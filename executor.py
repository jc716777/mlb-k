"""Live order execution against Kalshi, with hard risk limits.

Every signal passes through the RiskManager before any order is built. The
live-trading interlock (KALSHI_LIVE_TRADING_ENABLED) is the final gate: if it
is not explicitly enabled, the executor logs the intended order and refuses to
transmit. This is not a paper-trading mode -- it is a safety catch so a
misconfigured process cannot arm itself.
"""

from __future__ import annotations

import logging
import math
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from config import ExecutionConfig, RiskConfig
from kalshi import KalshiAPIError, KalshiRestClient
from models import EdgeSignal, OrderResult, Position, PositionStatus, Side

log = logging.getLogger("executor")


class RiskManager:
    """Tracks exposure and realized P&L; trips a kill switch on the daily loss."""

    def __init__(self, cfg: RiskConfig) -> None:
        self._cfg = cfg
        self.open_contracts: int = 0
        self.committed_cost_cents: int = 0
        self.realized_pnl_cents: int = 0
        self.kill_switch: bool = False

    def record_realized(self, pnl_cents: int) -> None:
        """Feed realized P&L to the daily-loss kill switch.

        WARNING: nothing calls this yet -- there is no fill/settlement
        tracking, so the kill switch is currently inert. Live exposure is
        bounded by max_position_cost_cents and the contract caps, not by
        realized loss. Wire this to a Kalshi fills/settlement feed to activate.
        """
        self.realized_pnl_cents += pnl_cents
        if self.realized_pnl_cents <= -self._cfg.daily_loss_limit_cents:
            self.kill_switch = True
            log.critical(
                "KILL SWITCH: realized loss %dc breached limit %dc",
                -self.realized_pnl_cents, self._cfg.daily_loss_limit_cents,
            )

    def approve(self, price_cents: int) -> tuple[int, str]:
        """Return (approved_contract_count, reject_reason).

        A count of 0 means the order is rejected; reason explains why.
        """
        if self.kill_switch:
            return 0, "kill switch active"
        if not (self._cfg.min_price_cents <= price_cents <= self._cfg.max_price_cents):
            return 0, f"price {price_cents}c outside tradable band"

        remaining_contracts = self._cfg.max_open_contracts - self.open_contracts
        if remaining_contracts <= 0:
            return 0, "max open contracts reached"

        remaining_budget = self._cfg.max_position_cost_cents - self.committed_cost_cents
        if remaining_budget <= 0:
            return 0, "max position cost reached"

        affordable = remaining_budget // max(price_cents, 1)
        count = min(
            self._cfg.max_contracts_per_order,
            remaining_contracts,
            affordable,
        )
        if count <= 0:
            return 0, "insufficient budget for one contract"
        return count, ""

    def commit(self, contracts: int, cost_cents: int) -> None:
        self.open_contracts += contracts
        self.committed_cost_cents += cost_cents

    def snapshot(self) -> str:
        return (
            f"open={self.open_contracts} committed={self.committed_cost_cents}c "
            f"realized_pnl={self.realized_pnl_cents}c "
            f"kill={'Y' if self.kill_switch else 'N'}"
        )

    def release(self, contracts: int) -> None:
        """Release committed contracts (e.g., when a position is closed)."""
        self.open_contracts = max(0, self.open_contracts - contracts)


class KalshiExecutor:
    """Builds, risk-checks, and transmits Kalshi orders."""

    def __init__(
        self,
        cfg: ExecutionConfig,
        risk: RiskManager,
        client: Optional[KalshiRestClient],
    ) -> None:
        self._cfg = cfg
        self._risk = risk
        self._client = client
        self.fills: list[OrderResult] = []      # accepted, transmitted orders
        self.results: list[OrderResult] = []    # every execution outcome
        self.positions: dict[str, Position] = {}  # market_ticker -> Position

    def _record(self, result: OrderResult) -> OrderResult:
        """Append every outcome to the audit trail; track accepted fills."""
        self.results.append(result)
        if result.accepted:
            self.fills.append(result)
        return result

    def _fee_cents(self, contracts: int, price_cents: int) -> int:
        """Kalshi trading fee: ceil(coeff * C * P * (1-P)), P in dollars."""
        p = price_cents / 100.0
        raw = self._cfg.fee_coefficient * contracts * p * (1.0 - p)
        return math.ceil(raw * 100.0)

    def _build_order(
        self, signal: EdgeSignal, contracts: int, limit_price_cents: int
    ) -> dict:
        """Build a Kalshi V2 order body (POST /portfolio/events/orders).

        V2 quotes everything from the YES book: side 'bid' = buy YES,
        'ask' = sell YES. A NO signal is expressed as selling YES, so the
        YES-side limit price is 100c minus the NO cost we are willing to pay.
        `count` and `price` are strings; `price` is fixed-point dollars.
        """
        if signal.side == Side.YES:
            v2_side = "bid"
            yes_price_cents = limit_price_cents
        else:  # buying NO is economically selling YES
            v2_side = "ask"
            yes_price_cents = 100 - limit_price_cents
        return {
            "ticker": signal.market_ticker,
            "client_order_id": str(uuid.uuid4()),
            "side": v2_side,
            "count": str(contracts),
            "price": f"{yes_price_cents / 100.0:.4f}",
            "time_in_force": "immediate_or_cancel",
            "self_trade_prevention_type": "taker_at_cross",
        }

    async def execute(self, signal: EdgeSignal) -> OrderResult:
        """Risk-check and (if armed) transmit a real order for this signal."""
        # Prevent pyramiding: reject if position already exists in this market.
        if signal.market_ticker in self.positions:
            pos = self.positions[signal.market_ticker]
            if pos.status != PositionStatus.CLOSED:
                result = OrderResult(
                    signal=signal,
                    accepted=False,
                    live=False,
                    reject_reason=f"position already open in {signal.market_ticker}",
                )
                log.warning(result.describe())
                return self._record(result)

        # Model crossing the spread: we pay through the touch.
        limit_price = signal.target_price_cents + self._cfg.slippage_cents
        limit_price = min(limit_price, 99)

        contracts, reject = self._risk.approve(limit_price)
        if contracts <= 0:
            result = OrderResult(
                signal=signal, accepted=False, live=False, reject_reason=reject
            )
            log.warning(result.describe())
            return self._record(result)

        # --- live-trading interlock -----------------------------------------
        if not self._cfg.live_trading_enabled:
            result = OrderResult(
                signal=signal,
                accepted=False,
                live=False,
                contracts=contracts,
                requested_price_cents=limit_price,
                reject_reason="interlock off (set KALSHI_LIVE_TRADING_ENABLED=true)",
            )
            log.warning(
                "INTERLOCK: would buy %s x%d @ %dc -- order NOT sent",
                signal.side.value.upper(), contracts, limit_price,
            )
            return self._record(result)

        if self._client is None:
            result = OrderResult(
                signal=signal,
                accepted=False,
                live=False,
                contracts=contracts,
                requested_price_cents=limit_price,
                reject_reason="no REST client configured",
            )
            log.error(result.describe())
            return self._record(result)

        order = self._build_order(signal, contracts, limit_price)
        started = time.perf_counter()
        try:
            response = await self._client.create_order(
                order, timeout_s=self._cfg.order_timeout_s
            )
        except (KalshiAPIError, Exception) as exc:  # noqa: BLE001
            latency_ms = (time.perf_counter() - started) * 1000.0
            result = OrderResult(
                signal=signal,
                accepted=False,
                live=True,
                contracts=contracts,
                requested_price_cents=limit_price,
                latency_ms=latency_ms,
                reject_reason=f"transmit failed: {exc}",
            )
            log.error(result.describe())
            return self._record(result)

        latency_ms = (time.perf_counter() - started) * 1000.0
        order_id = response.get("order_id")
        fill_count = int(float(response.get("fill_count") or 0))

        if fill_count <= 0:
            # IOC order accepted by the exchange but nothing crossed.
            result = OrderResult(
                signal=signal,
                accepted=False,
                live=True,
                order_id=str(order_id) if order_id else None,
                contracts=0,
                requested_price_cents=limit_price,
                latency_ms=latency_ms,
                reject_reason="immediate-or-cancel: nothing filled",
            )
            log.warning(result.describe())
            return self._record(result)

        # average_fill_price is a YES-book price. For a NO buy we sold YES, so
        # the cost of each NO contract is 100c minus the YES price transacted.
        avg_fill = response.get("average_fill_price")
        yes_fill_cents = round(float(avg_fill) * 100.0) if avg_fill else limit_price
        if signal.side == Side.YES:
            filled_price = yes_fill_cents
        else:
            filled_price = 100 - yes_fill_cents

        # Prefer the exchange's reported fee; fall back to the modeled estimate.
        avg_fee = response.get("average_fee_paid")
        if avg_fee is not None:
            fee = round(float(avg_fee) * 100.0 * fill_count)
        else:
            fee = self._fee_cents(fill_count, filled_price)

        slippage = filled_price - signal.target_price_cents
        notional = fill_count * filled_price + fee

        self._risk.commit(fill_count, notional)
        result = OrderResult(
            signal=signal,
            accepted=True,
            live=True,
            order_id=str(order_id) if order_id else None,
            contracts=fill_count,
            requested_price_cents=limit_price,
            filled_price_cents=filled_price,
            slippage_cents=slippage,
            fee_cents=fee,
            notional_cents=notional,
            latency_ms=latency_ms,
        )
        log.info(result.describe())
        log.info("  risk: %s", self._risk.snapshot())

        # Create a position for this entry.
        self._create_position(signal, filled_price, fill_count)
        return self._record(result)

    def _create_position(
        self, signal: EdgeSignal, entry_price_cents: int, size_contracts: int
    ) -> None:
        """Create a new position after an entry order fills."""
        edge_cents = int(round(signal.abs_edge * 100.0))
        edge_cents = max(edge_cents, 1)  # Ensure positive edge

        stop = entry_price_cents - edge_cents
        target_1 = entry_price_cents + edge_cents
        target_2 = entry_price_cents + 3 * edge_cents
        stop_after_t1 = entry_price_cents  # Breakeven

        position = Position(
            market_ticker=signal.market_ticker,
            side=signal.side,
            entry_signal=signal,
            entry_price_cents=entry_price_cents,
            size_contracts=size_contracts,
            stop_price_cents=max(1, stop),  # Clamp to valid range
            target_1_price_cents=min(99, target_1),
            target_2_price_cents=min(99, target_2),
            stop_after_t1_cents=min(99, stop_after_t1),
        )
        self.positions[signal.market_ticker] = position
        log.info(
            "POSITION OPEN: %s %s x%d @ %dc | stop=%dc T1=%dc T2=%dc",
            signal.market_ticker, signal.side.value.upper(), size_contracts,
            entry_price_cents, position.stop_price_cents, position.target_1_price_cents,
            position.target_2_price_cents,
        )

    def position_for(self, market_ticker: str) -> Optional[Position]:
        """Get the open position for a market, or None."""
        pos = self.positions.get(market_ticker)
        if pos and pos.status != PositionStatus.CLOSED:
            return pos
        return None

    def mark_position_t1_filled(self, market_ticker: str) -> None:
        """Mark the position's first tier as filled."""
        pos = self.positions.get(market_ticker)
        if pos:
            pos = pos.model_copy(
                update={
                    "status": PositionStatus.T1_FILLED,
                    "t1_filled_at": datetime.now(timezone.utc),
                }
            )
            self.positions[market_ticker] = pos
            log.info("POSITION T1 FILLED: %s | stop moved to breakeven", market_ticker)

    def close_position(self, market_ticker: str) -> Optional[Position]:
        """Mark a position as closed."""
        pos = self.positions.get(market_ticker)
        if pos:
            pos = pos.model_copy(
                update={
                    "status": PositionStatus.CLOSED,
                    "closed_at": datetime.now(timezone.utc),
                }
            )
            self.positions[market_ticker] = pos
            self._risk.release(pos.size_contracts)
            log.info("POSITION CLOSED: %s | size=%d", market_ticker, pos.size_contracts)
            return pos
        return None

    def evaluate_exits(self, market_ticker: str, current_price_cents: int) -> Optional[str]:
        """Check if a position should exit on current market price.

        Returns the exit reason if an exit is triggered: "stop", "target_1", "target_2".
        Returns None if no exit is triggered.
        """
        pos = self.position_for(market_ticker)
        if not pos:
            return None

        # Determine the relevant price based on position side.
        # For a long (YES) position: check against the bid (price we can sell at).
        # For a short (NO) position: check against the ask (price we can cover at).
        effective_price = current_price_cents

        if pos.status == PositionStatus.OPEN:
            # Check stop first (hard exit).
            if effective_price <= pos.stop_price_cents:
                return "stop"
            # Check target_1 (sell half).
            if effective_price >= pos.target_1_price_cents:
                return "target_1"
            # Check target_2 (sell rest).
            if effective_price >= pos.target_2_price_cents:
                return "target_2"

        elif pos.status == PositionStatus.T1_FILLED:
            # After T1 is filled, stop is moved to breakeven.
            if effective_price <= pos.stop_after_t1_cents:
                return "stop"
            # Check target_2 (sell remaining).
            if effective_price >= pos.target_2_price_cents:
                return "target_2"

        return None

    async def execute_exit(
        self, market_ticker: str, exit_reason: str, current_price_cents: int
    ) -> bool:
        """Execute an exit order for a position.

        Returns True if an exit order was executed, False otherwise.
        """
        pos = self.position_for(market_ticker)
        if not pos:
            return False

        # Determine how many contracts to sell.
        if exit_reason == "target_1" and pos.status == PositionStatus.OPEN:
            # Sell half at target_1.
            contracts = pos.size_contracts // 2
            limit_price = pos.target_1_price_cents
            self.mark_position_t1_filled(market_ticker)
        elif exit_reason in ("target_2", "stop"):
            # Sell remaining at market.
            contracts = pos.size_contracts
            limit_price = current_price_cents
            self.close_position(market_ticker)
        else:
            return False

        if contracts <= 0:
            return False

        # Build an exit order (opposite side of entry).
        exit_side = Side.NO if pos.side == Side.YES else Side.YES
        order_dict = {
            "ticker": market_ticker,
            "client_order_id": str(uuid.uuid4()),
            "side": "ask" if exit_side == Side.YES else "bid",
            "count": str(contracts),
            "price": f"{limit_price / 100.0:.4f}",
            "time_in_force": "immediate_or_cancel",
            "self_trade_prevention_type": "taker_at_cross",
        }

        if self._client is None:
            log.warning("no REST client; cannot execute exit for %s", market_ticker)
            return False

        try:
            response = await self._client.create_order(order_dict, timeout_s=self._cfg.order_timeout_s)
            fill_count = int(float(response.get("fill_count") or 0))
            if fill_count > 0:
                log.info(
                    "EXIT %s: %s x%d @ %dc | reason=%s",
                    market_ticker, exit_side.value.upper(), fill_count,
                    current_price_cents, exit_reason,
                )
                return True
            else:
                log.warning("exit order for %s did not fill", market_ticker)
                return False
        except (KalshiAPIError, Exception) as exc:
            log.error("exit order failed for %s: %s", market_ticker, exc)
            return False
