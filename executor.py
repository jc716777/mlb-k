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
from typing import Optional

from config import ExecutionConfig, RiskConfig
from kalshi import KalshiAPIError, KalshiRestClient
from models import EdgeSignal, OrderResult, Side

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
        return self._record(result)
