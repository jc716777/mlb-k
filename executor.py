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
        """Build a Kalshi order body.

        UNVERIFIED -- this is the LEGACY /portfolio/orders body. The V2
        endpoint (/portfolio/events/orders) expects different fields: `price`
        as fixed-point dollars, `count` as a string, side as bid/ask. Confirm
        against the docs before live use.
        """
        order: dict = {
            "ticker": signal.market_ticker,
            "client_order_id": str(uuid.uuid4()),
            "action": "buy",
            "side": signal.side.value,
            "count": contracts,
            "type": "limit",
        }
        # Kalshi limit orders name the price field after the side being bought.
        if signal.side == Side.YES:
            order["yes_price"] = limit_price_cents
        else:
            order["no_price"] = limit_price_cents
        return order

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
        order_obj = response.get("order", response)
        order_id = order_obj.get("order_id") or order_obj.get("id")

        # Kalshi limit orders may rest; we model an immediate cross to the limit.
        filled_price = limit_price
        slippage = filled_price - signal.target_price_cents
        fee = self._fee_cents(contracts, filled_price)
        notional = contracts * filled_price + fee

        self._risk.commit(contracts, notional)
        result = OrderResult(
            signal=signal,
            accepted=True,
            live=True,
            order_id=str(order_id) if order_id else None,
            contracts=contracts,
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
