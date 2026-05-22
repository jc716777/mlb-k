"""Centralized configuration for the MLB-Kalshi pullback trading framework.

All tunables live here. Secrets are read from the environment (see .env.example);
nothing sensitive is hard-coded.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_bool(name: str, default: bool = False) -> bool:
    return _env(name, str(default)).lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


@dataclass(frozen=True)
class KalshiConfig:
    """Connection + auth parameters for the Kalshi trading API."""

    api_key_id: str = field(default_factory=lambda: _env("KALSHI_API_KEY_ID"))
    private_key_path: str = field(default_factory=lambda: _env("KALSHI_PRIVATE_KEY_PATH"))
    # Defaults point at Kalshi's demo environment -- dry-run there first.
    rest_base: str = field(
        default_factory=lambda: _env(
            "KALSHI_REST_BASE", "https://external-api.demo.kalshi.co"
        )
    )
    ws_base: str = field(
        default_factory=lambda: _env(
            "KALSHI_WS_BASE", "wss://external-api-ws.demo.kalshi.co"
        )
    )
    # Path segments are stable across environments.
    rest_prefix: str = "/trade-api/v2"
    ws_path: str = "/trade-api/ws/v2"

    market_ticker: str = field(default_factory=lambda: _env("KALSHI_MARKET_TICKER"))

    def validate(self) -> None:
        if not self.api_key_id:
            raise ValueError("KALSHI_API_KEY_ID is not set")
        if not self.private_key_path or not Path(self.private_key_path).is_file():
            raise ValueError(
                f"KALSHI_PRIVATE_KEY_PATH does not point to a file: {self.private_key_path!r}"
            )
        if not self.market_ticker:
            raise ValueError("KALSHI_MARKET_TICKER is not set")


@dataclass(frozen=True)
class MLBConfig:
    """MLB StatsAPI (public, unauthenticated) game-feed parameters."""

    game_pk: int = field(default_factory=lambda: _env_int("MLB_GAME_PK", 0))
    statsapi_base: str = "https://statsapi.mlb.com/api"
    # StatsAPI has no push socket; we poll the live feed.
    poll_interval_s: float = field(
        default_factory=lambda: _env_float("MLB_POLL_INTERVAL_S", 3.0)
    )
    regulation_innings: int = 9

    def feed_url(self) -> str:
        return f"{self.statsapi_base}/v1.1/game/{self.game_pk}/feed/live"

    def validate(self) -> None:
        if self.game_pk <= 0:
            raise ValueError("MLB_GAME_PK is not set or invalid")


@dataclass(frozen=True)
class StrategyConfig:
    """Quant / signal parameters."""

    # Minimum |P_clean - P_model| edge required to fire a signal.
    alpha_threshold: float = field(default_factory=lambda: _env_float("ALPHA_THRESHOLD", 0.06))
    # Rolling window for computing line-movement velocity (dOdds/dt).
    momentum_window_s: float = field(
        default_factory=lambda: _env_float("MOMENTUM_WINDOW_S", 30.0)
    )
    # Ticks older than this are considered stale and discarded.
    stale_tick_s: float = field(default_factory=lambda: _env_float("STALE_TICK_S", 2.0))
    # |velocity| (probability units / second) above which a move is a "surge".
    surge_velocity_threshold: float = field(
        default_factory=lambda: _env_float("SURGE_VELOCITY_THRESHOLD", 0.035)
    )
    # A surge is "mishandled" only if the market moved this many times more than
    # the sabermetric model did over the same window.
    mishandle_ratio: float = field(default_factory=lambda: _env_float("MISHANDLE_RATIO", 2.5))
    # Per-half-inning run variance used to spread the win-prob distribution.
    run_variance_per_half_inning: float = 0.90
    # League-average runs scored by the leadoff (bases-empty, 0-out) state.
    league_avg_half_inning_runs: float = 0.481
    # Cooldown between signals on the same market (seconds).
    signal_cooldown_s: float = field(
        default_factory=lambda: _env_float("SIGNAL_COOLDOWN_S", 20.0)
    )
    # Weight on the current batter when adjusting the live half-inning's run
    # expectancy; the remainder of the inning regresses to league average.
    current_batter_weight: float = 0.35
    # Suppress signals in situations below this Leverage Index (0.0 disables).
    min_leverage_index: float = field(
        default_factory=lambda: _env_float("MIN_LEVERAGE_INDEX", 0.0)
    )


@dataclass(frozen=True)
class RiskConfig:
    """Hard limits. These are NOT optional and apply even in live-only mode."""

    max_contracts_per_order: int = field(
        default_factory=lambda: _env_int("MAX_CONTRACTS_PER_ORDER", 5)
    )
    max_open_contracts: int = field(
        default_factory=lambda: _env_int("MAX_OPEN_CONTRACTS", 25)
    )
    # Total capital the bot may put at risk, in cents. Default $25 of a $50 deposit.
    max_position_cost_cents: int = field(
        default_factory=lambda: _env_int("MAX_POSITION_COST_CENTS", 2500)
    )
    # Realized loss (cents) that trips the kill switch for the day.
    # NOTE: inert until P&L is wired from settlement -- see
    # RiskManager.record_realized. The position-cost cap is the live bound.
    daily_loss_limit_cents: int = field(
        default_factory=lambda: _env_int("DAILY_LOSS_LIMIT_CENTS", 1200)
    )
    # Refuse to trade markets priced this extreme (no liquidity / no edge room).
    min_price_cents: int = 5
    max_price_cents: int = 95


@dataclass(frozen=True)
class ExecutionConfig:
    """Order-routing behavior and the live-trading interlock."""

    # MUST be true or the executor will never transmit a real order.
    live_trading_enabled: bool = field(
        default_factory=lambda: _env_bool("KALSHI_LIVE_TRADING_ENABLED", False)
    )
    # Cents added through the touch on a limit order to model crossing/slippage.
    slippage_cents: int = field(default_factory=lambda: _env_int("SLIPPAGE_CENTS", 2))
    # Modeled round-trip latency from signal to ack (seconds), used for logging.
    expected_latency_ms: float = field(
        default_factory=lambda: _env_float("EXPECTED_LATENCY_MS", 250.0)
    )
    # Kalshi trading fee coefficient: fee = ceil(coeff * C * P * (1-P)).
    fee_coefficient: float = 0.07
    order_timeout_s: float = 5.0


@dataclass(frozen=True)
class AppConfig:
    kalshi: KalshiConfig = field(default_factory=KalshiConfig)
    mlb: MLBConfig = field(default_factory=MLBConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))

    def validate(self) -> None:
        """Fail fast on misconfiguration before any network or trading activity."""
        self.kalshi.validate()
        self.mlb.validate()


def load_config() -> AppConfig:
    """Build the config tree from the current environment."""
    return AppConfig()
