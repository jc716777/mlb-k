"""Strongly typed domain models for game state, market odds, and signals.

All models are Pydantic v2 for validation at the system boundary (feeds) and
immutability where it matters.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, computed_field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class HalfInning(str, Enum):
    TOP = "top"      # away team batting
    BOTTOM = "bottom"  # home team batting


class Side(str, Enum):
    YES = "yes"
    NO = "no"


# --------------------------------------------------------------------------
# Game state
# --------------------------------------------------------------------------
class GameState(BaseModel):
    """A single immutable snapshot of the on-field situation."""

    model_config = ConfigDict(frozen=True)

    game_pk: int
    inning: int = Field(ge=1)
    half_inning: HalfInning
    outs: int = Field(ge=0, le=3)
    # (first_base, second_base, third_base) occupancy.
    runners_on_base: tuple[bool, bool, bool]
    home_score: int = Field(ge=0)
    away_score: int = Field(ge=0)
    # Matchup / context, used for run-environment adjustments.
    pitcher_id: Optional[int] = None
    batter_id: Optional[int] = None
    venue: str = ""
    timestamp: datetime = Field(default_factory=utcnow)
    last_event: str = ""
    is_final: bool = False

    @property
    def batting_is_home(self) -> bool:
        return self.half_inning == HalfInning.BOTTOM

    @property
    def base_index(self) -> int:
        """Index 0-7 into the run-expectancy base dimension.

        Bit 0 = 1B, bit 1 = 2B, bit 2 = 3B.
        """
        r1, r2, r3 = self.runners_on_base
        return (int(r1) << 0) | (int(r2) << 1) | (int(r3) << 2)

    @property
    def runners_count(self) -> int:
        return sum(self.runners_on_base)

    @property
    def score_diff_home(self) -> int:
        """Home score minus away score."""
        return self.home_score - self.away_score

    def situation_key(self) -> tuple:
        """Hashable key identifying the discrete game situation (ignores time).

        Includes pitcher/batter so a new matchup re-triggers evaluation.
        """
        return (
            self.inning,
            self.half_inning,
            self.outs,
            self.runners_on_base,
            self.home_score,
            self.away_score,
            self.pitcher_id,
            self.batter_id,
        )


# --------------------------------------------------------------------------
# Market odds
# --------------------------------------------------------------------------
class MarketOdds(BaseModel):
    """A Kalshi order-book top-of-book snapshot.

    Kalshi prices are integer cents in [1, 99]. YES and NO are complementary:
    no_bid == 100 - yes_ask and no_ask == 100 - yes_bid, but we carry all four
    explicitly so a partial feed update never silently corrupts the book.
    """

    model_config = ConfigDict(frozen=True)

    market_ticker: str
    bookmaker: str = "kalshi"
    market_type: str = "h2h_winner"  # home team wins == YES
    yes_bid: int = Field(ge=0, le=100)
    yes_ask: int = Field(ge=0, le=100)
    no_bid: int = Field(ge=0, le=100)
    no_ask: int = Field(ge=0, le=100)
    timestamp: datetime = Field(default_factory=utcnow)

    # --- raw implied probabilities (include the bid/ask overround) ---
    @computed_field  # type: ignore[prop-decorator]
    @property
    def implied_prob_yes(self) -> float:
        """Cost to acquire one YES contract, as a probability (includes vig)."""
        return self.yes_ask / 100.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def implied_prob_no(self) -> float:
        return self.no_ask / 100.0

    @property
    def overround(self) -> float:
        """Sum of implied probabilities minus 1; the book's margin."""
        return self.implied_prob_yes + self.implied_prob_no - 1.0

    @property
    def yes_mid(self) -> float:
        return (self.yes_bid + self.yes_ask) / 200.0

    @property
    def spread_cents(self) -> int:
        return self.yes_ask - self.yes_bid

    def vig_free_prob_yes(self) -> float:
        """Proportional (Shin-free) margin strip:  P_clean = P_i / sum(P_i)."""
        total = self.implied_prob_yes + self.implied_prob_no
        if total <= 0:
            return 0.5
        return self.implied_prob_yes / total


# --------------------------------------------------------------------------
# Signals & trades
# --------------------------------------------------------------------------
class ModelEstimate(BaseModel):
    """Output of the sabermetric win-probability model."""

    model_config = ConfigDict(frozen=True)

    p_model_home: float          # model true probability the home team wins
    expected_runs_inning: float  # RE24 expected runs for the current half-inning
    mean_final_diff: float       # expected (home - away) final run differential
    sigma_final_diff: float      # std dev of the final differential
    leverage_index: float = 1.0  # WP volatility vs. a neutral mid-game state
    park_factor: float = 1.0     # ballpark run multiplier applied
    pitcher_factor: float = 1.0  # current pitcher run-suppression multiplier
    batter_factor: float = 1.0   # current batter run-creation multiplier
    matchup_factor: float = 1.0  # combined multiplier on the live half-inning
    timestamp: datetime = Field(default_factory=utcnow)


class EdgeSignal(BaseModel):
    """A detected mean-reversion / pullback opportunity."""

    model_config = ConfigDict(frozen=True)

    market_ticker: str
    side: Side                   # contract to buy
    p_clean: float               # vig-stripped market probability (YES)
    p_model: float               # model probability (YES / home win)
    edge: float                  # signed P_model - P_clean (YES basis)
    abs_edge: float
    line_velocity: float         # dProb/dt over the momentum window
    model_velocity: float        # dP_model/dt over the same window
    mishandled_surge: bool
    leverage_index: float = 1.0  # WP volatility of the situation
    target_price_cents: int      # price we intend to pay
    game_event: str
    timestamp: datetime = Field(default_factory=utcnow)

    def describe(self) -> str:
        return (
            f"[EDGE] {self.market_ticker} buy {self.side.value.upper()} "
            f"P_clean={self.p_clean:.3f} P_model={self.p_model:.3f} "
            f"edge={self.edge:+.3f} v_line={self.line_velocity:+.4f}/s "
            f"v_model={self.model_velocity:+.4f}/s "
            f"surge={'Y' if self.mishandled_surge else 'N'} "
            f"LI={self.leverage_index:.2f} "
            f"target={self.target_price_cents}c | {self.game_event}"
        )


class OrderResult(BaseModel):
    """Outcome of an execution attempt."""

    model_config = ConfigDict(frozen=True)

    signal: EdgeSignal
    accepted: bool
    live: bool                   # True if a real order was transmitted
    order_id: Optional[str] = None
    contracts: int = 0
    requested_price_cents: int = 0
    filled_price_cents: int = 0
    slippage_cents: int = 0
    fee_cents: int = 0
    notional_cents: int = 0
    latency_ms: float = 0.0
    reject_reason: str = ""
    timestamp: datetime = Field(default_factory=utcnow)

    def describe(self) -> str:
        if not self.accepted:
            return f"[ORDER REJECTED] {self.signal.market_ticker}: {self.reject_reason}"
        tag = "LIVE" if self.live else "BLOCKED(interlock)"
        return (
            f"[ORDER {tag}] {self.signal.market_ticker} "
            f"{self.signal.side.value.upper()} x{self.contracts} "
            f"req={self.requested_price_cents}c fill={self.filled_price_cents}c "
            f"slip={self.slippage_cents}c fee={self.fee_cents}c "
            f"notional={self.notional_cents}c lat={self.latency_ms:.0f}ms "
            f"id={self.order_id}"
        )


def normal_cdf(x: float) -> float:
    """Standard-normal CDF via erf; used by the win-probability model."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def normal_pdf(x: float) -> float:
    """Standard-normal PDF; used to gauge win-probability leverage."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
