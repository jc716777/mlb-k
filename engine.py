"""Quantitative core: vig stripping, sabermetric win probability, line
momentum, and mean-reversion ("pullback") signal generation.

The win-probability model adjusts the run environment for the ballpark and the
live pitcher/batter matchup, and reports a Leverage Index for the situation.
The model runs in a thread pool so a burst of feed events never stalls the
asyncio event loop.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Deque, NamedTuple, Optional

from config import StrategyConfig
from models import (
    EdgeSignal,
    GameState,
    HalfInning,
    MarketOdds,
    ModelEstimate,
    Side,
    normal_cdf,
    normal_pdf,
)
from sabermetrics import PlayerStatsCache, park_factor

log = logging.getLogger("engine")

# --------------------------------------------------------------------------
# Static 24-state Run Expectancy Matrix (RE24).
# Rows  = base state, indexed by bit pattern (1B=bit0, 2B=bit1, 3B=bit2).
# Cols  = outs (0, 1, 2). Values: expected runs scored in the rest of the
# half-inning. League-average era values.
# --------------------------------------------------------------------------
RUN_EXPECTANCY: dict[int, tuple[float, float, float]] = {
    0b000: (0.481, 0.254, 0.098),  # bases empty
    0b001: (0.859, 0.509, 0.224),  # 1B
    0b010: (1.100, 0.664, 0.319),  # 2B
    0b011: (1.437, 0.884, 0.429),  # 1B, 2B
    0b100: (1.350, 0.950, 0.353),  # 3B
    0b101: (1.784, 1.130, 0.478),  # 1B, 3B
    0b110: (1.964, 1.376, 0.580),  # 2B, 3B
    0b111: (2.292, 1.541, 0.752),  # bases loaded
}


def run_expectancy(base_index: int, outs: int) -> float:
    """Expected remaining runs for a (base state, outs) situation."""
    if outs >= 3:
        return 0.0
    return RUN_EXPECTANCY[base_index][outs]


class _Distribution(NamedTuple):
    """Intermediate result of the final-score-differential projection."""

    mean_diff: float        # expected (home - away) final differential
    score_state_diff: float # mean_diff excluding the live half-inning's runs
    sigma: float            # std dev of that differential
    re_now: float           # unadjusted RE24 value for the current state
    park_factor: float
    pitcher_factor: float
    batter_factor: float
    matchup_factor: float   # combined multiplier on the live half-inning


class _Series:
    """A timestamped rolling series supporting velocity over a window."""

    def __init__(self, window_s: float) -> None:
        self._window_s = window_s
        self._points: Deque[tuple[float, float]] = deque()

    def push(self, value: float, ts: Optional[float] = None) -> None:
        now = ts if ts is not None else time.monotonic()
        self._points.append((now, value))
        cutoff = now - self._window_s
        while len(self._points) > 1 and self._points[0][0] < cutoff:
            self._points.popleft()

    def velocity(self) -> float:
        """Mean rate of change (units/second) across the retained window."""
        if len(self._points) < 2:
            return 0.0
        t0, v0 = self._points[0]
        t1, v1 = self._points[-1]
        dt = t1 - t0
        return (v1 - v0) / dt if dt > 1e-6 else 0.0

    @property
    def latest(self) -> Optional[float]:
        return self._points[-1][1] if self._points else None


class WinProbabilityModel:
    """Sabermetric baseline win probability from the RE24 matrix.

    Combines current score, RE24 expected runs for the live half-inning, and a
    league-average projection of the remaining innings, then maps the expected
    final run differential to a probability through a normal distribution whose
    spread grows with the number of half-innings still to be played.

    Adjustments:
      * park        -- scales the run environment for both teams.
      * pitcher     -- the live pitcher's run-suppression multiplier.
      * batter      -- the live batter's run-creation multiplier (weighted, as
                       he is only one of the hitters due up this inning).

    It also reports a Leverage Index: the win-probability sensitivity to one
    run, normalized so a neutral mid-game state is ~1.0. This captures the
    closeness-and-lateness component of leverage.
    """

    def __init__(
        self,
        cfg: StrategyConfig,
        regulation_innings: int = 9,
        stats_cache: Optional[PlayerStatsCache] = None,
    ) -> None:
        self._cfg = cfg
        self._regulation = regulation_innings
        self._stats = stats_cache
        # Self-calibrate the leverage baseline from a canonical mid-game state
        # (tie game, top of the 5th, bases empty, nobody out).
        reference = GameState(
            game_pk=0,
            inning=regulation_innings // 2 + 1,
            half_inning=HalfInning.TOP,
            outs=0,
            runners_on_base=(False, False, False),
            home_score=0,
            away_score=0,
        )
        ref = self._project(reference)
        self._baseline_sensitivity = self._run_sensitivity(
            ref.score_state_diff, ref.sigma
        )

    # -- run-environment adjustments ----------------------------------------
    def _matchup(self, state: GameState) -> tuple[float, float, float, float]:
        """Return (park, pitcher_factor, batter_factor, matchup_factor)."""
        park = park_factor(state.venue)
        pitcher_factor = 1.0
        batter_factor = 1.0
        if self._stats is not None:
            pitcher = self._stats.pitcher(state.pitcher_id)
            if pitcher is not None:
                pitcher_factor = pitcher.run_factor()
            batter = self._stats.batter(state.batter_id)
            if batter is not None:
                batter_factor = batter.run_factor()

        # The current batter is only one of ~4-5 hitters due up this inning;
        # blend his factor toward league average for the rest of the frame.
        w = self._cfg.current_batter_weight
        effective_batter = w * batter_factor + (1.0 - w) * 1.0
        # Log5-style combination: both factors are already league-relative.
        matchup_factor = pitcher_factor * effective_batter
        return park, pitcher_factor, batter_factor, matchup_factor

    # -- final-score projection ---------------------------------------------
    def _project(self, state: GameState) -> _Distribution:
        re_now = run_expectancy(state.base_index, state.outs)
        park, pitcher_f, batter_f, matchup_f = self._matchup(state)

        # Count future *full* half-innings after the current (partial) one.
        innings_left = max(0, self._regulation - state.inning)
        if state.half_inning == HalfInning.TOP:
            away_future = innings_left          # away bats in each later inning
            home_future = innings_left + 1      # plus the bottom of this inning
        else:  # BOTTOM
            away_future = innings_left
            home_future = innings_left

        # Future innings: park-adjusted league average (matchups unknown).
        avg = self._cfg.league_avg_half_inning_runs * park
        away_total = state.away_score + away_future * avg
        home_total = state.home_score + home_future * avg
        # Differential from score + remaining innings only -- this drives the
        # Leverage Index, so the live at-bat's run expectancy does not deflate
        # how pivotal the situation is.
        score_state_diff = home_total - away_total

        # Current half-inning: RE24 adjusted for park and the live matchup.
        re_adjusted = re_now * matchup_f * park
        if state.batting_is_home:
            home_total += re_adjusted
        else:
            away_total += re_adjusted

        mean_diff = home_total - away_total

        total_half_innings_left = away_future + home_future + 1  # +1 = current
        variance = self._cfg.run_variance_per_half_inning * total_half_innings_left
        sigma = max(variance ** 0.5, 0.5)

        return _Distribution(
            mean_diff=mean_diff,
            score_state_diff=score_state_diff,
            sigma=sigma,
            re_now=re_now,
            park_factor=park,
            pitcher_factor=pitcher_f,
            batter_factor=batter_f,
            matchup_factor=matchup_f,
        )

    @staticmethod
    def _run_sensitivity(mean_diff: float, sigma: float) -> float:
        """dP(home win)/d(run): how much one run swings win probability."""
        return normal_pdf(mean_diff / sigma) / sigma

    def estimate(self, state: GameState) -> ModelEstimate:
        # Terminal states are deterministic.
        if state.is_final:
            p = 1.0 if state.score_diff_home > 0 else 0.0 if state.score_diff_home < 0 else 0.5
            return ModelEstimate(
                p_model_home=p,
                expected_runs_inning=0.0,
                mean_final_diff=float(state.score_diff_home),
                sigma_final_diff=1e-6,
                leverage_index=0.0,
            )

        dist = self._project(state)
        z = dist.mean_diff / dist.sigma
        p_home = min(max(normal_cdf(z), 0.0), 1.0)

        # Leverage is gauged at the score-and-innings state, so a lopsided
        # matchup does not understate how pivotal a close game is.
        sensitivity = self._run_sensitivity(dist.score_state_diff, dist.sigma)
        leverage = (
            sensitivity / self._baseline_sensitivity
            if self._baseline_sensitivity > 0
            else 1.0
        )

        return ModelEstimate(
            p_model_home=p_home,
            expected_runs_inning=dist.re_now,
            mean_final_diff=dist.mean_diff,
            sigma_final_diff=dist.sigma,
            leverage_index=leverage,
            park_factor=dist.park_factor,
            pitcher_factor=dist.pitcher_factor,
            batter_factor=dist.batter_factor,
            matchup_factor=dist.matchup_factor,
        )


class PullbackEngine:
    """Detects mean-reversion opportunities when the market overreacts.

    Feed both streams in via `update_game_state` / `update_odds`; call
    `evaluate` to (asynchronously) produce a signal when one exists.
    """

    def __init__(
        self,
        cfg: StrategyConfig,
        regulation_innings: int = 9,
        stats_cache: Optional[PlayerStatsCache] = None,
    ) -> None:
        self._cfg = cfg
        self._model = WinProbabilityModel(cfg, regulation_innings, stats_cache)
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="winprob")

        self._game_state: Optional[GameState] = None
        self._odds: Optional[MarketOdds] = None
        self._estimate: Optional[ModelEstimate] = None

        self._line_series = _Series(cfg.momentum_window_s)
        self._model_series = _Series(cfg.momentum_window_s)
        self._last_signal_ts: float = 0.0
        self._last_event_label: str = ""

    # -- ingestion -----------------------------------------------------------
    def update_game_state(self, state: GameState) -> None:
        self._game_state = state
        self._last_event_label = state.last_event or self._last_event_label

    def update_odds(self, odds: MarketOdds) -> None:
        self._odds = odds
        self._line_series.push(odds.vig_free_prob_yes())

    # -- evaluation ----------------------------------------------------------
    async def evaluate(self) -> Optional[EdgeSignal]:
        """Recompute the model and emit a signal if the edge clears alpha."""
        if self._game_state is None or self._odds is None:
            return None

        # Heavy(ish) matrix/stat work off the event loop.
        loop = asyncio.get_running_loop()
        estimate = await loop.run_in_executor(
            self._pool, self._model.estimate, self._game_state
        )
        self._estimate = estimate
        self._model_series.push(estimate.p_model_home)

        odds = self._odds
        p_clean = odds.vig_free_prob_yes()
        p_model = estimate.p_model_home

        edge = p_model - p_clean          # YES (home-win) basis
        abs_edge = abs(edge)

        line_velocity = self._line_series.velocity()
        model_velocity = self._model_series.velocity()

        # A "mishandled surge": the line lurched while the fundamentals barely
        # moved -- the market is reacting to an event the model has digested.
        fast_line = abs(line_velocity) >= self._cfg.surge_velocity_threshold
        outran_model = abs(line_velocity) >= self._cfg.mishandle_ratio * (
            abs(model_velocity) + 1e-6
        )
        mishandled_surge = fast_line and outran_model

        if abs_edge < self._cfg.alpha_threshold:
            return None

        # Skip low-leverage spots: small WP swings give little room to profit.
        if estimate.leverage_index < self._cfg.min_leverage_index:
            log.debug(
                "edge %.3f ignored: LI %.2f below min %.2f",
                abs_edge, estimate.leverage_index, self._cfg.min_leverage_index,
            )
            return None

        now = time.monotonic()
        if now - self._last_signal_ts < self._cfg.signal_cooldown_s:
            return None

        # Buy the side the market has under-priced.
        if edge > 0:
            side = Side.YES
            target_price = odds.yes_ask
        else:
            side = Side.NO
            target_price = odds.no_ask

        self._last_signal_ts = now
        signal = EdgeSignal(
            market_ticker=odds.market_ticker,
            side=side,
            p_clean=p_clean,
            p_model=p_model,
            edge=edge,
            abs_edge=abs_edge,
            line_velocity=line_velocity,
            model_velocity=model_velocity,
            mishandled_surge=mishandled_surge,
            leverage_index=estimate.leverage_index,
            target_price_cents=target_price,
            game_event=self._last_event_label[:80],
        )
        log.info(signal.describe())
        log.info(
            "  model: park=%.2f pitcher=%.2f batter=%.2f matchup=%.2f LI=%.2f",
            estimate.park_factor, estimate.pitcher_factor,
            estimate.batter_factor, estimate.matchup_factor,
            estimate.leverage_index,
        )
        if mishandled_surge:
            log.info(
                "  -> MISHANDLED SURGE: line moved %.4f/s vs model %.4f/s",
                line_velocity, model_velocity,
            )
        return signal

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
