"""Unit tests for the math and state-machine components.

Run with:  pytest test_framework.py -v

Async paths are exercised through asyncio.run so no pytest plugin is needed.
"""

from __future__ import annotations

import asyncio
import dataclasses

from config import ExecutionConfig, RiskConfig, StrategyConfig
from engine import PullbackEngine, WinProbabilityModel, _Series, run_expectancy
from executor import KalshiExecutor, RiskManager
from models import EdgeSignal, GameState, HalfInning, MarketOdds, Side
from sabermetrics import BatterStats, PitcherStats, _parse_innings, park_factor


def _game_state(**overrides) -> GameState:
    base = dict(
        game_pk=1,
        inning=5,
        half_inning=HalfInning.TOP,
        outs=0,
        runners_on_base=(False, False, False),
        home_score=0,
        away_score=0,
    )
    base.update(overrides)
    return GameState(**base)


# --------------------------------------------------------------------------
# Vig stripping
# --------------------------------------------------------------------------
def test_vig_free_probability_strips_overround():
    odds = MarketOdds(
        market_ticker="T", yes_bid=58, yes_ask=62, no_bid=38, no_ask=42
    )
    # implied: yes 0.62, no 0.42 -> overround 0.04.
    assert abs(odds.overround - 0.04) < 1e-9
    # proportional strip: 0.62 / (0.62 + 0.42).
    assert abs(odds.vig_free_prob_yes() - 0.62 / 1.04) < 1e-9


def test_vig_free_probability_is_a_probability():
    odds = MarketOdds(market_ticker="T", yes_bid=10, yes_ask=14, no_bid=86, no_ask=90)
    assert 0.0 < odds.vig_free_prob_yes() < 1.0


# --------------------------------------------------------------------------
# Run Expectancy Matrix
# --------------------------------------------------------------------------
def test_run_expectancy_known_values():
    assert run_expectancy(0b000, 0) == 0.481   # bases empty, 0 out
    assert run_expectancy(0b111, 0) == 2.292   # bases loaded, 0 out
    assert run_expectancy(0b111, 2) == 0.752   # bases loaded, 2 out


def test_run_expectancy_three_outs_is_zero():
    assert run_expectancy(0b111, 3) == 0.0


# --------------------------------------------------------------------------
# Win-probability model
# --------------------------------------------------------------------------
def test_final_state_is_deterministic():
    model = WinProbabilityModel(StrategyConfig())
    home_win = _game_state(
        inning=9, half_inning=HalfInning.BOTTOM, outs=3,
        home_score=5, away_score=2, is_final=True,
    )
    away_win = _game_state(
        inning=9, half_inning=HalfInning.BOTTOM, outs=3,
        home_score=2, away_score=5, is_final=True,
    )
    assert model.estimate(home_win).p_model_home == 1.0
    assert model.estimate(away_win).p_model_home == 0.0


def test_win_probability_monotonic_in_score():
    model = WinProbabilityModel(StrategyConfig())
    trailing = model.estimate(_game_state(home_score=0, away_score=4))
    leading = model.estimate(_game_state(home_score=4, away_score=0))
    assert leading.p_model_home > 0.5 > trailing.p_model_home


def test_leverage_index_ordering():
    model = WinProbabilityModel(StrategyConfig())
    late_close = model.estimate(_game_state(
        inning=9, half_inning=HalfInning.BOTTOM, outs=2,
        home_score=4, away_score=4,
    ))
    early = model.estimate(_game_state(inning=2))
    blowout = model.estimate(_game_state(
        inning=8, half_inning=HalfInning.TOP, home_score=10, away_score=1,
    ))
    assert late_close.leverage_index > early.leverage_index
    assert early.leverage_index > blowout.leverage_index
    assert late_close.leverage_index > 2.0
    assert blowout.leverage_index < 0.2


def test_pitcher_quality_suppresses_runs():
    cfg = StrategyConfig()
    cache_like = type("C", (), {
        "pitcher": lambda self, _id: PitcherStats(player_id=1, era=2.20, innings_pitched=150),
        "batter": lambda self, _id: None,
    })()
    model = WinProbabilityModel(cfg, stats_cache=cache_like)
    # Home batting with a strong opposing pitcher -> lower home win prob than
    # the same situation with a neutral pitcher.
    neutral = WinProbabilityModel(cfg).estimate(_game_state(
        inning=7, half_inning=HalfInning.BOTTOM, outs=1,
        runners_on_base=(True, True, True), home_score=3, away_score=3,
    ))
    with_ace = model.estimate(_game_state(
        inning=7, half_inning=HalfInning.BOTTOM, outs=1,
        runners_on_base=(True, True, True), home_score=3, away_score=3,
        pitcher_id=1,
    ))
    assert with_ace.pitcher_factor < 1.0
    assert with_ace.p_model_home < neutral.p_model_home


# --------------------------------------------------------------------------
# Sabermetrics
# --------------------------------------------------------------------------
def test_park_factor_lookup():
    assert park_factor("Coors Field") > 1.10
    assert park_factor("Oracle Park") < 1.00
    assert park_factor("Totally Made Up Stadium") == 1.00


def test_pitcher_run_factor_direction():
    ace = PitcherStats(player_id=1, era=2.40, innings_pitched=150)
    replacement = PitcherStats(player_id=2, era=6.00, innings_pitched=150)
    assert ace.run_factor() < 1.0 < replacement.run_factor()


def test_batter_run_factor_direction():
    star = BatterStats(player_id=1, ops=0.950, plate_appearances=500)
    weak = BatterStats(player_id=2, ops=0.560, plate_appearances=500)
    assert star.run_factor() > 1.0 > weak.run_factor()


def test_innings_pitched_parsing():
    assert abs(_parse_innings("45.1") - (45 + 1 / 3)) < 1e-9
    assert abs(_parse_innings("45.2") - (45 + 2 / 3)) < 1e-9
    assert _parse_innings("45.0") == 45.0
    assert _parse_innings("bogus") == 0.0


# --------------------------------------------------------------------------
# Momentum series
# --------------------------------------------------------------------------
def test_series_velocity():
    series = _Series(window_s=30.0)
    series.push(0.50, ts=0.0)
    series.push(0.80, ts=3.0)
    assert abs(series.velocity() - 0.10) < 1e-9


def test_series_drops_points_outside_window():
    series = _Series(window_s=10.0)
    series.push(0.50, ts=0.0)
    series.push(0.60, ts=6.0)
    series.push(0.70, ts=12.0)  # ts=0.0 (age 12s) now outside the 10s window
    # ts=0.0 is evicted; velocity is measured over the retained points.
    assert abs(series.velocity() - (0.70 - 0.60) / (12.0 - 6.0)) < 1e-9


# --------------------------------------------------------------------------
# Risk manager
# --------------------------------------------------------------------------
def test_risk_manager_approves_within_limits():
    risk = RiskManager(RiskConfig())
    count, reason = risk.approve(price_cents=50)
    assert count > 0 and reason == ""


def test_risk_manager_rejects_extreme_price():
    risk = RiskManager(RiskConfig())
    count, _ = risk.approve(price_cents=99)
    assert count == 0


def test_risk_manager_kill_switch_on_daily_loss():
    cfg = RiskConfig()
    risk = RiskManager(cfg)
    risk.record_realized(-cfg.daily_loss_limit_cents)
    assert risk.kill_switch is True
    count, reason = risk.approve(price_cents=50)
    assert count == 0 and "kill switch" in reason


# --------------------------------------------------------------------------
# Executor
# --------------------------------------------------------------------------
def _interlock_off_config() -> ExecutionConfig:
    return dataclasses.replace(ExecutionConfig(), live_trading_enabled=False)


def test_fee_formula():
    executor = KalshiExecutor(_interlock_off_config(), RiskManager(RiskConfig()), None)
    # 0.07 * 10 contracts * 0.5 * 0.5 = 0.175 dollars -> 18 cents (rounded up).
    assert executor._fee_cents(contracts=10, price_cents=50) == 18


def test_executor_interlock_blocks_orders():
    executor = KalshiExecutor(_interlock_off_config(), RiskManager(RiskConfig()), None)
    signal = EdgeSignal(
        market_ticker="T", side=Side.YES, p_clean=0.40, p_model=0.60,
        edge=0.20, abs_edge=0.20, line_velocity=0.0, model_velocity=0.0,
        mishandled_surge=False, target_price_cents=45, game_event="test",
    )
    result = asyncio.run(executor.execute(signal))
    assert result.accepted is False
    assert result.live is False
    assert "interlock" in result.reject_reason
    assert len(executor.results) == 1   # outcome recorded
    assert len(executor.fills) == 0     # nothing transmitted


# --------------------------------------------------------------------------
# Engine signal logic
# --------------------------------------------------------------------------
def _evaluate_once(odds: MarketOdds) -> EdgeSignal | None:
    async def _run():
        engine = PullbackEngine(StrategyConfig())
        engine.update_game_state(_game_state())  # inning 5, tied -> P_model ~0.50
        engine.update_odds(odds)
        signal = await engine.evaluate()
        engine.shutdown()
        return signal

    return asyncio.run(_run())


def test_engine_fires_when_market_misprices():
    # Market prices home far too low; model ~0.50 -> large positive edge.
    signal = _evaluate_once(
        MarketOdds(market_ticker="T", yes_bid=18, yes_ask=22, no_bid=78, no_ask=82)
    )
    assert signal is not None
    assert signal.side == Side.YES          # buy the under-priced side
    assert signal.abs_edge > StrategyConfig().alpha_threshold


def test_engine_silent_when_market_is_fair():
    # Market priced close to the model -> no edge.
    signal = _evaluate_once(
        MarketOdds(market_ticker="T", yes_bid=49, yes_ask=52, no_bid=48, no_ask=51)
    )
    assert signal is None


def test_engine_silent_on_final_game():
    # Game over: model is 1.0; even a lagging market must not produce a signal.
    async def _run():
        engine = PullbackEngine(StrategyConfig())
        engine.update_game_state(_game_state(
            inning=9, half_inning=HalfInning.BOTTOM, outs=3,
            home_score=5, away_score=2, is_final=True,
        ))
        engine.update_odds(
            MarketOdds(market_ticker="T", yes_bid=78, yes_ask=82, no_bid=18, no_ask=22)
        )
        signal = await engine.evaluate()
        engine.shutdown()
        return signal

    assert asyncio.run(_run()) is None
