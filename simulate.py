"""Offline end-to-end simulation.

Runs the real engine and executor against the scripted mock feeders, so the
full pipeline can be watched detecting a pullback edge -- with no network, no
API keys, and no possibility of a real order (the live interlock is forced
off here regardless of the environment).

    python simulate.py
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging

from config import AppConfig
from engine import PullbackEngine
from executor import KalshiExecutor, RiskManager
from main import consume
from mockfeed import MockGameFeeder, MockOddsFeeder
from sabermetrics import PlayerStatsCache


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)-9s %(message)s",
        datefmt="%H:%M:%S",
    )


async def run() -> None:
    log = logging.getLogger("sim")

    # Force the interlock OFF: a simulation must never be able to transmit.
    cfg = AppConfig()
    cfg = dataclasses.replace(
        cfg,
        execution=dataclasses.replace(cfg.execution, live_trading_enabled=False),
    )

    queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
    stop = asyncio.Event()

    stats_cache = PlayerStatsCache(cfg.mlb.statsapi_base)
    engine = PullbackEngine(cfg.strategy, cfg.mlb.regulation_innings, stats_cache)
    risk = RiskManager(cfg.risk)
    executor = KalshiExecutor(cfg.execution, risk, client=None)

    game_feeder = MockGameFeeder(queue)
    odds_feeder = MockOddsFeeder(queue)

    log.info("=== OFFLINE SIMULATION (no network, no orders) ===")
    tasks = [
        asyncio.create_task(game_feeder.run(stop), name="mock-game"),
        asyncio.create_task(odds_feeder.run(stop), name="mock-odds"),
        asyncio.create_task(
            consume(cfg, queue, engine, executor, stop), name="consumer"
        ),
    ]
    # If any task exits unexpectedly, trip `stop` so the run never hangs.
    for task in tasks:
        task.add_done_callback(lambda _t: stop.set())
    await stop.wait()
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    engine.shutdown()

    detected = len(executor.results)
    log.info("=== SIMULATION COMPLETE ===")
    log.info("edges detected: %d | risk: %s", detected, risk.snapshot())
    if detected:
        for result in executor.results:
            sig = result.signal
            log.info(
                "  %s edge=%+.3f LI=%.2f surge=%s -> %s",
                sig.side.value.upper(), sig.edge, sig.leverage_index,
                "Y" if sig.mishandled_surge else "N",
                "TX" if result.live else "blocked-by-interlock",
            )
    else:
        log.info("  no edge cleared the alpha threshold")


if __name__ == "__main__":
    _setup_logging()
    asyncio.run(run())
