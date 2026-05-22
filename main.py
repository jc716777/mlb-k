"""Orchestrator: one unified asyncio event loop wiring feeds -> engine ->
executor.

  python main.py        # reads configuration from the environment / .env
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from datetime import datetime, timezone

from config import AppConfig, load_config
from engine import PullbackEngine
from executor import KalshiExecutor, RiskManager
from feeder import FeedEvent, KalshiOddsFeeder, MLBGameFeeder
from kalshi import KalshiAuth, KalshiRestClient
from models import GameState, MarketOdds

log = logging.getLogger("main")


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)-9s %(message)s",
        datefmt="%H:%M:%S",
    )


def _age_seconds(ts: datetime) -> float:
    return (datetime.now(timezone.utc) - ts).total_seconds()


async def consume(
    cfg: AppConfig,
    queue: "asyncio.Queue[FeedEvent]",
    engine: PullbackEngine,
    executor: KalshiExecutor,
    stop: asyncio.Event,
) -> None:
    """Drain the feed queue, drop stale ticks, run the strategy."""
    stale_limit = cfg.strategy.stale_tick_s
    while not stop.is_set():
        try:
            event = await asyncio.wait_for(queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            continue

        age = _age_seconds(event.timestamp)

        if isinstance(event, MarketOdds):
            # Strict stale-data guard: ignore odds ticks older than the limit.
            if age > stale_limit:
                log.warning("dropping stale odds tick (age=%.2fs)", age)
                continue
            engine.update_odds(event)
        elif isinstance(event, GameState):
            engine.update_game_state(event)
            if event.is_final:
                log.info("final game state received; shutting down")
                stop.set()
        else:  # pragma: no cover - queue is typed
            continue

        signal_ = await engine.evaluate()
        if signal_ is not None:
            await executor.execute(signal_)


async def run(cfg: AppConfig) -> None:
    cfg.validate()

    if cfg.execution.live_trading_enabled:
        log.warning("LIVE TRADING ENABLED -- real orders will be sent to Kalshi")
    else:
        log.info("live-trading interlock OFF -- orders will be logged, not sent")

    auth = KalshiAuth(cfg.kalshi)
    queue: "asyncio.Queue[FeedEvent]" = asyncio.Queue(maxsize=1000)
    stop = asyncio.Event()

    engine = PullbackEngine(cfg.strategy, cfg.mlb.regulation_innings)
    risk = RiskManager(cfg.risk)

    # Install signal handlers for a clean shutdown.
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:  # pragma: no cover - non-POSIX
            pass

    async with KalshiRestClient(cfg.kalshi, auth) as client:
        # Confirm credentials and surface the starting balance before trading.
        try:
            balance = await client.get_balance()
            log.info("Kalshi balance: %s", balance)
        except Exception as exc:  # noqa: BLE001
            log.error("could not fetch Kalshi balance: %s", exc)

        executor = KalshiExecutor(cfg.execution, risk, client)
        mlb_feeder = MLBGameFeeder(cfg.mlb, queue)
        odds_feeder = KalshiOddsFeeder(cfg.kalshi, auth, queue)

        tasks = [
            asyncio.create_task(mlb_feeder.run(stop), name="mlb-feeder"),
            asyncio.create_task(odds_feeder.run(stop), name="odds-feeder"),
            asyncio.create_task(
                consume(cfg, queue, engine, executor, stop), name="consumer"
            ),
        ]
        await stop.wait()
        log.info("shutdown requested; cancelling tasks")
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    engine.shutdown()
    log.info("final risk state: %s", risk.snapshot())
    log.info("fills this session: %d", len(executor.fills))


def main() -> int:
    cfg = load_config()
    _setup_logging(cfg.log_level)
    try:
        asyncio.run(run(cfg))
    except ValueError as exc:
        log.error("configuration error: %s", exc)
        return 2
    except KeyboardInterrupt:  # pragma: no cover
        log.info("interrupted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
