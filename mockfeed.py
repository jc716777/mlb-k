"""Scripted mock feeders for offline end-to-end testing.

These stand in for the live MLB / Kalshi feeds so the full pipeline (engine,
momentum tracking, signal logic, executor) can be exercised with no network,
no API credentials, and no possibility of a real order.

The scripted scenario is a textbook "mishandled surge": a two-out walk -- an
event the win-probability model knows is nearly irrelevant -- triggers a
violent overreaction in the betting line. The engine should fade it.
"""

from __future__ import annotations

import asyncio
import logging

from models import GameState, HalfInning, MarketOdds

log = logging.getLogger("mockfeed")

MARKET_TICKER = "KXMLBGAME-MOCK-HOME"


class MockGameFeeder:
    """Emits two game states: a 2-out situation, then a walk that loads on."""

    def __init__(self, queue: "asyncio.Queue") -> None:
        self._queue = queue

    @staticmethod
    def _state(runners: tuple[bool, bool, bool], event: str) -> GameState:
        return GameState(
            game_pk=999000,
            inning=8,
            half_inning=HalfInning.BOTTOM,  # home batting
            outs=2,
            runners_on_base=runners,
            home_score=4,
            away_score=4,
            venue="Mock Park",
            last_event=event,
        )

    async def run(self, stop: asyncio.Event) -> None:
        # Pre-event: tie game, runner on second, two outs.
        await self._emit(
            self._state((False, True, False), "Flyout to center."), stop
        )
        # The "irrelevant" event: a two-out walk. No run scores, outs unchanged;
        # the model barely moves. Wait while the calm odds ticks stream.
        await asyncio.sleep(1.9)
        await self._emit(
            self._state((True, True, False), "Walk. Runners on first and second."),
            stop,
        )

    async def _emit(self, state: GameState, stop: asyncio.Event) -> None:
        if stop.is_set():
            return
        log.info("MOCK game event: %s", state.last_event)
        await self._queue.put(state)


class MockOddsFeeder:
    """Emits a calm, fair line, then a surge far outpacing the fundamentals."""

    def __init__(self, queue: "asyncio.Queue") -> None:
        self._queue = queue

    async def run(self, stop: asyncio.Event) -> None:
        # (delay_before_s, yes_bid). yes_ask is yes_bid + 3 (a 3c book).
        # Calm phase: priced near the model (~0.57 vig-free).
        calm = [(0.30, 56)] * 6
        # Surge phase: the line rockets after the (near-irrelevant) walk.
        surge = [(0.30, b) for b in (59, 63, 68, 73, 78, 83, 87, 89)]
        tail = [(0.40, 90), (0.40, 90)]

        for delay, yes_bid in calm + surge + tail:
            await asyncio.sleep(delay)
            if stop.is_set():
                return
            yes_ask = yes_bid + 3
            await self._queue.put(
                MarketOdds(
                    market_ticker=MARKET_TICKER,
                    yes_bid=yes_bid,
                    yes_ask=yes_ask,
                    no_bid=100 - yes_ask,
                    no_ask=100 - yes_bid,
                )
            )

        # Give the consumer a moment to drain the queue, then end the run.
        await asyncio.sleep(1.0)
        log.info("MOCK scenario complete")
        stop.set()
