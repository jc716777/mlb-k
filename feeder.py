"""Asynchronous live data feeders.

Two concurrent sources push onto a single shared queue:

  * MLBGameFeeder   -- polls the public MLB StatsAPI live feed for play-by-play
                       game state (StatsAPI offers no push socket).
  * KalshiOddsFeeder-- subscribes to the Kalshi market-data WebSocket for
                       top-of-book line movement.

Both reconnect with exponential backoff and never raise into the event loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional, Union

import aiohttp
import websockets

from config import KalshiConfig, MLBConfig
from kalshi import KalshiAuth
from models import GameState, HalfInning, MarketOdds
from sabermetrics import PlayerStatsCache

log = logging.getLogger("feeder")

FeedEvent = Union[GameState, MarketOdds]


async def _backoff(attempt: int, cap: float = 16.0) -> None:
    delay = min(cap, 2.0 ** attempt)
    log.warning("reconnecting in %.0fs (attempt %d)", delay, attempt)
    await asyncio.sleep(delay)


# --------------------------------------------------------------------------
# MLB StatsAPI play-by-play
# --------------------------------------------------------------------------
class MLBGameFeeder:
    """Polls the StatsAPI live feed and emits a GameState only when it changes."""

    def __init__(
        self,
        cfg: MLBConfig,
        queue: "asyncio.Queue[FeedEvent]",
        stats_cache: PlayerStatsCache,
    ) -> None:
        self._cfg = cfg
        self._queue = queue
        self._stats = stats_cache
        self._last_key: Optional[tuple] = None

    @staticmethod
    def _parse(payload: dict) -> Optional[GameState]:
        """Translate a StatsAPI live-feed payload into a GameState."""
        try:
            game_pk = payload["gamePk"]
            game_data = payload["gameData"]
            live = payload["liveData"]
            linescore = live["linescore"]

            abstract = game_data["status"]["abstractGameState"]
            is_final = abstract == "Final"

            half_raw = str(linescore.get("inningHalf", "")).lower()
            if half_raw not in ("top", "bottom"):
                # "Middle"/"End": between half-innings, no batting situation.
                return None
            half = HalfInning.TOP if half_raw == "top" else HalfInning.BOTTOM

            offense = linescore.get("offense", {})
            runners = (
                "first" in offense,
                "second" in offense,
                "third" in offense,
            )

            teams = linescore.get("teams", {})
            home_score = int(teams.get("home", {}).get("runs", 0) or 0)
            away_score = int(teams.get("away", {}).get("runs", 0) or 0)

            current_play = live.get("plays", {}).get("currentPlay", {})
            last_event = current_play.get("result", {}).get("description", "") or ""

            venue = str(game_data.get("venue", {}).get("name", "") or "")
            defense = linescore.get("defense", {})
            pitcher_id = defense.get("pitcher", {}).get("id")
            batter_id = offense.get("batter", {}).get("id")

            return GameState(
                game_pk=game_pk,
                inning=int(linescore.get("currentInning", 1) or 1),
                half_inning=half,
                outs=int(linescore.get("outs", 0) or 0),
                runners_on_base=runners,
                home_score=home_score,
                away_score=away_score,
                pitcher_id=pitcher_id,
                batter_id=batter_id,
                venue=venue,
                last_event=last_event,
                is_final=is_final,
            )
        except (KeyError, TypeError, ValueError) as exc:
            log.error("failed to parse MLB feed: %s", exc)
            return None

    async def run(self, stop: asyncio.Event) -> None:
        attempt = 0
        async with aiohttp.ClientSession() as session:
            while not stop.is_set():
                try:
                    async with session.get(
                        self._cfg.feed_url(),
                        timeout=aiohttp.ClientTimeout(total=10.0),
                    ) as resp:
                        resp.raise_for_status()
                        payload = await resp.json()
                    attempt = 0

                    state = self._parse(payload)
                    if state is not None and state.situation_key() != self._last_key:
                        self._last_key = state.situation_key()
                        # Warm the stats cache so the engine can read it sync.
                        await self._stats.ensure(
                            session, state.pitcher_id, state.batter_id
                        )
                        log.info(
                            "game state: I%d-%s outs=%d bases=%s %d-%d | %s",
                            state.inning, state.half_inning.value, state.outs,
                            state.runners_on_base, state.away_score,
                            state.home_score, state.last_event[:60],
                        )
                        await self._queue.put(state)

                    if state is not None and state.is_final:
                        log.info("game is final; MLB feeder stopping")
                        stop.set()
                        return

                    await asyncio.sleep(self._cfg.poll_interval_s)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 - never kill the loop
                    log.error("MLB feed error: %s", exc)
                    attempt += 1
                    await _backoff(attempt)


# --------------------------------------------------------------------------
# Kalshi market-data WebSocket
# --------------------------------------------------------------------------
class KalshiOddsFeeder:
    """Subscribes to the Kalshi `ticker` channel for one market."""

    def __init__(
        self,
        cfg: KalshiConfig,
        auth: KalshiAuth,
        queue: "asyncio.Queue[FeedEvent]",
    ) -> None:
        self._cfg = cfg
        self._auth = auth
        self._queue = queue

    def _subscribe_msg(self) -> str:
        return json.dumps(
            {
                "id": 1,
                "cmd": "subscribe",
                "params": {
                    "channels": ["ticker"],
                    "market_tickers": [self._cfg.market_ticker],
                },
            }
        )

    @staticmethod
    def _parse(msg: dict) -> Optional[MarketOdds]:
        """Build a MarketOdds from a Kalshi ticker-channel message.

        UNVERIFIED -- likely wrong vs. the current API. Web docs indicate the
        live `ticker` message uses `yes_bid_dollars`/`yes_ask_dollars` (price
        in DOLLARS) and `ts_ms`, not the integer-cent `yes_bid`/`yes_ask`
        fields read below. Confirm against a real message before live use.
        """
        if msg.get("type") != "ticker":
            return None
        body = msg.get("msg", {})
        try:
            yes_bid = int(body["yes_bid"])  # FIXME: likely yes_bid_dollars * 100
            yes_ask = int(body["yes_ask"])  # FIXME: likely yes_ask_dollars * 100
            # Kalshi NO book is the mirror of the YES book.
            no_bid = 100 - yes_ask
            no_ask = 100 - yes_bid
            return MarketOdds(
                market_ticker=body.get("market_ticker", ""),
                yes_bid=yes_bid,
                yes_ask=yes_ask,
                no_bid=no_bid,
                no_ask=no_ask,
            )
        except (KeyError, TypeError, ValueError) as exc:
            log.error("failed to parse Kalshi ticker: %s", exc)
            return None

    async def run(self, stop: asyncio.Event) -> None:
        url = self._cfg.ws_base + self._cfg.ws_path
        attempt = 0
        while not stop.is_set():
            try:
                headers = self._auth.headers("GET", self._cfg.ws_path)
                async with websockets.connect(
                    url, additional_headers=headers, ping_interval=10, ping_timeout=10
                ) as ws:
                    await ws.send(self._subscribe_msg())
                    log.info("Kalshi WS connected, subscribed to %s", self._cfg.market_ticker)
                    attempt = 0

                    async for raw in ws:
                        if stop.is_set():
                            break
                        try:
                            msg = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        odds = self._parse(msg)
                        if odds is not None:
                            log.debug(
                                "odds: yes %d/%d overround=%.3f",
                                odds.yes_bid, odds.yes_ask, odds.overround,
                            )
                            await self._queue.put(odds)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - never kill the loop
                log.error("Kalshi WS error: %s", exc)
                attempt += 1
                await _backoff(attempt)
