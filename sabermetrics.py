"""Sabermetric reference data and player-quality adjustments.

Provides ballpark run factors, league baselines, and an async cache that pulls
season stat lines from the public MLB StatsAPI. The win-probability model uses
these to adjust the run environment for the live pitcher/batter matchup and the
ballpark, rather than assuming a league-average context everywhere.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import aiohttp
from pydantic import BaseModel, ConfigDict

log = logging.getLogger("saber")


# --------------------------------------------------------------------------
# Park factors
# --------------------------------------------------------------------------
# Approximate multiplicative run park factors (1.00 = neutral run environment).
# Keyed by lower-cased StatsAPI venue name. These are coarse, season-agnostic
# estimates -- refresh them from a current source before relying on live edges.
PARK_FACTORS: dict[str, float] = {
    "coors field": 1.15,
    "fenway park": 1.03,
    "great american ball park": 1.06,
    "globe life field": 1.01,
    "chase field": 1.04,
    "citizens bank park": 1.04,
    "wrigley field": 1.02,
    "yankee stadium": 1.03,
    "oriole park at camden yards": 1.02,
    "truist park": 1.00,
    "dodger stadium": 0.98,
    "oracle park": 0.91,
    "t-mobile park": 0.93,
    "petco park": 0.96,
    "loandepot park": 0.97,
    "comerica park": 0.96,
    "kauffman stadium": 1.00,
    "angel stadium": 0.98,
    "rate field": 1.02,
    "guaranteed rate field": 1.02,
    "progressive field": 0.98,
    "target field": 1.00,
    "american family field": 1.02,
    "busch stadium": 0.96,
    "pnc park": 0.97,
    "nationals park": 1.01,
    "citi field": 0.96,
    "daikin park": 1.01,
    "minute maid park": 1.01,
    "tropicana field": 0.96,
    "rogers centre": 1.02,
    "oakland coliseum": 0.94,
    "sutter health park": 1.06,
    "george m. steinbrenner field": 1.03,
}
NEUTRAL_PARK = 1.00


def park_factor(venue: str) -> float:
    """Run park factor for a venue name; 1.00 if unknown."""
    if not venue:
        return NEUTRAL_PARK
    key = venue.strip().lower()
    if key in PARK_FACTORS:
        return PARK_FACTORS[key]
    # Tolerate minor naming variations (sponsor changes, punctuation).
    for name, factor in PARK_FACTORS.items():
        if name in key or key in name:
            return factor
    return NEUTRAL_PARK


# --------------------------------------------------------------------------
# League baselines
# --------------------------------------------------------------------------
class LeagueBaselines(BaseModel):
    """Recent-season league averages used to normalize player quality."""

    model_config = ConfigDict(frozen=True)

    era: float = 4.15
    ops: float = 0.715


LEAGUE = LeagueBaselines()


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# --------------------------------------------------------------------------
# Player stat lines + run-environment factors
# --------------------------------------------------------------------------
class PitcherStats(BaseModel):
    """Season pitching line. `run_factor` < 1 suppresses scoring."""

    model_config = ConfigDict(frozen=True)

    player_id: int
    name: str = ""
    era: float = LEAGUE.era
    innings_pitched: float = 0.0

    def run_factor(self, league: LeagueBaselines = LEAGUE) -> float:
        """Multiplier on expected runs, regressed to the mean by workload."""
        regression_ip = 45.0
        adjusted = (
            self.era * self.innings_pitched + league.era * regression_ip
        ) / (self.innings_pitched + regression_ip)
        return _clamp(adjusted / league.era, 0.62, 1.45)


class BatterStats(BaseModel):
    """Season hitting line. `run_factor` > 1 boosts scoring."""

    model_config = ConfigDict(frozen=True)

    player_id: int
    name: str = ""
    ops: float = LEAGUE.ops
    plate_appearances: int = 0

    def run_factor(self, league: LeagueBaselines = LEAGUE) -> float:
        """Multiplier on expected runs, regressed to the mean by sample size."""
        regression_pa = 80.0
        pa = max(self.plate_appearances, 0)
        adjusted = (self.ops * pa + league.ops * regression_pa) / (
            pa + regression_pa
        )
        return _clamp(adjusted / league.ops, 0.70, 1.45)


def _parse_innings(raw: object) -> float:
    """StatsAPI reports innings as e.g. '45.1' meaning 45 + 1/3 innings."""
    try:
        text = str(raw)
        if "." in text:
            whole, frac = text.split(".", 1)
            return int(whole) + (int(frac[0]) / 3.0 if frac else 0.0)
        return float(text)
    except (ValueError, TypeError):
        return 0.0


def _first_split(payload: dict) -> dict:
    """Extract the first season-stat split from a StatsAPI /stats response."""
    try:
        return payload["stats"][0]["splits"][0]["stat"]
    except (KeyError, IndexError, TypeError):
        return {}


class PlayerStatsCache:
    """Async, lazily-populated cache of pitcher/batter season stats.

    `ensure` fetches any missing players (deduplicating in-flight requests);
    `pitcher`/`batter` are synchronous dict reads safe to call from the
    win-probability worker thread.
    """

    def __init__(self, statsapi_base: str) -> None:
        self._base = statsapi_base
        self._pitchers: dict[int, PitcherStats] = {}
        self._batters: dict[int, BatterStats] = {}
        self._inflight: set[int] = set()

    def pitcher(self, player_id: Optional[int]) -> Optional[PitcherStats]:
        return self._pitchers.get(player_id) if player_id else None

    def batter(self, player_id: Optional[int]) -> Optional[BatterStats]:
        return self._batters.get(player_id) if player_id else None

    async def ensure(
        self,
        session: aiohttp.ClientSession,
        pitcher_id: Optional[int],
        batter_id: Optional[int],
    ) -> None:
        """Fetch any not-yet-cached players. Never raises into the caller."""
        tasks = []
        if (
            pitcher_id
            and pitcher_id not in self._pitchers
            and pitcher_id not in self._inflight
        ):
            tasks.append(self._fetch_pitcher(session, pitcher_id))
        if (
            batter_id
            and batter_id not in self._batters
            and batter_id not in self._inflight
        ):
            tasks.append(self._fetch_batter(session, batter_id))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _fetch_stat(
        self, session: aiohttp.ClientSession, player_id: int, group: str
    ) -> dict:
        url = f"{self._base}/v1/people/{player_id}/stats"
        params = {"stats": "season", "group": group}
        async with session.get(
            url, params=params, timeout=aiohttp.ClientTimeout(total=8.0)
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def _fetch_pitcher(
        self, session: aiohttp.ClientSession, player_id: int
    ) -> None:
        self._inflight.add(player_id)
        try:
            stat = _first_split(await self._fetch_stat(session, player_id, "pitching"))
            stats = PitcherStats(
                player_id=player_id,
                era=float(stat.get("era", LEAGUE.era) or LEAGUE.era),
                innings_pitched=_parse_innings(stat.get("inningsPitched", 0)),
            )
            self._pitchers[player_id] = stats
            log.info(
                "cached pitcher %d: ERA=%.2f IP=%.1f run_factor=%.3f",
                player_id, stats.era, stats.innings_pitched, stats.run_factor(),
            )
        except Exception as exc:  # noqa: BLE001 - degrade to neutral, never crash
            log.warning("pitcher %d stats unavailable (%s); using neutral", player_id, exc)
            self._pitchers[player_id] = PitcherStats(player_id=player_id)
        finally:
            self._inflight.discard(player_id)

    async def _fetch_batter(
        self, session: aiohttp.ClientSession, player_id: int
    ) -> None:
        self._inflight.add(player_id)
        try:
            stat = _first_split(await self._fetch_stat(session, player_id, "hitting"))
            stats = BatterStats(
                player_id=player_id,
                ops=float(stat.get("ops", LEAGUE.ops) or LEAGUE.ops),
                plate_appearances=int(stat.get("plateAppearances", 0) or 0),
            )
            self._batters[player_id] = stats
            log.info(
                "cached batter %d: OPS=%.3f PA=%d run_factor=%.3f",
                player_id, stats.ops, stats.plate_appearances, stats.run_factor(),
            )
        except Exception as exc:  # noqa: BLE001 - degrade to neutral, never crash
            log.warning("batter %d stats unavailable (%s); using neutral", player_id, exc)
            self._batters[player_id] = BatterStats(player_id=player_id)
        finally:
            self._inflight.discard(player_id)
