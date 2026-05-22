# mlb-k

Asynchronous framework that monitors live MLB game state and Kalshi market
odds, and flags mean-reversion ("pullback") opportunities where the live line
overreacts to an in-game event.

## Modules

| File              | Role |
|-------------------|------|
| `config.py`       | Centralized parameters: API creds, risk limits, alpha threshold, momentum windows. |
| `models.py`       | Typed `GameState`, `MarketOdds`, `EdgeSignal`, `OrderResult` (Pydantic v2). |
| `kalshi.py`       | Kalshi key-pair auth (RSA-PSS request signing) + async REST client. |
| `sabermetrics.py` | Ballpark run factors + async StatsAPI cache of pitcher/batter season stats. |
| `feeder.py`       | Real feeds: MLB StatsAPI play-by-play poller + Kalshi market-data WebSocket. |
| `engine.py`       | Vig stripping, RE24 win-probability model, line momentum, signal logic. |
| `executor.py`     | Risk manager + live Kalshi order placement with slippage/fee accounting. |
| `main.py`         | Unified asyncio orchestrator. |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # then fill in credentials and targets
python main.py
```

## Strategy summary

1. **Vig strip** — Kalshi YES/NO asks are converted to implied probabilities;
   the bid/ask overround is removed proportionally: `P_clean = P_i / ΣP_i`.
2. **Sabermetric baseline** — a static 24-state Run Expectancy Matrix (RE24)
   feeds a win-probability model: expected final run differential mapped
   through a normal distribution whose spread grows with innings remaining.
   The run environment is adjusted for:
   * **Park** — a ballpark run factor scales scoring for both teams.
   * **Pitcher** — the live pitcher's ERA (regressed to the mean by workload)
     suppresses or inflates the current half-inning's expected runs.
   * **Batter** — the live batter's OPS (regressed by sample size) does the
     same, weighted since he is only one of the hitters due up.
3. **Leverage** — each estimate reports a Leverage Index: win-probability
   sensitivity to a run at the score-and-innings state, normalized so a
   neutral mid-game spot is ~1.0 (close + late ⇒ high, decided ⇒ ~0). Set
   `MIN_LEVERAGE_INDEX` to ignore edges in low-leverage situations.
4. **Pullback trigger** — line velocity (`dProb/dt`) is tracked over a 30s
   window; a "mishandled surge" is flagged when the market lurches far faster
   than the model. A buy fires when `|P_clean − P_model|` exceeds the
   configurable alpha threshold.

## Safety — read before funding

* **Live-trading interlock.** `executor.py` will not transmit a real order
  unless `KALSHI_LIVE_TRADING_ENABLED=true`. Hard caps (max contracts, max
  position cost, daily-loss kill switch) always apply.
* **Untested live paths.** The Kalshi order and WebSocket-auth code is written
  to Kalshi's documented API but has **not** been tested against the exchange.
  Verify it on Kalshi's demo environment (`demo-api.kalshi.co`) first.
* **The strategy is unproven.** The model adjusts for park and the live
  pitcher/batter matchup and reports leverage, but it is still a simplified
  baseline: no bullpen projection, batting-order context, platoon splits,
  weather, or a full base-out leverage tree. Park factors are coarse static
  estimates. Treat any profit expectation skeptically; validate against real
  market data before funding.
