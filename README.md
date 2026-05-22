# mlb-k

Asynchronous framework that monitors live MLB game state and Kalshi market
odds, and flags mean-reversion ("pullback") opportunities where the live line
overreacts to an in-game event.

## Modules

| File          | Role |
|---------------|------|
| `config.py`   | Centralized parameters: API creds, risk limits, alpha threshold, momentum windows. |
| `models.py`   | Typed `GameState`, `MarketOdds`, `EdgeSignal`, `OrderResult` (Pydantic v2). |
| `kalshi.py`   | Kalshi key-pair auth (RSA-PSS request signing) + async REST client. |
| `feeder.py`   | Real feeds: MLB StatsAPI play-by-play poller + Kalshi market-data WebSocket. |
| `engine.py`   | Vig stripping, RE24 win-probability model, line momentum, signal logic. |
| `executor.py` | Risk manager + live Kalshi order placement with slippage/fee accounting. |
| `main.py`     | Unified asyncio orchestrator. |

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
3. **Pullback trigger** — line velocity (`dProb/dt`) is tracked over a 30s
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
* **The strategy is unproven.** The RE24 model is a coarse baseline and does
  not account for pitcher/batter quality, leverage, or park. Treat any profit
  expectation skeptically; validate against real market data before funding.
