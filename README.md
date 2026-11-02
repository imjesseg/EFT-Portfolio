# ETF portfolio agent

Give it a target asset mix; it ranks ETFs in each category on 1, 3, and 12-month **peer-relative** performance and **reward/risk**, then proposes sample portfolios and backtests them against a category-average baseline.

```bash
python3 -m etf_agent "40% short term fixed income, 10% senior AAA rated fixed income, 10% commodities, 20% growth equities, 20% defensive equities"
```

No dependencies beyond Python 3.10+. Prices are daily adjusted closes from Yahoo Finance's public chart endpoint, cached in `.cache/prices/` for the day. Optionally `pip install -e .` to get an `etf-agent` command.

In Amp, just ask in this repo (for example "suggest ETFs for 60% growth, 40% short-term bonds, conservative") — the `suggesting-etf-portfolios` skill runs the agent and summarizes the result.

## How it scores

For each ETF and each horizon (1m = 21, 3m = 63, 12m = 252 trading days):

- **Return** — total return including dividends.
- **vs peer median** — return minus the median of its category.
- **Reward/risk** — (return − T-bill return from BIL) ÷ volatility over that horizon (a per-horizon Sharpe ratio).

Within each category, an ETF's score is its percentile rank on reward/risk and on return, blended by `--risk-weight` (default 0.5) and averaged across horizons with `--horizon-weights` (default 0.2,0.3,0.5). Ranking is only against peers, so a T-bill fund is never compared to a growth fund.

## Sample portfolios

| Portfolio | Construction |
|---|---|
| Best-in-class | Top-scoring ETF per category gets the whole category weight |
| Diversified | Top `--picks` (3) per category, skipping near-duplicates (daily-return correlation ≥ `--max-corr`, 0.95, e.g. QQQ vs QQQM), inverse-volatility weighted |
| Low-volatility | Two lowest-volatility ETFs from the top half of each category, inverse-volatility weighted |
| Category average | Every ETF in each category equally weighted — the baseline to beat |

Each is backtested over the last 12 months with daily rebalancing (returns, volatility, reward/risk, max drawdown). The backtest uses the same window the ETFs were selected on, so it is optimistic; past performance does not predict future returns, and this is not investment advice.

## Fidelity mutual fund equivalents

Find the Fidelity mutual fund that most closely tracks any ETF:

```bash
python3 -m etf_agent --match XLP QQQ SGOV
```

Or add `--fidelity` to a portfolio run to get a Fidelity equivalent for every suggested ETF. Funds are ranked by **tracking error** (annualized volatility of the daily return difference over the last 12 months), which also penalizes funds that move in the same direction but by more or less than the ETF. Each match shows correlation, beta, the 12-month return gap, and a close / fair / loose label; "loose" means Fidelity has no mutual fund with that exposure (e.g. gold bullion or commodity futures). Fund tickers link to their Fidelity research page.

The candidate list of ~100 retail Fidelity mutual funds (Select sector, index, active equity, allocation, and bond funds) lives in [`etf_agent/fidelity_funds.json`](etf_agent/fidelity_funds.json). Money market funds are excluded because their constant $1 price carries no tracking information.

## Categories

`python3 -m etf_agent --list-categories` shows every category, the phrases it matches, and its ETFs. Edit [`etf_agent/universe.json`](etf_agent/universe.json) to add categories or tickers, or pass `--universe my.json`. Allocations can also use exact keys: `"short_term_fixed_income=40, growth_equity=60"`.

## Tests

```bash
python3 -m unittest discover -s tests
```
