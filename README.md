# ETF portfolio agent

Give it a target asset mix; it ranks ETFs in each category on 1, 3, and 12-month **peer-relative** performance and **reward/risk**, then proposes sample portfolios and backtests them against a category-average baseline.

```bash
python3 -m etf_agent "40% short term fixed income, 10% senior AAA rated fixed income, 10% commodities, 20% growth equities, 20% defensive equities"
```

No dependencies beyond Python 3.10+. Optionally `pip install -e .` to get an `etf-agent` command.

For a slide-style overview of the approach, options, usage, and caveats, open [`docs/overview.html`](docs/overview.html) in a browser (arrow keys move between slides; Print saves a PDF, one slide per page). The same deck is in PowerPoint form, with editable text and tables, as [`docs/overview.pptx`](docs/overview.pptx).

**Price data.** Daily dividend-adjusted closes come from Yahoo Finance's public chart endpoint and are cached in `.cache/prices/` for the day. Yahoo has no official API, published limits, or guarantees, and it throttles heavy or cloud traffic. If a download fails and the `TIINGO_API_KEY` environment variable is set, the tool falls back to [Tiingo](https://www.tiingo.com)'s official end-of-day API for that ticker. Tiingo's free plan (sign up for a key) allows 50 requests an hour, 1,000 a day, and 500 different tickers a month, and covers ETFs and mutual funds. Google Finance has no API and MSN's is undocumented, so neither is used.

In Amp, just ask in this repo (for example "suggest ETFs for 60% growth, 40% short-term bonds, conservative") — the `suggesting-etf-portfolios` skill runs the agent and summarizes the result.

## How it scores

For each ETF and each horizon (1m = 21, 3m = 63, 12m = 252 trading days):

- **Return** — total return including dividends.
- **vs peer median** — return minus the median of its category.
- **Reward/risk** — (return − T-bill return from BIL) ÷ volatility over that horizon (a per-horizon Sharpe ratio).
- **Sortino** — (return − T-bill return) ÷ downside deviation over that horizon. Only days that trailed T-bills count as risk, so upside swings are not penalized.
- **Calmar** — (return − T-bill return) ÷ worst peak-to-trough drawdown over that horizon. Drawdowns under 0.1% count as 0.1%, so near-cash funds don't get near-infinite ratios.

Within each category, an ETF's score is its percentile rank on reward/risk and on return, blended by `--risk-weight` (default 0.5) and averaged across horizons with `--horizon-weights` (default 0.2,0.3,0.5). Ranking is only against peers, so a T-bill fund is never compared to a growth fund.

### `--measure sharpe|sortino|calmar`

Chooses the risk-adjusted measure behind every decision. `--sortino` and `--calmar` are shortcuts.

| | `sharpe` (default) | `sortino` | `calmar` |
|---|---|---|---|
| Risk counted | All volatility | Days that trailed T-bills | Worst drawdown |
| Ranking uses | Reward/risk | Sortino | Calmar |
| Weights within Diversified and Low-volatility | Inverse volatility | Proportional to 12-month Sortino | Proportional to 12-month Calmar |
| Fidelity fund suggested | Lowest tracking error | Highest Sortino among equally good matches | Highest Calmar among equally good matches |

With `sortino` or `calmar`, picks whose 12-month ratio is ≤ 0 (no return above T-bills) get no weight, so a category can end up in a single ETF. Fidelity matching always shortlists the 3 funds with the lowest tracking error first and only reorders those whose match quality is as good as the closest fund's, so a better-performing fund never wins by being a worse match.

## Sample portfolios

| Portfolio | Construction |
|---|---|
| Best-in-class | Top-scoring ETF per category gets the whole category weight |
| Diversified | Top `--picks` (3) per category, skipping near-duplicates (daily-return correlation ≥ `--max-corr`, 0.95, e.g. QQQ vs QQQM), inverse-volatility weighted (or by `--measure`) |
| Low-volatility | Two lowest-volatility ETFs from the top half of each category, inverse-volatility weighted (or by `--measure`) |
| Category average | Every ETF in each category equally weighted — the baseline to beat |

Each is backtested over the last 12 months with daily rebalancing (returns, volatility, reward/risk, Sortino, Calmar, max drawdown). The backtest uses the same window the ETFs were selected on, so it is optimistic; past performance does not predict future returns, and this is not investment advice.

## Fidelity mutual fund equivalents

Find the Fidelity mutual fund that most closely tracks any ETF:

```bash
python3 -m etf_agent --match XLP QQQ SGOV
```

Or add `--fidelity` to a portfolio run to get a Fidelity equivalent for every suggested ETF. Funds are ranked by **tracking error** (annualized volatility of the daily return difference over the last 12 months), which also penalizes funds that move in the same direction but by more or less than the ETF. Each match shows correlation, beta, the 12-month return gap, the 12-month Sortino (or Calmar with `--calmar`) of the ETF and the fund, and a close / fair / loose label; "loose" means Fidelity has no mutual fund with that exposure (e.g. gold bullion or commodity futures). Fund tickers link to their Fidelity research page.

The candidate list of ~100 retail Fidelity mutual funds (Select sector, index, active equity, allocation, and bond funds) lives in [`etf_agent/fidelity_funds.json`](etf_agent/fidelity_funds.json). Money market funds are excluded because their constant $1 price carries no tracking information.

## Daily portfolio monitor

Create a portfolio file from what you own, then check it:

```bash
python3 -m etf_agent --import-holdings positions.csv "40% short term, 10% AAA, 10% commodities, 20% growth, 20% defensive" > portfolio.json
python3 -m etf_agent --monitor portfolio.json --sortino
```

`--import-holdings` reads either of two formats; see the samples in [`examples/`](examples/):

- **Fidelity positions export** ([`fidelity_positions.csv`](examples/fidelity_positions.csv)): on Fidelity.com, open **Positions** and choose **Download**. Money market core positions (such as `SPAXX**`) count as cash; pending activity is skipped. For an export with several accounts, choose one with `--account NUMBER`. Columns are found by name (`Symbol`, `Quantity`, `Current Value`, `Account Number`).
- **A plain list** ([`holdings.txt`](examples/holdings.txt)): one `TICKER SHARES [CATEGORY]` per line, plus `cash AMOUNT`. Money market funds (five-letter tickers ending in `XX`, such as `FDRXX`) count as cash.

With a mix, each category's target is split among the funds you hold there by `--measure` (inverse volatility by default; Sortino or Calmar weights can set a fund with no return above T-bills to 0%). Without a mix, your current weights, including cash, become the targets. Funds outside the ETF lists, such as Fidelity mutual funds, are placed in the category of the ETF they track most closely, and the import says so. Use `--account-type taxable` for brokerage accounts. Review the printed file before relying on it.

The portfolio file ([`examples/portfolio.json`](examples/portfolio.json)) lists each holding's `ticker`, `shares`, and `target` (percent), plus a `category` for funds outside the ETF lists. You can also write or edit it by hand. `cash` is counted in the total. Set `cash_target` (percent) to keep part of the portfolio in cash, for example a money market fund held as your short-term allocation; holding targets plus `cash_target` must add up to 100. Cash is checked against its target with the same 5/25 rule, and proposed trades leave it at its target. Without a `cash_target`, cash is invested once it passes 5% of the portfolio. `portfolio.json` is git-ignored so your holdings stay private. Don't commit it even once: git keeps tracking a committed file and ignores `.gitignore` for it, so later edits with real holdings would reach the public repo.

To try the monitor without your own holdings, use the committed sample [`examples/three_funds.json`](examples/three_funds.json), built from [`examples/three_funds.txt`](examples/three_funds.txt): $100,000 in the FDRXX money market (kept as cash with a 21.2% `cash_target`), FDFAX, and FBGRX.

```bash
python3 -m etf_agent --monitor examples/three_funds.json
```

The check suggests a rebalance when either rule fires:

- **Drift** (5/25 rule): a holding's weight is more than `drift_points` (5) percentage points or more than `drift_relative` (25%) of its target away from its target.
- **Persistent laggard**: a holding ranked in the bottom half of its category at each of the last `laggard_weeks` (4) weekly checks, and the best ETF it doesn't hold beat its score by at least `replace_margin` (0.25) at every one of them. One bad week never triggers a swap. Scores use the same ranking and `--measure` as the rest of the tool.

When a rule fires, it proposes trades back to the targets (with any replacement taking over the laggard's target), in dollars and approximate shares. If any holding's price is missing or more than `stale_days` (4) days old, it suggests nothing. Rules can be overridden in the file's `rules` object. `--json` prints a machine-readable report whose `action_needed` field says whether to act.

Set `"account_type": "tax_deferred"` for IRAs and 401(k)s, where taxes don't apply. `"taxable"` adds capital-gains and wash-sale warnings to proposed sales. Lot-level cost basis isn't tracked yet. Replacement candidates come from the category's ETF list; check that your plan offers them, or use `--match` for a Fidelity mutual fund equivalent.

### Email and scheduling

Add `--email` to send the report by email only when a rebalance is suggested or prices are missing, so quiet days stay quiet. Use `--email always` to get it after every check. Mail goes through any SMTP server, configured with environment variables. For Gmail, use an [app password](https://myaccount.google.com/apppasswords), which needs 2-Step Verification:

```bash
# ~/.etf-agent.env  (chmod 600 — it holds a password)
export SMTP_HOST=smtp.gmail.com
export SMTP_PORT=587            # 465 for SSL-only servers
export SMTP_USER=you@gmail.com
export SMTP_PASSWORD='your-app-password'
export EMAIL_TO=you@gmail.com   # comma-separate several addresses
export TIINGO_API_KEY=...       # optional backup price source
```

Then schedule it with `crontab -e`, for example at 9 p.m. on weekdays, after mutual fund prices are published:

```cron
0 21 * * 1-5  . $HOME/.etf-agent.env && cd $HOME/EFT-Portfolio && python3 -m etf_agent --monitor portfolio.json --sortino --email >> $HOME/etf-monitor.log 2>&1
```

Cron uses the machine's time zone, so adjust the hour if it isn't Eastern. Until you have your own `portfolio.json`, you can point it at `examples/three_funds.json`.

## Categories

`python3 -m etf_agent --list-categories` shows every category, the phrases it matches, and its ETFs. Edit [`etf_agent/universe.json`](etf_agent/universe.json) to add categories or tickers, or pass `--universe my.json`. Allocations can also use exact keys: `"short_term_fixed_income=40, growth_equity=60"`.

## Tests

```bash
python3 -m unittest discover -s tests
```
