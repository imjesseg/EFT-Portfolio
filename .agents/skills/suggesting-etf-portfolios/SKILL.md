---
name: suggesting-etf-portfolios
description: Suggests and monitors ETF portfolios. Suggests sample ETF portfolios for a target sector/asset-class mix by ranking ETFs on 1, 3, and 12-month peer-relative performance and reward/risk, and finds the Fidelity mutual funds that most closely track given ETFs. Use when asked for ETF ideas, portfolio suggestions, the best ETFs for an allocation such as "40% short-term fixed income, 20% growth equities", a Fidelity fund equivalent of an ETF (e.g. XLP to FDFAX), or a check of whether an existing portfolio needs rebalancing.
---

# Suggesting ETF portfolios

Run the bundled agent from the repository root (Python 3.10+, no dependencies):

```bash
python3 -m etf_agent "40% short term fixed income, 10% senior AAA rated fixed income, 10% commodities, 20% growth equities, 20% defensive equities"
```

1. Pass the user's mix as free text. Percentages are matched to categories by alias; run `python3 -m etf_agent --list-categories` if a category does not match, and use the exact key (e.g. `aaa_fixed_income=10`).
2. Map the user's preferences to flags:
   - More return-seeking: `--risk-weight 0.2`; more risk-averse: `--risk-weight 0.8` (default 0.5).
   - Emphasize recent momentum: `--horizon-weights 0.4,0.4,0.2` (default 0.2,0.3,0.5 for 1m,3m,12m).
   - Risk measure: `--sortino` (downside deviation) or `--calmar` (worst drawdown) replaces volatility for ranking, weighting picks, and choosing among equally good Fidelity matches. Use Sortino when the user worries about losses rather than swings, Calmar when they care most about the worst drop.
   - More or fewer funds: `--picks N` (Diversified portfolio), `--top N` (rows per ranking table).
   - Fidelity mutual fund equivalents for the suggested ETFs: `--fidelity`. For specific ETFs only: `python3 -m etf_agent --match XLP QQQ`.
   - Check an existing portfolio for drift and laggards: `python3 -m etf_agent --monitor portfolio.json` (copy `portfolio.example.json` to start). Report the headline, any reasons, and the proposed trades. `--email` (or `--email action`) mails the report using the SMTP_* and EMAIL_TO environment variables; see the README for cron setup.
   - Use `--json` when post-processing results.
3. When showing Fidelity equivalents, call out "loose" matches: Fidelity has no mutual fund with that exposure, so the ETF may be the better vehicle.
4. Report the sample portfolios table and the recommended holdings, then point out notable tradeoffs (concentration, duplicates, volatility, drawdown) and how each portfolio compares with the Category average baseline.
5. Always repeat the caveat that backtests are in-sample and past performance does not predict returns.

To add categories or ETFs, edit `etf_agent/universe.json` (each category has `name`, `aliases`, `tickers`).
