"""Portfolio check: drift from target weights and persistent laggards, with a proposed trade list.

Meant to run daily. It only suggests trades; it never places them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from . import analytics as an
from .data import History

DEFAULT_RULES = {
    "drift_points": 5.0,  # rebalance when a weight is more than 5 percentage points off target...
    "drift_relative": 0.25,  # ...or more than 25% of its target (the common "5/25" rule)
    "laggard_weeks": 4,  # replace a fund only after this many weekly checks in the bottom half of its category
    "replace_margin": 0.25,  # ...with the alternative's score ahead by at least this much at every check
    "stale_days": 4,  # a holding priced this many days behind the newest data blocks any action
}


@dataclass
class Holding:
    ticker: str
    shares: float
    target: float  # fraction of the portfolio
    category: str


@dataclass
class Portfolio:
    holdings: list[Holding]
    cash: float
    account_type: str  # "tax_deferred" (IRA, 401k: taxes ignored) or "taxable"
    rules: dict


def load_portfolio(path: Path, universe: dict) -> Portfolio:
    data = json.loads(Path(path).read_text())
    category_of = {t: key for key, spec in universe.items() for t in spec["tickers"]}
    holdings = []
    for h in data["holdings"]:
        ticker = h["ticker"].upper()
        category = h.get("category") or category_of.get(ticker)
        if category not in universe:
            raise ValueError(f"{ticker}: add a \"category\" to its holding (one of: {', '.join(universe)})")
        holdings.append(Holding(ticker, float(h["shares"]), float(h["target"]) / 100, category))
    total = sum(h.target for h in holdings)
    if abs(total - 1) > 0.001:
        raise ValueError(f"Holding targets add up to {total * 100:g}%, not 100%")
    account_type = data.get("account_type", "tax_deferred")
    if account_type not in ("tax_deferred", "taxable"):
        raise ValueError('"account_type" must be "tax_deferred" or "taxable"')
    return Portfolio(holdings, float(data.get("cash", 0)), account_type, {**DEFAULT_RULES, **data.get("rules", {})})


def tickers_needed(portfolio: Portfolio, universe: dict) -> list[str]:
    held = [h.ticker for h in portfolio.holdings]
    peers = [t for c in dict.fromkeys(h.category for h in portfolio.holdings) for t in universe[c]["tickers"]]
    return list(dict.fromkeys(held + peers + [an.RISK_FREE_TICKER]))


def _truncated(history: History, days_back: int) -> History:
    if not days_back:
        return history
    return History(history.ticker, history.name, history.dates[:-days_back], history.prices[:-days_back], history.source)


def _scores(tickers: list[str], histories: dict[str, History], days_back: int, horizon_weights, risk_weight, measure):
    """Category scores as they stood ``days_back`` trading days ago."""
    risk_free = {}
    if an.RISK_FREE_TICKER in histories:
        bil = _truncated(histories[an.RISK_FREE_TICKER], days_back).prices
        risk_free = {h: an.total_return(bil, d) for h, d in an.HORIZONS.items()}
    metrics = [
        m
        for t in tickers
        if t in histories and (m := an.compute_metrics(_truncated(histories[t], days_back), "", risk_free))
    ]
    return {m.ticker: m.score for m in an.score_category(metrics, horizon_weights, risk_weight, measure)} if metrics else {}


def check(portfolio: Portfolio, universe: dict, histories: dict[str, History], horizon_weights, risk_weight, measure) -> dict:
    rules = portfolio.rules
    as_of = max(h.dates[-1] for h in histories.values())
    stale_before = (date.fromisoformat(as_of) - timedelta(days=rules["stale_days"])).isoformat()
    problems = []
    for h in portfolio.holdings:
        if h.ticker not in histories:
            problems.append(f"{h.ticker}: no price data")
        elif histories[h.ticker].dates[-1] < stale_before:
            problems.append(f"{h.ticker}: last price is from {histories[h.ticker].dates[-1]}")

    price = {t: hist.prices[-1] for t, hist in histories.items()}
    value = {h.ticker: h.shares * price[h.ticker] for h in portfolio.holdings if h.ticker in price}
    total = sum(value.values()) + portfolio.cash
    reasons, rows = [], []
    for h in portfolio.holdings:
        actual = value.get(h.ticker, 0.0) / total if total else 0.0
        drift = actual - h.target
        breached = abs(drift) * 100 > rules["drift_points"] or abs(drift) > rules["drift_relative"] * h.target
        if breached and h.ticker in value:
            reasons.append(f"{h.ticker} is {actual:.1%} of the portfolio vs a {h.target:.1%} target")
        rows.append({"ticker": h.ticker, "category": h.category, "value": value.get(h.ticker), "weight": actual,
                     "target": h.target, "drift": drift, "breached": breached})

    held = {h.ticker for h in portfolio.holdings}
    watch, replacements = [], {}
    for category in dict.fromkeys(h.category for h in portfolio.holdings):
        peers = list(dict.fromkeys(universe[category]["tickers"] + [h.ticker for h in portfolio.holdings if h.category == category]))
        checks = [_scores(peers, histories, 5 * week, horizon_weights, risk_weight, measure) for week in range(rules["laggard_weeks"])]
        now = checks[0]
        alternatives = [t for t in sorted(now, key=now.get, reverse=True) if t not in held]
        best_alt = alternatives[0] if alternatives else None
        for h in (h for h in portfolio.holdings if h.category == category):
            if h.ticker not in now:
                watch.append({"ticker": h.ticker, "score": None})
                continue
            weeks_bottom = 0
            for scores in checks:
                order = sorted(scores, key=scores.get, reverse=True)
                if h.ticker in order and order.index(h.ticker) + 1 > len(order) / 2:
                    weeks_bottom += 1
                else:
                    break
            replace = (
                best_alt is not None
                and weeks_bottom == len(checks)
                and all(best_alt in s and h.ticker in s and s[best_alt] - s[h.ticker] >= rules["replace_margin"] for s in checks)
            )
            if replace:
                replacements[h.ticker] = best_alt
                reasons.append(
                    f"{h.ticker} has ranked in the bottom half of its category for {weeks_bottom} weekly checks; "
                    f"{best_alt} scores {now[best_alt]:.2f} vs {now[h.ticker]:.2f}"
                )
            order = sorted(now, key=now.get, reverse=True)
            watch.append({"ticker": h.ticker, "score": now[h.ticker], "rank": order.index(h.ticker) + 1, "of": len(order),
                          "weeks_in_bottom_half": weeks_bottom, "best_alternative": best_alt,
                          "alternative_score": now.get(best_alt), "replace": replace})

    action = bool(reasons) and not problems
    trades = []
    if action:
        new_target = {h.ticker: h.target for h in portfolio.holdings}
        for old, new in replacements.items():
            new_target[new] = new_target.get(new, 0.0) + new_target.pop(old)
        for t in dict.fromkeys([h.ticker for h in portfolio.holdings] + list(new_target)):
            amount = new_target.get(t, 0.0) * total - value.get(t, 0.0)
            if abs(amount) >= 1:
                trades.append({"ticker": t, "action": "buy" if amount > 0 else "sell", "amount": abs(amount),
                               "shares": abs(amount) / price[t], "new_target": new_target.get(t, 0.0)})

    notes = []
    if portfolio.account_type == "taxable" and any(t["action"] == "sell" for t in trades):
        notes.append("Sales in a taxable account realize capital gains or losses. If you are adding money, buying "
                     "the underweight funds with new cash first can avoid some of these sales.")
        if replacements:
            notes.append("If a replaced fund is sold at a loss, the wash-sale rule disallows the loss when you buy a "
                         "substantially identical fund within 30 days.")
    return {"as_of": as_of, "action_needed": action, "reasons": reasons, "data_problems": problems,
            "total_value": total, "account_type": portfolio.account_type, "holdings": rows, "watch": watch,
            "trades": trades, "notes": notes}


def headline(report: dict) -> str:
    if report["data_problems"]:
        return "Data problem"
    return "Rebalance suggested" if report["action_needed"] else "No action needed"


def render(report: dict) -> str:
    money = lambda x: f"${x:,.0f}"  # noqa: E731
    detail = {
        "Data problem": "No action is suggested until prices are complete.",
        "Rebalance suggested": "",
        "No action needed": "All holdings are within their drift bands and none is a persistent laggard.",
    }[headline(report)]
    out = [f"# Portfolio check (prices as of {report['as_of']})", "", f"**{headline(report)}.** {detail}".rstrip(), ""]
    if report["data_problems"] or report["reasons"]:
        out += [f"- {r}" for r in report["data_problems"] + report["reasons"]] + [""]
    out += [
        f"Total value {money(report['total_value'])} ({report['account_type'].replace('_', '-')} account).",
        "",
        "## Holdings",
        "",
        "| Ticker | Category | Value | Weight | Target | Drift | Status |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for r in report["holdings"]:
        value = money(r["value"]) if r["value"] is not None else "—"
        status = "outside band" if r["breached"] else "ok"
        out.append(f"| {r['ticker']} | {r['category']} | {value} | {r['weight']:.1%} | {r['target']:.1%} | "
                   f"{r['drift'] * 100:+.1f} pts | {status} |")
    out += ["", "## Replacement watch", "",
            "| Ticker | Score | Rank in category | Weeks in bottom half | Best alternative | Status |",
            "|---|---:|---|---:|---|---|"]
    for w in report["watch"]:
        if w["score"] is None:
            out.append(f"| {w['ticker']} | — | not enough history | | | |")
            continue
        alt = f"{w['best_alternative']} ({w['alternative_score']:.2f})" if w["best_alternative"] else "—"
        status = "replace" if w["replace"] else ("watch" if w["weeks_in_bottom_half"] else "ok")
        out.append(f"| {w['ticker']} | {w['score']:.2f} | {w['rank']} of {w['of']} | {w['weeks_in_bottom_half']} | {alt} | {status} |")
    if report["trades"]:
        out += ["", "## Proposed trades", "", "| Ticker | Action | Amount | ≈ Shares | New target |", "|---|---|---:|---:|---:|"]
        for t in report["trades"]:
            out.append(f"| {t['ticker']} | {t['action']} | {money(t['amount'])} | {t['shares']:,.1f} | {t['new_target']:.1%} |")
    out += [""] + [line for n in report["notes"] for line in (f"> {n}", "")]
    out.append("_Suggestions only: review them and place any trades yourself. Not investment advice._")
    return "\n".join(out)
