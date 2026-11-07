"""Command-line agent: turn a target sector mix into ranked ETF candidates and sample portfolios."""

from __future__ import annotations

import argparse
import json
import re
import sys
from functools import partial
from pathlib import Path

from . import analytics as an
from . import monitor, notify
from .data import fetch_all

DEFAULT_UNIVERSE = Path(__file__).with_name("universe.json")
FIDELITY_FUNDS = Path(__file__).with_name("fidelity_funds.json")
FIDELITY_QUOTE_URL = "https://digital.fidelity.com/prgw/digital/research/quote/dashboard/summary?symbol={}"
# --measure choice -> (Metrics attribute, description). Volatility-based "sharpe" is the default.
MEASURES = {
    "sharpe": ("reward_risk", "reward/risk (return above T-bills ÷ volatility)"),
    "sortino": ("sortino", "Sortino (return above T-bills ÷ downside deviation)"),
    "calmar": ("calmar", "Calmar (return above T-bills ÷ worst drawdown)"),
}
EXAMPLE = (
    "40% short term fixed income, 10% senior AAA rated fixed income, 10% commodities, "
    "20% growth equities, 20% defensive equities"
)


def _normalize(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9%.]+", " ", text.lower().replace("_", " ")).split())


def match_category(label: str, universe: dict) -> str | None:
    """Map free text like 'growth equities' to a category key (exact key, else longest alias found)."""
    text = _normalize(label)
    for key in universe:
        if text == _normalize(key):
            return key
    best, best_len = None, 0
    for key, spec in universe.items():
        for alias in spec["aliases"]:
            if re.search(r"\b" + re.escape(_normalize(alias)), text) and len(alias) > best_len:
                best, best_len = key, len(alias)
    return best


def parse_allocation(text: str, universe: dict) -> tuple[dict[str, float], list[str]]:
    """Parse '40% short term fixed income, 20% growth' (or 'growth=20') into {category: percent}."""
    allocation: dict[str, float] = {}
    notes: list[str] = []
    chunks = [c for c in re.split(r"[,;\n+]|\s+and\s+(?=\d)", text) if c.strip()]
    for chunk in chunks:
        # Prefer a number marked with %, so labels like "1-3 year treasuries 20%" work.
        number = re.search(r"\d+(?:\.\d+)?(?=\s*%)", chunk) or re.search(r"\d+(?:\.\d+)?", chunk)
        if not number:
            raise ValueError(f"No percentage found in '{chunk.strip()}'")
        label = (chunk[: number.start()] + " " + chunk[number.end():]).replace("%", " ").replace("=", " ").replace(":", " ")
        key = match_category(label, universe)
        if key is None:
            raise ValueError(f"Could not match '{label.strip()}' to a category. Run with --list-categories.")
        allocation[key] = allocation.get(key, 0.0) + float(number.group())
    total = sum(allocation.values())
    if abs(total - 100) > 0.01:
        scale = 100 if abs(total - 1) <= 0.01 else 100 / total
        if scale != 100:
            notes.append(f"Weights summed to {total:g}; rescaled to 100%.")
        allocation = {k: v * scale for k, v in allocation.items()}
    return allocation, notes


def pct(x: float, signed: bool = False) -> str:
    return f"{x * 100:+.1f}%" if signed else f"{x * 100:.1f}%"


def triple(d: dict[str, float], fmt) -> str:
    return " / ".join(fmt(d[h]) for h in an.HORIZONS)


def fidelity_matches(
    etfs: list[str], histories: dict, use_cache: bool, prefer: str | None
) -> tuple[dict[str, dict], dict[str, str]]:
    """Closest Fidelity mutual funds for each ETF: {etf: {"sortino_12m", "calmar_12m", "funds"}}.

    Returns (matches, fetch errors). ``prefer`` ("sortino"/"calmar") breaks ties among equally good matches.
    """
    funds = [t for group in json.loads(FIDELITY_FUNDS.read_text()).values() for t in group]
    wanted = dict.fromkeys(etfs + funds + [an.RISK_FREE_TICKER])
    fetched, errors = fetch_all([t for t in wanted if t not in histories], use_cache)
    histories = {**histories, **fetched}
    rf = an.total_return(histories[an.RISK_FREE_TICKER].prices, an.TRADING_DAYS) if an.RISK_FREE_TICKER in histories else 0.0
    matches = {
        e: {
            **an.measures_12m(histories[e].prices, rf),
            "funds": an.closest_funds(e, funds, histories, rf, prefer=prefer),
        }
        for e in etfs
        if e in histories
    }
    return matches, errors


def prefer_measure(args) -> str | None:
    """Fidelity tie-break measure: none for the default (pure tracking error), else the chosen ratio."""
    return None if args.measure == "sharpe" else args.measure


def fund_link(ticker: str) -> str:
    return f"[{ticker}]({FIDELITY_QUOTE_URL.format(ticker)})"


def render_fidelity_table(matches: dict[str, dict], prefer: str | None) -> list[str]:
    shown = prefer or "sortino"
    key, label = f"{shown}_12m", shown.capitalize()
    out = [
        "| ETF | Suggested Fidelity mutual fund | Match | Tracking error | Correlation | Beta | 12m return vs ETF "
        f"| {label} 12m (ETF → fund) | Other candidates (tracking error, {label}) |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for etf, match in matches.items():
        if not match["funds"]:
            out.append(f"| {etf} | no Fidelity fund with enough shared history | | | | | | | |")
            continue
        best, rest = match["funds"][0], match["funds"][1:]
        others = ", ".join(f"{fund_link(f['ticker'])} ({pct(f['tracking_error'])}, {f[key]:.2f})" for f in rest)
        out.append(
            f"| {etf} | {fund_link(best['ticker'])} {best['name']} | {an.match_quality(best)} | {pct(best['tracking_error'])} | "
            f"{best['correlation']:.2f} | {best['beta']:.2f} | {pct(best['return_gap_12m'], True)} | "
            f"{match[key]:.2f} → {best[key]:.2f} | {others} |"
        )
    selection = (
        "Candidates are the 3 funds with the lowest tracking error; among those with a match as good as the closest one, "
        f"the highest 12-month {label} is suggested."
        if prefer
        else "The fund with the lowest tracking error is suggested."
    )
    out += [
        "",
        f"_{selection} Tracking error is the annualized volatility of the daily return difference over the last "
        "12 months (lower = follows the ETF more closely). \"12m return vs ETF\" is the fund's 12-month return minus "
        "the ETF's. Match: close = tracking error ≤ 2% or correlation ≥ 0.95; fair = ≤ 6% or ≥ 0.80; loose = no close "
        "Fidelity mutual fund equivalent. Check expense ratios, loads, and minimums on fidelity.com before switching._",
    ]
    return out


def build_report(allocation: dict[str, float], universe: dict, args) -> dict:
    tickers = sorted({t for key in allocation for t in universe[key]["tickers"]} | {an.RISK_FREE_TICKER})
    histories, errors = fetch_all(tickers, use_cache=not args.no_cache)
    risk_free = {}
    if an.RISK_FREE_TICKER in histories:
        prices = histories[an.RISK_FREE_TICKER].prices
        risk_free = {h: an.total_return(prices, d) for h, d in an.HORIZONS.items()}

    horizon_weights = dict(zip(an.HORIZONS, args.horizon_weights))
    ranked: dict[str, list[an.Metrics]] = {}
    skipped: dict[str, str] = dict(errors)
    for key in allocation:
        metrics = []
        for t in universe[key]["tickers"]:
            if t not in histories:
                continue
            m = an.compute_metrics(histories[t], key, risk_free)
            if m:
                metrics.append(m)
            else:
                skipped[t] = "less than 12 months of history"
        if not metrics:
            raise SystemExit(f"No usable price data for category '{key}'.")
        ranked[key] = an.score_category(metrics, horizon_weights, args.risk_weight, MEASURES[args.measure][0])

    if args.measure == "sharpe":
        weigh, weighting = an.inverse_vol_weights, "inverse-volatility weights"
    else:
        label = args.measure.capitalize()
        weigh = partial(an.measure_weights, measure=args.measure)
        weighting = f"weights proportional to 12-month {label} (picks with {label} ≤ 0 get none)"
    portfolios = {}
    builders = {
        "Best-in-class": "Top-scoring ETF in each category gets the full category weight.",
        "Diversified": f"Top {args.picks} per category, skipping near-duplicates (daily-return correlation "
        f">= {args.max_corr}); {weighting} within each category.",
        "Low-volatility": f"The 2 lowest-volatility ETFs from the top half of each category; {weighting}.",
        "Category average (baseline)": "Every ETF in each category equally weighted - what the picks must beat.",
    }
    for name, description in builders.items():
        weights: dict[str, float] = {}
        for key, share in allocation.items():
            share /= 100
            peers = ranked[key]
            if name == "Best-in-class":
                part = {peers[0].ticker: share}
            elif name == "Diversified":
                part = weigh(an.pick_decorrelated(peers, histories, args.picks, args.max_corr), share)
            elif name == "Low-volatility":
                part = weigh(an.pick_low_vol(peers, 2), share)
            else:
                part = {m.ticker: share / len(peers) for m in peers}
            for t, w in part.items():
                if w > 0:
                    weights[t] = weights.get(t, 0.0) + w
        portfolios[name] = {
            "description": description,
            "weights": weights,
            "stats": an.portfolio_stats(weights, histories, risk_free),
        }

    fidelity = None
    if args.fidelity:
        held = sorted({t for name, p in portfolios.items() if not name.startswith("Category average") for t in p["weights"]})
        fidelity, fund_errors = fidelity_matches(held, histories, not args.no_cache, prefer_measure(args))
        skipped.update(fund_errors)

    as_of = max(histories[t].dates[-1] for t in histories)
    return {
        "as_of": as_of,
        "fidelity": fidelity,
        "allocation": allocation,
        "risk_free_returns": risk_free,
        "ranked": ranked,
        "portfolios": portfolios,
        "skipped": skipped,
        "names": {t: h.name for t, h in histories.items()},
        "category_of": {t: key for key, ms in ranked.items() for t in (m.ticker for m in ms)},
    }


def render_markdown(report: dict, universe: dict, args, notes: list[str]) -> str:
    out = [f"# ETF portfolio suggestions (prices as of {report['as_of']})", ""]
    out += [f"> {n}" for n in notes]
    out.append(
        "Target mix: " + ", ".join(f"{v:g}% {universe[k]['name']}" for k, v in report["allocation"].items())
    )
    w = args.horizon_weights
    measure = MEASURES[args.measure][1]
    out += [
        "",
        f"Scoring: each ETF is ranked against its category peers on 1m/3m/12m performance "
        f"(horizon weights {w[0]:g}/{w[1]:g}/{w[2]:g}); {args.risk_weight:.0%} of each rank is {measure} "
        f"and {1 - args.risk_weight:.0%} is raw return. Score 1.00 = best in category.",
        "",
        "## Sample portfolios",
        "",
        "| Portfolio | 1m | 3m | 12m | Vol (12m) | Reward/risk (12m) | Sortino (12m) | Calmar (12m) | Max drawdown (12m) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, p in report["portfolios"].items():
        s = p["stats"]
        out.append(
            f"| {name} | {pct(s['returns']['1m'], True)} | {pct(s['returns']['3m'], True)} | "
            f"{pct(s['returns']['12m'], True)} | {pct(s['vol_12m'])} | {s['reward_risk_12m']:.2f} | "
            f"{s['sortino_12m']:.2f} | {s['calmar_12m']:.2f} | {pct(s['max_drawdown_12m'])} |"
        )
    fidelity = report["fidelity"]
    extra = (" Fidelity mutual fund |", "---|") if fidelity is not None else ("", "")
    for name, p in report["portfolios"].items():
        if name.startswith("Category average"):
            continue
        out += ["", f"### {name}", "", p["description"], "", "| Weight | Ticker | Category | Fund |" + extra[0], "|---:|---|---|---|" + extra[1]]
        for t, wt in sorted(p["weights"].items(), key=lambda kv: -kv[1]):
            row = f"| {pct(wt)} | {t} | {universe[report['category_of'][t]]['name']} | {report['names'][t]} |"
            if fidelity is not None:
                funds = fidelity.get(t, {}).get("funds")
                row += f" {fund_link(funds[0]['ticker'])} ({an.match_quality(funds[0])}) |" if funds else " — |"
            out.append(row)
    if fidelity is not None:
        out += ["", "## Fidelity mutual fund equivalents", ""] + render_fidelity_table(fidelity, prefer_measure(args))

    out += ["", "## Category rankings", ""]
    for key, metrics in report["ranked"].items():
        out += [
            f"### {universe[key]['name']} — {report['allocation'][key]:g}% ({len(metrics)} ETFs ranked)",
            "",
            "| # | Ticker | Score | Return 1m / 3m / 12m | vs peer median 1m / 3m / 12m | Reward/risk 1m / 3m / 12m "
            "| Sortino 1m / 3m / 12m | Calmar 1m / 3m / 12m | Vol (12m) | Max DD | Fund |",
            "|---:|---|---:|---|---|---|---|---|---:|---:|---|",
        ]
        for i, m in enumerate(metrics[: args.top], 1):
            out.append(
                f"| {i} | {m.ticker} | {m.score:.2f} | {triple(m.returns, lambda x: pct(x, True))} | "
                f"{triple(m.relative, lambda x: pct(x, True))} | {triple(m.reward_risk, lambda x: f'{x:.2f}')} | "
                f"{triple(m.sortino, lambda x: f'{x:.2f}')} | {triple(m.calmar, lambda x: f'{x:.2f}')} | {pct(m.vols['12m'])} | {pct(m.max_drawdown)} | {m.name} |"
            )
        out.append("")

    if report["skipped"]:
        out.append("Skipped: " + "; ".join(f"{t} ({why})" for t, why in sorted(report["skipped"].items())))
        out.append("")
    out.append(
        "_Backtests use the same 12-month window the ETFs were selected on, so they overstate what the picks "
        "will do next. Past performance does not predict future returns; this is not investment advice._"
    )
    return "\n".join(out)


def to_json(report: dict) -> str:
    def metrics_dict(m: an.Metrics) -> dict:
        keys = ("ticker", "name", "score", "returns", "relative", "reward_risk", "sortino", "calmar", "vols", "max_drawdown")
        return {k: getattr(m, k) for k in keys}

    payload = {**report, "ranked": {k: [metrics_dict(m) for m in ms] for k, ms in report["ranked"].items()}}
    del payload["category_of"]
    return json.dumps(payload, indent=2)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="etf-agent",
        description="Suggest ETF portfolios for a target sector mix by ranking ETFs on 1/3/12-month "
        "peer-relative performance and reward/risk.",
        epilog=f'Example: etf-agent "{EXAMPLE}"',
    )
    parser.add_argument("allocation", nargs="*", help="Target mix, e.g. \"40%% short term fixed income, 60%% growth\"")
    parser.add_argument("--list-categories", action="store_true", help="Show categories, aliases, and ETFs, then exit")
    parser.add_argument("--top", type=int, default=5, help="Ranked ETFs to show per category (default 5)")
    parser.add_argument("--picks", type=int, default=3, help="ETFs per category in the Diversified portfolio (default 3)")
    parser.add_argument("--max-corr", type=float, default=0.95, help="Correlation at which two ETFs count as duplicates (default 0.95)")
    parser.add_argument("--risk-weight", type=float, default=0.5, help="0 = rank on raw return only, 1 = reward/risk only (default 0.5)")
    parser.add_argument(
        "--horizon-weights",
        type=lambda s: [float(x) for x in s.split(",")],
        default=[0.2, 0.3, 0.5],
        help="Weights for 1m,3m,12m (default 0.2,0.3,0.5)",
    )
    parser.add_argument(
        "--measure",
        choices=MEASURES,
        default="sharpe",
        help="Risk-adjusted measure used to rank ETFs, weight picks, and break ties between equally good Fidelity "
        "matches: sharpe = volatility (inverse-volatility weights, default), sortino = downside deviation, "
        "calmar = worst drawdown",
    )
    parser.add_argument("--sortino", dest="measure", action="store_const", const="sortino", help="Same as --measure sortino")
    parser.add_argument("--calmar", dest="measure", action="store_const", const="calmar", help="Same as --measure calmar")
    parser.add_argument("--fidelity", action="store_true", help="Add the closest Fidelity mutual fund for every suggested ETF")
    parser.add_argument("--match", nargs="+", metavar="ETF", help="Only find the closest Fidelity mutual funds for these ETFs, e.g. --match XLP QQQ")
    parser.add_argument(
        "--monitor",
        nargs="?",
        const=Path("portfolio.json"),
        type=Path,
        metavar="PORTFOLIO",
        help="Check a portfolio file (default portfolio.json) for drift and laggards and propose trades",
    )
    parser.add_argument(
        "--import-holdings",
        type=Path,
        metavar="FILE",
        help="Build a portfolio file from a Fidelity positions CSV or a 'TICKER SHARES' list and print it; targets "
        "come from a mix given as the allocation (split by --measure), else from current weights",
    )
    parser.add_argument("--account", help="With --import-holdings, the account number to use from a multi-account export")
    parser.add_argument(
        "--account-type", choices=["tax_deferred", "taxable"], default="tax_deferred",
        help="With --import-holdings: tax_deferred (IRA, 401k; default) or taxable",
    )
    parser.add_argument(
        "--email",
        nargs="?",
        const="action",
        choices=["action", "always"],
        help="With --monitor, email the report (SMTP settings from environment variables): only when a rebalance "
        "is suggested or prices are missing (default), or 'always' after every run",
    )
    parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE, help="Category/ETF universe JSON file")
    parser.add_argument("--no-cache", action="store_true", help="Re-download prices even if cached today")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of Markdown")
    args = parser.parse_args(argv)

    universe = json.loads(args.universe.read_text())
    if args.list_categories:
        for key, spec in universe.items():
            print(f"{key}: {spec['name']}\n  aliases: {', '.join(spec['aliases'])}\n  ETFs: {' '.join(spec['tickers'])}")
        return
    if args.match:
        etfs = [t.upper() for t in args.match]
        matches, _ = fidelity_matches(etfs, {}, not args.no_cache, prefer_measure(args))
        if args.json:
            print(json.dumps(matches, indent=2))
        else:
            missing = [e for e in etfs if e not in matches]
            lines = ["# Closest Fidelity mutual funds (last 12 months)", ""] + render_fidelity_table(matches, prefer_measure(args))
            print("\n".join(lines + ([f"\nNo price data for: {', '.join(missing)}"] if missing else [])))
        return
    if len(args.horizon_weights) != 3 or not 0 <= args.risk_weight <= 1:
        parser.error("--horizon-weights needs 3 numbers and --risk-weight must be between 0 and 1")
    if args.import_holdings:
        try:
            positions, cash = monitor.parse_positions(args.import_holdings.read_text(), args.account)
            allocation = parse_allocation(" ".join(args.allocation), universe)[0] if args.allocation else None
        except (OSError, ValueError) as exc:
            parser.error(f"{args.import_holdings}: {exc}")
        needs_matching = any("category" not in p and not any(p["ticker"] in s["tickers"] for s in universe.values()) for p in positions)
        tickers = [p["ticker"] for p in positions] + [an.RISK_FREE_TICKER]
        if needs_matching:
            tickers += [t for spec in universe.values() for t in spec["tickers"]]
        histories, _ = fetch_all(list(dict.fromkeys(tickers)), use_cache=not args.no_cache)
        try:
            doc, notes = monitor.build_portfolio(
                positions, cash, universe, histories, allocation, MEASURES[args.measure][0], args.account_type
            )
        except ValueError as exc:
            parser.error(str(exc))
        print(monitor.portfolio_json(doc), end="")
        for note in notes:
            print(f"Note: {note}", file=sys.stderr)
        return
    if args.monitor:
        try:
            portfolio = monitor.load_portfolio(args.monitor, universe)
        except (OSError, ValueError, KeyError) as exc:
            parser.error(f"{args.monitor}: {exc}")
        histories, _ = fetch_all(monitor.tickers_needed(portfolio, universe), use_cache=not args.no_cache)
        report = monitor.check(
            portfolio, universe, histories, dict(zip(an.HORIZONS, args.horizon_weights)), args.risk_weight,
            MEASURES[args.measure][0],
        )
        text = monitor.render(report)
        print(json.dumps(report, indent=2) if args.json else text)
        if args.email == "always" or (args.email == "action" and (report["action_needed"] or report["data_problems"])):
            notify.send_email(f"ETF portfolio check {report['as_of']}: {monitor.headline(report)}", text)
        return

    try:
        allocation, notes = parse_allocation(" ".join(args.allocation) or EXAMPLE, universe)
    except ValueError as exc:
        parser.error(str(exc))
    report = build_report(allocation, universe, args)
    print(to_json(report) if args.json else render_markdown(report, universe, args, notes))


if __name__ == "__main__":
    sys.exit(main())
