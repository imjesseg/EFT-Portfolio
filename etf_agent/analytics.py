"""Performance metrics, peer-relative scoring, and sample portfolio construction."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field

from .data import History

TRADING_DAYS = 252
HORIZONS = {"1m": 21, "3m": 63, "12m": 252}
RISK_FREE_TICKER = "BIL"  # 1-3 month T-bills


@dataclass
class Metrics:
    ticker: str
    name: str
    category: str
    returns: dict[str, float]  # total return per horizon
    vols: dict[str, float]  # annualized volatility of daily returns per horizon
    reward_risk: dict[str, float]  # (return - T-bill return) / volatility over the same horizon
    sortino: dict[str, float]  # (return - T-bill return) / downside deviation over the same horizon
    calmar: dict[str, float]  # (return - T-bill return) / worst drawdown over the same horizon
    max_drawdown: float  # worst peak-to-trough over the last 12 months
    relative: dict[str, float] = field(default_factory=dict)  # return minus category median
    score: float = 0.0  # 0..1 composite rank within the category


def total_return(prices: list[float], days: int) -> float:
    return prices[-1] / prices[-1 - days] - 1


def daily_returns(prices: list[float]) -> list[float]:
    return [b / a - 1 for a, b in zip(prices, prices[1:])]


def annual_vol(returns: list[float]) -> float:
    return statistics.stdev(returns) * math.sqrt(TRADING_DAYS) if len(returns) > 1 else 0.0


def max_drawdown(prices: list[float]) -> float:
    peak, worst = prices[0], 0.0
    for price in prices:
        peak = max(peak, price)
        worst = min(worst, price / peak - 1)
    return worst


def reward_risk(ret: float, risk_free: float, vol: float, days: int) -> float:
    """Sharpe-style ratio for a single horizon: excess return per unit of volatility over that horizon."""
    period_vol = vol * math.sqrt(days / TRADING_DAYS)
    return (ret - risk_free) / period_vol if period_vol > 0 else 0.0


def sortino(prices: list[float], risk_free: float, days: int) -> float:
    """Sortino ratio for a single horizon: like ``reward_risk``, but only days that trailed T-bills count as risk."""
    window = prices[-1 - days:]
    hurdle = (1 + risk_free) ** (1 / days) - 1  # T-bill return per day
    shortfalls = [min(r - hurdle, 0.0) for r in daily_returns(window)]
    period_downside = math.sqrt(sum(s * s for s in shortfalls) / len(shortfalls) * days)
    return (total_return(window, days) - risk_free) / period_downside if period_downside > 0 else 0.0


MIN_DRAWDOWN = 0.001  # drawdowns this small are noise (NAV rounding, payout timing), not risk


def calmar(prices: list[float], risk_free: float, days: int) -> float:
    """Calmar-style ratio for a single horizon: excess return per unit of the worst peak-to-trough loss."""
    window = prices[-1 - days:]
    return (total_return(window, days) - risk_free) / max(-max_drawdown(window), MIN_DRAWDOWN)


def compute_metrics(history: History, category: str, risk_free: dict[str, float]) -> Metrics | None:
    prices = history.prices
    if len(prices) <= TRADING_DAYS:
        return None  # less than 12 months of history
    rets, vols, rr, sort, cal = {}, {}, {}, {}, {}
    for label, days in HORIZONS.items():
        rf = risk_free.get(label, 0.0)
        rets[label] = total_return(prices, days)
        vols[label] = annual_vol(daily_returns(prices[-1 - days:]))
        rr[label] = reward_risk(rets[label], rf, vols[label], days)
        sort[label] = sortino(prices, rf, days)
        cal[label] = calmar(prices, rf, days)
    return Metrics(history.ticker, history.name, category, rets, vols, rr, sort, cal, max_drawdown(prices[-1 - TRADING_DAYS:]))


def percentile_ranks(values: list[float]) -> list[float]:
    """0 for the lowest value, 1 for the highest; ties share their average rank."""
    n = len(values)
    if n == 1:
        return [1.0]
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 / (n - 1)
        i = j + 1
    return ranks


def score_category(
    metrics: list[Metrics], horizon_weights: dict[str, float], risk_weight: float, risk_measure: str = "reward_risk"
) -> list[Metrics]:
    """Rank ETFs against their category peers and return them best first.

    For each horizon an ETF earns a blend of its peer percentile on ``risk_measure`` ("reward_risk",
    "sortino", or "calmar"; weight ``risk_weight``) and on raw return (weight ``1 - risk_weight``); horizons are
    combined with ``horizon_weights``.
    """
    total_weight = sum(horizon_weights.values())
    for m in metrics:
        m.score = 0.0
    for label in HORIZONS:
        returns = [m.returns[label] for m in metrics]
        median = statistics.median(returns)
        return_ranks = percentile_ranks(returns)
        rr_ranks = percentile_ranks([getattr(m, risk_measure)[label] for m in metrics])
        for m, ret_rank, rr_rank in zip(metrics, return_ranks, rr_ranks):
            m.relative[label] = m.returns[label] - median
            blended = risk_weight * rr_rank + (1 - risk_weight) * ret_rank
            m.score += horizon_weights[label] / total_weight * blended
    return sorted(metrics, key=lambda m: m.score, reverse=True)


def aligned_returns(histories: dict[str, History], tickers: list[str], days: int = TRADING_DAYS) -> dict[str, list[float]]:
    """Daily returns over the last ``days`` trading days that all tickers share."""
    price_maps = {t: dict(zip(histories[t].dates, histories[t].prices)) for t in tickers}
    common = sorted(set.intersection(*(set(p) for p in price_maps.values())))[-(days + 1):]
    return {t: daily_returns([price_maps[t][d] for d in common]) for t in tickers}


def correlation(a: list[float], b: list[float]) -> float:
    return statistics.correlation(a, b) if statistics.pstdev(a) and statistics.pstdev(b) else 0.0


def tracking_stats(target: str, fund: str, histories: dict[str, History], risk_free_12m: float = 0.0) -> dict | None:
    """How closely ``fund`` followed ``target`` over the last 12 months of shared trading days."""
    series = aligned_returns(histories, [target, fund])
    a, b = series[target], series[fund]
    if len(a) < 60:
        return None
    variance = statistics.variance(a)
    return {
        "ticker": fund,
        "name": histories[fund].name,
        "tracking_error": annual_vol([y - x for x, y in zip(a, b)]),
        "correlation": correlation(a, b),
        "beta": statistics.covariance(a, b) / variance if variance else 0.0,
        "return_gap_12m": math.prod(1 + r for r in b) - math.prod(1 + r for r in a),
        **measures_12m(histories[fund].prices, risk_free_12m),
    }


def measures_12m(prices: list[float], risk_free_12m: float) -> dict[str, float]:
    days = min(TRADING_DAYS, len(prices) - 1)
    return {"sortino_12m": sortino(prices, risk_free_12m, days), "calmar_12m": calmar(prices, risk_free_12m, days)}


MATCH_TIERS = ("close", "fair", "loose")


def match_quality(match: dict) -> str:
    # Tracking error alone suits cash-like funds whose daily moves are mostly noise; correlation suits equities.
    if match["tracking_error"] <= 0.02 or match["correlation"] >= 0.95:
        return "close"
    if match["tracking_error"] <= 0.06 or match["correlation"] >= 0.80:
        return "fair"
    return "loose"


def closest_funds(
    target: str,
    funds: list[str],
    histories: dict[str, History],
    risk_free_12m: float = 0.0,
    count: int = 3,
    prefer: str | None = None,
) -> list[dict]:
    """The ``count`` funds with the lowest tracking error against ``target``, best first.

    Tracking error, not correlation, because a high-beta fund can correlate well yet track badly. With
    ``prefer`` ("sortino" or "calmar"), funds whose match quality is as good as the closest fund's are
    reordered by that 12-month measure, so the better risk-adjusted fund wins among equally good matches.
    """
    stats = [tracking_stats(target, f, histories, risk_free_12m) for f in funds if f in histories and f != target]
    ranked = sorted((s for s in stats if s), key=lambda s: s["tracking_error"])[:count]
    if prefer and ranked:
        best_tier = MATCH_TIERS.index(match_quality(ranked[0]))
        ranked.sort(
            key=lambda s: (0, -s[f"{prefer}_12m"])
            if MATCH_TIERS.index(match_quality(s)) <= best_tier
            else (1, s["tracking_error"])
        )
    return ranked


def pick_decorrelated(ranked: list[Metrics], histories: dict[str, History], count: int, max_corr: float) -> list[Metrics]:
    """Best-scoring ETFs, skipping near-duplicates of an ETF already picked (e.g. QQQ vs QQQM)."""
    if not ranked:
        return []
    series = aligned_returns(histories, [m.ticker for m in ranked])
    picks: list[Metrics] = []
    for m in ranked:
        if all(correlation(series[m.ticker], series[p.ticker]) < max_corr for p in picks):
            picks.append(m)
        if len(picks) == count:
            break
    return picks


def pick_low_vol(ranked: list[Metrics], count: int) -> list[Metrics]:
    """Lowest 12-month volatility among the top half of the category by score."""
    top_half = ranked[: max(1, math.ceil(len(ranked) / 2))]
    return sorted(top_half, key=lambda m: m.vols["12m"])[:count]


def inverse_vol_weights(picks: list[Metrics], allocation: float) -> dict[str, float]:
    inverse = {m.ticker: 1 / max(m.vols["12m"], 1e-6) for m in picks}
    total = sum(inverse.values())
    return {t: allocation * v / total for t, v in inverse.items()}


def measure_weights(picks: list[Metrics], allocation: float, measure: str) -> dict[str, float]:
    """Weights proportional to a 12-month ratio ("sortino" or "calmar"); picks with no excess return get
    nothing (equal weights if none qualify)."""
    positive = {m.ticker: max(getattr(m, measure)["12m"], 0.0) for m in picks}
    total = sum(positive.values())
    if total == 0:
        return {m.ticker: allocation / len(picks) for m in picks}
    return {t: allocation * v / total for t, v in positive.items()}


def portfolio_stats(weights: dict[str, float], histories: dict[str, History], risk_free: dict[str, float]) -> dict:
    """Backtest constant (daily-rebalanced) weights over the last 12 months."""
    series = aligned_returns(histories, list(weights))
    n = len(next(iter(series.values())))
    daily = [sum(w * series[t][i] for t, w in weights.items()) for i in range(n)]
    nav = [1.0]
    for r in daily:
        nav.append(nav[-1] * (1 + r))
    days = {label: min(d, n) for label, d in HORIZONS.items()}
    rets = {label: total_return(nav, d) for label, d in days.items()}
    vol = annual_vol(daily)
    return {
        "returns": rets,
        "vol_12m": vol,
        "reward_risk_12m": reward_risk(rets["12m"], risk_free.get("12m", 0.0), vol, days["12m"]),
        "sortino_12m": sortino(nav, risk_free.get("12m", 0.0), days["12m"]),
        "calmar_12m": calmar(nav, risk_free.get("12m", 0.0), days["12m"]),
        "max_drawdown_12m": max_drawdown(nav),
    }
