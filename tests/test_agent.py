import json
import math
import random
import unittest
from datetime import date, timedelta

from etf_agent import analytics as an
from etf_agent.cli import DEFAULT_UNIVERSE, parse_allocation
from etf_agent.data import History

UNIVERSE = json.loads(DEFAULT_UNIVERSE.read_text())


def history(ticker: str, daily_returns: list[float]) -> History:
    prices = [100.0]
    for r in daily_returns:
        prices.append(prices[-1] * (1 + r))
    start = date(2025, 1, 1)
    dates = [(start + timedelta(days=i)).isoformat() for i in range(len(prices))]
    return History(ticker, ticker, dates, prices)


def metrics(ticker: str, ret: float, vol: float) -> an.Metrics:
    same = lambda x: {h: x for h in an.HORIZONS}
    return an.Metrics(ticker, ticker, "x", same(ret), same(vol), same(ret / vol), 0.0)


class ParseAllocationTest(unittest.TestCase):
    def test_free_text_mix(self):
        allocation, notes = parse_allocation(
            "40% short term fixed income ETFs, 10% senior AAA rated fixed income ETFs, 10% commodities ETFs, "
            "20% Growth equities, and 20% Defensive equities",
            UNIVERSE,
        )
        self.assertEqual(
            allocation,
            {
                "short_term_fixed_income": 40,
                "aaa_fixed_income": 10,
                "commodities": 10,
                "growth_equity": 20,
                "defensive_equity": 20,
            },
        )
        self.assertEqual(notes, [])

    def test_most_specific_alias_and_fraction_weights(self):
        allocation, _ = parse_allocation("dividend growth 0.25; growth_equity=0.75", UNIVERSE)
        self.assertEqual(allocation, {"dividend_equity": 25, "growth_equity": 75})

    def test_unknown_category_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_allocation("50% crypto, 50% growth", UNIVERSE)


class ScoringTest(unittest.TestCase):
    def test_risk_weight_trades_return_against_volatility(self):
        def winner(risk_weight):
            # HOT has the highest return but terrible reward/risk; STEADY is the reverse.
            ranked = an.score_category(
                [metrics("HOT", 0.30, 0.60), metrics("MID", 0.10, 0.20), metrics("STEADY", 0.08, 0.04)],
                {"1m": 1, "3m": 1, "12m": 1},
                risk_weight,
            )
            return ranked[0].ticker

        self.assertEqual(winner(0.0), "HOT")
        self.assertEqual(winner(1.0), "STEADY")

    def test_metrics_horizons(self):
        m = an.compute_metrics(history("G", [0.001] * 300), "x", {})
        self.assertAlmostEqual(m.returns["1m"], 1.001**21 - 1)
        self.assertAlmostEqual(m.returns["12m"], 1.001**252 - 1)
        self.assertIsNone(an.compute_metrics(history("NEW", [0.001] * 200), "x", {}))


class PortfolioTest(unittest.TestCase):
    def test_duplicates_are_skipped(self):
        rng = random.Random(1)
        base = [rng.gauss(0, 0.01) for _ in range(300)]
        other = [rng.gauss(0, 0.01) for _ in range(300)]
        histories = {
            "A": history("A", base),
            "A2": history("A2", [r * 1.01 for r in base]),  # same fund, different wrapper
            "B": history("B", other),
        }
        ranked = [metrics(t, 0.1, 0.2) for t in ("A", "A2", "B")]
        picks = an.pick_decorrelated(ranked, histories, 2, 0.95)
        self.assertEqual([m.ticker for m in picks], ["A", "B"])

    def test_daily_rebalanced_backtest(self):
        histories = {"UP": history("UP", [0.002] * 300), "FLAT": history("FLAT", [0.0] * 300)}
        stats = an.portfolio_stats({"UP": 0.25, "FLAT": 0.75}, histories, {})
        self.assertAlmostEqual(stats["returns"]["12m"], 1.0005**252 - 1)
        self.assertAlmostEqual(stats["max_drawdown_12m"], 0.0)
        self.assertTrue(math.isclose(stats["vol_12m"], 0.0, abs_tol=1e-12))


if __name__ == "__main__":
    unittest.main()
