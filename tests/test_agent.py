import json
import math
import os
import random
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

from etf_agent import analytics as an
from etf_agent import data
from etf_agent import monitor, notify
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
    return an.Metrics(ticker, ticker, "x", same(ret), same(vol), same(ret / vol), same(ret / vol), same(ret / vol), 0.0)


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

    def test_sortino_counts_only_shortfalls_below_t_bills(self):
        prices = history("S", [0.02, -0.01] * 126).prices
        downside_per_day = math.sqrt(0.01**2 / 2)  # only the -1% days count; +2% days add no risk
        expected = ((1.02 * 0.99) ** 126 - 1) / (downside_per_day * math.sqrt(252))
        self.assertAlmostEqual(an.sortino(prices, 0.0, 252), expected)

    def test_calmar_divides_excess_return_by_worst_drawdown(self):
        crash = history("C", [0.01] * 100 + [-0.10] + [0.01] * 151).prices
        self.assertAlmostEqual(an.calmar(crash, 0.02, 252), (1.01**251 * 0.9 - 1 - 0.02) / 0.10)
        steady = history("S", [0.0001] * 252).prices  # never falls: drawdown floored, not divided by zero
        self.assertAlmostEqual(an.calmar(steady, 0.0, 252), (1.0001**252 - 1) / an.MIN_DRAWDOWN)

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

    def test_closest_fund_tracks_rather_than_merely_correlates(self):
        rng = random.Random(2)
        etf = [rng.gauss(0, 0.01) for _ in range(300)]
        histories = {
            "ETF": history("ETF", etf),
            "LEVER": history("LEVER", [2 * r for r in etf]),  # correlation 1.0, but moves twice as much
            "CLONE": history("CLONE", [r + rng.gauss(0, 0.001) for r in etf]),
            "OTHER": history("OTHER", [rng.gauss(0, 0.01) for _ in range(300)]),
        }
        matches = an.closest_funds("ETF", ["OTHER", "LEVER", "CLONE"], histories)
        self.assertEqual([m["ticker"] for m in matches], ["CLONE", "LEVER", "OTHER"])
        self.assertAlmostEqual(matches[1]["beta"], 2.0)

    def test_sortino_preference_only_among_equally_good_matches(self):
        rng = random.Random(3)
        etf = [rng.gauss(0, 0.01) for _ in range(300)]
        histories = {
            "ETF": history("ETF", etf),
            "TIGHT": history("TIGHT", [r + rng.gauss(0, 0.0005) for r in etf]),
            "BETTER": history("BETTER", [r + rng.gauss(0.0004, 0.0008) for r in etf]),  # close match, higher return
            "LOOSE": history("LOOSE", [rng.gauss(0.003, 0.01) for _ in range(300)]),  # best Sortino, unrelated
        }
        funds = ["LOOSE", "BETTER", "TIGHT"]
        by_tracking = an.closest_funds("ETF", funds, histories)
        by_sortino = an.closest_funds("ETF", funds, histories, prefer="sortino")
        self.assertEqual([m["ticker"] for m in by_tracking], ["TIGHT", "BETTER", "LOOSE"])
        self.assertEqual([m["ticker"] for m in by_sortino], ["BETTER", "TIGHT", "LOOSE"])
        self.assertGreater(by_sortino[2]["sortino_12m"], by_sortino[0]["sortino_12m"])

    def test_sortino_weights(self):
        def pick(ticker, s):
            m = metrics(ticker, 0.1, 0.2)
            m.sortino["12m"] = s
            return m

        self.assertEqual(an.measure_weights([pick("A", 3.0), pick("B", 1.0), pick("C", -2.0)], 1.0, "sortino"), {"A": 0.75, "B": 0.25, "C": 0.0})
        self.assertEqual(an.measure_weights([pick("A", -1.0), pick("B", 0.0)], 0.4, "sortino"), {"A": 0.2, "B": 0.2})

    def test_daily_rebalanced_backtest(self):
        histories = {"UP": history("UP", [0.002] * 300), "FLAT": history("FLAT", [0.0] * 300)}
        stats = an.portfolio_stats({"UP": 0.25, "FLAT": 0.75}, histories, {})
        self.assertAlmostEqual(stats["returns"]["12m"], 1.0005**252 - 1)
        self.assertAlmostEqual(stats["max_drawdown_12m"], 0.0)
        self.assertTrue(math.isclose(stats["vol_12m"], 0.0, abs_tol=1e-12))


class MonitorTest(unittest.TestCase):
    UNIVERSE = {"cat": {"name": "Category", "aliases": [], "tickers": ["A", "B", "C"]}}

    def run_check(self, shares_a, shares_b, histories=None):
        portfolio = monitor.Portfolio(
            [monitor.Holding("A", shares_a, 0.10, "cat"), monitor.Holding("B", shares_b, 0.90, "cat")],
            0.0, "tax_deferred", dict(monitor.DEFAULT_RULES),
        )
        flat = {t: history(t, [0.0] * 300) for t in ("A", "B", "C")}  # every price stays at 100
        return monitor.check(portfolio, self.UNIVERSE, histories or flat, {"1m": 1, "3m": 1, "12m": 1}, 0.5, "reward_risk")

    def test_drift_bands_use_points_or_share_of_target(self):
        # A at 13% vs 10% is only 3 points off but 30% of its target: outside the 5/25 band.
        report = self.run_check(13, 87)
        self.assertTrue(report["action_needed"])
        self.assertEqual([r["breached"] for r in report["holdings"]], [True, False])
        trades = {t["ticker"]: (t["action"], round(t["amount"], 2)) for t in report["trades"]}
        self.assertEqual(trades, {"A": ("sell", 300.0), "B": ("buy", 300.0)})
        # 12% is 2 points and 20% of target: inside both limits.
        self.assertFalse(self.run_check(12, 88)["action_needed"])

    def test_stale_prices_block_action(self):
        histories = {t: history(t, [0.0] * 300) for t in ("A", "C")}
        histories["B"] = history("B", [0.0] * 290)  # ten days behind the others
        report = self.run_check(13, 87, histories)
        self.assertFalse(report["action_needed"])
        self.assertEqual(report["trades"], [])
        self.assertIn("B", report["data_problems"][0])

    def test_replacement_needs_a_persistent_laggard(self):
        rng = random.Random(4)
        noisy = lambda drift, n=300: [drift + rng.gauss(0, 0.002) for _ in range(n)]  # noqa: E731
        universe = {"cat": {"name": "Category", "aliases": [], "tickers": ["H", "G", "M1", "M2"]}}
        peers = {"G": history("G", noisy(0.002)), "M1": history("M1", noisy(0.001)), "M2": history("M2", noisy(0.0005))}
        rules = {**monitor.DEFAULT_RULES, "laggard_weeks": 3}

        def check(held_returns):
            portfolio = monitor.Portfolio([monitor.Holding("H", 100, 1.0, "cat")], 0.0, "tax_deferred", rules)
            histories = {**peers, "H": history("H", held_returns)}
            return monitor.check(portfolio, universe, histories, {"1m": 1, "3m": 1, "12m": 1}, 0.5, "reward_risk")

        lagging = check(noisy(-0.001))
        self.assertTrue(lagging["watch"][0]["replace"])
        self.assertEqual({t["ticker"]: t["action"] for t in lagging["trades"]}, {"H": "sell", "G": "buy"})
        # A fund that led its category until a bad final week is only watched, not replaced.
        slump = check(noisy(0.003, 295) + [-0.05] * 5)
        self.assertFalse(slump["watch"][0]["replace"])
        self.assertFalse(slump["action_needed"])


class DataFallbackTest(unittest.TestCase):
    def test_tiingo_is_used_only_when_yahoo_fails_and_a_key_is_set(self):
        rows = [{"date": "2026-11-05T00:00:00.000Z", "adjClose": 10.0}, {"date": "2026-11-06T00:00:00.000Z", "adjClose": 10.5}]

        def fake_get(url):
            if "yahoo" in url:
                raise OSError("HTTP Error 429: Too Many Requests")
            self.assertIn("token=KEY", url)
            return rows

        with tempfile.TemporaryDirectory() as tmp, patch.object(data, "_get_json", fake_get), patch.object(data, "CACHE_DIR", Path(tmp)):
            with patch.dict(os.environ, {"TIINGO_API_KEY": ""}):
                self.assertRaises(OSError, data.fetch_history, "XLP")
            with patch.dict(os.environ, {"TIINGO_API_KEY": "KEY"}):
                h = data.fetch_history("XLP")
        self.assertEqual((h.source, h.dates, h.prices), ("tiingo", ["2026-11-05", "2026-11-06"], [10.0, 10.5]))


class EmailTest(unittest.TestCase):
    def test_report_is_sent_as_plain_text_and_html(self):
        sent = []

        class FakeSMTP:
            def __init__(self, host, port, timeout):
                self.address = (host, port)

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def ehlo(self):
                pass

            def has_extn(self, name):
                return False

            def login(self, user, password):
                sent.append(("login", user, password))

            def send_message(self, message):
                sent.append(message)

        report = "# Check\n\n**Rebalance suggested.**\n\n| Ticker | Amount |\n|---|---:|\n| [XLP](https://x.test/?a=1&b=2) | <$5 |"
        env = {"SMTP_HOST": "smtp.test", "SMTP_USER": "me@test", "SMTP_PASSWORD": "pw", "EMAIL_TO": "a@test, b@test"}
        with patch.dict(os.environ, env), patch.object(notify.smtplib, "SMTP", FakeSMTP):
            notify.send_email("ETF check: Rebalance suggested", report)
        login, message = sent
        self.assertEqual(login, ("login", "me@test", "pw"))
        self.assertEqual((message["To"], message["From"]), ("a@test, b@test", "me@test"))
        self.assertEqual(message.get_body(("plain",)).get_content().strip(), report)
        html = message.get_body(("html",)).get_content()
        self.assertIn("<strong>Rebalance suggested.</strong>", html)
        self.assertIn('text-align:right;">&lt;$5</td>', html)
        self.assertIn('<a href="https://x.test/?a=1&amp;b=2">XLP</a>', html)


if __name__ == "__main__":
    unittest.main()
