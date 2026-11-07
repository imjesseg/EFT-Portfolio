"""Daily adjusted-close history, cached per day.

Yahoo Finance's public chart endpoint is the primary source. It is unofficial and can throttle or block,
so when it fails for a ticker and ``TIINGO_API_KEY`` is set, Tiingo's end-of-day API is used instead.
"""

from __future__ import annotations

import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?range=2y&interval=1d"
TIINGO_URL = "https://api.tiingo.com/tiingo/daily/{ticker}/prices?startDate={start}&token={token}"
CACHE_DIR = Path(".cache/prices")


@dataclass
class History:
    ticker: str
    name: str
    dates: list[str]
    prices: list[float]  # dividend/split-adjusted closes, oldest first
    source: str = "yahoo"


def _get_json(url: str):
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def parse_yahoo(ticker: str, raw: dict) -> History:
    closes = raw["indicators"]["adjclose"][0]["adjclose"]
    dates, prices = [], []
    for ts, close in zip(raw["timestamp"], closes):
        if close is not None:
            dates.append(datetime.fromtimestamp(ts, timezone.utc).date().isoformat())
            prices.append(float(close))
    meta = raw["meta"]
    return History(ticker, meta.get("longName") or meta.get("shortName") or ticker, dates, prices, "yahoo")


def parse_tiingo(ticker: str, rows: list[dict]) -> History:
    rows = [r for r in rows if r.get("adjClose") is not None]
    return History(ticker, ticker, [r["date"][:10] for r in rows], [float(r["adjClose"]) for r in rows], "tiingo")


def fetch_history(ticker: str, use_cache: bool = True) -> History:
    cache_file = CACHE_DIR / f"{ticker}-{date.today().isoformat()}.history.json"
    if use_cache and cache_file.exists():
        return History(**json.loads(cache_file.read_text()))
    try:
        history = parse_yahoo(ticker, _get_json(YAHOO_URL.format(ticker=ticker))["chart"]["result"][0])
    except Exception as yahoo_error:
        token = os.environ.get("TIINGO_API_KEY")
        if not token:
            raise
        start = (date.today() - timedelta(days=730)).isoformat()
        try:
            history = parse_tiingo(ticker, _get_json(TIINGO_URL.format(ticker=ticker, start=start, token=token)))
        except Exception as tiingo_error:
            raise RuntimeError(f"Yahoo: {yahoo_error}; Tiingo: {tiingo_error}") from tiingo_error
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(asdict(history)))
    return history


def fetch_all(tickers: list[str], use_cache: bool = True) -> tuple[dict[str, History], dict[str, str]]:
    """Fetch tickers concurrently. Returns (histories, errors by ticker)."""
    histories: dict[str, History] = {}
    errors: dict[str, str] = {}

    def task(ticker: str):
        try:
            return ticker, fetch_history(ticker, use_cache), None
        except Exception as exc:  # network/HTTP/parse failures are reported, not fatal
            return ticker, None, str(exc)

    with ThreadPoolExecutor(max_workers=8) as pool:
        for ticker, history, error in pool.map(task, tickers):
            if history:
                histories[ticker] = history
            else:
                errors[ticker] = error
    return histories, errors
