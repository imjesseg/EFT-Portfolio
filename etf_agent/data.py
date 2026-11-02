"""Daily adjusted-close history from Yahoo Finance's public chart endpoint, cached per day."""

from __future__ import annotations

import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?range=2y&interval=1d"
CACHE_DIR = Path(".cache/prices")


@dataclass
class History:
    ticker: str
    name: str
    dates: list[str]
    prices: list[float]  # dividend/split-adjusted closes, oldest first


def _download(ticker: str) -> dict:
    request = urllib.request.Request(CHART_URL.format(ticker=ticker), headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)["chart"]["result"][0]


def fetch_history(ticker: str, use_cache: bool = True) -> History:
    cache_file = CACHE_DIR / f"{ticker}-{date.today().isoformat()}.json"
    if use_cache and cache_file.exists():
        raw = json.loads(cache_file.read_text())
    else:
        raw = _download(ticker)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(raw))

    closes = raw["indicators"]["adjclose"][0]["adjclose"]
    dates, prices = [], []
    for ts, close in zip(raw["timestamp"], closes):
        if close is not None:
            dates.append(datetime.fromtimestamp(ts, timezone.utc).date().isoformat())
            prices.append(float(close))
    meta = raw["meta"]
    return History(ticker, meta.get("longName") or meta.get("shortName") or ticker, dates, prices)


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
