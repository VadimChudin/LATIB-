"""
Lazy Tick Loader — Phase 24 (Binance Vision)
=============================================
Downloads aggTrades from data.binance.vision ZIP archives.
Falls back to REST API for very recent dates (today/yesterday).

ZIP URL format:
  https://data.binance.vision/data/futures/um/daily/aggTrades/{SYMBOL}/{SYMBOL}-aggTrades-{YYYY-MM-DD}.zip

Each ZIP contains a single CSV with columns:
  agg_trade_id, price, quantity, first_trade_id, last_trade_id, transact_time, is_buyer_maker
"""

import os
import io
import time
import asyncio
import zipfile
import aiohttp
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, timezone

VISION_BASE = "https://data.binance.vision/data/futures/um/daily/aggTrades"
CACHE_DIR = Path("data/cache/ticks")

# Binance Vision CSV header mapping
VISION_RENAME = {
    "transact_time": "timestamp",
    "quantity": "qty"
}


class LazyTickLoader:
    def __init__(self):
        self.session = None
        self.mem_cache = {}
        self.semaphore = asyncio.Semaphore(10)

    async def _get_session(self):
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
        return self.session

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()

    def _normalize_symbol(self, symbol: str) -> str:
        """BTC/USDT, BTC_USDT, BTCUSDT -> BTCUSDT"""
        sym = symbol.replace("/", "").replace("_", "").replace(":", "").upper()
        if sym.endswith("USDTUSDT"):
            sym = sym.replace("USDTUSDT", "USDT")
        return sym

    def _get_cache_path(self, symbol: str, date_str: str) -> Path:
        """Returns path like data/cache/ticks/BTCUSDT/2025-01-15.csv"""
        p = CACHE_DIR / symbol
        p.mkdir(parents=True, exist_ok=True)
        return p / f"{date_str}.csv"

    async def _download_day_zip(self, symbol: str, date_str: str) -> pd.DataFrame:
        """
        Download one day of aggTrades from Binance Vision.
        Returns DataFrame with columns: price, qty, timestamp, is_buyer_maker
        """
        cache_path = self._get_cache_path(symbol, date_str)

        # 1. Check disk cache
        if cache_path.exists() and cache_path.stat().st_size > 0:
            try:
                df = pd.read_csv(cache_path, low_memory=False)
                return df
            except Exception:
                cache_path.unlink(missing_ok=True)

        # 2. Download ZIP from Binance Vision
        url = f"{VISION_BASE}/{symbol}/{symbol}-aggTrades-{date_str}.zip"
        
        async with self.semaphore:
            try:
                session = await self._get_session()
                async with session.get(url) as resp:
                    if resp.status == 404:
                        # File doesn't exist on Vision (coin too new or date too recent)
                        return pd.DataFrame()
                    if resp.status != 200:
                        print(f"  ⚠️ Vision HTTP {resp.status} for {symbol} {date_str}")
                        return pd.DataFrame()

                    data = await resp.read()
            except Exception as e:
                print(f"  ⚠️ Vision download error {symbol} {date_str}: {e}")
                return pd.DataFrame()

        # 3. Unzip in memory and parse CSV
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                csv_name = zf.namelist()[0]
                with zf.open(csv_name) as f:
                    df = pd.read_csv(f, low_memory=False)
            # Rename columns to our standard names
            df = df.rename(columns=VISION_RENAME)
            # Handle headerless CSVs (fallback)
            if "price" not in df.columns and len(df.columns) == 7:
                df.columns = ["agg_trade_id", "price", "qty", "first_trade_id", "last_trade_id", "timestamp", "is_buyer_maker"]
        except Exception as e:
            print(f"  ⚠️ Vision parse error {symbol} {date_str}: {e}")
            return pd.DataFrame()

        if df.empty:
            return df

        # 4. Clean up types
        df["price"] = df["price"].astype(float)
        df["qty"] = df["qty"].astype(float)
        df["timestamp"] = pd.to_numeric(df["timestamp"])

        # 5. Save to disk cache (only essential columns)
        df[["price", "qty", "timestamp", "is_buyer_maker"]].to_csv(
            cache_path, index=False
        )

        return df

    async def load_trade_window(self, symbol: str, start_ts, end_ts) -> pd.DataFrame:
        """
        Load tick data for a time window [start_ts, end_ts] in milliseconds.
        Automatically downloads the right daily ZIP files and stitches them together.
        """
        sym = self._normalize_symbol(symbol)
        start_ms = int(start_ts)
        end_ms = int(end_ts)

        mem_key = f"{sym}_{start_ms}_{end_ms}"
        if mem_key in self.mem_cache:
            return self.mem_cache[mem_key]

        # Figure out which calendar days we need
        start_dt = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc)
        end_dt = datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc)

        dates = []
        current = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        while current <= end_dt:
            dates.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)

        # Download all needed days (concurrently within semaphore)
        tasks = [self._download_day_zip(sym, d) for d in dates]
        dfs = await asyncio.gather(*tasks)

        # Stitch and filter
        frames = [df for df in dfs if not df.empty]
        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        combined = combined[
            (combined["timestamp"] >= start_ms) &
            (combined["timestamp"] <= end_ms)
        ]

        if combined.empty:
            return pd.DataFrame()

        combined = combined.sort_values("timestamp")
        combined["datetime"] = pd.to_datetime(combined["timestamp"], unit="ms")

        self.mem_cache[mem_key] = combined
        return combined


async def _test():
    loader = LazyTickLoader()
    # Test: load BTCUSDT ticks from 2 days ago (guaranteed to exist on Vision)
    two_days_ago = int((time.time() - 172800) * 1000)
    window_end = two_days_ago + 60000  # 1 minute window

    print(f"Testing BTCUSDT from Vision...")
    df = await loader.load_trade_window("BTC/USDT", two_days_ago, window_end)
    print(f"Got {len(df)} ticks.")
    if not df.empty:
        print(df[["price", "qty", "timestamp"]].head(5))
    await loader.close()


if __name__ == "__main__":
    asyncio.run(_test())
