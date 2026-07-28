"""
Historical Data Downloader
==========================
Downloads 730 days (~2 years) of 5-minute OHLCV data from Binance Futures
using CCXT pagination. Saves to data/cache/{SYMBOL}_5m_730d.csv.

Usage: python download_historical.py
"""
import os
import sys
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
import asyncio
import time
import json
import pandas as pd
import ccxt.async_support as ccxt
from datetime import datetime, timedelta, timezone

# Strip proxies
for pv in ['http_proxy', 'https_proxy', 'all_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY']:
    os.environ.pop(pv, None)

# ── Configuration ────────────────────────────────────────────────────────────

MAX_INSTRUMENTS = 100  # Top N most liquid instruments by 24h volume

TIMEFRAMES = ["5m", "1m"]  # 5m for most strategies, 1m for ScalpMTF
DAYS = 730  # Default 2 years for 5m
DAYS_1M = 30  # 30 days for 1m (epicenter detection)
CACHE_DIR = os.path.join(os.path.dirname(__file__), "data", "cache")
BATCH_SIZE = 1500  # Binance max per request
DELAY_BETWEEN_REQUESTS = 1.0  # Rate limit safety


def _cache_path(symbol: str, timeframe: str, days: int = None) -> str:
    """Generate cache file path for a symbol."""
    safe_name = symbol.replace("/", "_")
    days_val = days or DAYS
    return os.path.join(CACHE_DIR, f"{safe_name}_{timeframe}_{days_val}d.csv")

# ── Concurrency ──────────────────────────────────────────────────────────────
MAX_CONCURRENT_SYMBOLS = 30  # Increased from 5 to 30 for speed
semaphore = asyncio.Semaphore(MAX_CONCURRENT_SYMBOLS)

async def download_symbol(exchange, symbol: str, timeframe: str, force: bool = False, clean_name: str = None, days_override: int = None):
    """Download history for one symbol with incremental updates."""
    async with semaphore:
        display_name = clean_name or symbol
        path = _cache_path(display_name, timeframe, days_override)
        
        now_ms = int(time.time() * 1000)
        actual_days = days_override or DAYS
        start_ms = now_ms - (actual_days * 24 * 60 * 60 * 1000)
        
        df_existing = None
        if os.path.exists(path) and not force:
            try:
                df_existing = pd.read_csv(path)
                if not df_existing.empty:
                    df_existing["timestamp"] = pd.to_datetime(df_existing["timestamp"])
                    last_ts = int(df_existing["timestamp"].iloc[-1].timestamp() * 1000)
                    if now_ms - last_ts < 300000: # Less than 5 mins old
                        print(f"  ⏭️  {display_name} ({timeframe}): Up to date.")
                        return True
                    start_ms = last_ts + 1
                    print(f"  🔄 {display_name} ({timeframe}): Incremental sync since {df_existing['timestamp'].iloc[-1]}...")
            except Exception as e:
                print(f"  ⚠️ Error reading cache for {display_name}: {e}. Re-downloading...")
                df_existing = None

        all_candles = []
        since = start_ms
        raw_symbol = symbol.replace("/", "")  # BTC/USDT → BTCUSDT for direct API
        while since < now_ms:
            try:
                # Use direct Binance API to get ALL 12 kline fields (CCXT fetch_ohlcv drops fields 6-11)
                raw_klines = await exchange.fapiPublicGetKlines({
                    'symbol': raw_symbol,
                    'interval': timeframe,
                    'startTime': since,
                    'limit': BATCH_SIZE,
                })
                if not raw_klines: break
                for k in raw_klines:
                    all_candles.append([
                        int(k[0]),       # 0: open_time
                        float(k[1]),     # 1: open
                        float(k[2]),     # 2: high
                        float(k[3]),     # 3: low
                        float(k[4]),     # 4: close
                        float(k[5]),     # 5: volume
                        int(k[8]),       # 6: num_trades
                        float(k[9]),     # 7: taker_buy_volume
                        float(k[7]),     # 8: quote_volume (vol in USDT)
                    ])
                last_ts = int(raw_klines[-1][0])
                if last_ts <= since: break
                since = last_ts + 1
                await asyncio.sleep(0.01) # Faster polling
            except Exception as e:
                print(f"     ❌ Error on {display_name}: {e}. Retrying...")
                await asyncio.sleep(1)
                continue

        if not all_candles and df_existing is None:
            print(f"  ❌ {display_name}: No data received!")
            return False

        if all_candles:
            df_new = pd.DataFrame(all_candles, columns=[
                "timestamp", "open", "high", "low", "close", "volume",
                "num_trades", "taker_buy_volume", "quote_volume",
            ])
            df_new["timestamp"] = pd.to_datetime(df_new["timestamp"], unit="ms")
            if df_existing is not None:
                df = pd.concat([df_existing, df_new]).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
            else:
                df = df_new
        else:
            df = df_existing

        # Fetch Funding Rate (Optimized)
        funding_start = int(df["timestamp"].iloc[0].timestamp() * 1000)
        if "funding_rate" not in df.columns or force:
            print(f"     📥 Fetching funding rates for {display_name}...")
            all_funding = []
            f_since = funding_start
            while f_since < now_ms:
                try:
                    fr_data = await exchange.fapiPublicGetFundingRate({'symbol': symbol.replace('/', ''), 'startTime': f_since, 'limit': 1000})
                    if not fr_data: break
                    for f in fr_data:
                        all_funding.append({"timestamp": int(f['fundingTime']), "funding_rate": float(f['fundingRate'])})
                    f_since = int(fr_data[-1]['fundingTime']) + 1
                    await asyncio.sleep(0.01)
                except: break
            
            if all_funding:
                fr_df = pd.DataFrame(all_funding)
                fr_df['timestamp'] = pd.to_datetime(fr_df['timestamp'], unit='ms')
                df = pd.merge_asof(df.sort_values("timestamp"), fr_df.sort_values("timestamp"), on="timestamp", direction="backward")
                df['funding_rate'] = df['funding_rate'].ffill().bfill().fillna(0.0)
            else:
                if "funding_rate" not in df.columns:
                    df['funding_rate'] = 0.0

        os.makedirs(CACHE_DIR, exist_ok=True)
        df.to_csv(path, index=False)
        print(f"  ✅ {display_name} ({timeframe}): Processed {len(df)} candles.")
        return True

async def main():
    exchange = ccxt.binance({
        "enableRateLimit": True,
        "timeout": 30000,
        "trust_env": False,
        "options": {"defaultType": "future"},
    })

    try:
        markets = await exchange.load_markets()
        dynamic_symbols = []
        for symbol, market in markets.items():
            if market.get('active', False) and market.get('linear', False) and market.get('settle') == 'USDT':
                dynamic_symbols.append((f"{market['base']}/{market['quote']}", symbol))
                
        tickers = await exchange.fetch_tickers()
        dynamic_symbols.sort(key=lambda p: tickers.get(p[1], {}).get('quoteVolume', 0) or 0, reverse=True)
        filtered = dynamic_symbols[:MAX_INSTRUMENTS]
        
        active_list = [clean_sym.replace("/", "_") for clean_sym, _ in filtered]
        os.makedirs("data", exist_ok=True)
        with open("data/top_symbols.json", "w") as f:
            json.dump(active_list, f, indent=2)

        # Smart Timeframe Filter: 1m only for ScalpMTF or Core symbols
        core_1m_symbols = {"BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "AVAX/USDT"}
        try:
            config_path = "data/active_config.json"
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    configs = json.load(f)
                    for cfg in configs:
                        if "ScalpMTF" in cfg.get("strategy", ""):
                            core_1m_symbols.add(cfg["symbol"].replace("_", "/"))
        except: pass

        print(f"🚀 Launching OPTIMIZED download for {len(filtered)} symbols (Concurrency: {MAX_CONCURRENT_SYMBOLS})...\n")
        tasks = []
        for clean_sym, ccxt_sym in filtered:
            # 5m: 730 days for all strategies
            tasks.append(download_symbol(exchange, ccxt_sym, "5m", clean_name=clean_sym))
            
            # 1m: only 30 days (for epicenter detection)
            tasks.append(download_symbol(exchange, ccxt_sym, "1m", clean_name=clean_sym, days_override=DAYS_1M))
        
        results = await asyncio.gather(*tasks)
        success = sum(1 for r in results if r)
        print(f"\n🏆 COMPLETED: {success}/{len(tasks)} tasks successful.")

    finally:
        await exchange.close()

if __name__ == "__main__":
    asyncio.run(main())
