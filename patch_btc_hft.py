"""
Quick patch: re-download BTC_USDT 5m with HFT fields (num_trades, taker_buy_volume, quote_volume)
"""
import asyncio, os, sys, time, pandas as pd
import ccxt.async_support as ccxt

for pv in ['http_proxy', 'https_proxy', 'all_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY']:
    os.environ.pop(pv, None)

CSV_PATH = "data/cache/BTC_USDT_5m_730d.csv"
DAYS = 730
BATCH_SIZE = 1500

async def main():
    exchange = ccxt.binance({
        "enableRateLimit": True, "timeout": 30000, "trust_env": False,
        "options": {"defaultType": "future"},
    })

    try:
        now_ms = int(time.time() * 1000)
        start_ms = now_ms - (DAYS * 24 * 60 * 60 * 1000)
        since = start_ms
        all_candles = []

        print(f"📥 Downloading BTC/USDT 5m with HFT fields ({DAYS} days)...")
        while since < now_ms:
            raw = await exchange.fapiPublicGetKlines({
                'symbol': 'BTCUSDT', 'interval': '5m',
                'startTime': since, 'limit': BATCH_SIZE,
            })
            if not raw: break
            for k in raw:
                all_candles.append([
                    int(k[0]), float(k[1]), float(k[2]), float(k[3]),
                    float(k[4]), float(k[5]),
                    int(k[8]),      # num_trades
                    float(k[9]),    # taker_buy_volume
                    float(k[7]),    # quote_volume
                ])
            last_ts = int(raw[-1][0])
            if last_ts <= since: break
            since = last_ts + 1
            if len(all_candles) % 10000 == 0:
                print(f"   ... {len(all_candles)} candles")
            await asyncio.sleep(0.01)

        df = pd.DataFrame(all_candles, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "num_trades", "taker_buy_volume", "quote_volume",
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

        # Verify HFT fields have data
        print(f"\n📊 Stats:")
        print(f"   Total candles: {len(df)}")
        print(f"   num_trades range: {df['num_trades'].min():.0f} - {df['num_trades'].max():.0f}")
        print(f"   taker_buy_volume range: {df['taker_buy_volume'].min():.2f} - {df['taker_buy_volume'].max():.2f}")
        print(f"   delta (computed): min={( 2*df['taker_buy_volume'] - df['volume']).min():.2f}, max={(2*df['taker_buy_volume'] - df['volume']).max():.2f}")

        os.makedirs("data/cache", exist_ok=True)
        df.to_csv(CSV_PATH, index=False)
        print(f"\n✅ Saved to {CSV_PATH}")

    finally:
        await exchange.close()

if __name__ == "__main__":
    asyncio.run(main())
