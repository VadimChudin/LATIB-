"""Quick: download BTC 1m with HFT fields"""
import asyncio, os, time, pandas as pd
import ccxt.async_support as ccxt

for pv in ['http_proxy','https_proxy','all_proxy','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY']:
    os.environ.pop(pv, None)

CSV_PATH = "data/cache/BTC_USDT_1m_730d.csv"
DAYS = 730
BATCH = 1500

async def main():
    ex = ccxt.binance({"enableRateLimit":True,"timeout":30000,"trust_env":False,"options":{"defaultType":"future"}})
    try:
        now_ms = int(time.time()*1000)
        since = now_ms - (DAYS*86400000)
        all_c = []
        print(f"📥 BTC/USDT 1m + HFT fields ({DAYS}d)...")
        while since < now_ms:
            raw = await ex.fapiPublicGetKlines({'symbol':'BTCUSDT','interval':'1m','startTime':since,'limit':BATCH})
            if not raw: break
            for k in raw:
                all_c.append([int(k[0]),float(k[1]),float(k[2]),float(k[3]),float(k[4]),float(k[5]),int(k[8]),float(k[9]),float(k[7])])
            last = int(raw[-1][0])
            if last <= since: break
            since = last+1
            if len(all_c) % 50000 == 0: print(f"   ... {len(all_c)} candles")
            await asyncio.sleep(0.01)
        df = pd.DataFrame(all_c, columns=["timestamp","open","high","low","close","volume","num_trades","taker_buy_volume","quote_volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        print(f"\n📊 {len(df)} candles | trades: {df['num_trades'].min():.0f}-{df['num_trades'].max():.0f} | delta: {(2*df['taker_buy_volume']-df['volume']).min():.0f} to {(2*df['taker_buy_volume']-df['volume']).max():.0f}")
        os.makedirs("data/cache", exist_ok=True)
        df.to_csv(CSV_PATH, index=False)
        print(f"✅ Saved {CSV_PATH}")
    finally:
        await ex.close()

if __name__ == "__main__":
    asyncio.run(main())
