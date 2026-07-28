import asyncio
import aiohttp
import time
import requests

async def test_binance_raw():
    symbol = "DOGEUSDT"
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=5m&limit=300"
    
    print(f"[{time.strftime('%H:%M:%S')}] Attempting raw aiohttp fetch from {url}...")
    try:
        async with aiohttp.ClientSession(trust_env=False) as session:
            async with session.get(url, timeout=5.0) as resp:
                data = await resp.json()
                print(f"[{time.strftime('%H:%M:%S')}] aiohttp Success! Loaded {len(data)} candles.")
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] aiohttp Failed: {e}")

    print(f"\n[{time.strftime('%H:%M:%S')}] Attempting raw requests (sync) fetch from {url}...")
    try:
        # Naked synchronous request bypassing asyncio event loops entirely
        resp = requests.get(url, timeout=5.0)
        data = resp.json()
        print(f"[{time.strftime('%H:%M:%S')}] requests Success! Loaded {len(data)} candles.")
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] requests Failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_binance_raw())
