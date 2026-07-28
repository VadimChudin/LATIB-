import asyncio
import ccxt.async_support as ccxt

async def main():
    exchange = ccxt.binanceusdm()
    try:
        markets = await exchange.load_markets()
        print("Successfully loaded markets.")
        symbols = list(markets.keys())
        print(f"Total symbols: {len(symbols)}")
        print("Checking for ZEC related symbols:")
        for s in symbols:
            if "ZEC" in s:
                print(f" - {s} (ID: {markets[s]['id']})")
    finally:
        await exchange.close()

if __name__ == "__main__":
    asyncio.run(main())
