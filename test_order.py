"""
test_order.py — Test real order placement on Binance Futures Testnet
Sends a DOGE/USDT LONG + SL + TP and verifies all orders were placed correctly.
Run: python test_order.py
"""

import asyncio
import os
import ccxt.async_support as ccxt
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

API_KEY    = os.getenv("BINANCE_API_KEY", "")
API_SECRET = os.getenv("BINANCE_API_SECRET", "")

SYMBOL    = "DOGE/USDT"
CONTRACTS = 100  # 100 DOGE contracts (~small test amount)
LEVERAGE  = 5

TESTNET_URLS = {
    'fapiPublic':    'https://testnet.binancefuture.com/fapi/v1',
    'fapiPublicV2':  'https://testnet.binancefuture.com/fapi/v2',
    'fapiPrivate':   'https://testnet.binancefuture.com/fapi/v1',
    'fapiPrivateV2': 'https://testnet.binancefuture.com/fapi/v2',
}


async def main():
    # Data exchange (production, public) — get current price
    data_ex = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'future'},
    })

    # Trading exchange (testnet, authenticated)
    trade_ex = ccxt.binanceusdm({
        'enableRateLimit': True,
        'urls': {'api': TESTNET_URLS},
    })
    trade_ex.apiKey = API_KEY
    trade_ex.secret = API_SECRET

    try:
        print("=" * 60)
        print("  AUTOCORE — TEST ORDER ON BINANCE FUTURES TESTNET")
        print("=" * 60)

        # 1. Get current price
        await data_ex.load_markets()
        ticker = await data_ex.fetch_ticker(SYMBOL)
        price  = ticker['last']
        print(f"\n[1] Current {SYMBOL} price: {price:.5f}")

        # 2. Set SL and TP (1% SL, 2% TP)
        sl_price = round(price * 0.99, 5)
        tp_price = round(price * 1.02, 5)
        print(f"    Entry:  {price:.5f}")
        print(f"    SL:     {sl_price:.5f}  (-1%)")
        print(f"    TP:     {tp_price:.5f}  (+2%)")

        # 3. Check balance
        balance = await trade_ex.fetch_balance()
        usdt_balance = float(balance['total'].get('USDT', 0))
        print(f"\n[2] Testnet USDT balance: {usdt_balance:.2f}")
        if usdt_balance < 10:
            print("    ⚠️  Balance too low! Go to testnet.binancefuture.com and refresh your balance.")
            return

        # 4. Set leverage
        await trade_ex.set_leverage(LEVERAGE, SYMBOL)
        print(f"\n[3] Leverage set to {LEVERAGE}x ✅")

        # 5. Place MARKET LONG entry
        print(f"\n[4] Placing MARKET BUY {CONTRACTS} {SYMBOL}...")
        entry_order = await trade_ex.create_order(
            symbol=SYMBOL,
            type='market',
            side='buy',
            amount=CONTRACTS
        )
        print(f"    Entry order placed! ID: {entry_order['id']} ✅")

        await asyncio.sleep(1)  # Short wait for fill

        # 6. Place STOP_MARKET (SL)
        print(f"\n[5] Placing STOP_MARKET SL at {sl_price:.5f}...")
        sl_order = await trade_ex.create_order(
            symbol=SYMBOL,
            type='STOP_MARKET',
            side='sell',
            amount=CONTRACTS,
            params={'stopPrice': sl_price, 'reduceOnly': True}
        )
        print(f"    SL order placed! ID: {sl_order['id']} ✅")

        # 7. Place TAKE_PROFIT_MARKET (TP)
        print(f"\n[6] Placing TAKE_PROFIT_MARKET TP at {tp_price:.5f}...")
        tp_order = await trade_ex.create_order(
            symbol=SYMBOL,
            type='TAKE_PROFIT_MARKET',
            side='sell',
            amount=CONTRACTS,
            params={'stopPrice': tp_price, 'reduceOnly': True}
        )
        print(f"    TP order placed! ID: {tp_order['id']} ✅")

        # 8. Verify open orders
        print(f"\n[7] Verifying open orders on testnet...")
        open_orders = await trade_ex.fetch_open_orders(SYMBOL)
        print(f"    Open orders count: {len(open_orders)}")
        for o in open_orders:
            print(f"    • {o['type']:25s} | {o['side']:4s} | stopPrice={o.get('stopPrice') or o.get('info', {}).get('stopPrice', 'N/A')}")

        # 9. Cancel all test orders
        print(f"\n[8] Cancelling all test orders + closing position...")
        await trade_ex.cancel_all_orders(SYMBOL)
        close_order = await trade_ex.create_order(
            symbol=SYMBOL,
            type='market',
            side='sell',
            amount=CONTRACTS,
            params={'reduceOnly': True}
        )
        print(f"    Position closed. ✅")

        print("\n" + "=" * 60)
        print("  ✅  ALL TESTS PASSED! Order placement works on testnet.")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await data_ex.close()
        await trade_ex.close()


if __name__ == "__main__":
    asyncio.run(main())
