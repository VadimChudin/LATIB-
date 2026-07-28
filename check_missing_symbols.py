import ccxt
from download_historical import SYMBOLS

def check_binance_symbols():
    exchange = ccxt.binance({
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    })
    
    markets = exchange.load_markets()
    active_usdt_perps = []
    
    for symbol, market in markets.items():
        if market['active'] and market['linear'] and market['quote'] == 'USDT' and market['settle'] == 'USDT':
            active_usdt_perps.append(symbol)
            
    current_symbols = set(SYMBOLS)
    active_set = set(active_usdt_perps)
    
    missing = active_set - current_symbols
    dead = current_symbols - active_set
    
    print(f"Total Active USDT Perps on Binance: {len(active_set)}")
    print(f"Currently in our hardcoded list: {len(current_symbols)}")
    print(f"Missing from our list: {len(missing)}")
    print(f"Dead/Delisted in our list: {len(dead)}")
    
    if dead:
        print(f"Dead symbols: {dead}")
        
    print(f"\nExample missing symbols (Top 20):")
    print(sorted(list(missing))[:20])

if __name__ == "__main__":
    check_binance_symbols()
