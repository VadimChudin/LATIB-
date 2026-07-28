import json
import numpy as np
import os

LOG_FILE = 'data/trade_log.jsonl'
OUTPUT_CONFIG = 'configs/active_configs.json'

MIN_TRADES = 10
MIN_WINRATE = 0.45  # RL often has lower WR but high TP:SL
MIN_TOTAL_R = 1.0   # Must be cleanly profitable

def analyze():
    if not os.path.exists(LOG_FILE):
        print(f"Error: {LOG_FILE} not found. Run the Rust backtester first!")
        return

    # 1. Parse trades
    trades_by_symbol = {}
    with open(LOG_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if '"EXIT"' not in line:
                continue
            try:
                trade = json.loads(line)
                # We want knife_tick that uses the RL model 
                # (You can modify this check if needed for density)
                if trade.get('strategy') == 'knife_tick':
                    sym = trade.get('symbol')
                    if sym not in trades_by_symbol:
                        trades_by_symbol[sym] = []
                    trades_by_symbol[sym].append(trade)
            except:
                pass

    if not trades_by_symbol:
        print("No knife_tick RL trades found in log.")
        return

    # 2. Evaluate each symbol
    passed_symbols = []
    
    print("\n========= RL KNIFE TICK RESULTS =========")
    print(f"{'Symbol':<12} | {'Trades':<8} | {'WR %':<8} | {'Total R':<8}")
    print("-" * 50)
    
    for sym, trades in trades_by_symbol.items():
        wins = sum(1 for t in trades if t.get('pnl_r', 0) > 0)
        total = len(trades)
        wr = wins / total
        
        # PnL in R (assuming Risk = 1R per trade)
        total_r = sum(t.get('pnl_r', 0) for t in trades)
        
        if total >= MIN_TRADES:
            print(f"{sym:<12} | {total:<8} | {wr*100:>5.1f}% | {total_r:>5.2f} R")
            if wr >= MIN_WINRATE and total_r >= MIN_TOTAL_R:
                passed_symbols.append(sym)
                
    # 3. Generate new config
    if passed_symbols:
        print(f"\n✅ Found {len(passed_symbols)} highly profitable coins!")
        
        # Creating basic configs for the Rust Engine to consume
        configs = []
        for sym in passed_symbols:
            cfg = {
                "symbol": sym,
                "timeframe": "1s",  # Tick/1s for Knife
                "strategy": "knife_tick",
                "tier": 1,
                "leverage": 10,
                # Parameters are now mostly handled by the RL 'brain'
                "params": {
                    "rl_enabled": True
                }
            }
            configs.append(cfg)
            
        with open(OUTPUT_CONFIG, 'w', encoding='utf-8') as f:
            json.dump(configs, f, indent=2)
            
        print(f"✅ Generated {OUTPUT_CONFIG} with {len(passed_symbols)} symbols ready for LIVE!")
    else:
        print("\n❌ No symbols passed the profitability thresholds. (Check RL training or adjust thresholds).")

if __name__ == "__main__":
    analyze()
