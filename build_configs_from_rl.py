import json
import re

RESULTS_FILE = "rust_engine/rl_results.txt"
CONFIG_FILE = "data/active_configs.json"

MIN_TRADES = 150        # Ignore statistically insignificant coins
MIN_WINRATE = 55.0      # At least 55% WR

def main():
    configs = []
    
    with open(RESULTS_FILE, 'r', encoding='utf-16') as f:
        lines = f.readlines()
        
    for line in lines:
        # Example format:
        # ZRO_USDT   | Trades: 9237 | WinRate: 81.1% | PnL: 2914.67%
        if "|" in line and "Trades:" in line and "WinRate:" in line:
            parts = [p.strip() for p in line.split("|")]
            sym = parts[0].strip()
            
            # extract numbers
            try:
                trades_str = re.search(r'Trades:\s*(\d+)', parts[1]).group(1)
                wr_str = re.search(r'WinRate:\s*([\d\.]+)%', parts[2]).group(1)
                pnl_str = re.search(r'PnL:\s*([-\d\.]+)%', parts[3]).group(1)
                
                trades = int(trades_str)
                wr = float(wr_str)
                pnl = float(pnl_str)
                
                if trades >= MIN_TRADES and wr >= MIN_WINRATE and pnl > 0.0:
                    configs.append({
                        "symbol": sym,
                        "timeframe": "1s",
                        "strategy": "knife_tick",
                        "tier": 1,
                        "leverage": 10,
                        "params": {
                            "rl_enabled": True
                        }
                    })
            except Exception as e:
                pass

    if len(configs) > 0:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(configs, f, indent=2)
        print(f"✅ Generated {CONFIG_FILE} with {len(configs)} highly profitable symbols!")
    else:
        print("❌ No symbols met the criteria. Check thresholds.")

if __name__ == "__main__":
    main()
