import json
import re

RESULTS_FILE = "rust_engine/rl_results.txt"
MD_FILE = "top_rl_coins.md"

def main():
    lines = []
    with open(RESULTS_FILE, 'r', encoding='utf-16') as f:
        lines = f.readlines()
        
    coins = []
    for line in lines:
        if "|" in line and "Trades:" in line and "WinRate:" in line:
            parts = [p.strip() for p in line.split("|")]
            sym = parts[0].strip()
            
            try:
                trades_str = re.search(r'Trades:\s*(\d+)', parts[1]).group(1)
                wr_str = re.search(r'WinRate:\s*([\d\.]+)%', parts[2]).group(1)
                pnl_str = re.search(r'PnL:\s*([-\d\.]+)%', parts[3]).group(1)
                
                trades = int(trades_str)
                wr = float(wr_str)
                pnl = float(pnl_str)
                
                # Filter criteria
                if trades >= 150 and wr >= 55.0 and pnl > 0.0:
                    coins.append({
                        "sym": sym,
                        "trades": trades,
                        "wr": wr,
                        "pnl": pnl
                    })
            except Exception as e:
                pass
                
    # Sort by PnL Descending
    coins.sort(key=lambda x: x['pnl'], reverse=True)
    
    # Generate MD
    md = [
        "# 🏆 Top Profitable Knife Tick RL Symbols",
        "Based on `EvaluateRlAgent` with dynamic reinforcement learning sizing. Filtered for minimum 150 trades and >55% WinRate.",
        "",
        "| Symbol | Trades | WinRate (%) | Total PnL (%) |",
        "|--------|--------|-------------|---------------|"
    ]
    
    for c in coins:
        flag = "🔥" if c['wr'] > 80 else "✅" if c['wr'] > 70 else "⚖️"
        md.append(f"| {flag} `{c['sym']}` | {c['trades']:,} | {c['wr']}% | +{c['pnl']:,.2f}% |")
        
    with open(MD_FILE, 'w', encoding='utf-8') as f:
        f.write("\n".join(md))
        
    print(f"Generated {MD_FILE} with {len(coins)} symbols")

if __name__ == "__main__":
    main()
