import os
import json

def main():
    params_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'tick_params')
    config_out = os.path.join(os.path.dirname(__file__), '..', 'data', 'active_config.json')

    if not os.path.exists(params_dir):
        print(f"Error: {params_dir} does not exist. Did the DE finish?")
        return

    profitable = []
    total_files = 0

    for filename in os.listdir(params_dir):
        if not filename.endswith('.json'):
            continue
        total_files += 1
        filepath = os.path.join(params_dir, filename)
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                
                # Minimum filters: PnL >= 0.20, WinRate >= 50%, Trades >= 3
                test_pnl = data.get('test_pnl_r', -1.0)
                test_wr = data.get('test_wr', 0.0)
                total_trades = data.get('total_trades', 0)
                train_pnl = data.get('train_pnl_r', 0.0)
                
                if test_pnl >= 0.20 and test_wr >= 50 and total_trades >= 3:
                    profitable.append(data)
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    # Sort by Test PnL descending
    profitable.sort(key=lambda x: x.get('test_pnl_r', 0), reverse=True)

    print("="*60)
    print(f"Total coins analyzed: {total_files}")
    print(f"Profitable coins found (Whitelisted): {len(profitable)}")
    print("="*60)
    print(f"{'SYMBOL':<15} | {'TEST PNL (R)':<12} | {'TEST WR':<10} | {'TRADES'}")
    print("-" * 60)
    
    for p in profitable:
        sym = p.get('symbol', 'UNKNOWN')
        pnl = p.get('test_pnl_r', 0)
        wr = p.get('test_wr', 0)
        tr = p.get('total_trades', 0)
        print(f"{sym:<15} | +{pnl:<11.2f} | {wr:>5.1f}%     | {tr:>3}")
    print("="*60)

    # If you want to merge these into active_config.json
    if len(profitable) > 0:
        
        out_data = []

        # Add only profitable ones to config formatted exactly for config_loader.rs
        for p in profitable:
            sym = p.get('symbol')
            if sym:
                params_obj = p.get('params', {})
                # Format exactly as config_loader.rs expects:
                config_item = {
                    "symbol": sym,
                    "timeframe": "tick",
                    "strategy": "knife_tick",
                    "tier": 1,
                    "leverage": 10,
                    "params": params_obj,
                    "metrics": {
                        "win_rate": p.get('test_wr', 0) / 100.0,
                        "total_trades": p.get('total_trades', 0),
                        "score": p.get('test_pnl_r', 0)
                    }
                }
                out_data.append(config_item)

        # Save active config
        with open(config_out, 'w') as f:
            json.dump(out_data, f, indent=4)
        
        print(f"\n✅ Successfully saved active_config.json with {len(profitable)} whitelisted symbols!")
        print("You MUST run 'cargo build --release' in rust_engine if you haven't yet.")
        print("You can now run `python main.py` to start trading!")

if __name__ == '__main__':
    main()
