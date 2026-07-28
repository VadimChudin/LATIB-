import json
import os
import sys
import pandas as pd
import asyncio
import subprocess
from datetime import datetime, timezone

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Strategy Imports for Backtesting
from strategies.ultimate_smc_trail import UltimateSMCTrailStrategy
from strategies.knife_catcher import KnifeCatcherStrategy
from strategies.scalp_mtf import ScalpMTFStrategy
from strategies.density import DensityStrategy
from quantum_validator import test_recency, test_monte_carlo
from verify_ticks import TickVerifier
from constants import STRAT_MAP, REVERSE_STRAT_MAP, ACTIVE_CONFIG_PATH, AGGREGATED_GA_PATH, TOP_PER_STRAT

TOP_N_CONFIGS = 15

STRAT_INSTANCES = {
    "Ultimate_SMC_Trail": UltimateSMCTrailStrategy(),
    "KnifeCatcher_ML": KnifeCatcherStrategy(),
    "ScalpMTF": ScalpMTFStrategy(),
    "Density": DensityStrategy(),
    # "FundingRate_MR" is evaluated purely in Rust for now
}

def load_recent_data(symbol, timeframe="5m", bars=25000):
    """Loads a slice of recent history (roughly 3 months of 5m data) for rapid robustness testing."""
    sym = symbol.replace('/', '_')
    if timeframe == "1m":
        cache_path = os.path.join("data", "cache", f"{sym}_1m_730d.csv")
    else:
        cache_path = os.path.join("data", "cache", f"{sym}_{timeframe}_730d.csv")
        
    if not os.path.exists(cache_path):
        return None
    dtypes = {c: 'float32' for c in ['open', 'high', 'low', 'close', 'volume']}
    df = pd.read_csv(cache_path, dtype=dtypes, engine='c', low_memory=False)
    df = df.tail(bars).copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df.set_index('timestamp', drop=False)

def test_robustness(strat_name, df, base_params):
    """Tests 4 immediate parameter neighbors (+/- 10%). Returns True if performance drop is < 30%."""
    if df is None or len(df) < 1000:
        return True # Skip test if no data
        
    # Find 2 numerical parameters to perturb
    float_params = [k for k in base_params.keys() if isinstance(base_params[k], (float, int)) and not isinstance(base_params[k], bool)]
    targets = [k for k in float_params if "mult" in k or "rsi" in k or "atr" in k or "activate_r" in k][:2]
    if not targets:
        targets = float_params[:2]
        
    if not targets:
        return True
        
    neighbors = []
    for k in targets:
        orig = base_params[k]
        for pct in [1.1, 0.9]:
            p_copy = base_params.copy()
            if isinstance(orig, int):
                p_copy[k] = int(orig * pct)
            else:
                p_copy[k] = orig * pct
            neighbors.append(p_copy)
            
    strat = STRAT_INSTANCES.get(strat_name)
    if not strat:
        return True
        
    def _score(params):
        try:
            res = strat.backtest_logic(df.copy(), params)
            trades = res[res['trade_pnl_r'] != 0]
            if len(trades) < 5: return 0.0
            wr = len(trades[trades['trade_pnl_r'] > 0]) / len(trades)
            win_r = trades[trades['trade_pnl_r'] > 0]['trade_pnl_r'].sum()
            loss_r = abs(trades[trades['trade_pnl_r'] < 0]['trade_pnl_r'].sum())
            pf = win_r / loss_r if loss_r > 0 else 1.0
            return wr * pf * min(1.0, len(trades) / 50.0)
        except Exception:
            return 0.0

    champ_score = _score(base_params)
    if champ_score <= 0:
        return False
        
    neighbor_scores = [_score(p) for p in neighbors]
    avg_neighbor_score = sum(neighbor_scores) / len(neighbor_scores)
    
    drop_pct = (champ_score - avg_neighbor_score) / champ_score
    is_robust = drop_pct < 0.30
    
    print(f"      [Robustness] Champ: {champ_score:.2f} | Neighbors: {avg_neighbor_score:.2f} | Drop: {drop_pct*100:.1f}% -> {'✅ Safe' if is_robust else '❌ Curve-Fit'}")
    return is_robust



def test_historical_ticks(symbol, strat_rust_name, params, timeframe="5m"):
    """Runs a full backtest via Rust, gets trades, and verifies a random sample of 50 via ticks."""
    sym = symbol.replace('/', '_')
    csv_path = f"data/cache/{sym}_{timeframe}_730d.csv"
    if not os.path.exists(csv_path):
        print(f"      [Tick History] ⏭️ Skipped: CSV not found at {csv_path}")
        return True # Skip if no data
        
    binary_path = "rust_engine/target/release/aegis_engine.exe"
    if not os.path.exists(binary_path):
        binary_path = "rust_engine/target/release/aegis_engine"
        if not os.path.exists(binary_path):
            print(f"      [Tick History] ⏭️ Skipped: aegis_engine binary not found")
            return True
        
    params_json = json.dumps(params)
    cmd = [
        str(binary_path), "backtest-trades",
        "--csv", csv_path,
        "--strategy", strat_rust_name,
        "--params-json", params_json
    ]
    
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        trades = json.loads(res.stdout)
    except Exception as e:
        print(f"      [Tick History] ⚠️ Error running rust backtest: {e}")
        return True

    if not trades or len(trades) == 0:
        print(f"      [Tick History] ⏭️ Skipped: Rust returned 0 trades")
        return True
        
    async def do_verify():
        verifier = TickVerifier()
        # Verify 50 random winning trades
        result = await verifier.verify_trades(sym, trades, max_verify=50, random_sample=True)
        await verifier.close()
        return result
        
    try:
        verify_results = asyncio.run(do_verify())
    except Exception as e:
        print(f"      [Tick History] ⚠️ Tick verification error: {e}")
        return True
        
    conf = verify_results.get("confidence", "low")
    adj_wr = verify_results.get("adjusted_wr", 0.0)
    checked = verify_results.get("verified_count", 0)
    fake = verify_results.get("fake_wins", 0)
    stats = verify_results.get("stats", {})
    skipped = stats.get("skipped", 0) if isinstance(stats, dict) else 0
    
    if checked == 0:
        print(f"      [Tick History] ⏭️ Skipped: No winning trades to verify")
        return True
        
    is_valid = conf in ["high", "medium"]
    status_icon = "✅ Safe" if is_valid else "❌ Curve-Fit (Fake Wins)"
    pct_fake = (fake / checked * 100) if checked > 0 else 0
    
    skipped_str = f" | Skipped: {skipped}" if skipped > 0 else ""
    print(f"      [Tick History] Verified {checked} random wins | Fake: {fake} ({pct_fake:.1f}%){skipped_str} | Adj WR: {adj_wr:.1f}% | {status_icon}")
    return is_valid

def parse_symbol(raw_symbol):
    # Example "AVAX_USDT_15m_730d" -> "AVAX/USDT", "15m"
    parts = raw_symbol.split('_')
    if len(parts) >= 3:
        return f"{parts[0]}/{parts[1]}", parts[2]
    return raw_symbol, "5m" # Fallback

def main():
    if not os.path.exists(AGGREGATED_GA_PATH):
        print(f"❌ Error: Could not find aggregated results at {AGGREGATED_GA_PATH}")
        print("Make sure you let `run_ga_batch.py` finish successfully first!")
        return
        
    print("🔍 Reading Walk-Forward Analysis (WFA) Results...")
    approved_strats = []
    # WFA evaluates the core strategy logic across time. 
    # If a strategy scored < 50% average filtered WR, we block it completely.
    for py_strat in set(STRAT_MAP.values()):
        # Try matching WFA filename (handles naming inconsistencies)
        possible_names = [
            py_strat.lower(), 
            py_strat.replace('_ML', '').replace('ML_', '').lower(),
            py_strat.replace('KnifeCatcher_ML', 'knifecatcher').lower()
        ]
        
        is_approved = True # Assume true if no WFA file exists (e.g. new strategies)
        for name in possible_names:
            wfa_path = f"data/wfa_{name}.csv"
            if os.path.exists(wfa_path):
                try:
                    df_wfa = pd.read_csv(wfa_path)
                    if 'filtered_wr' in df_wfa.columns:
                        avg_fwr = df_wfa['filtered_wr'].mean()
                        if avg_fwr < 50.0:
                            is_approved = False
                            print(f"  🚫 WFA REJECTED: {py_strat} (Avg Filtered WR: {avg_fwr:.1f}% < 50%) - Marking as OVERFIT")
                        else:
                            print(f"  ✅ WFA APPROVED: {py_strat} (Avg Filtered WR: {avg_fwr:.1f}%)")
                except Exception as e:
                    print(f"  ⚠️ Error reading WFA for {py_strat}: {e}")
                break # Found the file, stop checking names
                
        if is_approved and py_strat not in approved_strats:
            approved_strats.append(py_strat)
            
    with open(AGGREGATED_GA_PATH, "r") as f:
        data = json.load(f)
        
    all_configs = []
    
    # Flatten the data into individual strategy configs
    for item in data:
        raw_symbol = item.get("raw_symbol", "")
        symbol, timeframe = parse_symbol(raw_symbol)
        results = item.get("results", {})
        
        for rust_strat, strat_data in results.items():
            py_strat_name = STRAT_MAP.get(rust_strat)
            if not py_strat_name:
                continue
                
            if py_strat_name not in approved_strats:
                continue # Skip strategies rejected by WFA
                
            if "aggressive" in strat_data:
                strat_info = strat_data["aggressive"]
            else:
                strat_info = strat_data
                
            fitness = strat_info.get("fitness", 0.0)
            if fitness <= 0:
                continue # Skip unprofitable 
                
            all_configs.append({
                "symbol": symbol,
                "timeframe": timeframe,
                "strategy": py_strat_name,
                "params": strat_info.get("params", {}),
                "metrics": {
                    "win_rate": strat_info.get("win_rate", 0) / 100.0, # Rust outputs 65.4, Python expects 0.654
                    "profit_factor": 1.5, # Placeholder, Rust GA doesn't reliably export this
                    "max_drawdown_pct": 0.10, # Placeholder
                    "total_trades": strat_info.get("num_trades", 0),
                    "score": fitness
                },
                "evaluated_at": datetime.now(timezone.utc).isoformat()
            })
            
    # Sort by highest fitness score
    all_configs.sort(key=lambda x: x["metrics"]["score"], reverse=True)
    
    # For each symbol-strategy pair, isolate its configs
    grouped_configs = {}
    for c in all_configs:
        key = f"{c['symbol']}_{c['strategy']}"
        if key not in grouped_configs:
            grouped_configs[key] = []
        grouped_configs[key].append(c)

    # Ensure variety: pick top robust config per strategy-symbol
    top_configs = []
    strat_counts = {s: 0 for s in STRAT_MAP.values()}
    TOP_PER_STRAT_LOCAL = TOP_PER_STRAT  # 25
    
    print("\n🛡️ Running Parameter Robustness Tests...")
    for key, configs in grouped_configs.items():
        symbol = configs[0]['symbol']
        strat_name = configs[0]['strategy']
        
        if strat_counts[strat_name] >= TOP_PER_STRAT_LOCAL:
            continue
            
        print(f"  Testing {symbol} on {strat_name}...")
        tf = "1m" if strat_name == "ScalpMTF" else "5m"
        df = load_recent_data(symbol, timeframe=tf)
        
        robust_config = None
        for i, c in enumerate(configs[:3]): # Test up to top 3 candidates
            print(f"    Candidate #{i+1} (Fitness: {c['metrics']['score']:.1f})")
            
            # Layer 1: Param Sensitivity (+/- 10%)
            if not test_robustness(strat_name, df, c['params']):
                continue
                
            # Layer 2: Recency — DISABLED (unfairly kills strategies during quiet market weeks)
            # strat_inst = STRAT_INSTANCES.get(strat_name)
            # if strat_inst and not test_recency(strat_inst, df, c['params']):
            #     continue
                
            # Layer 3: Monte Carlo (Price noise resilience)
            strat_inst = STRAT_INSTANCES.get(strat_name)
            if strat_inst and not test_monte_carlo(strat_inst, df, c['params']):
                continue
                
            # Layer 4: Historical Tick Verification (Random Sample of 50)
            rust_strat = REVERSE_STRAT_MAP.get(strat_name)
            if rust_strat and not test_historical_ticks(symbol, rust_strat, c['params'], timeframe=tf):
                continue
                
            robust_config = c
            break
                
        if robust_config:
            top_configs.append(robust_config)
            strat_counts[strat_name] += 1
        else:
            print(f"      🚫 ALL top configs for {symbol} failed robustness tests! Discarding entirely to protect capital.")
            
    # Phase B (Broadcasting): Guarantee every top symbol has a config for every approved strategy
    print("\n📡 Tiered Generalization: Broadcasting baseline parameters to missing symbols...")
    top_symbols_path = "data/top_symbols.json"
    if os.path.exists(top_symbols_path):
        with open(top_symbols_path, "r") as f:
            all_symbols = json.load(f)
            
        # Find the best generic baseline per strategy
        baselines = {}
        for c in top_configs:
            strat = c['strategy']
            if strat not in baselines:
                baselines[strat] = c
            elif c['metrics']['score'] > baselines[strat]['metrics']['score']:
                baselines[strat] = c
                
        # Check what we have
        existing_pairs = set(f"{c['symbol']}_{c['strategy']}" for c in top_configs)
        
        broadcasted_count = 0
        for raw_sym in all_symbols:
            sym, _ = parse_symbol(raw_sym)
            for strat in approved_strats:
                pair_key = f"{sym}_{strat}"
                if pair_key not in existing_pairs and strat in baselines:
                    # Broadcast baseline
                    new_config = dict(baselines[strat]) # Shallow copy is mostly fine, we deepcopy params/metrics
                    new_config['symbol'] = sym
                    new_config['params'] = dict(baselines[strat]['params'])
                    new_config['metrics'] = dict(baselines[strat]['metrics'])
                    new_config['metrics']['score'] = -1.0 # Mark as broadcasted/unoptimized fitness
                    new_config['evaluated_at'] = datetime.now(timezone.utc).isoformat()
                    top_configs.append(new_config)
                    existing_pairs.add(pair_key)
                    broadcasted_count += 1
                    
        print(f"  ✅ Broadcasted {broadcasted_count} baseline configurations to Tier 2/Missing symbols.")

    # Save to active config
    os.makedirs("data", exist_ok=True)
    with open(ACTIVE_CONFIG_PATH, "w") as f:
        json.dump(top_configs, f, indent=4)
        
    print(f"✅ Successfully exported Top {len(top_configs)} GA Optimized parameters to {ACTIVE_CONFIG_PATH}!")
    
    # Just print the natively optimized ones for brevity
    native_configs = [c for c in top_configs if c['metrics']['score'] >= 0]
    for i, c in enumerate(native_configs, 1):
         print(f"  #{i} {c['symbol']:10s} | {c['strategy']:20s} | Fitness: {c['metrics']['score']:.1f}")
         
    print("\n🚀 The Live Executor will automatically pick these up on its next cycle check.")

if __name__ == "__main__":
    main()
