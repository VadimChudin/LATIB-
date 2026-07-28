"""
DEEP AUDIT: Verify DE backtester knife_tick.rs logic in Python
=============================================================
This re-implements the EXACT same logic as knife_tick.rs evaluate_epicenter()
in Python so we can inspect every step and find hidden bugs.

We'll load real epicenter CSVs and trace through to see if results match
what DE claims.
"""
import os, json, csv, sys
from pathlib import Path

def load_ticks(csv_path):
    """Load tick data from CSV"""
    ticks = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) >= 4:
                ticks.append({
                    'ts_ms': int(row[0]),
                    'price': float(row[1]),
                    'qty': float(row[2]),
                    'is_buyer_maker': row[3].strip().lower() == 'true'
                })
    ticks.sort(key=lambda t: t['ts_ms'])
    return ticks


def evaluate_epicenter_python(ticks, direction, params):
    """
    EXACT replica of knife_tick.rs evaluate_epicenter in Python.
    Returns dict with trade details or None.
    """
    if len(params) < 5:
        return None

    window_ms = int(params[0])
    min_zscore = params[1]
    min_vol_spike = params[2] if len(params) > 2 else 1.5
    tp_recovery_pct = 0.8  # FIXED
    sl_buffer_pct = params[4]
    be_trigger_pct = params[5] if len(params) > 5 else 0.003
    trail_pct = params[6] if len(params) > 6 else 0.002
    micro_window_ms = int(params[7]) if len(params) > 7 else 500
    min_absorption = params[8] if len(params) > 8 else 2.0
    min_reclaim_pct = params[9] if len(params) > 9 else 0.001
    max_speed_mult = params[10] if len(params) > 10 else 3.0
    baseline_window_ms = int(params[11] * 1000) if len(params) > 11 else 30000
    max_absorber_ms = int(params[12] * 1000) if len(params) > 12 else 30000

    if len(ticks) < 20:
        return None

    # === BASELINE ===
    start_ts = ticks[0]['ts_ms']
    baseline_end = start_ts + baseline_window_ms

    baseline_prices = []
    baseline_volume = 0.0
    baseline_trade_count = 0

    for t in ticks:
        if t['ts_ms'] > baseline_end:
            break
        baseline_prices.append(t['price'])
        baseline_volume += t['price'] * t['qty']
        baseline_trade_count += 1

    if len(baseline_prices) < 10:
        return None
    if baseline_volume < 10000.0:
        baseline_volume = 10000.0
    if baseline_trade_count < 20:
        baseline_trade_count = 20

    baseline_secs = baseline_window_ms / 1000.0
    baseline_tps = baseline_trade_count / baseline_secs

    baseline_high = max(baseline_prices)
    baseline_low = min(baseline_prices)
    base_price = baseline_prices[-1]
    baseline_range_pct = (baseline_high - baseline_low) / base_price
    time_scale = (window_ms / baseline_window_ms) ** 0.5
    window_std = max(baseline_range_pct * time_scale, 0.00005)
    baseline_vol_per_window = baseline_volume * (window_ms / baseline_window_ms)

    baseline_absorption = (baseline_volume / baseline_range_pct) if baseline_range_pct > 0.00001 else 1000000.0

    # === STATE MACHINE ===
    from collections import deque
    window = deque()
    current_cvd = 0.0

    squeeze_detected = False
    squeeze_cvd = 0.0
    dump_start_price = base_price
    local_extreme = base_price
    local_extreme_ts = 0
    entry_idx = None

    for i, tick in enumerate(ticks):
        if tick['ts_ms'] <= baseline_end:
            continue

        # Update CVD
        quote_qty = tick['qty'] * tick['price']
        vol = -quote_qty if tick['is_buyer_maker'] else quote_qty
        current_cvd += vol

        # Maintain window
        window.append(i)
        while window and (tick['ts_ms'] - ticks[window[0]]['ts_ms']) > window_ms:
            window.popleft()

        if not squeeze_detected:
            if window:
                first_idx = window[0]
                first_tick = ticks[first_idx]

                if direction == "LONG":
                    price_move_pct = (first_tick['price'] - tick['price']) / first_tick['price']
                else:
                    price_move_pct = (tick['price'] - first_tick['price']) / first_tick['price']

                zscore = price_move_pct / window_std

                if zscore >= min_zscore:
                    window_volume = sum(ticks[idx]['price'] * ticks[idx]['qty'] for idx in window)
                    vol_spike = (window_volume / baseline_vol_per_window) if baseline_vol_per_window > 0 else 1.0

                    if vol_spike >= min_vol_spike:
                        squeeze_detected = True
                        squeeze_cvd = current_cvd
                        dump_start_price = first_tick['price']
                        local_extreme = tick['price']
                        local_extreme_ts = tick['ts_ms']
        else:
            # Track local extreme
            if direction == "LONG":
                if tick['price'] < local_extreme:
                    local_extreme = tick['price']
                    local_extreme_ts = tick['ts_ms']
            else:
                if tick['price'] > local_extreme:
                    local_extreme = tick['price']
                    local_extreme_ts = tick['ts_ms']

            if tick['ts_ms'] < local_extreme_ts + micro_window_ms:
                continue
            if tick['ts_ms'] > local_extreme_ts + max_absorber_ms:
                break

            # CHECK 1: CVD divergence
            if direction == "LONG":
                if not (current_cvd > squeeze_cvd):
                    continue
            else:
                if not (current_cvd < squeeze_cvd):
                    continue

            # CHECK 2: ABSORPTION
            target_micro_ts = tick['ts_ms'] - micro_window_ms
            micro_volume = 0.0
            micro_trades = 0
            micro_high = -1e18
            micro_low = 1e18

            for idx in reversed(list(window)):
                t = ticks[idx]
                if t['ts_ms'] < target_micro_ts:
                    break
                micro_volume += t['price'] * t['qty']
                micro_trades += 1
                if t['price'] > micro_high: micro_high = t['price']
                if t['price'] < micro_low: micro_low = t['price']

            if micro_trades < 3:
                continue

            micro_range = (micro_high - micro_low) / tick['price']
            if micro_range > 0.000001:
                absorption_ratio = micro_volume / (micro_range * tick['price'])
            else:
                absorption_ratio = micro_volume * 100.0

            if absorption_ratio < baseline_absorption * min_absorption:
                continue

            # CHECK 3: SPEED
            micro_seconds = max(micro_window_ms / 1000.0, 0.001)
            micro_tps = micro_trades / micro_seconds
            if micro_tps > baseline_tps * max_speed_mult:
                continue

            # CHECK 4: RECLAIM
            if direction == "LONG":
                reclaim = (tick['price'] - local_extreme) / local_extreme
            else:
                reclaim = (local_extreme - tick['price']) / local_extreme
            if reclaim < min_reclaim_pct:
                continue

            # ALL PASSED
            entry_idx = i
            break

    # === TRADE SIM ===
    if entry_idx is None:
        return None

    e_idx = entry_idx
    if e_idx + 5 >= len(ticks):
        return None

    taker_fee = 0.0005

    # Grid: avg of 5 ticks after entry signal
    grid_prices = [ticks[e_idx + offset]['price'] for offset in range(1, 6)]
    avg_entry_raw = sum(grid_prices) / len(grid_prices)

    if direction == "LONG":
        entry_price = avg_entry_raw * (1.0 + taker_fee)
    else:
        entry_price = avg_entry_raw * (1.0 - taker_fee)

    # Dynamic TP
    dump_size = abs(dump_start_price - local_extreme)
    tp_dist = dump_size * tp_recovery_pct

    if direction == "LONG":
        sl_price = local_extreme * (1.0 - sl_buffer_pct)
        tp_price = local_extreme + tp_dist
        be_trigger_price = entry_price * (1.0 + be_trigger_pct)
    else:
        sl_price = local_extreme * (1.0 + sl_buffer_pct)
        tp_price = local_extreme - tp_dist
        be_trigger_price = entry_price * (1.0 - be_trigger_pct)

    initial_sl = sl_price
    risk = abs(entry_price - sl_price)
    is_breakeven = False
    best_price = entry_price
    exit_price = entry_price
    pnl_r = 0.0
    mfe = 0.0
    exit_reason = "TIMEOUT"

    # =========================================================================
    # KEY AREA: Trade simulation loop - THIS IS WHERE BUGS COULD LURK
    # =========================================================================
    for tick in ticks[e_idx + 6:]:
        p_raw = tick['price']

        # FEE APPLICATION
        if direction == "LONG":
            p = p_raw * (1.0 - taker_fee)  # sell price is worse
        else:
            p = p_raw * (1.0 + taker_fee)  # buy-to-close price is worse

        # MFE tracking
        if direction == "LONG":
            favorable = (p - entry_price) / entry_price * 100.0
        else:
            favorable = (entry_price - p) / entry_price * 100.0
        if favorable > mfe:
            mfe = favorable

        if direction == "LONG":
            if p > best_price:
                best_price = p

            if not is_breakeven and p >= be_trigger_price:
                sl_price = entry_price
                is_breakeven = True

            if p >= tp_price:
                exit_price = p
                pnl_r = (exit_price - entry_price) / risk if risk > 0 else 0
                exit_reason = "TP"
                break

            if p <= sl_price:
                exit_price = p
                pnl_r = (exit_price - entry_price) / risk if risk > 0 else 0
                exit_reason = "SL"
                break

            if is_breakeven:
                trailing_sl = best_price * (1.0 - trail_pct)
                if trailing_sl > sl_price:
                    sl_price = trailing_sl
                if p <= sl_price:
                    exit_price = p
                    pnl_r = (exit_price - entry_price) / risk if risk > 0 else 0
                    exit_reason = "TRAIL"
                    break
        else:  # SHORT
            if p < best_price:
                best_price = p

            if not is_breakeven and p <= be_trigger_price:
                sl_price = entry_price
                is_breakeven = True

            if p <= tp_price:
                exit_price = p
                pnl_r = (entry_price - exit_price) / risk if risk > 0 else 0
                exit_reason = "TP"
                break

            if p >= sl_price:
                exit_price = p
                pnl_r = (entry_price - exit_price) / risk if risk > 0 else 0
                exit_reason = "SL"
                break

            if is_breakeven:
                trailing_sl = best_price * (1.0 + trail_pct)
                if trailing_sl < sl_price:
                    sl_price = trailing_sl
                if p >= sl_price:
                    exit_price = p
                    pnl_r = (entry_price - exit_price) / risk if risk > 0 else 0
                    exit_reason = "TRAIL"
                    break

    # Time runs out (last tick)
    if exit_reason == "TIMEOUT":
        last_p_raw = ticks[-1]['price']
        if direction == "LONG":
            exit_price = last_p_raw * (1.0 - taker_fee)
            pnl_r = (exit_price - entry_price) / risk if risk > 0 else 0
        else:
            exit_price = last_p_raw * (1.0 + taker_fee)
            pnl_r = (entry_price - exit_price) / risk if risk > 0 else 0

    return {
        'entry_idx': e_idx,
        'direction': direction,
        'entry_price': entry_price,
        'exit_price': exit_price,
        'sl_price': initial_sl,
        'tp_price': tp_price,
        'pnl_r': pnl_r,
        'risk_dist': risk,
        'mfe_pct': mfe,
        'exit_reason': exit_reason,
        'dump_start': dump_start_price,
        'local_extreme': local_extreme,
        'dump_size_pct': abs(dump_start_price - local_extreme) / dump_start_price * 100,
        'entry_vs_extreme_pct': abs(entry_price - local_extreme) / local_extreme * 100,
        'sl_dist_pct': risk / entry_price * 100,
        'tp_dist_pct': abs(tp_price - entry_price) / entry_price * 100,
        'remaining_ticks': len(ticks) - e_idx - 6,
    }


def main():
    # Load active config to get params
    with open('data/active_config.json', 'r') as f:
        configs = json.load(f)

    param_order = [
        "window_ms", "min_zscore", "min_vol_spike", "(unused_tp)", "sl_buffer_pct",
        "be_trigger_pct", "trail_pct", "micro_window_ms",
        "min_absorption", "min_reclaim_pct", "max_speed_mult",
        "baseline_window_sec", "max_absorber_sec", "rewake_cooldown_sec"
    ]

    symbols_found = 0
    total_trades = 0
    total_pnl = 0.0
    wins = 0
    losses = 0
    
    detailed_results = []
    
    # BUG TRACKERS
    bugs = {
        'tp_above_entry_long': 0,  # TP should be > entry for LONG
        'tp_below_entry_short': 0,
        'sl_above_entry_long': 0,  # SL should be < entry for LONG
        'sl_below_entry_short': 0,
        'entry_below_extreme_long': 0,  # Entry should be > local_extreme for LONG
        'entry_above_extreme_short': 0,
        'negative_risk': 0,
        'tiny_remaining_ticks': 0,
        'timeout_exits': 0,
        'dump_tiny': 0,  # dump size < 0.1%
    }

    for cfg in configs:
        if cfg.get('strategy') != 'knife_tick':
            continue
        
        sym = cfg['symbol']
        params_dict = cfg.get('params', {})
        
        # Build params vector
        params = []
        for name in param_order:
            val = params_dict.get(name, 0.0)
            if val is None:
                val = 0.0
            params.append(float(val))
        
        # Find epicenter files
        ep_dir = Path(f'data/epicenters_ticks/{sym}')
        if not ep_dir.exists():
            continue
        
        symbols_found += 1
        sym_trades = 0
        sym_pnl = 0.0
        sym_wins = 0
        
        for direction in ['LONG', 'SHORT']:
            dir_path = ep_dir / direction
            if not dir_path.exists():
                continue
            
            csv_files = sorted(dir_path.glob('*.csv'))
            
            for csv_file in csv_files:
                ticks = load_ticks(csv_file)
                if not ticks:
                    continue
                
                result = evaluate_epicenter_python(ticks, direction, params)
                
                if result is None:
                    continue
                
                total_trades += 1
                sym_trades += 1
                total_pnl += result['pnl_r']
                sym_pnl += result['pnl_r']
                
                if result['pnl_r'] > 0:
                    wins += 1
                    sym_wins += 1
                else:
                    losses += 1
                
                # BUG CHECKS
                if direction == "LONG":
                    if result['tp_price'] < result['entry_price']:
                        bugs['tp_above_entry_long'] += 1
                    if result['sl_price'] > result['entry_price']:
                        bugs['sl_above_entry_long'] += 1
                    if result['entry_price'] < result['local_extreme']:
                        bugs['entry_below_extreme_long'] += 1
                else:
                    if result['tp_price'] > result['entry_price']:
                        bugs['tp_below_entry_short'] += 1
                    if result['sl_price'] < result['entry_price']:
                        bugs['sl_below_entry_short'] += 1
                    if result['entry_price'] > result['local_extreme']:
                        bugs['entry_above_extreme_short'] += 1
                
                if result['risk_dist'] <= 0:
                    bugs['negative_risk'] += 1
                if result['remaining_ticks'] < 50:
                    bugs['tiny_remaining_ticks'] += 1
                if result['exit_reason'] == 'TIMEOUT':
                    bugs['timeout_exits'] += 1
                if result['dump_size_pct'] < 0.1:
                    bugs['dump_tiny'] += 1
                
                detailed_results.append({
                    'symbol': sym,
                    **result,
                    'csv_file': str(csv_file.name),
                })
        
        if sym_trades > 0:
            wr = sym_wins / sym_trades * 100
            print(f"  {sym:15s}: {sym_trades:3d} trades | WR={wr:5.1f}% | PnL R={sym_pnl:+8.2f}")

    print(f"\n{'='*80}")
    print(f"=== PYTHON REPLICA RESULTS ===")
    print(f"{'='*80}")
    print(f"Symbols: {symbols_found}")
    print(f"Total trades: {total_trades}")
    print(f"Wins: {wins}, Losses: {losses}")
    if total_trades > 0:
        print(f"Win Rate: {wins/total_trades*100:.1f}%")
        print(f"Total PnL R: {total_pnl:+.2f}")
        print(f"Avg PnL R: {total_pnl/total_trades:+.4f}")
    
    print(f"\n{'='*80}")
    print(f"=== BUG DETECTION ===")
    print(f"{'='*80}")
    for bug_name, count in bugs.items():
        flag = "!! BUG" if count > 0 else "   OK"
        print(f"  {flag} {bug_name}: {count}")
    
    # === CRITICAL CHECK: TP/SL ratio analysis ===
    print(f"\n{'='*80}")
    print(f"=== TP/SL RATIO ANALYSIS ===")
    print(f"{'='*80}")
    
    if detailed_results:
        tp_dists = [r['tp_dist_pct'] for r in detailed_results]
        sl_dists = [r['sl_dist_pct'] for r in detailed_results]
        rr_ratios = [r['tp_dist_pct'] / r['sl_dist_pct'] for r in detailed_results if r['sl_dist_pct'] > 0]
        dump_sizes = [r['dump_size_pct'] for r in detailed_results]
        entry_vs_ext = [r['entry_vs_extreme_pct'] for r in detailed_results]
        remaining = [r['remaining_ticks'] for r in detailed_results]
        
        print(f"  Avg TP dist: {sum(tp_dists)/len(tp_dists):.3f}%")
        print(f"  Avg SL dist: {sum(sl_dists)/len(sl_dists):.3f}%")
        print(f"  Avg R:R ratio: {sum(rr_ratios)/len(rr_ratios):.2f}")
        print(f"  Avg dump size: {sum(dump_sizes)/len(dump_sizes):.3f}%")
        print(f"  Avg entry vs extreme: {sum(entry_vs_ext)/len(entry_vs_ext):.3f}%")
        print(f"  Avg remaining ticks after entry: {sum(remaining)/len(remaining):.0f}")
        print(f"  Min remaining ticks: {min(remaining)}")
        
        # Exit reason breakdown
        reasons = {}
        for r in detailed_results:
            er = r['exit_reason']
            if er not in reasons:
                reasons[er] = {'count': 0, 'pnl': 0.0}
            reasons[er]['count'] += 1
            reasons[er]['pnl'] += r['pnl_r']
        
        print(f"\n  Exit reasons:")
        for er, data in sorted(reasons.items()):
            avg = data['pnl']/data['count']
            print(f"    {er:10s}: {data['count']:4d} trades | Total R: {data['pnl']:+8.2f} | Avg R: {avg:+.3f}")
        
        # MOST IMPORTANT: How many TIMEOUT trades there are and their PnL
        timeout_trades = [r for r in detailed_results if r['exit_reason'] == 'TIMEOUT']
        if timeout_trades:
            print(f"\n  === TIMEOUT DEEP DIVE ===")
            print(f"  Timeout trades: {len(timeout_trades)} ({len(timeout_trades)/len(detailed_results)*100:.1f}%)")
            timeout_pnl = [t['pnl_r'] for t in timeout_trades]
            timeout_wins = len([p for p in timeout_pnl if p > 0])
            print(f"  Timeout WR: {timeout_wins/len(timeout_pnl)*100:.1f}%")
            print(f"  Timeout Avg PnL R: {sum(timeout_pnl)/len(timeout_pnl):+.3f}")
            print(f"  Timeout Total PnL R: {sum(timeout_pnl):+.2f}")
            
            # CRITICAL: without timeouts, what's the real WR?  
            non_timeout = [r for r in detailed_results if r['exit_reason'] != 'TIMEOUT']
            if non_timeout:
                nt_wins = len([r for r in non_timeout if r['pnl_r'] > 0])
                nt_pnl = sum(r['pnl_r'] for r in non_timeout)
                print(f"\n  === WITHOUT TIMEOUTS ===")
                print(f"  Trades: {len(non_timeout)}")
                print(f"  WR: {nt_wins/len(non_timeout)*100:.1f}%")
                print(f"  Total PnL R: {nt_pnl:+.2f}")
                print(f"  Avg PnL R: {nt_pnl/len(non_timeout):+.3f}")


if __name__ == '__main__':
    main()
