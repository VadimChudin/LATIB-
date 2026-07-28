#!/usr/bin/env python3
"""
Build Tiered Active Config (Dynamic)
======================================
Reads ALL per-symbol params from data/tick_params/,
auto-assigns tiers based on OOS test_pnl_r thresholds,
and builds active_config.json with tier-based leverage.

Tier 1 (30x): test_pnl_r >= 30R    → high confidence
Tier 2 (15x): test_pnl_r 5-30R     → good
Tier 3  (5x): test_pnl_r 0-5R      → marginal
Excluded:     test_pnl_r < 0        → not profitable

Usage: python build_tiered_config.py
"""
import json, os

PARAMS_DIR = "data/tick_params"
OUTPUT = "data/active_config.json"

# PnL thresholds for tier assignment
TIER_1_THRESHOLD = 30.0   # test_pnl_r >= 30 → Tier 1 (30x)
TIER_2_THRESHOLD = 5.0    # test_pnl_r >= 5  → Tier 2 (15x)
# Anything 0 < test_pnl_r < 5 → Tier 3 (5x)
# test_pnl_r <= 0 → Excluded

TIER_LEVERAGE = {1: 30, 2: 15, 3: 5}

# Minimum test trades to trust the results
MIN_TOTAL_TRADES = 5


def get_tier(test_pnl_r, test_trades, train_pnl_r=None, train_trades=0, test_wr=0):
    """Assign tier based on OOS PnL. Strict filters — only genuinely profitable."""
    total_trades = (train_trades or 0) + (test_trades or 0)
    if total_trades < MIN_TOTAL_TRADES:
        return 0  # Not enough data
    
    # Require profitable training
    if train_pnl_r is not None and train_pnl_r <= 0:
        return 0
    
    # Require meaningful positive OOS
    if test_pnl_r < 0.5:
        return 0  # Too marginal or negative
    
    # Require decent OOS win rate
    if test_wr < 55.0:
        return 0
    
    # Tier by combined PnL
    combined_pnl = (train_pnl_r or 0) + (test_pnl_r or 0)
    
    if combined_pnl >= TIER_1_THRESHOLD:
        return 1
    elif combined_pnl >= TIER_2_THRESHOLD:
        return 2
    elif combined_pnl > 0:
        return 3
    return 0


def main():
    configs = []
    excluded = []
    
    for f in sorted(os.listdir(PARAMS_DIR)):
        if not f.endswith(".json"):
            continue
        with open(os.path.join(PARAMS_DIR, f)) as fh:
            data = json.load(fh)
        
        sym = f.replace(".json", "")
        test_pnl = data.get("test_pnl_r", 0)
        train_pnl = data.get("train_pnl_r", 0)
        test_wr = data.get("test_wr", 0)
        total_trades = data.get("total_trades", 0) or (data.get("train_trades", 0) + data.get("test_trades", 0))
        
        tier = get_tier(test_pnl, total_trades, train_pnl_r=train_pnl, train_trades=total_trades, test_wr=test_wr)
        
        if tier == 0:
            excluded.append((sym, test_pnl, total_trades))
            continue
        
        leverage = TIER_LEVERAGE[tier]
        params = data.get("params", {})
        
        # ═══ SAFETY FLOORS: prevent GA from optimizing to instant-SL values ═══
        PARAM_FLOORS = {
            "sl_pct":         0.0015,   # min 0.15% SL distance
            "trail_pct":      0.0010,   # min 0.10% trailing stop
            "be_trigger_pct": 0.0010,   # min 0.10% breakeven trigger
        }
        for key, floor in PARAM_FLOORS.items():
            if key in params and params[key] < floor:
                old = params[key]
                params[key] = floor
                print(f"  ⚠️ {sym}: {key} {old:.6f} → clamped to {floor:.4f}")
        
        config = {
            "symbol": sym,
            "timeframe": "tick",
            "strategy": "knife_tick",
            "tier": tier,
            "leverage": leverage,
            "params": params,
            "metrics": {
                "win_rate": data.get("test_wr", 0),
                "total_trades": total_trades,
                "score": test_pnl,
                "train_wr": data.get("train_wr", 0),
                "train_pnl_r": data.get("train_pnl_r", 0),
            }
        }
        configs.append(config)
    
    # Sort: Tier 1 first, then by PnL desc within tier
    configs.sort(key=lambda c: (c["tier"], -c["metrics"]["score"]))
    
    with open(OUTPUT, "w") as f:
        json.dump(configs, f, indent=2)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"  TIERED ACTIVE CONFIG")
    print(f"{'='*60}")
    print(f"{'Symbol':<20} {'Tier':>4} {'Lev':>4}x {'TestWR':>7} {'TestPnL':>8} {'Trades':>6}")
    print("-" * 55)
    
    for c in configs:
        m = c["metrics"]
        print(f"  {c['symbol']:<18} T{c['tier']}   {c['leverage']:>3}x {m['win_rate']:>6.1f}% {m['score']:>+7.1f}R {m['total_trades']:>6}")
    
    t1 = [c for c in configs if c['tier'] == 1]
    t2 = [c for c in configs if c['tier'] == 2]
    t3 = [c for c in configs if c['tier'] == 3]
    
    print(f"\n  Tier 1 (30x): {len(t1)} symbols  PnL={sum(c['metrics']['score'] for c in t1):+.1f}R")
    print(f"  Tier 2 (15x): {len(t2)} symbols  PnL={sum(c['metrics']['score'] for c in t2):+.1f}R")
    print(f"  Tier 3  (5x): {len(t3)} symbols  PnL={sum(c['metrics']['score'] for c in t3):+.1f}R")
    print(f"  Total:        {len(configs)} symbols  PnL={sum(c['metrics']['score'] for c in configs):+.1f}R")
    
    if excluded:
        print(f"\n  ❌ Excluded ({len(excluded)}):")
        for sym, pnl, trades in sorted(excluded, key=lambda x: x[1]):
            print(f"     {sym:<18} pnl={pnl:+.1f}R  trades={trades}")
    
    print(f"\n  📁 Saved to {OUTPUT}")


if __name__ == "__main__":
    main()
