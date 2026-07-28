"""
Trailing Stop Diagnostic — tests SwingICT KZ with trailing stop
and simulates compound growth toward $50 → $1000 target.

Run: python diagnose_trailing.py
"""
import os, sys, math
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def wr_needed_for_target(start, target, days, trades_per_day, avg_win_r, risk_pct=0.10):
    """
    Calculates minimum WR needed to compound start→target in `days` days.
    Assumes compound risk (risk_pct of current equity per trade).
    avg_win_r = average R captured on winning trades (trailing stop).
    """
    total_trades = trades_per_day * days
    target_mult  = target / start

    # Binary search for WR where expected compound outcome hits target
    lo, hi = 0.01, 0.99
    for _ in range(80):
        mid = (lo + hi) / 2
        # Per-trade multipliers
        win_mult  = 1 + risk_pct * avg_win_r
        loss_mult = 1 - risk_pct * 1.0
        # Expected value using geometric mean approximation
        per_trade = (win_mult ** mid) * (loss_mult ** (1 - mid))
        if per_trade ** total_trades < target_mult:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def simulate_compound(start, trades_per_day, days, win_rate, avg_win_r,
                      risk_pct=0.10, simulations=5000):
    """Monte Carlo compound growth simulation."""
    results = []
    busted  = 0

    for _ in range(simulations):
        equity = start
        for _ in range(trades_per_day * days):
            if np.random.rand() < win_rate:
                equity *= (1 + risk_pct * avg_win_r)
            else:
                equity *= (1 - risk_pct)
            if equity < start * 0.05:   # 95% drawdown = bust
                busted += 1
                equity = 0
                break
        results.append(equity)

    results = np.array(results)
    return {
        'p5':    np.percentile(results, 5),
        'p50':   np.percentile(results, 50),
        'p95':   np.percentile(results, 95),
        'bust':  busted / simulations,
        'reach': (results >= 1000).mean(),
    }


def main():
    # ── Load cache ──────────────────────────────────────────────────────────
    cache_file = os.path.join(os.path.dirname(__file__), 'data', 'cache', 'BTC_USDT_5m_730d.csv')
    if not os.path.exists(cache_file):
        print("Cache missing. Run diagnose.py first.")
        return

    print("Loading cache...")
    dtypes = {c: 'float32' for c in ['open', 'high', 'low', 'close', 'volume']}
    df = pd.read_csv(cache_file, dtype=dtypes, engine='c', low_memory=False)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.index = df['timestamp']
    print(f"Loaded {len(df):,} rows.\n")

    days = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).days or 1

    # ── Backtest: Fixed TP (baseline) ───────────────────────────────────────
    print("=== SwingICT KZ — Fixed TP 0.6R (baseline) ===")
    from strategies.swing_ict_kz import SwingICTKZStrategy
    strat = SwingICTKZStrategy()

    p_fixed = {'ema_fast': 13, 'ema_slow': 34, 'tp_rr': 0.6,
               'sl_atr_mult': 1.0, 'trailing_stop': False}
    r_fixed  = strat.backtest_logic(df.copy(), p_fixed)
    t_fixed  = r_fixed[r_fixed['trade_pnl_r'] != 0]['trade_pnl_r']
    wr_fixed = (t_fixed > 0).mean()
    avg_win_fixed = t_fixed[t_fixed > 0].mean()
    print(f"  Trades: {len(t_fixed)} ({len(t_fixed)/days:.1f}/day)")
    print(f"  WR:     {wr_fixed:.1%}")
    print(f"  Avg win R: {avg_win_fixed:.2f}R  |  Avg loss R: {t_fixed[t_fixed<0].mean():.2f}R")
    ev_fixed = wr_fixed * avg_win_fixed + (1 - wr_fixed) * t_fixed[t_fixed < 0].mean()
    print(f"  EV/trade: {ev_fixed:+.4f}R\n")

    # ── Backtest: Trailing Stop ─────────────────────────────────────────────
    print("=== SwingICT Trail — (activates at 1R, trails 0.5 ATR) ===")
    from strategies.swing_ict_trail import SwingICTTrailStrategy
    strat_trail = SwingICTTrailStrategy()
    p_trail = {'ema_fast': 13, 'ema_slow': 34, 'sl_atr_mult': 1.0,
               'trail_activate_r': 1.0, 'trail_atr_mult': 0.5}
    r_trail  = strat_trail.backtest_logic(df.copy(), p_trail)
    t_trail  = r_trail[r_trail['trade_pnl_r'] != 0]['trade_pnl_r']
    wr_trail = (t_trail > 0).mean()
    avg_win_trail = t_trail[t_trail > 0].mean()
    avg_loss_trail = t_trail[t_trail < 0].mean()
    ev_trail = wr_trail * avg_win_trail + (1 - wr_trail) * avg_loss_trail
    print(f"  Trades: {len(t_trail)} ({len(t_trail)/days:.1f}/day)")
    print(f"  WR:     {wr_trail:.1%}")
    print(f"  Avg win R: {avg_win_trail:.2f}R  |  Avg loss R: {avg_loss_trail:.2f}R")
    print(f"  EV/trade: {ev_trail:+.4f}R")
    print(f"  Max win R: {t_trail.max():.1f}R  |  Best 10% avg: {t_trail[t_trail>t_trail.quantile(0.9)].mean():.1f}R\n")

    # ── Required WR for target ──────────────────────────────────────────────
    print("=" * 60)
    print("  $50 → $1000 IN 7 DAYS — REQUIRED WIN RATE ANALYSIS")
    print("=" * 60)
    trades_per_day = int(len(t_trail) / days)
    print(f"\n  Settings: {trades_per_day} trades/day, SL=$5, risk=10%/trade")
    print(f"  Trailing avg win: {avg_win_trail:.2f}R\n")

    for avg_r in [avg_win_trail, 2.0, 2.5, 3.0]:
        req_wr = wr_needed_for_target(50, 1000, 7, trades_per_day, avg_r)
        ev     = req_wr * avg_r - (1 - req_wr) * 1.0
        print(f"  Avg win={avg_r:.1f}R → Need WR ≥ {req_wr:.1%}  (EV={ev:+.3f}R/trade)")

    # ── Monte Carlo for actual trailing WR ──────────────────────────────────
    print(f"\n=== Monte Carlo: actual trailing WR={wr_trail:.1%}, avg_win={avg_win_trail:.2f}R ===")
    print(f"    Risk 10%/trade, 7 days, {trades_per_day} trades/day, 5,000 simulations\n")
    mc = simulate_compound(50, trades_per_day, 7, wr_trail, avg_win_trail)
    print(f"  Worst  5%  outcome: ${mc['p5']:>8.2f}")
    print(f"  Median    outcome: ${mc['p50']:>8.2f}")
    print(f"  Best   95% outcome: ${mc['p95']:>8.2f}")
    print(f"  Probability of bust: {mc['bust']:.1%}")
    print(f"  Probability of reaching $1000: {mc['reach']:.1%}")
    print()

    # ── What WR we actually need ─────────────────────────────────────────────
    req_wr = wr_needed_for_target(50, 1000, 7, trades_per_day, avg_win_trail)
    gap    = req_wr - wr_trail
    print("=" * 60)
    print(f"  Actual WR:   {wr_trail:.1%}")
    print(f"  Required WR: {req_wr:.1%}")
    print(f"  Gap:         {gap:+.1%}  ({'ML Filter needed!' if gap > 0 else 'Already achievable!'})")
    print("=" * 60)


if __name__ == '__main__':
    main()
