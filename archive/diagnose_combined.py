"""
Combined Portfolio Diagnostic
=============================
Simulates the compound growth of running BOTH ML-filtered strategies simultaneously.
"""
from diagnose_trailing import simulate_compound, wr_needed_for_target

def main():
    # Swing_ICT at ML 0.63
    t1_day = 4.2
    wr1 = 0.885
    awin1 = 1.16

    # Ultimate SMC at ML 0.63
    t2_day = 2.2
    wr2 = 0.885
    awin2 = 1.19
    
    # Combined metrics (Approximation assuming independent distributions)
    total_trades_day = t1_day + t2_day
    comb_wr = ((t1_day * wr1) + (t2_day * wr2)) / total_trades_day
    comb_awin = ((t1_day * wr1 * awin1) + (t2_day * wr2 * awin2)) / (total_trades_day * comb_wr)

    print("=" * 60)
    print("  COMBINED PORTFOLIO (SwingICT + UltimateSMC) at ML=0.63")
    print("=" * 60)
    print(f"  Total Trades / Day: {total_trades_day:.1f}")
    print(f"  Combined Win Rate:  {comb_wr:.1%}")
    print(f"  Combined Avg Win:   {comb_awin:.2f}R")
    
    # Check Required WR for $50 -> $1000 in 7 days
    req_wr = wr_needed_for_target(50, 1000, 7, int(total_trades_day), comb_awin, risk_pct=0.10)
    gap = req_wr - comb_wr
    print(f"\n  Required WR for $1000: {req_wr:.1%}")
    print(f"  Gap: {gap:+.1%} (Goal {'MET!' if gap <= 0 else 'Short'})")

    print(f"\n=== Combined Monte Carlo (5000 sims) ===")
    mc = simulate_compound(50, int(total_trades_day), 7, comb_wr, comb_awin, risk_pct=0.10)
    print(f"  Worst  5%  outcome: ${mc['p5']:>8.2f}")
    print(f"  Median     outcome: ${mc['p50']:>8.2f}")
    print(f"  Best   95% outcome: ${mc['p95']:>8.2f}")
    print(f"  Probability of bust: {mc['bust']:.1%}")
    print(f"  Probability of reaching $1000: {mc['reach']:>5.1%}")

if __name__ == '__main__':
    main()
