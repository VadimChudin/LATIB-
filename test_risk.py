import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.risk_manager import RiskManager

def test_risk_manager():
    config = {
        'max_risk_per_trade_pct': 0.15,
        'base_risk_pct': 0.02,
        'use_kelly': True,
        'kelly_fraction': 0.5, # Half Kelly
        'max_leverage': 40.0
    }
    
    rm = RiskManager(config)
    
    print("=== Testing Risk Manager ===")
    
    # Test 1: High conviction trade
    # Win probability 88%, Average Win 1.17R
    # Full Kelly = 0.88 - ((1 - 0.88)/ 1.17) = 0.88 - 0.102 = 0.778
    # Half Kelly = 0.389 -> But capped at max_risk 0.15 (15%)
    res1 = rm.calculate_position_size(
        account_equity=50.0,
        entry_price=60000.0,
        sl_price=59900.0, # SL is $100 away
        ml_prob_win=0.88,
        avg_win_r=1.17
    )
    print("\nTest 1 (High Conviction, W=88%, R=1.17):")
    print(res1)
    
    # Test 2: Low conviction trade, high max risk cap
    config2 = config.copy()
    config2['max_risk_per_trade_pct'] = 0.50
    rm2 = RiskManager(config2)
    
    # Win Prob 60%, Avg Win 1R
    # Full Kelly: 0.60 - ((1 - 0.60)/1) = 0.60 - 0.40 = 0.20 (20%)
    # Half Kelly: 0.10 (10%)
    res2 = rm2.calculate_position_size(
        account_equity=1000.0,
        entry_price=65000.0,
        sl_price=64000.0, # SL is $1000 away
        ml_prob_win=0.60,
        avg_win_r=1.00
    )
    print("\nTest 2 (Low Conviction, W=60%, R=1.0):")
    print(res2)

if __name__ == '__main__':
    test_risk_manager()
