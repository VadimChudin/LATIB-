import os
import sys
import pandas as pd
import optuna
import logging
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from strategies.swing_ict_trail import SwingICTTrailStrategy
from strategies.ultimate_smc_trail import UltimateSMCTrailStrategy
from strategies.knife_catcher_ml import KnifeCatcherMLStrategy

# Suppress optuna logs below WARNING for cleaner output
optuna.logging.set_verbosity(optuna.logging.WARNING)

def load_data(symbol="BTC_USDT"):
    cache_file = os.path.join(os.path.dirname(__file__), 'data', 'cache', f'{symbol}_5m_730d.csv')
    if not os.path.exists(cache_file):
        # Try fallback to any valid csv in cache
        cache_dir = os.path.join(os.path.dirname(__file__), 'data', 'cache')
        if not os.path.exists(cache_dir):
            return None
        files = [f for f in os.listdir(cache_dir) if f.endswith('.csv')]
        if not files:
            print(f"❌ No cache files found in {cache_dir}")
            return None
        cache_file = os.path.join(cache_dir, files[0])
        print(f"⚠️ {symbol} cache missing. Using fallback: {files[0]}")
        
    dtypes = {c: 'float32' for c in ['open', 'high', 'low', 'close', 'volume']}
    df = pd.read_csv(cache_file, dtype=dtypes, engine='c', low_memory=False)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    # Do not set index to timestamp because many strategies expect it as a column
    # and use integer indexing via .iloc
    return df.tail(50000) # ≈ 173 days of 5m data

GLOBAL_DF = None

def get_objective(strategy_name):
    def objective(trial):
        if GLOBAL_DF is None: return 0.0

        if strategy_name == "SwingICT_Trail":
            params = {
                'ema_fast': trial.suggest_int('ema_fast', 3, 25),
                'ema_slow': trial.suggest_int('ema_slow', 25, 150),
                'sl_atr_mult': trial.suggest_float('sl_atr_mult', 0.5, 2.5, step=0.1),
                'trail_activate_r': trial.suggest_float('trail_activate_r', 0.5, 3.0, step=0.25),
                'trail_atr_mult': trial.suggest_float('trail_atr_mult', 0.1, 1.5, step=0.1)
            }
            if params['ema_fast'] >= params['ema_slow']: raise optuna.TrialPruned()
            strat = SwingICTTrailStrategy()
            
        elif strategy_name == "Ultimate_SMC_Trail":
            params = {
                'swing_length': trial.suggest_int('swing_length', 3, 15),
                'fvg_min_atr': trial.suggest_float('fvg_min_atr', 0.1, 1.0, step=0.1),
                'ob_min_score': trial.suggest_int('ob_min_score', 1, 4),
                'sl_atr_mult': trial.suggest_float('sl_atr_mult', 0.5, 2.5, step=0.1),
                'trail_activate_r': trial.suggest_float('trail_activate_r', 0.5, 3.0, step=0.25),
                'trail_atr_mult': trial.suggest_float('trail_atr_mult', 0.1, 1.5, step=0.1)
            }
            strat = UltimateSMCTrailStrategy()
            
        elif strategy_name == "KnifeCatcher_ML":
            params = {
                'rsi_oversold': trial.suggest_int('rsi_oversold', 15, 35),
                'bb_std': trial.suggest_float('bb_std', 2.0, 4.5, step=0.25),
                'vol_spike_mult': trial.suggest_float('vol_spike_mult', 1.5, 5.0, step=0.25),
                'tp_rr': trial.suggest_float('tp_rr', 0.5, 3.0, step=0.25),
                'sl_atr_mult': trial.suggest_float('sl_atr_mult', 0.5, 2.5, step=0.25)
            }
            strat = KnifeCatcherMLStrategy()
        elif strategy_name == "ML_ORB":
            from strategies.ml_orb import MLORBStrategy
            params = {
                'opening_bars': trial.suggest_int('opening_bars', 2, 8),
                'volume_mult': trial.suggest_float('volume_mult', 1.1, 2.5, step=0.1),
                'tp_mult': trial.suggest_float('tp_mult', 1.5, 4.0, step=0.5),
                'ml_min_prob': trial.suggest_float('ml_min_prob', 0.55, 0.70, step=0.05)
            }
            strat = MLORBStrategy()
        else:
            return 0.0

        res_df = strat.backtest_logic(GLOBAL_DF.copy(), params)
        if 'trade_pnl_r' not in res_df.columns: return 0.0
            
        trades = res_df[res_df['trade_pnl_r'] != 0]['trade_pnl_r']
        num_trades = len(trades)
        
        # Prune if too few trades (avoiding overfitting on noise)
        min_trades = 20 if strategy_name == "KnifeCatcher_ML" else 40
        if num_trades < min_trades: return 0.0
            
        wins = len(trades[trades > 0])
        win_rate = wins / num_trades
        
        gross_profit = trades[trades > 0].sum()
        gross_loss = abs(trades[trades < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 5.0
            
        # Balanced score: WinRate weighted by ProfitFactor
        score = win_rate * profit_factor
        
        trial.set_user_attr("win_rate", win_rate)
        trial.set_user_attr("profit_factor", profit_factor)
        trial.set_user_attr("trades", num_trades)
        
        return score
    return objective

def main():
    parser = argparse.ArgumentParser(description='Aegis Bayesian Optimizer')
    parser.add_argument('--strat', type=str, default='SwingICT_Trail', choices=['SwingICT_Trail', 'Ultimate_SMC_Trail', 'KnifeCatcher_ML', 'ML_ORB'])
    parser.add_argument('--symbol', type=str, default='BTC_USDT')
    parser.add_argument('--trials', type=int, default=100)
    args = parser.parse_args()

    global GLOBAL_DF
    print(f"🔍 Loading market data for {args.symbol}...")
    GLOBAL_DF = load_data(args.symbol)
    
    if GLOBAL_DF is None: return

    print(f"\n🚀 --- Starting Bayesian Optimization for {args.strat} ({args.symbol}) --- 🚀")
    
    study = optuna.create_study(direction='maximize')
    
    try:
        study.optimize(get_objective(args.strat), n_trials=args.trials, n_jobs=-1, show_progress_bar=True)
    except KeyboardInterrupt:
        print("\nInterrupted. Showing best so far...")

    if not study.trials:
        print("No successful trials.")
        return

    print(f"\n--- 🏆 TOP {args.strat} PARAMETERS 🏆 ---")
    top_trials = sorted([t for t in study.trials if t.value is not None], key=lambda t: t.value, reverse=True)[:5]
    
    for i, t in enumerate(top_trials):
        wr = t.user_attrs.get('win_rate', 0)
        pf = t.user_attrs.get('profit_factor', 0)
        tr = t.user_attrs.get('trades', 0)
        print(f"#{i+1} | Score: {t.value:.4f} | WR: {wr:.1%} | PF: {pf:.2f} | Trades: {tr}")
        print(f"      Params: {t.params}")

if __name__ == '__main__':
    main()
