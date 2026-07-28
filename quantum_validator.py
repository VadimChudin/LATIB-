import numpy as np
import pandas as pd
from datetime import timedelta

def test_recency(strat, df, params, days=7):
    """
    Проверка прибыльности на самом свежем участке данных (последние 'days' дней).
    """
    if df is None or len(df) < 500:
        return True
        
    last_date = df.index[-1]
    start_date = last_date - timedelta(days=days)
    df_recent = df[df.index >= start_date].copy()
    
    if len(df_recent) < 100:
        return True
        
    try:
        res = strat.backtest_logic(df_recent, params)
        trades = res[res['trade_pnl_r'] != 0]
        
        if len(trades) == 0:
            return False # Слишком редкие входы для текущего рынка
            
        total_pnl = trades['trade_pnl_r'].sum()
        # Параметр должен быть хотя бы не убыточным на последней неделе
        is_passed = total_pnl >= 0
        
        print(f"      [Recency] Last {days}d PnL: {total_pnl:.2f}R ({len(trades)} trades) -> {'✅ Pass' if is_passed else '❌ Fail'}")
        return is_passed
    except Exception as e:
        print(f"      [Recency] Error: {e}")
        return False

def test_monte_carlo(strat, df, params, iterations=10, jitter=0.0005):
    """
    Стресс-тест Монте-Карло: добавление случайного шума в цены.
    Проверяет, не является ли прибыльность результатом 'удачного' попадания в конкретные тики.
    """
    if df is None or len(df) < 500:
        return True
        
    # Берем срез последних 2000 баров для скорости
    df_slice = df.tail(2000).copy()
    
    success_count = 0
    pnl_results = []
    
    for i in range(iterations):
        try:
            df_noise = df_slice.copy()
            # Добавляем шум +/- jitter % к ценам Close, High, Low
            noise = 1 + (np.random.uniform(-jitter, jitter, len(df_noise)))
            for col in ['open', 'high', 'low', 'close']:
                df_noise[col] = df_noise[col] * noise
            
            res = strat.backtest_logic(df_noise, params)
            trades = res[res['trade_pnl_r'] != 0]
            
            if len(trades) > 0:
                pnl = trades['trade_pnl_r'].sum()
                if pnl > 0:
                    success_count += 1
                pnl_results.append(pnl)
            else:
                pnl_results.append(0)
        except Exception:
            pnl_results.append(0)
            
    pass_rate = success_count / iterations
    is_passed = pass_rate >= 0.7 # 70% итераций должны быть прибыльными
    
    avg_pnl = sum(pnl_results) / len(pnl_results)
    print(f"      [Monte Carlo] Pass Rate: {pass_rate*100:.0f}% | Avg PnL: {avg_pnl:.2f}R -> {'✅ Stable' if is_passed else '❌ Fragile'}")
    return is_passed
