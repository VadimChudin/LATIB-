"""
Run tick verification on knife V7 (CVD Divergence) trades.
Reads best params from ga_best_params.json, finds trades on 1m candles,
then verifies winning trades with verify_ticks.py.
"""
import asyncio
import json
import numpy as np
import pandas as pd
from pathlib import Path
from verify_ticks import TickVerifier

CSV = Path("data/cache/BTC_USDT_1m_730d.csv")
PARAMS_FILE = Path("data/ga_best_params.json")


def load_and_compute(csv_path):
    df = pd.read_csv(csv_path)
    n = len(df)

    # ATR(14)
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    df['atr'] = pd.Series(tr).ewm(span=14, adjust=False).mean().values

    # RSI(14)
    delta_price = pd.Series(close).diff()
    gain = delta_price.where(delta_price > 0, 0).ewm(span=14, adjust=False).mean()
    loss = (-delta_price.where(delta_price < 0, 0)).ewm(span=14, adjust=False).mean()
    rs = gain / loss.replace(0, 1e-10)
    df['rsi'] = 100 - 100 / (1 + rs)

    # ADX(14)
    df['adx'] = 25.0  # simplified — use fixed value for now

    # Delta
    df['delta'] = 2 * df['taker_buy_volume'] - df['volume']

    # CVD
    df['cvd'] = df['delta'].cumsum()

    # Volume SMA
    df['vol_sma'] = df['volume'].rolling(20).mean()

    return df


def find_swing_lows(candles_low, start, end, min_count=4):
    swings = []
    for j in range(max(start, 2), min(end - 2, len(candles_low) - 3)):
        if (candles_low[j] <= candles_low[j-1] and candles_low[j] <= candles_low[j-2]
                and candles_low[j] <= candles_low[j+1] and candles_low[j] <= candles_low[j+2]):
            swings.append(j)
            if len(swings) >= min_count:
                break
    return swings


def find_swing_highs(candles_high, start, end, min_count=4):
    swings = []
    for j in range(max(start, 2), min(end - 2, len(candles_high) - 3)):
        if (candles_high[j] >= candles_high[j-1] and candles_high[j] >= candles_high[j-2]
                and candles_high[j] >= candles_high[j+1] and candles_high[j] >= candles_high[j+2]):
            swings.append(j)
            if len(swings) >= min_count:
                break
    return swings


def run_strategy(df, params):
    lookback = int(params.get('lookback', 50))
    cvd_window = int(params.get('cvd_window', 10))
    min_divergence = params.get('min_divergence', 0.3)
    vol_mult = params.get('vol_mult', 3.0)
    tp_rr = params.get('tp_rr', 1.6)
    sl_buffer_atr = params.get('sl_buffer_atr', 0.9)
    adx_max = params.get('adx_max', 25.0)
    rsi_filter = params.get('rsi_filter', 40.0)
    min_drop_atr = params.get('min_drop_atr', 3.5)
    confirm_green = int(params.get('confirm_green', 0))

    n = len(df)
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    opens = df['open'].values
    atr = df['atr'].values
    rsi = df['rsi'].values
    cvd = df['cvd'].values
    vol = df['volume'].values
    vol_sma = df['vol_sma'].values
    timestamps = df['timestamp'].values

    trades = []
    in_pos = False
    entry_price = sl_price = tp_price = 0.0
    direction = ""
    entry_ts = ""
    cooldown = 0

    start = max(300, lookback + cvd_window + 5)

    for i in range(start, n - 1):
        if in_pos:
            if direction == "LONG":
                if lows[i] <= sl_price:
                    trades.append({"entry_ts": entry_ts, "direction": "LONG", "entry_price": entry_price,
                                   "sl_price": sl_price, "tp_price": tp_price, "exit_price": sl_price, "pnl_r": -1.0})
                    in_pos = False; cooldown = 5
                elif highs[i] >= tp_price:
                    trades.append({"entry_ts": entry_ts, "direction": "LONG", "entry_price": entry_price,
                                   "sl_price": sl_price, "tp_price": tp_price, "exit_price": tp_price, "pnl_r": tp_rr})
                    in_pos = False
            else:
                if highs[i] >= sl_price:
                    trades.append({"entry_ts": entry_ts, "direction": "SHORT", "entry_price": entry_price,
                                   "sl_price": sl_price, "tp_price": tp_price, "exit_price": sl_price, "pnl_r": -1.0})
                    in_pos = False; cooldown = 5
                elif lows[i] <= tp_price:
                    trades.append({"entry_ts": entry_ts, "direction": "SHORT", "entry_price": entry_price,
                                   "sl_price": sl_price, "tp_price": tp_price, "exit_price": tp_price, "pnl_r": tp_rr})
                    in_pos = False
            continue

        if cooldown > 0:
            cooldown -= 1
            continue

        atr_v = atr[i]
        if atr_v <= 0 or pd.isna(vol_sma[i]) or vol_sma[i] <= 0:
            continue

        # === LONG: CVD Bullish Divergence ===
        if rsi[i] < rsi_filter:
            lb_start = max(0, i - lookback)
            # Find swing lows (most recent first)
            swing_idxs = []
            for j in range(i - 3, lb_start + 1, -1):
                if j >= 2 and j + 2 < i:
                    if (lows[j] <= lows[j-1] and lows[j] <= lows[j-2]
                            and lows[j] <= lows[j+1] and lows[j] <= lows[j+2]):
                        swing_idxs.append(j)
                        if len(swing_idxs) >= 4:
                            break

            if len(swing_idxs) >= 2:
                for k in range(len(swing_idxs) - 1):
                    recent_j = swing_idxs[k]
                    prev_j = swing_idxs[k + 1]
                    # Price: lower low
                    if lows[recent_j] >= lows[prev_j]:
                        continue
                    # CVD: higher low (divergence)
                    if cvd[recent_j] <= cvd[prev_j]:
                        continue
                    # Divergence strength
                    cvd_range = max(cvd[lb_start:i].max() - cvd[lb_start:i].min(), 1.0)
                    cvd_div = abs(cvd[recent_j] - cvd[prev_j]) / cvd_range
                    if cvd_div < min_divergence:
                        continue
                    # Drop check
                    recent_high = highs[lb_start:i].max()
                    drop = (recent_high - closes[i]) / atr_v
                    if drop < min_drop_atr:
                        continue
                    # Volume
                    recent_vol = vol[max(0,i-3):i+1].mean()
                    if recent_vol < vol_sma[i] * vol_mult:
                        continue
                    # Confirm green
                    if confirm_green and closes[i] <= opens[i]:
                        continue
                    # Time proximity
                    if i - recent_j > cvd_window * 3:
                        continue

                    # TRADE
                    swing_low = min(lows[recent_j], lows[i])
                    sl = swing_low - atr_v * sl_buffer_atr
                    risk = closes[i] - sl
                    if risk <= 0 or risk > atr_v * 5:
                        continue

                    entry_price = closes[i]
                    sl_price = sl
                    tp_price = closes[i] + risk * tp_rr
                    direction = "LONG"
                    entry_ts = str(timestamps[i])
                    in_pos = True
                    break

        # === SHORT: CVD Bearish Divergence ===
        if not in_pos and rsi[i] > (100 - rsi_filter):
            lb_start = max(0, i - lookback)
            swing_idxs = []
            for j in range(i - 3, lb_start + 1, -1):
                if j >= 2 and j + 2 < i:
                    if (highs[j] >= highs[j-1] and highs[j] >= highs[j-2]
                            and highs[j] >= highs[j+1] and highs[j] >= highs[j+2]):
                        swing_idxs.append(j)
                        if len(swing_idxs) >= 4:
                            break

            if len(swing_idxs) >= 2:
                for k in range(len(swing_idxs) - 1):
                    recent_j = swing_idxs[k]
                    prev_j = swing_idxs[k + 1]
                    if highs[recent_j] <= highs[prev_j]:
                        continue
                    if cvd[recent_j] >= cvd[prev_j]:
                        continue
                    cvd_range = max(cvd[lb_start:i].max() - cvd[lb_start:i].min(), 1.0)
                    cvd_div = abs(cvd[recent_j] - cvd[prev_j]) / cvd_range
                    if cvd_div < min_divergence:
                        continue
                    recent_low = lows[lb_start:i].min()
                    rise = (closes[i] - recent_low) / atr_v
                    if rise < min_drop_atr:
                        continue
                    recent_vol = vol[max(0,i-3):i+1].mean()
                    if recent_vol < vol_sma[i] * vol_mult:
                        continue
                    if confirm_green and closes[i] >= opens[i]:
                        continue
                    if i - recent_j > cvd_window * 3:
                        continue

                    swing_high = max(highs[recent_j], highs[i])
                    sl = swing_high + atr_v * sl_buffer_atr
                    risk = sl - closes[i]
                    if risk <= 0 or risk > atr_v * 5:
                        continue

                    entry_price = closes[i]
                    sl_price = sl
                    tp_price = closes[i] - risk * tp_rr
                    direction = "SHORT"
                    entry_ts = str(timestamps[i])
                    in_pos = True
                    break

    return trades


async def main():
    print("═══ Tick Verification — Knife V7 CVD Divergence ═══\n")

    # Load params
    with open(PARAMS_FILE) as f:
        data = json.load(f)
    params = data["best_params"]
    print(f"Best params: WR={data['win_rate']:.1f}%, trades={data['num_trades']}")
    print(f"Params: {json.dumps(params, indent=2)}\n")

    # Load candles & run strategy
    print("Loading 1m candles...")
    df = load_and_compute(CSV)
    print(f"Loaded {len(df)} candles")

    print("Running strategy with best params...")
    trades = run_strategy(df, params)
    wins = sum(1 for t in trades if t['pnl_r'] > 0)
    losses = sum(1 for t in trades if t['pnl_r'] < 0)
    wr = wins / len(trades) * 100 if trades else 0
    print(f"Found {len(trades)} trades: {wins}W/{losses}L (WR={wr:.1f}%)\n")

    if not trades:
        print("No trades to verify!")
        return

    # Verify with ticks (random sample of winners)
    print(f"Verifying winning trades with tick data...")
    verifier = TickVerifier()
    result = await verifier.verify_trades("BTC_USDT", trades, max_verify=30, random_sample=True)
    await verifier.close()

    print(f"\n═══ RESULTS ═══")
    print(f"   Original: {result['original_trades']} trades, {result['original_wins']} wins")
    print(f"   Verified: {result['verified_count']} winning trades checked on ticks")
    print(f"   Fake wins: {result['fake_wins']}")
    print(f"   Adjusted WR: {result['adjusted_wr']}%")
    print(f"   Confidence: {result['confidence']}")


if __name__ == "__main__":
    asyncio.run(main())
