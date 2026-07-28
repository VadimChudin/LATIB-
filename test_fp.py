import sys
sys.path.insert(0, r'd:\smart-zones-pro\python_core')

from footprint_data import get_collector

c = get_collector()
c.load_all()

print()
print('=== STATS ===')
for tf, cnt in c.get_stats().items():
    print(f'  {tf}: {cnt} candles')

print()
candles = c.get_footprint('4h')
if candles:
    last = candles[-1]
    print(f'Last 4H candle: {last.time_str}')
    print(f'  O={last.open} H={last.high} L={last.low} C={last.close}')
    print(f'  is_real={last.is_real}')
    print(f'  levels count: {len(last.levels)}')
    top3 = sorted(last.levels.items(), key=lambda x: x[1]["buy"]+x[1]["sell"], reverse=True)[:3]
    for p, d in top3:
        print(f'    ${p}: BUY={d["buy"]:.1f} SELL={d["sell"]:.1f}')
    
    # Test update
    print()
    print('=== TESTING UPDATE ===')
    buf = c.buffers['4h']
    result = buf.update()
    print(f'  update() returned: {result}')
    
    candles2 = c.get_footprint('4h')
    last2 = candles2[-1]
    print(f'  After update last candle: {last2.time_str}')
    print(f'    O={last2.open} H={last2.high} L={last2.low} C={last2.close}')
    print(f'    is_real={last2.is_real}')
    print(f'    levels count: {len(last2.levels)}')
else:
    print('NO 4H CANDLES')
