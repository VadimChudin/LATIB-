import pandas as pd
import pandas_ta as ta

df = pd.DataFrame({'close': range(100)})
# Test float
df.ta.bbands(length=20, std=2.5, append=True)
print("Float '2.5' columns:", [c for c in df.columns if 'BB' in c])

# Test int
df.ta.bbands(length=20, std=3.0, append=True)
print("Float '3.0' columns:", [c for c in df.columns if 'BB' in c])
