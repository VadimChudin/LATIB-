import glob
import re

for file in glob.glob("train_ml_*.py"):
    with open(file, "r", encoding="utf-8") as f:
        content = f.read()

    # Find the SYMBOLS = [...] definition
    pattern = r'SYMBOLS\s*=\s*\[.*?\]'
    replacement = 'from download_historical import TOP_ALTCOINS\nSYMBOLS = [s.replace("/", "_") for s in TOP_ALTCOINS[:50]]'
    
    # Replace it
    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    if new_content != content:
        with open(file, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"Patched {file}")
