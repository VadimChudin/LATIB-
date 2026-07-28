"""Minimal test to see if webview opens at all."""
import sys
sys.path.insert(0, r'd:\smart-zones-pro\python_core')

print("[test] Step 1: imports OK")

print("[test] Step 2: loading footprint data...")
try:
    from footprint_data import get_collector
    c = get_collector()
    c.load_all()
    print(f"[test] Data loaded OK: {c.get_stats()}")
except Exception as e:
    print(f"[test] FAILED to load data: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("[test] Step 3: creating window...")
try:
    import webview
    from footprint_window import API, HTML
    api = API(c)
    
    window = webview.create_window(
        "TEST Footprint",
        html=HTML, js_api=api,
        width=800, height=600,
        background_color="#131722",
    )
    print("[test] Window created, starting webview...")
    webview.start(debug=True)
    print("[test] webview.start() returned (window closed)")
except Exception as e:
    print(f"[test] FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
