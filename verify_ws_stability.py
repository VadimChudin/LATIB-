import asyncio
import websockets
import json
import time

async def test_ui_bridge_stability():
    print("--- Aegis Terminal WebSocket Stability Test ---")
    url = "ws://localhost:8080"
    
    attempts = 0
    max_attempts = 5
    
    while attempts < max_attempts:
        try:
            async with websockets.connect(url, ping_interval=5, ping_timeout=2) as ws:
                print(f"[SUCCESS] Connected to Aegis UI Bridge on attempt {attempts+1}")
                
                # Test receiving initial state
                msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                data = json.loads(msg)
                print(f"[DATA] Received {data.get('type')}")
                
                # Stay connected for 10 seconds to monitor for silent drops
                print("[STABILITY] Monitoring connection for 10s...")
                await asyncio.sleep(10)
                print("[STABILITY] Connection stable.")
                return True
                
        except Exception as e:
            print(f"[FAIL] Connection attempt {attempts+1} failed: {e}")
            attempts += 1
            await asyncio.sleep(2)
            
    print("[ERROR] Could not establish stable connection to UI Bridge.")
    return False

if __name__ == "__main__":
    try:
        asyncio.run(test_ui_bridge_stability())
    except KeyboardInterrupt:
        pass
