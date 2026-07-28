"""
Statistical Arbitrage (Pairs Trading) Executor
==============================================
Standalone execution engine strictly for market-neutral pair trading.
Listens to websockets for paired assets (e.g., BTC and ETH), calculates the live
Z-score of their spread, and concurrently executes opposing legs (Long A / Short B).

Warning: Requires Binance Hedge Mode, properly funded USDS-M Futures account, and 
cross-margin or sufficient isolated margin on both assets.
"""
import os
import sys
import json
import ssl
import time
import asyncio
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import websockets
import ccxt.async_support as ccxt
from dotenv import load_dotenv

from core.db import CortexDB
from strategies.stat_arb import StatArbStrategy

load_dotenv(override=True)

# DROP ALL PROXY ENV VARS (websockets lib crashes on socks:// proxies)
import os
os.environ['NO_PROXY'] = '*'
os.environ['no_proxy'] = '*'
bad_vars = [
    'http_proxy', 'https_proxy', 'all_proxy', 
    'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY'
]
for pv in bad_vars:
    if pv in os.environ:
        del os.environ[pv]

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ArbExecutor")

PAPER_MODE = os.getenv('ARB_PAPER_MODE', os.getenv('PAPER_MODE', 'False')).lower() == 'true'
TIMEFRAME = '5m'
LOOKBACK_BARS = 100
TRADE_RISK_USD = float(os.getenv('TRADE_AMOUNT', '50.0')) # Dollar risk per leg per trade

def _create_ssl_context():
    ctx = ssl.create_default_context()
    ctx.check_hostname = True
    ctx.verify_mode = ssl.CERT_REQUIRED
    return ctx

async def broadcast_arb_log(msg: str, level: str = "info"):
    """Sends a log message specifically to the Aegis Terminal Arb UI."""
    payload = {
        "type": "arb_log",
        "message": msg,
        "level": level
    }
    try:
        async with websockets.connect("ws://127.0.0.1:8080") as ws:
            await ws.send(json.dumps(payload))
    except Exception:
        pass # UI might not be open, that's fine

class ArbExecutor:
    def __init__(self, pairs_file="data/arb_pairs.json"):
        self.pairs_file = pairs_file
        self.pairs = self._load_pairs()
        self.db = CortexDB()
        self.strategy = StatArbStrategy(lookback_bars=LOOKBACK_BARS, entry_z=2.0, exit_z=0.2, sl_z=4.0)
        
        # State Tracking
        self.exchange = None
        self.buffers = {} # {symbol: pd.DataFrame of historical klines}
        self.open_arb_positions = {} # {pair_tuple: "LONG_A_SHORT_B" / "SHORT_A_LONG_B"}
        self.last_eval = {} # Throttle evaluation internally
        
        # Track individual kline close for synchronization
        self.latest_close = {}
        
    def _load_pairs(self) -> list:
        if os.path.exists(self.pairs_file):
            try:
                with open(self.pairs_file, 'r') as f:
                    data = json.load(f)
                    return data.get('pairs', [])
            except Exception as e:
                logger.error(f"Failed to load arb pairs: {e}")
        return []

    async def _initialize_exchange(self):
        self.exchange = ccxt.binance({
            'apiKey': os.getenv('ARB_BINANCE_API_KEY', os.getenv('BINANCE_API_KEY', '')),
            'secret': os.getenv('ARB_BINANCE_SECRET', os.getenv('BINANCE_SECRET', '')),
            'enableRateLimit': True,
            'options': {'defaultType': 'future'},
        })
        if PAPER_MODE:
            self.exchange.set_sandbox_mode(True)
            logger.info("🏢 PAPER TRADING MODE ACTIVE for Arb Executor")
        await self.exchange.load_markets()

    async def _seed_buffers(self):
        """Pre-fetch history required for Z-score calculation."""
        flattened_symbols = list(set([sym for pair in self.pairs for sym in pair]))
        logger.info(f"Seeding historical buffers for {len(flattened_symbols)} unique symbols...")
        
        for sym in flattened_symbols:
            try:
                klines = await self.exchange.fetch_ohlcv(sym, TIMEFRAME, limit=LOOKBACK_BARS + 50)
                df = pd.DataFrame(klines, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.drop_duplicates(subset=["timestamp"], inplace=True)
                df.set_index('timestamp', inplace=False)
                self.buffers[sym] = df
            except Exception as e:
                logger.error(f"Failed to seed buffer for {sym}: {e}")
            await asyncio.sleep(0.5)

    async def evaluate_pair(self, sym_a: str, sym_b: str):
        """Evaluate the statistical arbitrage logic on a specific pair."""
        pair_key = (sym_a, sym_b)
        
        df_a = self.buffers.get(sym_a)
        df_b = self.buffers.get(sym_b)
        
        if df_a is None or df_b is None or len(df_a) < LOOKBACK_BARS or len(df_b) < LOOKBACK_BARS:
            return

        # Throttle evaluation to avoid duplicate trades on the same minute
        now = time.time()
        if now - self.last_eval.get(pair_key, 0) < 60:
            return
            
        # Run Strategy Logic
        try:
            results = self.strategy.generate_signals(df_a, df_b)
            latest = results.iloc[-1]
        except Exception as e:
            logger.error(f"Strategy eval failed for {pair_key}: {e}")
            return
            
        z_score = latest.get('z_score', 0)
        signal = latest.get('signal', 0)
        exit_sig = latest.get('exit_signal', False)
        stop_loss = latest.get('stop_loss', False)
        
        current_pos = self.open_arb_positions.get(pair_key)
        
        logger.info(f"⚖️ Pair [{sym_a} / {sym_b}] | Z-Score: {z_score:.2f} | Open Pos: {current_pos}")

        if current_pos:
            # We are holding a position on this pair; look for exits
            if exit_sig or stop_loss:
                reason = "TAKE_PROFIT" if exit_sig else "STOP_LOSS"
                msg = f"🚨 {reason} TRIGGERED for [{sym_a} / {sym_b}] at Z={z_score:.2f}"
                logger.warning(msg)
                await broadcast_arb_log(msg, "info")
                await self.execute_exit(sym_a, sym_b, current_pos)
                del self.open_arb_positions[pair_key]
                self.last_eval[pair_key] = now
        else:
            # Look for entries
            if signal == 1:
                # Spread too high -> Short A, Long B
                msg = f"🚀 ENTRY TRIGGERED: ARB SHORT {sym_a} & LONG {sym_b} (Z={z_score:.2f})"
                logger.warning(msg)
                await broadcast_arb_log(msg, "info")
                await self.execute_entry(sym_a, sym_b, "SHORT", "LONG")
                self.open_arb_positions[pair_key] = "SHORT_A_LONG_B"
                self.last_eval[pair_key] = now
            elif signal == -1:
                # Spread too low -> Long A, Short B
                msg = f"🚀 ENTRY TRIGGERED: ARB LONG {sym_a} & SHORT {sym_b} (Z={z_score:.2f})"
                logger.warning(msg)
                await broadcast_arb_log(msg, "info")
                await self.execute_entry(sym_a, sym_b, "LONG", "SHORT")
                self.open_arb_positions[pair_key] = "LONG_A_SHORT_B"
                self.last_eval[pair_key] = now

    async def execute_entry(self, sym_a, sym_b, side_a, side_b):
        """Executes the dual entry using parallel market orders."""
        if PAPER_MODE:
            logger.info(f"📄 PAPER: Executed {side_a} on {sym_a} and {side_b} on {sym_b} for ${TRADE_RISK_USD}")
            return
            
        try:
            # Calculate position sizes based on current price
            ticker_a = await self.exchange.fetch_ticker(sym_a)
            ticker_b = await self.exchange.fetch_ticker(sym_b)
            
            qty_a = TRADE_RISK_USD / ticker_a['last']
            qty_b = TRADE_RISK_USD / ticker_b['last']
            
            # Format according to exchange rules
            m_a = self.exchange.market(sym_a)
            m_b = self.exchange.market(sym_b)
            
            qty_a_fmt = self.exchange.amount_to_precision(sym_a, qty_a)
            qty_b_fmt = self.exchange.amount_to_precision(sym_b, qty_b)
            
            # Execute concurrently to minimize slippage latency between legs
            order_a = self.exchange.create_market_order(sym_a, side_a.lower(), qty_a_fmt)
            order_b = self.exchange.create_market_order(sym_b, side_b.lower(), qty_b_fmt)
            
            res_a, res_b = await asyncio.gather(order_a, order_b, return_exceptions=True)
            
            # Record in DB
            self.db.log_trade(sym_a, side_a.upper(), float(qty_a_fmt), ticker_a['last'], "stat_arb", "LIVE")
            self.db.log_trade(sym_b, side_b.upper(), float(qty_b_fmt), ticker_b['last'], "stat_arb", "LIVE")
            
            logger.info(f"✅ Executed Arb Pair: {res_a} | {res_b}")
            await broadcast_arb_log(f"✅ Successfully executed dual entry", "info")
            
        except Exception as e:
            msg = f"Failed to execute entry for {sym_a}/{sym_b}: {e}"
            logger.error(msg)
            await broadcast_arb_log(msg, "error")

    async def execute_exit(self, sym_a, sym_b, current_pos):
        """Crosses the spread to close both legs."""
        if current_pos == "SHORT_A_LONG_B":
            side_a, side_b = "BUY", "SELL"
        else:
            side_a, side_b = "SELL", "BUY"
            
        if PAPER_MODE:
            try:
                ticker_a = await self.exchange.fetch_ticker(sym_a)
                ticker_b = await self.exchange.fetch_ticker(sym_b)
                self.db.log_trade(sym_a, side_a.upper(), 0.0, ticker_a['last'], "stat_arb", "PAPER_CLOSE")
                self.db.log_trade(sym_b, side_b.upper(), 0.0, ticker_b['last'], "stat_arb", "PAPER_CLOSE")
                logger.info(f"📄 PAPER: Closed Arb Pair [{sym_a}/{sym_b}] at {ticker_a['last']} and {ticker_b['last']}")
            except Exception as e:
                logger.error(f"Paper close failed: {e}")
            return
            
        try:
            # 1. Fetch exact position sizes to close 100% of the active pair
            balance = await self.exchange.fetch_balance()
            positions = balance.get('info', {}).get('positions', [])
            
            # Map symbol names from CCXT 'BTC/USDT' back to Binance raw 'BTCUSDT'
            raw_a = sym_a.replace('/', '')
            raw_b = sym_b.replace('/', '')
            
            qty_a_str = next((p['positionAmt'] for p in positions if p['symbol'] == raw_a and float(p['positionAmt']) != 0), "0")
            qty_b_str = next((p['positionAmt'] for p in positions if p['symbol'] == raw_b and float(p['positionAmt']) != 0), "0")
            
            qty_a = abs(float(qty_a_str))
            qty_b = abs(float(qty_b_str))
            
            # 2. Execute Market Close (Cross the Spread Simultaneously)
            tasks = []
            if qty_a > 0:
                qty_a_fmt = self.exchange.amount_to_precision(sym_a, qty_a)
                tasks.append(self.exchange.create_market_order(sym_a, side_a.lower(), qty_a_fmt, params={'reduceOnly': True}))
            
            if qty_b > 0:
                qty_b_fmt = self.exchange.amount_to_precision(sym_b, qty_b)
                tasks.append(self.exchange.create_market_order(sym_b, side_b.lower(), qty_b_fmt, params={'reduceOnly': True}))
                
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Retrieve execution prices
                ticker_a = await self.exchange.fetch_ticker(sym_a)
                ticker_b = await self.exchange.fetch_ticker(sym_b)
                self.db.log_trade(sym_a, side_a.upper(), qty_a, ticker_a['last'], "stat_arb_close", "LIVE")
                self.db.log_trade(sym_b, side_b.upper(), qty_b, ticker_b['last'], "stat_arb_close", "LIVE")
                
                msg = f"✅ Closed Arb Pair: {sym_a}/{sym_b} (PROFIT COLLECTED)"
                logger.info(msg)
                await broadcast_arb_log(msg, "info")
            else:
                logger.warning(f"No active positions found to close for {sym_a}/{sym_b}")
                
        except Exception as e:
            msg = f"Failed to execute exit for {sym_a}/{sym_b}: {e}"
            logger.error(msg)
            await broadcast_arb_log(msg, "error")

    async def _update_buffer(self, symbol, kline):
        """Append a new closed kline to the DataFrame buffer."""
        df = self.buffers.get(symbol)
        if df is None:
            return
            
        new_row = pd.DataFrame([{
            'timestamp': pd.to_datetime(kline['t'], unit='ms'),
            'open': float(kline['o']),
            'high': float(kline['h']),
            'low': float(kline['l']),
            'close': float(kline['c']),
            'volume': float(kline['v'])
        }])
        
        df = pd.concat([df, new_row], ignore_index=True)
        # Keep buffer manageable
        if len(df) > LOOKBACK_BARS + 200:
            df = df.iloc[-LOOKBACK_BARS-50:].reset_index(drop=True)
            
        self.buffers[symbol] = df

    async def stream_binance(self):
        """Connects to Binance WS for all unique pair symbols."""
        flattened_symbols = list(set([sym for pair in self.pairs for sym in pair]))
        if not flattened_symbols:
            logger.error("No pairs found. Exiting Stream.")
            return

        streams = []
        for symbol in flattened_symbols:
            safe_sym = symbol.replace("/", "").lower()
            streams.append(f"{safe_sym}@kline_{TIMEFRAME}")
            
        stream_str = "/".join(streams)
        ws_url = f"wss://fstream.binance.com/stream?streams={stream_str}"
        
        ssl_ctx = _create_ssl_context()
        logger.info(f"Connecting to WS: {ws_url[:80]}...")
        
        while True:
            try:
                async with websockets.connect(ws_url, ssl=ssl_ctx) as ws:
                    logger.info("✅ WebSocket connected successfully.")
                    while True:
                        msg = await ws.recv()
                        data = json.loads(msg)
                        
                        if 'data' in data and 'k' in data['data']:
                            sym = data['data']['s']
                            # Reformat mapping to CCXT style (safely handle '1000SHIBUSDT')
                            if sym.endswith('USDT'):
                                formatted_sym = f"{sym[:-4]}/{sym[-4:]}"
                            else:
                                formatted_sym = f"{sym[:-4]}/{sym[-4:]}" # Fallback
                            kline = data['data']['k']
                            
                            # Only process closed klines to trigger pair evaluations
                            if kline.get('x'):
                                await self._update_buffer(formatted_sym, kline)
                                
                                # Check if this symbol completes both legs of any active pair
                                for pair in self.pairs:
                                    if formatted_sym in pair:
                                        await self.evaluate_pair(pair[0], pair[1])
                                        
            except Exception as e:
                logger.error(f"WebSocket Error: {e}. Reconnecting in 5s...")
                await asyncio.sleep(5)

    async def run(self):
        await broadcast_arb_log("Booting Arb Executor Engine...", "info")
        await self._initialize_exchange()
        await broadcast_arb_log("Exchange connected. Prepping buffers...", "info")
        await self._seed_buffers()
        logger.info("Initializing WebSocket Stream Loop...")
        await broadcast_arb_log(f"Listening to live streams for {len(self.pairs)} pairs...", "info")
        await self.stream_binance()


if __name__ == "__main__":
    try:
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        executor = ArbExecutor()
        asyncio.run(executor.run())
    except KeyboardInterrupt:
        print("\nShutdown Arb Executor")
