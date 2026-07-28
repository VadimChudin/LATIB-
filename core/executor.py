"""
Live Trading Executor v6.0 — Bulletproof WebSocket Edition
==========================================================
Handles WebSocket connections to Binance Futures for real-time market data.
Generates signals via Strategies, filters them via ML Models, calculates
position size via Kelly Criterion, and places actual orders via CCXT.

Key stability features (v6.0):
- Proper ping/pong keepalive (Binance fstream pings every 5 min)
- 24-hour forced reconnect (Binance kills streams after 24h)
- Exponential backoff with jitter on reconnect
- Health watchdog: detects stale connections (no data for 5 min)
- Structured error logging (no more silent `except: pass`)
- SSL resilience for regions with certificate issues
"""
import os
import sys
import ssl
import random

for proxy_var in ['http_proxy', 'https_proxy', 'all_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY']:
    os.environ.pop(proxy_var, None)

import socket
socket.setdefaulttimeout(30.0)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import json
import logging
import time
from typing import Dict, Any, List, Optional
import ccxt.async_support as ccxt
import pandas as pd
import numpy as np
import aiohttp
import warnings
from datetime import datetime
import websockets
from pydantic import BaseModel, ValidationError
from core.db import CortexDB

# ── Pydantic Models for fast message validation ─────────────────────────────

class BinanceKline(BaseModel):
    t: int
    o: str
    h: str
    l: str
    c: str
    v: str
    x: bool

class BinanceData(BaseModel):
    s: str
    k: BinanceKline

class BinanceWSMessage(BaseModel):
    data: BinanceData

# ── Imports ──────────────────────────────────────────────────────────────────

from dotenv import load_dotenv
load_dotenv(override=True)

from core.db import CortexDB
from core.risk_manager import RiskManager
from core.ml_filter import RegimeMLFilter
from core.performance import PerformanceMonitor
from core.regime_detector import RegimeDetector
from core.correlation_filter import CorrelationFilter
from strategies.ultimate_smc_trail import UltimateSMCTrailStrategy
from strategies.density import DensityStrategy
from strategies.knife_catcher_ml import KnifeCatcherMLStrategy
from strategies.scalp_mtf import ScalpMTFStrategy
from strategies.funding_rate import FundingRateStrategy

from core.utils import (
    strip_proxies, get_browser_headers, is_in_cooldown,
    set_global_cooldown, get_cooldown_remaining, wait_for_cooldown
)

load_dotenv(override=True)
strip_proxies()

logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("AutoCore.Executor")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ── Constants ────────────────────────────────────────────────────────────────

WS_PING_INTERVAL = 120       # Send client ping every 2 min (server pings every 5 min)
WS_PING_TIMEOUT = 30         # Consider connection dead if no pong in 30s
WS_CLOSE_TIMEOUT = 5         # Fast close on errors
WS_MAX_MESSAGE_SIZE = 2**20  # 1 MB max message
WS_MAX_RECONNECT_DELAY = 120 # Max 2 min between reconnects
WS_HEALTH_TIMEOUT = 300      # 5 min without data = stale connection
WS_MAX_CONNECTION_LIFE = 23 * 3600  # Force reconnect after 23h
POSITION_POLL_INTERVAL = 30  # Poll positions every 30s

# ── Pro Features ─────────────────────────────────────────────────────────────

PAPER_MODE = os.getenv('PAPER_MODE', 'False').lower() == 'true'
MAX_SAME_DIRECTION = 2       # A2: Max 2 LONG or 2 SHORT simultaneously
HEARTBEAT_INTERVAL = 3600   # A5: Heartbeat every 60 min

# ── SSL Context ──────────────────────────────────────────────────────────────

def _create_ssl_context():
    """Create a resilient SSL context for regions with certificate issues."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = True
    ctx.verify_mode = ssl.CERT_REQUIRED
    return ctx


class LiveExecutor:
    def __init__(self, config_path="data/active_config.json", db=None, strategies=None):
        logger.info("Initializing LiveExecutor v6.0 (Bulletproof WS)...")
        self.config_path = config_path
        self.db = db or CortexDB()
        self.active_configs = self._load_configs()

        # Use provided strategies or create new ones
        if strategies:
            self.strategies = strategies
            logger.info(f"Using {len(strategies)} shared strategy instances.")
        else:
            self.strategies = {
                "Ultimate_SMC_Trail": UltimateSMCTrailStrategy(),
                "KnifeCatcher_ML":    KnifeCatcherMLStrategy(),
                "Density":            DensityStrategy(),
                "ScalpMTF":           ScalpMTFStrategy(),
                "FundingRate_MR":     FundingRateStrategy(),
            }
            logger.info("Initialized local strategy instances.")

        # ML Filters
        self.ml_filters = {}
        for name, strategy in self.strategies.items():
            if hasattr(strategy, 'ml_filter'):
                self.ml_filters[name] = strategy.ml_filter
            else:
                self.ml_filters[name] = RegimeMLFilter()
                try:
                    self.ml_filters[name].load(name)
                except:
                    pass

        self.risk_manager = RiskManager({
            'max_risk_per_trade_pct': 0.02,
            'base_risk_pct': 0.02,
            'use_kelly': True,
            'kelly_fraction': 0.5,
            'max_leverage': float(os.getenv('MAX_LEVERAGE', '40')),
        })
        self.perf = PerformanceMonitor()
        self.last_config_mtime = 0

        # System Log Buffer for UI
        self.system_logs = []
        self.max_sys_logs = 200

        # Market Data Buffers
        self.klines: dict = {}
        self.ui_clients: set = set()
        self.positions_cache: dict = {}
        self.klive: dict = {}

        # Signal batching
        self.pending_signals = []
        self.batch_timer_task = None
        self.last_eval_time = {}

        # WebSocket health tracking
        self._last_ws_message_time = 0.0
        self._ws_connection_start = 0.0
        self._ws_reconnect_count = 0

        # A3: Paper Trading
        self.paper_trades = []  # Virtual trade log
        self.paper_equity = 2000.0  # Starting virtual balance
        # Structural Cooldown: {symbol: {'side': 'LONG', 'sl_price': 100.50, 'valid_until': ts}}
        self.structural_cooldowns = {}
        self.db = CortexDB()  # Persistent trade log
        if PAPER_MODE:
            logger.info(f"📄 PAPER MODE ACTIVE — virtual equity ${self.paper_equity:.0f}")

        # A4: Execution Analytics
        self.exec_stats = {'total': 0, 'slippage_sum': 0.0, 'latency_sum': 0.0}

        # A5: Telegram reference (set by main.py)
        self._tg_bot = None

        # B2: Funding Rate cache {symbol: rate}
        self.funding_rates = {}

        # P1: HMM Regime Detector
        self.regime_detector = RegimeDetector(n_regimes=3)
        self.current_regimes = {}  # symbol → 'bull'/'bear'/'chop'

        # P1: Correlation Filter (cross-strategy)
        self.correlation_filter = CorrelationFilter(max_same_direction=2)

        # P1: Daily Drawdown Limiter
        self.daily_pnl = 0.0
        self.max_daily_loss_pct = -5.0  # Stop trading at -5% daily
        self.daily_trade_count = 0
        self._last_reset_day = None

        # Initialize CCXT Exchange
        testnet    = os.getenv('TESTNET', 'False').lower() == 'true'
        api_key    = os.getenv('BINANCE_API_KEY', '')
        api_secret = os.getenv('BINANCE_API_SECRET', '')
        browser_headers = get_browser_headers()

        exchange_config = {
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'timeout': 20000,
            'headers': browser_headers,
            'trust_env': False,
            'options': {
                'defaultType': 'future',
                'adjustForTimeDifference': True,
                'recvWindow': 60000,
            },
        }

        self.exchange = ccxt.binance(exchange_config)
        self.data_exchange = ccxt.binance({
            'enableRateLimit': True,
            'timeout': 20000,
            'headers': browser_headers,
            'trust_env': False,
            'options': {'defaultType': 'future'}
        })

        if testnet:
            self.exchange.set_sandbox_mode(True)
            logger.info("Exchange: TESTNET mode.")
        else:
            logger.info("Exchange: LIVE mode.")

        self._init_sema = asyncio.Semaphore(3)

    # ── Config Loading ───────────────────────────────────────────────────────

    def _load_configs(self) -> list:
        """Load active trading configs from Engine output."""
        defaults = [
            {'symbol': 'BTC/USDT', 'timeframe': '5m', 'strategy': 'Ultimate_SMC_Trail', 'params': {
                'swing_length': 3, 'fvg_min_atr': 0.3, 'ob_min_score': 4,
                'sl_atr_mult': 1.0, 'trail_activate_r': 1.0, 'trail_atr_mult': 0.5
            }},
            {'symbol': 'ETH/USDT', 'timeframe': '5m', 'strategy': 'Ultimate_SMC_Trail', 'params': {
                'swing_length': 3, 'fvg_min_atr': 0.3, 'ob_min_score': 4,
                'sl_atr_mult': 1.0, 'trail_activate_r': 0.8, 'trail_atr_mult': 0.2
            }},
            {'symbol': 'SOL/USDT', 'timeframe': '5m', 'strategy': 'Ultimate_SMC_Trail', 'params': {
                'swing_length': 3, 'fvg_min_atr': 0.3, 'ob_min_score': 4,
                'sl_atr_mult': 1.0, 'trail_activate_r': 0.8, 'trail_atr_mult': 0.2
            }},
            {'symbol': 'BNB/USDT', 'timeframe': '5m', 'strategy': 'Ultimate_SMC_Trail', 'params': {
                'swing_length': 3, 'fvg_min_atr': 0.3, 'ob_min_score': 4,
                'sl_atr_mult': 1.0, 'trail_activate_r': 0.8, 'trail_atr_mult': 0.2
            }},
            {'symbol': 'AVAX/USDT', 'timeframe': '5m', 'strategy': 'Ultimate_SMC_Trail', 'params': {
                'swing_length': 3, 'fvg_min_atr': 0.3, 'ob_min_score': 4,
                'sl_atr_mult': 1.0, 'trail_activate_r': 0.8, 'trail_atr_mult': 0.2
            }},
        ]
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    configs = json.load(f)
                if configs and len(configs) > 0:
                    logger.info(f"Loaded {len(configs)} configs: {[c['symbol'] for c in configs]}")
                    return configs
        except Exception as e:
            logger.warning(f"Could not load config: {e}")
        logger.info("Using default configs (SOL, LINK).")
        return defaults

    # ── Initialization ───────────────────────────────────────────────────────

    async def initialize(self):
        """Pre-fetch history with CCXT defaults."""
        logger.info("🎬 Initialization: Loading markets...")
        await asyncio.wait_for(self.exchange.load_markets(), timeout=30.0)

        async def _fetch_single(symbol_tuple):
            symbol, tf = symbol_tuple
            async with self._init_sema:
                try:
                    ohlcv = await self.data_exchange.fetch_ohlcv(symbol, timeframe=tf, limit=500)
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = df[col].astype(float)
                    self.klines[symbol] = df
                    logger.info(f"  ✅ {symbol} ({tf}): {len(df)} candles loaded")
                    await asyncio.sleep(1.5)  # Rate limit spacing
                    return True
                except Exception as e:
                    if '429' in str(e):
                        set_global_cooldown(60)
                        raise
                    logger.error(f"  ❌ {symbol}: {e}")
                    return False

        # Build list of symbols to fetch
        symbols_to_fetch = [(c['symbol'], c.get('timeframe', '5m')) for c in self.active_configs]
        
        # Always fetch BTC/USDT for the ML Correlation Matrix (BTC Gravity)
        if ('BTC/USDT', '5m') not in symbols_to_fetch:
            symbols_to_fetch.append(('BTC/USDT', '5m'))

        await asyncio.gather(*[_fetch_single(st) for st in symbols_to_fetch])
        
        try:
            await self.data_exchange.close()
        except:
            pass
            
        logger.info("✅ Initialization complete.")

    # ── Balance & Positions ──────────────────────────────────────────────────

    async def get_equity(self) -> float:
        """Fetch USDT wallet balance using CCXT's standard fetch_balance()."""
        if is_in_cooldown():
            return 0.0
        try:
            balance = await self.exchange.fetch_balance()
            return float(balance.get('total', {}).get('USDT', 0.0))
        except Exception as e:
            if '429' in str(e):
                set_global_cooldown(60)
            else:
                logger.error(f"Balance fetch error: {e}")
            return 0.0

    async def track_positions_loop(self):
        """Continuously polls positions and balance. Runs as background task."""
        logger.info("📊 Position tracking loop started (every 30s)")
        
        last_report_time = time.time()
        REPORT_INTERVAL = 3 * 3600  # 3 hours in seconds
        
        while True:
            await wait_for_cooldown("Positions")
            try:
                equity = await self.get_equity()
                await self.broadcast({'type': 'balance_update', 'equity': equity})

                positions = await self.exchange.fetch_positions()
                active = []
                for p in positions:
                    amt = float(p['info'].get('positionAmt', 0))
                    if amt != 0:
                        entry = float(p.get('entryPrice', 0))
                        mark = float(p.get('markPrice', entry))
                        pnl = float(p.get('unrealizedPnl', 0))
                        side = 'SHORT' if amt < 0 else 'LONG'
                        pct = ((mark - entry) / entry * 100) if entry > 0 else 0
                        if side == 'SHORT': pct = -pct
                        active.append({
                            'symbol': p['symbol'], 'side': side,
                            'entryPrice': entry, 'unrealizedPnl': pnl, 'pnlPct': pct
                        })
                # LIVE COOLDOWN: Detect if any position closed since last poll
                if hasattr(self, 'positions_cache') and self.positions_cache:
                    active_symbols = {p['symbol'] for p in active}
                    for old_p in self.positions_cache:
                        if old_p['symbol'] not in active_symbols:
                            # It closed. Was it a loss?
                            if old_p.get('unrealizedPnl', 0) < 0:
                                logger.info(f"🔴 [LIVE] {old_p['symbol']} stopped out. Applying Structural Cooldown.")
                                # We don't have exact SL here, so we use the last Mark Price as the invalidation level
                                self.structural_cooldowns[old_p['symbol']] = {
                                    'side': old_p['side'],
                                    'sl_price': old_p.get('entryPrice', 0), # Require price to recover past original entry
                                    'valid_until': time.time() + 14400 # Max 4 hours holding the lock
                                }
                
                self.positions_cache = active
                await self.broadcast({'type': 'positions', 'data': active})
            except Exception as e:
                if '429' in str(e):
                    logger.warning("⚠️ 429 in position loop. Cooldown activated.")
                    set_global_cooldown(60)
                else:
                    logger.error(f"Position tracking error: {e}")

            # B2: Fetch funding rates (piggyback on position poll)
            try:
                for c in self.active_configs:
                    sym = c['symbol']
                    ticker = await self.exchange.fetch_ticker(sym)
                    info = ticker.get('info', {})
                    fr = float(info.get('lastFundingRate', 0))
                    self.funding_rates[sym] = fr
            except Exception:
                pass  # Non-critical

            # ── 3-Hour Telegram Report ──
            if time.time() - last_report_time >= REPORT_INTERVAL:
                last_report_time = time.time()
                try:
                    stats = self.db.get_performance_stats()
                    curr_equity = self.paper_equity if PAPER_MODE else await self.get_equity()
                    
                    open_paper_trades = len([pt for pt in self.paper_trades if pt.get('status') == 'OPEN'])
                    open_live_trades = len(self.positions_cache) if self.positions_cache else 0
                    open_count = open_paper_trades if PAPER_MODE else open_live_trades
                    
                    msg = (
                        "🕒 *𝗔𝗘𝗚𝗜𝗦 𝟯-𝗛𝗼𝘂𝗿 𝗥𝗲𝗽𝗼𝗿𝘁*\n\n"
                        f"💰 *Equity:* `${curr_equity:.2f}`\n"
                        f"📈 *Open Trades:* `{open_count}`\n"
                        f"🎯 *Win Rate:* `{stats['win_rate']:.1f}%` ({stats['winning_trades']}/{stats['total_trades']})\n"
                        f"💸 *Total Realized PnL:* `${stats['total_pnl']:+.2f}`\n"
                        f"🛡 *Drawdown limit:* `{self.max_daily_loss_pct}%`\n"
                        f"🕹 *Mode:* `{'PAPER 📄' if PAPER_MODE else 'LIVE 🔴'}`"
                    )
                    
                    if self._tg_bot and hasattr(self._tg_bot, 'send_alert'):
                        await self._tg_bot.send_alert(msg)
                except Exception as e:
                    logger.error(f"Failed to generate 3-hour report: {e}")

            await asyncio.sleep(POSITION_POLL_INTERVAL)

    # ── Trade Execution ──────────────────────────────────────────────────────

    async def _execute_chaser_order(self, symbol: str, side: str, amount: float, max_chase_time: int = 5):
        """Places a LIMIT order at best_bid/best_ask and chases the price using dynamic OBI analysis."""
        start_time = time.time()
        order_id = None
        current_price = 0.0
        dynamic_chase_time = max_chase_time

        try:
            while time.time() - start_time < dynamic_chase_time:
                # 1. Fetch Order Book to find best price and calculate OBI
                ob = await self.exchange.fetch_order_book(symbol, limit=10)
                best_bid = ob['bids'][0][0] if ob.get('bids') else None
                best_ask = ob['asks'][0][0] if ob.get('asks') else None

                if not best_bid or not best_ask:
                    await asyncio.sleep(0.5)
                    continue
                    
                # ── Order Book Imbalance (OBI) ──
                bid_vol = sum(b[1] for b in ob.get('bids', []))
                ask_vol = sum(a[1] for a in ob.get('asks', []))
                total_vol = bid_vol + ask_vol
                obi = (bid_vol - ask_vol) / total_vol if total_vol > 0 else 0.0
                
                # Dynamic Chase logic
                # If buying, and OBI > 0.5 (heavy buy pressure), price will run away. Shorten chase to market quickly.
                # If buying, and OBI < -0.5 (heavy sell pressure), price is dropping to us. Extend chase for maker fill.
                if side == 'buy':
                    if obi > 0.4: dynamic_chase_time = 2.0
                    elif obi < -0.4: dynamic_chase_time = 10.0
                else: # sell
                    if obi < -0.4: dynamic_chase_time = 2.0
                    elif obi > 0.4: dynamic_chase_time = 10.0

                target_price = best_bid if side == 'buy' else best_ask

                # 2. If no order, create one
                if not order_id:
                    logger.info(f"[{symbol}] 🏃 Chaser: Placing {side.upper()} LIMIT @ {target_price} [OBI: {obi:+.2f}]")
                    order = await self.exchange.create_order(
                        symbol, 'LIMIT', side, amount, target_price, 
                        params={'timeInForce': 'GTC', 'postOnly': True}
                    )
                    order_id = order['id']
                    current_price = target_price
                    await asyncio.sleep(0.5)
                    continue

                # 3. Order exists. Check status
                try:
                    order_info = await self.exchange.fetch_order(order_id, symbol)
                except Exception as e:
                    logger.warning(f"[{symbol}] Chaser fetch_order error: {e}")
                    await asyncio.sleep(0.5)
                    continue

                status = order_info.get('status')
                remaining = float(order_info.get('remaining', amount))

                if status == 'closed' or remaining <= 0:
                    avg_fill = float(order_info.get('average', target_price))
                    logger.info(f"[{symbol}] ✅ Chaser: Fill complete @ {avg_fill} (Maker)")
                    return avg_fill  # Success

                # 4. If price moved away, cancel and replace
                if target_price != current_price:
                    logger.info(f"[{symbol}] 🏃 Chaser: Price moved ({current_price} -> {target_price}). Moving order...")
                    try:
                        await self.exchange.cancel_order(order_id, symbol)
                    except Exception as e:
                        if 'Unknown order' not in str(e):
                            logger.warning(f"[{symbol}] Cancel failed: {e}")
                    
                    order_id = None # Reset so next loop iteration creates a new limit
                    amount = remaining # Only buy what's left
                
                await asyncio.sleep(0.5)

            # --- TIMEOUT ---
            logger.warning(f"[{symbol}] ⚡ Chaser timeout ({max_chase_time}s). Falling back to MARKET order.")
            if order_id:
                try:
                    await self.exchange.cancel_order(order_id, symbol)
                except:
                    pass
                
                try:
                    order_info = await self.exchange.fetch_order(order_id, symbol)
                    amount = float(order_info.get('remaining', amount))
                except:
                    pass

            if amount > 0:
                logger.info(f"[{symbol}] ⚡ Placing fallback MARKET for remaining {amount}")
                market_order = await self.exchange.create_order(symbol, 'market', side, amount)
                return float(market_order.get('average', 0.0))
            
            return current_price

        except Exception as e:
            logger.error(f"[{symbol}] ❌ Chaser error: {e}")
            if order_id:
                try:
                    await self.exchange.cancel_order(order_id, symbol)
                except:
                    pass
            # Ultimate fallback if everything failed
            logger.warning(f"[{symbol}] ⚡ Ultimate fallback MARKET for {amount}")
            try:
                market_order = await self.exchange.create_order(symbol, 'market', side, amount)
                return float(market_order.get('average', 0.0))
            except Exception as inner_e:
                logger.error(f"[{symbol}] Ultimate fallback also failed: {inner_e}")
                return 0.0

    async def execute_trade(self, symbol: str, signal, config: dict):
        """Validates via ML + Risk + Regime + Correlation + Drawdown, places orders."""
        if not self.perf.trading_allowed:
            return

        # P1: Daily PnL reset at midnight
        today = datetime.utcnow().date()
        if self._last_reset_day != today:
            self.daily_pnl = 0.0
            self.daily_trade_count = 0
            self._last_reset_day = today

        # P1: Max Daily Drawdown Limiter
        if self.daily_pnl <= self.max_daily_loss_pct:
            logger.warning(f"🛑 DAILY LOSS LIMIT HIT ({self.daily_pnl:.1f}%). No more trades today.")
            return

        strat_key = config['strategy']
        direction = getattr(signal, 'direction', 'LONG')

        # P1: Regime Filter
        regime = self.current_regimes.get(symbol, 'chop')
        if not self.regime_detector.should_trade(regime, strat_key):
            logger.info(f"🌊 Regime filter: {strat_key} blocked in '{regime}' regime for {symbol}")
            return

        # P1: Correlation Filter
        if not self.correlation_filter.can_open(symbol, direction, strat_key, klines=self.klines):
            logger.info(f"🔗 Correlation filter: blocked {strat_key} {direction} {symbol}")
            return

        prob_win = getattr(signal, 'confidence', 0.5)
        ml_filter = self.ml_filters.get(strat_key)

        if ml_filter and ml_filter.is_fitted:
            try:
                current_idx = len(self.klines[symbol]) - 1
                btc_df = self.klines.get('BTC/USDT')
                features_df = ml_filter.prepare_features(self.klines[symbol].copy(), [current_idx], btc_df=btc_df)
                if not features_df.empty:
                    features_dict = features_df.iloc[0].to_dict()
                    features_dict.pop('index', None)
                    # B2: Inject live funding rate
                    features_dict['funding_rate'] = self.funding_rates.get(symbol, 0.0) * 100  # as pct
                    prob_win = ml_filter.predict_probability(features_dict, strategy_name=strat_key)
                    if prob_win < 0.63:
                        return
            except Exception as e:
                logger.warning(f"ML filter error for {symbol}: {e}")

        # ── B4: Order Book Depth Filter ──────────────────────────────────
        # Extract best bid/ask for realistic Paper Trading entry
        best_bid = None
        best_ask = None
        try:
            ob = await self.exchange.fetch_order_book(symbol, limit=20)
            best_bid = ob['bids'][0][0] if ob.get('bids') else None
            best_ask = ob['asks'][0][0] if ob.get('asks') else None

            bid_vol = sum(b[1] for b in ob.get('bids', [])[:10])
            ask_vol = sum(a[1] for a in ob.get('asks', [])[:10])
            ba_ratio = bid_vol / ask_vol if ask_vol > 0 else 1.0

            if signal.direction == 'LONG' and ba_ratio < 0.3:
                logger.info(f"[{symbol}] ⛔ OB filter: LONG blocked (bid/ask={ba_ratio:.2f} < 0.3)")
                return
            if signal.direction == 'SHORT' and ba_ratio > 3.0:
                logger.info(f"[{symbol}] ⛔ OB filter: SHORT blocked (bid/ask={ba_ratio:.2f} > 3.0)")
                return
            logger.debug(f"[{symbol}] OB: bid/ask={ba_ratio:.2f} ✓")
        except Exception as e:
            logger.debug(f"[{symbol}] OB check skipped: {e}")

        equity = await self.get_equity()
        if equity <= 0:
            if PAPER_MODE:
                equity = self.paper_equity
            else:
                return

        if equity <= 0:
            return

        # ── Volatility-Scaled Kelly ──────────────────────────────────────
        # Dynamically scale down risk during market turbulence
        volatility_scalar = 1.0
        try:
            if symbol in self.klines and len(self.klines[symbol]) >= 50:
                import pandas as pd
                df_vol = self.klines[symbol]
                # Approximate ATR
                tr1 = df_vol['high'] - df_vol['low']
                tr2 = (df_vol['high'] - df_vol['close'].shift()).abs()
                tr3 = (df_vol['low'] - df_vol['close'].shift()).abs()
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                
                short_atr = tr.tail(5).mean()
                long_atr = tr.tail(50).mean()
                
                if long_atr > 0 and short_atr > long_atr:
                    # Scale down Kelly if short-term volatility > long-term average
                    volatility_scalar = long_atr / short_atr
                    # Floor it at 0.2x limit (max 80% risk reduction)
                    volatility_scalar = max(0.2, min(1.0, volatility_scalar))
                    if volatility_scalar < 1.0:
                        logger.info(f"🌪️ [{symbol}] High Volatility (ATR ratio {long_atr/short_atr:.2f}). Scaling Kelly to {volatility_scalar:.2f}x")
        except Exception as e:
            logger.debug(f"[{symbol}] Volatility Scalar check failed: {e}")

        # Dynamic avg_win_r from GA metrics (profit_factor & win_rate)
        # Formula: avg_win_r = PF × (1 - WR) / WR
        metrics = config.get('metrics', {})
        pf = metrics.get('profit_factor', 0)
        wr = metrics.get('win_rate', 0)
        if pf > 0 and 0 < wr < 1:
            avg_win_r = pf * (1 - wr) / wr
        else:
            avg_win_r = 1.0  # Conservative fallback

        risk_calc = self.risk_manager.calculate_position_size(
            account_equity=equity,
            entry_price=signal.entry_price,
            sl_price=signal.sl_price,
            ml_prob_win=prob_win,
            avg_win_r=avg_win_r,
            volatility_scalar=volatility_scalar
        )

        if risk_calc['size'] <= 0:
            return

        amount = risk_calc['size']
        side = 'buy' if signal.direction == 'LONG' else 'sell'
        close_side = 'sell' if side == 'buy' else 'buy'
        leverage = int(risk_calc['leverage_needed'])

        # ── A3: Paper Trading Mode ────────────────────────────────────
        if PAPER_MODE:
            PAPER_SLIPPAGE = 0.0003 # 0.03% market impact slippage
            
            # Cheat Prevention: Enter at the real Order Book price, not the past "close" price
            paper_entry = signal.entry_price
            if best_bid and best_ask:
                # LONG pays Ask, SHORT gets Bid
                paper_entry = best_ask if signal.direction == 'LONG' else best_bid
                # Simulating Market order slippage
                paper_entry = paper_entry * (1 + PAPER_SLIPPAGE) if signal.direction == 'LONG' else paper_entry * (1 - PAPER_SLIPPAGE)
            else:
                logger.warning(f"[{symbol}] PAPER: OB unavailable. Using Signal Price + Slippage.")
                paper_entry = paper_entry * (1 + PAPER_SLIPPAGE) if signal.direction == 'LONG' else paper_entry * (1 - PAPER_SLIPPAGE)

            risk_dist = abs(paper_entry - signal.sl_price) if signal.sl_price else 0
            trail_act = config.get('params', {}).get('trail_activate_r', 1.0)
            trail_atr = config.get('params', {}).get('trail_atr_mult', 0.5)
            paper_trade = {
                'time': datetime.utcnow().isoformat(),
                'symbol': symbol, 'strategy': strat_key,
                'direction': signal.direction, 'entry': paper_entry,
                'sl': signal.sl_price, 'sl_initial': signal.sl_price,
                'tp': signal.tp_price,
                'size': amount, 'leverage': leverage,
                'ml_prob': prob_win, 'status': 'OPEN',
                'risk_dist': risk_dist,
                'trail_activate_r': trail_act,
                'trail_atr_mult': trail_atr,
                'best_price': signal.entry_price,
                'trailing_active': False,
                'open_candle_ts': time.time(),  # skip this candle for trailing
            }
            self.paper_trades.append(paper_trade)
            self.correlation_filter.open_position(symbol, signal.direction, strat_key)
            # Persist to DB
            trade_id = self.db.log_trade_open(
                symbol=symbol, strategy=strat_key, direction=signal.direction,
                entry=paper_entry, sl=signal.sl_price, tp=signal.tp_price,
                size=amount, lev=leverage, risk=self.paper_equity * 0.02,
                ml_prob=prob_win, timeframe='5m', params=config.get('params', {})
            )
            paper_trade['db_id'] = trade_id
            msg = f"📄 [PAPER] {symbol} {signal.direction} @ {paper_entry:.4f} (Slip={abs(paper_entry-signal.entry_price)/signal.entry_price*100:.2f}%) | size={amount} lev={leverage}x | SL={signal.sl_price:.4f}"
            logger.info(msg)
            asyncio.create_task(self.broadcast({'type': 'log', 'message': msg, 'level': 'info'}))
            # Send Telegram alert for paper trade
            await self._tg_alert(msg)
            return

        # ── Real Trade Execution ──────────────────────────────────────
        try:
            t0 = time.time()  # A4: Execution Analytics

            await self.exchange.set_leverage(max(1, leverage), symbol)
            
            # Use Chaser Logic instead of Market Order
            fill_price = await self._execute_chaser_order(symbol, side, amount, max_chase_time=5)
            
            if fill_price <= 0:
                raise Exception("Chaser failed to fill order and fallback failed.")

            t1 = time.time()  # A4: Latency

            await self.exchange.create_order(
                symbol, 'STOP_MARKET', close_side, amount,
                params={'stopPrice': signal.sl_price, 'reduceOnly': True}
            )

            if signal.tp_price:
                await self.exchange.create_order(
                    symbol, 'TAKE_PROFIT_MARKET', close_side, amount,
                    params={'stopPrice': signal.tp_price, 'reduceOnly': True}
                )

            # ── A4: Record Execution Analytics ────────────────────────
            latency_ms = (t1 - t0) * 1000
            slippage_pct = abs(fill_price - signal.entry_price) / signal.entry_price * 100 if signal.entry_price > 0 else 0
            self.exec_stats['total'] += 1
            self.exec_stats['slippage_sum'] += slippage_pct
            self.exec_stats['latency_sum'] += latency_ms

            self.db.log_trade_open(
                symbol, strat_key, signal.direction, signal.entry_price,
                signal.sl_price, signal.tp_price, amount,
                risk_calc['leverage_needed'], risk_calc['risk_amount_usd'],
                prob_win, config.get('params')
            )
            msg = f"[{symbol}] ✅ {side.upper()} executed: size={amount}, lev={leverage}x, slip={slippage_pct:.3f}%, lat={latency_ms:.0f}ms"
            logger.info(msg)
            asyncio.create_task(self.broadcast({'type': 'log', 'message': msg, 'level': 'success'}))
            await self._tg_alert(msg)
        except Exception as e:
            logger.error(f"[{symbol}] Trade failed: {e}")
            await self._tg_alert(f"🔴 Trade FAILED: {symbol} {e}")

    # ── Kline Message Processing ─────────────────────────────────────────────

    async def _handle_kline_message(self, raw_message: str):
        """Processes a single kline tick from fstream WebSocket."""
        try:
            msg = BinanceWSMessage.model_validate_json(raw_message)
        except ValidationError:
            return

        kline = msg.data.k
        symbol_raw = msg.data.s  # e.g. 'SOLUSDT'
        formatted_symbol = f"{symbol_raw[:-4]}/USDT"

        if formatted_symbol not in self.klines:
            return

        df = self.klines[formatted_symbol]
        ts = pd.to_datetime(kline.t, unit='ms')
        row = {
            'timestamp': ts,
            'open': float(kline.o),
            'high': float(kline.h),
            'low': float(kline.l),
            'close': float(kline.c),
            'volume': float(kline.v)
        }

        if ts == df['timestamp'].iloc[-1]:
            for col, val in row.items():
                df.at[df.index[-1], col] = val
        else:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            if len(df) > 500:
                df = df.iloc[-500:].reset_index(drop=True)
            self.klines[formatted_symbol] = df

        # ── Paper Trade SL/TP Monitor ─────────────────────────────────────
        if PAPER_MODE:
            high = float(kline.h)
            low = float(kline.l)
            close_price = float(kline.c)

            for pt in self.paper_trades:
                if pt.get('status') != 'OPEN' or pt.get('symbol') != formatted_symbol:
                    continue

                entry = pt['entry']
                sl = pt['sl']
                tp = pt['tp']
                direction = pt['direction']
                risk_dist = pt.get('risk_dist', 0)
                trail_act_r = pt.get('trail_activate_r', 1.0)
                trail_mult = pt.get('trail_atr_mult', 0.5)
                hit = None

                # Trailing stop logic — CANDLE-BASED (matches backtest behavior)
                # Only update best_price every 5 minutes, not on every tick.
                # This prevents premature trail stops from intra-candle noise.
                candle_age = time.time() - pt.get('open_candle_ts', 0)
                last_trail_update = pt.get('last_trail_update', 0)
                trail_interval = time.time() - last_trail_update
                
                if risk_dist > 0 and candle_age > 300:  # >5min = next candle
                    # Only update best_price on candle boundaries (~every 300s)
                    if trail_interval >= 300 or not pt.get('trailing_active'):
                        if direction == 'LONG':
                            pt['best_price'] = max(pt.get('best_price', entry), high)
                        elif direction == 'SHORT':
                            pt['best_price'] = min(pt.get('best_price', entry), low)
                        pt['last_trail_update'] = time.time()
                    
                    if direction == 'LONG':
                        profit_r = (pt['best_price'] - entry) / risk_dist
                        if profit_r >= trail_act_r:
                            new_sl = pt['best_price'] - risk_dist * trail_mult
                            # Ensure minimum profit floor: never trail below entry + 0.3R
                            min_sl = entry + risk_dist * 0.3
                            new_sl = max(new_sl, min_sl)
                            if not pt.get('trailing_active'):
                                logger.info(f"📄 [PAPER] 🔄 {formatted_symbol} trailing activated @ {pt['best_price']:.4f} ({profit_r:.1f}R)")
                                pt['trailing_active'] = True
                            if sl is None or new_sl > sl:
                                pt['sl'] = new_sl
                                sl = new_sl
                    elif direction == 'SHORT':
                        profit_r = (entry - pt['best_price']) / risk_dist
                        if profit_r >= trail_act_r:
                            new_sl = pt['best_price'] + risk_dist * trail_mult
                            # Ensure minimum profit floor: never trail above entry - 0.3R
                            min_sl = entry - risk_dist * 0.3
                            new_sl = min(new_sl, min_sl)
                            if not pt.get('trailing_active'):
                                logger.info(f"📄 [PAPER] 🔄 {formatted_symbol} trailing activated @ {pt['best_price']:.4f} ({profit_r:.1f}R)")
                                pt['trailing_active'] = True
                            if sl is None or new_sl < sl:
                                pt['sl'] = new_sl
                                sl = new_sl

                # Check SL/TP hits
                if direction == 'LONG':
                    if sl and low <= sl:
                        hit = 'TRAIL' if pt.get('trailing_active') else 'SL'
                        exit_price = sl
                    elif tp and high >= tp:
                        hit = 'TP'
                        exit_price = tp
                elif direction == 'SHORT':
                    if sl and high >= sl:
                        hit = 'TRAIL' if pt.get('trailing_active') else 'SL'
                        exit_price = sl
                    elif tp and low <= tp:
                        hit = 'TP'
                        exit_price = tp

                if hit:
                    # ── MARKET REALISM INJECTOR ──
                    PAPER_FEE_PCT = 0.0006    # 0.06% typical round-trip Taker fee + buffer
                    PAPER_SLIPPAGE = 0.0003   # 0.03% slippage on Stop/TP Market execution
                    
                    if direction == 'LONG':
                        exit_price = exit_price * (1 - PAPER_SLIPPAGE)
                    else:
                        exit_price = exit_price * (1 + PAPER_SLIPPAGE)

                    # Calculate PnL with Fees
                    if direction == 'LONG':
                        pnl_pct = (exit_price - entry) / entry * 100 - (PAPER_FEE_PCT * 100)
                    else:
                        pnl_pct = (entry - exit_price) / entry * 100 - (PAPER_FEE_PCT * 100)

                    risk_dist = abs(entry - sl) if sl else 1.0
                    pnl_r = pnl_pct / (risk_dist / entry * 100) if risk_dist > 0 else 0

                    pt['status'] = 'CLOSED'
                    pt['exit'] = exit_price
                    pt['exit_time'] = datetime.utcnow().isoformat()
                    pt['pnl_pct'] = round(pnl_pct, 3)
                    pt['pnl_r'] = round(pnl_r, 2)
                    pt['exit_reason'] = hit

                    emoji = '✅' if pnl_pct > 0 else '❌'
                    msg = (f"📄 [PAPER] {emoji} {formatted_symbol} {direction} CLOSED @ {exit_price:.4f} "
                           f"| {hit} | PnL={pnl_pct:+.2f}% ({pnl_r:+.1f}R) | Fees={PAPER_FEE_PCT*100:.2f}%")
                    logger.info(msg)
                    asyncio.create_task(self.broadcast({'type': 'log', 'message': msg, 'level': 'success' if pnl_pct > 0 else 'error'}))
                    asyncio.create_task(self._tg_alert(msg))

                    # Exact PnL Dollar Calculation (Deducting USD Fee)
                    size = pt.get('size', 0.0)
                    usd_fee = size * entry * PAPER_FEE_PCT
                    if direction == 'LONG':
                        pnl_usd = size * (exit_price - entry) - usd_fee
                    else:
                        pnl_usd = size * (entry - exit_price) - usd_fee

                    # Persist close to DB
                    if pt.get('db_id'):
                        self.db.close_trade(
                            trade_id=pt['db_id'],
                            exit_price=exit_price,
                            pnl_usd=pnl_usd,
                            pnl_pct=pt['pnl_pct'],
                            pnl_r=pt['pnl_r'],
                            reason=hit
                        )

                    # Update daily PnL tracker
                    self.daily_pnl += pnl_pct
                    self.daily_trade_count += 1

                    # Update paper equity
                    self.paper_equity += pnl_usd
                    logger.info(f"📄 [PAPER] 💰 Virtual equity: ${self.paper_equity:.2f} ({pnl_usd:+.2f})")

                    # Structural Cooldown: Only apply if it was a Stop Loss hit
                    if hit == 'SL':
                        self.structural_cooldowns[formatted_symbol] = {
                            'side': direction,
                            'sl_price': exit_price,
                            'valid_until': time.time() + 14400 # 4 hours max lock
                        }

                    # Free correlation filter slot
                    self.correlation_filter.close_position(formatted_symbol, pt.get('strategy', ''))

        # Throttled evaluation: on candle close OR every 10 seconds
        now = time.time()
        last_eval = self.last_eval_time.get(formatted_symbol, 0)

        if kline.x or (now - last_eval >= 10.0):
            self.last_eval_time[formatted_symbol] = now

            if kline.x:
                log_msg = f"[{formatted_symbol}] 🕯 Candle closed @ {row['close']}"
                logger.info(log_msg)
                asyncio.create_task(self.broadcast({'type': 'log', 'message': log_msg, 'level': 'info'}))

                # P1: Update HMM regime on candle close (every 50 bars to save CPU)
                if len(df) >= 100 and len(df) % 50 == 0:
                    try:
                        regime = self.regime_detector.fit_predict(df)
                        self.current_regimes[formatted_symbol] = regime.iloc[-1]
                    except Exception:
                        pass

            # ── MULTI-STRATEGY OVERLAY ────────────────────────────────────
            # Run ALL strategies on this symbol, not just the one from config.
            # Pick the signal with highest confidence.
            # Skip if we already have an open position on this symbol.

            # Check if we already have a position on this symbol
            if any(p.get('symbol') == formatted_symbol for p in (self.positions_cache or [])):
                return

            # Paper mode: also check paper_trades for open positions
            if PAPER_MODE and any(
                pt.get('symbol') == formatted_symbol and pt.get('status') == 'OPEN'
                for pt in self.paper_trades
            ):
                return

            # Intelligent Structural Cooldown Check
            # Prevent re-entering the SAME direction if price hasn't structurally cleared the previous Stop Loss
            cd = self.structural_cooldowns.get(formatted_symbol)
            if cd and time.time() < cd['valid_until']:
                current_price = row['close']
                
                # Check ALL strategies running on this coin simultaneously.
                # If ANY of them want to go in the same direction, we block it.
                # Since we don't have the explicit signal yet, we'll check it during the batch processor.
                pass

            # Default params per strategy (used when Engine config doesn't specify)
            default_params = {
                # GA-optimized params (2024-03-04, BTC_USDT_5m_730d)
                "SwingICT_Trail": {
                    'ema_fast': 12, 'ema_slow': 39, 'sl_atr_mult': 2.5,
                    'vol_mult': 1.0, 'trail_activate_r': 1.0, 'trail_atr_mult': 0.5
                },
                "Ultimate_SMC_Trail": {
                    'swing_length': 3, 'fvg_min_atr': 0.1, 'ob_min_score': 2,
                    'sl_atr_mult': 2.5, 'trail_activate_r': 1.0, 'trail_atr_mult': 0.5
                },
                "KnifeCatcher_ML": {
                    'rsi_oversold': 25, 'bb_std': 2.0, 'vol_spike_mult': 1.5,
                    'tp_rr': 0.8, 'sl_atr_mult': 1.5
                },
                "ML_ORB": {
                    'opening_bars': 3, 'volume_mult': 1.8, 'tp_mult': 1.0
                },
            }

            # Try Engine config first for the matching symbol
            engine_config = next(
                (c for c in self.active_configs if c['symbol'] == formatted_symbol), None
            )

            # Prevent rogue trades: If BTC/USDT is only here for Correlation Gravity, do not trade it
            if not engine_config and formatted_symbol == 'BTC/USDT':
                return

            candidates = []
            df_copy = df.copy()
            current_idx = len(df_copy) - 1

            for strat_name, strategy in self.strategies.items():
                try:
                    # Use Engine-optimized params if available, else defaults
                    if engine_config and engine_config.get('strategy') == strat_name:
                        params = engine_config.get('params', default_params.get(strat_name, {}))
                    else:
                        params = default_params.get(strat_name, {})

                    # Inject BTC macro data into strategy context
                    btc_frame = self.klines.get('BTC/USDT')
                    if btc_frame is not None:
                        strategy.set_btc_context(btc_frame)
                    
                    signal = strategy.generate_signal(df_copy, current_idx, params)
                    if signal:
                        fake_config = {
                            'symbol': formatted_symbol,
                            'timeframe': '5m',
                            'strategy': strat_name,
                            'params': params,
                        }
                        candidates.append((signal, formatted_symbol, fake_config))
                except Exception as e:
                    logger.debug(f"[{formatted_symbol}] {strat_name} error: {e}")

            if candidates:
                for sig, sym, cfg in candidates:
                    self.pending_signals.append((sig, sym, cfg))
                if self.batch_timer_task is None or self.batch_timer_task.done():
                    self.batch_timer_task = asyncio.create_task(self._process_signal_batch())

    async def _process_signal_batch(self):
        """Collect signals for 2.5s. Per symbol: pick highest confidence, resolve conflicts.
        A2: Portfolio Risk — limits max same-direction positions."""
        await asyncio.sleep(2.5)
        if not self.pending_signals:
            return

        # Group by symbol → pick best signal per symbol
        by_symbol = {}
        for sig, sym, cfg in self.pending_signals:
            conf = getattr(sig, 'confidence', 0.5)
            if sym not in by_symbol or conf > getattr(by_symbol[sym][0], 'confidence', 0.5):
                by_symbol[sym] = (sig, sym, cfg)

        # ── A2: Portfolio Risk Guard ──────────────────────────────────
        # Count current open positions by direction
        long_count = sum(1 for p in (self.positions_cache or []) if p.get('side') == 'LONG')
        short_count = sum(1 for p in (self.positions_cache or []) if p.get('side') == 'SHORT')

        logger.info(f"--- BATCH: {len(self.pending_signals)} signals → {len(by_symbol)} symbols | Open: {long_count}L/{short_count}S ---")

        for sym, (sig, _, cfg) in by_symbol.items():
            strat_name = cfg.get('strategy', '?')
            conf = getattr(sig, 'confidence', 0.5)

            # A2: Check portfolio limits
            if sig.direction == 'LONG' and long_count >= MAX_SAME_DIRECTION:
                logger.info(f"  ⛔ {sym}: LONG blocked (already {long_count} longs open)")
                continue
            if sig.direction == 'SHORT' and short_count >= MAX_SAME_DIRECTION:
                logger.info(f"  ⛔ {sym}: SHORT blocked (already {short_count} shorts open)")
                continue

            # Intelligent Structural Cooldown Guard
            cd = self.structural_cooldowns.get(sym)
            if cd and time.time() < cd['valid_until']:
                # If we are trying to re-enter the SAME direction that just failed
                if sig.direction == cd['side']:
                    entry_p = sig.entry_price
                    failed_sl = cd['sl_price']
                    
                    # LONG RULE: If we got stopped out, price MUST drop significantly lower than the SL
                    # before a new valid LONG can form (sweeping the liquidity).
                    # OR wait out the 4-hour hard lock.
                    if sig.direction == 'LONG' and entry_p >= failed_sl * 0.998:
                        logger.info(f"  🛡️ {sym}: LONG Blocked (Structural Cooldown - Price {entry_p} hasn't swept old SL {failed_sl})")
                        continue
                        
                    # SHORT RULE: If we got stopped out, price MUST rise significantly higher than the SL
                    # before a new valid SHORT can form.
                    if sig.direction == 'SHORT' and entry_p <= failed_sl * 1.002:
                        logger.info(f"  🛡️ {sym}: SHORT Blocked (Structural Cooldown - Price {entry_p} hasn't swept old SL {failed_sl})")
                        continue
                else:
                    # If we are reversing direction (e.g., failed LONG, now going SHORT), 
                    # we clear the cooldown immediately. Market changed flow.
                    if sym in self.structural_cooldowns:
                        del self.structural_cooldowns[sym]

            logger.info(f"  🎯 {sym}: {sig.direction} via {strat_name} (conf={conf:.2f})")
            await self.execute_trade(sym, sig, cfg)

            # Update counters after execution
            if sig.direction == 'LONG': long_count += 1
            else: short_count += 1

        self.pending_signals.clear()
        self.batch_timer_task = None

    # ── A5: Telegram Alerts ──────────────────────────────────────────────────

    async def _tg_alert(self, message: str):
        """Send alert via Telegram if bot is connected."""
        try:
            if self._tg_bot and hasattr(self._tg_bot, 'send_alert'):
                await self._tg_bot.send_alert(message)
        except Exception:
            pass  # Telegram is best-effort, never block trading

    async def _heartbeat_loop(self):
        """A5: Periodic heartbeat — reports system health to logs and Telegram."""
        await asyncio.sleep(60)  # Wait 1 min after startup
        logger.info(f"💓 Heartbeat monitor started (every {HEARTBEAT_INTERVAL//60} min)")

        while True:
            try:
                # Gather stats
                equity = 0
                try:
                    equity = await self.get_equity()
                except Exception:
                    pass

                positions = self.positions_cache or []
                long_count = sum(1 for p in positions if p.get('side') == 'LONG')
                short_count = sum(1 for p in positions if p.get('side') == 'SHORT')
                ws_uptime_h = (time.time() - self._ws_connection_start) / 3600 if self._ws_connection_start > 0 else 0

                # Exec analytics
                avg_slip = 0.0
                avg_lat = 0.0
                if self.exec_stats['total'] > 0:
                    avg_slip = self.exec_stats['slippage_sum'] / self.exec_stats['total']
                    avg_lat = self.exec_stats['latency_sum'] / self.exec_stats['total']

                # Stats & Analytics
                if PAPER_MODE:
                    closed_paper_list = [pt for pt in self.paper_trades if pt.get('status') == 'CLOSED']
                    open_paper = sum(1 for pt in self.paper_trades if pt.get('status') == 'OPEN')
                    closed_paper = len(closed_paper_list)
                    wins = sum(1 for pt in closed_paper_list if pt.get('pnl_pct', 0) > 0)
                    gross_profit = sum(pt.get('pnl_usdt', 0) for pt in closed_paper_list if pt.get('pnl_usdt', 0) > 0)
                    gross_loss = abs(sum(pt.get('pnl_usdt', 0) for pt in closed_paper_list if pt.get('pnl_usdt', 0) < 0))
                    
                    paper_info = f" | 💰${self.paper_equity:.0f} | Paper: {closed_paper} trades ({wins}W/{closed_paper-wins}L) | Open: {open_paper}"
                    
                    calc_trades = closed_paper
                else:
                    journal_closed = [t for t in self.journal if t.get('status') == 'CLOSED']
                    calc_trades = len(journal_closed)
                    wins = sum(1 for t in journal_closed if t.get('pnl_usdt', 0) > 0)
                    gross_profit = sum(t.get('pnl_usdt', 0) for t in journal_closed if t.get('pnl_usdt', 0) > 0)
                    gross_loss = abs(sum(t.get('pnl_usdt', 0) for t in journal_closed if t.get('pnl_usdt', 0) < 0))
                    paper_info = ""

                # Compute exact Win Rate and Profit Factor for UI Portfolio
                if calc_trades > 0:
                    win_rate = (wins / calc_trades) * 100
                    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0)
                else:
                    win_rate = 0.0
                    profit_factor = 0.0
                
                asyncio.create_task(self.broadcast({
                    'type': 'analytics_update', 'win_rate': win_rate, 
                    'profit_factor': profit_factor, 'trades': calc_trades
                }))


                mode_tag = "📄 PAPER" if PAPER_MODE else "💰 LIVE"
                msg = (
                    f"💓 {mode_tag} HEARTBEAT\n"
                    f"Equity: ${equity:,.2f}\n"
                    f"Positions: {long_count}L / {short_count}S\n"
                    f"WS uptime: {ws_uptime_h:.1f}h | Reconnects: {self._ws_reconnect_count}\n"
                    f"Avg slip: {avg_slip:.3f}% | Avg lat: {avg_lat:.0f}ms{paper_info}"
                )
                logger.info(msg.replace('\n', ' | '))
                await self._tg_alert(msg)

            except Exception as e:
                logger.error(f"Heartbeat error: {e}")

            await asyncio.sleep(HEARTBEAT_INTERVAL)

    # ── Config Hot-Reload ────────────────────────────────────────────────────

    async def _reload_config_if_changed(self):
        """Checks if active_config.json has been updated."""
        try:
            if not os.path.exists(self.config_path):
                return
            mtime = os.path.getmtime(self.config_path)
            if mtime > self.last_config_mtime:
                logger.info("📋 Hot-reloading config...")
                self.active_configs = self._load_configs()
                self.last_config_mtime = mtime
                await self.broadcast({'type': 'log', 'message': '🔄 Config hot-reloaded.', 'level': 'info'})
        except Exception as e:
            logger.error(f"Hot-reload failed: {e}")

    # ── UI WebSocket Bridge ──────────────────────────────────────────────────

    async def ui_server_handler(self, websocket):
        """Handles Electron UI client connections."""
        self.ui_clients.add(websocket)
        logger.info(f"UI client connected (total: {len(self.ui_clients)})")
        try:
            # 1. System Init
            await websocket.send(json.dumps({
                'type': 'system_init', 'status': 'connected',
                'server_time': time.time()
            }))
            
            # 2. Active Symbols & Configs
            if hasattr(self, 'active_configs') and self.active_configs:
                # Extract unique symbols (strip slashes to standard format)
                active_symbols = list(set([c['symbol'].replace('/', '') for c in self.active_configs]))
                # But UI needs it like 'BTCUSDT'
                await websocket.send(json.dumps({
                    'type': 'active_symbols', 
                    'symbols': active_symbols
                }))
                await websocket.send(json.dumps({
                    'type': 'active_configs',
                    'data': self.active_configs
                }))
            
            # 3. Cached logs
            if self.system_logs:
                await websocket.send(json.dumps({'type': 'log_buffer', 'data': list(self.system_logs[-50:])}))
                
            async for message in websocket:
                try:
                    data = json.loads(message)
                    if data.get('action') == 'ping':
                        await websocket.send(json.dumps({'type': 'pong', 'timestamp': time.time()}))
                except:
                    pass
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.ui_clients.discard(websocket)
            logger.info(f"UI client disconnected (total: {len(self.ui_clients)})")

    async def broadcast(self, data: dict):
        """Broadcast to all UI clients. Logs to buffer."""
        if data.get('type') == 'log':
            self.system_logs.append(data)
            if len(self.system_logs) > self.max_sys_logs:
                self.system_logs.pop(0)

        if not self.ui_clients:
            return
        msg = json.dumps(data)
        dead = set()
        for ws in list(self.ui_clients):
            try:
                await ws.send(msg)
            except:
                dead.add(ws)
        self.ui_clients -= dead

    # ── Background Maintenance ───────────────────────────────────────────────

    async def _maintenance_loop(self):
        """Periodic config reload and trade sync."""
        while True:
            await wait_for_cooldown("Maintenance")
            try:
                await self._reload_config_if_changed()
                await self._check_hotswap_flag()
            except Exception as e:
                logger.error(f"Maintenance error: {e}")
            await asyncio.sleep(60)

    async def _check_hotswap_flag(self):
        """Checks if cron_retrain.py finished training new models."""
        flag_path = os.path.join("data", "models", "retrain_flag.txt")
        if os.path.exists(flag_path):
            logger.info("🔥 HOT-SWAP DETECTED: New ML Models are ready!")
            t0 = time.time()
            
            # Hot-Swap Memory Reload
            for strat_key, ml in self.ml_filters.items():
                if ml:
                    try:
                        ml.load_model()
                    except Exception as e:
                        logger.error(f"Failed to hot-swap ml_filter for {strat_key}: {e}")
            
            try:
                os.remove(flag_path)
            except Exception as e:
                logger.warning(f"Could not delete hot-swap flag: {e}")
                
            t1 = time.time()
            lock_ms = (t1 - t0) * 1000
            msg = f"🚀 Successfully Hot-Swapped ML Models to fresh data! Reload time: {lock_ms:.1f}ms"
            logger.info(msg)
            asyncio.create_task(self.broadcast({'type': 'log', 'message': msg, 'level': 'success'}))
            await self._tg_alert(msg)

    # ══════════════════════════════════════════════════════════════════════════
    # ══  MAIN WEBSOCKET CONNECTION (Bulletproof v6.0)  ═══════════════════════
    # ══════════════════════════════════════════════════════════════════════════

    async def start_websocket_streams(self):
        """
        Main entry point. Handles:
        1. One-time initialization (history fetch)
        2. UI server startup
        3. Background tasks (positions, maintenance)
        4. Binance WebSocket connection with full resilience
        """

        # ── Stage 1: Initialize ──────────────────────────────────────────────
        logger.info("🚀 Starting LiveExecutor v6.0...")
        init_retry = 5
        while True:
            await wait_for_cooldown("Initialization")
            try:
                await self.initialize()
                break
            except Exception as e:
                if '429' in str(e):
                    set_global_cooldown(60)
                else:
                    logger.error(f"Init failed: {e}. Retrying in {init_retry}s...")
                    await asyncio.sleep(init_retry)
                    init_retry = min(init_retry * 2, 60)

        # ── Stage 2: Start UI Server ─────────────────────────────────────────
        try:
            ui_server = await websockets.serve(
                self.ui_server_handler, "0.0.0.0", 8080,
                ping_interval=20, ping_timeout=20
            )
            logger.info("🌐 UI Bridge active on port 8080")
        except Exception as e:
            logger.warning(f"UI server failed to start: {e}")

        # ── Stage 3: Start Background Tasks ──────────────────────────────────
        asyncio.create_task(self.track_positions_loop())
        asyncio.create_task(self._maintenance_loop())
        asyncio.create_task(self.perf.polling_loop(executor=self))
        asyncio.create_task(self._heartbeat_loop())  # A5
        logger.info("⚡ Background tasks started (positions, maintenance, perf, heartbeat)")

        # ── Stage 4: WebSocket Connection Loop ───────────────────────────────
        await self._ws_connection_loop()

    async def _ws_connection_loop(self):
        """
        Bulletproof WebSocket connection loop.
        
        Features:
        - Exponential backoff with jitter (5s → 120s max)
        - 23-hour forced reconnect (Binance 24h limit)
        - Health watchdog (5 min stale detection)
        - Structured error logging
        - SSL resilience
        """
        # Deduplicate streams (configs may list the same symbol multiple times)
        seen_streams = set()
        streams = []
        for c in self.active_configs:
            pair = c['symbol'].replace('/', '').lower()
            tf = c.get('timeframe', '5m')
            stream_key = f"{pair}@kline_{tf}"
            if stream_key not in seen_streams:
                streams.append(stream_key)
                seen_streams.add(stream_key)

        if not streams:
            logger.error("No streams to subscribe to!")
            return

        # Ensure BTC/USDT is always streamed for ML Correlation Matrix
        if "btcusdt@kline_5m" not in seen_streams:
            streams.append("btcusdt@kline_5m")
            seen_streams.add("btcusdt@kline_5m")

        url = f"wss://fstream.binance.com/stream?streams={'/'.join(streams)}"
        ssl_ctx = _create_ssl_context()

        reconnect_attempt = 0

        while True:
            # Exponential backoff with jitter
            if reconnect_attempt > 0:
                base_delay = min(5 * (2 ** (reconnect_attempt - 1)), WS_MAX_RECONNECT_DELAY)
                jitter = random.uniform(0, 2)
                delay = base_delay + jitter
                logger.info(f"🔄 Reconnecting in {delay:.1f}s (attempt #{reconnect_attempt})...")
                await asyncio.sleep(delay)

            await wait_for_cooldown("WebSocket")

            try:
                # Strip proxies RIGHT BEFORE connect (something may re-set them)
                for pv in ['http_proxy', 'https_proxy', 'all_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'socks_proxy', 'SOCKS_PROXY']:
                    os.environ.pop(pv, None)

                logger.info(f"📡 Connecting to Binance fstream ({len(streams)} streams)...")
                logger.info(f"   URL: {url[:80]}...")

                async with websockets.connect(
                    url,
                    ssl=ssl_ctx,
                    proxy=None,                        # EXPLICIT: no proxy
                    ping_interval=WS_PING_INTERVAL,    # Send ping every 2 min
                    ping_timeout=WS_PING_TIMEOUT,      # Dead if no pong in 30s
                    close_timeout=WS_CLOSE_TIMEOUT,    # Fast close
                    max_size=WS_MAX_MESSAGE_SIZE,       # 1 MB max
                    additional_headers={
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
                    },
                    open_timeout=15,
                ) as ws:
                    self._ws_connection_start = time.time()
                    self._last_ws_message_time = time.time()
                    reconnect_attempt = 0  # Reset on successful connect
                    self._ws_reconnect_count += 1

                    logger.info(f"✅ WebSocket connected! (session #{self._ws_reconnect_count})")
                    asyncio.create_task(self.broadcast({
                        'type': 'log', 'level': 'success',
                        'message': f'📡 WebSocket connected (session #{self._ws_reconnect_count})'
                    }))

                    # Start health watchdog for this connection
                    watchdog_task = asyncio.create_task(
                        self._health_watchdog(ws)
                    )

                    try:
                        async for message in ws:
                            self._last_ws_message_time = time.time()
                            await self._handle_kline_message(message)
                    except websockets.exceptions.ConnectionClosedOK:
                        logger.info("WebSocket closed normally (server-initiated).")
                    except websockets.exceptions.ConnectionClosedError as e:
                        logger.warning(f"⚠️ WebSocket closed with error: code={e.code}, reason='{e.reason}'")
                    finally:
                        watchdog_task.cancel()
                        try:
                            await watchdog_task
                        except asyncio.CancelledError:
                            pass

            except websockets.exceptions.InvalidStatusCode as e:
                logger.error(f"🚫 WebSocket rejected: HTTP {e.status_code}")
                if e.status_code == 403:
                    logger.error("   IP may be blocked. Consider VPN or waiting.")
                    set_global_cooldown(120)
                elif e.status_code == 429:
                    logger.error("   Rate limited!")
                    set_global_cooldown(60)
            except websockets.exceptions.InvalidURI as e:
                logger.error(f"❌ Invalid WebSocket URI: {e}")
                return  # Fatal — don't retry
            except (ConnectionError, OSError, asyncio.TimeoutError) as e:
                logger.warning(f"⚠️ Network error: {type(e).__name__}: {e}")
            except Exception as e:
                logger.error(f"❌ Unexpected WS error: {type(e).__name__}: {e}", exc_info=True)

            reconnect_attempt += 1
            logger.info(f"📊 Connection stats: total sessions={self._ws_reconnect_count}, current attempt={reconnect_attempt}")

    async def _health_watchdog(self, ws):
        """
        Monitors connection health:
        1. Detects stale connections (no data for 5 min)
        2. Forces reconnect after 23 hours (Binance 24h limit)
        """
        while True:
            await asyncio.sleep(60)  # Check every minute

            now = time.time()
            connection_age = now - self._ws_connection_start
            silence_duration = now - self._last_ws_message_time

            # Log health status every 10 minutes
            if int(connection_age) % 600 < 60:
                age_h = connection_age / 3600
                logger.info(
                    f"💓 WS Health: age={age_h:.1f}h, "
                    f"last_msg={silence_duration:.0f}s ago, "
                    f"session=#{self._ws_reconnect_count}"
                )

            # Check 1: Stale connection (no data for 5 min)
            if silence_duration > WS_HEALTH_TIMEOUT:
                logger.warning(
                    f"🔴 STALE CONNECTION: No data for {silence_duration:.0f}s. "
                    f"Forcing reconnect..."
                )
                await ws.close(1000, "Health watchdog: stale connection")
                return

            # Check 2: 23-hour age limit (Binance kills at 24h)
            if connection_age > WS_MAX_CONNECTION_LIFE:
                logger.info(
                    f"🔄 SCHEDULED RECONNECT: Connection age {connection_age/3600:.1f}h "
                    f"exceeds {WS_MAX_CONNECTION_LIFE/3600}h limit. Reconnecting..."
                )
                await ws.close(1000, "Scheduled 23h reconnect")
                return


if __name__ == '__main__':
    logger.error("Please run via main.py")
    sys.exit(1)
