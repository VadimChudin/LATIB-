///! Live Trading Orchestrator
///! ==========================
///! The brain of the Rust Live Engine. Connects all components:
///!
///! WebSocket Feed → Candle Buffer → Strategy Signals → ML Filter → Order Router
///!                                                                      ↓
///!                              Smart Trailer ← Position Manager ← Orders
///!
///! On each KlineClose:
///!   1. Append candle to rolling buffer (last 250 bars)
///!   2. Run active strategies on the buffer
///!   3. If signal found → extract features → ML predict
///!   4. If ML probability > threshold → place Market order
///!
///! On each Trade tick:
///!   1. Update SmartTrailer composite score
///!   2. If force-exit signal → close via Market order

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use tracing::{info, warn, error};

use super::wall_tracker::WallStore;
use super::tape_reader::{TapeStore, OrderFlowSignal};
use super::absorber::{AbsorberTracker, HftTarget, AbsorberResult};
use super::level_tracker::{LevelStore, self};
use super::scalp_monitor::{self as scalp_mon, ScalpDirection};
use super::spot_probe;
use super::trade_logger;
use super::liquidation_feed;
// use super::market_session;
use super::whale_detector;

use crate::backtest;
use crate::ml_inference::{self, LgbmModel, MetaInfo};
use crate::strategies;
use crate::live::config_loader::{self, LiveConfig};
use crate::live::order_book::OrderBookStore;
use crate::live::order_router::{OrderRouter, Side};
use crate::live::position_manager::*;
use crate::live::smart_trailer::*;
use crate::live::ws_feed;

// ── Constants ──────────────────────────────────────────────────────────────

const CANDLE_BUFFER_SIZE: usize = 250;
const ML_THRESHOLD: f64 = 0.50;
const MAX_POSITIONS: usize = 5;
const MAX_SAME_DIRECTION: usize = 3;
const DAILY_LOSS_LIMIT_R: f64 = 5.0;     // Stop trading after -5R daily
const EQUITY_TRAIL_PCT: f64 = 0.03;       // Reduce size at 3% drawdown from peak
const MAX_CORRELATED_POSITIONS: usize = 3; // Max positions in correlated alts
const MTF_EMA_PERIOD: usize = 20;
const MTF_15M_BUFFER_SIZE: usize = 100;

/// MTF (Multi-Timeframe) trend derived from 15m EMA
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MtfTrend {
    Bullish,   // Price above 15m EMA → allow LONG
    Bearish,   // Price below 15m EMA → allow SHORT
    Neutral,   // Not enough data yet
}

// ── Orchestrator ───────────────────────────────────────────────────────────

pub struct HftFireEvent {
    pub symbol: String,
    pub trade: crate::backtest::Trade,
    pub config: crate::live::config_loader::LiveConfig,
    pub entry_price: f64,
    pub target_wall_price: Option<f64>,
    pub is_wall_backed: bool,
}

/// Phase 30C: Entry Confirmation Tick
/// After absorber fires, wait up to CONFIRM_TIMEOUT for price to move
/// at least CONFIRM_MIN_PCT in the expected direction before entering.
const CONFIRM_TIMEOUT_SECS: f64 = 3.0;
const CONFIRM_MIN_PCT: f64 = 0.0003; // 0.03%

#[derive(Clone)]
pub struct PendingConfirmation {
    pub signal_price: f64,
    pub created_at: Instant,
    pub direction: Direction,
    pub trade: crate::backtest::Trade,
    pub config: crate::live::config_loader::LiveConfig,
    pub target_wall_price: Option<f64>,
    pub is_wall_backed: bool,
}

// Phase 10: Ensemble ML support
pub struct Ensemble {
    pub lgbm: LgbmModel,
    pub xgb: Option<LgbmModel>, // Using same parser as LGBM (compatible JSON format)
    pub rf: Option<LgbmModel>,  // Using same parser as LGBM
}

pub struct Orchestrator {
    configs: Vec<LiveConfig>,
    candle_buffers: HashMap<String, Vec<backtest::Candle>>,  // symbol → rolling candles
    position_manager: PositionManager,
    smart_trailer: SmartTrailer,
    order_router: Arc<OrderRouter>,
    ml_models: HashMap<String, Ensemble>,  // model_name → loaded ensemble
    paper_mode: bool,
    last_signal_time: HashMap<String, Instant>,  // cooldown per symbol
    // MTF (Multi-Timeframe) 15m EMA trend filter
    mtf_trend_store: HashMap<String, MtfTrend>,
    mtf_15m_buffers: HashMap<String, Vec<f64>>,  // symbol → last N close prices
    // Sniper execution: wall + order flow stores
    wall_store: WallStore,
    tape_store: TapeStore,
    // Live funding rates
    funding_rates: HashMap<String, f64>,
    // FR history: last 3 settlement rates per symbol
    fr_history: HashMap<String, Vec<f64>>,
    // HFT Absorber for knife catches
    absorber: Arc<AbsorberTracker>,
    // S/R Level tracker for ORB v2
    level_store: LevelStore,
    // Phase 30.5: Density Breakout Monitor
    density_radar: super::density_radar::DensityRadar,
    // Phase 11: Spot Probe HTTP client (reused across calls)
    http_client: reqwest::Client,
    // === Phase 10: Risk Management ===
    daily_pnl_r: f64,                         // Running daily PnL in R
    daily_trade_count: usize,                  // Daily trade count
    last_reset_day: u64,                       // Day number for daily reset
    equity_peak: f64,                          // Peak equity for trailing
    current_equity: f64,                       // Current equity
    position_locks: HashMap<String, String>,   // symbol → direction ("LONG"/"SHORT")
    // Phase 11: Trade Logger — active trade IDs per symbol
    trade_ids: HashMap<String, String>,           // symbol → trade_id
    // Phase 13: Liquidation Radar
    liq_store: liquidation_feed::LiqStore,
    // Phase 12: Whale Detector
    whale_detector: whale_detector::WhaleDetector,
    ob_store: OrderBookStore,
    // Phase 11.4: Meta-Inference
    meta_model: Option<LgbmModel>,
    meta_info: Option<MetaInfo>,
    // Phase 29C: RL Continuous Sizing Inference
    pub rl_agent: Option<crate::ml_inference_rl::RlAgent>,
    models_dir: PathBuf, // Phase 11 Hot-swap
    // Phase 14: Telegram Bot channel
    pub tg_tx: Option<tokio::sync::mpsc::Sender<String>>,
    pub active_modes: HashMap<String, String>, // "symbol_strat" -> "conservative" | "aggressive"
    pub shadow_stats: HashMap<String, ModeStats>, // "symbol_strat_mode" -> stats
    pub journal_analyzer: super::journal_analyzer::JournalAnalyzer, // Phase 22
    // Phase 26: HFT Signal channel
    pub hft_tx: tokio::sync::mpsc::UnboundedSender<HftFireEvent>,
    pub hft_rx: Option<tokio::sync::mpsc::UnboundedReceiver<HftFireEvent>>,
    pub live_stats: crate::live::live_stats::SharedStats,
    // Phase 30: Per-symbol loss cooldown (2 losses in 1h → pause)
    symbol_loss_times: HashMap<String, Vec<Instant>>,
    // Phase 30C: Pending entry confirmations (symbol → pending)
    pending_confirmations: HashMap<String, PendingConfirmation>,
    // Phase 36 FIX: CVD at squeeze detection (for divergence check matching backtester)
    squeeze_cvd: HashMap<String, (f64, Instant)>,  // symbol → (cvd_at_squeeze, when)
}

#[derive(Debug, Clone, Default)]
pub struct ModeStats {
    pub equity: f64,
    pub trades_count: usize,
    pub wins: usize,
}

impl Orchestrator {
    pub fn new(
        config_path: PathBuf,
        models_dir: PathBuf,
        api_key: String,
        api_secret: String,
        paper_mode: bool,
        wall_store: WallStore,
        tape_store: TapeStore,
        ob_store: OrderBookStore,
        liq_store: liquidation_feed::LiqStore,
        live_stats: crate::live::live_stats::SharedStats,
    ) -> Self {
        // Load active configs
        let configs = config_loader::load_active_configs(&config_path);

        // Load ML models (Phase 10: Ensemble)
        let mut ml_models = HashMap::new();
        let model_names: Vec<String> = configs.iter()
            .map(|c| c.model_name().to_string())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        for model_name in &model_names {
            // Load base LGBM
            let lgbm_path = models_dir.join(format!("{}.json", model_name));
            if let Ok(lgbm) = LgbmModel::load(&lgbm_path) {
                // Try load XGB
                let xgb_path = models_dir.join(format!("xgb_{}.json", model_name));
                let xgb = LgbmModel::load(&xgb_path).ok();
                
                // Try load RF
                let rf_path = models_dir.join(format!("rf_{}.json", model_name));
                let rf = LgbmModel::load(&rf_path).ok();

                let ensemble_size = 1 + if xgb.is_some() { 1 } else { 0 } + if rf.is_some() { 1 } else { 0 };
                info!("🧠 Loaded ML Ensemble: {} ({} models active)", model_name, ensemble_size);

                ml_models.insert(model_name.clone(), Ensemble { lgbm, xgb, rf });
            } else {
                warn!("⚠️ Could not load base ML model {} (trading without ML filter)", model_name);
            }
        }

        let (hft_tx, hft_rx) = tokio::sync::mpsc::unbounded_channel();

        // Load Meta-Model (Phase 11.4)
        let meta_json_path = models_dir.join("meta_model.json");
        let meta_info_path = models_dir.join("meta_model_info.json");
        
        let meta_model = LgbmModel::load(&meta_json_path).ok();
        let meta_info: Option<MetaInfo> = if meta_info_path.exists() {
            let data = std::fs::read_to_string(&meta_info_path).unwrap_or_default();
            serde_json::from_str(&data).ok()
        } else {
            None
        };

        if let Some(ref info) = meta_info {
            info!("🧠 Meta-Model MetaInfo loaded. Train size: {} trades.", info.train_size);
        }

        // Phase 29C: Load RL Agent Weights
        let rl_path_local = std::path::PathBuf::from("../data/models/knife_ppo_weights.json");
        let rl_path_prod = std::path::PathBuf::from("data/models/knife_ppo_weights.json");
        let mut rl_agent = crate::ml_inference_rl::RlAgent::load_from_json(rl_path_local.to_str().unwrap());
        if rl_agent.is_some() {
            info!("🧠 RL Agent loaded successfully for Continuous Action sizing!");
        } else {
            rl_agent = crate::ml_inference_rl::RlAgent::load_from_json(rl_path_prod.to_str().unwrap());
            if rl_agent.is_some() {
                info!("🧠 RL Agent loaded successfully for Continuous Action sizing!");
            } else {
                warn!("⚠️ RL Agent not found. Will use standard discrete sizing.");
            }
        }

        let order_router = Arc::new(OrderRouter::new(api_key, api_secret, paper_mode));

        info!("🎯 Orchestrator initialized:");
        info!("   {} active configs", configs.len());
        info!("   {} ML ensembles loaded", ml_models.len());
        info!("   Mode: {}", if paper_mode { "PAPER" } else { "LIVE" });

        Self {
            configs,
            candle_buffers: HashMap::new(),
            position_manager: PositionManager::new(),
            smart_trailer: SmartTrailer::new(TrailerConfig::default()),
            order_router,
            ml_models,
            rl_agent,
            paper_mode,
            last_signal_time: HashMap::new(),
            mtf_trend_store: HashMap::new(),
            mtf_15m_buffers: HashMap::new(),
            wall_store,
            tape_store,
            funding_rates: HashMap::new(),
            fr_history: HashMap::new(),
            absorber: Arc::new(AbsorberTracker::new()),
            level_store: level_tracker::new_store(),
            // Phase 11: Spot Probe HTTP client
            http_client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(3))
                .build()
                .unwrap_or_default(),
            // Phase 10: Risk Management
            daily_pnl_r: 0.0,
            daily_trade_count: 0,
            last_reset_day: 0,
            equity_peak: 10000.0,  // Starting equity assumption
            current_equity: 10000.0,
            position_locks: HashMap::new(),
            trade_ids: HashMap::new(),
            liq_store,
            ob_store,
            whale_detector: whale_detector::WhaleDetector::new(),
            meta_model,
            meta_info,
            models_dir: models_dir.clone(),
            tg_tx: None,
            active_modes: HashMap::new(),
            shadow_stats: HashMap::new(),
            journal_analyzer: {
                let mut ja = super::journal_analyzer::JournalAnalyzer::new();
                ja.reload_from_file("data/trade_log.jsonl");
                ja
            },
            hft_tx,
            hft_rx: Some(hft_rx),
            live_stats,
            symbol_loss_times: HashMap::new(),
            pending_confirmations: HashMap::new(),
            squeeze_cvd: HashMap::new(),
            density_radar: super::density_radar::DensityRadar::new(),
        }
    }

    /// Get all unique symbols for WebSocket subscription
    pub fn get_symbols(&self) -> Vec<String> {
        config_loader::get_active_symbols(&self.configs)
    }

    pub async fn execute_hft_entry(&mut self, event: HftFireEvent) {
        let mut trade = event.trade.clone();
        
        // The macro signal was generated at a sub-optimal price.
        // The HFT Absorber waited and found the absolute bottom/top.
        // We update the entry price to the real executed price.
        let old_price = trade.entry_price;
        trade.entry_price = event.entry_price;
        
        // Macro sl_price was calculated as entry_price * (1 +/- sl_pct).
        // To maintain the exact GA risk logic, we recalculate SL and TP based on the new entry price.
        if trade.direction == "LONG" {
            let sl_pct = (old_price - trade.sl_price) / old_price;
            let tp_pct = (trade.tp_price - old_price) / old_price;
            trade.sl_price = event.entry_price * (1.0 - sl_pct);
            trade.tp_price = event.entry_price * (1.0 + tp_pct);
        } else {
            let sl_pct = (trade.sl_price - old_price) / old_price;
            let tp_pct = (old_price - trade.tp_price) / old_price;
            trade.sl_price = event.entry_price * (1.0 + sl_pct);
            trade.tp_price = event.entry_price * (1.0 - tp_pct);
        }

        // FIX: Recalculate risk_dist to match new entry/SL (was keeping stale macro value)
        let old_risk_dist = trade.risk_dist;
        trade.risk_dist = (trade.entry_price - trade.sl_price).abs();
        tracing::info!("🔪 [HFT Exec] Routing {} {:?} @ {:.4} (Macro was {:.4}) | risk_dist: {:.6} → {:.6}", 
            event.symbol, trade.direction, event.entry_price, old_price, old_risk_dist, trade.risk_dist);

        // Phase 30.6: Instant entry bypass for Density Breakout
        if event.config.strategy == "breakout" {
            info!("🚀 [Confirm] {} {:?} @ {:.6} — INSTANT ENTRY for Density Breakout!",
                event.symbol, trade.direction, event.entry_price);
            self.execute_entry(&event.symbol, &trade, &event.config, 0.0, None, event.target_wall_price, event.is_wall_backed).await;
            return;
        }

        // Phase 30C: Don't enter immediately — store pending confirmation
        // Wait for price to move in our direction before committing capital
        let direction = if trade.direction == "LONG" { Direction::Long } else { Direction::Short };
        
        // Skip confirmation if we already have a pending or position on this symbol
        if self.position_manager.has_position(&event.symbol) {
            info!("⏭️ [Confirm] {} already has position — skipping", event.symbol);
            return;
        }
        if self.pending_confirmations.contains_key(&event.symbol) {
            info!("⏭️ [Confirm] {} already has pending confirmation — skipping", event.symbol);
            return;
        }

        info!("⏳ [Confirm] {} {:?} @ {:.6} — waiting for {:.2}% move within {:.0}s",
            event.symbol, direction, event.entry_price, CONFIRM_MIN_PCT * 100.0, CONFIRM_TIMEOUT_SECS);

        self.pending_confirmations.insert(event.symbol.clone(), PendingConfirmation {
            signal_price: event.entry_price,
            created_at: Instant::now(),
            direction,
            trade,
            config: event.config,
            target_wall_price: event.target_wall_price,
            is_wall_backed: event.is_wall_backed,
        });
    }

    /// Phase 30C: Check and process pending entry confirmations
    /// Called on every tick BEFORE position checks
    async fn check_pending_confirmation(&mut self, symbol: &str, price: f64) {
        // Quick check: do we have a pending for this symbol?
        let pending = match self.pending_confirmations.get(symbol) {
            Some(p) => p.clone(),
            None => return,
        };

        let elapsed = pending.created_at.elapsed().as_secs_f64();
        let move_pct = (price - pending.signal_price) / pending.signal_price;

        // Check if price moved in the expected direction
        let confirmed = match pending.direction {
            Direction::Long => move_pct >= CONFIRM_MIN_PCT,
            Direction::Short => move_pct <= -CONFIRM_MIN_PCT,
        };

        if confirmed {
            // 🎯 Bounce confirmed! Execute entry at CURRENT price (not signal price)
            info!("✅ [Confirm] {} {:?} CONFIRMED in {:.1}s — price moved {:.4}% (signal={:.6}, now={:.6})",
                symbol, pending.direction, elapsed, move_pct * 100.0, pending.signal_price, price);

            // Update entry price to current (slightly worse but confirmed)
            let mut trade = pending.trade.clone();
            let old_entry = trade.entry_price;
            trade.entry_price = price;

            // Recalculate SL/TP distances proportionally
            if trade.direction == "LONG" {
                let sl_dist = old_entry - trade.sl_price;
                let tp_dist = trade.tp_price - old_entry;
                trade.sl_price = price - sl_dist;
                trade.tp_price = price + tp_dist;
            } else {
                let sl_dist = trade.sl_price - old_entry;
                let tp_dist = old_entry - trade.tp_price;
                trade.sl_price = price + sl_dist;
                trade.tp_price = price - tp_dist;
            }
            trade.risk_dist = (trade.entry_price - trade.sl_price).abs();

            self.pending_confirmations.remove(symbol);
            self.execute_entry(symbol, &trade, &pending.config, 0.0, None, pending.target_wall_price, pending.is_wall_backed).await;
            return;
        }

        // Check timeout
        if elapsed >= CONFIRM_TIMEOUT_SECS {
            // ❌ No bounce within timeout — skip entry
            info!("❌ [Confirm] {} {:?} EXPIRED after {:.1}s — price={:.6} (signal={:.6}, move={:.4}%). NO BOUNCE.",
                symbol, pending.direction, elapsed, price, pending.signal_price, move_pct * 100.0);
            self.pending_confirmations.remove(symbol);
        }
    }

    /// Process funding rate update from Binance markPrice stream
    pub fn on_funding_rate_update(&mut self, symbol: &str, rate: f64) {
        self.funding_rates.insert(symbol.to_string(), rate);
        // Track history (keep last 3)
        let history = self.fr_history.entry(symbol.to_string()).or_insert_with(Vec::new);
        history.push(rate);
        if history.len() > 3 {
            history.remove(0);
        }
    }

    /// Process a closed candle (KlineClose event)
    /// This is where strategy evaluation and entry signals happen
    pub async fn on_candle_close(&mut self, symbol: &str, ws_candle: &ws_feed::Candle) {
        // Convert ws_feed::Candle to backtest::Candle
        let bt_candle = backtest::Candle {
            timestamp: ws_candle.timestamp.to_string(),
            open: ws_candle.open,
            high: ws_candle.high,
            low: ws_candle.low,
            close: ws_candle.close,
            volume: ws_candle.volume,
            num_trades: 0.0,
            taker_buy_volume: 0.0,
            quote_volume: 0.0,
        };

        // Update rolling buffer and get a snapshot
        {
            let buffer = self.candle_buffers.entry(symbol.to_string()).or_insert_with(Vec::new);
            buffer.push(bt_candle);
            if buffer.len() > CANDLE_BUFFER_SIZE {
                buffer.drain(..buffer.len() - CANDLE_BUFFER_SIZE);
            }
        }

        // Get buffer reference (no longer mutably borrowed)
        let buffer = match self.candle_buffers.get(symbol) {
            Some(b) => b,
            None => return,
        };

        // Need minimum candles for indicator calculation
        if buffer.len() < 201 {
            // Log warmup progress every 10th candle
            if buffer.len() % 10 == 0 {
                info!("⏳ [Warming Up] {} buffer: {}/201 candles", symbol, buffer.len());
            }
            return;
        }

        // Update S/R levels from hourly aggregation (every candle)
        self.update_levels_for_symbol(symbol);

        // Skip if we already have a position on this symbol
        if self.position_manager.has_position(symbol) {
            // But update trailing on candle close
            if let Some(pos) = self.position_manager.positions.get_mut(symbol) {
                // Skip candle trail for knife_tick — DE uses tick-level BE/trail only
                if pos.strategy != "knife_tick" {
                    let updated = pos.update_trail_candle(ws_candle.high, ws_candle.low);
                    if updated {
                        info!("📐 {} trail updated: SL={:.4}", symbol, pos.sl_price);
                    }
                }
            }
            return;
        }

        // Max position limit
        if self.position_manager.positions.len() >= MAX_POSITIONS {
            return;
        }

        // === RISK GATE 4: Correlation Limit ===
        // Group highly correlated altcoins to avoid concentrated risk
        let correlated_alts = ["ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", "XRPUSDT"];
        let clean_symbol = symbol.replace("/", "").to_uppercase();
        if correlated_alts.contains(&clean_symbol.as_str()) {
            let correlated_count = self.position_manager.positions.values()
                .filter(|p| {
                    let p_sym = p.symbol.replace("/", "").to_uppercase();
                    correlated_alts.contains(&p_sym.as_str())
                })
                .count();

            if correlated_count >= MAX_CORRELATED_POSITIONS {
                info!("🚫 Correlation limit reached: {} correlated alts already open, skipping {}", 
                    correlated_count, clean_symbol);
                return;
            }
        }

        // Cooldown check (30 seconds between signals per symbol)
        if let Some(last) = self.last_signal_time.get(symbol) {
            if Instant::now().checked_duration_since(*last).unwrap_or_default().as_secs() < 30 {
                return;
            }
        }

        // === DENSITY BREAKOUT SCANNER ===
        if let Some(target) = self.density_radar.check(symbol, ws_candle.close, &self.level_store, &self.wall_store, &self.tape_store) {
            let absorber = self.absorber.clone();
            if absorber.try_accept(&target).await {
                let ws = self.wall_store.clone();
                let ts = self.tape_store.clone();
                let sym = symbol.to_string();
                let direction = target.direction;
                let hft_tx_clone = self.hft_tx.clone();
                
                info!("🧱🔪 Breakout Absorber SPAWNED for {} {:?} — watching wall", sym, direction);

                tokio::spawn(async move {
                    let result = super::absorber::track_absorption(
                        sym.clone(), direction, target.strategy_type.clone(), ws, ts, absorber,
                    ).await;
                    
                    if let AbsorberResult::Fired { symbol, direction, confidence, entry_price, target_wall_price, is_wall_backed } = result {
                        info!("🧱🔥 DENSITY BREAKOUT FIRED: {} {:?} @ {:.4} (score={})",
                            symbol, direction, entry_price, confidence);
                            
                        let trade = crate::backtest::Trade {
                            entry_price,
                            tp_price: entry_price * if direction == crate::live::position_manager::Direction::Long { 1.05 } else { 0.95 },
                            sl_price: entry_price * if direction == crate::live::position_manager::Direction::Long { 0.98 } else { 1.02 },
                            direction: if direction == crate::live::position_manager::Direction::Long { "LONG".to_string() } else { "SHORT".to_string() },
                            ..Default::default()
                        };
                        
                        let config = crate::live::config_loader::LiveConfig {
                            symbol: symbol.clone(),
                            timeframe: "1m".to_string(),
                            strategy: "breakout".to_string(),
                            tier: 1, 
                            leverage: 10,
                            conservative: None,
                            aggressive: None,
                            params: Some(serde_json::json!({})),
                            metrics: Default::default(),
                        };
                        
                        let _ = hft_tx_clone.send(HftFireEvent {
                            symbol,
                            trade,
                            config,
                            entry_price,
                            target_wall_price,
                            is_wall_backed,
                        });
                    }
                });
            }
        }

        // Find configs for this symbol — clone to break borrow on self
        let buffer = match self.candle_buffers.get(symbol) {
            Some(b) => b.clone(), // Phase 21: Clone to allow &mut self later in the loop
            None => return,
        };

        // Strip out both '/' and '_' for uniform comparison (e.g. TRIA/USDT vs TRIA_USDT vs TRIAUSDT)
        let clean_symbol = symbol.replace("/", "").replace("_", "").to_uppercase();
        let matching_configs: Vec<LiveConfig> = self.configs.iter()
            .filter(|c| c.symbol.replace("/", "").replace("_", "").to_uppercase() == clean_symbol)
            .filter(|c| c.rust_strategy_name() != "orb")
            .cloned()
            .collect();

        if matching_configs.is_empty() {
            return;
        }

        // --- PRECOMPUTE (Turbo Mode) ---
        let closes: Vec<f64> = buffer.iter().map(|c| c.close).collect();
        let (bb_upper, bb_lower, bb_mid) = backtest::calc_bollinger_bands(&closes, 20, 2.0);
        let precomputed = backtest::PrecomputedData {
            atr: backtest::calc_atr(&buffer, 14),
            rsi: backtest::calc_rsi(&closes, 14),
            ema_fast: backtest::calc_ema(&closes, 9),
            ema_slow: backtest::calc_ema(&closes, 50),
            ema_200: backtest::calc_ema(&closes, 200),
            adx: backtest::calc_adx(&buffer, 14),
            bb_upper,
            bb_lower,
            bb_mid,
            bitsets: None,
            btc_vol: None,
            delta: buffer.iter().map(|c| 2.0 * c.taker_buy_volume - c.volume).collect(),
            tape_speed: buffer.iter().map(|c| c.num_trades).collect(),
        };

        // Result collection for Borrow Checker peace
        let mut mode_results = Vec::new();

        // Evaluate each active strategy
        for config in &matching_configs {
            let rust_strat = config.rust_strategy_name();
            let key = format!("{}_{}", symbol, rust_strat);
            
            // Get modes (local copies)
            let active_mode = self.active_modes.get(&key).cloned().unwrap_or_else(|| "conservative".to_string());
            let shadow_mode = if active_mode == "conservative" { "aggressive" } else { "conservative" };
            
            let params_active = config.params_vec(&active_mode);
            let params_shadow = config.params_vec(&shadow_mode);

            // 1. Run Shadow/Active Comparison (Performance tracking)
            let trades_active = match rust_strat {
                "smc" => strategies::smc::run_backtest_with_params(&buffer, &precomputed, &params_active),
                "knifetick" => strategies::knife_tick_macro::run_backtest_with_params(&buffer, &precomputed, &params_active),
                "scalpmtf" => strategies::scalp_mtf::run_backtest_with_params(&buffer, &precomputed, &params_active),
                "fundingrate" => strategies::funding_rate::run_backtest_with_params(&buffer, &precomputed, &params_active),
                "density" => strategies::density::run_backtest_with_params(&buffer, &precomputed, &params_active),
                _ => vec![],
            };
            
            let trades_shadow = match rust_strat {
                "smc" => strategies::smc::run_backtest_with_params(&buffer, &precomputed, &params_shadow),
                "knifetick" => strategies::knife_tick_macro::run_backtest_with_params(&buffer, &precomputed, &params_shadow),
                "scalpmtf" => strategies::scalp_mtf::run_backtest_with_params(&buffer, &precomputed, &params_shadow),
                "fundingrate" => strategies::funding_rate::run_backtest_with_params(&buffer, &precomputed, &params_shadow),
                "density" => strategies::density::run_backtest_with_params(&buffer, &precomputed, &params_shadow),
                _ => vec![],
            };

            mode_results.push((config.clone(), key, active_mode, shadow_mode, trades_active, trades_shadow));
        }

        // 2. Mutable Section: Update stats and check swaps
        for (_, key, active_mode, shadow_mode, trades_active, trades_shadow) in &mode_results {
            self.update_mode_stats(&format!("{}_{}", key, active_mode), trades_active);
            self.update_mode_stats(&format!("{}_{}", key, shadow_mode), trades_shadow);
            self.check_hotswap(symbol, "n/a", key);
        }

        // 3. Signal Processing Section
        for (config, key, _, _, trades_active, trades_shadow) in mode_results {
            let rust_strat = config.rust_strategy_name();
            let final_mode = self.active_modes.get(&key).cloned().unwrap_or_else(|| "conservative".to_string());
            
            // Pick trades from the now-active mode
            let trades = if final_mode == "aggressive" { trades_shadow } else { trades_active };

            // ... proceed with signal check for `trades`
            if trades.is_empty() { continue; }

            // Check if the strategy wants to be in a position RIGHT NOW
            if let Some(last_trade) = trades.last() {
                let is_unclosed = last_trade.exit_price == 0.0 && last_trade.pnl_r == 0.0;
                let is_fresh = last_trade.entry_idx >= buffer.len().saturating_sub(3);
                
                if is_unclosed && !is_fresh {
                    info!("⏭️ STALE SIGNAL: {} {} entry_idx={} (need >= {}) — skipping",
                        last_trade.direction, symbol, last_trade.entry_idx, buffer.len().saturating_sub(3));
                    continue;
                }
                
                if is_unclosed && is_fresh {
                    info!("⚡ LIVE SIGNAL: {} {} on {} (strategy: {}, entry_idx={}/{})",
                        last_trade.direction, symbol, config.strategy, rust_strat,
                        last_trade.entry_idx, buffer.len());

                    // MTF Filter: block trades against 15m EMA trend
                    let mtf = self.mtf_trend_store.get(symbol).copied().unwrap_or(MtfTrend::Neutral);
                    let is_long_signal = last_trade.direction == "LONG";
                    if is_long_signal && mtf == MtfTrend::Bearish {
                        info!("🔻 MTF BLOCKED: LONG {} rejected (15m EMA bearish)", symbol);
                        continue;
                    }
                    if !is_long_signal && mtf == MtfTrend::Bullish {
                        info!("🔺 MTF BLOCKED: SHORT {} rejected (15m EMA bullish)", symbol);
                        continue;
                    }

                    // Phase 22: Master Advisor - Journal-based Risk Blocking
                    if !self.journal_analyzer.is_strategy_allowed(&config.strategy) {
                        continue;
                    }

                    // Calculate BTC Gravity features
                    let mut btc_trend = 0.0;
                    let mut btc_vol = 0.0;
                    let mut btc_dump = 0.0;

                    let btc_key = if self.candle_buffers.contains_key("BTC/USDT") { "BTC/USDT" } else { "BTCUSDT" };
                    if let Some(btc_buffer) = self.candle_buffers.get(btc_key) {
                        let len = btc_buffer.len();
                        if len > 200 {
                            let curr = len - 1;
                            let closes: Vec<f64> = btc_buffer.iter().map(|c| c.close).collect();
                            
                            let ema_50: f64 = closes[curr.saturating_sub(49)..=curr].iter().sum::<f64>() / 50.0;
                            let ema_200: f64 = closes[curr.saturating_sub(199)..=curr].iter().sum::<f64>() / 200.0;
                            
                            if ema_200 > 0.0 {
                                btc_trend = (ema_50 - ema_200) / ema_200 * 100.0;
                            }
                            
                            let atr = crate::backtest::calc_atr(&btc_buffer, 14);
                            let c = btc_buffer[curr].close;
                            if c > 0.0 {
                                btc_vol = atr[curr] / c * 100.0;
                            }
                            
                            let past_idx = curr.saturating_sub(2);
                            let past_open = btc_buffer[past_idx].open;
                            if past_open > 0.0 {
                                btc_dump = (c - past_open) / past_open * 100.0;
                            }
                        }
                    }

                    let funding_rate = self.funding_rates.get(symbol).copied().unwrap_or(0.0);

                    // ML Filter — pass buffer as slice
                    let ml_pass = self.check_ml_filter(&config, &buffer, last_trade.entry_idx, btc_trend, btc_vol, btc_dump, funding_rate);

                    if !ml_pass {
                        info!("🧠 ML BLOCKED: {} {} rejected by ensemble", last_trade.direction, symbol);
                        continue;
                    }

                    info!("🧠 ML PASSED: {} {} approved by ensemble", last_trade.direction, symbol);
                    {
                        // Direction check
                        let is_long = last_trade.direction == "LONG";
                        if is_long && self.position_manager.long_count() >= MAX_SAME_DIRECTION {
                            info!("⏭️ Skipping LONG {}: max same-direction limit", symbol);
                            continue;
                        }
                        if !is_long && self.position_manager.short_count() >= MAX_SAME_DIRECTION {
                            info!("⏭️ Skipping SHORT {}: max same-direction limit", symbol);
                            continue;
                        }

                        // === Phase 30: VOLATILITY GATE (knife_tick only) ===
                        // If the coin's 2h range is less than 2× SL distance, it's in a dead chop.
                        // DE found good params on trending data, but can't profit in a sideways channel.
                        if config.rust_strategy_name() == "knifetick" {
                            let lookback = 120.min(buffer.len()); // 120 × 1min = 2 hours
                            let recent = &buffer[buffer.len() - lookback..];
                            let range_high = recent.iter().map(|c| c.high).fold(f64::MIN, f64::max);
                            let range_low = recent.iter().map(|c| c.low).fold(f64::MAX, f64::min);
                            let range_pct = if range_low > 0.0 { (range_high - range_low) / range_low } else { 0.0 };
                            
                            // SL from config params
                            let sl_pct = config.params.as_ref()
                                .and_then(|p| p.get("sl_pct").or_else(|| p.get("sl_buffer_pct")))
                                .and_then(|v| v.as_f64())
                                .unwrap_or(0.003);
                            
                            let min_range = sl_pct * 2.0; // Need at least 2× SL width of movement
                            
                            if range_pct < min_range {
                                info!("📉 VOL GATE BLOCKED: {} — 2h range={:.3}% < min={:.3}% (SL={:.3}%). Market is flat.",
                                    symbol, range_pct * 100.0, min_range * 100.0, sl_pct * 100.0);
                                continue;
                            }
                            info!("📈 VOL GATE OK: {} — 2h range={:.3}% ≥ min={:.3}%", 
                                symbol, range_pct * 100.0, min_range * 100.0);

                            // === Phase 30B: SPEED FILTER REMOVED ===
                            // Analysis of trade_log.jsonl showed profitable knife_tick bounces
                            // often occur at tape_speed 0.2-0.6 (after volume exhaustion).
                            // The speed < 1.0 filter was blocking winning trades.
                            // Observation Window in position_manager now handles "dead market"
                            // exits more intelligently.
                        }

                        // Sniper Confirmation: check walls + order flow
                        // NOTE: Skip sniper for knifetick — knives catch bounces, not breakouts.
                        // Wall-eating checks are irrelevant and block 100% of valid knife signals.
                        let (sniper_pass, dynamic_tp) = if config.rust_strategy_name() == "knifetick" {
                            info!("🎯 Sniper SKIPPED for knifetick {} {} — bounce strategy, walls irrelevant", last_trade.direction, symbol);
                            (true, None) // No Dynamic TP for knifetick — use GA-optimized tp_pct only
                        } else {
                            self.sniper_confirm(symbol, is_long, last_trade.entry_price).await
                        };
                        if !sniper_pass {
                            info!("🎯 SNIPER BLOCKED: {} {} — wall/flow mismatch", last_trade.direction, symbol);
                            continue;
                        }
                        info!("🎯 SNIPER PASSED: {} {} — executing trade!", last_trade.direction, symbol);

                        // Clone trade data to break borrows, then execute
                        let mut trade_clone = last_trade.clone();
                        // Override TP with dynamic wall-based TP if available
                        if let Some(dtp) = dynamic_tp {
                            info!("🎯 Dynamic TP: {:.4} → {:.4} (next wall)", trade_clone.tp_price, dtp);
                            trade_clone.tp_price = dtp;
                        }

                        // === HFT ABSORBER FORK ===
                        // KnifeTick has been migrated to `check_macro_triggers()` (True Tick Engine).
                        // It no longer triggers on 1m candle closes.
                        if config.rust_strategy_name() == "knifetick" {
                            info!("🔪 True Tick Migration: {} skipped in on_candle_close, handled by polling loop.", symbol);
                        } else if config.strategy == "density" {
                            // === DENSITY BREAKOUT: Wall + Tape + Sniper ===
                            let (sniper_ok, dtp) = self.sniper_confirm(symbol, is_long, last_trade.entry_price).await;
                            
                            let mut hft_confirmed = false;
                            if let Some(wall_snap) = self.wall_store.get(symbol) {
                                if !wall_snap.is_warming_up {
                                    let walls = if is_long { wall_snap.ask_walls() } else { wall_snap.bid_walls() };
                                    if let Some(wall) = walls.iter().find(|w| {
                                        let dist = (w.price - last_trade.entry_price).abs() / last_trade.entry_price;
                                        dist < 0.005 // Wall must be very close to breakout entry
                                    }) {
                                        if wall.touch_count >= 2 && wall.eaten_pct() > 0.30 {
                                            hft_confirmed = true;
                                            info!("📊🔥 DENSITY [{}]: Wall {:.4} confirmed (touches={}, eaten={:.1}%)", 
                                                symbol, wall.price, wall.touch_count, wall.eaten_pct() * 100.0);
                                        }
                                    }
                                }
                            }

                            // Tape confirmation
                            if hft_confirmed {
                                if let Some(tape) = self.tape_store.get(symbol) {
                                    let delta = tape.normalized_delta();
                                    if (is_long && delta < 0.1) || (!is_long && delta > -0.1) {
                                        hft_confirmed = false; // Delta must support the breakout
                                        info!("📊⏸️ DENSITY [{}]: Tape delta {:.2} weak for breakout", symbol, delta);
                                    }
                                }
                            }

                            if hft_confirmed && sniper_ok {
                                info!("📊🚀 DENSITY BREAKOUT CONFIRMED [{}] — executing!", symbol);
                                let mut trade_final = trade_clone.clone();
                                if let Some(dtp_val) = dtp {
                                    trade_final.tp_price = dtp_val;
                                }
                                let config_clone = config.clone();
                                self.execute_entry(symbol, &trade_final, &config_clone, 0.0, None, None, false).await;
                            } else {
                                info!("📊⏭️ DENSITY [{}] conditions not met (HFT: {}, Sniper: {})", 
                                    symbol, hft_confirmed, sniper_ok);
                            }
                        } else if config.strategy == "smc" {
                            // === SMC v2: Sniper + Level + Iceberg Gate ===
                            let (sniper_ok, dtp) = self.sniper_confirm(symbol, is_long, last_trade.entry_price).await;
                            if !sniper_ok {
                                info!("🏛️❌ SMC sniper BLOCKED {} {} — walls/flow mismatch", 
                                    if is_long { "LONG" } else { "SHORT" }, symbol);
                            } else {
                                // Additional institutional checks
                                let mut bonus_confirms = 0u32;

                                // Check S/R level nearby
                                if let Some(snap) = self.level_store.get(symbol) {
                                    if let Some(level) = snap.nearest_level(last_trade.entry_price, 2.0) {
                                        if level.touches >= 2 {
                                            bonus_confirms += 1;
                                            info!("🏛️📊 SMC [{}]: S/R level {:.4} ({} touches)", symbol, level.price, level.touches);
                                        }
                                    }
                                    if snap.poc_confirms_level(last_trade.entry_price) {
                                        bonus_confirms += 1;
                                        info!("🏛️📊 SMC [{}]: POC confirms entry zone", symbol);
                                    }
                                }

                                // Check iceberg pressure
                                if let Some(state) = self.tape_store.get(symbol) {
                                    let (ice_buy, ice_sell) = state.iceberg_pressure();
                                    if is_long && ice_buy > 0.2 {
                                        bonus_confirms += 1;
                                        info!("🏛️🧊 SMC [{}]: Iceberg BUY pressure {:.2}", symbol, ice_buy);
                                    } else if !is_long && ice_sell > 0.2 {
                                        bonus_confirms += 1;
                                        info!("🏛️🧊 SMC [{}]: Iceberg SELL pressure {:.2}", symbol, ice_sell);
                                    }
                                }

                                info!("🏛️ SMC [{}] sniper=✅ bonus={}/3 — executing!", symbol, bonus_confirms);
                                let mut trade_final = trade_clone.clone();
                                if let Some(dtp_val) = dtp {
                                    trade_final.tp_price = dtp_val;
                                }
                                let config_clone = config.clone();
                                self.execute_entry(symbol, &trade_final, &config_clone, 0.0, None, None, false).await;
                            }
                        } else if config.strategy == "scalpmtf" {
                            // === SCALPMTF v2: Sniper + Dynamic SL + ScalpMonitor ===
                            let (sniper_ok, dtp) = self.sniper_confirm(symbol, is_long, last_trade.entry_price).await;
                            if !sniper_ok {
                                info!("📈❌ ScalpMTF sniper BLOCKED {} {} — walls/flow mismatch",
                                    if is_long { "LONG" } else { "SHORT" }, symbol);
                            } else {
                                // Dynamic SL: use nearest wall or S/R level
                                let mut dynamic_sl = last_trade.sl_price;
                                let atr_est = (last_trade.entry_price - last_trade.sl_price).abs();

                                // Try wall-based SL
                                if let Some(wall_snap) = self.wall_store.get(symbol) {
                                    if !wall_snap.is_warming_up {
                                        if is_long {
                                            let bid_walls = wall_snap.bid_walls();
                                            if let Some(wall) = bid_walls.iter().find(|w| {
                                                let dist = (last_trade.entry_price - w.price) / last_trade.entry_price;
                                                dist > 0.001 && dist < 0.015 && w.current_size_usd > 20_000.0
                                            }) {
                                                dynamic_sl = wall.price - atr_est * 0.1; // SL just behind wall
                                                info!("📈🧱 ScalpMTF [{}] SL behind bid wall {:.4}", symbol, wall.price);
                                            }
                                        } else {
                                            let ask_walls = wall_snap.ask_walls();
                                            if let Some(wall) = ask_walls.iter().find(|w| {
                                                let dist = (w.price - last_trade.entry_price) / last_trade.entry_price;
                                                dist > 0.001 && dist < 0.015 && w.current_size_usd > 20_000.0
                                            }) {
                                                dynamic_sl = wall.price + atr_est * 0.1;
                                                info!("📈🧱 ScalpMTF [{}] SL behind ask wall {:.4}", symbol, wall.price);
                                            }
                                        }
                                    }
                                }

                                // Clamp SL: min ATR×0.5, max ATR×1.5
                                let sl_dist = (last_trade.entry_price - dynamic_sl).abs();
                                if sl_dist < atr_est * 0.5 {
                                    dynamic_sl = if is_long {
                                        last_trade.entry_price - atr_est * 0.5
                                    } else {
                                        last_trade.entry_price + atr_est * 0.5
                                    };
                                } else if sl_dist > atr_est * 1.5 {
                                    dynamic_sl = if is_long {
                                        last_trade.entry_price - atr_est * 1.5
                                    } else {
                                        last_trade.entry_price + atr_est * 1.5
                                    };
                                }

                                let mut trade_final = trade_clone.clone();
                                trade_final.sl_price = dynamic_sl;
                                if let Some(dtp_val) = dtp {
                                    trade_final.tp_price = dtp_val;
                                }

                                info!("📈 ScalpMTF [{}] ENTER {} SL={:.4} (dynamic)",
                                    symbol, if is_long { "LONG" } else { "SHORT" }, dynamic_sl);

                                let config_clone = config.clone();
                                let ml_prob = 0.0;
                                let cvd = None;
                                self.execute_entry(symbol, &trade_final, &config_clone, ml_prob, cvd, None, false).await;

                                // Spawn ScalpMonitor for microstructure exit management
                                let ws = self.wall_store.clone();
                                let ts = self.tape_store.clone();
                                let sym = symbol.to_string();
                                let dir = if is_long { ScalpDirection::Long } else { ScalpDirection::Short };
                                let ep = last_trade.entry_price;
                                let router = self.order_router.clone();
                                let close_side = if is_long { Side::Sell } else { Side::Buy };
                                let _qty = trade_final.tp_price; // We store qty context

                                tokio::spawn(async move {
                                    let exit = scalp_mon::monitor_scalp(sym.clone(), dir, ep, ws, ts).await;
                                    let should_close = match &exit {
                                        scalp_mon::ScalpExit::DeltaReversal { held_secs, .. } => {
                                            info!("📈🔴 ScalpMonitor [{}] EXIT: delta reversal at {:.1}s", sym, held_secs);
                                            true
                                        }
                                        scalp_mon::ScalpExit::WallGone { held_secs, .. } => {
                                            info!("📈🧱 ScalpMonitor [{}] EXIT: wall gone at {:.1}s", sym, held_secs);
                                            true
                                        }
                                        scalp_mon::ScalpExit::IcebergAgainst { pressure, held_secs, .. } => {
                                            info!("📈🧊 ScalpMonitor [{}] EXIT: iceberg {:.2} at {:.1}s", sym, pressure, held_secs);
                                            true
                                        }
                                        scalp_mon::ScalpExit::WallTP { wall_price, held_secs, .. } => {
                                            info!("📈🎯 ScalpMonitor [{}] EXIT: wall-TP {:.4} at {:.1}s", sym, wall_price, held_secs);
                                            true
                                        }
                                        scalp_mon::ScalpExit::Timeout { .. } => {
                                            info!("📈⏱️ ScalpMonitor [{}] TIMEOUT → standard SL/TP", sym);
                                            false // Don't force close, let SL/TP handle it
                                        }
                                        scalp_mon::ScalpExit::NoData => false,
                                    };

                                    // Phase 10: Actually close position via order_router
                                    if should_close {
                                        let _ = router.cancel_all_orders(&sym).await;
                                        match router.market_order(&sym, close_side, 0.0, 0.0).await {
                                            Ok(_) => info!("📈✅ ScalpMonitor [{}] position CLOSED", sym),
                                            Err(e) => info!("📈❌ ScalpMonitor [{}] close failed: {}", sym, e),
                                        }
                                    }
                                });
                            }
                        } else if config.strategy == "fundingrate" {
                            // === FUNDING RATE: FR history + Sniper + Iceberg ===
                            let fr = self.funding_rates.get(symbol).copied().unwrap_or(0.0);
                            let fr_history = self.fr_history.get(symbol);
                            
                            // Check: 2+ consecutive extreme FRs
                            let fr_extreme = if let Some(hist) = fr_history {
                                if hist.len() >= 2 {
                                    let prev = hist[hist.len() - 2];
                                    let curr = hist[hist.len() - 1];
                                    if is_long {
                                        // LONG: FR should be negative (shorts overcrowded)
                                        prev < -0.0003 && curr < -0.0003
                                    } else {
                                        // SHORT: FR should be positive (longs overcrowded)
                                        prev > 0.0005 && curr > 0.0005
                                    }
                                } else { false }
                            } else { false };

                            if !fr_extreme {
                                info!("💰⏸️ FundingRate [{}] FR={:.6} — not extreme enough (need 2+ consecutive)",
                                    symbol, fr);
                            } else {
                                let (sniper_ok, dtp) = self.sniper_confirm(symbol, is_long, last_trade.entry_price).await;
                                if !sniper_ok {
                                    info!("💰❌ FundingRate sniper BLOCKED {} {} — walls/flow mismatch",
                                        if is_long { "LONG" } else { "SHORT" }, symbol);
                                } else {
                                    // Bonus: iceberg confirmation
                                    let mut bonus = 0u32;
                                    if let Some(tape) = self.tape_store.get(symbol) {
                                        let (ice_buy, ice_sell) = tape.iceberg_pressure();
                                        if is_long && ice_buy > 0.2 { bonus += 1; }
                                        else if !is_long && ice_sell > 0.2 { bonus += 1; }
                                    }

                                    info!("💰✅ FundingRate [{}] {} FR={:.6} (2+ extreme) sniper=✅ bonus={} — executing!",
                                        symbol, if is_long { "LONG" } else { "SHORT" }, fr, bonus);

                                    let mut trade_final = trade_clone.clone();
                                    if let Some(dtp_val) = dtp {
                                        trade_final.tp_price = dtp_val;
                                    }
                                    let config_clone = config.clone();
                                    let ml_prob = 0.0;
                                    let cvd = None;
                                    self.execute_entry(symbol, &trade_final, &config_clone, ml_prob, cvd, None, false).await;
                                }
                            }
                        } else {
                            // All other strategies: execute immediately
                            let config_clone = config.clone();
                            let ml_prob = 0.0;
                            let cvd = None;
                            self.execute_entry(symbol, &trade_clone, &config_clone, ml_prob, cvd, None, false).await;
                        }

                        self.last_signal_time.insert(symbol.to_string(), Instant::now());
                        break; // One signal per candle per symbol
                    }
                }
            }
        }
    }

    /// Run ML inference on the signal (Phase 10: Ensemble Voting)
    fn check_ml_filter(&self, config: &LiveConfig, candles: &[backtest::Candle], idx: usize, btc_trend: f64, btc_vol: f64, btc_dump: f64, funding_rate: f64) -> bool {
        let model_name = config.model_name();

        if let Some(ensemble) = self.ml_models.get(model_name) {
            let features = ml_inference::extract_features(candles, idx, btc_trend, btc_vol, btc_dump, funding_rate);
            
            let mut total_models = 0;
            let mut yes_votes = 0;

            // 1. Base LGBM (always present if ensemble loaded)
            total_models += 1;
            let p_lgbm = ensemble.lgbm.predict_proba(&features);
            if p_lgbm >= ML_THRESHOLD { yes_votes += 1; }

            // 2. Optional XGB
            let mut p_xgb = 0.0;
            if let Some(xgb) = &ensemble.xgb {
                total_models += 1;
                p_xgb = xgb.predict_proba(&features);
                if p_xgb >= ML_THRESHOLD { yes_votes += 1; }
            }

            // 3. Optional RF
            let mut p_rf = 0.0;
            if let Some(rf) = &ensemble.rf {
                total_models += 1;
                p_rf = rf.predict_proba(&features);
                if p_rf >= ML_THRESHOLD { yes_votes += 1; }
            }

            let approved = yes_votes * 2 > total_models; // > 50% majority (e.g. 2 out of 3, or 1 out of 1)
            
            info!("🧠 ML Ensemble '{}' | Votes: {}/{} | LGBM={:.0}% XGB={:.0}% RF={:.0}% | Result: {}",
                model_name, yes_votes, total_models, 
                p_lgbm * 100.0, p_xgb * 100.0, p_rf * 100.0,
                if approved { "✅ APPROVED" } else { "❌ REJECTED" });

            approved
        } else {
            // No ML ensemble loaded — trade without filter (risky but functional)
            warn!("⚠️ No ML ensemble for {}, allowing signal without filter", model_name);
            true
        }
    }

    /// Sniper Confirmation: validate entry against wall absorption + order flow
    /// Returns (should_enter, dynamic_tp_price)
    async fn sniper_confirm(&self, symbol: &str, is_long: bool, entry_price: f64) -> (bool, Option<f64>) {
        // Get wall snapshot for this symbol
        let wall_snap = match self.wall_store.get(symbol) {
            Some(snap) => snap.clone(),
            None => {
                // No wall data yet — pass through (don't block during warmup)
                info!("🎯 Sniper: no wall data for {} — auto-pass", symbol);
                return (true, None);
            }
        };

        // During warming up, don't block trades — just provide dynamic TP if possible
        if wall_snap.is_warming_up {
            let dtp = self.find_dynamic_tp(symbol, is_long, entry_price);
            return (true, dtp);
        }

        // Get order flow signal
        let flow_signal = match self.tape_store.get(symbol) {
            Some(state) => {
                // We need a snapshot without &mut — use the available read methods
                OrderFlowSignal {
                    delta: state.normalized_delta(),
                    cvd: state.cvd,
                    cvd_trend: state.cvd_trend(),
                    imbalance_ratio: state.imbalance_ratio(),
                    tape_speed: state.tape_speed(),
                    speed_acceleration: state.speed_acceleration(),
                    large_prints_buy: state.large_prints().0,
                    large_prints_sell: state.large_prints().1,
                    trade_count: state.trade_count(),
                    iceberg_buy_pressure: state.iceberg_pressure().0,
                    iceberg_sell_pressure: state.iceberg_pressure().1,
                    volume_zscore: state.volume_zscore(),
                    sweep_score: state.sweep_score().0,
                    sweep_direction_is_buy: state.sweep_score().1,
                    max_single_print_usd: state.max_print_usd(),
                }
            }
            None => {
                info!("🎯 Sniper: no tape data for {} — auto-pass", symbol);
                return (true, None);
            }
        };

        // Log flow state
        info!("🎯 Sniper [{}] delta={:.2} imb={:.1} speed={:.0}t/s accel={:.1}x whale_buy={} whale_sell={}",
            symbol, flow_signal.delta, flow_signal.imbalance_ratio,
            flow_signal.tape_speed, flow_signal.speed_acceleration,
            flow_signal.large_prints_buy, flow_signal.large_prints_sell);

        // Dynamic wall significance threshold (20% of the coin's macro wall size)
        let min_wall_usd = wall_snap.wall_threshold_usd * 0.2;
        info!("🎯 Dynamic wall threshold for {}: ${:.0} (base: ${:.0})", symbol, min_wall_usd, wall_snap.wall_threshold_usd);

        // Detect spoofer/support algorithms: fresh bid walls appearing near price (age < 5min)
        let spoofer_support = wall_snap.bid_walls().iter()
            .any(|w| w.age_secs() < 300 && (entry_price - w.price) / entry_price < 0.01 && w.current_size_usd > min_wall_usd);
        let spoofer_resistance = wall_snap.ask_walls().iter()
            .any(|w| w.age_secs() < 300 && (w.price - entry_price) / entry_price < 0.01 && w.current_size_usd > min_wall_usd);

        let confirmed = if is_long {
            // LONG: need ask walls being eaten + buyer dominance
            let ask_walls = wall_snap.ask_walls();
            let nearby_wall = ask_walls.iter()
                .find(|w| w.price > entry_price && (w.price - entry_price) / entry_price < 0.01 && w.current_size_usd > min_wall_usd);

            let mut block_reason = String::new();

            let wall_condition = match nearby_wall {
                Some(wall) => {
                    let eaten = wall.eaten_pct();
                    info!("🎯 Ask wall @ {:.4} (${:.0}): eaten {:.0}%, age {:.1}h, refresh {}x",
                        wall.price, wall.current_size_usd, eaten * 100.0, wall.age_hours(), wall.refresh_count);
                    if eaten > 0.30 {
                        true
                    } else {
                        block_reason.push_str(&format!("Wall {:.0}$ not eaten ({:.0}%). ", wall.current_size_usd, eaten*100.0));
                        false
                    }
                }
                None => {
                    info!("🎯 No significant ask wall (>${:.0}) near {} — clear path for LONG", min_wall_usd, entry_price);
                    true
                }
            };

            // Spoofer bonus: fresh bid walls below us = algorithms pushing price up
            if spoofer_support {
                info!("🎯 🤖 SPOOFER DETECTED: Fresh bid wall <5min below {} — algorithmic support!", entry_price);
            }

            let flow_condition = {
                // Dynamic: delta must be in the right direction
                let delta_ok = flow_signal.delta > 0.0;
                // Imbalance: buyers outweigh sellers (relative, not hardcoded)
                let imb_ok = flow_signal.imbalance_ratio > 1.2;
                // Whale activity: large prints detected  
                let whale_ok = flow_signal.large_prints_buy > 0;
                // Tape momentum: speed increasing
                let accel_ok = flow_signal.speed_acceleration > 1.2;
                // CVD trend: buying pressure growing
                let cvd_ok = flow_signal.cvd_trend > 0.0;
                // Relaxed: delta in right direction OR at least 2 other confirmations
                let confirms = imb_ok as u8 + whale_ok as u8 + accel_ok as u8 + cvd_ok as u8;
                info!("🎯 LONG flow: delta_ok={} imb_ok={} whale_ok={} accel_ok={} cvd_ok={} confirms={}",
                    delta_ok, imb_ok, whale_ok, accel_ok, cvd_ok, confirms);
                
                let ok = delta_ok || confirms >= 2;
                if !ok {
                    block_reason.push_str(&format!("Flow weak (delta={:.2}, confs={}). ", flow_signal.delta, confirms));
                }
                ok
            };

            if !wall_condition || !flow_condition {
                info!("🎯 ❌ Sniper REJECTED LONG {}: {}", symbol, block_reason);
                false
            } else { true }
        } else {
            // SHORT: need bid walls being eaten + seller dominance
            let bid_walls = wall_snap.bid_walls();
            let nearby_wall = bid_walls.iter()
                .find(|w| w.price < entry_price && (entry_price - w.price) / entry_price < 0.01 && w.current_size_usd > min_wall_usd);

            let mut block_reason = String::new();

            let wall_condition = match nearby_wall {
                Some(wall) => {
                    let eaten = wall.eaten_pct();
                    info!("🎯 Bid wall @ {:.4} (${:.0}): eaten {:.0}%, age {:.1}h, refresh {}x",
                        wall.price, wall.current_size_usd, eaten * 100.0, wall.age_hours(), wall.refresh_count);
                    if eaten > 0.30 {
                        true
                    } else {
                        block_reason.push_str(&format!("Wall {:.0}$ not eaten ({:.0}%). ", wall.current_size_usd, eaten*100.0));
                        false
                    }
                }
                None => {
                    info!("🎯 No significant bid wall (>${:.0}) near {} — clear path for SHORT", min_wall_usd, entry_price);
                    true
                }
            };

            // Spoofer bonus: fresh ask walls above us = algorithms pushing price down
            if spoofer_resistance {
                info!("🎯 🤖 SPOOFER DETECTED: Fresh ask wall <5min above {} — algorithmic resistance!", entry_price);
            }

            let flow_condition = {
                let delta_ok = flow_signal.delta < 0.0;
                let imb_ok = flow_signal.imbalance_ratio < (1.0 / 1.2); // sellers > 1.2x buyers
                let whale_ok = flow_signal.large_prints_sell > 0;
                let accel_ok = flow_signal.speed_acceleration > 1.2;
                let cvd_ok = flow_signal.cvd_trend < 0.0;
                // Relaxed: delta in right direction OR at least 2 other confirmations
                let confirms = imb_ok as u8 + whale_ok as u8 + accel_ok as u8 + cvd_ok as u8;
                info!("🎯 SHORT flow: delta_ok={} imb_ok={} whale_ok={} accel_ok={} cvd_ok={} confirms={}",
                    delta_ok, imb_ok, whale_ok, accel_ok, cvd_ok, confirms);
                
                let ok = delta_ok || confirms >= 2;
                if !ok {
                    block_reason.push_str(&format!("Flow weak (delta={:.2}, confs={}). ", flow_signal.delta, confirms));
                }
                ok
            };

            if !wall_condition || !flow_condition {
                info!("🎯 ❌ Sniper REJECTED SHORT {}: {}", symbol, block_reason);
                false
            } else { true }
        };

        if !confirmed {
            let dtp = self.find_dynamic_tp(symbol, is_long, entry_price);
            return (false, dtp);
        }

        // === Phase 11: Spot Probe — check physical Spot market ===
        let spot_env = spot_probe::probe_spot_depth(
            &self.http_client, symbol, entry_price, is_long
        ).await;

        let final_confirmed = match &spot_env {
            Some(env) if env.hidden_barrier => {
                // Hidden Spot barrier contradicts our direction → BLOCK
                info!("🔍 ❌ SpotProbe BLOCKED {} {} — hidden {:?} barrier on Spot",
                    if is_long { "LONG" } else { "SHORT" }, symbol, env.barrier_side);
                false
            }
            Some(env) => {
                let spot_confirms = if is_long { env.confirms_long } else { env.confirms_short };
                if spot_confirms {
                    info!("🔍 ✅ SpotProbe CONFIRMED {} {} — Spot wall supports direction",
                        if is_long { "LONG" } else { "SHORT" }, symbol);
                } else {
                    info!("🔍 🟡 SpotProbe NEUTRAL {} {} — no Spot anomaly (passing)",
                        if is_long { "LONG" } else { "SHORT" }, symbol);
                }
                true  // Pass through (confirmed or neutral)
            }
            None => {
                // Spot probe failed (timeout/error) — don't block, pass through
                info!("🔍 🟡 SpotProbe UNAVAILABLE {} — passing without Spot check", symbol);
                true
            }
        };

        if final_confirmed {
            info!("🎯 ✅ Sniper CONFIRMED {} {}", if is_long { "LONG" } else { "SHORT" }, symbol);
        }

        let dtp = self.find_dynamic_tp(symbol, is_long, entry_price);
        (final_confirmed, dtp)
    }

    /// Phase 2: Compute HFT metrics for SMC logic
    fn collect_hft_metrics(&self, symbol: &str, price: f64) -> strategies::smc::HftMetrics {
        let mut metrics = strategies::smc::HftMetrics::default();

        // 1. Wall Presence
        if let Some(snap) = self.wall_store.get(symbol) {
            let threshold = snap.wall_threshold_usd * 0.5;
            let walls = snap.nearby_walls(price, 0.005); // within 0.5%
            let max_wall = walls.iter()
                .filter(|w| w.current_size_usd > threshold)
                .map(|w| w.current_size_usd)
                .fold(0.0, f64::max);
            
            if max_wall > 0.0 {
                metrics.nearby_wall_presence = (max_wall / (threshold * 5.0)).min(1.0);
            }
        }

        // 2. Tape Momentum (CVD)
        if let Some(tape) = self.tape_store.get(symbol) {
            metrics.tape_delta_momentum = tape.normalized_delta();
        }

        // 3. Liquidation Sweep
        if let Some(cascade) = self.liq_store.get(symbol).or_else(|| self.liq_store.get("__global__")) {
            metrics.liquidation_proximity = if cascade.is_cascade || cascade.knife_prepare_active { 1.0 } else { 0.0 };
        }

        // 4. Order Book Imbalance
        if let Some(book) = self.ob_store.get(symbol) {
            metrics.order_book_imbalance = book.imbalance();
        }

        metrics
    }

    /// Find a dynamic take-profit target based on the next wall
    fn find_dynamic_tp(&self, symbol: &str, is_long: bool, entry_price: f64) -> Option<f64> {
        let snap = self.wall_store.get(symbol)?;

        if is_long {
            // Find next ask wall above entry (at least 0.3% away)
            let walls = snap.ask_walls();
            for wall in &walls {
                let dist_pct = (wall.price - entry_price) / entry_price;
                if dist_pct > 0.003 && wall.current_size_usd > 30_000.0 {
                    // Set TP slightly before the wall (90% of the way)
                    let tp = entry_price + (wall.price - entry_price) * 0.90;
                    info!("🎯 Dynamic TP LONG {}: wall @ {:.4} (${:.0}k) → TP = {:.4}",
                        symbol, wall.price, wall.current_size_usd / 1000.0, tp);
                    return Some(tp);
                }
            }
        } else {
            // Find next bid wall below entry
            let walls = snap.bid_walls();
            for wall in &walls {
                let dist_pct = (entry_price - wall.price) / entry_price;
                if dist_pct > 0.003 && wall.current_size_usd > 30_000.0 {
                    let tp = entry_price - (entry_price - wall.price) * 0.90;
                    info!("🎯 Dynamic TP SHORT {}: wall @ {:.4} (${:.0}k) → TP = {:.4}",
                        symbol, wall.price, wall.current_size_usd / 1000.0, tp);
                    return Some(tp);
                }
            }
        }

        None // No suitable wall found, keep original TP
    }

    /// Update S/R levels for a symbol when new candle data arrives
    fn update_levels_for_symbol(&self, symbol: &str) {
        if let Some(buffer) = self.candle_buffers.get(symbol) {
            level_tracker::update_levels(&self.level_store, symbol, buffer);
        }
    }

    /// Process a closed 15m candle — update MTF EMA trend (no strategy execution)
    pub fn on_15m_candle_close(&mut self, symbol: &str, candle: &ws_feed::Candle) {
        // Append close price to 15m buffer
        let buffer = self.mtf_15m_buffers
            .entry(symbol.to_string())
            .or_insert_with(|| Vec::with_capacity(MTF_15M_BUFFER_SIZE + 10));

        buffer.push(candle.close);

        // Trim to last N
        if buffer.len() > MTF_15M_BUFFER_SIZE {
            buffer.drain(..buffer.len() - MTF_15M_BUFFER_SIZE);
        }

        // Need enough data for EMA
        if buffer.len() < MTF_EMA_PERIOD {
            self.mtf_trend_store.insert(symbol.to_string(), MtfTrend::Neutral);
            return;
        }

        // Calculate EMA(20) on 15m closes
        let ema = Self::calc_ema(buffer, MTF_EMA_PERIOD);
        let current_price = candle.close;

        let trend = if current_price > ema {
            MtfTrend::Bullish
        } else {
            MtfTrend::Bearish
        };

        let prev = self.mtf_trend_store.insert(symbol.to_string(), trend);
        if prev.as_ref() != Some(&trend) {
            info!("📐 MTF [{}] 15m EMA({}) = {:.4} | Price = {:.4} → {:?}",
                symbol, MTF_EMA_PERIOD, ema, current_price, trend);
        }
    }

    /// Simple EMA calculation over a slice of values
    fn calc_ema(values: &[f64], period: usize) -> f64 {
        if values.len() < period {
            return values.last().copied().unwrap_or(0.0);
        }
        let k = 2.0 / (period as f64 + 1.0);
        // Seed with SMA of first `period` values
        let sma: f64 = values[..period].iter().sum::<f64>() / period as f64;
        let mut ema = sma;
        for &val in &values[period..] {
            ema = val * k + ema * (1.0 - k);
        }
        ema
    }

    /// Calculate Meta-Model score for risk modulation (Phase 11.4)
    fn calculate_meta_score(
        &self,
        symbol: &str,
        strategy_name: &str,
        entry_price: f64,
        risk_dist: f64,
    ) -> f64 {
        let info = match &self.meta_info {
            Some(i) => i,
            None => return 100.0, // No model yet
        };

        let model = match &self.meta_model {
            Some(m) => m,
            None => return 100.0,
        };

        // Rule: ignore if train size < 50
        if info.train_size < 50 {
            return 100.0;
        }

        // Strategy encoding
        let strategy_enc = *info.strategy_map.get(strategy_name).unwrap_or(&0) as f64;

        // Spot probe encoding (placeholder for now)
        let spot_probe_enc = 1.0; // neutral

        // Tape features
        let (cvd_d, imb_r, t_speed) = self.tape_store.get(symbol)
            .map(|s| (s.normalized_delta(), s.imbalance_ratio(), s.tape_speed()))
            .unwrap_or((0.0, 1.0, 0.0));

        // Wall features (pick the largest wall)
        let (w_size, w_age, w_eaten) = self.wall_store.get(symbol)
            .and_then(|snap| {
                snap.walls.iter()
                    .max_by(|a, b| a.current_size_usd.partial_cmp(&b.current_size_usd).unwrap())
                    .map(|w| (w.current_size_usd, w.age_hours(), w.eaten_pct()))
            })
            .unwrap_or((0.0, 0.0, 0.0));

        // Feature vector matching aggregate_journal.py:
        // ["strategy_enc", "spot_probe_enc", "wall_size_usd", "wall_age_h", "wall_eaten_pct", "cvd_delta", "imbalance_ratio", "tape_speed", "entry_price", "risk_dist"]
        let features = vec![
            strategy_enc,
            spot_probe_enc,
            w_size,
            w_age,
            w_eaten,
            cvd_d,
            imb_r,
            t_speed,
            entry_price,
            risk_dist,
        ];

        // Probability of success * 100
        model.predict_proba(&features) * 100.0
    }

    /// Execute a trade entry
    async fn execute_entry(
        &mut self,
        symbol: &str,
        trade: &backtest::Trade,
        config: &LiveConfig,
        ml_prob: f64,
        cvd_delta: Option<f64>,
        target_wall_price: Option<f64>,
        is_wall_backed: bool,
    ) {
        let is_long = trade.direction == "LONG";

        // === FIX 2: Imbalance Ratio Entry Filter ===
        // Reject entries with weak order-book imbalance (< 1.3)
        // But for KnifeTick, we bypass this macro block because the HFT Absorber 
        // natively handles tick-level tape speed and imbalance in real time.
        if config.rust_strategy_name() != "knifetick" {
            if let Some(tape) = self.tape_store.get(symbol) {
                let imb = tape.imbalance_ratio();
                if imb < 1.3 {
                    info!("📊 IMBALANCE BLOCKED: {} imb_ratio={:.2} < 1.3 — skipping entry", symbol, imb);
                    return;
                }
            }
        }

        // === Phase 11.4: Meta-Inference & Risk Modulation ===
        let meta_score = self.calculate_meta_score(&symbol, &config.strategy, trade.entry_price, trade.risk_dist);
        let volume_multiplier = if meta_score > 80.0 {
            info!("🛡️ Meta-Score: {:.1} → GREEN (100% volume)", meta_score);
            1.0 // Green
        } else if meta_score >= 40.0 {
            info!("🛡️ Meta-Score: {:.1} → YELLOW (50% volume WARN)", meta_score);
            0.5 // Yellow
        } else {
            info!("🛡️ Meta-Score: {:.1} → RED (Micro-lot 0.01x for data collection)", meta_score);
            0.01 // Red (Micro-lot)
        };

        // Pre-calculate volume reduction based on Meta-Score
        // (Final calculation happens after risk_dist is determined)

        let side = if is_long { Side::Buy } else { Side::Sell };
        let current_price = trade.entry_price;
        let mut final_sl = trade.sl_price;

        // === Phase 31C: SL BEHIND GRID BOTTOM (density zone / wall) ===
        // For knife_tick with grid: SL goes 0.1% BEHIND the grid bottom (cascade/wall)
        // This hides the stop behind physical support. To get stopped out,
        // price must break through the ENTIRE wall — which is the invalidation point.
        //
        // BUG FIX Phase 36: Clamp grid_sl to max 2× the DE-optimized SL distance.
        // Without this, a far wall (e.g. -2%) overwrites a tight DE SL (0.3-0.5%),
        // causing live to risk 4-6× more per trade than the backtester assumed.
        if config.strategy.to_lowercase().contains("knife") {
            if let Some(wall) = target_wall_price {
                let de_sl_dist = (current_price - trade.sl_price).abs();
                let max_sl_dist = de_sl_dist * 2.0; // never more than 2× DE SL
                
                if is_long {
                    let grid_sl = wall * 0.999; // 0.1% below grid bottom
                    let grid_dist = (current_price - grid_sl).abs();
                    let clamped_sl = if grid_dist > max_sl_dist {
                        let clamped = current_price - max_sl_dist;
                        info!("🔪🛡️ SL BEHIND GRID CLAMPED: {} LONG — grid_sl={:.4} too far ({:.2}%), clamped to {:.4} ({:.2}%)",
                            symbol, grid_sl, grid_dist/current_price*100.0, clamped, max_sl_dist/current_price*100.0);
                        clamped
                    } else {
                        info!("🔪🛡️ SL BEHIND GRID: {} LONG — grid_bottom={:.4}, SL={:.4} (was {:.4})",
                            symbol, wall, grid_sl, final_sl);
                        grid_sl
                    };
                    final_sl = clamped_sl;
                } else {
                    let grid_sl = wall * 1.001; // 0.1% above grid top
                    let grid_dist = (grid_sl - current_price).abs();
                    let clamped_sl = if grid_dist > max_sl_dist {
                        let clamped = current_price + max_sl_dist;
                        info!("🔪🛡️ SL BEHIND GRID CLAMPED: {} SHORT — grid_sl={:.4} too far ({:.2}%), clamped to {:.4} ({:.2}%)",
                            symbol, grid_sl, grid_dist/current_price*100.0, clamped, max_sl_dist/current_price*100.0);
                        clamped
                    } else {
                        info!("🔪🛡️ SL BEHIND GRID: {} SHORT — grid_top={:.4}, SL={:.4} (was {:.4})",
                            symbol, wall, grid_sl, final_sl);
                        grid_sl
                    };
                    final_sl = clamped_sl;
                }
            }
        }

        // === RISK GATE 0: Coin Blacklist (Phase 11.4) ===
        if is_coin_blacklisted(symbol) {
            info!("🚫 BLACKLISTED: {} — skipping trade (cooldown active)", symbol);
            return;
        }

        // === RISK GATE 0.1: Per-symbol loss cooldown (Phase 30) ===
        // If 2+ losses on this symbol in the last hour → skip
        if let Some(loss_times) = self.symbol_loss_times.get(symbol) {
            let recent_losses = loss_times.iter().filter(|t| t.elapsed().as_secs() < 3600).count();
            if recent_losses >= 2 {
                info!("🧊 COOLDOWN: {} has {} losses in last hour — skipping", symbol, recent_losses);
                return;
            }
        }

        // === RISK GATE 0.5: Liquidation Cascade (Phase 13) ===
        if let Some(cascade) = self.liq_store.get("__global__") {
            let is_knife = config.strategy.contains("Knife");
            // During ACTIVE cascade: block everything
            if cascade.is_cascade {
                info!("🌊 CASCADE ACTIVE (Z={:.1}) — blocking {} entry for {}",
                    cascade.velocity_zscore, config.strategy, symbol);
                return;
            }
            // After cascade: trending strategies get 5min cooldown
            if cascade.trending_warn_active && !is_knife {
                info!("🌊 CASCADE WARN — {} blocked for {} (5min cooldown)",
                    config.strategy, symbol);
                return;
            }
            // After cascade: KnifeCatcher gets PREPARE bonus (logged only)
            if cascade.knife_prepare_active && is_knife {
                info!("🌊 KNIFE PREPARE — {} gets +0.15 ML bonus for {} (post-cascade sweep)",
                    config.strategy, symbol);
                // Bonus is applied in sniper_confirm (ML prob adjustment)
            }
        }

        // === RISK GATE 0.7: Whale Flow (Phase 12) ===
        let whale_tag = self.whale_detector.get_tag(symbol);
        let is_long = trade.direction == "LONG";
        if whale_tag.blocks_long() && is_long {
            info!("🐋 WHALE SELL active on {} — blocking LONG entry", symbol);
            return;
        }
        if whale_tag.blocks_short() && !is_long {
            info!("🐋 WHALE BUY active on {} — blocking SHORT entry", symbol);
            return;
        }

        // === RISK GATE 1: Daily loss limit ===
        // Reset daily counters at 00:00 UTC
        let now_day = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() / 86400;
        if now_day != self.last_reset_day {
            if self.daily_pnl_r != 0.0 {
                info!("📅 Daily reset: PnL={:.2}R, trades={}", self.daily_pnl_r, self.daily_trade_count);
            }
            self.daily_pnl_r = 0.0;
            self.daily_trade_count = 0;
            self.last_reset_day = now_day;
            // Clear position locks for closed positions
            self.position_locks.retain(|sym, _| {
                self.position_manager.has_position(sym)
            });
        }

        if self.daily_pnl_r <= -DAILY_LOSS_LIMIT_R {
            warn!("🛑 DAILY LOSS LIMIT HIT: {:.2}R. No more trades today!", self.daily_pnl_r);
            return;
        }

        // === RISK GATE 2: Symbol position lock ===
        if let Some(locked_dir) = self.position_locks.get(symbol) {
            if (locked_dir == "LONG" && !is_long) || (locked_dir == "SHORT" && is_long) {
                warn!("🔒 Symbol lock: {} already has {} position → blocking {} entry",
                    symbol, locked_dir, trade.direction);
                return;
            }
        }

        // Position sizing: fixed fractional (2% risk)
        let equity = match self.order_router.get_equity().await {
            Ok(eq) => {
                // Update equity tracking
                self.current_equity = eq;
                if eq > self.equity_peak {
                    self.equity_peak = eq;
                }
                eq
            },
            Err(e) => {
                warn!("⚠️ Could not get equity: {}. Using $2000 default.", e);
                2000.0
            }
        };

        // === RISK GATE 3: Equity drawdown → reduce position size ===
        let drawdown_pct = if self.equity_peak > 0.0 {
            (self.equity_peak - self.current_equity) / self.equity_peak
        } else { 0.0 };

        let risk_pct = if drawdown_pct > EQUITY_TRAIL_PCT {
            let reduced = 0.02 * (1.0 - drawdown_pct); // Scale down risk
            info!("📉 Equity drawdown {:.1}% → risk reduced to {:.2}%", drawdown_pct * 100.0, reduced * 100.0);
            reduced.max(0.005) // Minimum 0.5% risk
        } else {
            0.02 // Normal 2% risk
        };

        let ja_mult = self.journal_analyzer.get_risk_multiplier(&config.strategy);
        let risk_amount = equity * risk_pct * ja_mult;
        let mut risk_dist = (trade.entry_price - final_sl).abs();

        // DEBUG: Track risk_dist discrepancies (Phase 30 bugfix)
        info!("🔍 risk_dist DEBUG [{}]: entry={:.6} final_sl={:.6} trade.sl={:.6} risk_dist={:.6} trade.risk_dist={:.6}",
            symbol, trade.entry_price, final_sl, trade.sl_price, risk_dist, trade.risk_dist);

        if risk_dist <= 0.0 {
            warn!("⚠️ Zero risk distance for {}, skipping", symbol);
            return;
        }

        // Minimum risk_dist guard: prevent R-multiple inflation
        // when dynamic SL (wall-based) places stop too close to entry.
        // Also moves the actual SL order to match the clamped distance.
        let min_risk_dist = trade.entry_price * 0.0015; // FIX D: 0.15% safety net
        
        // Phase 37: HARD SKIP for knife_tick when SL is impossibly tight.
        // NOM-style coins: risk_dist=0.036% → SL is 1 tick away → any move = -1.3R blowout.
        // The clamp below only fixes sizing but the SL stays tight → still get blown through.
        // Solution: don't enter at all if the spread/SL geometry is this broken.
        if config.strategy.to_lowercase().contains("knife") && risk_dist < min_risk_dist {
            let pct = risk_dist / trade.entry_price * 100.0;
            warn!("🚫 SKIP: {} risk_dist={:.6} ({:.3}%) < min {:.3}% — SL too tight for knife entry",
                symbol, risk_dist, pct, 0.15);
            return;
        }
        
        if risk_dist < min_risk_dist {
            warn!("⚠️ risk_dist {:.6} too small for {} (min={:.6}), clamping SL", risk_dist, symbol, min_risk_dist);
            risk_dist = min_risk_dist;
            
            // CRITICAL FIX: Only move the actual SL order if it's NOT a knife catcher.
            // Knife catcher MUST keep SL directly behind the wall, 
            // but we use the clamped risk_dist to safely size the position without inflation.
            if !config.strategy.to_lowercase().contains("knife") {
                if is_long {
                    final_sl = current_price - risk_dist;
                } else {
                    final_sl = current_price + risk_dist;
                }
                info!("   → final_sl adjusted to {:.6} ({}% from entry)", final_sl, risk_dist / current_price * 100.0);
            } else {
                info!("   → Keeping final_sl at {:.6} behind wall, but risk clamped to {:.6} for safety", final_sl, risk_dist);
            }
        }

        let mut quantity = risk_amount / risk_dist;
        quantity *= volume_multiplier;

        // Ensure micro-lot doesn't drop to 0
        if volume_multiplier > 0.0 && quantity < 0.001 {
            quantity = 0.001;
        }

        info!("💰 Sizing: equity=${:.0}, risk=${:.0} ({:.1}%, Advisor={:.1}x), multiplier={:.1}x, dist={:.4}, qty={:.6}",
            equity, risk_amount, risk_pct * 100.0, ja_mult, volume_multiplier, risk_dist, quantity);

        // Phase 31C: Grid Scaling Logic (50 Limit Orders, NO market order)
        // target_wall_price is ALWAYS Some() now (cascade/wall/default spread)
        let mut initial_size = quantity;
        let mut pending_grid = Vec::new();
        let mut place_grid_orders = false;

        if let Some(wall) = target_wall_price {
            if config.rust_strategy_name() == "knifetick" {
                let steps = 50;
                // Phase 31C: 3 ANOMALOUS fat orders at the bottom get 70%, 
                // 47 thin probes get 30% (each ~$6 at typical sizing)
                // Fat orders are ~18x bigger than thin — catches the wall bounce with massive size
                let fat_count = 3;
                let fat_total = quantity * 0.70;
                let fat_each = fat_total / fat_count as f64;
                let thin_total = quantity * 0.30;
                let thin_each = thin_total / (steps - fat_count) as f64;
                
                // First limit order at current price (replaces market Step 0)
                initial_size = thin_each;
                
                let dist = (current_price - wall).abs();
                let price_step = dist / (steps as f64 - 1.0);
                
                for i in 1..steps {
                    let level_price = if is_long {
                        current_price - price_step * (i as f64)
                    } else {
                        current_price + price_step * (i as f64)
                    };
                    let level_size = if i >= steps - fat_count { fat_each } else { thin_each };
                    pending_grid.push((level_price, level_size));
                }
                place_grid_orders = true;
                info!("🔪🧲 GRID 50: {} — 47 thin ({:.4}) + 3 fat ({:.4}) → bottom {:.4}", 
                    symbol, thin_each, fat_each, wall);
            }
        }

        // Place Step 0 as LIMIT order at current price (maker rebate!)
        match self.order_router.limit_order(symbol, side, initial_size, current_price, true).await {
            Ok(result) => {
                let status = result.status.as_deref().unwrap_or("POSTED");
                info!("✅ LIMIT POSTED (Step 0): {} {} {} @ {:.4} | Status: {}",
                    trade.direction, symbol, initial_size, current_price, status);

                // Dispatch the rest of the 49 limit orders to Binance asynchronously
                if place_grid_orders {
                    let router = self.order_router.clone();
                    let sym = symbol.to_string();
                    let grid_snapshot = pending_grid.clone();
                    tokio::spawn(async move {
                        for (p, s) in grid_snapshot {
                            let _ = router.limit_order(&sym, side, s, p, false).await;
                        }
                    });
                }

                // Register position
                let mode = self.active_modes.get(&format!("{}_{}", symbol, config.strategy))
                    .map(|s| s.as_str())
                    .unwrap_or("conservative");
                let params = config.params_vec(mode);
                let be_trigger = if params.len() > 5 { params[5] } else { 0.0 };
                let mut trail = if params.len() > 6 { params[6] } else { 0.0 };
                
                // Phase 31C: Enforce minimum trail 0.4% for knife_tick
                // 0.1% trail = exit on noise. Alt shakes ±0.2% every second.
                if config.strategy.to_lowercase().contains("knife") {
                    trail = trail.max(0.004); // 0.4% minimum
                    info!("🔪📐 Trail enforced to {:.2}% for knife_tick", trail * 100.0);
                }
                
                // Fetch current CVD to baseline the RL trajectory
                let squeeze_cvd = self.tape_store.get(symbol).map(|t| t.cvd).unwrap_or(0.0);
                
                let position = Position {
                    symbol: symbol.to_string(),
                    direction: if is_long { Direction::Long } else { Direction::Short },
                    entry_price: current_price,
                    size: initial_size,
                    target_size: quantity,
                    sl_price: final_sl,
                    tp_price: Some(trade.tp_price),
                    strategy: config.strategy.clone(),
                    risk_dist,
                    trail_activate_r: 1.0,
                    trail_atr_mult: 0.5,
                    be_trigger_pct: be_trigger,
                    trail_pct: trail,
                    is_breakeven: false,
                    is_wall_backed,
                    trailing: TrailingState {
                        active: false,
                        best_price: current_price,
                        current_sl: final_sl,
                        last_candle_update: Instant::now(),
                    },
                    partially_exited: false,
                    open_time: Instant::now(),
                    observation: ObservationWindow::new(), // Phase 30B
                    pending_grid, // Phase 29C+1 internal grid tracker
                    // Phase 36: RL
                    squeeze_cvd,
                    local_extreme: current_price,
                };
                info!("📐 Position params: BE={:.4}% trail={:.4}% cvd_base={:.1}", be_trigger * 100.0, trail * 100.0, squeeze_cvd);
                // Phase 30B: Enforce minimum TP distance so commission never eats profit
                let mut position = position;
                position.enforce_min_tp();
                self.position_manager.open(position);
                
                // Phase 14: Send Telegram Entry Alert
                if let Some(tx) = &self.tg_tx {
                    use crate::live::telegram_bot::escape_variable as esc;
                    
                    let dir_icon = if is_long { "🟢" } else { "🔴" };
                    let dir_text = if is_long { "LONG" } else { "SHORT" };
                    let conf_pct = (ml_prob * 100.0) as i32;
                    
                    let msg = format!(
                        "{} *ENTRY {}* \\| `#{}`\n\
                        ⚙️ Strategy: {}\n\
                        💵 Price: ${}\n\
                        🎯 Target: ${} \\| 🛑 Stop: ${}\n\
                        🧠 Confidence: {}% / {}",
                        dir_icon, dir_text, esc(symbol),
                        esc(&config.strategy),
                        esc(&format!("{:.4}", current_price)),
                        esc(&format!("{:.4}", trade.tp_price)),
                        esc(&format!("{:.4}", final_sl)),
                        conf_pct,
                        esc(&format!("{:.1}", cvd_delta.unwrap_or(0.0)))
                    );
                    
                    let _ = tx.try_send(msg);
                }

                // Phase 10: Register position lock + daily counter
                self.position_locks.insert(symbol.to_string(), trade.direction.clone());
                self.daily_trade_count += 1;
                info!("🔒 Symbol lock: {} → {} | Daily trades: {}", 
                    symbol, trade.direction, self.daily_trade_count);

                // Phase 14: Update Live Stats open positions
                {
                    let mut stats = self.live_stats.lock().await;
                    stats.open_positions.push(crate::live::live_stats::OpenPosInfo {
                        symbol: symbol.to_string(),
                        direction: trade.direction.clone(),
                        entry_price: current_price,
                        current_price,
                        pnl_pct: 0.0,
                        duration_secs: 0,
                        strategy: config.strategy.clone(),
                        size: initial_size, // Fix E0382: position was already moved
                        target_size: quantity,
                    });
                }

                // Phase 11: Trade Logger — ENTRY event
                let trade_id = trade_logger::generate_trade_id(symbol);
                self.trade_ids.insert(symbol.to_string(), trade_id.clone());

                // Grab spot probe status and flow metrics
                let spot_status = "neutral".to_string(); // Will be enriched when we pass it through
                let (cvd_d, imb_r, t_speed) = self.tape_store.get(symbol)
                    .map(|s| (Some(s.normalized_delta()), Some(s.imbalance_ratio()), Some(s.tape_speed())))
                    .unwrap_or((None, None, None));
                let (w_side, w_price, w_size, w_age, w_eaten) = self.wall_store.get(symbol)
                    .and_then(|snap| {
                        snap.walls.iter()
                            .max_by(|a, b| a.current_size_usd.partial_cmp(&b.current_size_usd).unwrap())
                            .map(|w| (
                                Some(format!("{:?}", w.side)),
                                Some(w.price),
                                Some(w.current_size_usd),
                                Some(w.age_hours()),
                                Some(w.eaten_pct()),
                            ))
                    })
                    .unwrap_or((None, None, None, None, None));

                trade_logger::log_entry(&trade_logger::TradeEntry {
                    event: "ENTRY",
                    ts: trade_logger::now_iso(),
                    trade_id,
                    symbol: symbol.to_string(),
                    strategy: config.strategy.clone(),
                    direction: trade.direction.clone(),
                    entry_price: current_price,
                    sl_price: final_sl,  // FIX: log actual SL (after wall-widening), not macro SL
                    tp_price: trade.tp_price,
                    quantity,
                    risk_dist,
                    ml_prob: None, // TODO: pass from ML inference
                    spot_probe: spot_status,
                    wall_side: w_side,
                    wall_price: w_price,
                    wall_size_usd: w_size,
                    wall_age_h: w_age,
                    wall_eaten_pct: w_eaten,
                    cvd_delta: cvd_d,
                    imbalance_ratio: imb_r,
                    tape_speed: t_speed,
                    whale_tag: {
                        let wt = self.whale_detector.get_tag(symbol);
                        if wt != whale_detector::WhaleTag::Neutral {
                            Some(wt.as_str().to_string())
                        } else { None }
                    },
                    liq_zscore: None,
                    correlation_warn: None,
                    meta_warn_score: Some(meta_score),
                });
                // Place stop-loss order
                let close_side = if is_long { Side::Sell } else { Side::Buy };
                if let Err(e) = self.order_router.stop_market_order(
                    symbol, close_side, quantity, final_sl
                ).await {
                    error!("❌ Failed to place SL for {}: {}", symbol, e);
                }

                // Phase 31.1: Place LIMIT Take Profit order (Maker Rebate)
                if let Err(e) = self.order_router.limit_order(
                    symbol, close_side, quantity, trade.tp_price, true // post_only = true
                ).await {
                    error!("❌ Failed to place LIMIT TP for {}: {}", symbol, e);
                }
            }
            Err(e) => {
                error!("❌ ORDER FAILED for {}: {}", symbol, e);
            }
        }
    }

    /// Process a tick update (for position monitoring and smart trailing)
    pub async fn on_tick(
        &mut self,
        symbol: &str,
        price: f64,
        ob_store: &OrderBookStore,
        tape_store: &TapeStore,
    ) {
        // Phase 30C: Check pending confirmations BEFORE position check
        self.check_pending_confirmation(symbol, price).await;

        // Check if we have a position on this symbol
        if !self.position_manager.has_position(symbol) {
            return;
        }

        // Phase 29C+1: Spoofing Protection — Check if wall vanished
        let mut clear_grid = false;
        if let Some(pos) = self.position_manager.positions.get(symbol) {
            // Only apply spoof protection if the grid was actually placed on a real wall
            if pos.is_wall_backed && pos.size < pos.target_size && !pos.pending_grid.is_empty() {
                if let Some(wall_price) = pos.pending_grid.last().map(|p| p.0) {
                    if let Some(wall_snap) = self.wall_store.get(symbol) {
                        let min_wall_usd = wall_snap.wall_threshold_usd * 0.2;
                        let walls = if pos.direction == Direction::Long { wall_snap.bid_walls() } else { wall_snap.ask_walls() };
                        let is_wall_present = walls.iter().any(|w| (w.price - wall_price).abs() / wall_price < 0.002 && w.current_size_usd >= min_wall_usd);
                        
                        if !is_wall_present {
                            info!("🧊💥 SPOOF DETECTED: Wall at {:.4} vanished for {}! Aborting remaining Grid limit orders.", wall_price, symbol);
                            clear_grid = true;
                        }
                    }
                }
            }
        }

        if clear_grid {
            if let Some(pos) = self.position_manager.positions.get_mut(symbol) {
                pos.pending_grid.clear();
                let sym = symbol.to_string();
                let router = self.order_router.clone();
                let qty = pos.size;
                let close_side = if pos.direction == Direction::Long { Side::Sell } else { Side::Buy };
                let sl = pos.sl_price;
                let tp = pos.tp_price;
                tokio::spawn(async move {
                    let _ = router.cancel_all_orders(&sym).await; // Cancel vanished grid limits
                    let _ = router.stop_market_order(&sym, close_side, qty, sl).await;
                    if let Some(tp_val) = tp {
                        let _ = router.limit_order(&sym, close_side, qty, tp_val, true).await;
                    }
                });
            }
        }

        // Tick-level BE + trailing
        // Phase 30: knife_tick uses Smart Trail (delta-adaptive), others use static trail
        if let Some(pos) = self.position_manager.positions.get_mut(symbol) {
            let is_knife = pos.strategy == "knife_tick";
            
            let (updated, early_cut) = if is_knife {
                // Phase 36: Integrate RL Agent continuous tracking!
                let trail_updated = pos.update_trail_tick(price);
                let mut rl_early_cut = false;
                
                // 1. Update Extreme
                if pos.direction == Direction::Long {
                    pos.local_extreme = pos.local_extreme.min(price);
                } else {
                    pos.local_extreme = pos.local_extreme.max(price);
                }
                
                if let Some(agent) = &mut self.rl_agent {
                    if let Some(tape) = tape_store.get(symbol) {
                        let base_price = pos.entry_price;
                        let direction_bias = if pos.direction == Direction::Long { 1.0 } else { -1.0 };
                        
                        // Extract Baseline (Rolling 30-sec window)
                        let (baseline_tps, _baseline_avg_size, _baseline_flow, baseline_absorption) = tape.get_baseline_metrics(30, 500);
                        
                        // Extract Micro metrics (500ms window)
                        let (micro_tps, micro_quote_vol, _micro_delta, micro_high, micro_low, _micro_ticks, current_cvd) = tape.get_micro_absorption_metrics(500);
                        
                        // Absorption calculation
                        let micro_range = if price > 0.0 { (micro_high - micro_low) / price } else { 0.0 };
                        let raw_absorption_ratio = if micro_range > 0.000001 {
                            micro_quote_vol / (micro_range * price)
                        } else {
                            micro_quote_vol * 100.0
                        };
                        let normalized_absorption = raw_absorption_ratio / baseline_absorption.max(1.0);
                        let normalized_tps = micro_tps / baseline_tps.max(1.0);
                        
                        let reclaim_pct = if pos.direction == Direction::Long {
                            (price - pos.local_extreme) / pos.local_extreme
                        } else {
                            (pos.local_extreme - price) / pos.local_extreme
                        };
                        
                        let cvd_divergence = if pos.direction == Direction::Long {
                            current_cvd - pos.squeeze_cvd
                        } else {
                            pos.squeeze_cvd - current_cvd
                        };
                        
                        let micro_abs_vol = if price > 0.0 { micro_quote_vol / price } else { 0.0 };
                        
                        // Construct 15-dimensional RL feature vector specifically matching Gym `num_features=15`
                        // (Training skipped index 0 [ts] and index 1 [price])
                        let mut features: [f32; 15] = [0.0; 15];
                        features[0] = normalized_absorption as f32;
                        features[1] = normalized_tps as f32;
                        features[2] = (reclaim_pct * 1000.0) as f32;
                        features[3] = (cvd_divergence / 1000.0) as f32;
                        features[4] = ((price - base_price) / base_price * 1000.0) as f32;
                        features[5] = direction_bias as f32;
                        features[6] = (micro_abs_vol / 1000.0) as f32;
                        // Dynamic features matching Gym env:
                        // Gym: features[7] = current_step / max_steps (1200)
                        // Live ticks arrive ~250ms apart, so we estimate Gym-equivalent step count
                        let elapsed_ms = pos.open_time.elapsed().as_millis() as f32;
                        let gym_equivalent_step = elapsed_ms / 250.0; // ~250ms per Gym tick
                        features[7] = (gym_equivalent_step / 1200.0).min(1.0);
                        let pnl_pct = if pos.direction == Direction::Long {
                            (price - base_price) / base_price
                        } else {
                            (base_price - price) / base_price
                        };
                        features[8] = (pnl_pct as f32 * 100.0).clamp(-10.0, 10.0); // unrealized PnL %
                        
                        let action = agent.predict_action(&features);
                        
                        // Phase 37: RL exit-manager — EXACT MIRROR of Gym env
                        // Gym: BURN_IN=10 steps (~250ms each = 2.5s), threshold=±0.1
                        // We replicate this exactly so the agent's trained policy applies correctly.
                        let burn_in_ms = 2500; // 10 Gym steps × 250ms
                        let past_burn_in = pos.open_time.elapsed().as_millis() >= burn_in_ms;
                        
                        if past_burn_in {
                            // Exact same thresholds as Gym step(): ±0.1
                            let wants_exit = if pos.direction == Direction::Long {
                                action <= -0.1
                            } else {
                                action >= 0.1
                            };
                            
                            if wants_exit {
                                rl_early_cut = true;
                            }
                        }
                        
                        if rl_early_cut {
                            info!("🔪🤖 RL AGENT EARLY CUT: {} (action={:.4}, pnl={:.2}%) [abs={:.2}, tps={:.2}, reclaim={:.2}%, cvd_div={:.2}]", 
                                symbol, action, ((price - base_price) / base_price * 100.0 * direction_bias), normalized_absorption, normalized_tps, reclaim_pct * 100.0, cvd_divergence);
                        }
                    }
                }
                
                (trail_updated, rl_early_cut)
            } else {
                // Phase 29C+2: Full Probabilistic Brain — collects tags + Decision Matrix
                let flow = tape_store.get_mut(symbol)
                    .map(|mut s| s.order_flow_signal())
                    .unwrap_or_default();

                // Get wall snapshot for this symbol
                let wall_snap = self.wall_store.get(symbol).map(|r| r.clone());

                // Calculate BTC 5m momentum for Macro Pillar
                let btc_momentum = {
                    let btc_key = if self.candle_buffers.contains_key("BTC/USDT") { "BTC/USDT" } else { "BTCUSDT" };
                    self.candle_buffers.get(btc_key).map(|buf| {
                        let lookback = 5.min(buf.len());
                        if lookback >= 2 {
                            let old = buf[buf.len() - lookback].close;
                            let new_price = buf[buf.len() - 1].close;
                            if old > 0.0 { (new_price - old) / old } else { 0.0 }
                        } else { 0.0 }
                    })
                };

                pos.update_with_brain(price, &flow, wall_snap.as_ref(), btc_momentum)
            };

            // Early cut-loss: delta strongly against + position in loss → close now
            if early_cut {
                let sym = symbol.to_string();
                self.close_position(&sym, ExitReason::SmartExit, price).await;
                return;
            }

            if updated {
                // Update SL order on exchange when trail moves
                let sl_side = match pos.direction {
                    Direction::Long => Side::Sell,
                    Direction::Short => Side::Buy,
                };
                let new_sl = pos.sl_price;
                let qty = pos.size;
                let sym = symbol.to_string();
                info!("📐 {} tick trail → SL={:.4}", sym, new_sl);
                // Cancel old SL and place new one
                let _ = self.order_router.cancel_all_orders(&sym).await;
                if let Err(e) = self.order_router.stop_market_order(&sym, sl_side, qty, new_sl).await {
                    error!("❌ Failed to update trail SL for {}: {}", sym, e);
                }
            }
        }

        // Phase 30.5: Density Breakout 50% Scale-Out (Halfway point)
        let mut density_scale_out_size = 0.0;
        let mut density_side = Side::Buy;

        if let Some(pos) = self.position_manager.positions.get_mut(symbol) {
            // "breakout" is the strategy type we spawn in DensityRadar
            if pos.strategy == "breakout" && !pos.partially_exited {
                if let Some(tp) = pos.tp_price {
                    let total_dist = (tp - pos.entry_price).abs();
                    // Has it moved 50% of the way to the target?
                    let moved_50_pct = match pos.direction {
                        Direction::Long => price >= pos.entry_price + total_dist * 0.5,
                        Direction::Short => price <= pos.entry_price - total_dist * 0.5,
                    };

                    if moved_50_pct {
                        density_scale_out_size = pos.scale_out(0.5);
                        pos.sl_price = pos.entry_price; // Move stop to Breakeven
                        pos.trailing.current_sl = pos.entry_price;
                        density_side = match pos.direction {
                            Direction::Long => Side::Sell,
                            Direction::Short => Side::Buy,
                        };
                    }
                }
            }
        }

        if density_scale_out_size > 0.0 {
            let sym = symbol.to_string();
            info!("✂️🧱 {} DENSITY SCALE-OUT: Hit 50% to target! Exiting {} contracts -> SL to BE", 
                sym, density_scale_out_size);
            let _ = self.order_router.market_order(&sym, density_side, density_scale_out_size, price).await;
        }

        // Check SL/TP hits
        let exit_reason = {
            let pos = self.position_manager.positions.get(symbol).unwrap();
            pos.check_exit(price, price) // Using price as both high and low for tick
        };

        if let Some(reason) = exit_reason {
            self.close_position(symbol, reason, price).await;
            return;
        }

        // Smart Trailer evaluation
        let action = {
            let pos = self.position_manager.positions.get(symbol).unwrap();
            self.smart_trailer.evaluate(pos, price, ob_store, tape_store)
        };

        match action {
            TrailAction::ForceClose(reason) => {
                // Skip ForceClose for knife — DE doesn't know about SmartTrailer,
                // and counter-trend entries naturally have negative composite scores
                let is_knife = self.position_manager.positions.get(symbol)
                    .map(|p| p.strategy == "knife_tick")
                    .unwrap_or(false);
                if is_knife {
                    info!("🔪 SmartTrailer ForceClose SKIPPED for knife {} (DE-only exits)", symbol);
                } else {
                    self.close_position(symbol, reason, price).await;
                }
            }
            TrailAction::PartialExit => {
                let (partial_qty, side, direction) = {
                    let pos = self.position_manager.positions.get(symbol).unwrap();
                    let side = if pos.direction == Direction::Long { Side::Sell } else { Side::Buy };
                    (pos.size * 0.5, side, pos.direction)
                };

                info!("📉 Executing 50% PARTIAL EXIT for {} {:?} @ {:.4}", symbol, direction, price);

                match self.order_router.market_order(symbol, side, partial_qty, price).await {
                    Ok(_) => {
                        if let Some(pos) = self.position_manager.positions.get_mut(symbol) {
                            pos.size -= partial_qty;
                            pos.partially_exited = true;
                            let old_sl = pos.sl_price;
                            // Move SL to Breakeven
                            pos.sl_price = pos.entry_price;
                            pos.trailing.current_sl = pos.entry_price;
                            info!("🛡️ Moved Stop-Loss to Breakeven for {}: {:.4} -> {:.4}", symbol, old_sl, pos.sl_price);
                        }
                    }
                    Err(e) => {
                        error!("❌ Failed to execute partial exit for {}: {}", symbol, e);
                    }
                }
            }
            TrailAction::TightenTrail { factor } => {
                if let Some(pos) = self.position_manager.positions.get_mut(symbol) {
                    let old_sl = pos.sl_price;
                    match pos.direction {
                        Direction::Long => {
                            let new_sl = price - (price - pos.sl_price) * factor;
                            if new_sl > pos.sl_price {
                                pos.sl_price = new_sl;
                                info!("🔧 {} trail tightened: {:.4} → {:.4}", symbol, old_sl, new_sl);
                            }
                        }
                        Direction::Short => {
                            let new_sl = price + (pos.sl_price - price) * factor;
                            if new_sl < pos.sl_price {
                                pos.sl_price = new_sl;
                                info!("🔧 {} trail tightened: {:.4} → {:.4}", symbol, old_sl, new_sl);
                            }
                        }
                    }
                }
            }
            TrailAction::Hold => {}
        }

        // Phase 14: Update open positions live stats
        {
            let pos = self.position_manager.positions.get(symbol).unwrap();
            let mut stats = self.live_stats.lock().await;
            if let Some(mut p) = stats.open_positions.iter_mut().find(|x| x.symbol == symbol) {
                p.current_price = price;
                p.duration_secs = pos.open_time.elapsed().as_secs();
                p.pnl_pct = match pos.direction {
                    Direction::Long => (price - pos.entry_price) / pos.entry_price * 100.0,
                    Direction::Short => (pos.entry_price - price) / pos.entry_price * 100.0,
                };
                p.size = pos.size; // Phase 29C+1: Update size dynamically for grid
                p.target_size = pos.target_size;
            }
        }
    }

    /// Close a position
    async fn close_position(&mut self, symbol: &str, reason: ExitReason, price: f64) {
        if let Some(pos) = self.position_manager.positions.get(symbol) {
            let close_side = match pos.direction {
                Direction::Long => Side::Sell,
                Direction::Short => Side::Buy,
            };
            // CRITICAL: Use target_size (full grid volume), not size (Step 0 only).
            // The grid orders fill on Binance asynchronously, so pos.size may only
            // reflect the initial thin order. target_size is the full intended position.
            let quantity = pos.target_size.max(pos.size);

            // Calculate PnL with round-trip fee (matching backtest knife_tick.rs)
            let round_trip_fee_pct = 0.10; // 0.05% entry + 0.05% exit = 0.10% total
            let gross_pnl_pct = match pos.direction {
                Direction::Long => (price - pos.entry_price) / pos.entry_price * 100.0,
                Direction::Short => (pos.entry_price - price) / pos.entry_price * 100.0,
            };
            let pnl_pct = gross_pnl_pct - round_trip_fee_pct; // Net PnL after fees

            let risk_pct = (pos.risk_dist / pos.entry_price) * 100.0;
            let pnl_r = if risk_pct > 0.0 {
                pnl_pct / risk_pct
            } else {
                0.0
            };

            info!("🔴 CLOSING {} {} | Reason: {} | PnL: {:.2}R | Entry: {:.4} → Exit: {:.4}",
                pos.direction_str(), symbol, reason.as_str(), pnl_r, pos.entry_price, price);

            // Phase 14: Send Telegram Exit Alert
            if let Some(tx) = &self.tg_tx {
                use crate::live::telegram_bot::escape_variable as esc;
                
                let pnl_icon = if pnl_r > 0.0 { "✅" } else { "🔻" };
                
                let msg = format!(
                    "{} *TRADE CLOSED* \\| `#{}` \\({}\\)\n\
                    🏁 Reason: *{}*\n\
                    💵 Exit: ${}\n\
                    📈 PnL: {:.2}% \\({} {:.2} R\\)\n\
                    ⏱ Duration: {}m\n\
                    💰 Balance: ${:.0}",
                    pnl_icon, esc(symbol), pos.direction_str(),
                    esc(reason.as_str()),
                    esc(&format!("{:.4}", price)),
                    pnl_pct, 
                    if pnl_r > 0.0 { "+" } else { "" }, pnl_r,
                    pos.open_time.elapsed().as_secs() / 60,
                    self.current_equity
                );
                
                let _ = tx.try_send(msg);
            }

            // Phase 14: Update Live Stats
            {
                let mut stats = self.live_stats.lock().await;
                stats.daily_pnl_r += pnl_r;
                stats.total_pnl_r += pnl_r;
                // daily_trade_count is already updated on entry, so we just update total
                stats.total_trades += 1;
                
                if pnl_r > 0.1 {
                    stats.wins += 1;
                } else if pnl_r < -0.1 {
                    stats.losses += 1;
                } else {
                    stats.be_count += 1;
                }
                
                if pnl_r > stats.best_trade_r { stats.best_trade_r = pnl_r; }
                if pnl_r < stats.worst_trade_r { stats.worst_trade_r = pnl_r; }

                // Determine risk size per trade based on equity
                stats.equity = self.current_equity;
                // Average risk is roughly equity * 0.02 * (ja_mult ~ 1.0) * (drawdown_scaler ~ 1.0)
                stats.risk_per_trade = self.current_equity * 0.02;
                
                // Remove from open positions
                stats.open_positions.retain(|p| p.symbol != symbol);
            }

            // Phase 11: Trade Logger — EXIT event
            let trade_id = self.trade_ids.remove(symbol)
                .unwrap_or_else(|| format!("unknown_{}", symbol));
            let duration_secs = pos.open_time.elapsed().as_secs();
            // MFE/MAE from trailing state (best_price tracks the peak)
            let mfe_pct = match pos.direction {
                Direction::Long => ((pos.trailing.best_price - pos.entry_price) / pos.entry_price * 100.0).max(0.0),
                Direction::Short => ((pos.entry_price - pos.trailing.best_price) / pos.entry_price * 100.0).max(0.0),
            };
            let mae_pct = pnl_pct.min(0.0).abs(); // Worst unrealized loss approximation

            trade_logger::log_exit(&trade_logger::TradeExit {
                event: "EXIT",
                ts: trade_logger::now_iso(),
                trade_id,
                symbol: symbol.to_string(),
                strategy: pos.strategy.clone(),
                direction: pos.direction_str().to_string(),
                entry_price: pos.entry_price,
                exit_price: price,
                exit_reason: reason.as_str().to_string(),
                pnl_r,
                pnl_pct,
                duration_secs,
                mfe_pct,
                mae_pct,
                whale_tag: None,
                liq_zscore: None,
            });

            // Phase 30: Record loss for per-symbol cooldown
            if pnl_r < -0.1 {
                self.symbol_loss_times
                    .entry(symbol.to_string())
                    .or_insert_with(Vec::new)
                    .push(Instant::now());
                // Clean up old entries (keep only last 2 hours)
                if let Some(times) = self.symbol_loss_times.get_mut(symbol) {
                    times.retain(|t| t.elapsed().as_secs() < 7200);
                }
            }

            // Cancel existing stop orders
            let _ = self.order_router.cancel_all_orders(symbol).await;

            // Place market close order
            match self.order_router.market_order(symbol, close_side, quantity, price).await {
                Ok(_) => {
                    info!("✅ Close order filled for {}", symbol);
                    // Phase 10: Record PnL for daily tracking
                    self.daily_pnl_r += pnl_r;
                    info!("📊 Daily PnL: {:.2}R (limit: -{:.0}R)", 
                        self.daily_pnl_r, DAILY_LOSS_LIMIT_R);
                }
                Err(e) => {
                    error!("❌ Close order FAILED for {}: {}", symbol, e);
                }
            }
        }

        // Phase 10: Release position lock
        self.position_locks.remove(symbol);
        self.position_manager.close(symbol, reason);
    }

    /// Reload configs (hot-swap from Python pipeline)
    pub fn reload_configs(&mut self, config_path: &std::path::Path) {
        info!("🔄 Hot-swapping active configs & ML models...");
        self.configs = config_loader::load_active_configs(config_path);
        
        // Reload ML Ensembles (Phase 11: Full Hot-swap)
        let mut ml_models = HashMap::new();
        let model_names: Vec<String> = self.configs.iter()
            .map(|c| c.model_name().to_string())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        for model_name in &model_names {
            let lgbm_path = self.models_dir.join(format!("{}.json", model_name));
            if let Ok(lgbm) = LgbmModel::load(&lgbm_path) {
                let xgb_path = self.models_dir.join(format!("xgb_{}.json", model_name));
                let xgb = LgbmModel::load(&xgb_path).ok();
                let rf_path = self.models_dir.join(format!("rf_{}.json", model_name));
                let rf = LgbmModel::load(&rf_path).ok();
                ml_models.insert(model_name.clone(), Ensemble { lgbm, xgb, rf });
            }
        }
        self.ml_models = ml_models;

        // Reload Meta-Model
        let meta_json_path = self.models_dir.join("meta_model.json");
        self.meta_model = LgbmModel::load(&meta_json_path).ok();

        info!("   Reloaded {} configs and {} ML ensembles", self.configs.len(), self.ml_models.len());
    }

    /// Feed a trade to the Whale Detector (Phase 12)
    pub fn record_whale_trade(&mut self, symbol: &str, price: f64, quantity: f64, is_buyer_maker: bool) {
        self.whale_detector.record_trade(symbol, price, quantity, is_buyer_maker);
    }

    /// Preload recent historical candles via Binance REST API
    /// This fills the 201-candle warmup buffer instantly on startup (Phase 12 addition)
    pub async fn preload_historical_candles(&mut self) {
        let symbols = self.get_symbols();
        info!("⏳ Preloading historical candles for {} symbols...", symbols.len());
        
        let mut handles = Vec::new();
        
        for symbol in symbols {
            let api_sym = symbol.replace("/", "").replace("_", "").to_uppercase();
            let ws_key = super::ws_feed::format_symbol(&api_sym);
            let url = format!("https://fapi.binance.com/fapi/v1/klines?symbol={}&interval=5m&limit=250", api_sym);
            let client = self.http_client.clone();
            let sym_clone = symbol.clone();
            
            handles.push(tokio::spawn(async move {
                let mut buffer = Vec::with_capacity(250);
                match client.get(&url).send().await {
                    Ok(resp) if resp.status().is_success() => {
                        if let Ok(klines) = resp.json::<Vec<Vec<serde_json::Value>>>().await {
                            for k in klines {
                                if k.len() >= 6 {
                                    let parse_f64 = |v: &serde_json::Value| -> f64 {
                                        v.as_str().and_then(|s| s.parse().ok()).unwrap_or(0.0)
                                    };
                                    let candle = crate::backtest::Candle {
                                        timestamp: k[6].as_u64().unwrap_or(0).to_string(),
                                        open: parse_f64(&k[1]),
                                        high: parse_f64(&k[2]),
                                        low: parse_f64(&k[3]),
                                        close: parse_f64(&k[4]),
                                        volume: parse_f64(&k[5]),
                                        num_trades: k[8].as_u64().unwrap_or(0) as f64,
                                        taker_buy_volume: parse_f64(&k[9]),
                                        quote_volume: parse_f64(&k[7]),
                                    };
                                    buffer.push(candle);
                                }
                            }
                        }
                    }
                    Ok(resp) => tracing::warn!("Failed to preload {}: HTTP {}", sym_clone, resp.status()),
                    Err(e) => tracing::warn!("Failed to preload {}: {}", sym_clone, e),
                }
                (sym_clone, ws_key, buffer)
            }));
            
            // tiny sleep to stagger requests and avoid rate limits
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        
        let mut loaded_count = 0;
        for handle in handles {
            if let Ok((symbol, ws_key, buffer)) = handle.await {
                if !buffer.is_empty() {
                    let buf_len = buffer.len();
                    info!("   ✅ {} → {} candles loaded (key='{}')", symbol, buf_len, ws_key);
                    self.candle_buffers.insert(ws_key, buffer);
                    loaded_count += 1;
                }
            }
        }
        
        info!("✅ Preloaded historical candles for {} symbols", loaded_count);
    }

    pub fn update_mode_stats(&mut self, key: &str, trades: &[backtest::Trade]) {
        let stats = self.shadow_stats.entry(key.to_string()).or_insert(ModeStats::default());
        stats.trades_count = trades.len();
        if !trades.is_empty() {
            let wins = trades.iter().filter(|t| t.pnl_r > 0.0).count();
            stats.wins = wins;
            // Simplified equity tracking for the buffer window
            stats.equity = trades.iter().map(|t| t.pnl_r).sum();
        }
    }

    pub fn check_hotswap(&mut self, symbol: &str, strategy: &str, base_key: &str) {
        let cons_key = format!("{}_conservative", base_key);
        let aggr_key = format!("{}_aggressive", base_key);

        let cons_stats = self.shadow_stats.get(&cons_key).cloned().unwrap_or_default();
        let aggr_stats = self.shadow_stats.get(&aggr_key).cloned().unwrap_or_default();

        let current_mode = self.active_modes.get(base_key).map(|s| s.as_str()).unwrap_or("conservative");

        if current_mode == "conservative" {
            // Switch to Aggressive if it's significantly better in recent buffer
            if aggr_stats.trades_count > 5 && aggr_stats.equity > cons_stats.equity + 1.0 {
                info!("🔥 [HOT-SWAP] {} {}: Switching to AGGRESSIVE (Aggr_Pnl={:.1}R vs Cons_Pnl={:.1}R)",
                    symbol, strategy, aggr_stats.equity, cons_stats.equity);
                self.active_modes.insert(base_key.to_string(), "aggressive".to_string());
                
                if let Some(tx) = &self.tg_tx {
                    let _ = tx.try_send(format!("🔄 *HOT-SWAP: {}*\nPhase shift detected. Switched to **AGGRESSIVE** parameters.\nRecent: Aggr. PnL: {:.2}R", symbol, aggr_stats.equity));
                }
            }
        } else {
            // Switch back to Conservative if Aggressive starts failing or Cons is safer
            if cons_stats.equity > aggr_stats.equity || aggr_stats.equity < -2.0 {
                info!("🛡️ [HOT-SWAP] {} {}: Switching to CONSERVATIVE (Cons_Pnl={:.1}R vs Aggr_Pnl={:.1}R)",
                    symbol, strategy, cons_stats.equity, aggr_stats.equity);
                self.active_modes.insert(base_key.to_string(), "conservative".to_string());
                
                if let Some(tx) = &self.tg_tx {
                     let _ = tx.try_send(format!("🛡️ *HOT-SWAP: {}*\nStabilizing strategy. Switched back to **CONSERVATIVE** parameters.", symbol));
                }
            }
        }
    }

    /// Phase 29C: True Tick Macro Trigger
    /// Evaluates high-frequency drops/spikes using TapeStore and Orderbook, 
    /// disconnected from 1-minute candle limitations.
    pub async fn check_macro_triggers(&mut self, ob_store: &OrderBookStore) {
        // Evaluate all active configs that belong to "knifetick"
        let knife_configs: Vec<LiveConfig> = self.configs.iter()
            .filter(|c| c.rust_strategy_name() == "knifetick")
            .cloned()
            .collect();

        if knife_configs.is_empty() { return; }

        let active_symbols: Vec<String> = self.tape_store.iter().map(|kv| kv.key().clone()).collect();

        for symbol in active_symbols {
            // 1. Skip if we already hold a position
            if self.position_manager.has_position(&symbol) {
                continue;
            }

            // 2. Cooldown check (30 seconds between signals per symbol)
            if let Some(last) = self.last_signal_time.get(&symbol) {
                if Instant::now().checked_duration_since(*last).unwrap_or_default().as_secs() < 30 {
                    continue;
                }
            }

            // 3. Skip if this symbol is blacklisted or hit 2 losses
            if is_coin_blacklisted(&symbol) { continue; }
            if let Some(loss_times) = self.symbol_loss_times.get(&symbol) {
                let recent_losses = loss_times.iter().filter(|t| t.elapsed().as_secs() < 3600).count();
                if recent_losses >= 2 { continue; }
            }

            // 4. True Tick Analysis from TapeStore
            let tape_state = match self.tape_store.get(&symbol) {
                Some(state) => state,
                None => continue,
            };

            let zscore = tape_state.volume_zscore();
            let current_cvd = tape_state.cvd;
            let _tape_speed = tape_state.tape_speed();
            let imb_ratio = tape_state.imbalance_ratio();

            // Find matching config for this symbol
            let clean_symbol = symbol.replace("/", "").replace("_", "").to_uppercase();
            let config = knife_configs.iter().find(|c| c.symbol.replace("/", "").replace("_", "").to_uppercase() == clean_symbol);
            
            if let Some(cfg) = config {
                // Phase 31B: Use GA-optimized min_zscore from config instead of hardcoded 2.0
                let cfg_min_zscore = cfg.params.as_ref()
                    .and_then(|p| p.get("min_zscore"))
                    .and_then(|v| v.as_f64())
                    .unwrap_or(2.0);

                // Phase 1: Squeeze detection — zscore + imbalance confirm panic
                let is_long_squeeze = zscore > cfg_min_zscore && imb_ratio < 0.9;
                let is_short_squeeze = zscore > cfg_min_zscore && imb_ratio > 1.1;
                
                // Phase 36 FIX: Two-phase CVD divergence (matches backtester exactly).
                // Step 1: When zscore triggers, SAVE current CVD as squeeze_cvd.
                // Step 2: Entry only when CVD has IMPROVED from squeeze_cvd
                //   (LONG: cvd > squeeze_cvd = buyers returning after dump)
                //   (SHORT: cvd < squeeze_cvd = sellers returning after pump)
                let mut is_long_panic = false;
                let mut is_short_panic = false;
                
                if is_long_squeeze || is_short_squeeze {
                    // Record squeeze CVD if not already stored (or expired > 60s)
                    let should_store = match self.squeeze_cvd.get(&symbol) {
                        None => true,
                        Some((_, when)) => when.elapsed().as_secs() > 60, // Expire after 60s
                    };
                    if should_store {
                        self.squeeze_cvd.insert(symbol.clone(), (current_cvd, Instant::now()));
                        info!("🔪📌 SQUEEZE DETECTED [{}]: Z={:.1} CVD_saved={:.4} IMB={:.2} — waiting for CVD divergence",
                            symbol, zscore, current_cvd, imb_ratio);
                    }
                }
                
                // Check CVD divergence against stored squeeze point
                if let Some((saved_cvd, when)) = self.squeeze_cvd.get(&symbol) {
                    let age_secs = when.elapsed().as_secs();
                    if age_secs <= 60 {
                        // LONG: CVD must have improved (buyers returning)
                        if is_long_squeeze && current_cvd > *saved_cvd {
                            is_long_panic = true;
                            info!("🔪✅ CVD DIVERGENCE [{}] LONG: cvd={:.4} > squeeze={:.4} (Δ={:.4})",
                                symbol, current_cvd, saved_cvd, current_cvd - saved_cvd);
                        }
                        // SHORT: CVD must have worsened (sellers returning)  
                        if is_short_squeeze && current_cvd < *saved_cvd {
                            is_short_panic = true;
                            info!("🔪✅ CVD DIVERGENCE [{}] SHORT: cvd={:.4} < squeeze={:.4} (Δ={:.4})",
                                symbol, current_cvd, saved_cvd, saved_cvd - current_cvd);
                        }
                    } else {
                        // Expired — remove
                        self.squeeze_cvd.remove(&symbol);
                    }
                }

                // Phase 31B: Verbose rejection logging
                if !is_long_panic && !is_short_panic && zscore > 1.0 {
                    // Only log when zscore is at least mildly elevated, to avoid spam
                    if zscore > cfg_min_zscore * 0.8 {
                        info!("🔪📊 SIGNAL NEAR-MISS {}: Z={:.2} (need>{:.2}) CVD={:.4} IMB={:.2}",
                            symbol, zscore, cfg_min_zscore, current_cvd, imb_ratio);
                    }
                }

                if is_long_panic || is_short_panic {
                    let direction = if is_long_panic { Direction::Long } else { Direction::Short };

                    // === BUG FIX #12: Match backtester entry filters ===
                    // Live was missing vol_spike, absorption, speed, and reclaim checks.
                    // This caused 34 entries per morning vs ~15 per MONTH in backtest.
                    
                    // Read DE-optimized params from config
                    let cfg_min_vol_spike = cfg.params.as_ref()
                        .and_then(|p| p.get("min_vol_spike")).and_then(|v| v.as_f64()).unwrap_or(1.5);
                    let cfg_min_absorption = cfg.params.as_ref()
                        .and_then(|p| p.get("min_absorption")).and_then(|v| v.as_f64()).unwrap_or(5.0);
                    let cfg_max_speed_mult = cfg.params.as_ref()
                        .and_then(|p| p.get("max_speed_mult")).and_then(|v| v.as_f64()).unwrap_or(3.0);
                    let cfg_min_reclaim_pct = cfg.params.as_ref()
                        .and_then(|p| p.get("min_reclaim_pct")).and_then(|v| v.as_f64()).unwrap_or(0.001);
                    let cfg_micro_window_ms = cfg.params.as_ref()
                        .and_then(|p| p.get("micro_window_ms")).and_then(|v| v.as_f64()).unwrap_or(1500.0) as i64;
                    let cfg_baseline_window_sec = cfg.params.as_ref()
                        .and_then(|p| p.get("baseline_window_sec")).and_then(|v| v.as_f64()).unwrap_or(30.0) as i64;
                    
                    // Get micro absorption metrics from tape
                    let (micro_tps, micro_quote_vol, _micro_delta, micro_high, micro_low, _micro_trade_count, _cvd) = 
                        tape_state.get_micro_absorption_metrics(cfg_micro_window_ms);
                    
                    // Get baseline metrics
                    let (baseline_tps, _baseline_avg_size, _baseline_flow, _baseline_absorption) = 
                        tape_state.get_baseline_metrics(cfg_baseline_window_sec, cfg_micro_window_ms);
                    
                    // FILTER 1: Volume spike — current trade rate must be cfg_min_vol_spike × baseline
                    let vol_ratio = if baseline_tps > 0.1 { micro_tps / baseline_tps } else { 1.0 };
                    if vol_ratio < cfg_min_vol_spike {
                        info!("🔪❌ VOL_SPIKE REJECT [{}]: ratio={:.2} < {:.2}", symbol, vol_ratio, cfg_min_vol_spike);
                        continue;
                    }
                    
                    // FILTER 2: Absorption — high volume but price barely moved (wall absorbed it)
                    let micro_range = if micro_high > micro_low && tape_state.last_price > 0.0 {
                        (micro_high - micro_low) / tape_state.last_price
                    } else { 0.0 };
                    let absorption_ratio = if micro_range > 0.000001 {
                        micro_quote_vol as f64 / (micro_range * tape_state.last_price)
                    } else {
                        micro_quote_vol * 100.0 // Price didn't move at all = infinite absorption
                    };
                    // Note: backtester checks absorption_ratio > baseline_absorption * min_absorption.
                    // We approximate baseline_absorption from known values.
                    let min_absorption_abs = cfg_min_absorption * 1000.0; // Scale factor for live
                    if absorption_ratio < min_absorption_abs {
                        info!("🔪❌ ABSORPTION REJECT [{}]: ratio={:.0} < {:.0}", symbol, absorption_ratio, min_absorption_abs);
                        continue;
                    }
                    
                    // FILTER 3: Speed — tape not too fast (not a stop-hunt cascade)
                    let speed_ratio = if baseline_tps > 0.1 { micro_tps / baseline_tps } else { 1.0 };
                    if speed_ratio > cfg_max_speed_mult {
                        info!("🔪❌ SPEED REJECT [{}]: speed_ratio={:.1} > {:.1}", symbol, speed_ratio, cfg_max_speed_mult);
                        continue;
                    }
                    
                    // FILTER 4: Reclaim — price must have bounced from the extreme
                    let reclaim_pct = if micro_high > 0.0 && micro_low > 0.0 && micro_low < micro_high {
                        if is_long_panic {
                            // LONG: price must have bounced UP from the low
                            (tape_state.last_price - micro_low) / micro_low
                        } else {
                            // SHORT: price must have bounced DOWN from the high
                            (micro_high - tape_state.last_price) / micro_high
                        }
                    } else { 0.0 };
                    if reclaim_pct < cfg_min_reclaim_pct {
                        info!("🔪❌ RECLAIM REJECT [{}]: reclaim={:.4}% < {:.4}%", symbol, reclaim_pct*100.0, cfg_min_reclaim_pct*100.0);
                        continue;
                    }
                    
                    info!("🔪✅ ALL FILTERS PASSED [{}]: Z={:.1} vol_spike={:.1}x absorption={:.0} speed={:.1}x reclaim={:.3}%",
                        symbol, zscore, vol_ratio, absorption_ratio, speed_ratio, reclaim_pct*100.0);
                    
                    // Clear squeeze CVD — entry is proceeding
                    self.squeeze_cvd.remove(&symbol);

                    // === Phase 30: VOLATILITY GATE ===
                    let mut vol_gate_passed = true;
                    if let Some(buffer) = self.candle_buffers.get(&symbol) {
                        let lookback = 120.min(buffer.len()); // 2 hours
                        if lookback > 10 {
                            let recent = &buffer[buffer.len() - lookback..];
                            let range_high = recent.iter().map(|c| c.high).fold(f64::MIN, f64::max);
                            let range_low = recent.iter().map(|c| c.low).fold(f64::MAX, f64::min);
                            let _range_pct = if range_low > 0.0 { (range_high - range_low) / range_low } else { 0.0 };
                            let sl_pct = cfg.params.as_ref()
                                .and_then(|p| p.get("sl_pct").or_else(|| p.get("sl_buffer_pct")))
                                .and_then(|v| v.as_f64())
                                .unwrap_or(0.003);
                            
                            if _range_pct < sl_pct * 2.0 {
                                vol_gate_passed = false;
                            }
                        }
                    }

                    if !vol_gate_passed { continue; }

                    let mode = self.active_modes.get(&format!("{}_{}", symbol, cfg.strategy))
                        .map(|s| s.as_str())
                        .unwrap_or("conservative");
                    let params = cfg.params_vec(mode);
                    let sl_pct = cfg.params.as_ref().and_then(|p| p.get("sl_pct").or_else(|| p.get("sl_buffer_pct"))).and_then(|v| v.as_f64()).unwrap_or(0.003);
                    
                    // BUG FIX #9: TP placeholder — will be recalculated after grid_bottom_price is known.
                    // Using sl_pct * 1.5 as initial estimate (matches backtester minimum R:R).
                    let tp_pct_placeholder = sl_pct * 1.5;
                    
                    let dummy_price = if let Some(ob) = ob_store.get(&symbol) { ob.mid_price().unwrap_or(1.0) } else { 1.0 };

                    let mut trade_clone = crate::backtest::Trade {
                        entry_idx: 0,
                        direction: if is_long_panic { "LONG".to_string() } else { "SHORT".to_string() },
                        entry_price: dummy_price,
                        exit_price: 0.0,
                        sl_price: if is_long_panic { dummy_price * (1.0 - sl_pct) } else { dummy_price * (1.0 + sl_pct) },
                        tp_price: if is_long_panic { dummy_price * (1.0 + tp_pct_placeholder) } else { dummy_price * (1.0 - tp_pct_placeholder) },
                        pnl_r: 0.0,
                        risk_dist: dummy_price * sl_pct,
                        pnl_abs: 0.0,
                        mfe_pct: 0.0,
                    };

                    let initial_delta = tape_state.normalized_delta().abs();
                    
                    // === Phase 31C: Density Zone Detection (Cascade-based) ===
                    // Always enter with grid. Find the best target:
                    //   1) Cascade cluster (multiple walls within 0.5% — accumulate their USD)
                    //   2) Single heavy wall nearby
                    //   3) Default: spread grid 0.5% from current price
                    let current_price = tape_state.last_price;
                    let grid_bottom_price;
                    let mut is_real_wall = false;
                    
                    if let Some(wall_snap) = self.wall_store.get(&symbol) {
                        let target_side = if is_long_panic {
                            crate::live::wall_tracker::WallSide::Bid
                        } else {
                            crate::live::wall_tracker::WallSide::Ask
                        };
                        
                        // Priority 1: Find cascade cluster within 2% of price
                        let best_cascade = wall_snap.cascades.iter()
                            .filter(|c| c.side == target_side && c.thickness >= 2)
                            .filter(|c| {
                                let center = (c.bottom_price + c.top_price) / 2.0;
                                let dist = (current_price - center).abs() / current_price;
                                dist <= 0.02 // Within 2%
                            })
                            .max_by(|a, b| a.total_size_usd.partial_cmp(&b.total_size_usd).unwrap());
                        
                        if let Some(cascade) = best_cascade {
                            // Use the furthest edge of the cascade as grid bottom
                            grid_bottom_price = if is_long_panic { cascade.bottom_price } else { cascade.top_price };
                            is_real_wall = true;
                            info!("🔪🧲 CASCADE ZONE: {} — {} walls, ${:.0} total, range {:.4}–{:.4}",
                                symbol, cascade.thickness, cascade.total_size_usd, cascade.bottom_price, cascade.top_price);
                        } else {
                            // Priority 2: Single wall nearby
                            let walls = if is_long_panic { wall_snap.bid_walls() } else { wall_snap.ask_walls() };
                            let close_wall = walls.iter().find(|w| {
                                let dist = (current_price - w.price).abs() / current_price;
                                dist <= 0.015
                            });
                            
                            if let Some(w) = close_wall {
                                grid_bottom_price = w.price;
                                is_real_wall = true;
                                info!("🔪🧲 SINGLE WALL: {} — Wall at {:.4} ${:.0}", symbol, w.price, w.current_size_usd);
                            } else {
                                // Priority 3: Default spread 0.5%
                                grid_bottom_price = if is_long_panic {
                                    current_price * 0.995
                                } else {
                                    current_price * 1.005
                                };
                                info!("🔪📡 NO WALL: {} — Default grid spread ±0.5%", symbol);
                            }
                        }
                    } else {
                        // No wall data at all — default spread
                        grid_bottom_price = if is_long_panic {
                            current_price * 0.995
                        } else {
                            current_price * 1.005
                        };
                        info!("🔪📡 NO WALL DATA: {} — Default grid spread ±0.5%", symbol);
                    }
                    
                    let target_wall_price = Some(grid_bottom_price);

                    // BUG FIX #9: Recalculate TP from actual dump size (now that grid_bottom_price is known).
                    // Backtester: tp_dist = dump_size * 0.8, then tp_dist.max(risk * 1.5).
                    // dump_size ≈ distance from current_price to grid_bottom_price.
                    {
                        let dump_size_pct = (current_price - grid_bottom_price).abs() / current_price;
                        let tp_recovery = 0.8; // Matches backtester tp_recovery_pct
                        let tp_pct_real = (dump_size_pct * tp_recovery).max(sl_pct * 1.5);
                        let new_tp = if is_long_panic {
                            dummy_price * (1.0 + tp_pct_real)
                        } else {
                            dummy_price * (1.0 - tp_pct_real)
                        };
                        info!("🔪📊 TP CALC [{}]: dump={:.3}% × 80% = {:.3}%, sl={:.3}%, final_tp={:.3}%",
                            symbol, dump_size_pct*100.0, dump_size_pct*tp_recovery*100.0, sl_pct*100.0, tp_pct_real*100.0);
                        trade_clone.tp_price = new_tp;
                    }

                    // === Phase 31B: Smart BTC Guide V2 (Combo: 5min Price Trend + CVD) ===
                    // Block ONLY when BOTH conditions confirm BTC crash:
                    //   1) BTC price dropped >0.3% in last 5 minutes (real trend, not noise)
                    //   2) BTC CVD is strongly negative right now (selling pressure)
                    let mut btc_guide_passed = true;
                    if symbol != "BTC/USDT" && symbol != "BTCUSDT" {
                        let btc_sym = if symbol.contains('/') { "BTC/USDT" } else { "BTCUSDT" };
                        
                        // Get BTC 5-minute price change from candle buffers
                        let btc_price_change_5m = self.candle_buffers.get(btc_sym)
                            .or_else(|| self.candle_buffers.get("BTCUSDT"))
                            .map(|buf| {
                                let lookback = 5.min(buf.len());
                                if lookback >= 2 {
                                    let old = buf[buf.len() - lookback].close;
                                    let new = buf[buf.len() - 1].close;
                                    if old > 0.0 { (new - old) / old } else { 0.0 }
                                } else { 0.0 }
                            }).unwrap_or(0.0);
                        
                        // Get BTC instant CVD
                        let btc_cvd = self.tape_store.get(btc_sym)
                            .or_else(|| self.tape_store.get("BTCUSDT"))
                            .map(|t| t.cvd_trend())
                            .unwrap_or(0.0);
                        
                        // Combo logic: block only when BOTH price trend AND CVD confirm
                        if direction == Direction::Long && btc_price_change_5m < -0.003 && btc_cvd < -1.0 {
                            info!("🛑 BTC GUIDE V2: {} LONG blocked — BTC price {:.2}% (5m) + CVD {:.2}",
                                symbol, btc_price_change_5m * 100.0, btc_cvd);
                            btc_guide_passed = false;
                        } else if direction == Direction::Short && btc_price_change_5m > 0.003 && btc_cvd > 1.0 {
                            info!("🛑 BTC GUIDE V2: {} SHORT blocked — BTC price +{:.2}% (5m) + CVD {:.2}",
                                symbol, btc_price_change_5m * 100.0, btc_cvd);
                            btc_guide_passed = false;
                        } else {
                            info!("✅ BTC GUIDE V2: {} OK — BTC 5m={:.2}% CVD={:.2} → neutral/aligned",
                                symbol, btc_price_change_5m * 100.0, btc_cvd);
                        }
                    }

                    if !btc_guide_passed { continue; }

                    let target = HftTarget {
                        symbol: symbol.to_string(),
                        direction,
                        created_at: Instant::now(),
                        initial_delta_abs: initial_delta,
                        strategy_type: "knifetick".to_string(),
                        target_wall_price,
                    };

                    let absorber = self.absorber.clone();
                    if absorber.try_accept(&target).await {
                        let ws = self.wall_store.clone();
                        let ts = self.tape_store.clone();
                        let sym = symbol.to_string();
                        let _cfg = cfg.clone();
                        let params_clone = params.clone();
                        let hft_tx = self.hft_tx.clone();

                        info!("🔪🚀 TRUE TICK MACRO TRIGGERED for {} ({:?}) | ZScore: {:.1} | CVD: {:.4}",
                            sym, direction, zscore, current_cvd);

                        self.last_signal_time.insert(sym.clone(), Instant::now());

                        let w_price = target_wall_price;
                        tokio::spawn(async move {
                            let result = super::absorber::track_knife_tick_v3(
                                sym.clone(), direction, "knifetick".to_string(), ws, ts, absorber, params_clone, w_price
                            ).await;

                            match result {
                                AbsorberResult::Fired { symbol, direction: _, confidence: _, entry_price, target_wall_price, is_wall_backed } => {
                                    let _ = hft_tx.send(HftFireEvent {
                                        symbol,
                                        trade: trade_clone,
                                        config: _cfg,
                                        entry_price,
                                        target_wall_price,
                                        is_wall_backed,
                                    });
                                }
                                AbsorberResult::Timeout { symbol } => {
                                    info!("🔪 ⏱️ TrueTick Absorber timeout on {} — no bottom found", symbol);
                                }
                                AbsorberResult::Rejected { symbol } => {
                                    info!("🔪 🚫 TrueTick Absorber rejected {} — slots full", symbol);
                                }
                            }
                        });
                    }
                }
            }
        }
    }
}

// ── Phase 11.4: Coin Blacklist Check ────────────────────────────────────────

/// Check if a coin is blacklisted (consecutive losses cooldown).
/// Reads `data/coin_blacklist.json` produced by `aggregate_journal.py`.
fn is_coin_blacklisted(symbol: &str) -> bool {
    let path = std::path::Path::new("data/coin_blacklist.json");
    if !path.exists() {
        return false; // No blacklist file = no restrictions
    }

    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return false,
    };

    let blacklist: HashMap<String, serde_json::Value> = match serde_json::from_str(&content) {
        Ok(b) => b,
        Err(_) => return false,
    };

    if let Some(entry) = blacklist.get(symbol) {
        // Check if the cooldown has expired
        if let Some(expires_str) = entry.get("expires_at").and_then(|v| v.as_str()) {
            // Compare ISO timestamps: if current time < expires_at, still banned
            let now = trade_logger::now_iso();
            return now < expires_str.to_string();
        }
        return true; // Has entry but no expiry = banned
    }

    false
}
