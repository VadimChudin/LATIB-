///! Binance Futures WebSocket Feed
///! ===============================
///! Connects to wss://fstream.binance.com/stream and processes:
///! - @kline_5m  → candle updates (signal generation)
///! - @aggTrade  → raw trade tape (CVD / Trade Delta)
///! - @depth@100ms → full order book snapshots
///!
///! Features:
///! - Ping/pong keepalive
///! - 23-hour forced reconnect (Binance 24h limit)
///! - Health watchdog (stale connection detection)
///! - Exponential backoff with jitter

use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;
use futures_util::{SinkExt, StreamExt};
use serde::Deserialize;
use tokio::sync::broadcast;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use tracing::{error, info, warn};

// ── Data Types ──────────────────────────────────────────────────────────────

/// A single OHLCV candle bar
#[derive(Debug, Clone, Default)]
pub struct Candle {
    pub timestamp: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub is_closed: bool,
}

/// A single aggressive trade from the tape
#[derive(Debug, Clone)]
pub struct AggTrade {
    pub symbol: String,
    pub price: f64,
    pub quantity: f64,
    pub is_buyer_maker: bool, // true = seller aggressor, false = buyer aggressor
    pub timestamp: i64,
}

/// Events broadcasted to consumers (position manager, smart trailer, IPC)
#[derive(Debug, Clone)]
pub enum MarketEvent {
    /// A candle tick (updated every ~250ms from Binance)
    KlineTick {
        symbol: String,
        candle: Candle,
    },
    /// A candle has officially closed (x=true)
    KlineClose {
        symbol: String,
        candle: Candle,
    },
    /// An aggregated trade from the tape
    Trade(AggTrade),
    /// A funding rate update from markPrice
    FundingRateUpdate {
        symbol: String,
        rate: f64,
    },
}

// ── Binance WS Message Parsing ──────────────────────────────────────────────

#[derive(Deserialize)]
struct WsWrapper {
    stream: String,
    data: serde_json::Value,
}

#[derive(Deserialize)]
struct BinanceKlineData {
    s: String, // Symbol e.g. "BTCUSDT"
    k: BinanceKline,
}

#[derive(Deserialize)]
struct BinanceKline {
    t: i64,  // Open time
    o: String,
    h: String,
    l: String,
    c: String,
    v: String,
    x: bool, // Is this kline closed?
}

#[derive(Deserialize)]
struct BinanceAggTrade {
    s: String,  // Symbol
    p: String,  // Price
    q: String,  // Quantity
    m: bool,    // Is buyer maker? (true = sell aggressor)
    #[serde(rename = "T")]
    timestamp: i64,
}

#[derive(Deserialize)]
struct BinanceMarkPrice {
    s: String, // Symbol
    r: String, // Funding Rate
}

// ── Constants ───────────────────────────────────────────────────────────────

const WS_BASE_URL: &str = "wss://fstream.binance.com/stream?streams=";
const MAX_RECONNECT_DELAY_SECS: u64 = 120;
const HEALTH_TIMEOUT_SECS: u64 = 300;     // 5 min stale = dead
const MAX_CONNECTION_LIFE_SECS: u64 = 23 * 3600; // 23h forced reconnect
const RECONNECT_BASE_DELAY_SECS: u64 = 5;

// ── WebSocket Feed ──────────────────────────────────────────────────────────

pub struct WsFeed {
    /// Symbols to subscribe (e.g. ["BTCUSDT", "ETHUSDT"])
    symbols: Vec<String>,
    /// Broadcast channel for market events
    event_tx: broadcast::Sender<MarketEvent>,
    /// Live candle buffers per symbol (last N candles)
    pub candle_buffers: Arc<DashMap<String, Vec<Candle>>>,
    /// Connection stats
    reconnect_count: u32,
}

impl WsFeed {
    pub fn new(symbols: Vec<String>, event_tx: broadcast::Sender<MarketEvent>) -> Self {
        let candle_buffers = Arc::new(DashMap::new());
        // Pre-allocate buffers
        for sym in &symbols {
            candle_buffers.insert(sym.clone(), Vec::with_capacity(500));
        }

        Self {
            symbols,
            event_tx,
            candle_buffers,
            reconnect_count: 0,
        }
    }

    /// Build the combined stream URL for all symbols
    fn build_url(&self) -> String {
        let mut streams = Vec::new();
        for sym in &self.symbols {
            let lower = sym.to_lowercase().replace("/", "").replace("_", "");
            streams.push(format!("{}@kline_1m", lower));
            streams.push(format!("{}@aggTrade", lower));
            streams.push(format!("{}@markPrice", lower));
        }
        // Always include BTC for correlation
        if !self.symbols.iter().any(|s| s.to_uppercase() == "BTCUSDT") {
            streams.push("btcusdt@kline_1m".to_string());
            streams.push("btcusdt@aggTrade".to_string());
            streams.push("btcusdt@markPrice".to_string());
        }
        format!("{}{}", WS_BASE_URL, streams.join("/"))
    }

    /// Main connection loop with exponential backoff
    pub async fn run(&mut self) {
        let mut attempt: u32 = 0;

        loop {
            if attempt > 0 {
                let base = RECONNECT_BASE_DELAY_SECS * 2u64.pow(attempt.min(6) - 1);
                let jitter = rand::random::<u64>() % 3;
                let delay = base.min(MAX_RECONNECT_DELAY_SECS) + jitter;
                info!("🔄 Reconnecting in {}s (attempt #{})...", delay, attempt);
                tokio::time::sleep(Duration::from_secs(delay)).await;
            }

            let url = self.build_url();
            info!("📡 Connecting to Binance fstream ({} streams)...", self.symbols.len() * 2);

            match connect_async(&url).await {
                Ok((ws_stream, _response)) => {
                    self.reconnect_count += 1;
                    attempt = 0;
                    info!("✅ WebSocket connected! (session #{})", self.reconnect_count);

                    let connection_start = Instant::now();
                    let mut last_message_time = Instant::now();

                    let (mut _write, mut read) = ws_stream.split();

                    loop {
                        // Health checks
                        let silence = last_message_time.elapsed().as_secs();
                        let age = connection_start.elapsed().as_secs();

                        if silence > HEALTH_TIMEOUT_SECS {
                            warn!("🔴 STALE: No data for {}s. Reconnecting...", silence);
                            break;
                        }
                        if age > MAX_CONNECTION_LIFE_SECS {
                            info!("🔄 23h limit reached. Scheduled reconnect.");
                            break;
                        }

                        // Read next message with timeout
                        let msg = tokio::time::timeout(
                            Duration::from_secs(30),
                            read.next(),
                        )
                        .await;

                        match msg {
                            Ok(Some(Ok(Message::Text(text)))) => {
                                last_message_time = Instant::now();
                                self.handle_message(&text);
                            }
                            Ok(Some(Ok(Message::Ping(data)))) => {
                                last_message_time = Instant::now();
                                if let Err(e) = _write.send(Message::Pong(data)).await {
                                    warn!("Pong send failed: {}", e);
                                    break;
                                }
                            }
                            Ok(Some(Ok(Message::Close(_)))) => {
                                info!("WebSocket closed by server.");
                                break;
                            }
                            Ok(Some(Err(e))) => {
                                warn!("⚠️ WS error: {}", e);
                                break;
                            }
                            Ok(None) => {
                                info!("WebSocket stream ended.");
                                break;
                            }
                            Err(_) => {
                                // Timeout — no message in 30s, continue (ping will keep alive)
                                continue;
                            }
                            _ => continue,
                        }
                    }
                }
                Err(e) => {
                    error!("🚫 Connection failed: {}", e);
                }
            }

            attempt += 1;
        }
    }

    /// Parse and dispatch a single WebSocket message
    fn handle_message(&self, raw: &str) {
        let wrapper: WsWrapper = match serde_json::from_str(raw) {
            Ok(w) => w,
            Err(_) => return,
        };

        if wrapper.stream.contains("kline") {
            self.handle_kline(&wrapper.data);
        } else if wrapper.stream.contains("aggTrade") {
            self.handle_agg_trade(&wrapper.data);
        } else if wrapper.stream.contains("markPrice") {
            self.handle_mark_price(&wrapper.data);
        }
    }

    /// Process a kline message → update candle buffer + broadcast event
    fn handle_kline(&self, data: &serde_json::Value) {
        let kd: BinanceKlineData = match serde_json::from_value(data.clone()) {
            Ok(k) => k,
            Err(_) => return,
        };

        let symbol = format_symbol(&kd.s);
        let candle = Candle {
            timestamp: kd.k.t,
            open: kd.k.o.parse().unwrap_or(0.0),
            high: kd.k.h.parse().unwrap_or(0.0),
            low: kd.k.l.parse().unwrap_or(0.0),
            close: kd.k.c.parse().unwrap_or(0.0),
            volume: kd.k.v.parse().unwrap_or(0.0),
            is_closed: kd.k.x,
        };

        // Update candle buffer
        if let Some(mut buf) = self.candle_buffers.get_mut(&symbol) {
            let should_update = buf.last().map_or(false, |last| last.timestamp == candle.timestamp);

            if should_update {
                if let Some(last) = buf.last_mut() {
                    last.high = candle.high;
                    last.low = candle.low;
                    last.close = candle.close;
                    last.volume = candle.volume;
                    last.is_closed = candle.is_closed;
                }
            } else {
                buf.push(candle.clone());
                if buf.len() > 500 {
                    let drain_to = buf.len() - 500;
                    buf.drain(..drain_to);
                }
            }
        }

        // Broadcast event
        let event = if candle.is_closed {
            MarketEvent::KlineClose { symbol, candle }
        } else {
            MarketEvent::KlineTick { symbol, candle }
        };
        let _ = self.event_tx.send(event);
    }

    /// Process an aggTrade message → broadcast for CVD/Delta calculation
    fn handle_agg_trade(&self, data: &serde_json::Value) {
        let trade: BinanceAggTrade = match serde_json::from_value(data.clone()) {
            Ok(t) => t,
            Err(_) => return,
        };

        let event = MarketEvent::Trade(AggTrade {
            symbol: format_symbol(&trade.s),
            price: trade.p.parse().unwrap_or(0.0),
            quantity: trade.q.parse().unwrap_or(0.0),
            is_buyer_maker: trade.m,
            timestamp: trade.timestamp,
        });
        let _ = self.event_tx.send(event);
    }

    /// Process a markPriceUpdate message → broadcast Funding Rate
    fn handle_mark_price(&self, data: &serde_json::Value) {
        let mp: BinanceMarkPrice = match serde_json::from_value(data.clone()) {
            Ok(m) => m,
            Err(_) => return,
        };

        if let Ok(rate) = mp.r.parse::<f64>() {
            let event = MarketEvent::FundingRateUpdate {
                symbol: format_symbol(&mp.s),
                rate,
            };
            let _ = self.event_tx.send(event);
        }
    }
}

/// Convert "BTCUSDT" → "BTC/USDT"
pub fn format_symbol(raw: &str) -> String {
    if raw.ends_with("USDT") {
        let base = &raw[..raw.len() - 4];
        format!("{}/USDT", base)
    } else {
        raw.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_symbol() {
        assert_eq!(format_symbol("BTCUSDT"), "BTC/USDT");
        assert_eq!(format_symbol("SOLUSDT"), "SOL/USDT");
        assert_eq!(format_symbol("DOGEUSDT"), "DOGE/USDT");
    }

    #[test]
    fn test_build_url() {
        let (tx, _rx) = broadcast::channel(100);
        let feed = WsFeed::new(vec!["BTCUSDT".into(), "ETHUSDT".into()], tx);
        let url = feed.build_url();
        assert!(url.contains("btcusdt@kline_5m"));
        assert!(url.contains("ethusdt@aggTrade"));
    }
}
