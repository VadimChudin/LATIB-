///! IPC Bridge (TCP JSON-RPC)
///! ==========================
///! Bi-directional communication between Rust executor and Python orchestrator.
///!
///! Protocol: Newline-delimited JSON over TCP (port 9090)
///!
///! Python → Rust (Commands):
///!   {"cmd": "open_trade", "symbol": "BTC/USDT", "direction": "LONG", ...}
///!   {"cmd": "close_trade", "symbol": "BTC/USDT"}
///!   {"cmd": "update_config", "configs": [...]}
///!   {"cmd": "ping"}
///!
///! Rust → Python (Events):
///!   {"event": "trade_opened", "symbol": "BTC/USDT", ...}
///!   {"event": "trade_closed", "symbol": "BTC/USDT", "pnl_pct": 1.5, ...}
///!   {"event": "heartbeat", "positions": 2, "equity": 2150.0}
///!   {"event": "signal", "symbol": "BTC/USDT", "direction": "LONG", ...}
///!   {"event": "pong"}

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpListener;
use tokio::sync::broadcast;
use tracing::{error, info, warn};

const IPC_PORT: u16 = 9090;

/// Commands from Python → Rust
#[derive(Debug, Deserialize, Clone)]
#[serde(tag = "cmd")]
pub enum Command {
    #[serde(rename = "open_trade")]
    OpenTrade {
        symbol: String,
        direction: String,
        entry_price: f64,
        sl_price: f64,
        tp_price: Option<f64>,
        size: f64,
        strategy: String,
        trail_activate_r: Option<f64>,
        trail_atr_mult: Option<f64>,
    },
    #[serde(rename = "close_trade")]
    CloseTrade {
        symbol: String,
    },
    #[serde(rename = "update_config")]
    UpdateConfig {
        configs: serde_json::Value,
    },
    #[serde(rename = "ping")]
    Ping,
}

/// Events from Rust → Python
#[derive(Debug, Serialize, Clone)]
#[serde(tag = "event")]
pub enum Event {
    #[serde(rename = "trade_opened")]
    TradeOpened {
        symbol: String,
        direction: String,
        entry_price: f64,
        sl_price: f64,
        size: f64,
        strategy: String,
    },
    #[serde(rename = "trade_closed")]
    TradeClosed {
        symbol: String,
        direction: String,
        exit_price: f64,
        pnl_pct: f64,
        pnl_r: f64,
        reason: String,
    },
    #[serde(rename = "heartbeat")]
    Heartbeat {
        long_count: usize,
        short_count: usize,
        ws_uptime_h: f64,
        reconnect_count: u32,
    },
    #[serde(rename = "signal")]
    Signal {
        symbol: String,
        direction: String,
        strategy: String,
        confidence: f64,
        entry_price: f64,
        sl_price: f64,
    },
    #[serde(rename = "pong")]
    Pong,
}

/// IPC Server
pub struct IpcBridge {
    /// Channel for commands received from Python
    pub cmd_tx: broadcast::Sender<Command>,
    /// Channel for events to send to Python  
    _event_rx: broadcast::Receiver<Event>,
    pub event_tx: broadcast::Sender<Event>,
}

impl IpcBridge {
    pub fn new() -> Self {
        let (cmd_tx, _) = broadcast::channel(256);
        let (event_tx, event_rx) = broadcast::channel(256);
        Self { cmd_tx, _event_rx: event_rx, event_tx }
    }

    /// Start the TCP server and handle connections
    pub async fn run(self: Arc<Self>) {
        let addr = format!("0.0.0.0:{}", IPC_PORT);
        let listener = match TcpListener::bind(&addr).await {
            Ok(l) => {
                info!("🌐 IPC Bridge listening on {}", addr);
                l
            }
            Err(e) => {
                error!("Failed to bind IPC port {}: {} (engine continues without IPC)", IPC_PORT, e);
                // Don't return — that would kill the engine via tokio::select!
                // Instead, sleep forever so the engine keeps running
                loop { tokio::time::sleep(tokio::time::Duration::from_secs(3600)).await; }
            }
        };

        loop {
            match listener.accept().await {
                Ok((stream, peer)) => {
                    info!("📡 IPC client connected: {}", peer);
                    let bridge = Arc::clone(&self);
                    tokio::spawn(async move {
                        bridge.handle_client(stream).await;
                        info!("📡 IPC client disconnected: {}", peer);
                    });
                }
                Err(e) => {
                    warn!("IPC accept error: {}", e);
                }
            }
        }
    }

    async fn handle_client(&self, stream: tokio::net::TcpStream) {
        let (reader, mut writer) = stream.into_split();
        let mut buf_reader = BufReader::new(reader);
        let mut event_rx = self.event_tx.subscribe();

        // Task 1: Read commands from Python
        let cmd_tx = self.cmd_tx.clone();
        let read_task = tokio::spawn(async move {
            let mut line = String::new();
            loop {
                line.clear();
                match buf_reader.read_line(&mut line).await {
                    Ok(0) => break, // EOF
                    Ok(_) => {
                        let trimmed = line.trim();
                        if trimmed.is_empty() {
                            continue;
                        }
                        match serde_json::from_str::<Command>(trimmed) {
                            Ok(cmd) => {
                                let _ = cmd_tx.send(cmd);
                            }
                            Err(e) => {
                                warn!("IPC parse error: {} | raw: {}", e, trimmed);
                            }
                        }
                    }
                    Err(e) => {
                        warn!("IPC read error: {}", e);
                        break;
                    }
                }
            }
        });

        // Task 2: Send events to Python
        let write_task = tokio::spawn(async move {
            loop {
                match event_rx.recv().await {
                    Ok(event) => {
                        let json = match serde_json::to_string(&event) {
                            Ok(j) => j,
                            Err(_) => continue,
                        };
                        let line = format!("{}\n", json);
                        if writer.write_all(line.as_bytes()).await.is_err() {
                            break;
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        warn!("IPC event channel lagged by {}", n);
                    }
                    Err(_) => break,
                }
            }
        });

        tokio::select! {
            _ = read_task => {}
            _ = write_task => {}
        }
    }
}
