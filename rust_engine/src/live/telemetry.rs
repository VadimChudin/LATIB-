use std::net::SocketAddr;
// use std::sync::Arc;
use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
// use serde::Serialize;
use serde_json::json;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::broadcast;
use tokio_tungstenite::accept_async;
use tokio_tungstenite::tungstenite::Message;
use tracing::{info, warn, error};

use super::wall_tracker::WallStore;
use super::tape_reader::TapeStore;

/// Starts the WebSocket Telemetry server on port 8081.
/// Broadcasts WallTracker and TapeReader state every 500ms to all connected clients.
pub async fn start_server(wall_store: WallStore, tape_store: TapeStore, symbols: Vec<String>) {
    let addr = "127.0.0.1:8081";
    let listener = match TcpListener::bind(&addr).await {
        Ok(l) => l,
        Err(e) => {
            error!("❌ Failed to bind Telemetry server to {}: {}", addr, e);
            return;
        }
    };
    info!("📡 AEGIS Telemetry Server listening on ws://{}", addr);

    // Channel for broadcasting state to all connected websocket clients
    let (tx, _) = broadcast::channel::<String>(100);

    // Spawn the aggregator/broadcaster task
    let tx_clone = tx.clone();
    let symbols_clone = symbols.clone();
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_millis(500)).await;

            // Only broadcast if there are listeners (receiver count > 0)
            if tx_clone.receiver_count() == 0 {
                continue;
            }

            let mut heatmap_data = Vec::new();
            let mut screener_data = Vec::new();

            // 1. Collect Heatmap data for top symbols (like BTC, ETH, SOL)
            let watch_symbols = symbols_clone.iter().take(5).collect::<Vec<_>>();
            for sym in watch_symbols {
                let wall_data = match wall_store.get(sym) {
                    Some(snap) => {
                        let ask = snap.ask_walls().into_iter().map(|w| {
                            json!({ "p": w.price, "s": w.current_size_usd, "e": w.eaten_pct(), "a": w.age_hours() })
                        }).collect::<Vec<_>>();
                        let bid = snap.bid_walls().into_iter().map(|w| {
                            json!({ "p": w.price, "s": w.current_size_usd, "e": w.eaten_pct(), "a": w.age_hours() })
                        }).collect::<Vec<_>>();
                        json!({ "ask": ask, "bid": bid })
                    }
                    None => json!({ "ask": [], "bid": [] }),
                };

                let flow_data = match tape_store.get(sym) {
                    Some(state) => {
                        json!({
                            "delta": state.normalized_delta(),
                            "imbalance": state.imbalance_ratio(),
                            "speed": state.tape_speed(),
                            "accel": state.speed_acceleration(),
                        })
                    }
                    None => json!({ "delta": 0.0, "imbalance": 1.0, "speed": 0.0, "accel": 1.0 }),
                };

                heatmap_data.push(json!({
                    "symbol": sym,
                    "walls": wall_data,
                    "flow": flow_data
                }));
            }

            // 2. Collect Screener data: Top closest/biggest walls across ALL symbols
            for sym in &symbols_clone {
                if let Some(snap) = wall_store.get(sym) {
                    let flow_speed = tape_store.get(sym).map(|s| s.tape_speed()).unwrap_or(0.0);
                    let current_price = tape_store.get(sym).map(|s| s.last_price).unwrap_or(0.0);
                    
                    if current_price > 0.0 {
                        // Grab the single closest/biggest ASK wall (Resistance)
                        if let Some((w, dist)) = snap.ask_walls().into_iter()
                            .map(|w| (w, (w.price - current_price).abs() / current_price * 100.0))
                            .filter(|(_, d)| *d < 3.0)
                            .max_by_key(|(w, _)| w.current_size_usd as u64) 
                        {
                            screener_data.push(json!({
                                "sym": sym,
                                "side": "ASK",
                                "price": w.price,
                                "dist": dist,
                                "size": w.current_size_usd,
                                "eaten": w.eaten_pct(),
                                "speed": flow_speed
                            }));
                        }
                        
                        // Grab the single closest/biggest BID wall (Support)
                        if let Some((w, dist)) = snap.bid_walls().into_iter()
                            .map(|w| (w, (w.price - current_price).abs() / current_price * 100.0))
                            .filter(|(_, d)| *d < 3.0)
                            .max_by_key(|(w, _)| w.current_size_usd as u64)
                        {
                            screener_data.push(json!({
                                "sym": sym,
                                "side": "BID",
                                "price": w.price,
                                "dist": dist,
                                "size": w.current_size_usd,
                                "eaten": w.eaten_pct(),
                                "speed": flow_speed
                            }));
                        }
                    }
                }
            }

            // Sort screener data by size (largest walls first) and take top 25
            screener_data.sort_by(|a, b| {
                let s_a = a["size"].as_f64().unwrap_or(0.0);
                let s_b = b["size"].as_f64().unwrap_or(0.0);
                s_b.partial_cmp(&s_a).unwrap_or(std::cmp::Ordering::Equal)
            });
            screener_data.truncate(25);

            if !heatmap_data.is_empty() || !screener_data.is_empty() {
                // Wrap in a type identifier so AEGIS Terminal knows what to do
                let msg_obj = json!({
                    "type": "telemetry_update",
                    "heatmap": heatmap_data,
                    "screener": screener_data
                });
                
                if let Ok(msg_str) = serde_json::to_string(&msg_obj) {
                    let _ = tx_clone.send(msg_str);
                }
            }
        }
    });

    // Accept incoming WebSocket connections
    while let Ok((stream, addr)) = listener.accept().await {
        let tx_client = tx.clone();
        tokio::spawn(handle_connection(stream, addr, tx_client));
    }
}

async fn handle_connection(stream: TcpStream, addr: SocketAddr, tx: broadcast::Sender<String>) {
    info!("📡 Client connected to telemetry: {}", addr);
    
    let ws_stream = match accept_async(stream).await {
        Ok(ws) => ws,
        Err(e) => {
            warn!("⚠️ Telemetry WS handshake failed for {}: {}", addr, e);
            return;
        }
    };

    let (mut ws_sender, mut ws_receiver) = ws_stream.split();
    let mut rx = tx.subscribe();

    // Send loop: forward broadcast channel messages to this client
    let send_task = tokio::spawn(async move {
        while let Ok(msg) = rx.recv().await {
            if ws_sender.send(Message::Text(msg)).await.is_err() {
                break; // Client disconnected
            }
        }
    });

    // Receive loop: Just to detect disconnection
    let recv_task = tokio::spawn(async move {
        while let Some(msg) = ws_receiver.next().await {
            if msg.is_err() || msg.unwrap().is_close() {
                break;
            }
        }
    });

    tokio::select! {
        _ = send_task => {},
        _ = recv_task => {},
    }

    info!("📡 Client disconnected from telemetry: {}", addr);
}
