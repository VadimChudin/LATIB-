///! HFT Logger — High-Frequency Tick & Order Book Snapshotter
///! =========================================================
///! Records sub-second metrics during critical "Density Approach" events.
///! Used for HFT strategy playback, micro-structure ML training, and debugging.
///! Data is saved to `data/hft_snapshots.jsonl`.

use std::fs::OpenOptions;
use std::io::Write;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use tracing::{info, warn};

const HFT_LOG_PATH: &str = "data/hft_snapshots.jsonl";

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct HftSnapshot {
    pub event_id: String,       // Unique ID for the tracking session
    pub ts_ms: u128,            // Millisecond unix timestamp
    pub symbol: String,
    pub price: f64,
    pub direction: String,      // "Long" / "Short"
    pub strategy: String,       // "breakout" / "knife"
    
    // Tape metrics
    pub tape_speed: f64,
    pub cvd: f64,
    pub delta: f64,
    pub whale_buys: usize,
    pub whale_sells: usize,

    // Order Book metrics (nearest wall)
    pub wall_dist_pct: f64,
    pub wall_size_usd: f64,
    pub wall_eaten_pct: f64,
    
    // Absorber state
    pub score: i32,
    pub extensions: u32,
    pub status: String,         // "Tracking" / "Fired" / "Timeout" / "Reject"
}

/// Log a high-frequency snapshot to the JSONL file
pub fn append_snapshot(snapshot: &HftSnapshot) {
    if let Ok(json_line) = serde_json::to_string(snapshot) {
        let _ = std::fs::create_dir_all("data");

        match OpenOptions::new()
            .create(true)
            .append(true)
            .open(HFT_LOG_PATH)
        {
            Ok(mut file) => {
                let _ = writeln!(file, "{}", json_line);
            }
            Err(e) => {
                warn!("🧱 HFTLogger: failed to open {}: {}", HFT_LOG_PATH, e);
            }
        }
    }
}

/// Helper to get current ms
pub fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

/// Special meta-row to mark the start/end of an event in the log
pub fn log_event_meta(event_id: &str, symbol: &str, status: &str) {
    let meta = HftSnapshot {
        event_id: event_id.to_string(),
        ts_ms: now_ms(),
        symbol: symbol.to_string(),
        price: 0.0,
        direction: "".to_string(),
        strategy: "".to_string(),
        tape_speed: 0.0,
        cvd: 0.0,
        delta: 0.0,
        whale_buys: 0,
        whale_sells: 0,
        wall_dist_pct: 0.0,
        wall_size_usd: 0.0,
        wall_eaten_pct: 0.0,
        score: 0,
        extensions: 0,
        status: status.to_string(),
    };
    append_snapshot(&meta);
    info!("🧱 HFTLogger: Event {} MARKED as {}", event_id, status);
}
