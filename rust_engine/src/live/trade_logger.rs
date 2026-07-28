///! Trade Logger — JSONL Append-Only Trade Journal
///! ================================================
///! Records every trade entry and exit as JSON Lines to `data/trade_log.jsonl`.
///! Designed for:
///!   - Phase 11: Trade analytics and debugging
///!   - Phase 11.4: Meta-Labeling RL Feedback (aggregate_journal.py reads this)
///!   - Phase 12-13: Reserved fields for whale_tag, liq_zscore, correlation_warn
///!
///! Each trade produces TWO events:
///!   1. ENTRY — when position is opened (snap all entry metrics)
///!   2. EXIT  — when position is closed (PnL, MFE, MAE, exit reason)

use std::fs::OpenOptions;
use std::io::Write;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;
use tracing::{info, warn};

// ── Log file path ───────────────────────────────────────────────────────────

const TRADE_LOG_PATH: &str = "data/trade_log.jsonl";

// ── Data Types ──────────────────────────────────────────────────────────────

/// ENTRY event — snapshot of all conditions at trade opening
#[derive(Debug, Serialize)]
pub struct TradeEntry {
    pub event: &'static str,               // "ENTRY"
    pub ts: String,                         // ISO 8601 timestamp
    pub trade_id: String,                   // Unique ID: "{symbol}_{ts_unix}"
    pub symbol: String,
    pub strategy: String,
    pub direction: String,                  // "LONG" / "SHORT"
    pub entry_price: f64,
    pub sl_price: f64,
    pub tp_price: f64,
    pub quantity: f64,
    pub risk_dist: f64,
    // ML metrics
    pub ml_prob: Option<f64>,               // ML prediction probability
    // Spot Probe (Phase 11)
    pub spot_probe: String,                 // "confirmed" / "neutral" / "blocked" / "unavailable"
    // Wall state
    pub wall_side: Option<String>,          // "Bid" / "Ask"
    pub wall_price: Option<f64>,
    pub wall_size_usd: Option<f64>,
    pub wall_age_h: Option<f64>,
    pub wall_eaten_pct: Option<f64>,
    // Order flow
    pub cvd_delta: Option<f64>,
    pub imbalance_ratio: Option<f64>,
    pub tape_speed: Option<f64>,
    // Phase 12 reserved
    pub whale_tag: Option<String>,
    // Phase 13 reserved
    pub liq_zscore: Option<f64>,
    pub correlation_warn: Option<f64>,
    // Phase 11.4 reserved
    pub meta_warn_score: Option<f64>,
}

/// EXIT event — trade result with MFE/MAE
#[derive(Debug, Serialize)]
pub struct TradeExit {
    pub event: &'static str,               // "EXIT"
    pub ts: String,
    pub trade_id: String,                   // Must match the ENTRY trade_id
    pub symbol: String,
    pub strategy: String,
    pub direction: String,
    pub entry_price: f64,
    pub exit_price: f64,
    pub exit_reason: String,               // "TP" / "SL" / "Trailing" / "SmartExit" / "PanicSell"
    pub pnl_r: f64,                        // Profit in R-units
    pub pnl_pct: f64,                      // Profit in %
    pub duration_secs: u64,                // How long the trade lasted
    // Excursion analysis
    pub mfe_pct: f64,                      // Maximum Favorable Excursion (best unrealized %)
    pub mae_pct: f64,                      // Maximum Adverse Excursion (worst unrealized %)
    // Phase 12 reserved
    pub whale_tag: Option<String>,
    // Phase 13 reserved
    pub liq_zscore: Option<f64>,
}

// ── Public API ──────────────────────────────────────────────────────────────

/// Generate a unique trade ID from symbol and current timestamp
pub fn generate_trade_id(symbol: &str) -> String {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    format!("{}_{}", symbol.replace("/", "").replace("_", ""), ts)
}

/// Get current ISO 8601 timestamp string
pub fn now_iso() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    // Simple ISO format without chrono dependency
    let days = secs / 86400;
    let time_secs = secs % 86400;
    let hours = time_secs / 3600;
    let mins = (time_secs % 3600) / 60;
    let s = time_secs % 60;

    // Rough date calculation (good enough for logging)
    let mut y = 1970i64;
    let mut remaining_days = days as i64;
    loop {
        let days_in_year = if y % 4 == 0 && (y % 100 != 0 || y % 400 == 0) { 366 } else { 365 };
        if remaining_days < days_in_year { break; }
        remaining_days -= days_in_year;
        y += 1;
    }
    let months_days = if y % 4 == 0 && (y % 100 != 0 || y % 400 == 0) {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };
    let mut m = 0;
    for (i, &d) in months_days.iter().enumerate() {
        if remaining_days < d as i64 { m = i + 1; break; }
        remaining_days -= d as i64;
    }
    let day = remaining_days + 1;

    format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z", y, m, day, hours, mins, s)
}

/// Log a trade ENTRY event to the JSONL file
pub fn log_entry(entry: &TradeEntry) {
    append_json(entry, "ENTRY", &entry.symbol);
}

/// Log a trade EXIT event to the JSONL file
pub fn log_exit(exit: &TradeExit) {
    append_json(exit, "EXIT", &exit.symbol);
}

// ── Internal ────────────────────────────────────────────────────────────────

fn append_json<T: Serialize>(record: &T, event_type: &str, symbol: &str) {
    match serde_json::to_string(record) {
        Ok(json_line) => {
            // Ensure data directory exists
            let _ = std::fs::create_dir_all("data");

            match OpenOptions::new()
                .create(true)
                .append(true)
                .open(TRADE_LOG_PATH)
            {
                Ok(mut file) => {
                    if let Err(e) = writeln!(file, "{}", json_line) {
                        warn!("📝 TradeLogger: failed to write {} {}: {}", event_type, symbol, e);
                    } else {
                        info!("📝 TradeLogger: {} {} logged to {}", event_type, symbol, TRADE_LOG_PATH);
                    }
                }
                Err(e) => {
                    warn!("📝 TradeLogger: failed to open {}: {}", TRADE_LOG_PATH, e);
                }
            }
        }
        Err(e) => {
            warn!("📝 TradeLogger: failed to serialize {} {}: {}", event_type, symbol, e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_trade_id() {
        let id = generate_trade_id("BTC_USDT");
        assert!(id.starts_with("BTCUSDT_"));
        assert!(id.len() > 10);
    }

    #[test]
    fn test_now_iso() {
        let ts = now_iso();
        assert!(ts.starts_with("20")); // 2020+
        assert!(ts.ends_with("Z"));
        assert!(ts.contains("T"));
    }

    #[test]
    fn test_serialize_entry() {
        let entry = TradeEntry {
            event: "ENTRY",
            ts: "2026-03-14T01:30:00Z".to_string(),
            trade_id: "BTCUSDT_12345".to_string(),
            symbol: "BTC_USDT".to_string(),
            strategy: "KnifeCatcher_ML".to_string(),
            direction: "LONG".to_string(),
            entry_price: 65000.0,
            sl_price: 64500.0,
            tp_price: 66000.0,
            quantity: 0.01,
            risk_dist: 500.0,
            ml_prob: Some(0.72),
            spot_probe: "confirmed".to_string(),
            wall_side: Some("Bid".to_string()),
            wall_price: Some(64800.0),
            wall_size_usd: Some(120000.0),
            wall_age_h: Some(4.2),
            wall_eaten_pct: Some(0.55),
            cvd_delta: Some(1.45),
            imbalance_ratio: Some(2.1),
            tape_speed: Some(120.0),
            whale_tag: None,
            liq_zscore: None,
            correlation_warn: None,
            meta_warn_score: None,
        };
        let json = serde_json::to_string(&entry).unwrap();
        assert!(json.contains("ENTRY"));
        assert!(json.contains("65000"));
        assert!(json.contains("whale_tag"));
    }
}
