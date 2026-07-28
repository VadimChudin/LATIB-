use std::sync::Arc;
use tokio::sync::Mutex;

/// Shared live trading statistics, updated by Orchestrator, read by TG bot.
#[derive(Debug, Clone)]
pub struct LiveStats {
    pub daily_pnl_r: f64,
    pub total_pnl_r: f64,
    pub daily_trades: usize,
    pub total_trades: usize,
    pub wins: usize,
    pub losses: usize,
    pub be_count: usize,
    pub best_trade_r: f64,
    pub worst_trade_r: f64,
    pub open_positions: Vec<OpenPosInfo>,
    pub equity: f64,
    pub risk_per_trade: f64,     // in USD (equity * 0.02)
}

#[derive(Debug, Clone)]
pub struct OpenPosInfo {
    pub symbol: String,
    pub direction: String,
    pub entry_price: f64,
    pub current_price: f64,
    pub pnl_pct: f64,
    pub duration_secs: u64,
    pub strategy: String,
    pub size: f64,
    pub target_size: f64,
}

pub type SharedStats = Arc<Mutex<LiveStats>>;

pub fn new_shared() -> SharedStats {
    Arc::new(Mutex::new(LiveStats {
        daily_pnl_r: 0.0,
        total_pnl_r: 0.0,
        daily_trades: 0,
        total_trades: 0,
        wins: 0,
        losses: 0,
        be_count: 0,
        best_trade_r: f64::NEG_INFINITY,
        worst_trade_r: f64::INFINITY,
        open_positions: Vec::new(),
        equity: 0.0,
        risk_per_trade: 0.0,
    }))
}
