use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use tracing::{info, warn};

#[derive(Debug, Deserialize)]
#[serde(tag = "event")]
pub enum JournalEvent {
    #[serde(rename = "ENTRY")]
    Entry {
        trade_id: String,
        symbol: String,
        strategy: String,
        entry_price: f64,
        ts: String,
    },
    #[serde(rename = "EXIT")]
    Exit {
        trade_id: String,
        pnl_r: f64,
        exit_reason: String,
    },
}

#[derive(Debug, Clone, Default)]
pub struct StrategyStats {
    pub total_trades: usize,
    pub wins: usize,
    pub total_pnl_r: f64,
    pub win_rate: f64,
    pub current_streak: i32, // + for wins, - for losses
    pub risk_multiplier: f64,
}

pub struct JournalAnalyzer {
    pub stats_by_strategy: HashMap<String, StrategyStats>,
    /// Threshold: if WinRate < 35% AND trades > 10, block it.
    pub block_threshold_wr: f64,
    pub min_trades_to_block: usize,
}

impl JournalAnalyzer {
    pub fn new() -> Self {
        Self {
            stats_by_strategy: HashMap::new(),
            block_threshold_wr: 0.0,
            min_trades_to_block: 5,
        }
    }

    pub fn reload_from_file(&mut self, path: &str) {
        let file = match File::open(path) {
            Ok(f) => f,
            Err(_) => {
                warn!("📝 JournalAnalyzer: No trade_log.jsonl found at {}", path);
                return;
            }
        };

        let reader = BufReader::new(file);
        let mut temp_stats: HashMap<String, (usize, usize, f64, i32)> = HashMap::new(); // strategy -> (total, wins, pnl, streak)
        
        // We'll track trade_id -> strategy to link Exit with Strategy
        let mut trade_map: HashMap<String, String> = HashMap::new();

        for line in reader.lines() {
            if let Ok(l) = line {
                if let Ok(event) = serde_json::from_str::<JournalEvent>(&l) {
                    match event {
                        JournalEvent::Entry { trade_id, strategy, .. } => {
                            trade_map.insert(trade_id, strategy);
                        }
                        JournalEvent::Exit { trade_id, pnl_r, .. } => {
                            if let Some(strategy) = trade_map.get(&trade_id) {
                                let entry = temp_stats.entry(strategy.clone()).or_insert((0, 0, 0.0, 0));
                                entry.0 += 1;
                                if pnl_r > 0.0 {
                                    entry.1 += 1;
                                    // Reset or increment win streak
                                    if entry.3 < 0 { entry.3 = 1; } else { entry.3 += 1; }
                                } else if pnl_r < -0.1 {
                                    // Reset or increment loss streak
                                    if entry.3 > 0 { entry.3 = -1; } else { entry.3 -= 1; }
                                }
                                entry.2 += pnl_r;
                            }
                        }
                    }
                }
            }
        }

        // Finalize stats
        self.stats_by_strategy.clear();
        for (strat, (total, wins, pnl, streak)) in temp_stats {
            let wr = if total > 0 { (wins as f64 / total as f64) * 100.0 } else { 0.0 };
            
            // Calc risk multiplier: 1.0 base, 0.5 if streak <= -2, 0.25 if streak <= -4
            let risk_mult = if streak <= -4 { 0.25 }
                           else if streak <= -2 { 0.5 }
                           else { 1.0 };

            self.stats_by_strategy.insert(strat, StrategyStats {
                total_trades: total,
                wins,
                total_pnl_r: pnl,
                win_rate: wr,
                current_streak: streak,
                risk_multiplier: risk_mult,
            });
        }

        info!("🧠 Master Advisor: Analyzed {} strategies from log", self.stats_by_strategy.len());
    }

    pub fn is_strategy_allowed(&self, strategy: &str) -> bool {
        if let Some(stats) = self.stats_by_strategy.get(strategy) {
            if stats.total_trades >= self.min_trades_to_block {
                if stats.win_rate < self.block_threshold_wr {
                    warn!("🛡️ Master Advisor: BLOCKED strategy {} (WR={:.1}% < {}%)", 
                        strategy, stats.win_rate, self.block_threshold_wr);
                    return false;
                }
            }
        }
        true
    }

    pub fn get_risk_multiplier(&self, strategy: &str) -> f64 {
        self.stats_by_strategy.get(strategy)
            .map(|s| s.risk_multiplier)
            .unwrap_or(1.0)
    }
}
