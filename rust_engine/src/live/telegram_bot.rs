use tokio::sync::mpsc;
use reqwest::Client;
use tracing::{info, warn, error};
use super::live_stats::SharedStats;

/// Telegram Bot for AEGIS Engine
/// Runs asynchronously in the background:
/// 1. Processes outgoing message queue (ENTRY/EXIT alerts)
/// 2. Polls Telegram getUpdates for /stats command
pub struct TelegramBot {
    token: String,
    chat_id: String,
    client: Client,
    rx: mpsc::Receiver<String>,
    stats: SharedStats,
    last_update_id: i64,
}

impl TelegramBot {
    pub fn new(token: String, chat_id: String, rx: mpsc::Receiver<String>, stats: SharedStats) -> Self {
        Self {
            token,
            chat_id,
            client: Client::new(),
            rx,
            stats,
            last_update_id: 0,
        }
    }

    /// Background task: send outgoing messages + poll incoming commands
    pub async fn run(mut self) {
        info!("📱 Telegram Bot background task started (with /stats polling)");
        
        self.send_startup_message().await;

        let mut poll_interval = tokio::time::interval(tokio::time::Duration::from_secs(3));

        loop {
            tokio::select! {
                // Outgoing messages (ENTRY/EXIT alerts)
                msg = self.rx.recv() => {
                    match msg {
                        Some(text) => {
                            self.send_telegram_message(&text).await;
                            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                        }
                        None => break, // Channel closed
                    }
                }
                // Incoming commands (poll every 3s)
                _ = poll_interval.tick() => {
                    self.poll_commands().await;
                }
            }
        }
        
        info!("📱 Telegram Bot background task exiting");
    }
    
    async fn send_startup_message(&self) {
        let msg = "🤖 *AEGIS Rust LiveExecutor v1\\.0* \\- СТАРТ ОСНОВНОГО ЦИКЛА\n\nМодуль уведомлений успешно: ✅ Активен\\. Ждем сигналов\\.\n\nКоманды:\n/stats \\- статистика торговли";
        self.send_telegram_message(msg).await;
    }

    /// Poll Telegram getUpdates for /stats command
    async fn poll_commands(&mut self) {
        let url = format!(
            "https://api.telegram.org/bot{}/getUpdates?offset={}&timeout=1&allowed_updates=[\"message\"]",
            self.token, self.last_update_id + 1
        );

        let resp = match self.client.get(&url).send().await {
            Ok(r) => r,
            Err(e) => {
                warn!("[TG] Poll error: {}", e);
                return;
            }
        };

        let body: serde_json::Value = match resp.json().await {
            Ok(v) => v,
            Err(e) => {
                warn!("[TG] Parse error: {}", e);
                return;
            }
        };

        if let Some(results) = body.get("result").and_then(|r| r.as_array()) {
            for update in results {
                if let Some(update_id) = update.get("update_id").and_then(|u| u.as_i64()) {
                    self.last_update_id = update_id;
                }

                let text = update
                    .get("message")
                    .and_then(|m| m.get("text"))
                    .and_then(|t| t.as_str())
                    .unwrap_or("");

                let chat_id = update
                    .get("message")
                    .and_then(|m| m.get("chat"))
                    .and_then(|c| c.get("id"))
                    .and_then(|id| id.as_i64());

                if text.starts_with("/stats") {
                    if let Some(cid) = chat_id {
                        let msg = self.format_stats().await;
                        self.send_to_chat(cid, &msg).await;
                    }
                }
            }
        }
    }

    /// Format beautiful stats message from SharedStats
    async fn format_stats(&self) -> String {
        let s = self.stats.lock().await;
        
        let win_rate = if s.total_trades > 0 {
            s.wins as f64 / s.total_trades as f64 * 100.0
        } else {
            0.0
        };

        let pnl_usd = s.total_pnl_r * s.risk_per_trade;
        let daily_usd = s.daily_pnl_r * s.risk_per_trade;

        let pnl_icon = if s.total_pnl_r >= 0.0 { "📈" } else { "📉" };
        let daily_icon = if s.daily_pnl_r >= 0.0 { "🟢" } else { "🔴" };

        let best_str = if s.best_trade_r > f64::NEG_INFINITY {
            format!("{:+.2}R", s.best_trade_r)
        } else { "—".to_string() };
        let worst_str = if s.worst_trade_r < f64::INFINITY {
            format!("{:+.2}R", s.worst_trade_r)
        } else { "—".to_string() };

        // Open positions section
        let positions_text = if s.open_positions.is_empty() {
            "   Нет открытых позиций".to_string()
        } else {
            s.open_positions.iter().map(|p| {
                let fill_pct = if p.target_size > 0.0 {
                    (p.size / p.target_size * 100.0).round()
                } else {
                    100.0
                };
                let icon = if p.direction == "LONG" { "🟢" } else { "🔴" };
                let dur_min = p.duration_secs / 60;
                format!("   {} {} {} @ ${:.4} ({:+.2}%) {}мин\n      └ Объем: {:.4} ({}%)",
                    icon, p.symbol.replace("/USDT", ""), p.direction,
                    p.entry_price, p.pnl_pct, dur_min, p.size, fill_pct)
            }).collect::<Vec<_>>().join("\n")
        };

        // Build message (plain text, no markdown to avoid escaping issues)
        format!(
            "📊 AEGIS Статистика\n\
            ━━━━━━━━━━━━━━━━━━━\n\
            {} Всего PnL: {:+.2}R (${:+.1})\n\
            {} Сегодня: {:+.2}R (${:+.1}) | {} сделок\n\
            \n\
            🎯 Win Rate: {:.1}% ({}W / {}L / {}BE)\n\
            🏆 Best: {} | Worst: {}\n\
            💼 Equity: ${:.0} | Risk: ${:.1}/trade\n\
            \n\
            📂 Открытые позиции ({}):\n\
            {}",
            pnl_icon, s.total_pnl_r, pnl_usd,
            daily_icon, s.daily_pnl_r, daily_usd, s.daily_trades,
            win_rate, s.wins, s.losses, s.be_count,
            best_str, worst_str,
            s.equity, s.risk_per_trade,
            s.open_positions.len(),
            positions_text
        )
    }

    /// Send message to a specific chat (for command responses)
    async fn send_to_chat(&self, chat_id: i64, text: &str) {
        let url = format!("https://api.telegram.org/bot{}/sendMessage", self.token);
        
        let mut payload = serde_json::Map::new();
        payload.insert("chat_id".to_string(), serde_json::Value::Number(serde_json::Number::from(chat_id)));
        payload.insert("text".to_string(), serde_json::Value::String(text.to_string()));
        payload.insert("disable_web_page_preview".to_string(), serde_json::Value::Bool(true));

        match self.client.post(&url).json(&payload).send().await {
            Ok(resp) if !resp.status().is_success() => {
                let err = resp.text().await.unwrap_or_default();
                warn!("[TG] Stats reply failed: {}", err);
            }
            Err(e) => error!("[TG] Stats reply error: {}", e),
            _ => {}
        }
    }

    async fn send_telegram_message(&self, text: &str) {
        let url = format!("https://api.telegram.org/bot{}/sendMessage", self.token);
        
        let mut payload = serde_json::Map::new();
        payload.insert("chat_id".to_string(), serde_json::Value::String(self.chat_id.clone()));
        payload.insert("text".to_string(), serde_json::Value::String(text.to_string()));
        payload.insert("parse_mode".to_string(), serde_json::Value::String("MarkdownV2".to_string()));
        payload.insert("disable_web_page_preview".to_string(), serde_json::Value::Bool(true));

        match self.client.post(&url).json(&payload).send().await {
            Ok(resp) => {
                let status = resp.status();
                if !status.is_success() {
                    let err_text = resp.text().await.unwrap_or_default();
                    warn!("[TG] Failed to send message. HTTP status: {} Body: {}", status, err_text);
                }
            }
            Err(e) => {
                error!("[TG] Request error: {}", e);
            }
        }
    }
}

/// Helper function to escape special characters for Telegram MarkdownV2
pub fn escape_markdown_v2(text: &str) -> String {
    let chars_to_escape = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!'];
    let mut escaped = String::with_capacity(text.len() + 10);
    
    for c in text.chars() {
        if chars_to_escape.contains(&c) {
            escaped.push('\\');
        }
        escaped.push(c);
    }
    escaped
}

/// Helper to format a string but specifically only escaping problematic chars from numbers/tickers
pub fn escape_variable(val: &str) -> String {
    val.replace("-", "\\-")
       .replace(".", "\\.")
       .replace("!", "\\!")
       .replace("_", "\\_")
       .replace("|", "\\|")
       .replace("(", "\\(")
       .replace(")", "\\)")
}
