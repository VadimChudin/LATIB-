///! Binance Futures Order Router
///! =============================
///! Places and manages orders via Binance Futures REST API.
///! Supports: Market, Limit, and Stop-Market orders.
///! Authentication: HMAC-SHA256 signed requests.
///!
///! In PAPER mode, simulates orders locally without hitting the API.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use hmac::{Hmac, Mac};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use tracing::{info, warn};

type HmacSha256 = Hmac<Sha256>;

// ── Config ──────────────────────────────────────────────────────────────────

const BINANCE_FAPI_URL: &str = "https://fapi.binance.com";

/// Order side
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub enum Side {
    #[serde(rename = "BUY")]
    Buy,
    #[serde(rename = "SELL")]
    Sell,
}

impl Side {
    pub fn as_str(&self) -> &'static str {
        match self {
            Side::Buy => "BUY",
            Side::Sell => "SELL",
        }
    }
}

/// Order type
#[derive(Debug, Clone, Copy)]
pub enum OrderType {
    Market,
    Limit,
    StopMarket,
}

impl OrderType {
    pub fn _as_str(&self) -> &'static str {
        match self {
            OrderType::Market => "MARKET",
            OrderType::Limit => "LIMIT",
            OrderType::StopMarket => "STOP_MARKET",
        }
    }
}

/// Result of placing an order
#[derive(Debug, Clone, Deserialize)]
pub struct OrderResult {
    #[serde(rename = "orderId")]
    pub order_id: Option<i64>,
    pub symbol: Option<String>,
    pub status: Option<String>,
    #[serde(rename = "avgPrice")]
    pub avg_price: Option<String>,
    #[serde(rename = "executedQty")]
    pub executed_qty: Option<String>,
    pub side: Option<String>,
    // Error fields
    pub code: Option<i32>,
    pub msg: Option<String>,
}

/// Paper trade result (simulated)
#[derive(Debug, Clone)]
pub struct PaperResult {
    pub symbol: String,
    pub side: Side,
    pub quantity: f64,
    pub fill_price: f64,
    pub slippage_pct: f64,
}

// ── Order Router ────────────────────────────────────────────────────────────

pub struct OrderRouter {
    client: Client,
    api_key: String,
    api_secret: String,
    paper_mode: bool,
}

impl OrderRouter {
    pub fn new(api_key: String, api_secret: String, paper_mode: bool) -> Self {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .build()
            .expect("Failed to create HTTP client");

        info!(
            "📋 Order Router initialized ({})",
            if paper_mode { "PAPER" } else { "LIVE" }
        );

        Self {
            client,
            api_key,
            api_secret,
            paper_mode,
        }
    }

    /// Place a market order
    pub async fn market_order(
        &self,
        symbol: &str,
        side: Side,
        quantity: f64,
        current_price: f64,
    ) -> Result<OrderResult, String> {
        if self.paper_mode {
            let slippage = 0.0003; // 0.03%
            let fill_price = match side {
                Side::Buy => current_price * (1.0 + slippage),
                Side::Sell => current_price * (1.0 - slippage),
            };
            info!(
                "📄 PAPER {} {} {:.6} @ {:.4} (slip={:.2}%)",
                side.as_str(),
                symbol,
                quantity,
                fill_price,
                slippage * 100.0
            );
            return Ok(OrderResult {
                order_id: Some(chrono::Utc::now().timestamp()),
                symbol: Some(symbol.to_string()),
                status: Some("FILLED".to_string()),
                avg_price: Some(format!("{:.8}", fill_price)),
                executed_qty: Some(format!("{:.8}", quantity)),
                side: Some(side.as_str().to_string()),
                code: None,
                msg: None,
            });
        }

        // Real order
        let clean_symbol = symbol.replace("/", "").replace("_", "");
        let mut params = HashMap::new();
        params.insert("symbol", clean_symbol.clone());
        params.insert("side", side.as_str().to_string());
        params.insert("type", "MARKET".to_string());
        params.insert("quantity", format!("{:.8}", quantity));

        self.signed_request("POST", "/fapi/v1/order", params).await
    }

    /// Place a stop-market order (for SL)
    pub async fn stop_market_order(
        &self,
        symbol: &str,
        side: Side,
        quantity: f64,
        stop_price: f64,
    ) -> Result<OrderResult, String> {
        if self.paper_mode {
            info!(
                "📄 PAPER STOP {} {} {:.6} @ stop={:.4}",
                side.as_str(),
                symbol,
                quantity,
                stop_price
            );
            return Ok(OrderResult {
                order_id: Some(chrono::Utc::now().timestamp()),
                symbol: Some(symbol.to_string()),
                status: Some("NEW".to_string()),
                avg_price: None,
                executed_qty: Some(format!("{:.8}", quantity)),
                side: Some(side.as_str().to_string()),
                code: None,
                msg: None,
            });
        }

        let clean_symbol = symbol.replace("/", "").replace("_", "");
        let mut params = HashMap::new();
        params.insert("symbol", clean_symbol.clone());
        params.insert("side", side.as_str().to_string());
        params.insert("type", "STOP_MARKET".to_string());
        params.insert("quantity", format!("{:.8}", quantity));
        params.insert("stopPrice", format!("{:.8}", stop_price));
        params.insert("closePosition", "true".to_string());

        self.signed_request("POST", "/fapi/v1/order", params).await
    }

    /// Place a limit order (for TP, Maker only)
    pub async fn limit_order(
        &self,
        symbol: &str,
        side: Side,
        quantity: f64,
        price: f64,
        post_only: bool,
    ) -> Result<OrderResult, String> {
        if self.paper_mode {
            info!(
                "📄 PAPER LIMIT {} {} {:.6} @ {:.4}",
                side.as_str(),
                symbol,
                quantity,
                price
            );
            return Ok(OrderResult {
                order_id: Some(chrono::Utc::now().timestamp()),
                symbol: Some(symbol.to_string()),
                status: Some("NEW".to_string()),
                avg_price: None,
                executed_qty: Some("0".to_string()),
                side: Some(side.as_str().to_string()),
                code: None,
                msg: None,
            });
        }

        let clean_symbol = symbol.replace("/", "").replace("_", "");
        let mut params = HashMap::new();
        params.insert("symbol", clean_symbol.clone());
        params.insert("side", side.as_str().to_string());
        params.insert("type", "LIMIT".to_string());
        params.insert("quantity", format!("{:.8}", quantity));
        params.insert("price", format!("{:.8}", price));
        // Force maker order if post_only is true
        if post_only {
            params.insert("timeInForce", "GTX".to_string()); // Good Till Crossed (Post Only)
        } else {
            params.insert("timeInForce", "GTC".to_string()); // Good Till Canceled
        }

        self.signed_request("POST", "/fapi/v1/order", params).await
    }

    /// Cancel all open orders for a symbol
    pub async fn cancel_all_orders(&self, symbol: &str) -> Result<(), String> {
        if self.paper_mode {
            info!("📄 PAPER cancel all orders for {}", symbol);
            return Ok(());
        }

        let clean_symbol = symbol.replace("/", "").replace("_", "");
        let mut params = HashMap::new();
        params.insert("symbol", clean_symbol);

        let _result: serde_json::Value = self
            .signed_request_raw("DELETE", "/fapi/v1/allOpenOrders", params)
            .await?;
        Ok(())
    }

    /// Get account balance (equity)
    pub async fn get_equity(&self) -> Result<f64, String> {
        if self.paper_mode {
            return Ok(70.0); // Simulate $70 deposit
        }

        let params = HashMap::new();
        let result: Vec<serde_json::Value> = self
            .signed_request_raw("GET", "/fapi/v2/balance", params)
            .await?;

        for asset in &result {
            if asset["asset"].as_str() == Some("USDT") {
                if let Some(balance) = asset["balance"].as_str() {
                    return balance.parse::<f64>().map_err(|e| e.to_string());
                }
            }
        }
        Err("USDT balance not found".to_string())
    }

    // ── Signing ─────────────────────────────────────────────────────────

    /// Execute a signed REST request and parse as OrderResult
    async fn signed_request(
        &self,
        method: &str,
        path: &str,
        params: HashMap<&str, String>,
    ) -> Result<OrderResult, String> {
        let body = self.execute_signed(method, path, params).await?;
        serde_json::from_str::<OrderResult>(&body).map_err(|e| format!("Parse error: {}", e))
    }

    /// Execute a signed REST request and parse as generic JSON
    async fn signed_request_raw<T: serde::de::DeserializeOwned>(
        &self,
        method: &str,
        path: &str,
        params: HashMap<&str, String>,
    ) -> Result<T, String> {
        let body = self.execute_signed(method, path, params).await?;
        serde_json::from_str::<T>(&body).map_err(|e| format!("Parse error: {}", e))
    }

    /// Build signed query string and execute HTTP request
    async fn execute_signed(
        &self,
        method: &str,
        path: &str,
        params: HashMap<&str, String>,
    ) -> Result<String, String> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis();

        let mut query_parts: Vec<String> = params
            .iter()
            .map(|(k, v)| format!("{}={}", k, v))
            .collect();
        query_parts.push(format!("timestamp={}", timestamp));
        query_parts.push(format!("recvWindow={}", 5000));
        let query = query_parts.join("&");

        // HMAC-SHA256 signature
        let mut mac =
            HmacSha256::new_from_slice(self.api_secret.as_bytes()).map_err(|e| e.to_string())?;
        mac.update(query.as_bytes());
        let signature = hex::encode(mac.finalize().into_bytes());

        let url = format!("{}{}?{}&signature={}", BINANCE_FAPI_URL, path, query, signature);

        let request = match method {
            "POST" => self.client.post(&url),
            "DELETE" => self.client.delete(&url),
            _ => self.client.get(&url),
        };

        let response = request
            .header("X-MBX-APIKEY", &self.api_key)
            .send()
            .await
            .map_err(|e| format!("HTTP error: {}", e))?;

        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|e| format!("Read error: {}", e))?;

        if !status.is_success() {
            warn!("⚠️ API error {}: {}", status, body);
            // Parse error response
            if let Ok(err) = serde_json::from_str::<OrderResult>(&body) {
                if let Some(msg) = &err.msg {
                    return Err(format!("Binance error {}: {}", err.code.unwrap_or(-1), msg));
                }
            }
            return Err(format!("HTTP {}: {}", status, body));
        }

        Ok(body)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_paper_mode() {
        let router = OrderRouter::new("test".into(), "test".into(), true);
        assert!(router.paper_mode);
    }

    #[test]
    fn test_side_serialization() {
        assert_eq!(Side::Buy.as_str(), "BUY");
        assert_eq!(Side::Sell.as_str(), "SELL");
    }
}
