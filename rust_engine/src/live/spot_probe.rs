///! Spot Probe — Lazy On-Demand Spot Orderbook Scanner
///! ===================================================
///! Called ONLY when sniper_confirm() needs to verify a Futures wall
///! against the physical Spot market. Uses REST (not WebSocket) to
///! minimize connections.
///!
///! Flow:
///!   1. sniper_confirm() detects wall on Futures
///!   2. Calls probe_spot_depth(symbol, price_center, range_pct)
///!   3. REST GET https://api.binance.com/api/v3/depth?symbol=X&limit=20
///!   4. Returns SpotEnvironment with confirms/hidden_barrier flags

use serde::Deserialize;
use tracing::{info, warn};

use super::wall_tracker::WallSide;

// ── Configuration ───────────────────────────────────────────────────────────

/// How much bigger than average a level must be to qualify as a "wall" (3× = anomaly)
const SPOT_WALL_MULTIPLIER: f64 = 3.0;

/// Price range to search for spot walls (±0.5% from center price)
const SPOT_RANGE_PCT: f64 = 0.005;

/// Minimum wall size in USD to be considered significant
const SPOT_MIN_WALL_USD: f64 = 20_000.0;

/// Timeout for REST request (milliseconds)
const SPOT_TIMEOUT_MS: u64 = 2000;

// ── Data Types ──────────────────────────────────────────────────────────────

/// Result of a Spot orderbook probe — "Environment Report" for strategies
#[derive(Debug, Clone)]
pub struct SpotEnvironment {
    /// Biggest Bid (buy) wall found in the price range
    pub spot_bid_wall: Option<SpotWall>,
    /// Biggest Ask (sell) wall found in the price range
    pub spot_ask_wall: Option<SpotWall>,
    /// Spot Bid-wall supports a LONG (buy pressure in the zone)
    pub confirms_long: bool,
    /// Spot Ask-wall supports a SHORT (sell pressure in the zone)
    pub confirms_short: bool,
    /// Hidden barrier: wall on Spot that CONTRADICTS the intended direction
    pub hidden_barrier: bool,
    /// Which side the hidden barrier is on (if any)
    pub barrier_side: Option<WallSide>,
}

/// A single wall detected on Spot
#[derive(Debug, Clone)]
pub struct SpotWall {
    pub price: f64,
    pub size_usd: f64,
    pub side: WallSide,
}

/// Binance REST depth response
#[derive(Deserialize)]
struct DepthResponse {
    bids: Vec<[String; 2]>,  // [[price, qty], ...]
    asks: Vec<[String; 2]>,
}

// ── Public API ──────────────────────────────────────────────────────────────

/// Probe Binance Spot orderbook for a given symbol and price zone.
/// Returns SpotEnvironment describing walls found near the center price.
///
/// - `symbol`: e.g. "BTC_USDT" (converted to "BTCUSDT" internally)
/// - `price_center`: approximate entry price to search around
/// - `is_long`: intended trade direction (used for hidden barrier detection)
pub async fn probe_spot_depth(
    client: &reqwest::Client,
    symbol: &str,
    price_center: f64,
    is_long: bool,
) -> Option<SpotEnvironment> {
    // Convert symbol format: "BTC_USDT" → "BTCUSDT"
    let api_symbol = symbol
        .replace("_", "")
        .replace("/", "")
        .to_uppercase();

    let url = format!(
        "https://api.binance.com/api/v3/depth?symbol={}&limit=20",
        api_symbol
    );

    let resp = match tokio::time::timeout(
        std::time::Duration::from_millis(SPOT_TIMEOUT_MS),
        client.get(&url).send(),
    ).await {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => {
            warn!("🔍 SpotProbe [{}] HTTP error: {}", symbol, e);
            return None;
        }
        Err(_) => {
            warn!("🔍 SpotProbe [{}] timeout ({}ms)", symbol, SPOT_TIMEOUT_MS);
            return None;
        }
    };

    let depth: DepthResponse = match resp.json().await {
        Ok(d) => d,
        Err(e) => {
            warn!("🔍 SpotProbe [{}] parse error: {}", symbol, e);
            return None;
        }
    };

    // Parse levels and find anomalies in the price range
    let range_low = price_center * (1.0 - SPOT_RANGE_PCT);
    let range_high = price_center * (1.0 + SPOT_RANGE_PCT);

    let bid_levels = parse_levels(&depth.bids, range_low, range_high);
    let ask_levels = parse_levels(&depth.asks, range_low, range_high);

    // Calculate average level size for anomaly detection
    let all_levels: Vec<f64> = bid_levels.iter().chain(ask_levels.iter())
        .map(|(_, size_usd)| *size_usd)
        .collect();

    if all_levels.is_empty() {
        info!("🔍 SpotProbe [{}] no levels in range {:.2}-{:.2}", symbol, range_low, range_high);
        return Some(SpotEnvironment {
            spot_bid_wall: None,
            spot_ask_wall: None,
            confirms_long: false,
            confirms_short: false,
            hidden_barrier: false,
            barrier_side: None,
        });
    }

    let avg_size: f64 = all_levels.iter().sum::<f64>() / all_levels.len() as f64;
    let wall_threshold = (avg_size * SPOT_WALL_MULTIPLIER).max(SPOT_MIN_WALL_USD);

    // Find biggest bid wall (support)
    let spot_bid_wall = bid_levels.iter()
        .filter(|(_, size)| *size >= wall_threshold)
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .map(|(price, size_usd)| SpotWall {
            price: *price,
            size_usd: *size_usd,
            side: WallSide::Bid,
        });

    // Find biggest ask wall (resistance)
    let spot_ask_wall = ask_levels.iter()
        .filter(|(_, size)| *size >= wall_threshold)
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .map(|(price, size_usd)| SpotWall {
            price: *price,
            size_usd: *size_usd,
            side: WallSide::Ask,
        });

    // Determine confirmation and hidden barriers
    let confirms_long = spot_bid_wall.is_some();   // Bid wall = buy support = good for LONG
    let confirms_short = spot_ask_wall.is_some();   // Ask wall = sell resistance = good for SHORT

    // Hidden barrier: wall on the WRONG side for our intended direction
    let (hidden_barrier, barrier_side) = if is_long && spot_ask_wall.is_some() {
        // Going LONG but there's a sell wall on Spot → ceiling above us
        (true, Some(WallSide::Ask))
    } else if !is_long && spot_bid_wall.is_some() {
        // Going SHORT but there's a buy wall on Spot → floor below us
        (true, Some(WallSide::Bid))
    } else {
        (false, None)
    };

    // Log the result
    if let Some(ref w) = spot_bid_wall {
        info!("🔍 SpotProbe [{}] Bid wall: {:.4} ${:.0}k", symbol, w.price, w.size_usd / 1000.0);
    }
    if let Some(ref w) = spot_ask_wall {
        info!("🔍 SpotProbe [{}] Ask wall: {:.4} ${:.0}k", symbol, w.price, w.size_usd / 1000.0);
    }
    if hidden_barrier {
        info!("🔍 SpotProbe [{}] ⚠️ HIDDEN BARRIER detected ({:?})", symbol, barrier_side);
    }

    Some(SpotEnvironment {
        spot_bid_wall,
        spot_ask_wall,
        confirms_long,
        confirms_short,
        hidden_barrier,
        barrier_side,
    })
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Parse depth levels into (price, size_usd) tuples within a price range
fn parse_levels(levels: &[[String; 2]], range_low: f64, range_high: f64) -> Vec<(f64, f64)> {
    let mut result = Vec::new();
    for level in levels {
        let price: f64 = match level[0].parse() {
            Ok(p) => p,
            Err(_) => continue,
        };
        let qty: f64 = match level[1].parse() {
            Ok(q) => q,
            Err(_) => continue,
        };

        if price >= range_low && price <= range_high {
            result.push((price, price * qty));
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_levels() {
        let levels = vec![
            ["65000.0".to_string(), "10.0".to_string()],  // $650k
            ["65100.0".to_string(), "0.5".to_string()],    // $32.5k
            ["66000.0".to_string(), "100.0".to_string()],  // $6.6M — out of range
        ];

        let result = parse_levels(&levels, 64500.0, 65500.0);
        assert_eq!(result.len(), 2);
        assert!((result[0].1 - 650_000.0).abs() < 1.0);
    }

    #[test]
    fn test_empty_environment() {
        let env = SpotEnvironment {
            spot_bid_wall: None,
            spot_ask_wall: None,
            confirms_long: false,
            confirms_short: false,
            hidden_barrier: false,
            barrier_side: None,
        };
        assert!(!env.confirms_long);
        assert!(!env.hidden_barrier);
    }
}
