///! Full Order Book Manager
///! ========================
///! Maintains a lock-free full-depth order book per symbol.
///! Updated from `@depth@100ms` WebSocket stream.

use dashmap::DashMap;
use serde::Deserialize;
use std::sync::Arc;
use tracing::warn;

/// A single price level in the book
#[derive(Debug, Clone)]
pub struct Level {
    pub price: f64,
    pub quantity: f64,
}

/// Full order book for one symbol
#[derive(Debug, Clone, Default)]
pub struct OrderBook {
    pub bids: Vec<Level>, // Sorted descending by price (best bid first)
    pub asks: Vec<Level>, // Sorted ascending by price (best ask first)
    pub last_update_id: u64,
}

impl OrderBook {
    /// Calculate Order Book Imbalance: (ΣBid - ΣAsk) / (ΣBid + ΣAsk)
    /// Range: -1.0 (all sellers) to +1.0 (all buyers)
    pub fn imbalance(&self) -> f64 {
        let bid_vol: f64 = self.bids.iter().map(|l| l.quantity * l.price).sum();
        let ask_vol: f64 = self.asks.iter().map(|l| l.quantity * l.price).sum();
        let total = bid_vol + ask_vol;
        if total <= 0.0 {
            return 0.0;
        }
        (bid_vol - ask_vol) / total
    }

    /// Best bid price
    pub fn best_bid(&self) -> Option<f64> {
        self.bids.first().map(|l| l.price)
    }

    /// Best ask price
    pub fn best_ask(&self) -> Option<f64> {
        self.asks.first().map(|l| l.price)
    }

    /// Mid price
    pub fn mid_price(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some((bid + ask) / 2.0),
            _ => None,
        }
    }

    /// Spread in basis points
    pub fn spread_bps(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) if bid > 0.0 => {
                Some((ask - bid) / bid * 10_000.0)
            }
            _ => None,
        }
    }
}

/// Shared order book store (lock-free)
pub type OrderBookStore = Arc<DashMap<String, OrderBook>>;

/// Create a new shared order book store
pub fn new_store() -> OrderBookStore {
    Arc::new(DashMap::new())
}

/// Parse a Binance depth update and apply it to the store
pub fn apply_depth_update(store: &OrderBookStore, raw: &serde_json::Value) {
    #[derive(Deserialize)]
    #[allow(dead_code)]
    struct DepthUpdate {
        s: String,                    // Symbol
        #[serde(rename = "U")]
        first_update_id: u64,
        u: u64,                       // Last update ID
        b: Vec<[String; 2]>,         // Bids [[price, qty], ...]
        a: Vec<[String; 2]>,         // Asks [[price, qty], ...]
    }

    let update: DepthUpdate = match serde_json::from_value(raw.clone()) {
        Ok(u) => u,
        Err(e) => {
            warn!("Failed to parse depth update: {}", e);
            return;
        }
    };

    let symbol = super::ws_feed::format_symbol(&update.s);

    let mut book = store.entry(symbol).or_insert_with(OrderBook::default);

    // Helper: apply level updates to a price vector
    fn apply_levels(levels: &mut Vec<Level>, updates: &[[String; 2]]) {
        for level in updates {
            let price: f64 = level[0].parse().unwrap_or(0.0);
            let qty: f64 = level[1].parse().unwrap_or(0.0);

            if qty == 0.0 {
                levels.retain(|l| (l.price - price).abs() > f64::EPSILON);
            } else {
                let mut found = false;
                for l in levels.iter_mut() {
                    if (l.price - price).abs() < f64::EPSILON {
                        l.quantity = qty;
                        found = true;
                        break;
                    }
                }
                if !found {
                    levels.push(Level { price, quantity: qty });
                }
            }
        }
    }

    apply_levels(&mut book.bids, &update.b);
    apply_levels(&mut book.asks, &update.a);

    // Keep sorted
    book.bids.sort_by(|a, b| b.price.partial_cmp(&a.price).unwrap());
    book.asks.sort_by(|a, b| a.price.partial_cmp(&b.price).unwrap());
    book.last_update_id = update.u;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_imbalance() {
        let book = OrderBook {
            bids: vec![
                Level { price: 100.0, quantity: 10.0 },
                Level { price: 99.0, quantity: 5.0 },
            ],
            asks: vec![
                Level { price: 101.0, quantity: 3.0 },
                Level { price: 102.0, quantity: 2.0 },
            ],
            last_update_id: 0,
        };
        let imb = book.imbalance();
        assert!(imb > 0.0, "More bid volume should give positive imbalance");
    }
}
