use crate::live::ws_feed::Candle;

/// Dynamically aggregates 1-minute WebSocket candles into higher timeframes (5m, 15m)
/// This saves connection limits with Binance by reusing the 1m stream for all MTF needs.
#[derive(Debug, Default)]
pub struct CandleAggregator {
    pending_5m: Option<Candle>,
    pending_15m: Option<Candle>,
}

impl CandleAggregator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Process a closed 1-minute candle.
    /// Returns a tuple of `(Option<closed_5m_candle>, Option<closed_15m_candle>)`.
    /// The returned candles represent fully formed higher timeframe bars.
    pub fn process_1m_close(&mut self, candle_1m: &Candle) -> (Option<Candle>, Option<Candle>) {
        let mut completed_5m = None;
        let mut completed_15m = None;

        // The timestamp from Binance is the open time in milliseconds
        // Typical Minute from timestamp:
        let ts = candle_1m.timestamp;
        let minute = (ts / 60_000) % 60;

        // --- 5m Aggregation ---
        if let Some(mut p5) = self.pending_5m.take() {
            p5.high = p5.high.max(candle_1m.high);
            p5.low = p5.low.min(candle_1m.low);
            p5.close = candle_1m.close;
            p5.volume += candle_1m.volume;

            // 5m candle closes at minute 4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59
            if (minute + 1) % 5 == 0 {
                p5.is_closed = true;
                completed_5m = Some(p5);
            } else {
                self.pending_5m = Some(p5);
            }
        } else {
            // Start a new 5m candle
            let mut new5 = candle_1m.clone();
            new5.is_closed = false;
            // Align the timestamp backwards to the nearest 5m start
            new5.timestamp = ts - ((minute % 5) * 60_000); 

            if (minute + 1) % 5 == 0 {
                new5.is_closed = true;
                completed_5m = Some(new5);
            } else {
                self.pending_5m = Some(new5);
            }
        }

        // --- 15m Aggregation ---
        if let Some(mut p15) = self.pending_15m.take() {
            p15.high = p15.high.max(candle_1m.high);
            p15.low = p15.low.min(candle_1m.low);
            p15.close = candle_1m.close;
            p15.volume += candle_1m.volume;

            // 15m candle closes at minute 14, 29, 44, 59
            if (minute + 1) % 15 == 0 {
                p15.is_closed = true;
                completed_15m = Some(p15);
            } else {
                self.pending_15m = Some(p15);
            }
        } else {
            // Start a new 15m candle
            let mut new15 = candle_1m.clone();
            new15.is_closed = false;
            // Align timestamp backwards to nearest 15m start
            new15.timestamp = ts - ((minute % 15) * 60_000);

            if (minute + 1) % 15 == 0 {
                new15.is_closed = true;
                completed_15m = Some(new15);
            } else {
                self.pending_15m = Some(new15);
            }
        }

        (completed_5m, completed_15m)
    }

    /// Update current pending candles with a tick (unfinished 1m candle).
    /// Returns the current state of (5m_tick, 15m_tick)
    pub fn process_1m_tick(&mut self, tick_1m: &Candle) -> (Candle, Candle) {
        let ts = tick_1m.timestamp;
        let minute = (ts / 60_000) % 60;

        let mut t5 = match &self.pending_5m {
            Some(p) => {
                let mut c = p.clone();
                c.high = c.high.max(tick_1m.high);
                c.low = c.low.min(tick_1m.low);
                c.close = tick_1m.close;
                c.volume += tick_1m.volume;
                c
            },
            None => {
                let mut c = tick_1m.clone();
                c.timestamp = ts - ((minute % 5) * 60_000);
                c
            }
        };

        let mut t15 = match &self.pending_15m {
            Some(p) => {
                let mut c = p.clone();
                c.high = c.high.max(tick_1m.high);
                c.low = c.low.min(tick_1m.low);
                c.close = tick_1m.close;
                c.volume += tick_1m.volume;
                c
            },
            None => {
                let mut c = tick_1m.clone();
                c.timestamp = ts - ((minute % 15) * 60_000);
                c
            }
        };

        t5.is_closed = false;
        t15.is_closed = false;

        (t5, t15)
    }
}
