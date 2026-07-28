///! Decision Matrix — Tag-based Trade Management
///! ==============================================
///! Phase 29C+2: Probabilistic Exit Engine
///!
///! Instead of a single "Confidence Score 0-100%" which loses information
///! when conflicting signals cancel out, we use a TAG SYSTEM:
///!   - Each Pillar emits tags (STALE_LOSING, SWEEP_FOR_US, ICEBERG_AGAINST...)
///!   - Tags are combined via a priority-based Decision Matrix
///!   - Output: a concrete TradeAction (Hold, Tighten, Widen, Eject, Ride)
///!
///! This preserves signal granularity: BTC_WIND_FOR + ICEBERG_AGAINST = EJECT,
///! not "50% confidence = hold".

use std::fmt;

// ── Trade Tags ─────────────────────────────────────────────────────────────

/// Tags emitted by the various analysis pillars.
/// Each tag represents a discrete market observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TradeTag {
    // ── Time Decay Pillar ──
    /// Position open > 60s with pnl < +0.1R (stagnating)
    StaleTrade,
    /// Position open > 90s with pnl < 0 (stagnating AND losing)
    StaleLosing,

    // ── Order Book Pillar ──
    /// The Dynamic Target wall is a suspected spoofer (stability < 0.6)
    SpoofWallAhead,
    /// Iceberg detected BEHIND us (refill_count >= 3, protects our position)
    IcebergShield,
    /// Iceberg detected AHEAD of us (refill_count >= 3, absorbs our bounce)
    IcebergAgainst,
    /// A real, stable wall exists ahead as our Dynamic Target
    DynamicTargetSet,

    // ── Tape & HFT Pillar ──
    /// Sweep bot detected pushing price IN our favor (sweep_score > 15)
    SweepForUs,
    /// Sweep bot detected pushing price AGAINST us (sweep_score > 15)
    SweepAgainst,
    /// Whale print > $50k in our direction
    WhalePrintFor,
    /// Whale print > $50k against our direction
    WhalePrintAgainst,

    // ── BTC Macro Pillar ──
    /// BTC 5m momentum supports our trade direction
    BtcWindFor,
    /// BTC 5m momentum opposes our trade direction
    BtcWindAgainst,

    // ── Phase 30.5 Density Breakout Pillar ──
    /// The wall we broke out of was reclaimed by the maker (spoofing trap)
    WallReclaimed,
    /// The momentum died completely right after our breakout (no follow-through)
    MomentumDeath,
}

impl fmt::Display for TradeTag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TradeTag::StaleTrade => write!(f, "STALE_TRADE"),
            TradeTag::StaleLosing => write!(f, "STALE_LOSING"),
            TradeTag::SpoofWallAhead => write!(f, "SPOOF_WALL"),
            TradeTag::IcebergShield => write!(f, "ICE_SHIELD"),
            TradeTag::IcebergAgainst => write!(f, "ICE_AGAINST"),
            TradeTag::DynamicTargetSet => write!(f, "DYN_TARGET"),
            TradeTag::SweepForUs => write!(f, "SWEEP_FOR"),
            TradeTag::SweepAgainst => write!(f, "SWEEP_AGAINST"),
            TradeTag::WhalePrintFor => write!(f, "WHALE_FOR"),
            TradeTag::WhalePrintAgainst => write!(f, "WHALE_AGAINST"),
            TradeTag::BtcWindFor => write!(f, "BTC_WIND_FOR"),
            TradeTag::BtcWindAgainst => write!(f, "BTC_WIND_AGAINST"),
            TradeTag::WallReclaimed => write!(f, "WALL_RECLAIMED"),
            TradeTag::MomentumDeath => write!(f, "MOMENTUM_DEATH"),
        }
    }
}

// ── Trade Actions ──────────────────────────────────────────────────────────

/// The concrete action the position manager should take.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TradeAction {
    /// Continue with standard trailing / observation window logic
    Hold,
    /// Tighten trailing stop to 0.15% (exit on first micro-pullback)
    TightenTrail,
    /// Widen trailing stop to 0.8% and block early exits (ride the wave)
    WidenTrail,
    /// Immediate market exit — override everything
    PanicEject,
    /// God Mode: ride sweep to Dynamic Target, wide trail, block Profit Exhaustion
    RideSweep,
}

impl fmt::Display for TradeAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TradeAction::Hold => write!(f, "HOLD"),
            TradeAction::TightenTrail => write!(f, "TIGHTEN"),
            TradeAction::WidenTrail => write!(f, "WIDEN"),
            TradeAction::PanicEject => write!(f, "PANIC_EJECT"),
            TradeAction::RideSweep => write!(f, "RIDE_SWEEP"),
        }
    }
}

// ── Decision Logic ─────────────────────────────────────────────────────────

/// Priority-based decision matrix.
///
/// Rules are evaluated top-to-bottom. First match wins.
/// Priority ordering ensures safety: EJECT rules always win over RIDE rules.
pub fn decide(tags: &[TradeTag]) -> TradeAction {
    let has = |t: TradeTag| tags.contains(&t);

    // ═══════════════════════════════════════════════════════════════════════
    // PRIORITY 1: IMMEDIATE DANGER — always eject, no questions asked
    // ═══════════════════════════════════════════════════════════════════════
    if has(TradeTag::IcebergAgainst) {
        return TradeAction::PanicEject;
    }
    if has(TradeTag::SweepAgainst) {
        return TradeAction::PanicEject;
    }
    // Whale smash against us while we're already stale = death
    if has(TradeTag::WhalePrintAgainst) && has(TradeTag::StaleLosing) {
        return TradeAction::PanicEject;
    }
    // Phase 30.5: Breakout wall was a fake/trap and got rebuilt against us
    if has(TradeTag::WallReclaimed) {
        return TradeAction::PanicEject;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // PRIORITY 2: GOD MODE — ride the sweep wave
    // ═══════════════════════════════════════════════════════════════════════
    // Sweep in our favor + BTC supports us = maximum confidence
    if has(TradeTag::SweepForUs) && has(TradeTag::BtcWindFor) {
        return TradeAction::RideSweep;
    }
    // Sweep in our favor + Iceberg protecting us = very strong
    if has(TradeTag::SweepForUs) && has(TradeTag::IcebergShield) {
        return TradeAction::RideSweep;
    }
    // Sweep alone = still worth riding
    if has(TradeTag::SweepForUs) {
        return TradeAction::WidenTrail;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // PRIORITY 3: FAVORABLE CONDITIONS — widen trail, let profits run
    // ═══════════════════════════════════════════════════════════════════════
    // Iceberg shield + BTC wind = very safe, let it run
    if has(TradeTag::IcebergShield) && has(TradeTag::BtcWindFor) {
        return TradeAction::WidenTrail;
    }
    // Whale pushing for us = momentum, widen
    if has(TradeTag::WhalePrintFor) && !has(TradeTag::StaleLosing) {
        return TradeAction::WidenTrail;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // PRIORITY 4: DETERIORATING — tighten trail, prepare to exit
    // ═══════════════════════════════════════════════════════════════════════
    // Stale AND losing with no positive tags = squeeze trail
    if has(TradeTag::StaleLosing) && !has(TradeTag::BtcWindFor) && !has(TradeTag::IcebergShield) {
        return TradeAction::TightenTrail;
    }
    // BTC against us while stale = tighten
    if has(TradeTag::BtcWindAgainst) && has(TradeTag::StaleTrade) {
        return TradeAction::TightenTrail;
    }
    // Whale against us (but not combined with stale losing, which was eject above)
    if has(TradeTag::WhalePrintAgainst) {
        return TradeAction::TightenTrail;
    }
    // Phase 30.5: Breakout lost momentum but wall not reclaimed yet
    if has(TradeTag::MomentumDeath) {
        return TradeAction::TightenTrail;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // DEFAULT: business as usual
    // ═══════════════════════════════════════════════════════════════════════
    TradeAction::Hold
}

/// Format tags for logging (e.g. "[STALE_LOSING|SWEEP_FOR|BTC_WIND_FOR]")
pub fn format_tags(tags: &[TradeTag]) -> String {
    if tags.is_empty() {
        return "[]".to_string();
    }
    let parts: Vec<String> = tags.iter().map(|t| t.to_string()).collect();
    format!("[{}]", parts.join("|"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iceberg_against_always_ejects() {
        let tags = vec![TradeTag::IcebergAgainst, TradeTag::BtcWindFor, TradeTag::SweepForUs];
        assert_eq!(decide(&tags), TradeAction::PanicEject);
    }

    #[test]
    fn test_sweep_plus_btc_rides() {
        let tags = vec![TradeTag::SweepForUs, TradeTag::BtcWindFor];
        assert_eq!(decide(&tags), TradeAction::RideSweep);
    }

    #[test]
    fn test_stale_losing_tightens() {
        let tags = vec![TradeTag::StaleLosing];
        assert_eq!(decide(&tags), TradeAction::TightenTrail);
    }

    #[test]
    fn test_empty_holds() {
        let tags: Vec<TradeTag> = vec![];
        assert_eq!(decide(&tags), TradeAction::Hold);
    }

    #[test]
    fn test_sweep_against_ejects() {
        let tags = vec![TradeTag::SweepAgainst];
        assert_eq!(decide(&tags), TradeAction::PanicEject);
    }

    #[test]
    fn test_format_tags() {
        let tags = vec![TradeTag::StaleLosing, TradeTag::BtcWindFor];
        assert_eq!(format_tags(&tags), "[STALE_LOSING|BTC_WIND_FOR]");
    }

    #[test]
    fn test_whale_against_plus_stale_ejects() {
        let tags = vec![TradeTag::WhalePrintAgainst, TradeTag::StaleLosing];
        assert_eq!(decide(&tags), TradeAction::PanicEject);
    }

    #[test]
    fn test_whale_against_alone_tightens() {
        let tags = vec![TradeTag::WhalePrintAgainst];
        assert_eq!(decide(&tags), TradeAction::TightenTrail);
    }
}
