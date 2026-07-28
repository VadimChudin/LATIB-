use super::absorber::{calculate_confidence_score, TapeBaseline};
use super::position_manager::Direction;
use super::wall_tracker::{WallSnapshot, WallInfo, WallSide};
use std::time::Instant;

#[derive(Debug, Clone, Copy)]
pub enum Scenario {
    AggressiveAbsorption,
    WashTrading,
    FakeWall,
    KnifeExhaustion,
}

pub fn run_synthetic_benchmark() {
    println!("🧪 Starting HFT Synthetic Benchmark...");

    let scenarios = [
        Scenario::AggressiveAbsorption,
        Scenario::WashTrading,
        Scenario::FakeWall,
        Scenario::KnifeExhaustion,
    ];

    for &scenario in &scenarios {
        let (strategy, direction, baseline, speed, delta, whales, wall, peak_speed, expected_min_score) = 
            setup_scenario(scenario);

        let (score, _) = calculate_confidence_score(
            strategy,
            direction,
            &baseline,
            speed,
            delta,
            whales.0,
            whales.1,
            100.0, // last_price
            wall.as_ref(),
            peak_speed,
        );

        let status = if score >= expected_min_score { "✅ PASS" } else { "❌ FAIL" };
        println!("Benchmark [{:?}]: Score={} (Expected >= {}) | {}", 
            scenario, score, expected_min_score, status);
        
        if score < expected_min_score {
            println!("  Metrics: speed={:.1} delta={:.3} whales={:?}", speed, delta, whales);
        }
    }

    println!("🧪 Benchmark Finished.");
}

fn setup_scenario(scenario: Scenario) -> (
    &'static str, // strategy
    Direction,
    TapeBaseline,
    f64,          // current speed
    f64,          // current delta
    (usize, usize), // whales (buys, sells)
    Option<WallSnapshot>,
    f64,          // peak speed seen
    i32,          // expected min score
) {
    match scenario {
        Scenario::AggressiveAbsorption => (
            "breakout",
            Direction::Long,
            TapeBaseline { peak_speed: 10.0, initial_cvd: 0.0, initial_delta: 0.1 },
            20.0,  // speed acceleration (10 -> 20 = 2x)
            0.4,   // delta spike (0.1 -> 0.4 = +0.3)
            (2, 0), // whale buys
            Some(mock_wall_snapshot(WallSide::Ask, 100.1, 0.6)), // wall being eaten (60%)
            20.0,
            80, // High score expected
        ),
        Scenario::WashTrading => (
            "breakout",
            Direction::Long,
            TapeBaseline { peak_speed: 10.0, initial_cvd: 0.0, initial_delta: 0.1 },
            20.0,  // high speed
            0.1,   // NO delta spike (neutral)
            (0, 0),
            None,
            20.0,
            20, // Low score expected (only speed)
        ),
        Scenario::FakeWall => (
            "breakout",
            Direction::Long,
            TapeBaseline { peak_speed: 10.0, initial_cvd: 0.0, initial_delta: 0.1 },
            5.0,    // slow tape
            0.1,
            (0, 0),
            Some(mock_wall_snapshot(WallSide::Ask, 100.1, 0.0)), // untouched wall
            10.0,
            0,
        ),
        Scenario::KnifeExhaustion => (
            "knife",
            Direction::Long,
            TapeBaseline { peak_speed: 50.0, initial_cvd: -100.0, initial_delta: -0.2 },
            10.0,   // speed drop (50 -> 10)
            0.0,    // delta reversal (-0.2 -> 0.0)
            (2, 0), // whale buys
            Some(mock_wall_snapshot(WallSide::Bid, 99.9, 0.1)), // fresh support wall
            50.0,
            70,
        ),
    }
}

fn mock_wall_snapshot(side: WallSide, price: f64, eaten_pct: f64) -> WallSnapshot {
    let max_size = 100_000.0;
    let current_size = max_size * (1.0 - eaten_pct);
    
    let wall = WallInfo {
        price,
        side,
        first_seen: Instant::now() - std::time::Duration::from_secs(3600), // 1h old
        max_size_usd: max_size,
        current_size_usd: current_size,
        refresh_count: 0,
        touch_count: 1,
        last_touch: Some(Instant::now()),
        presence_history: std::collections::VecDeque::new(),
        stability: 1.0,
        prev_size_usd: 0.0,
        is_iceberg: false,
        is_spoof: false,
    };

    WallSnapshot {
        walls: vec![wall],
        cascades: vec![],
        is_warming_up: false,
        wall_threshold_usd: 50_000.0,
    }
}
