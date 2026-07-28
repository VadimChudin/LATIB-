use crate::tick_backtest::Epicenter;
use crate::strategies::knife_tick;
use rayon::prelude::*;
use rand::Rng;
use std::time::Instant;

const POPULATION_SIZE: usize = 100;
const F_SCALE: f64 = 0.8;
const CR: f64 = 0.9;

/// Phase 29C: v3 entry + v5 risk + adaptive baseline/duration/cooldown
fn get_bounds() -> Vec<(f64, f64)> {
    vec![
        (1000.0, 10000.0),  // 0: window_ms
        (1.5, 5.0),         // 1: min_zscore
        (1.0, 5.0),         // 2: min_vol_spike
        (0.3, 1.0),         // 3: (unused, tp_recovery=0.8 hardcoded)
        (0.001, 0.005),     // 4: sl_buffer_pct
        (0.001, 0.01),      // 5: be_trigger_pct
        (0.001, 0.008),     // 6: trail_pct
        (200.0, 3000.0),    // 7: micro_window_ms
        (2.0, 15.0),        // 8: min_absorption — floor at 2.0
        (0.0005, 0.005),    // 9: min_reclaim_pct
        (0.5, 5.0),         // 10: max_speed_mult
        (10.0, 300.0),      // 11: baseline_window_sec (Pre-Panic Baseline)
        (10.0, 120.0),      // 12: max_absorber_duration_sec
        (10.0, 120.0),      // 13: rewake_cooldown_sec
    ]
}

pub fn optimize_ticks(epicenters: &[Epicenter], _direction: &str, generations: usize) {
    if epicenters.is_empty() {
        println!("No epicenters provided for optimization.");
        return;
    }

    let bounds = get_bounds();
    let dim = bounds.len();

    // ─── TRAIN / TEST SPLIT (70 / 30) ───
    // Split only REAL epicenters chronologically, then mix in false ones
    let real_eps: Vec<&Epicenter> = epicenters.iter().filter(|e| e.has_bounce).collect();
    let false_eps: Vec<&Epicenter> = epicenters.iter().filter(|e| !e.has_bounce).collect();
    
    let real_split = (real_eps.len() as f64 * 0.70) as usize;
    let false_split = (false_eps.len() as f64 * 0.70) as usize;
    
    // Combine real + false for each set
    let mut train: Vec<&Epicenter> = Vec::new();
    train.extend_from_slice(&real_eps[..real_split]);
    if !false_eps.is_empty() {
        train.extend_from_slice(&false_eps[..false_split.min(false_eps.len())]);
    }
    
    let mut test: Vec<&Epicenter> = Vec::new();
    test.extend_from_slice(&real_eps[real_split..]);
    if false_eps.len() > false_split {
        test.extend_from_slice(&false_eps[false_split..]);
    }

    println!("📊 Train/Test Split: {} train / {} test (total {})", train.len(), test.len(), epicenters.len());
    println!("   Real: {} train + {} test | False: {} train + {} test", 
        real_split, real_eps.len() - real_split,
        false_split.min(false_eps.len()), false_eps.len().saturating_sub(false_split));
    println!("🧬 Phase 31 DE Optimizer started (Sharpe + Boundary + FalseEP + MFE)");
    let start_time = Instant::now();

    let mut population = vec![vec![0.0; dim]; POPULATION_SIZE];
    let mut fitnesses = vec![0.0; POPULATION_SIZE];

    let mut rng = rand::thread_rng();
    for i in 0..POPULATION_SIZE {
        for j in 0..dim {
            population[i][j] = rng.gen_range(bounds[j].0..=bounds[j].1);
        }
        fitnesses[i] = evaluate_genome_v2(&train, &population[i], &bounds);
    }

    let mut best_fitness = -99999.0;
    let mut best_genome = population[0].clone();

    for gen in 1..=generations {
        // DE/rand/1/bin
        let trial_fitnesses: Vec<(usize, Vec<f64>, f64)> = (0..POPULATION_SIZE).into_par_iter().map(|i| {
            let mut rng = rand::thread_rng();
            let r1 = rng.gen_range(0..POPULATION_SIZE);
            let mut r2 = rng.gen_range(0..POPULATION_SIZE);
            while r2 == r1 { r2 = rng.gen_range(0..POPULATION_SIZE); }
            let mut r3 = rng.gen_range(0..POPULATION_SIZE);
            while r3 == r1 || r3 == r2 { r3 = rng.gen_range(0..POPULATION_SIZE); }

            let mut trial = population[i].clone();
            let rand_j = rng.gen_range(0..dim);

            for j in 0..dim {
                if rng.gen::<f64>() < CR || j == rand_j {
                    trial[j] = population[r1][j] + F_SCALE * (population[r2][j] - population[r3][j]);
                    // Bounce back if out of bounds
                    if trial[j] < bounds[j].0 { trial[j] = bounds[j].0; }
                    if trial[j] > bounds[j].1 { trial[j] = bounds[j].1; }
                }
            }

            let fit = evaluate_genome_v2(&train, &trial, &bounds);
            (i, trial, fit)
        }).collect();

        for (i, trial, fit) in trial_fitnesses {
            if fit > fitnesses[i] {
                population[i] = trial;
                fitnesses[i] = fit;
            }
        }

        // Find best
        let mut gen_best_fit = -99999.0;
        let mut gen_best_idx = 0;
        for i in 0..POPULATION_SIZE {
            if fitnesses[i] > gen_best_fit {
                gen_best_fit = fitnesses[i];
                gen_best_idx = i;
            }
        }

        if gen_best_fit > best_fitness {
            best_fitness = gen_best_fit;
            best_genome = population[gen_best_idx].clone();
        }

        if gen % 10 == 0 || gen == 1 {
            println!("Gen {:3}/{}: Fit={:.2} | win={:.0}ms Z={:.2}σ vol={:.1}x sl={:.3}% be={:.3}% trail={:.3}% µwin={:.0}ms abs={:.1} recl={:.4}% spd={:.2}x base={:.0}s dur={:.0}s cool={:.0}s",
                gen, generations, best_fitness, best_genome[0], best_genome[1], best_genome[2], best_genome[4]*100.0, best_genome[5]*100.0, best_genome[6]*100.0, best_genome[7], best_genome[8], best_genome[9]*100.0, best_genome[10], best_genome[11], best_genome[12], best_genome[13]);
        }
    }

    let duration_sec = start_time.elapsed().as_secs_f64();
    println!("✅ Optimization finished in {:.2}s. Best Fitness: {:.2}", duration_sec, best_fitness);
    
    // ─── TRAIN RESULTS ───
    let train_stats = eval_set_detailed(&train, &best_genome);
    println!("\n📈 TRAIN SET: WR={:.1}% ({}/{}), Total R={:.2}, Sharpe={:.3}, MFE>0.1%={:.0}%, FP={}", 
        train_stats.wr * 100.0, train_stats.wins, train_stats.total, train_stats.total_pnl,
        train_stats.sharpe, train_stats.mfe_positive_ratio * 100.0, train_stats.false_positives);

    // ─── TEST RESULTS (OUT-OF-SAMPLE) ───
    let test_stats = eval_set_detailed(&test, &best_genome);
    println!("🧪 TEST  SET: WR={:.1}% ({}/{}), Total R={:.2}, Sharpe={:.3}, MFE>0.1%={:.0}%, FP={}", 
        test_stats.wr * 100.0, test_stats.wins, test_stats.total, test_stats.total_pnl,
        test_stats.sharpe, test_stats.mfe_positive_ratio * 100.0, test_stats.false_positives);
    
    // ─── BOUNDARY CHECK ───
    let param_names = [
        "window_ms", "min_zscore", "min_vol_spike", "(unused_tp)", "sl_buffer_pct",
        "be_trigger_pct", "trail_pct", "micro_window_ms",
        "min_absorption", "min_reclaim_pct", "max_speed_mult",
        "baseline_window_sec", "max_absorber_sec", "rewake_cooldown_sec",
    ];
    println!("\n🔍 PARAMETER BOUNDARY CHECK:");
    for idx in [0, 1, 2, 4, 7, 8, 9, 10, 11, 12, 13] {
        let (lo, hi) = bounds[idx];
        let val = best_genome[idx];
        let range = hi - lo;
        let dist_lo = (val - lo) / range * 100.0;
        let dist_hi = (hi - val) / range * 100.0;
        let status = if dist_lo < 10.0 || dist_hi < 10.0 { "⚠️ BOUNDARY" } else { "✅ OK" };
        println!("  {} {}: {:.4} (lo={:.1}%, hi={:.1}%)", status, param_names[idx], val, dist_lo, dist_hi);
    }

    // ─── VERDICT ───
    let wr_gap = (train_stats.wr - test_stats.wr).abs();
    if test_stats.total >= 5 && test_stats.wr >= 0.48 && test_stats.total_pnl > 0.0 && wr_gap < 0.15 {
        println!("\n✅ VERDICT: STRATEGY VALIDATED! OOS passed. WR gap={:.1}%", wr_gap * 100.0);
    } else if test_stats.total < 5 {
        println!("\n⚠️ VERDICT: NOT ENOUGH OOS TRADES ({}). Need more data.", test_stats.total);
    } else if wr_gap > 0.15 {
        println!("\n❌ VERDICT: OVERFIT DETECTED. Train WR={:.1}% but Test WR={:.1}% (gap={:.1}%)", 
            train_stats.wr * 100.0, test_stats.wr * 100.0, wr_gap * 100.0);
    } else {
        println!("\n❌ VERDICT: UNPROFITABLE. Test PnL={:.2}R, WR={:.1}%", test_stats.total_pnl, test_stats.wr * 100.0);
    }

    // ─── DUMP TRADES + PARAMS TO JSON ───
    dump_results(epicenters, &best_genome, &param_names, &train_stats, &test_stats, best_fitness);
}

pub struct BatchResult {
    pub total_trades: usize,
    pub train_wr: f64,
    pub train_pnl: f64,
    pub test_wr: f64,
    pub test_pnl: f64,
    pub test_mfe_ratio: f64,
    pub best_genome: Vec<f64>,
}

pub fn optimize_ticks_batch(epicenters: &[Epicenter], _direction: &str, generations: usize) -> BatchResult {
    if epicenters.len() < 5 {
        return BatchResult { total_trades: 0, train_wr: 0.0, train_pnl: 0.0, test_wr: 0.0, test_pnl: 0.0, test_mfe_ratio: 0.0, best_genome: vec![] };
    }

    let bounds = get_bounds();
    let dim = bounds.len();

    // Train/test split (70/30) — same as optimize_ticks
    let real_eps: Vec<&Epicenter> = epicenters.iter().filter(|e| e.has_bounce).collect();
    let false_eps: Vec<&Epicenter> = epicenters.iter().filter(|e| !e.has_bounce).collect();
    
    let real_split = (real_eps.len() as f64 * 0.70) as usize;
    let false_split = (false_eps.len() as f64 * 0.70) as usize;
    
    let mut train: Vec<&Epicenter> = Vec::new();
    train.extend_from_slice(&real_eps[..real_split]);
    if !false_eps.is_empty() {
        train.extend_from_slice(&false_eps[..false_split.min(false_eps.len())]);
    }
    
    let mut test: Vec<&Epicenter> = Vec::new();
    test.extend_from_slice(&real_eps[real_split..]);
    if false_eps.len() > false_split {
        test.extend_from_slice(&false_eps[false_split..]);
    }

    // Initialize population
    let mut population = vec![vec![0.0; dim]; POPULATION_SIZE];
    let mut rng = rand::thread_rng();
    let mut fitnesses = vec![0.0f64; POPULATION_SIZE];
    for i in 0..POPULATION_SIZE {
        for j in 0..dim {
            population[i][j] = rng.gen_range(bounds[j].0..=bounds[j].1);
        }
        fitnesses[i] = evaluate_genome_v2(&train, &population[i], &bounds);
    }

    let mut best_fitness = -99999.0;
    let mut best_genome = population[0].clone();
    let timeout = Instant::now();

    for _gen in 1..=generations {
        // Timeout: max 10 min per symbol
        if timeout.elapsed().as_secs() > 600 {
            break;
        }

        let trial_fitnesses: Vec<(usize, Vec<f64>, f64)> = (0..POPULATION_SIZE).into_par_iter().map(|i| {
            let mut rng = rand::thread_rng();
            let r1 = rng.gen_range(0..POPULATION_SIZE);
            let mut r2 = rng.gen_range(0..POPULATION_SIZE);
            while r2 == r1 { r2 = rng.gen_range(0..POPULATION_SIZE); }
            let mut r3 = rng.gen_range(0..POPULATION_SIZE);
            while r3 == r1 || r3 == r2 { r3 = rng.gen_range(0..POPULATION_SIZE); }

            let mut trial = population[i].clone();
            let rand_j = rng.gen_range(0..dim);
            for j in 0..dim {
                if rng.gen::<f64>() < CR || j == rand_j {
                    trial[j] = population[r1][j] + F_SCALE * (population[r2][j] - population[r3][j]);
                    if trial[j] < bounds[j].0 { trial[j] = bounds[j].0; }
                    if trial[j] > bounds[j].1 { trial[j] = bounds[j].1; }
                }
            }

            let fit = evaluate_genome_v2(&train, &trial, &bounds);
            (i, trial, fit)
        }).collect();

        for (i, trial, fit) in trial_fitnesses {
            if fit >= fitnesses[i] {
                population[i] = trial.clone();
                fitnesses[i] = fit;
                if fit > best_fitness {
                    best_fitness = fit;
                    best_genome = trial;
                }
            }
        }
    }

    let train_stats = eval_set_detailed(&train, &best_genome);
    let test_stats = eval_set_detailed(&test, &best_genome);

    BatchResult {
        total_trades: train_stats.total + test_stats.total,
        train_wr: train_stats.wr,
        train_pnl: train_stats.total_pnl,
        test_wr: test_stats.wr,
        test_pnl: test_stats.total_pnl,
        test_mfe_ratio: test_stats.mfe_positive_ratio,
        best_genome: best_genome,
    }
}

// ═══ Phase 31: New Fitness Function ═══

struct EvalStats {
    wins: usize,
    total: usize,
    wr: f64,
    total_pnl: f64,
    sharpe: f64,
    mfe_positive_ratio: f64,
    false_positives: usize,
    correct_rejects: usize,
}

fn eval_set_detailed(epicenters: &[&Epicenter], params: &[f64]) -> EvalStats {
    let mut wins = 0usize;
    let mut total = 0usize;
    let mut pnl_list: Vec<f64> = Vec::new();
    let mut mfe_positive = 0usize;
    let mut false_positives = 0usize;
    let mut correct_rejects = 0usize;

    for ev in epicenters {
        let trade = knife_tick::evaluate_epicenter(ev, params);
        
        if ev.has_bounce {
            // Real epicenter — normal evaluation
            if let Some(t) = trade {
                pnl_list.push(t.pnl_r);
                total += 1;
                if t.pnl_r > 0.0 { wins += 1; }
                if t.mfe_pct > 0.1 { mfe_positive += 1; } // price moved at least 0.1%
            }
        } else {
            // FALSE epicenter — entering here is BAD
            if trade.is_some() {
                false_positives += 1;
            } else {
                correct_rejects += 1;
            }
        }
    }

    let wr = if total > 0 { wins as f64 / total as f64 } else { 0.0 };
    let total_pnl: f64 = pnl_list.iter().sum();
    let mfe_positive_ratio = if total > 0 { mfe_positive as f64 / total as f64 } else { 0.0 };
    
    // Sharpe Ratio
    let sharpe = if total > 1 {
        let mean = total_pnl / total as f64;
        let variance = pnl_list.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / total as f64;
        let std_dev = variance.sqrt().max(0.01);
        mean / std_dev
    } else {
        0.0
    };

    EvalStats { wins, total, wr, total_pnl, sharpe, mfe_positive_ratio, false_positives, correct_rejects }
}

/// Phase 31: Boundary Penalty — penalize parameters sitting at the edge of their range
fn boundary_penalty(params: &[f64], bounds: &[(f64, f64)]) -> f64 {
    // Entry-relevant parameters that MUST be used meaningfully
    let entry_indices = [0, 7, 8, 9, 10]; // window_ms, micro_win, delta, size, speed
    let mut penalty = 0.0;
    
    for &idx in &entry_indices {
        let (lo, hi) = bounds[idx];
        let range = hi - lo;
        let val = params[idx];
        
        // Distance to nearest boundary as fraction of range
        let dist_to_edge = ((val - lo).min(hi - val)) / range;
        
        // Soft penalty: increases smoothly as parameter approaches boundary
        if dist_to_edge < 0.15 {
            penalty += (0.15 - dist_to_edge) * 80.0; // max ~12 per param
        }
    }
    penalty
}

/// Phase 31: Price Jitter — average tick-to-tick price change (noise floor)
fn calc_jitter(epicenter: &Epicenter, window_ms: u64) -> f64 {
    let ticks = &epicenter.ticks;
    if ticks.len() < 10 { return 0.0; }
    
    let mut changes = Vec::new();
    for pair in ticks.windows(2) {
        if pair[1].ts_ms - pair[0].ts_ms < window_ms {
            let pct = ((pair[1].price - pair[0].price) / pair[0].price).abs() as f64;
            changes.push(pct);
        }
    }
    if changes.is_empty() { return 0.0; }
    changes.iter().sum::<f64>() / changes.len() as f64
}

/// Phase 31: Main fitness function with all anti-overfit mechanisms
fn evaluate_genome_v2(epicenters: &[&Epicenter], params: &[f64], bounds: &[(f64, f64)]) -> f64 {
    let mut pnl_list: Vec<f64> = Vec::new();
    let mut false_positive_count = 0usize;
    let mut correct_reject_count = 0usize;
    let mut mfe_positive_count = 0usize;
    let mut jitter_reject_count = 0usize; // entries where SL < 3×jitter (noisy)
    
    let sl_pct = params[4];
    let window_ms = params[0] as u64;
    
    for ev in epicenters.iter() {
        let trade = knife_tick::evaluate_epicenter(ev, params);
        
        // Phase 31: Jitter check — if SL < 3× noise floor, mark as bad entry
        let jitter = calc_jitter(ev, window_ms);
        let jitter_bad = jitter > 0.0 && sl_pct < jitter * 3.0;
        
        if ev.has_bounce {
            // Real epicenter
            if let Some(t) = trade {
                if jitter_bad {
                    jitter_reject_count += 1;
                    continue; // Don't count this trade — SL is in noise zone
                }
                pnl_list.push(t.pnl_r);
                if t.mfe_pct > 0.1 { mfe_positive_count += 1; }
            }
        } else {
            // FALSE epicenter
            if trade.is_some() {
                false_positive_count += 1;
            } else {
                correct_reject_count += 1;
            }
        }
    }
    
    let total = pnl_list.len();
    
    // Minimum trades: 10 (need enough for statistics but don't force weak entries)
    if total < 10 {
        return -1000.0 + (total as f64 * 40.0); // gradient toward more trades
    }
    
    // ── Sharpe Ratio ──
    let mean = pnl_list.iter().sum::<f64>() / total as f64;
    let variance = pnl_list.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / total as f64;
    let std_dev = variance.sqrt().max(0.01);
    let sharpe = mean / std_dev;
    
    // ── MFE ratio (% of trades where price went our way at least 0.1%) ──
    let mfe_ratio = mfe_positive_count as f64 / total as f64;
    
    // ── Boundary penalty ──
    let bp = boundary_penalty(params, bounds);
    
    // ── False positive penalty ── 
    let fp_penalty = false_positive_count as f64 * 3.0;
    
    // ── Entry rate (% of real epicenters where we entered) ──
    let real_count = epicenters.iter().filter(|e| e.has_bounce).count();
    let entry_rate = if real_count > 0 { total as f64 / real_count as f64 } else { 0.0 };
    
    // ══ FINAL FITNESS ══
    // Core: Sharpe × sqrt(trades) — consistent profit at scale (dominates)
    // Entry rate bonus: reward strategies that find MORE entries (not just avoid everything)
    // MFE bonus: trades where price actually bounced in our favor
    // FP penalty: don't enter false epicenters
    // Boundary penalty: use all parameters meaningfully
    // NO correct_reject bonus — rejecting is free, shouldn't be rewarded
    
    let core = sharpe * (total as f64).sqrt() * 5.0;  // ×5 to make profit dominant
    let entry_bonus = entry_rate * 30.0;               // reward higher entry rate (up to ~30 pts)
    let mfe_bonus = mfe_ratio * 15.0;                  // reward actual bounces
    
    core + entry_bonus + mfe_bonus - bp - fp_penalty - jitter_reject_count as f64 * 1.0
}

fn dump_results(
    epicenters: &[Epicenter], 
    best_genome: &[f64], 
    param_names: &[&str],
    train_stats: &EvalStats,
    test_stats: &EvalStats,
    best_fitness: f64,
) {
    let mut all_trades_json: Vec<serde_json::Value> = Vec::new();
    for ev in epicenters {
        if !ev.has_bounce { continue; } // Only dump real trades
        if let Some(trade) = knife_tick::evaluate_epicenter(ev, best_genome) {
            all_trades_json.push(serde_json::json!({
                "epicenter_ts_ms": ev.ts_ms,
                "direction": trade.direction,
                "entry_idx": trade.entry_idx,
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "sl_price": trade.sl_price,
                "tp_price": trade.tp_price,
                "pnl_r": trade.pnl_r,
                "risk_dist": trade.risk_dist,
                "mfe_pct": trade.mfe_pct,
            }));
        }
    }

    let trades_path = "../data/tick_trades_knifetick.json";
    match serde_json::to_string_pretty(&all_trades_json) {
        Ok(json_str) => {
            std::fs::write(trades_path, &json_str).ok();
            println!("\n📁 Saved {} trades to {}", all_trades_json.len(), trades_path);
        }
        Err(e) => eprintln!("Failed to serialize trades: {}", e),
    }

    let mut params_map = serde_json::Map::new();
    for (i, val) in best_genome.iter().enumerate() {
        let name = param_names.get(i).unwrap_or(&"unknown");
        params_map.insert(name.to_string(), serde_json::json!(val));
    }
    let params_output = serde_json::json!({
        "strategy": "knife_tick",
        "version": "phase31",
        "params": params_map,
        "train_wr": train_stats.wr * 100.0,
        "test_wr": test_stats.wr * 100.0,
        "train_trades": train_stats.total,
        "test_trades": test_stats.total,
        "train_pnl_r": train_stats.total_pnl,
        "test_pnl_r": test_stats.total_pnl,
        "train_sharpe": train_stats.sharpe,
        "test_sharpe": test_stats.sharpe,
        "train_mfe_ratio": train_stats.mfe_positive_ratio,
        "test_mfe_ratio": test_stats.mfe_positive_ratio,
        "false_positives_train": train_stats.false_positives,
        "false_positives_test": test_stats.false_positives,
        "fitness": best_fitness,
    });
    let params_path = "../data/ga_best_tick_params.json";
    match serde_json::to_string_pretty(&params_output) {
        Ok(json_str) => {
            std::fs::write(params_path, &json_str).ok();
            println!("📁 Saved best params to {}", params_path);
        }
        Err(e) => eprintln!("Failed to serialize params: {}", e),
    }
}
