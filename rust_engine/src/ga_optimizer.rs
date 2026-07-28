/// Genetic Algorithm Optimizer for Strategy Parameters
/// ====================================================
///
/// Evolves optimal strategy parameters using:
/// - Tournament selection
/// - Uniform crossover
/// - Gaussian mutation
/// - Elitism (top 10% survive)
///
/// Fitness = Sharpe-like ratio: (WR - 50%) * sqrt(num_trades)
/// This rewards high WR AND sufficient trade count.

// use rand::prelude::*;
use rayon::prelude::*;
use crate::backtest::{Candle, Trade, PrecomputedData};
use crate::strategies;
use rand::Rng;
use std::fmt;

/// A genome = set of strategy parameters
#[derive(Clone, Debug)]
pub struct Genome {
    pub params: Vec<f64>,
    pub fitness: f64,
    pub win_rate: f64,
    pub num_trades: usize,
}

/// Parameter definition with min/max/step
#[derive(Clone)]
pub struct ParamDef {
    pub name: String,
    pub min: f64,
    pub max: f64,
    pub step: f64,
}

/// GA configuration
pub struct GaConfig {
    pub population_size: usize,
    pub generations: usize,
    pub mutation_rate: f64,
    pub elitism_pct: f64,
    pub tournament_size: usize,
    // Phase 21: Hot-Swap Tuning
    pub recency_base: f64,       // e.g. 0.3 (lower = more aggressive recency)
    pub stress_penalty_mult: f64, // e.g. 1.3 (higher = more conservative in stress)
}

impl Default for GaConfig {
    fn default() -> Self {
        GaConfig {
            population_size: 100,
            generations: 50,
            mutation_rate: 0.15,
            elitism_pct: 0.10,
            tournament_size: 5,
            recency_base: 0.3,
            stress_penalty_mult: 1.3,
        }
    }
}

impl fmt::Display for Genome {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "WR={:.1}% trades={} fitness={:.3}", self.win_rate, self.num_trades, self.fitness)
    }
}

/// Get parameter definitions for a strategy (~8-9 params each = ~35 total)
pub fn get_param_defs(strategy: &str) -> Vec<ParamDef> {
    let mut defs = match strategy {
        "smc" => vec![
            // Core strategy (idx 0-5)
            ParamDef { name: "swing_length".into(),    min: 3.0,  max: 10.0, step: 1.0 },
            ParamDef { name: "fvg_min_atr".into(),     min: 0.1,  max: 0.8,  step: 0.1 },
            ParamDef { name: "ob_min_score".into(),     min: 2.0,  max: 5.0,  step: 1.0 },
            ParamDef { name: "sl_atr_mult".into(),      min: 0.5,  max: 2.5,  step: 0.25 },
            ParamDef { name: "trail_activate_r".into(), min: 0.5,  max: 2.5,  step: 0.25 },
            ParamDef { name: "trail_atr_mult".into(),   min: 0.2,  max: 1.2,  step: 0.1 },
        ],
        "knifetick" => vec![
            // V3 HFT Tick strategy params
            ParamDef { name: "window_ms".into(),         min: 1000.0, max: 10000.0, step: 500.0 },
            ParamDef { name: "min_drop_pct".into(),      min: 0.1,  max: 1.0,   step: 0.05 },
            ParamDef { name: "tp_pct".into(),            min: 0.5,  max: 5.0,   step: 0.1 },
            ParamDef { name: "sl_pct".into(),            min: 0.1,  max: 2.0,   step: 0.05 },
            ParamDef { name: "be_trigger_pct".into(),    min: 0.1,  max: 1.0,   step: 0.05 },
            ParamDef { name: "trail_pct".into(),         min: 0.1,  max: 1.0,   step: 0.05 },
            ParamDef { name: "micro_window_ms".into(),   min: 20.0, max: 500.0, step: 10.0 },
            ParamDef { name: "max_speed_mult".into(),    min: 0.1,  max: 5.0,   step: 0.1 },
            ParamDef { name: "min_size_mult".into(),     min: 0.1,  max: 5.0,   step: 0.1 },
            ParamDef { name: "min_delta_mult".into(),    min: -2.0, max: 2.0,   step: 0.1 },
            ParamDef { name: "reserved".into(),          min: 0.0,  max: 1.0,   step: 1.0 },
        ],
        "scalpmtf" => vec![
            ParamDef { name: "fast_ema".into(),         min: 5.0,  max: 21.0, step: 1.0 },
            ParamDef { name: "slow_ema".into(),         min: 30.0, max: 100.0, step: 5.0 },
            ParamDef { name: "rsi_thresh".into(),       min: 15.0, max: 45.0, step: 5.0 },
            ParamDef { name: "tp_rr".into(),            min: 0.8,  max: 3.0,  step: 0.2 },
        ],
        "fundingrate" => vec![
            ParamDef { name: "fr_long_thresh".into(),    min: 0.01, max: 0.08, step: 0.01 },
            ParamDef { name: "fr_short_thresh".into(),   min: 0.02, max: 0.10, step: 0.01 },
            ParamDef { name: "sl_atr_mult".into(),       min: 1.0,  max: 2.5,  step: 0.25 },
            ParamDef { name: "trail_activate_r".into(),  min: 0.5,  max: 2.0,  step: 0.25 },
            ParamDef { name: "trail_atr_mult".into(),    min: 0.3,  max: 1.0,  step: 0.1 },
            ParamDef { name: "cooldown_bars".into(),     min: 3.0,  max: 12.0, step: 1.0 },
        ],
        "density" => vec![
            ParamDef { name: "vol_spike_mult".into(),   min: 1.5,  max: 5.0,  step: 0.5 },
            ParamDef { name: "min_touches".into(),      min: 2.0,  max: 5.0,  step: 1.0 },
            ParamDef { name: "shakeout_pct".into(),     min: 0.003, max: 0.015, step: 0.001 },
            ParamDef { name: "tp_rr".into(),            min: 1.0,  max: 4.0,  step: 0.5 },
            ParamDef { name: "sl_atr_mult".into(),      min: 0.5,  max: 2.0,  step: 0.25 },
        ],
        _ => vec![],
    };

    // Common params appended to ALL strategies (after strategy-specific ones)
    defs.push(ParamDef { name: "cooldown_bars".into(),     min: 0.0,  max: 30.0,  step: 5.0 });
    defs.push(ParamDef { name: "max_trades_day".into(),    min: 3.0,  max: 20.0,  step: 1.0 });
    defs.push(ParamDef { name: "max_drawdown_r".into(),    min: 3.0,  max: 8.0,   step: 1.0 });

    defs
}

/// Calculate fitness — Quant-Grade V6
/// ===================================
/// - WR computed from RAW pnl (no recency/stress distortion)
/// - Sortino computed from weighted equity curve (recency applies to risk metrics only)
/// - No stress penalty on wins (removes distortion)
/// - Expected R bonus rewards actual profitability
fn calc_fitness(trades: &[Trade], _data: &PrecomputedData, n_total: usize, _config: &GaConfig) -> (f64, f64, usize) {
    let total = trades.len();
    if total < 100 { return (-100.0, 0.0, total); }

    // Pass 1: Raw statistics (undistorted)
    let mut raw_wins = 0usize;
    let mut raw_losses = 0usize;
    let mut raw_profit = 0.0f64;
    let mut raw_loss = 0.0f64;

    // Pass 2: Equity curve for Sortino / Max DD
    let mut equity = 0.0f64;
    let mut peak_equity = 0.0f64;
    let mut max_dd = 0.0f64;
    let mut downside_sq = 0.0f64;

    for t in trades {
        let raw_pnl = t.pnl_r;

        // --- RAW stats (clean, no weighting) ---
        if raw_pnl > 0.0 {
            raw_wins += 1;
            raw_profit += raw_pnl;
        } else if raw_pnl < 0.0 {
            raw_losses += 1;
            raw_loss += raw_pnl.abs();
        }

        // --- Equity curve: light recency (newer trades weighted slightly more) ---
        let recency = if n_total > 0 {
            let ratio = t.entry_idx as f64 / n_total as f64;
            0.7 + 0.3 * ratio  // range: 0.7 (old) to 1.0 (new) — subtle, not destructive
        } else {
            1.0
        };
        let weighted_pnl = raw_pnl * recency;
        equity += weighted_pnl;

        if equity > peak_equity { peak_equity = equity; }
        let dd = peak_equity - equity;
        if dd > max_dd { max_dd = dd; }

        if weighted_pnl < 0.0 {
            downside_sq += weighted_pnl * weighted_pnl;
        }
    }

    // --- Metrics ---
    let wr = raw_wins as f64 / total as f64 * 100.0;
    let avg_pnl = equity / total as f64;

    // Sortino = avg return / downside deviation
    let downside_dev = (downside_sq / total as f64).sqrt();
    let sortino = if downside_dev > 0.001 {
        avg_pnl / downside_dev
    } else if avg_pnl > 0.0 {
        avg_pnl * 10.0
    } else {
        -1.0
    };

    // Profit Factor = gross profit / gross loss (from RAW pnl)
    let pf = if raw_loss > 0.001 {
        (raw_profit / raw_loss).min(10.0)
    } else if raw_profit > 0.0 {
        10.0
    } else {
        0.0
    };

    // Expected R per trade (RAW)
    let raw_total = (raw_wins + raw_losses).max(1) as f64;
    let expected_r = (raw_profit - raw_loss) / raw_total;

    // --- Fitness composition ---
    // 1. Base: Sortino (risk-adjusted return)
    // 2. Scale: sqrt(trades) — more trades = more confidence
    // 3. Quality: PF — win dollars vs loss dollars
    // 4. Safety: DD penalty — punish large drawdowns
    // 5. Profitability: ER bonus — reward actual expected profit per trade
    // 6. Gate: WR must be >= 35% (below = not viable for execution)

    let trade_scale = (total as f64).sqrt();

    let dd_penalty = if max_dd > 3.0 {
        (1.0 - (max_dd - 3.0) * 0.1).max(0.1)
    } else {
        1.0
    };

    let er_bonus = if expected_r > 0.0 {
        1.0 + (expected_r * 8.0).min(5.0)  // 0.1R→1.8, 0.2R→2.6, 0.5R→5.0
    } else {
        0.1
    };

    // Smooth WR scaling: WR=20%→0.3, WR=30%→0.7, WR=40%→1.0, WR=60%→1.0
    let wr_scale = if wr >= 40.0 { 1.0 }
                   else if wr >= 20.0 { (wr - 20.0) / 20.0 * 0.7 + 0.3 }
                   else { 0.1 };

    let fitness = sortino * trade_scale * pf * dd_penalty * er_bonus * wr_scale;

    (fitness, wr, total)
}

/// Run the strategy with custom parameters and return trades
fn evaluate_genome(candles: &[Candle], data: &PrecomputedData, strategy: &str, params: &[f64]) -> Vec<Trade> {
    // Strategy-specific params are the first N, common params are at the end
    let n_strat = match strategy {
        "knifetick" => 11,
        "smc" | "fundingrate" => 6,
        "density" => 5,
        "scalpmtf" => 4,
        _ => 0,
    };

    let strat_params = &params[..n_strat.min(params.len())];

    let mut trades = match strategy {
        "smc" => strategies::smc::run_backtest_with_params(candles, data, strat_params),
        "knifetick" => strategies::knife_tick_macro::run_backtest_with_params(candles, data, strat_params),
        "scalpmtf" => strategies::scalp_mtf::run_backtest_with_params(candles, data, strat_params),
        "fundingrate" => strategies::funding_rate::run_backtest_with_params(candles, data, strat_params),
        "density" => strategies::density::run_backtest_with_params(candles, data, strat_params),
        _ => vec![],
    };

    // Apply slippage
    crate::backtest::apply_slippage(&mut trades);

    // Common params: cooldown_bars, max_trades_day, max_drawdown_r
    let cooldown = params.get(n_strat).copied().unwrap_or(0.0) as usize;
    let max_trades_day = params.get(n_strat + 1).copied().unwrap_or(20.0) as usize;
    let max_dd_r = params.get(n_strat + 2).copied().unwrap_or(5.0);

    // Apply cooldown filter
    if cooldown > 0 && trades.len() > 1 {
        let mut filtered = vec![trades[0].clone()];
        for t in trades.iter().skip(1) {
            let last = filtered.last().unwrap();
            if t.entry_idx >= last.entry_idx + cooldown {
                filtered.push(t.clone());
            }
        }
        trades = filtered;
    }

    // Apply max trades per day (288 bars = 1 day of 5m)
    {
        let mut daily_count = 0usize;
        let mut current_day = 0usize;
        trades.retain(|t| {
            let day = t.entry_idx / 288;
            if day != current_day {
                current_day = day;
                daily_count = 0;
            }
            daily_count += 1;
            daily_count <= max_trades_day
        });
    }

    // Apply max drawdown
    trades = crate::backtest::apply_max_drawdown(&trades, max_dd_r);

    trades
}

/// Initialize random population (used by DE)
fn init_population(param_defs: &[ParamDef], size: usize) -> Vec<Genome> {
    let mut rng = rand::thread_rng();
    let mut pop = Vec::with_capacity(size);

    for _ in 0..size {
        let params: Vec<f64> = param_defs.iter().map(|pd| {
            let steps = ((pd.max - pd.min) / pd.step).round() as usize;
            let step_idx = rng.gen_range(0..=steps);
            pd.min + step_idx as f64 * pd.step
        }).collect();
        pop.push(Genome { params, fitness: f64::NEG_INFINITY, win_rate: 0.0, num_trades: 0 });
    }
    pop
}

/// Snap value to parameter grid
#[inline]
fn snap_to_grid(val: f64, pd: &ParamDef) -> f64 {
    let clamped = val.clamp(pd.min, pd.max);
    ((clamped - pd.min) / pd.step).round() * pd.step + pd.min
}

/// Differential Evolution optimizer (DE/rand/1/bin)
/// ================================================
/// Industry-standard black-box optimizer.
/// NO tournament, NO elitism domination.
/// Each vector competes ONLY with its own trial — 1 vs 1.
/// Mutation = vector DIFFERENCE → auto-scales to fitness landscape.
pub fn optimize(
    candles: &[Candle],
    data: &PrecomputedData,
    strategy: &str,
    config: &GaConfig,
    _gpu_ctx: &mut Option<crate::gpu_evaluator::GpuEvaluator>,
) -> Vec<Genome> {
    let param_defs = get_param_defs(strategy);
    let n_params = param_defs.len();
    if n_params == 0 {
        eprintln!("No parameter definitions for strategy: {}", strategy);
        return vec![];
    }

    let f_scale = 0.8;  // Mutation scaling factor
    let cr = 0.9;       // Crossover rate
    let np = config.population_size;

    println!("🧬 Differential Evolution Optimizer");
    println!("   Strategy: {}", strategy);
    println!("   Parameters: {}", param_defs.iter().map(|p| p.name.as_str()).collect::<Vec<_>>().join(", "));
    println!("   Population: {} | Generations: {} | F={:.1} | CR={:.1}", np, config.generations, f_scale, cr);

    // Initialize & evaluate
    let mut population = init_population(&param_defs, np);
    population.par_iter_mut().for_each(|genome| {
        let trades = evaluate_genome(candles, data, strategy, &genome.params);
        let (fitness, wr, num) = calc_fitness(&trades, data, candles.len(), config);
        genome.fitness = fitness;
        genome.win_rate = wr;
        genome.num_trades = num;
    });

    let mut best_ever = population.iter()
        .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap())
        .cloned().unwrap();

    for gen in 0..config.generations {
        // === DE CORE: create trial for each vector, evaluate, compete 1-vs-1 ===
        let mut trials: Vec<(usize, Genome)> = Vec::with_capacity(np);
        {
            let mut rng = rand::thread_rng();
            for i in 0..np {
                // Pick 3 distinct random indices ≠ i
                let mut r1 = i; while r1 == i { r1 = rng.gen_range(0..np); }
                let mut r2 = i; while r2 == i || r2 == r1 { r2 = rng.gen_range(0..np); }
                let mut r3 = i; while r3 == i || r3 == r1 || r3 == r2 { r3 = rng.gen_range(0..np); }

                let j_rand = rng.gen_range(0..n_params);
                let mut trial_params = Vec::with_capacity(n_params);

                for j in 0..n_params {
                    if rng.gen::<f64>() < cr || j == j_rand {
                        // Mutant: v = r1 + F*(r2-r3)
                        let v = population[r1].params[j]
                              + f_scale * (population[r2].params[j] - population[r3].params[j]);
                        trial_params.push(snap_to_grid(v, &param_defs[j]));
                    } else {
                        trial_params.push(population[i].params[j]);
                    }
                }

                trials.push((i, Genome {
                    params: trial_params,
                    fitness: f64::NEG_INFINITY,
                    win_rate: 0.0,
                    num_trades: 0,
                }));
            }
        }

        // Evaluate trials in parallel
        trials.par_iter_mut().for_each(|(_, trial)| {
            let trades = evaluate_genome(candles, data, strategy, &trial.params);
            let (fitness, wr, num) = calc_fitness(&trades, data, candles.len(), config);
            trial.fitness = fitness;
            trial.win_rate = wr;
            trial.num_trades = num;
        });

        // Greedy selection: trial replaces parent ONLY if better
        let mut improved = 0usize;
        for (i, trial) in trials {
            if trial.fitness >= population[i].fitness {
                population[i] = trial;
                improved += 1;
            }
        }

        // Track best
        for g in &population {
            if g.fitness > best_ever.fitness {
                best_ever = g.clone();
            }
        }

        // Progress
        if gen % 5 == 0 || gen == config.generations - 1 {
            let avg_fitness: f64 = population.iter().map(|g| g.fitness).sum::<f64>() / np as f64;
            let mut sorted_idx: Vec<usize> = (0..np).collect();
            sorted_idx.sort_by(|&a, &b| population[b].fitness.partial_cmp(&population[a].fitness).unwrap_or(std::cmp::Ordering::Equal));
            let top10_avg: f64 = sorted_idx.iter().take(10).map(|&i| population[i].fitness).sum::<f64>() / 10.0;
            let best_idx = sorted_idx[0];
            println!("   Gen {:3}/{}: Best={} | Top10={:.3} | Avg={:.3} | Improved={}/{}",
                gen + 1, config.generations, population[best_idx], top10_avg, avg_fitness, improved, np);
        }
    }

    // Sort final
    population.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap_or(std::cmp::Ordering::Equal));

    // Show TOP 5 diverse results
    println!("\n🏆 TOP 5 GENOMES:");
    let mut shown = 0;
    let mut seen: Vec<f64> = vec![];
    for g in &population {
        if shown >= 5 { break; }
        let is_dup = seen.iter().any(|&f| (g.fitness - f).abs() < f.abs() * 0.001 + 0.001);
        if is_dup && shown > 0 { continue; }
        shown += 1;
        seen.push(g.fitness);
        print!("   #{}: {} | Params: [", shown, g);
        for (j, p) in g.params.iter().enumerate() {
            if j > 0 { print!(", "); }
            print!("{}={:.2}", param_defs.get(j).map(|pd| pd.name.as_str()).unwrap_or("?"), p);
        }
        println!("]");
    }

    population
}

