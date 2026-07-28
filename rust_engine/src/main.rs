/*
 * AEGIS Rust Compute & Live Execution Engine
 * ===========================================
 * High-performance backtest + live trading engine.
 *
 * Usage:
 *   aegis_engine backtest --csv BTC_USDT.csv --strategy smc
 *   aegis_engine scan --data-dir ./data/cache --strategy smc
 *   aegis_engine live --symbols BTCUSDT,ETHUSDT,SOLUSDT
 *
 * Input:  CSV files from data/cache/ (OHLCV 5m candles)
 * Output: JSON results to stdout or file
 */

mod strategies;
mod backtest;
mod indicators;
pub mod bitset_engine;
pub mod ml_inference;
pub mod ga_optimizer;
pub mod live;
pub mod gpu_evaluator;
pub mod tick_backtest;
pub mod ga_tick_optimizer;
pub mod rl_exporter;
pub mod ml_inference_rl;

use clap::{Parser, Subcommand};
use rayon::prelude::*;
use std::path::PathBuf;
use std::time::Instant;
use std::sync::Arc;

#[derive(Parser)]
#[command(name = "aegis_engine", about = "AEGIS High-Performance Backtest & Live Engine")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run backtest on a single symbol
    Backtest {
        #[arg(long)]
        csv: PathBuf,
        #[arg(long, default_value = "smc")]
        strategy: String,
    },
    /// Scan all CSV files in a directory
    Scan {
        #[arg(long)]
        data_dir: PathBuf,
        #[arg(long, default_value = "smc")]
        strategy: String,
    },
    /// Walk-Forward Analysis on a single symbol
    Wfa {
        #[arg(long)]
        csv: PathBuf,
        #[arg(long, default_value = "smc")]
        strategy: String,
        #[arg(long, default_value = "6")]
        train_months: usize,
        #[arg(long, default_value = "1")]
        test_months: usize,
    },
    Optimize {
        #[arg(long)]
        csv: PathBuf,
        #[arg(long, default_value = "smc")]
        strategy: String,
        #[arg(long, default_value = "50")]
        generations: usize,
        #[arg(long, default_value = "100")]
        population: usize,
    },
    /// Run DE on raw ticks from epicenters
    OptimizeTicks {
        #[arg(long, default_value = "BTC_USDT")]
        symbol: String,
        #[arg(long, default_value = "LONG")]
        direction: String,
        #[arg(long, default_value = "50")]
        generations: usize,
        #[arg(long)]
        limit: Option<usize>,
        /// Skip symbols alphabetically before this one (resume batch)
        #[arg(long)]
        skip_to: Option<String>,
        /// Only process these symbols (comma-separated, e.g. KITE_USDT,WIF_USDT)
        #[arg(long)]
        only: Option<String>,
    },
    /// Run WFA on ALL CSVs × ALL strategies with ML filter
    WfaAll {
        #[arg(long)]
        data_dir: PathBuf,
        #[arg(long)]
        models_dir: Option<PathBuf>,
    },
    /// Run GA optimization on ALL strategies sequentially
    OptimizeAll {
        #[arg(long)]
        csv: PathBuf,
        #[arg(long, default_value = "50")]
        generations: usize,
        #[arg(long, default_value = "100")]
        population: usize,
        /// Custom output path (for parallel execution)
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Evaluate a pool of candidate parameters using GPU
    EvaluatePool {
        #[arg(long)]
        csv: PathBuf,
        #[arg(long, default_value = "density")]
        strategy: String,
        #[arg(long)]
        json_pool: PathBuf,
    },
    /// Run backtest with specific params and output trade list (for tick verification)
    BacktestTrades {
        #[arg(long)]
        csv: PathBuf,
        #[arg(long)]
        strategy: String,
        /// JSON string of named params, e.g. '{"tp_rr":2.0,"sl_atr_mult":1.0}'
        #[arg(long)]
        params_json: String,
    },
    /// 🔥 LIVE EXECUTOR: Real-time WebSocket trading with OB + Tape trailing
    Live {
        /// Comma-separated symbols (e.g. BTCUSDT,ETHUSDT,SOLUSDT)
        #[arg(long, value_delimiter = ',')]
        symbols: Vec<String>,
        /// IPC port for Python bridge
        #[arg(long, default_value = "9090")]
        ipc_port: u16,
    },
    /// 🔍 Mine historical levels and export to JSON (Phase 15.2)
    MineLevels {
        #[arg(long)]
        csv: PathBuf,
        /// Tolerance for level grouping (e.g. 0.0001 for 0.01% of price)
        #[arg(long, default_value = "0.0001")]
        tolerance: f64,
    },
    Playback {
        #[arg(long)]
        event_id: String,
        #[arg(long, default_value = "70")]
        threshold: i32,
    },
    /// 🧪 Benchmark HFT logic using synthetic scans (Phase 15.4)
    BenchmarkHft,
    /// 🏋️ Export tick features to CSV for RL Gymnasium training
    ExportRlTrajectories {
        #[arg(long, default_value = "ALL")]
        symbol: String,
        #[arg(long, default_value = "LONG")]
        direction: String,
    },
    /// 🧠 Run backtest using the trained PPO RL Agent
    EvaluateRlAgent {
        #[arg(long, default_value = "ALL")]
        symbol: String,
        #[arg(long, default_value = "LONG")]
        direction: String,
    },
}

fn get_btc_volatility(n_target: usize) -> Vec<f64> {
    let btc_path = std::path::Path::new("data/cache/BTC_USDT_5m_730d.csv");
    if btc_path.exists() {
        let btc_candles = backtest::load_csv(btc_path);
        let mut vol = backtest::calc_atr(&btc_candles, 14);
        let btc_closes: Vec<f64> = btc_candles.iter().map(|c| c.close).collect();
        for i in 0..vol.len() {
            if btc_closes[i] > 0.0 {
                vol[i] = (vol[i] / btc_closes[i]) * 100.0;
            }
        }
        if vol.len() >= n_target {
            vol.split_off(vol.len() - n_target)
        } else {
            let mut padded = vec![0.0; n_target - vol.len()];
            padded.extend(vol);
            padded
        }
    } else {
        vec![0.0; n_target]
    }
}

fn main() {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();
    let start = Instant::now();

    match cli.command {
        Commands::Backtest { csv, strategy } => {
            println!("📊 Backtest: {:?} with {}", csv, strategy);
            let candles = backtest::load_csv(&csv);
            println!("   Loaded {} candles", candles.len());

            let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
            let precomputed = backtest::PrecomputedData {
                atr: backtest::calc_atr(&candles, 14),
                rsi: backtest::calc_rsi(&closes, 14),
                ema_fast: backtest::calc_ema(&closes, 9),
                ema_slow: backtest::calc_ema(&closes, 50),
                ema_200: backtest::calc_ema(&closes, 200),
                adx: backtest::calc_adx(&candles, 14),
                bb_upper: backtest::calc_bollinger_bands(&closes, 20, 2.0).0,
                bb_lower: backtest::calc_bollinger_bands(&closes, 20, 2.0).1,
                bb_mid: backtest::calc_bollinger_bands(&closes, 20, 2.0).2,
                bitsets: None,
                btc_vol: None,
                delta: candles.iter().map(|c| 2.0 * c.taker_buy_volume - c.volume).collect(),
                tape_speed: candles.iter().map(|c| c.num_trades).collect(),
            };

            let trades = match strategy.as_str() {
                "smc" => strategies::smc::run_backtest(&candles, &precomputed),
                "knifetick" => strategies::knife_tick_macro::run_backtest_with_params(&candles, &precomputed, &[]),
                "scalpmtf" => strategies::scalp_mtf::run_backtest(&candles, &precomputed),
                "fundingrate" => strategies::funding_rate::run_backtest(&candles, &precomputed),
                "density" => strategies::density::run_backtest_with_params(&candles, &precomputed, &[2.5, 2.0, 0.006, 2.0, 1.0]),
                _ => {
                    eprintln!("Unknown strategy: {}", strategy);
                    return;
                }
            };

            let wins = trades.iter().filter(|t| t.pnl_r > 0.0).count();
            let total = trades.len();
            let wr = if total > 0 { wins as f64 / total as f64 * 100.0 } else { 0.0 };

            println!("   Trades: {} | Wins: {} | WR: {:.1}%", total, wins, wr);
            println!("   Time: {:.2}s", start.elapsed().as_secs_f64());

            // Output JSON
            let result = serde_json::json!({
                "symbol": csv.file_stem().unwrap().to_str().unwrap(),
                "strategy": strategy,
                "trades": total,
                "wins": wins,
                "win_rate": wr,
                "elapsed_sec": start.elapsed().as_secs_f64(),
            });
            println!("{}", serde_json::to_string_pretty(&result).unwrap());
        }

        Commands::Scan { data_dir, strategy } => {
            println!("🔍 Scanning {} for CSVs...", data_dir.display());

            let csv_files: Vec<PathBuf> = std::fs::read_dir(&data_dir)
                .unwrap()
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.extension().map_or(false, |ext| ext == "csv"))
                .collect();

            println!("   Found {} CSV files. Running {} in parallel...", csv_files.len(), strategy);

            // Parallel scan using rayon
            let results: Vec<serde_json::Value> = csv_files
                .par_iter()
                .map(|csv_path| {
                    let candles = backtest::load_csv(csv_path);
                    let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
                    let precomputed = backtest::PrecomputedData {
                        atr: backtest::calc_atr(&candles, 14),
                        rsi: backtest::calc_rsi(&closes, 14),
                        ema_fast: backtest::calc_ema(&closes, 9),
                        ema_slow: backtest::calc_ema(&closes, 50),
                        ema_200: backtest::calc_ema(&closes, 200),
                        adx: backtest::calc_adx(&candles, 14),
                        bb_upper: vec![], bb_lower: vec![], bb_mid: vec![],
                        bitsets: None,
                        btc_vol: None,
                delta: candles.iter().map(|c| 2.0 * c.taker_buy_volume - c.volume).collect(),
                tape_speed: candles.iter().map(|c| c.num_trades).collect(),
                    };

                    let trades = match strategy.as_str() {
                        "smc" => strategies::smc::run_backtest(&candles, &precomputed),
                        "knifetick" => strategies::knife_tick_macro::run_backtest_with_params(&candles, &precomputed, &[]),
                        "scalpmtf" => strategies::scalp_mtf::run_backtest(&candles, &precomputed),
                        "fundingrate" => strategies::funding_rate::run_backtest(&candles, &precomputed),
                        "density" => strategies::density::run_backtest_with_params(&candles, &precomputed, &[2.5, 2.0, 0.006, 2.0, 1.0]),
                        _ => vec![],
                    };

                    let wins = trades.iter().filter(|t| t.pnl_r > 0.0).count();
                    let total = trades.len();
                    let wr = if total > 0 { wins as f64 / total as f64 * 100.0 } else { 0.0 };

                    let symbol = csv_path.file_stem().unwrap().to_str().unwrap().to_string();
                    println!("   {} → {} trades, WR={:.1}%", symbol, total, wr);

                    serde_json::json!({
                        "symbol": symbol,
                        "trades": total,
                        "wins": wins,
                        "win_rate": wr,
                    })
                })
                .collect();

            // Sort by win rate descending
            let mut results = results;
            results.sort_by(|a, b| {
                b["win_rate"].as_f64().unwrap()
                    .partial_cmp(&a["win_rate"].as_f64().unwrap())
                    .unwrap()
            });

            println!("\n🏆 TOP RESULTS:");
            for (i, r) in results.iter().take(10).enumerate() {
                println!("   #{}: {} → WR={:.1}% ({} trades)",
                    i + 1,
                    r["symbol"].as_str().unwrap(),
                    r["win_rate"].as_f64().unwrap(),
                    r["trades"].as_u64().unwrap(),
                );
            }

            println!("\n⏱️ Total scan time: {:.2}s", start.elapsed().as_secs_f64());

            let output = serde_json::to_string_pretty(&results).unwrap();
            std::fs::write("data/scan_results.json", &output).ok();
            println!("📁 Results saved to data/scan_results.json");
        }

        Commands::Wfa { csv, strategy, train_months, test_months } => {
            println!("📈 Walk-Forward Analysis: {:?}", csv);
            println!("   Engine: AEGIS Rust Compute + LightGBM Inference");
            println!("   Params: Train={}mo, Test={}mo, Strategy={}", train_months, test_months, strategy);
            
            let candles = backtest::load_csv(&csv);
            println!("   Loaded {} candles", candles.len());
            
            let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
            let precomputed = backtest::PrecomputedData {
                atr: backtest::calc_atr(&candles, 14),
                rsi: backtest::calc_rsi(&closes, 14),
                ema_fast: backtest::calc_ema(&closes, 9),
                ema_slow: backtest::calc_ema(&closes, 50),
                ema_200: backtest::calc_ema(&closes, 200),
                adx: backtest::calc_adx(&candles, 14),
                bb_upper: vec![], bb_lower: vec![], bb_mid: vec![],
                bitsets: None,
                btc_vol: None,
                delta: candles.iter().map(|c| 2.0 * c.taker_buy_volume - c.volume).collect(),
                tape_speed: candles.iter().map(|c| c.num_trades).collect(),
            };

            // 1. Run raw strategy to get candidate trades
            let start_sim = Instant::now();
            let trades = match strategy.as_str() {
                "smc" => strategies::smc::run_backtest(&candles, &precomputed),
                "knifetick" => strategies::knife_tick_macro::run_backtest_with_params(&candles, &precomputed, &[]),
                "scalpmtf" => strategies::scalp_mtf::run_backtest(&candles, &precomputed),
                "fundingrate" => strategies::funding_rate::run_backtest(&candles, &precomputed),
                "density" => strategies::density::run_backtest_with_params(&candles, &precomputed, &[2.5, 2.0, 0.006, 2.0, 1.0]),
                _ => {
                    eprintln!("Unknown strategy: {}", strategy);
                    return;
                }
            };
            
            let sim_time = start_sim.elapsed().as_secs_f64();
            println!("   Raw simulation time: {:.3}s ({} candidate trades)", sim_time, trades.len());
            
            // 2. Load ML Model
            let model_path = PathBuf::from("data/models_json").join(format!("{}.json", strategy));
            if !model_path.exists() {
                eprintln!("❌ ML Model not found at {:?}. Run export_models_json.py first.", model_path);
                return;
            }
            
            let ml_start = Instant::now();
            let model = crate::ml_inference::LgbmModel::load(&model_path).expect("Failed to load model");
            println!("   Loaded ML Model: {} trees, {} features", model.num_trees, model.num_features);
            
            // 3. Filter trades with ML (assuming ML handles the timeline logic via features)
            let mut filtered_wins = 0;
            let mut filtered_total = 0;
            
            for trade in &trades {
                // Extract features for the candle where the trade fired
                let features = crate::ml_inference::extract_features(&candles, trade.entry_idx, 0.0, 0.0, 0.0, 0.0);
                let proba = model.predict_proba(&features);
                
                // Keep trades with > 50% ML confidence
                if proba >= 0.5 {
                    filtered_total += 1;
                    if trade.pnl_r > 0.0 {
                        filtered_wins += 1;
                    }
                }
            }
            
            let ml_time = ml_start.elapsed().as_secs_f64();
            let wr = if filtered_total > 0 { filtered_wins as f64 / filtered_total as f64 * 100.0 } else { 0.0 };
            
            println!("\n✅ WFA COMPLETE!");
            println!("   Filtered Trades: {} | Wins: {} | New WR: {:.1}%", filtered_total, filtered_wins, wr);
            println!("   ML Inference time: {:.3}s", ml_time);
            println!("   Total Exec Time:   {:.3}s", start.elapsed().as_secs_f64());
            
            let result = serde_json::json!({
                "symbol": csv.file_stem().unwrap().to_str().unwrap(),
                "strategy": strategy,
                "raw_trades": trades.len(),
                "filtered_trades": filtered_total,
                "filtered_win_rate": wr,
                "total_sec": start.elapsed().as_secs_f64()
            });
            println!("\n{}", serde_json::to_string_pretty(&result).unwrap());
        }

        Commands::Optimize { csv, strategy, generations, population } => {
            println!("🧬 Genetic Algorithm Optimizer");
            println!("   CSV: {:?}", csv);

            let candles = backtest::load_csv(&csv);
            println!("   Loaded {} candles", candles.len());

            // --- PRECOMPUTE (Turbo Mode) ---
            let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
            let ema_200 = backtest::calc_ema(&closes, 200);
            let adx = backtest::calc_adx(&candles, 14);
            let (bb_upper, bb_lower, bb_mid) = backtest::calc_bollinger_bands(&closes, 20, 2.2);
            let bitsets = Some(bitset_engine::build_bitsets(&candles, &ema_200, &adx));
            
            let precomputed = backtest::PrecomputedData {
                atr: backtest::calc_atr(&candles, 14),
                rsi: backtest::calc_rsi(&closes, 14),
                ema_fast: backtest::calc_ema(&closes, 9),
                ema_slow: backtest::calc_ema(&closes, 50),
                ema_200,
                adx,
                bb_upper,
                bb_lower,
                bb_mid,
                bitsets,
                btc_vol: Some(get_btc_volatility(candles.len())),
                delta: candles.iter().map(|c| 2.0 * c.taker_buy_volume - c.volume).collect(),
                tape_speed: candles.iter().map(|c| c.num_trades).collect(),
            };

            let config = ga_optimizer::GaConfig {
                population_size: population,
                generations,
                ..Default::default()
            };

            let mut gpu_ctx = if population >= 1000 {
                crate::gpu_evaluator::GpuEvaluator::new().ok()
            } else { None };

            let results = ga_optimizer::optimize(&candles, &precomputed, &strategy, &config, &mut gpu_ctx);

            println!("\n⏱️ Total optimization time: {:.2}s", start.elapsed().as_secs_f64());

            // Save best params as JSON
            if let Some(best) = results.first() {
                let param_defs = ga_optimizer::get_param_defs(&strategy);
                let mut params_map = serde_json::Map::new();
                for (i, pd) in param_defs.iter().enumerate() {
                    params_map.insert(pd.name.clone(), serde_json::json!(best.params[i]));
                }
                let output = serde_json::json!({
                    "strategy": strategy,
                    "best_params": params_map,
                    "win_rate": best.win_rate,
                    "num_trades": best.num_trades,
                    "fitness": best.fitness,
                });
                let json_str = serde_json::to_string_pretty(&output).unwrap();
                println!("\n{}", json_str);
                std::fs::write("data/ga_best_params.json", &json_str).ok();
                println!("📁 Saved to data/ga_best_params.json");
            }
        }

        Commands::OptimizeTicks { symbol, direction, generations, limit, skip_to, only } => {
            if symbol == "ALL" {
                // ═══ BATCH MODE: run on all epicenter files ═══
                let epic_dir = std::path::Path::new("../data/epicenters");
                let mut files: Vec<PathBuf> = std::fs::read_dir(epic_dir)
                    .expect("Cannot read epicenters directory")
                    .filter_map(|e| e.ok())
                    .map(|e| e.path())
                    .filter(|p| p.extension().map_or(false, |ext| ext == "json"))
                    .collect();
                files.sort();
                
                println!("🔥 BATCH OPTIMIZE: {} symbols × {} generations\n", files.len(), generations);
                println!("{:<25} {:>6} {:>8} {:>10} {:>8} {:>10} {:>8}", 
                    "Symbol", "Trades", "TrainWR", "TrainPnL", "TestWR", "TestPnL", "MFE>0.1");
                println!("{}", "─".repeat(85));
                
                let mut summary: Vec<(String, usize, f64, f64, f64, f64, f64)> = Vec::new();
                
                for (_i, path) in files.iter().enumerate() {
                    let sym = path.file_stem().unwrap().to_str().unwrap()
                        .replace("_epicenters", "");
                    
                    // Skip already-processed symbols
                    if let Some(ref skip) = skip_to {
                        if sym.as_str() < skip.as_str() {
                            continue;
                        }
                    }
                    // Filter to specific symbols
                    if let Some(ref only_list) = only {
                        let allowed: Vec<&str> = only_list.split(',').collect();
                        if !allowed.contains(&sym.as_str()) {
                            continue;
                        }
                    }
                    
                    let epicenters = tick_backtest::load_epicenters(&sym, &direction, limit);
                    if epicenters.len() < 5 {
                        println!("{:<25} SKIP (only {} epicenters)", sym, epicenters.len());
                        continue;
                    }
                    
                    // Capture results by running optimize
                    let result = ga_tick_optimizer::optimize_ticks_batch(&epicenters, &direction, generations);
                    
                    println!("{:<25} {:>6} {:>7.1}% {:>+9.2}R {:>7.1}% {:>+9.2}R {:>7.0}%",
                        sym, result.total_trades, 
                        result.train_wr * 100.0, result.train_pnl,
                        result.test_wr * 100.0, result.test_pnl,
                        result.test_mfe_ratio * 100.0);
                    
                    // Save per-symbol params
                    if !result.best_genome.is_empty() {
                        let param_names = ["window_ms","min_zscore","min_vol_spike","(unused_tp)","sl_buffer_pct","be_trigger_pct","trail_pct","micro_window_ms","min_absorption","min_reclaim_pct","max_speed_mult","baseline_window_sec","max_absorber_sec","rewake_cooldown_sec"];
                        let mut params_map = serde_json::Map::new();
                        for (i, &val) in result.best_genome.iter().enumerate() {
                            if i < param_names.len() {
                                params_map.insert(param_names[i].to_string(), serde_json::Value::from(val));
                            }
                        }
                        let mut out = serde_json::Map::new();
                        out.insert("symbol".into(), serde_json::Value::String(sym.clone()));
                        out.insert("strategy".into(), serde_json::Value::String("knife_tick".into()));
                        out.insert("version".into(), serde_json::Value::String("phase31".into()));
                        out.insert("params".into(), serde_json::Value::Object(params_map));
                        out.insert("train_wr".into(), serde_json::Value::from(result.train_wr * 100.0));
                        out.insert("test_wr".into(), serde_json::Value::from(result.test_wr * 100.0));
                        out.insert("train_pnl_r".into(), serde_json::Value::from(result.train_pnl));
                        out.insert("test_pnl_r".into(), serde_json::Value::from(result.test_pnl));
                        out.insert("test_mfe_ratio".into(), serde_json::Value::from(result.test_mfe_ratio));
                        out.insert("total_trades".into(), serde_json::Value::from(result.total_trades as u64));
                        let params_dir = std::path::Path::new("../data/tick_params");
                        std::fs::create_dir_all(params_dir).ok();
                        let path = params_dir.join(format!("{}.json", sym.clone()));
                        let json_out = serde_json::Value::Object(out);
                        std::fs::write(&path, serde_json::to_string_pretty(&json_out).unwrap()).ok();
                    }
                    
                    summary.push((sym, result.total_trades, result.train_wr, result.train_pnl, result.test_wr, result.test_pnl, result.test_mfe_ratio));
                }
                
                // ═══ TOP RESULTS ═══
                println!("\n{}", "═".repeat(85));
                println!("🏆 TOP BY OOS WIN RATE (min 5 OOS trades):\n");
                let mut top: Vec<_> = summary.iter()
                    .filter(|(_, trades, _, _, _, _, _)| *trades >= 15)
                    .collect();
                top.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap());
                
                for (sym, trades, train_wr, train_pnl, test_wr, test_pnl, mfe) in top.iter().take(15) {
                    let verdict = if *test_wr > 0.55 && *test_pnl > 0.0 { "✅" }
                        else if *test_wr > 0.50 { "⚠️" }
                        else { "❌" };
                    println!("  {} {:<22} {:>3} trades | Train: {:.0}% {:.1}R | Test: {:.0}% {:+.1}R | MFE: {:.0}%",
                        verdict, sym, trades, train_wr*100.0, train_pnl, test_wr*100.0, test_pnl, mfe*100.0);
                }
                
                println!("\n⏱️ Total time: {:.1}s", start.elapsed().as_secs_f64());
            } else {
                let epicenters = tick_backtest::load_epicenters(&symbol, &direction, limit);
                ga_tick_optimizer::optimize_ticks(&epicenters, &direction, generations);
            }
        }

        Commands::ExportRlTrajectories { symbol, direction } => {
            println!("🏋️ Exporting RL Trajectories for Gym Environment...");
            let out_dir = std::path::Path::new("../data/rl_trajectories");
            std::fs::create_dir_all(out_dir).unwrap();

            let symbols_to_process = if symbol == "ALL" {
                let ticks_dir = std::path::Path::new("../data/epicenters_ticks");
                if let Ok(entries) = std::fs::read_dir(ticks_dir) {
                    entries.filter_map(|e| e.ok())
                        .filter(|e| e.path().is_dir())
                        .map(|e| e.file_name().into_string().unwrap())
                        .collect()
                } else {
                    vec![]
                }
            } else {
                vec![symbol.clone()]
            };

            let mut total_exported = 0;
            
            for sym in symbols_to_process {
                let epicenters = tick_backtest::load_epicenters(&sym, &direction, None);
                println!("   Loaded {} epicenters for {}", epicenters.len(), sym);
                
                let mut exported = 0;
                for (i, ep) in epicenters.iter().enumerate() {
                    if let Some(traj) = rl_exporter::export_trajectory(ep, &direction, 50) {
                        let path = out_dir.join(format!("{}_{}_{}.csv", sym, direction, i));
                        let mut content = String::with_capacity(traj.len() * 100);
                        // Add header
                        content.push_str("ts,price,f_abs,f_spd,f_rec,f_cvd,f_move,f_dir,f_vol,f_time,f_pnl,f_11,f_12,f_13,f_14,f_15,f_16,f_17,f_18\n");
                        for row in traj {
                            let str_row: Vec<String> = row.iter().map(|f| format!("{:.5}", f)).collect();
                            content.push_str(&str_row.join(","));
                            content.push('\n');
                        }
                        std::fs::write(path, content).unwrap();
                        exported += 1;
                        total_exported += 1;
                    }
                }
            }
            println!("✅ Successfully exported {} trajectory sequences to CSV!", total_exported);
        }

        Commands::EvaluateRlAgent { symbol, direction } => {
            println!("🧠 Evaluating RL Agent on Historical Epicenters...");
            
            let agent = match ml_inference_rl::RlAgent::load_from_json("../data/models/knife_ppo_weights.json") {
                Some(a) => a,
                None => {
                    println!("❌ Failed to load JSON weights. Did train_rl_knife.py finish?");
                    return;
                }
            };
            
            let symbols_to_process = if symbol == "ALL" {
                let config_str = std::fs::read_to_string("../data/active_config.json").unwrap_or_else(|_| "[]".to_string());
                if let Ok(config_json) = serde_json::from_str::<serde_json::Value>(&config_str) {
                    if let Some(arr) = config_json.as_array() {
                        arr.iter()
                           .filter_map(|v| v.get("symbol").and_then(|s| s.as_str()).map(|s| s.to_string()))
                           .collect()
                    } else {
                        vec![]
                    }
                } else {
                    vec![]
                }
            } else {
                vec![symbol.clone()]
            };

            let results: Vec<_> = symbols_to_process.into_par_iter().map(|sym| {
                let epicenters = tick_backtest::load_epicenters(&sym, &direction, None);
                let mut sym_trades = 0;
                let mut sym_wins = 0;
                let mut sym_pnl = 0.0;
                let is_long = direction == "LONG";
                const BURN_IN: usize = 10;
                const MAX_STEPS: usize = 1200;
                const STOP_LOSS_PCT: f32 = 0.015; // 1.5% matching Gym
                
                for ep in epicenters.iter() {
                    if let Some(traj) = rl_exporter::export_trajectory(ep, &direction, 50) {
                        if traj.len() < BURN_IN + 1 { continue; }
                        
                        // EXIT-MANAGER MODE: Force entry at trajectory start
                        let entry_price = traj[0][1];
                        let mut exited = false;
                        
                        for (step, row) in traj.iter().enumerate() {
                            if step >= MAX_STEPS { break; }
                            
                            let price = row[1];
                            
                            // Build feature vector with dynamic injections
                            let mut features: Vec<f32> = row[2..].to_vec();
                            // Ensure at least 15 features
                            while features.len() < 15 { features.push(0.0); }
                            features.truncate(15);
                            
                            // Inject dynamic features matching Gym env
                            features[7] = (step as f32) / (MAX_STEPS as f32); // time_in_trade
                            let pnl_now = if is_long {
                                (price - entry_price) / entry_price
                            } else {
                                (entry_price - price) / entry_price
                            };
                            features[8] = (pnl_now * 100.0).clamp(-10.0, 10.0); // unrealized_pnl
                            
                            // Skip burn-in period
                            if step < BURN_IN { continue; }
                            
                            let action = agent.predict_action(&features);
                            
                            // Exit thresholds matching Gym env
                            let wants_exit = if is_long {
                                action <= -0.1
                            } else {
                                action >= 0.1
                            };
                            
                            // Stop-loss check (after burn-in)
                            let stop_loss_hit = pnl_now <= -STOP_LOSS_PCT;
                            
                            if wants_exit || stop_loss_hit {
                                let fee = 0.0004 * 2.0;
                                let net_pnl = (pnl_now * 100.0) - (fee * 100.0);
                                
                                sym_trades += 1;
                                sym_pnl += net_pnl;
                                if net_pnl > 0.0 { sym_wins += 1; }
                                exited = true;
                                break;
                            }
                        }
                        
                        // Force close at end of trajectory
                        if !exited {
                            let last_price = traj.last().unwrap()[1];
                            let pnl_pct = if is_long {
                                (last_price - entry_price) / entry_price
                            } else {
                                (entry_price - last_price) / entry_price
                            };
                            let fee = 0.0004 * 2.0;
                            let net_pnl = (pnl_pct * 100.0) - (fee * 100.0);
                            
                            sym_trades += 1;
                            sym_pnl += net_pnl;
                            if net_pnl > 0.0 { sym_wins += 1; }
                        }
                    }
                }
                
                (sym, sym_trades, sym_wins, sym_pnl)
            }).collect();

            let mut total_trades = 0;
            let mut win_trades = 0;
            let mut total_pnl = 0.0;
            
            let mut valid_results = Vec::new();

            for (sym, trades, wins, pnl) in results {
                if trades > 0 {
                    let win_rate = (wins as f32 / trades as f32) * 100.0;
                    total_trades += trades;
                    win_trades += wins;
                    total_pnl += pnl;
                    valid_results.push((sym, trades, win_rate, pnl));
                }
            }

            valid_results.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

            println!("\n🏆 TOP COINS BY WINRATE:");
            for (sym, trades, wr, pnl) in valid_results.iter() {
                println!("{:<10} | Trades: {:<4} | WinRate: {:.1}% | PnL: {:.2}%", 
                    sym, trades, wr, pnl);
            }
            
            println!("-----------------------------------------------------------");
            let win_rate = if total_trades > 0 { (win_trades as f32 / total_trades as f32) * 100.0 } else { 0.0 };
            println!("🎯 OVERALL: Trades: {} | WinRate: {:.1}% | Total PnL: {:.2}%", total_trades, win_rate, total_pnl);
        }

        Commands::EvaluatePool { csv, strategy, json_pool } => {
            println!("🧬 GPU POOL EVALUATOR");
            println!("   CSV: {:?}", csv);
            println!("   Pool: {:?}", json_pool);

            let candles = backtest::load_csv(&csv);
            if candles.is_empty() { panic!("Failed to load CSV or CSV is empty"); }
            let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();

            let ema_200 = backtest::calc_ema(&closes, 200);
            let adx = backtest::calc_adx(&candles, 14);
            let (bb_upper, bb_lower, bb_mid) = backtest::calc_bollinger_bands(&closes, 20, 2.0);

            let bitsets = Some(bitset_engine::build_bitsets(&candles, &ema_200, &adx));

            let data = backtest::PrecomputedData {
                atr: backtest::calc_atr(&candles, 14),
                rsi: backtest::calc_rsi(&closes, 14),
                ema_fast: backtest::calc_ema(&closes, 9),
                ema_slow: backtest::calc_ema(&closes, 50),
                ema_200, adx, bb_upper, bb_lower, bb_mid, bitsets,
                btc_vol: Some(get_btc_volatility(candles.len())),
                delta: candles.iter().map(|c| 2.0 * c.taker_buy_volume - c.volume).collect(),
                tape_speed: candles.iter().map(|c| c.num_trades).collect(),
            };

            // Load pool
            let pool_str = std::fs::read_to_string(json_pool).expect("Failed to read JSON pool");
            let candidates: Vec<serde_json::Value> = serde_json::from_str(&pool_str).expect("Invalid JSON pool");
            
            println!("   Candidates: {}", candidates.len());

            let mut genomes = Vec::new();
            for c in candidates {
                if let Some(params_obj) = c.get("best_params").and_then(|p| p.as_object()) {
                    let mut p_vec = Vec::new();
                    let defs = ga_optimizer::get_param_defs(&strategy);
                    for d in defs {
                        p_vec.push(params_obj.get(&d.name).and_then(|v| v.as_f64()).unwrap_or(0.0));
                    }
                    genomes.push(ga_optimizer::Genome {
                        params: p_vec, fitness: 0.0, win_rate: 0.0, num_trades: 0,
                    });
                }
            }

            // Run GPU evaluation
            let mut gpu = gpu_evaluator::GpuEvaluator::new().expect("Failed to init GPU");
            let bitsets_raw = data.bitsets.as_ref().unwrap().precalculate_combinations();
            let prices_f32: Vec<f32> = candles.iter().map(|c| c.close as f32).collect();
            let atrs_f32: Vec<f32> = data.atr.iter().map(|&a| a as f32).collect();
            let btc_vols_f32: Vec<f32> = data.btc_vol.as_ref().unwrap().iter().map(|&x| x as f32).collect();
            
            gpu.set_constants(&bitsets_raw, &prices_f32, &atrs_f32, &btc_vols_f32).expect("GPU Upload failed");

            let mut params_f32 = Vec::new();
            for g in &genomes {
                for &p in &g.params { params_f32.push(p as f32); }
            }

            let n_u32 = (candles.len() + 31) / 32;
            let results = gpu.evaluate_batch(&params_f32, candles.len() as i32, n_u32 as i32, genomes.len() as i32).expect("GPU Eval failed");

            let mut final_results = Vec::new();
            for (i, fitness) in results.into_iter().enumerate() {
                final_results.push(serde_json::json!({
                    "params": genomes[i].params,
                    "fitness": fitness,
                }));
            }

            println!("{}", serde_json::to_string(&final_results).unwrap());
        }

        Commands::BacktestTrades { csv, strategy, params_json } => {
            let candles = backtest::load_csv(&csv);
            if candles.is_empty() { panic!("CSV is empty"); }
            let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
            let ema_200 = backtest::calc_ema(&closes, 200);
            let adx = backtest::calc_adx(&candles, 14);
            let (bb_upper, bb_lower, bb_mid) = backtest::calc_bollinger_bands(&closes, 20, 2.2);
            let bitsets = Some(bitset_engine::build_bitsets(&candles, &ema_200, &adx));

            let data = backtest::PrecomputedData {
                atr: backtest::calc_atr(&candles, 14),
                rsi: backtest::calc_rsi(&closes, 14),
                ema_fast: backtest::calc_ema(&closes, 9),
                ema_slow: backtest::calc_ema(&closes, 50),
                ema_200, adx, bb_upper, bb_lower, bb_mid, bitsets,
                btc_vol: Some(get_btc_volatility(candles.len())),
                delta: candles.iter().map(|c| 2.0 * c.taker_buy_volume - c.volume).collect(),
                tape_speed: candles.iter().map(|c| c.num_trades).collect(),
            };

            // Parse named params
            let params_map: serde_json::Map<String, serde_json::Value> =
                serde_json::from_str(&params_json).expect("Invalid params JSON");
            let defs = ga_optimizer::get_param_defs(&strategy);
            let params: Vec<f64> = defs.iter()
                .map(|d| params_map.get(&d.name).and_then(|v| v.as_f64()).unwrap_or(0.0))
                .collect();

            // Run backtest
            let n_strat = match strategy.as_str() {
                "knife" => 10,
                "smc" | "fundingrate" => 6,
                "density" => 5,
                "scalpmtf" => 4,
                _ => 0,
            };
            let strat_params = &params[..n_strat.min(params.len())];
            let mut trades = match strategy.as_str() {
                "smc" => strategies::smc::run_backtest_with_params(&candles, &data, strat_params),
                "scalpmtf" => strategies::scalp_mtf::run_backtest_with_params(&candles, &data, strat_params),
                "fundingrate" => strategies::funding_rate::run_backtest_with_params(&candles, &data, strat_params),
                "density" => strategies::density::run_backtest_with_params(&candles, &data, strat_params),
                _ => vec![],
            };
            backtest::apply_slippage(&mut trades);

            // Convert to JSON with candle timestamps
            let mut output = Vec::new();
            for t in &trades {
                let ts = if t.entry_idx < candles.len() {
                    candles[t.entry_idx].timestamp.clone()
                } else {
                    "unknown".to_string()
                };
                output.push(serde_json::json!({
                    "entry_idx": t.entry_idx,
                    "entry_ts": ts,
                    "direction": t.direction,
                    "entry_price": t.entry_price,
                    "sl_price": t.sl_price,
                    "tp_price": t.tp_price,
                    "exit_price": t.exit_price,
                    "pnl_r": t.pnl_r,
                }));
            }
            println!("{}", serde_json::to_string(&output).unwrap());
        }

        Commands::WfaAll { data_dir, models_dir } => {
            println!("🏆 FULL WFA: All Symbols × All Strategies + ML Filter");
            println!("   Data dir: {}", data_dir.display());

            let models_path = models_dir.unwrap_or_else(|| PathBuf::from("data/models_json"));
            let strategies = ["smc", "knife", "scalpmtf", "fundingrate", "density"];

            // Strategy name → model JSON name mapping
            let model_names = [
                ("smc", "ultimate_smc_trail"),
                ("knife", "knife_catcher"),
                ("scalpmtf", "scalpmtf_model"),
            ];

            // Load all ML models
            let mut models = std::collections::HashMap::new();
            for (strat, model_name) in &model_names {
                let path = models_path.join(format!("{}.json", model_name));
                if path.exists() {
                    match ml_inference::LgbmModel::load(&path) {
                        Ok(m) => {
                            println!("   ✅ Loaded ML model: {} ({} trees)", strat, m.num_trees);
                            models.insert(strat.to_string(), m);
                        }
                        Err(e) => println!("   ⚠️ Failed to load {}: {}", strat, e),
                    }
                } else {
                    println!("   ⏭️ No model for {}, will show raw WR only", strat);
                }
            }

            // Find all CSV files
            let csv_files: Vec<PathBuf> = std::fs::read_dir(&data_dir)
                .unwrap()
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.extension().map_or(false, |ext| ext == "csv"))
                .collect();

            println!("   Found {} CSV files\n", csv_files.len());

            // Results storage
            let mut all_results: Vec<serde_json::Value> = Vec::new();

            println!("{:<25} {:<10} {:>6} {:>8} {:>8} {:>10}",
                "Symbol", "Strategy", "Trades", "Raw WR", "ML WR", "Time(ms)");
            println!("{}", "─".repeat(75));

            for csv_path in &csv_files {
                let symbol = csv_path.file_stem().unwrap().to_str().unwrap().to_string();
                let candles = backtest::load_csv(csv_path);
                if candles.len() < 300 { continue; }
                let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();

                let precomputed = backtest::PrecomputedData {
                    atr: backtest::calc_atr(&candles, 14),
                    rsi: backtest::calc_rsi(&closes, 14),
                    ema_fast: backtest::calc_ema(&closes, 9),
                    ema_slow: backtest::calc_ema(&closes, 50),
                    ema_200: backtest::calc_ema(&closes, 200),
                    adx: backtest::calc_adx(&candles, 14),
                    bb_upper: vec![],
                    bb_lower: vec![],
                    bb_mid: vec![],
                    bitsets: None,
                    btc_vol: None,
                delta: candles.iter().map(|c| 2.0 * c.taker_buy_volume - c.volume).collect(),
                tape_speed: candles.iter().map(|c| c.num_trades).collect(),
                };

                for strat_name in &strategies {
                    let t0 = Instant::now();

                    let trades = match *strat_name {
                        "smc" => strategies::smc::run_backtest(&candles, &precomputed),
                        _ => vec![],
                    };

                    let total = trades.len();
                    let wins = trades.iter().filter(|t| t.pnl_r > 0.0).count();
                    let raw_wr = if total > 0 { wins as f64 / total as f64 * 100.0 } else { 0.0 };

                    // ML filter
                    let (ml_wr, ml_trades) = if let Some(model) = models.get(*strat_name) {
                        let mut f_wins = 0;
                        let mut f_total = 0;
                        for trade in &trades {
                            let features = ml_inference::extract_features(&candles, trade.entry_idx, 0.0, 0.0, 0.0, 0.0);
                            let proba = model.predict_proba(&features);
                            if proba >= 0.5 {
                                f_total += 1;
                                if trade.pnl_r > 0.0 { f_wins += 1; }
                            }
                        }
                        let wr = if f_total > 0 { f_wins as f64 / f_total as f64 * 100.0 } else { 0.0 };
                        (format!("{:.1}%", wr), f_total)
                    } else {
                        ("N/A".to_string(), 0)
                    };

                    let elapsed_ms = t0.elapsed().as_millis();

                    println!("{:<25} {:<10} {:>6} {:>7.1}% {:>8} {:>8}ms",
                        symbol, strat_name, total, raw_wr, ml_wr, elapsed_ms);

                    all_results.push(serde_json::json!({
                        "symbol": symbol,
                        "strategy": strat_name,
                        "raw_trades": total,
                        "raw_wr": raw_wr,
                        "ml_filtered_trades": ml_trades,
                        "ml_wr": ml_wr,
                    }));
                }
            }

            println!("\n{}", "═".repeat(75));
            println!("⏱️ Total time: {:.2}s", start.elapsed().as_secs_f64());
            println!("📊 Tested: {} symbols × {} strategies = {} combinations",
                csv_files.len(), strategies.len(), csv_files.len() * strategies.len());

            // Save results
            let json_out = serde_json::to_string_pretty(&all_results).unwrap();
            std::fs::write("data/wfa_all_results.json", &json_out).ok();
            println!("📁 Saved to data/wfa_all_results.json");
        }

        Commands::OptimizeAll { csv, generations, population, output } => {
            println!("🧬 GA OPTIMIZE ALL: 6 strategies × {} generations × {} population", generations, population);
            println!("   CSV: {}\n", csv.display());

            let candles = backtest::load_csv(&csv);
            println!("   Loaded {} candles\n", candles.len());

            // --- PRECOMPUTE INDICATORS (Turbo Mode) ---
            let t_pre = Instant::now();
            let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
            let atr = backtest::calc_atr(&candles, 14);
            let rsi = backtest::calc_rsi(&closes, 14);
            let ema_fast = backtest::calc_ema(&closes, 9);
            let ema_slow = backtest::calc_ema(&closes, 50);
            let ema_200 = backtest::calc_ema(&closes, 200);
            let adx = backtest::calc_adx(&candles, 14);
            let (bb_upper, bb_lower, bb_mid) = backtest::calc_bollinger_bands(&closes, 20, 2.2);

            let bitsets = Some(bitset_engine::build_bitsets(&candles, &ema_200, &adx));
            let precomputed = backtest::PrecomputedData {
                atr,
                rsi,
                ema_fast,
                ema_slow,
                ema_200,
                adx,
                bb_upper,
                bb_lower,
                bb_mid,
                bitsets,
                btc_vol: Some(get_btc_volatility(candles.len())),
                delta: candles.iter().map(|c| 2.0 * c.taker_buy_volume - c.volume).collect(),
                tape_speed: candles.iter().map(|c| c.num_trades).collect(),
            };
            println!("   ⚡ Precomputed all indicators in {:.1}ms", t_pre.elapsed().as_secs_f64()*1000.0);

            let strategies_list = ["smc", "knife", "scalpmtf", "fundingrate", "density"];
            let config_cons = ga_optimizer::GaConfig {
                population_size: population,
                generations,
                recency_base: 0.5,       // More historical weight
                stress_penalty_mult: 1.0, // Standard stress
                ..Default::default()
            };

            let config_aggr = ga_optimizer::GaConfig {
                population_size: population,
                generations,
                recency_base: 0.1,       // High recency weight
                stress_penalty_mult: 1.5, // Aggressive stress penalty
                ..Default::default()
            };

            let mut all_best = serde_json::Map::new();
            let mut gpu_ctx = if population >= 1000 {
                crate::gpu_evaluator::GpuEvaluator::new().ok()
            } else { None };

            if gpu_ctx.is_some() {
                println!("   🚀 GPU Acceleration enabled for ALL strategies");
            }

            for strat_name in &strategies_list {
                println!("\n{}", "═".repeat(60));
                println!("🎯 Dual-Mode Optimizing: {}", strat_name);
                println!("{}", "═".repeat(60));

                // 1. Conservative Run
                println!("   [MODE: CONSERVATIVE]");
                let results_cons = ga_optimizer::optimize(&candles, &precomputed, strat_name, &config_cons, &mut gpu_ctx);
                
                // 2. Aggressive Run
                println!("\n   [MODE: AGGRESSIVE]");
                let results_aggr = ga_optimizer::optimize(&candles, &precomputed, strat_name, &config_aggr, &mut gpu_ctx);

                if let (Some(best_c), Some(best_a)) = (results_cons.first(), results_aggr.first()) {
                    let param_defs = ga_optimizer::get_param_defs(strat_name);
                    
                    let mut params_c = serde_json::Map::new();
                    let mut params_a = serde_json::Map::new();

                    for (i, pd) in param_defs.iter().enumerate() {
                        if i < best_c.params.len() {
                            params_c.insert(pd.name.clone(), serde_json::json!(best_c.params[i]));
                        }
                        if i < best_a.params.len() {
                            params_a.insert(pd.name.clone(), serde_json::json!(best_a.params[i]));
                        }
                    }

                    let mut candidates = Vec::new();
                    for genome in results_aggr.iter().take(20) {
                        let mut c_params = serde_json::Map::new();
                        for (i, pd) in param_defs.iter().enumerate() {
                            c_params.insert(pd.name.clone(), serde_json::json!(genome.params[i]));
                        }
                        candidates.push(serde_json::json!({
                            "best_params": c_params,
                            "fitness": genome.fitness,
                        }));
                    }

                    let entry = serde_json::json!({
                        "conservative": {
                            "params": params_c,
                            "fitness": best_c.fitness,
                            "win_rate": best_c.win_rate,
                            "num_trades": best_c.num_trades,
                        },
                        "aggressive": {
                            "params": params_a,
                            "fitness": best_a.fitness,
                            "win_rate": best_a.win_rate,
                            "num_trades": best_a.num_trades,
                        },
                        "candidates": candidates
                    });

                    println!("\n   ✅ {} complete: Cons_WR={:.1}% | Aggr_WR={:.1}% | Pool={}",
                        strat_name, best_c.win_rate, best_a.win_rate, candidates.len());

                    all_best.insert(strat_name.to_string(), entry);
                }
            }

            let total_time = start.elapsed().as_secs_f64();
            println!("\n{}", "═".repeat(60));
            println!("🏆 ALL OPTIMIZATIONS COMPLETE in {:.1}s", total_time);
            println!("{}", "═".repeat(60));

            let json_res = serde_json::to_string_pretty(&all_best);
            match json_res {
                Ok(json_str) => {
                    println!("\n{}", json_str);
                    let out_path = output.unwrap_or_else(|| PathBuf::from("data/ga_best_params_all.json"));
                    if let Some(parent) = out_path.parent() {
                        std::fs::create_dir_all(parent).ok();
                    }
                    if let Err(e) = std::fs::write(&out_path, &json_str) {
                        eprintln!("❌ Failed to save results to {}: {}", out_path.display(), e);
                    } else {
                        println!("📁 Saved to {}", out_path.display());
                    }
                }
                Err(e) => {
                    eprintln!("❌ Failed to serialize GA results to JSON: {}", e);
                }
            }
        }

        Commands::Live { symbols: cli_symbols, ipc_port: _ } => {
            // Initialize tokio runtime for async live executor
            let rt = tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");
            rt.block_on(async {
                use live::{ws_feed, order_book, tape_reader, ipc_bridge, candle_aggregator, orchestrator, wall_tracker, liquidation_feed};
                use std::time::Duration;
                use tokio::sync::Mutex;

                // Load .env for API keys
                dotenv::dotenv().ok();
                let api_key = std::env::var("BINANCE_API_KEY").unwrap_or_default();
                let api_secret = std::env::var("BINANCE_API_SECRET").unwrap_or_default();
                let paper_mode = std::env::var("PAPER_MODE")
                    .map(|v| v == "true" || v == "True" || v == "1")
                    .unwrap_or(true); // Default to paper mode for safety

                println!("🔥 AEGIS Rust LiveExecutor v1.0.0");
                println!("═══════════════════════════════════");
                println!("   Mode: {}", if paper_mode { "📝 PAPER TRADING" } else { "💰 LIVE TRADING" });

                // Initialize orchestrator from active_config.json
                let config_path = PathBuf::from("data/active_config.json");
                let models_dir = PathBuf::from("data/models");

                println!("   Strategy: OB + Trade Tape + CVD Smart Trailing");
                println!("   IPC: TCP:9090");
                println!("═══════════════════════════════════\n");

                // Create shared event channel
                let (event_tx, _event_rx) = tokio::sync::broadcast::channel::<ws_feed::MarketEvent>(4096);

                // Create shared stores
                let ob_store = order_book::new_store();
                let tape_store = tape_reader::new_store(&["placeholder".to_string()], Duration::from_secs(300));
                let wall_store = wall_tracker::new_store();
                let liq_store = liquidation_feed::new_store();

                // Phase 14: Shared Live Stats
                let live_stats = crate::live::live_stats::new_shared();

                // Create orchestrator (needs stores)
                let mut orch = orchestrator::Orchestrator::new(
                    config_path.clone(),
                    models_dir,
                    api_key,
                    api_secret,
                    paper_mode,
                    wall_store.clone(),
                    tape_store.clone(),
                    ob_store.clone(),
                    liq_store.clone(),
                    live_stats.clone(),
                );

                // Get symbols from config (or use CLI override)
                let mut symbols = if cli_symbols.is_empty() {
                    orch.get_symbols()
                } else {
                    cli_symbols
                };

                // Phase 31B: Always track BTC for macro guidance
                if !symbols.contains(&"BTC/USDT".to_string()) {
                    symbols.push("BTC/USDT".to_string());
                }

                if symbols.is_empty() {
                    tracing::error!("❌ No symbols to trade. Check active_config.json or --symbols flag.");
                    return;
                }

                // Initialize orchestrator with historical data so we don't wait 3.5h for indicators (Phase 12)
                orch.preload_historical_candles().await;

                // Re-initialize tape store with actual symbols
                for sym in &symbols {
                    tape_store.insert(sym.clone(), tape_reader::TapeState::new(Duration::from_secs(300)));
                    // Same for OrderBookStore and WallStore if they need explicit keys
                    wall_store.insert(sym.clone(), wall_tracker::WallSnapshot {
                        walls: Vec::new(),
                        cascades: Vec::new(),
                        is_warming_up: true,
                        wall_threshold_usd: 50_000.0,
                    });
                }

                println!("   Symbols: {:?}", symbols);

                // Start AEGIS Telemetry Server (WebSocket for GUI)
                tokio::spawn(crate::live::telemetry::start_server(
                    wall_store.clone(),
                    tape_store.clone(),
                    symbols.clone()
                ));

                // WebSocket Feed
                let mut ws = ws_feed::WsFeed::new(symbols.clone(), event_tx.clone());

                // IPC Bridge
                let ipc = Arc::new(ipc_bridge::IpcBridge::new());
                let ipc_clone = Arc::clone(&ipc);

                // Candle Aggregator (1m → 5m, 15m) per symbol
                let aggregators: Arc<Mutex<std::collections::HashMap<String, candle_aggregator::CandleAggregator>>> = 
                    Arc::new(Mutex::new(std::collections::HashMap::new()));

                // Phase 14: Init Telegram Notifications
                let tg_token = std::env::var("TELEGRAM_BOT_TOKEN").unwrap_or_default();
                let tg_chat = std::env::var("TELEGRAM_CHAT_ID").unwrap_or_default();
                if !tg_token.is_empty() && !tg_chat.is_empty() {
                    tracing::info!("📱 Telegram notifications enabled");
                    let (tg_tx, tg_rx) = tokio::sync::mpsc::channel(100);
                    let bot = crate::live::telegram_bot::TelegramBot::new(
                        tg_token, tg_chat, tg_rx, live_stats.clone()
                    );
                    tokio::spawn(bot.run());
                    orch.tg_tx = Some(tg_tx);
                } else {
                    tracing::warn!("📱 Telegram notifications disabled (missing token or chat_id)");
                }

                // Orchestrator shared reference
                let orch = Arc::new(Mutex::new(orch));

                // Hot-swap watcher (checks for retrain_flag.txt every 60s)
                let config_path_clone = config_path.clone();
                let orch_hotswap = Arc::clone(&orch);
                let hotswap_handle = tokio::spawn(async move {
                    let flag_path = PathBuf::from("data/models/retrain_flag.txt");
                    let mut last_mtime = std::time::SystemTime::UNIX_EPOCH;
                    
                    loop {
                        tokio::time::sleep(Duration::from_secs(60)).await;
                        if let Ok(meta) = std::fs::metadata(&flag_path) {
                            if let Ok(mtime) = meta.modified() {
                                if mtime > last_mtime {
                                    last_mtime = mtime;
                                    tracing::info!("🔄 Hot-swap flag detected! Reloading configs...");
                                    let mut orch = orch_hotswap.lock().await;
                                    orch.reload_configs(&config_path_clone);
                                }
                            }
                        }
                    }
                });

                // Spawn tasks
                let ws_handle = tokio::spawn(async move {
                    ws.run().await;
                });

                let ipc_handle = tokio::spawn(async move {
                    ipc_clone.run().await;
                });

                // Wall Tracker (independent density screener)
                let ob_store_walls = ob_store.clone();
                let wall_store_task = wall_store.clone();
                let wall_tracker_handle = tokio::spawn(async move {
                    wall_tracker::run(ob_store_walls, wall_store_task).await;
                });

                // Depth Scanner (WebSocket for ALL 100 symbols @depth@100ms)
                let ob_store_depth = ob_store.clone();
                let depth_scanner_handle = tokio::spawn(async move {
                    wall_tracker::run_depth_scanner(ob_store_depth).await;
                });

                // Liquidation Feed (Phase 13: @forceOrder stream)
                let liq_store_task = liq_store.clone();
                let liq_feed_handle = tokio::spawn(async move {
                    liquidation_feed::run_liquidation_feed(liq_store_task).await;
                });

                // Phase 29C: True Tick Polling (Adaptive 250ms)
                let orch_macro = Arc::clone(&orch);
                let ob_store_macro = ob_store.clone();
                let macro_polling_handle = tokio::spawn(async move {
                    let mut interval = tokio::time::interval(tokio::time::Duration::from_millis(250));
                    loop {
                        interval.tick().await;
                        let mut o = orch_macro.lock().await;
                        o.check_macro_triggers(&ob_store_macro).await;
                    }
                });

                // Main event processing loop
                let ob_store_clone = ob_store.clone();
                let tape_store_clone = tape_store.clone();
                let orch_events = Arc::clone(&orch);
                let aggregators_clone = Arc::clone(&aggregators);
                let mut market_rx = event_tx.subscribe();
                let mut hft_rx = orch.lock().await.hft_rx.take().expect("HFT RX already taken");

                let event_loop = tokio::spawn(async move {
                    loop {
                        tokio::select! {
                            msg = market_rx.recv() => {
                                match msg {
                            Ok(ws_feed::MarketEvent::Trade(trade)) => {
                                // Record trade in tape
                                tape_reader::record_trade(
                                    &tape_store_clone,
                                    &trade.symbol,
                                    trade.price,
                                    trade.quantity,
                                    trade.is_buyer_maker,
                                    trade.timestamp,
                                );

                                // Feed trade to Whale Detector (Phase 12)
                                {
                                    let mut orch = orch_events.lock().await;
                                    orch.record_whale_trade(
                                        &trade.symbol,
                                        trade.price,
                                        trade.quantity,
                                        trade.is_buyer_maker,
                                    );
                                }

                                // Feed tick to orchestrator for position monitoring
                                let mut orch = orch_events.lock().await;
                                orch.on_tick(
                                    &trade.symbol,
                                    trade.price,
                                    &ob_store_clone,
                                    &tape_store_clone,
                                ).await;
                            }
                            Ok(ws_feed::MarketEvent::KlineClose { symbol, candle }) => {
                                tracing::debug!("[{}] 🕯 1m Candle closed @ {:.4}", symbol, candle.close);

                                // 1. Aggregate to 5m/15m and feed those to orchestrator
                                let (completed_5m, completed_15m) = {
                                    let mut aggs = aggregators_clone.lock().await;
                                    let agg = aggs.entry(symbol.clone())
                                        .or_insert_with(candle_aggregator::CandleAggregator::new);
                                    agg.process_1m_close(&candle)
                                };

                                if let Some(candle_5m) = completed_5m {
                                    tracing::debug!("[{}] 📊 5m Candle closed @ {:.4}", symbol, candle_5m.close);
                                    let mut orch = orch_events.lock().await;
                                    orch.on_candle_close(&symbol, &candle_5m).await;
                                }

                                if let Some(candle_15m) = completed_15m {
                                    tracing::debug!("[{}] 📈 15m Candle closed @ {:.4}", symbol, candle_15m.close);
                                    let mut orch = orch_events.lock().await;
                                    orch.on_15m_candle_close(&symbol, &candle_15m);
                                }
                            }
                            Ok(ws_feed::MarketEvent::FundingRateUpdate { symbol, rate }) => {
                                let mut orch = orch_events.lock().await;
                                orch.on_funding_rate_update(&symbol, rate);
                            }
                            Ok(_) => {} // KlineTick — handled passively
                            Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                                tracing::warn!("Event channel lagged by {}", n);
                            }
                            Err(_) => break,
                                }
                            }
                            Some(hft_event) = hft_rx.recv() => {
                                let mut o = orch_events.lock().await;
                                o.execute_hft_entry(hft_event).await;
                            }
                        }
                    }
                });

                // Wait for all tasks (runs forever until Ctrl+C)
                tokio::select! {
                    _ = ws_handle => tracing::error!("WS feed exited!"),
                    _ = ipc_handle => tracing::error!("IPC bridge exited!"),
                    _ = event_loop => tracing::error!("Event loop exited!"),
                    _ = hotswap_handle => tracing::error!("Hotswap watcher exited!"),
                    _ = wall_tracker_handle => tracing::error!("WallTracker exited!"),
                    _ = depth_scanner_handle => tracing::error!("DepthScanner exited!"),
                    _ = liq_feed_handle => tracing::error!("LiqFeed exited!"),
                    _ = macro_polling_handle => tracing::error!("Macro polling loop exited!"),
                    _ = tokio::signal::ctrl_c() => {
                        tracing::info!("🛑 Shutdown requested. Cleaning up...");
                    }
                }
            });
        }

        Commands::MineLevels { csv, tolerance } => {
            println!("🔍 Mining Levels for: {:?}", csv);
            let candles = backtest::load_csv(&csv);
            let levels = backtest::find_historical_levels(&candles, tolerance);
            
            println!("   Found {} significant levels", levels.len());
            
            let symbol = csv.file_stem().unwrap().to_str().unwrap();
            let output_path = format!("data/levels_{}.json", symbol);
            let json_out = serde_json::to_string_pretty(&levels).unwrap();
            
            std::fs::create_dir_all("data").ok();
            std::fs::write(&output_path, &json_out).expect("Failed to write levels JSON");
            println!("📁 Levels saved to {}", output_path);
        }

        Commands::Playback { event_id, threshold } => {
            println!("🎥 HFT Playback: Event ID {}", event_id);
            // We'll implement the actual logic in live/playback.rs
            live::playback::run_playback(&event_id, threshold);
        }

        Commands::BenchmarkHft => {
            live::synthetic_tests::run_synthetic_benchmark();
        }
    }
}

