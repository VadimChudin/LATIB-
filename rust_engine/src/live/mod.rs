/// AEGIS Live Trading Module
/// =========================
/// Real-time WebSocket feed, order management, and smart trailing.

pub mod ws_feed;
pub mod order_book;
pub mod tape_reader;
pub mod order_router;
pub mod smart_trailer;
pub mod position_manager;
pub mod ipc_bridge;
pub mod candle_aggregator;
pub mod config_loader;
pub mod orchestrator;
pub mod wall_tracker;
pub mod telemetry;
pub mod absorber;
pub mod level_tracker;
pub mod scalp_monitor;
pub mod spot_probe;
pub mod telegram_bot; // Phase 14: Native Rust telegram bot_logger;
pub mod live_stats;
pub mod trade_logger;
pub mod journal_analyzer;
pub mod hft_logger;
pub mod playback;
pub mod synthetic_tests;
pub mod liquidation_feed;
pub mod market_session;
pub mod whale_detector;
pub mod decision_matrix; // Phase 29C+2: Tag-based trade management
pub mod density_radar;   // Phase 30.5: Density breakout S/R proximity monitor
