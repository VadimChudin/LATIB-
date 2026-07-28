use std::fs;
use std::path::{Path, PathBuf};
use std::io::{BufRead, BufReader};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TickEvent {
    pub ts_ms: u64,
    pub price: f32,
    pub qty: f32,
    pub is_buyer_maker: bool,
}

#[derive(Debug, Clone)]
pub struct Epicenter {
    pub ts_ms: u64,
    pub ticks: Vec<TickEvent>,
    pub direction: String,
    /// Phase 31: true = real bounce happened, false = drop without bounce (negative example)
    pub has_bounce: bool,
}

/// Helper function to read ticks from a single CSV file
fn read_ticks_csv(file_path: &Path) -> Vec<TickEvent> {
    let mut ticks = Vec::new();
    if let Ok(file) = fs::File::open(file_path) {
        let reader = BufReader::new(file);
        let mut lines = reader.lines();
        
        // Skip header
        lines.next();

        for line in lines.filter_map(Result::ok) {
            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() >= 4 {
                if let (Ok(ts), Ok(price), Ok(qty)) = (
                    parts[0].parse::<u64>(),
                    parts[1].parse::<f32>(),
                    parts[2].parse::<f32>(),
                ) {
                    let is_bm = parts[3].to_lowercase().trim().parse::<bool>().unwrap_or(false);
                    ticks.push(TickEvent {
                        ts_ms: ts,
                        price,
                        qty,
                        is_buyer_maker: is_bm,
                    });
                }
            }
        }
    }
    // Ensure ticks are sorted by time (they usually are from Binance, but just to be safe)
    ticks.sort_by_key(|t| t.ts_ms);
    ticks
}

/// Load ticks from CSV files in the epicenter directory
pub fn load_epicenters(symbol: &str, direction: &str, limit: Option<usize>) -> Vec<Epicenter> {
    let dir_upper = direction.to_uppercase();
    let dirs_to_load = if dir_upper == "ALL" {
        vec!["LONG", "SHORT"]
    } else {
        vec![dir_upper.as_str()]
    };

    let mut epicenters = Vec::new();

    for dir in dirs_to_load {
        // Load REAL epicenters (has_bounce = true)
        load_from_dir(&format!("../data/epicenters_ticks/{}/{}", symbol, dir), dir, true, &mut epicenters);
        
        // Load FALSE epicenters (has_bounce = false) — Phase 31
        load_from_dir(&format!("../data/epicenters_ticks/{}/{}_FALSE", symbol, dir), dir, false, &mut epicenters);
    }

    epicenters.sort_by_key(|e| e.ts_ms);

    if let Some(l) = limit {
        if epicenters.len() > l {
            let skip = epicenters.len() - l;
            epicenters.drain(0..skip);
        }
    }

    let real_count = epicenters.iter().filter(|e| e.has_bounce).count();
    let false_count = epicenters.iter().filter(|e| !e.has_bounce).count();
    println!("Loaded {} epicenters for {} ({} real + {} false)", epicenters.len(), symbol, real_count, false_count);
    epicenters
}

/// Helper: load epicenter CSVs from a directory
fn load_from_dir(dir_path: &str, direction: &str, has_bounce: bool, epicenters: &mut Vec<Epicenter>) {
    let path = Path::new(dir_path);
    if !path.exists() {
        return; // silently skip if dir doesn't exist (FALSE dirs may not exist yet)
    }

    if let Ok(entries) = fs::read_dir(path) {
        let mut files: Vec<_> = entries.filter_map(|e| e.ok()).collect();
        files.sort_by_key(|a| a.file_name());

        for entry in files {
            let p = entry.path();
            if let Some(ext) = p.extension() {
                if ext == "csv" {
                    if let Some(stem) = p.file_stem() {
                        if let Ok(ts_ms) = stem.to_string_lossy().parse::<u64>() {
                            let ticks = read_ticks_csv(&p);
                            if !ticks.is_empty() {
                                epicenters.push(Epicenter {
                                    ts_ms,
                                    ticks,
                                    direction: direction.to_string(),
                                    has_bounce,
                                });
                            }
                        }
                    }
                }
            }
        }
    }
}
