use std::fs::File;
use std::io::{BufRead, BufReader};
use super::hft_logger::HftSnapshot;

/// Playback recorded HFT events from data/hft_snapshots.jsonl
pub fn run_playback(target_event_id: &str, fire_threshold: i32) {
    let path = "data/hft_snapshots.jsonl";
    let file = match File::open(path) {
        Ok(f) => f,
        Err(_) => {
            eprintln!("❌ Could not open {}", path);
            return;
        }
    };

    let reader = BufReader::new(file);
    let mut snapshots: Vec<HftSnapshot> = Vec::new();
    let mut found = false;

    // 1. Gather all snapshots for this event_id
    for line in reader.lines() {
        if let Ok(l) = line {
            if let Ok(snap) = serde_json::from_str::<HftSnapshot>(&l) {
                if snap.event_id == target_event_id {
                    snapshots.push(snap);
                    found = true;
                }
            }
        }
    }

    if !found {
        println!("❌ Event ID {} not found in logs", target_event_id);
        return;
    }

    println!("✅ Loaded {} snapshots for event {}", snapshots.len(), target_event_id);
    println!("⏱️ Replaying logic with threshold={}...", fire_threshold);

    // 2. Replay strategy scoring (Calibration Mode)
    let mut fired_at: Option<usize> = None;
    let mut max_score = 0;

    for (i, snap) in snapshots.iter().enumerate() {
        if snap.score > max_score {
            max_score = snap.score;
        }

        if fired_at.is_none() && snap.score >= fire_threshold {
            fired_at = Some(i);
        }
    }

    // 3. Output results
    match fired_at {
        Some(idx) => {
            let s = &snapshots[idx];
            println!("\n🔥 [PLAYBACK SUCCESS]");
            println!("   Fired at tick: {}", idx);
            println!("   Price: {}", s.price);
            println!("   Final Score: {} (Peak Score: {})", s.score, max_score);
            println!("   Strategy: {}", s.strategy);
        }
        None => {
            println!("\n❄️ [PLAYBACK TIMEOUT]");
            println!("   Threshold {} never reached (Peak Score: {})", fire_threshold, max_score);
        }
    }
}
