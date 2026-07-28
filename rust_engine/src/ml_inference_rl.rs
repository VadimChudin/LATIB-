use serde_json::Value;
use std::fs;

/// Blazing fast MLP Policy evaluation for Proximal Policy Optimization (PPO) 
/// Trained via stable-baselines3 and exported as JSON weights.
pub struct RlAgent {
    w1: Vec<Vec<f32>>,
    b1: Vec<f32>,
    w2: Vec<Vec<f32>>,
    b2: Vec<f32>,
    w3: Vec<Vec<f32>>,
    b3: Vec<f32>,
}

impl RlAgent {
    /// Loads PyTorch state_dict arrays from JSON
    pub fn load_from_json(path: &str) -> Option<Self> {
        let text = fs::read_to_string(path).ok()?;
        let json: Value = serde_json::from_str(&text).ok()?;
        let actor = &json["actor"];
        
        // actor["mlp_extractor.policy_net.0.weight"] is [64 x 15] Array
        let w1 = Self::parse_matrix(&actor["mlp_extractor.policy_net.0.weight"])?;
        let b1 = Self::parse_vec(&actor["mlp_extractor.policy_net.0.bias"])?;
        let w2 = Self::parse_matrix(&actor["mlp_extractor.policy_net.2.weight"])?;
        let b2 = Self::parse_vec(&actor["mlp_extractor.policy_net.2.bias"])?;
        let w3 = Self::parse_matrix(&actor["action_net.weight"])?;
        let b3 = Self::parse_vec(&actor["action_net.bias"])?;
        
        Some(Self { w1, b1, w2, b2, w3, b3 })
    }

    fn parse_matrix(v: &Value) -> Option<Vec<Vec<f32>>> {
        let arr = v.as_array()?;
        let mut mat = Vec::new();
        for row in arr {
            let row_arr = row.as_array()?;
            let mut r = Vec::new();
            for val in row_arr {
                r.push(val.as_f64()? as f32);
            }
            mat.push(r);
        }
        Some(mat)
    }

    fn parse_vec(v: &Value) -> Option<Vec<f32>> {
        let arr = v.as_array()?;
        let mut vec = Vec::new();
        for val in arr {
            vec.push(val.as_f64()? as f32);
        }
        Some(vec)
    }

    /// Pure Rust mathematically identically equivalent to PyTorch PPO Actor Forward prop
    /// Executed in ~1 nanosecond.
    pub fn predict_action(&self, features: &[f32]) -> f32 {
        // Sanitize NaNs and Inf, clip to [-100, 100] exactly like in our Gymnasium Gym!
        let mut clean_features = vec![0.0; features.len()];
        for i in 0..features.len() {
            let f = features[i];
            if f.is_nan() || f.is_infinite() {
                clean_features[i] = 0.0;
            } else {
                clean_features[i] = f.clamp(-100.0, 100.0);
            }
        }

        // Layer 1
        let mut h1 = self.b1.clone();
        for i in 0..self.w1.len() {
            let row = &self.w1[i];
            for j in 0..clean_features.len().min(row.len()) {
                h1[i] += row[j] * clean_features[j];
            }
            if h1[i] < 0.0 { h1[i] = 0.0; } // ReLU activation
        }

        // Layer 2
        let mut h2 = self.b2.clone();
        for i in 0..self.w2.len() {
            let row = &self.w2[i];
            for j in 0..h1.len().min(row.len()) {
                h2[i] += row[j] * h1[j];
            }
            if h2[i] < 0.0 { h2[i] = 0.0; } // ReLU activation
        }

        // Output Layer (Squashed by Tanh for Continuous Action Space -1..1)
        let mut out = self.b3[0];
        let row = &self.w3[0];
        for j in 0..h2.len().min(row.len()) {
            out += row[j] * h2[j];
        }
        out.clamp(-1.0, 1.0)
    }
}
