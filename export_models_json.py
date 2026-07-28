#!/usr/bin/env python3
"""
export_models_json.py — Export Python ML models (.joblib) to Rust-compatible JSON.

Rust LgbmModel expects:
{
  "num_trees": N,
  "num_features": 23,
  "feature_names": [...],
  "trees": [
    { "split_feature": 0, "threshold": 0.5, "decision_type": "<=",
      "left_child": { ... or "leaf_value": 0.1 },
      "right_child": { ... or "leaf_value": -0.2 }
    },
    ...
  ]
}
"""
import os
import json
import joblib
import numpy as np

MODELS_DIR = "data/models"

# Feature names matching Rust ml_inference.rs extract_features() order
FEATURE_NAMES = [
    "hurst", "atr_pct", "vol_ratio", "hour", "dist_to_ema",
    "rsi", "adx", "macd_hist", "bb_pos", "stoch",
    "cci", "mfi", "rsi_change_5", "vol_change_5", "atr_change_5",
    "macd_slope", "close_pos", "ema_slope", "bb_width", "stoch_dist_50",
    "htf_trend", "h1_direction", "funding_rate",
]

META_FEATURE_NAMES = [
    "strategy_enc", "spot_probe_enc", "wall_size_usd", "wall_age_h",
    "wall_eaten_pct", "cvd_delta", "imbalance_ratio", "tape_speed",
    "entry_price", "risk_dist"
]

# Model files to export (Rust model_name -> joblib filename)
MODEL_MAP = {
    "ultimate_smc_trail": "ultimate_smc_trail_model.joblib",
    "knife_catcher": "knife_catcher_model.joblib",
    "scalpmtf_model": "scalpmtf_model.joblib",
    "funding_rate_model": "funding_rate_model.joblib",
    "density_model": "density_model.joblib",
    "meta_model": "meta_model.joblib",
}


def lgbm_tree_to_json(tree_dict):
    """Convert a LightGBM tree dict (from model.dump_model()) to our JSON format."""
    def parse_node(node):
        if "leaf_value" in node:
            return {"leaf_value": float(node["leaf_value"])}
        
        result = {
            "split_feature": int(node.get("split_feature", 0)),
            "threshold": float(node.get("threshold", 0.0)),
            "decision_type": node.get("decision_type", "<="),
        }
        
        if "left_child" in node:
            result["left_child"] = parse_node(node["left_child"])
        if "right_child" in node:
            result["right_child"] = parse_node(node["right_child"])
        
        return result
    
    return parse_node(tree_dict["tree_structure"])


def xgb_tree_to_json(tree_str, num_features):
    """Convert an XGBoost text dump tree to our JSON format."""
    lines = tree_str.strip().split("\n")
    nodes = {}
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Parse node ID
        parts = line.split(":")
        node_id = int(parts[0])
        rest = parts[1].strip() if len(parts) > 1 else ""
        
        if "leaf=" in rest:
            val = float(rest.split("leaf=")[1].split(",")[0])
            nodes[node_id] = {"leaf_value": val}
        elif "[f" in rest:
            # [f5<0.5] yes=1,no=2
            feat_str = rest.split("[f")[1].split("<")[0].split(">")[0].split("<=")[0]
            feat_idx = int(feat_str)
            
            if "<=" in rest.split("[")[1].split("]")[0]:
                thresh = float(rest.split("<=")[1].split("]")[0])
                decision = "<="
            elif "<" in rest.split("[")[1].split("]")[0]:
                thresh = float(rest.split("<")[1].split("]")[0])
                decision = "<"
            else:
                thresh = 0.0
                decision = "<="
            
            yes_id = int(rest.split("yes=")[1].split(",")[0])
            no_id = int(rest.split("no=")[1].split(",")[0])
            
            nodes[node_id] = {
                "split_feature": feat_idx,
                "threshold": thresh,
                "decision_type": decision,
                "_yes": yes_id,
                "_no": no_id,
            }
    
    def build_tree(node_id):
        node = nodes[node_id]
        if "leaf_value" in node:
            return {"leaf_value": node["leaf_value"]}
        
        result = {
            "split_feature": node["split_feature"],
            "threshold": node["threshold"],
            "decision_type": node["decision_type"],
            "left_child": build_tree(node["_yes"]),
            "right_child": build_tree(node["_no"]),
        }
        return result
    
    return build_tree(0)


def sklearn_tree_to_json(tree, feature_idx=0):
    """Convert sklearn DecisionTree to our JSON format."""
    tree_ = tree.tree_
    
    def recurse(node_id):
        if tree_.children_left[node_id] == -1:  # Leaf
            val = float(tree_.value[node_id].flatten()[0])
            return {"leaf_value": val}
        
        return {
            "split_feature": int(tree_.feature[node_id]),
            "threshold": float(tree_.threshold[node_id]),
            "decision_type": "<=",
            "left_child": recurse(int(tree_.children_left[node_id])),
            "right_child": recurse(int(tree_.children_right[node_id])),
        }
    
    return recurse(0)


def export_model(model, rust_name, num_features=23, custom_features=None):
    """Export a model to Rust-compatible JSON."""
    model_type = type(model).__name__
    print(f"  Model type: {model_type}")
    
    trees_json = []
    
    if model_type == "LGBMClassifier":
        booster = model.booster_
        model_dump = booster.dump_model()
        for tree_info in model_dump["tree_info"]:
            trees_json.append(lgbm_tree_to_json(tree_info))
        num_features = model_dump.get("max_feature_idx", num_features - 1) + 1
    
    elif model_type == "XGBClassifier":
        booster = model.get_booster()
        tree_dump = booster.get_dump()
        for tree_str in tree_dump:
            trees_json.append(xgb_tree_to_json(tree_str, num_features))
        num_features = booster.num_features()
    
    elif model_type in ("GradientBoostingClassifier", "RandomForestClassifier"):
        for estimator in getattr(model, 'estimators_', []):
            try:
                for tree in iter(estimator): # type: ignore
                    trees_json.append(sklearn_tree_to_json(tree))
            except TypeError:
                trees_json.append(sklearn_tree_to_json(estimator))
    
    else:
        print(f"  ⚠️ Unknown model type: {model_type} — trying sklearn-style export")
        for est in getattr(model, 'estimators_', []):
            try:
                for sub in iter(est): # type: ignore
                    if hasattr(sub, 'tree_'):
                        trees_json.append(sklearn_tree_to_json(sub))
            except TypeError:
                if hasattr(est, 'tree_'):
                    trees_json.append(sklearn_tree_to_json(est))
    
    if not trees_json:
        print(f"  ❌ Could not extract trees from {model_type}")
        return False
    
    feature_list = custom_features if custom_features else FEATURE_NAMES
    
    result = {
        "num_trees": len(trees_json),
        "num_features": int(num_features),
        "feature_names": feature_list[:int(num_features)],
        "trees": trees_json,
    }
    
    out_path = os.path.join(MODELS_DIR, f"{rust_name}.json")
    with open(out_path, "w") as f:
        json.dump(result, f)
    
    size_kb = os.path.getsize(out_path) / 1024
    print(f"  ✅ Exported {len(trees_json)} trees → {out_path} ({size_kb:.1f} KB)")
    return True


def main():
    print("=" * 60)
    print("  ML Model Exporter: VotingClassifier (.joblib) → Rust JSON Ensemble")
    print("=" * 60)
    
    exported = 0
    failed = 0
    
    for rust_name, joblib_file in MODEL_MAP.items():
        joblib_path = os.path.join(MODELS_DIR, joblib_file)
        print(f"\n🔄 {rust_name} ({joblib_file})")
        
        if not os.path.exists(joblib_path):
            print(f"  ❌ File not found: {joblib_path}")
            failed += 1
            continue
        
        try:
            model = joblib.load(joblib_path)
            
            # Phase 10: Handle VotingClassifier (Ensemble of 3 models)
            if type(model).__name__ == "VotingClassifier":
                print("  📦 Detected VotingClassifier Ensemble. Extracting sub-models...")
                
                # Default assume the order from ml_filter.py: [('rf', clf1), ('xgb', clf2), ('lgb', clf3)]
                # If the model is fitted, the estimators are in model.estimators_
                estimators = getattr(model, 'estimators_', None)
                if not estimators:
                    print("  ❌ VotingClassifier is not fitted. Cannot export.")
                    failed += 1
                    continue
                
                rf_model, xgb_model, lgb_model = None, None, None
                for est in estimators:
                    name = type(est).__name__
                    if name == "RandomForestClassifier": rf_model = est
                    elif name == "XGBClassifier": xgb_model = est
                    elif name == "LGBMClassifier": lgb_model = est
                
                if lgb_model and export_model(lgb_model, rust_name):
                    exported += 1
                else:
                    print("  ❌ Failed to export Base LGBM model")
                    failed += 1
                    
                if xgb_model and export_model(xgb_model, f"xgb_{rust_name}"):
                    exported += 1
                else:
                    print("  ⚠️ Failed/Skipped XGB export")
                    
                if rf_model and export_model(rf_model, f"rf_{rust_name}"):
                    exported += 1
                else:
                    print("  ⚠️ Failed/Skipped RF export")
                    
            else:
                # Fallback to single model export
                if rust_name == "meta_model":
                    export_model(model, rust_name, num_features=len(META_FEATURE_NAMES), custom_features=META_FEATURE_NAMES)
                    exported += 1
                elif export_model(model, rust_name):
                    exported += 1
                else:
                    failed += 1
        except Exception as e:
            print(f"  ❌ Error: {e}")
            failed += 1
    
    print(f"\n{'=' * 60}")
    print(f"  Done: {exported} exported sub-models, {failed} failed")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
