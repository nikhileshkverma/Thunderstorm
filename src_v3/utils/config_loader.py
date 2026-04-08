"""
utils/config_loader.py — Load config.yaml and provide typed access.
All pipeline scripts import from here.
"""
import yaml
from pathlib import Path
from typing import Any

_CONFIG = None

def load_config(config_path: str = None) -> dict:
    """Load config.yaml and return as dict. Cached after first load."""
    global _CONFIG
    if _CONFIG is not None:
        return _CONFIG

    if config_path is None:
        # Auto-find: look in same dir as this file's parent (src/)
        config_path = Path(__file__).parent.parent / "config.yaml"

    with open(config_path, "r") as f:
        _CONFIG = yaml.safe_load(f)

    # Convert path strings to Path objects
    for key, val in _CONFIG["paths"].items():
        _CONFIG["paths"][key] = Path(val)

    # Create output directories
    for key in ["features_dir", "labels_dir", "dataset_dir",
                "models_dir", "results_dir", "logs_dir"]:
        _CONFIG["paths"][key].mkdir(parents=True, exist_ok=True)
    (_CONFIG["paths"]["results_dir"] / "plots").mkdir(parents=True, exist_ok=True)

    return _CONFIG


def get_active_regions(cfg: dict) -> list:
    """Return only active regions from config."""
    return [r for r in cfg["regions"] if r.get("active", True)]


def get_plot_context(cfg: dict, model_name: str = "", split: str = "",
                     n_train: int = 0, n_val: int = 0, n_test: int = 0,
                     extra: str = "") -> str:
    """
    Build a standardised plot subtitle with full context.
    Professor Tissot: every plot must show model, hyperparams, what is predicted,
    date range, region, FH, and sample counts.
    """
    p = cfg["plots"]
    m = cfg["models"].get(model_name.lower().replace(" ","_"), {})

    lines = [
        f"Target: {p['prediction_target']}",
        f"Region: {p['region_label']}  |  Period: {p['date_range']}",
        f"FH: {cfg['hrrr']['forecast_hours']}  |  Cycles: {cfg['hrrr']['cycles']}",
    ]
    if model_name:
        hp_str = ""
        if "random_forest" in model_name.lower():
            hp_str = (f"n_est={m.get('n_estimators','?')}  "
                      f"max_depth={m.get('max_depth','None')}  "
                      f"min_leaf={m.get('min_samples_leaf','?')}  "
                      f"class_weight={m.get('class_weight','?')}")
        elif "xgb" in model_name.lower():
            hp_str = (f"n_est={m.get('n_estimators','?')}  "
                      f"depth={m.get('max_depth','?')}  "
                      f"lr={m.get('learning_rate','?')}  "
                      f"subsample={m.get('subsample','?')}")
        lines.append(f"Model: {model_name}  |  {hp_str}")

    if split:
        counts = []
        if n_train: counts.append(f"train n={n_train}")
        if n_val:   counts.append(f"val n={n_val}")
        if n_test:  counts.append(f"test n={n_test}")
        if counts:
            lines.append(f"Split: {' | '.join(counts)}")

    if extra:
        lines.append(extra)

    return "\n".join(lines)
