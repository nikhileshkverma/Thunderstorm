"""
================================================================================
config.py — Central configuration for the Deep Learning Thunderstorm Pipeline
================================================================================
Project : Deep Learning Thunderstorm Model for the CONUS
Authors : Verma, Collins, Kamangir, King, Tissot  |  3 December 2025

Edit ONLY this file to change paths, dates, regions, or model parameters.
All other pipeline scripts import from here.
================================================================================
"""

from pathlib import Path
from datetime import date

# =============================================================================
#  SERVER PATHS
# =============================================================================
HRRR_BASE     = Path("/Storage03/nverma1/HRRR")        # root; year appended below
GLM_BASE      = Path("/Storage03/nverma1/GOES16_GLM")        # root; year/jday/hour appended
WGRIB2        = Path("/home/nverma1/bin/wgrib2")

PROJECT_ROOT  = Path("/Storage03/nverma1/lightning_project")
SRC_DIR       = PROJECT_ROOT / "src"

# Output directories (created automatically by pipeline)
DATA_DIR      = PROJECT_ROOT / "data"
FEATURES_DIR  = DATA_DIR / "features"      # raw extracted+derived CSVs per date
LABELS_DIR    = DATA_DIR / "labels"        # lightning label CSVs per date
DATASET_DIR   = DATA_DIR / "dataset"       # merged dataset ready for ML
MODELS_DIR    = PROJECT_ROOT / "models"    # saved model files
RESULTS_DIR   = PROJECT_ROOT / "results"   # metrics, plots
LOGS_DIR      = PROJECT_ROOT / "logs"      # run logs

for d in [FEATURES_DIR, LABELS_DIR, DATASET_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# =============================================================================
#  HRRR EXTRACTION CONFIG
# =============================================================================
HRRR_CYCLES        = ["00", "06", "12", "18"]   # all 4 daily cycles
HRRR_FORECAST_HOURS = [6, 12, 18, 24, 30, 36]   # per Waylon spec

# Physical constants (do not change)
R_DRY  = 287.0    # J kg-1 K-1
G      = 9.8      # m s-2
CPD    = 1005.7   # J kg-1 K-1
KAPPA  = 0.4      # von Kármán

# =============================================================================
#  STUDY REGION(S)
#  Each region is a dict: name, center_lat, center_lon, box_km
#  box_km = 24 (the 24km×24km box per Waylon PDF)
#  Add more regions once Waylon confirms the 3 official regions.
# =============================================================================
REGIONS = [
    {
        "name":       "SouthTexas_CorpusChristi",
        "lat":        28.51052,
        "lon":        -97.5052,
        "box_km":     24,           # ±12 km from center in each direction
    },
    # TODO: Add 2 more regions once Waylon confirms
    # {"name": "Region2", "lat": ..., "lon": ..., "box_km": 24},
    # {"name": "Region3", "lat": ..., "lon": ..., "box_km": 24},
]

# =============================================================================
#  LIGHTNING LABELING CONFIG
# =============================================================================
LIGHTNING_WINDOW_HOURS = 2    # ±2 hours around valid time (per PDF page 15)

# Half-width of box in degrees (approximate: 24km / 111km/deg ≈ 0.108°)
BOX_HALF_DEG = 24.0 / 2.0 / 111.0   # ~0.108 degrees

# Minimum flash count to label as lightning=1
LIGHTNING_FLASH_THRESHOLD = 1   # even 1 flash in box/window → label=1

# =============================================================================
#  DATE RANGE FOR DATASET CONSTRUCTION
# =============================================================================
# Phase 1: January 2024 (proof of concept, ~744 rows per region)
DATASET_START = date(2024, 1, 1)
DATASET_END   = date(2024, 1, 31)

# Phase 2 (future): Full year or multi-year
# DATASET_START = date(2019, 1, 1)
# DATASET_END   = date(2024, 12, 31)

# =============================================================================
#  ML MODEL CONFIG
# =============================================================================
TEST_SIZE       = 0.15     # 15% held out as test set (chronological split)
VAL_SIZE        = 0.15     # 15% validation from remaining training data
RANDOM_STATE    = 42

# Random Forest hyperparameters
RF_PARAMS = {
    "n_estimators":  300,
    "max_depth":     None,    # grow full trees
    "min_samples_leaf": 5,
    "n_jobs":        -1,
    "random_state":  RANDOM_STATE,
    "class_weight":  "balanced",   # important: class imbalance (few storms)
}

# XGBoost hyperparameters
XGB_PARAMS = {
    "n_estimators":      500,
    "max_depth":         6,
    "learning_rate":     0.05,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "use_label_encoder": False,
    "eval_metric":       "logloss",
    "random_state":      RANDOM_STATE,
    "n_jobs":            -1,
    "scale_pos_weight":  None,   # set dynamically to n_neg/n_pos
}

# =============================================================================
#  COLUMNS TO DROP BEFORE MODEL TRAINING
#  (metadata / not input features per Waylon PDF)
# =============================================================================
NON_FEATURE_COLS = [
    "cycle_time", "forecast_hour", "latitude", "longitude",
    "valid_time", "label",
    # MAXUWU/MAXVWV not in wrfnat files — exclude until Waylon confirms
    "MAXUWU", "MAXVWV",
    # FRICV, SHTFL, LHTFL, SFCR, SPFH_2m, SPFH_1..50 are extracted but NOT
    # used as input features per PDF page 5 ("EXTRACTED FEATURES NOT USED AS INPUT")
    # They ARE used to compute derived features (BowenRatio, ZL, EPT, etc.)
    # so they stay in the CSV but are removed before model fitting.
    "FRICV", "SHTFL", "LHTFL", "SFCR", "SPFH_2m",
] + [f"SPFH_{i}" for i in range(1, 51)]

# =============================================================================
#  EVALUATION METRICS THRESHOLDS
# =============================================================================
DECISION_THRESHOLD = 0.5   # probability cutoff for binary prediction