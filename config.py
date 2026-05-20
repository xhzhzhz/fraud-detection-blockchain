import os
from pathlib import Path

# =============================================================
# PATH CONFIGURATION
# =============================================================
BASE_DIR = Path(__file__).resolve().parent

DATA_RAW_DIR       = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
DATA_SPLITS_DIR    = BASE_DIR / "data" / "splits"

MODELS_DIR         = BASE_DIR / "models" / "saved"
EVALUATION_DIR     = BASE_DIR / "models" / "evaluation"

REPORTS_DIR        = BASE_DIR / "reports"
FIGURES_DIR        = REPORTS_DIR / "figures"
SHAP_DIR           = REPORTS_DIR / "shap"
AUDIT_LOGS_DIR     = REPORTS_DIR / "audit_logs"

LOGS_DIR           = BASE_DIR / "logs"

for d in [DATA_RAW_DIR, DATA_PROCESSED_DIR, DATA_SPLITS_DIR,
          MODELS_DIR, EVALUATION_DIR, FIGURES_DIR,
          SHAP_DIR, AUDIT_LOGS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# =============================================================
# DATASET CONFIGURATION
# =============================================================
RAW_DATASET_FILENAME = "PS_20174392719_1491204439457_log.csv"
RAW_DATASET_PATH     = DATA_RAW_DIR / RAW_DATASET_FILENAME

# Kolom yang dieksklusi sebelum modeling 
COLUMNS_TO_DROP = [
    "isFlaggedFraud",  # Target leakage
    "nameOrig",        # High-cardinality identifier
    "nameDest",        # High-cardinality identifier
]

TARGET_COLUMN = "isFraud"

# =============================================================
# TEMPORAL SPLIT CONFIGURATION
# =============================================================
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# =============================================================
# SMOTE CONFIGURATION
# =============================================================
SMOTE_CONFIGS = {
    "baseline": None,   # Tanpa SMOTE, distribusi asli
    "smote_1":  0.01,   # 1% fraud dari total training set
    "smote_2":  0.05,   # 5% fraud dari total training set
    "smote_3":  0.20,   # 20% fraud dari total training set
}

SMOTE_K_NEIGHBORS  = 5
RANDOM_STATE       = 42 

# =============================================================
# MODEL CONFIGURATION
# =============================================================

# XGBoost: ruang parameter untuk RandomizedSearchCV
XGBOOST_PARAM_DIST = {
    "max_depth":        [3, 5, 7, 10],
    "learning_rate":    [0.01, 0.05, 0.1, 0.3],
    "subsample":        [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
}

XGBOOST_FIXED_PARAMS = {
    "objective":        "binary:logistic",
    "eval_metric":      "aucpr",
    "tree_method":      "hist", 
    "n_estimators":     500,     
    "random_state":     RANDOM_STATE,
    "n_jobs":           1,
}

XGBOOST_N_ITER          = 40   
XGBOOST_CV_FOLDS        = 5
XGBOOST_EARLY_STOPPING  = 50

# Random Forest — ruang parameter untuk GridSearchCV
RF_PARAM_GRID = {
    "n_estimators":     [100, 200],
    "max_depth":        [10, 20],
    "min_samples_split":[5, 10],
    "max_features":     ["sqrt", "log2"],
}

RF_FIXED_PARAMS = {
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "max_samples": 0.3,
    "n_jobs":       1,
}

RF_CV_FOLDS   = 5

# =============================================================
# EVALUATION THRESHOLDS 
# =============================================================
TARGET_RECALL    = 0.85
TARGET_PRECISION = 0.70
TARGET_F1        = 0.80
TARGET_ROC_AUC   = 0.90
TARGET_PR_AUC    = 0.75

# =============================================================
# SHAP CONFIGURATION
# =============================================================
SHAP_TOP_K_FEATURES = 5   # Untuk local explanation
SHAP_TOP_K_AUDIT    = 3   # Untuk shapHash di blockchain

# Target konsistensi SHAP 
SHAP_SPEARMAN_TARGET = 0.80
SHAP_CV_TARGET       = 0.30
SHAP_JACCARD_TARGET  = 0.60

# =============================================================
# BLOCKCHAIN CONFIGURATION
# =============================================================
BLOCKCHAIN_NETWORK      = "sepolia"
HIGH_VALUE_THRESHOLD    = 10_000_000   # Rp 10 juta (Kriteria 2 selective audit)
FRAUD_SCORE_THRESHOLD   = None         
UNCERTAINTY_THRESHOLD   = 0.15        # Kriteria 3 selective audit

# =============================================================
# API CONFIGURATION
# =============================================================
API_HOST            = "0.0.0.0"
API_PORT            = 8000
ML_TIMEOUT_SECONDS  = 2.0

# =============================================================
# AUDIT LOG CONFIGURATION
# =============================================================
BLOCKCHAIN_RETRY_INTERVAL_MINUTES = 5
BLOCKCHAIN_MAX_RETRIES            = 5