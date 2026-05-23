import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import joblib
import pandas as pd
from config import MODELS_DIR, EVALUATION_DIR, DATA_SPLITS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_final_pipeline():
    # Membaca konfigurasi optimal dari Fase 3
    with open(EVALUATION_DIR / "best_config_final.json") as f:
        best_final = json.load(f)

    # Fase 3 memilih baseline: load langsung dari nama konfigurasi
    # bukan dari best_pipeline.joblib (yang berisi smote_1)
    optimal_config = best_final["shap_analysis"]["optimal_config_shap"]
    model_prefix   = "xgb"   
    logger.info(f"Konfigurasi optimal dari Fase 3: {optimal_config}")

    # Load feature columns dari split_info
    with open(DATA_SPLITS_DIR / "split_info.json") as f:
        split_info = json.load(f)
    feature_cols = split_info["feature_columns"]

    # Load dual thresholds
    with open(EVALUATION_DIR / "dual_thresholds.json") as f:
        dual = json.load(f)
    threshold_key = f"{optimal_config}"
    theta_low     = dual[threshold_key]["theta_low"]
    theta_high    = dual[threshold_key]["theta_high"]

    # Load all_metrics untuk test metrics
    df_metrics   = pd.read_csv(EVALUATION_DIR / "all_metrics.csv")
    metrics_row  = df_metrics[
        (df_metrics["config"] == optimal_config) &
        (df_metrics["model"] == "XGBoost") &
        (df_metrics["split"] == "test")
    ].iloc[0]

    # Load model
    model_path = MODELS_DIR / f"{model_prefix}_{optimal_config}.joblib"
    model      = joblib.load(model_path)

    # Membangun pipeline final
    final_pipeline = {
        "model":              model,
        "config_name":        optimal_config,
        "model_name":         "XGBoost",
        "feature_columns":    feature_cols,
        "theta_low":          theta_low,
        "theta_high":         theta_high,
        "theta_optimal":      theta_high,  # Untuk uncertainty
        "model_version":      f"xgboost-{optimal_config}-v1.0",
        "test_metrics":       metrics_row.to_dict(),
        "selection_basis":    "Fase 3: Konsistensi SHAP tertinggi (Spearman=1.0, Jaccard=1.0)",
        "shap_analysis":      best_final["shap_analysis"]
    }

    save_path = MODELS_DIR / "best_pipeline_final.joblib"
    joblib.dump(final_pipeline, save_path)

    logger.info(f"best_pipeline_final.joblib disimpan: {save_path}")
    logger.info(f"  Config     : {optimal_config} (XGBoost)")
    logger.info(f"  theta_low  : {theta_low:.4f}")
    logger.info(f"  theta_high : {theta_high:.4f}")
    logger.info(f"  PR-AUC     : {metrics_row['pr_auc']}")
    logger.info(f"  Recall     : {metrics_row['recall']}")
    logger.info(f"  Model file : {model_path.name}")

    return final_pipeline


if __name__ == "__main__":
    build_final_pipeline()
    print("best_pipeline_final.joblib berhasil dibuat.")