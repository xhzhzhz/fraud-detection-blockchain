import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

from config import (
    DATA_SPLITS_DIR, MODELS_DIR, EVALUATION_DIR, FIGURES_DIR,
    TARGET_COLUMN,
    XGBOOST_PARAM_DIST, XGBOOST_FIXED_PARAMS,
    XGBOOST_N_ITER, XGBOOST_CV_FOLDS, XGBOOST_EARLY_STOPPING,
    RF_PARAM_GRID, RF_FIXED_PARAMS, RF_CV_FOLDS,
    SMOTE_CONFIGS, RANDOM_STATE
)
from src.modeling.smote_handler import apply_smote, get_scale_pos_weight
from src.modeling.evaluator import (
    find_optimal_threshold, compute_metrics,
    plot_confusion_matrix, plot_pr_roc_curves,
    plot_metrics_comparison
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# =============================================================
# LOAD DATA
# =============================================================

def load_splits() -> tuple:
    """

    Test set dimuat tapi tidak disentuh hingga evaluasi final.
    """
    logger.info("Memuat split data...")

    train_df = pd.read_csv(DATA_SPLITS_DIR / "train.csv")
    val_df   = pd.read_csv(DATA_SPLITS_DIR / "val.csv")
    test_df  = pd.read_csv(DATA_SPLITS_DIR / "test.csv")

    with open(DATA_SPLITS_DIR / "split_info.json") as f:
        split_info = json.load(f)

    # Kolom fitur: semua kecuali target dan step
    feature_cols = split_info["feature_columns"]

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COLUMN]

    X_val   = val_df[feature_cols]
    y_val   = val_df[TARGET_COLUMN]

    X_test  = test_df[feature_cols]
    y_test  = test_df[TARGET_COLUMN]

    logger.info(f"Train: {X_train.shape} | "
                f"Fraud: {y_train.sum():,} ({y_train.mean()*100:.4f}%)")
    logger.info(f"Val  : {X_val.shape} | "
                f"Fraud: {y_val.sum():,} ({y_val.mean()*100:.4f}%)")
    logger.info(f"Test : {X_test.shape} | "
                f"Fraud: {y_test.sum():,} ({y_test.mean()*100:.4f}%) [LOCKED]")

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols


# =============================================================
# TRAIN XGBOOST
# =============================================================

def train_xgboost(
    X_train_res, y_train_res,
    X_val, y_val,
    scale_pos_weight, config_name
):
    logger.info(f"[{config_name}|XGBoost] Memulai RandomizedSearchCV "
                f"(n_iter={XGBOOST_N_ITER}, cv={XGBOOST_CV_FOLDS})...")

    t_start = time.time()

    # Tahap 1: Tuning hyperparameter via CV, tanpa early stopping
    # Early stopping tidak digunakan di sini karena eval_set eksternal
    # akan bocor ke setiap fold CV, menghasilkan estimasi yang tidak valid.
    # n_estimators fixed 300 sebagai nilai tengah untuk CV.
    cv_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        n_estimators=300,          
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbosity=0
    )

    cv = StratifiedKFold(
        n_splits=XGBOOST_CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    search = RandomizedSearchCV(
        estimator=cv_model,
        param_distributions=XGBOOST_PARAM_DIST,
        n_iter=XGBOOST_N_ITER,
        scoring="average_precision",
        cv=cv,
        refit=False,               # tidak refit, refit manual di tahap 2
        n_jobs=8,
        random_state=RANDOM_STATE,
        verbose=1,
        error_score="raise"
    )

    search.fit(X_train_res, y_train_res)

    best_params   = search.best_params_
    best_cv_score = search.best_score_

    logger.info(
        f"[{config_name}|XGBoost] Selesai dalam {(time.time()-t_start)/60:.1f} menit | "
        f"Best CV PR-AUC: {best_cv_score:.4f}"
    )
    logger.info(f"[{config_name}|XGBoost] Best params: {best_params}")

    # Tahap 2: Refit dengan hyperparameter terbaik + early stopping
    # Early stopping di sini valid karena menggunakan val set secara eksplisit
    # setelah CV selesai, tidak ada kontaminasi fold.
    final_params = {**best_params}
    final_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        n_estimators=XGBOOST_FIXED_PARAMS["n_estimators"],  # 500, batas atas
        early_stopping_rounds=XGBOOST_EARLY_STOPPING,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=8,
        verbosity=0,
        **final_params
    )

    final_model.fit(
        X_train_res, y_train_res,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    logger.info(
        f"[{config_name}|XGBoost] n_estimators optimal (early stopping): "
        f"{final_model.best_iteration + 1}"
    )

    return final_model, best_params, best_cv_score


# =============================================================
# TRAIN RANDOM FOREST
# =============================================================

def train_random_forest(
    X_train_res, y_train_res,
    config_name
):
    """
    Random Forest baseline dengan exhaustive GridSearchCV.
    Ruang parameter terbatas (16 kombinasi) memungkinkan eksplorasi
    menyeluruh.
    """
    logger.info(
        f"[{config_name}|RandomForest] Memulai GridSearchCV "
        f"(exhaustive 16 kombinasi, cv={RF_CV_FOLDS})..."
    )

    t_start = time.time()

    base_model = RandomForestClassifier(**RF_FIXED_PARAMS)

    cv = StratifiedKFold(
        n_splits=RF_CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    search = GridSearchCV(
        estimator=base_model,
        param_grid=RF_PARAM_GRID,   
        scoring="average_precision",
        cv=cv,
        refit=True,
        n_jobs=8,
        verbose=1,
        error_score="raise"
    )

    search.fit(X_train_res, y_train_res)

    best_model    = search.best_estimator_
    best_params   = search.best_params_
    best_cv_score = search.best_score_

    elapsed = time.time() - t_start
    logger.info(
        f"[{config_name}|RandomForest] Selesai dalam {elapsed/60:.1f} menit | "
        f"Best CV PR-AUC: {best_cv_score:.4f}"
    )
    logger.info(f"[{config_name}|RandomForest] Best params: {best_params}")

    return best_model, best_params, best_cv_score

# =============================================================
# PLOT FEATURE IMPORTANCE
# Visualisasi feature importance XGBoost sebagai sanity check
# sebelum analisis SHAP untuk memastikan feature engineering
# berjalan dengan benar.
# =============================================================

def plot_feature_importance(
    model,
    feature_names: list,
    config_name: str,
    model_name: str,
    top_n: int = 15
) -> None:

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        logger.warning(f"Model tidak memiliki feature_importances_. Skip.")
        return

    fi_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(fi_df["feature"], fi_df["importance"],
                   color="#2196F3", edgecolor="white", height=0.7)
    ax.set_xlabel("Feature Importance (Gain)")
    ax.set_title(
        f"Top {top_n} Feature Importance\n"
        f"{model_name} | {config_name}",
        fontweight="bold"
    )
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                f"{width:.4f}", va="center", fontsize=8)
    plt.tight_layout()

    save_path = FIGURES_DIR / f"fi_{config_name}_{model_name}.png"
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    logger.info(f"Feature importance disimpan: {save_path}")


# =============================================================
# MAIN TRAINING LOOP
# =============================================================

import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

def run_training_pipeline() -> None:
    logger.info("=" * 65)
    logger.info("FASE 2: PIPELINE TRAINING DIMULAI ")
    logger.info("=" * 65)

    # Load data
    (X_train, y_train, X_val, y_val,
     X_test, y_test, feature_cols) = load_splits()

    # Struktur untuk menyimpan hasil semua konfigurasi
    all_metrics_rows  = []
    thresholds_dict   = {}
    results_per_config = {}   # Untuk grafik PR/ROC gabungan

    # ==========================================================
    # LOOP UTAMA: ITERASI SETIAP KONFIGURASI SMOTE
    # ==========================================================
    for config_name in SMOTE_CONFIGS.keys():

        logger.info("=" * 65)
        logger.info(f"KONFIGURASI: {config_name.upper()}")
        logger.info("=" * 65)

        results_per_config[config_name] = {}

        # 1. Menerapkan SMOTE pada training set
        X_train_res, y_train_res = apply_smote(
            X_train, y_train, config_name
        )

        # 2. Menghitung scale_pos_weight untuk XGBoost
        spw = get_scale_pos_weight(y_train_res, config_name)

        # ======================================================
        # TRAINING & EVALUASI XGBOOST
        # ======================================================
        logger.info(f"--- [{config_name}] XGBoost ---")

        xgb_model, xgb_params, xgb_cv_score = train_xgboost(
            X_train_res, y_train_res,
            X_val, y_val,
            spw, config_name
        )

        # Prediksi probabilitas
        y_prob_val_xgb  = xgb_model.predict_proba(X_val)[:, 1]
        y_prob_test_xgb = xgb_model.predict_proba(X_test)[:, 1]

        # Threshold dari validation set
        threshold_xgb = find_optimal_threshold(
            y_val.values, y_prob_val_xgb,
            config_name, "XGBoost"
        )

        # Metrik pada validation set
        val_metrics_xgb = compute_metrics(
            y_val.values, y_prob_val_xgb,
            threshold_xgb, config_name, "XGBoost", "val"
        )
        val_metrics_xgb["cv_pr_auc"] = round(xgb_cv_score, 4)
        val_metrics_xgb["best_params"] = str(xgb_params)

        # Metrik pada test set (final evaluation)
        test_metrics_xgb = compute_metrics(
            y_test.values, y_prob_test_xgb,
            threshold_xgb, config_name, "XGBoost", "test"
        )
        test_metrics_xgb["cv_pr_auc"] = round(xgb_cv_score, 4)
        test_metrics_xgb["best_params"] = str(xgb_params)

        all_metrics_rows.extend([val_metrics_xgb, test_metrics_xgb])

        # Confusion matrix test set
        plot_confusion_matrix(
            y_test.values, y_prob_test_xgb,
            threshold_xgb, config_name, "XGBoost"
        )

        # Feature importance
        plot_feature_importance(
            xgb_model, feature_cols,
            config_name, "XGBoost"
        )

        # Simpan model
        model_path = MODELS_DIR / f"xgb_{config_name}.joblib"
        joblib.dump(xgb_model, model_path)
        logger.info(f"Model disimpan: {model_path}")

        # Simpan hasil untuk grafik gabungan
        results_per_config[config_name]["XGBoost"] = {
            "y_prob_test":  y_prob_test_xgb,
            "test_metrics": test_metrics_xgb,
            "threshold":    threshold_xgb
        }

        # Threshold record
        thresholds_dict[f"{config_name}_XGBoost"] = {
            "threshold":    round(threshold_xgb, 4),
            "val_f1":       val_metrics_xgb["f1"],
            "val_recall":   val_metrics_xgb["recall"],
            "val_precision":val_metrics_xgb["precision"],
        }

        # ======================================================
        # TRAINING & EVALUASI RANDOM FOREST
        # ======================================================
        logger.info(f"--- [{config_name}] Random Forest ---")

        rf_model, rf_params, rf_cv_score = train_random_forest(
            X_train_res, y_train_res, config_name
        )

        y_prob_val_rf  = rf_model.predict_proba(X_val)[:, 1]
        y_prob_test_rf = rf_model.predict_proba(X_test)[:, 1]

        threshold_rf = find_optimal_threshold(
            y_val.values, y_prob_val_rf,
            config_name, "RandomForest"
        )

        val_metrics_rf = compute_metrics(
            y_val.values, y_prob_val_rf,
            threshold_rf, config_name, "RandomForest", "val"
        )
        val_metrics_rf["cv_pr_auc"] = round(rf_cv_score, 4)
        val_metrics_rf["best_params"] = str(rf_params)

        test_metrics_rf = compute_metrics(
            y_test.values, y_prob_test_rf,
            threshold_rf, config_name, "RandomForest", "test"
        )
        test_metrics_rf["cv_pr_auc"] = round(rf_cv_score, 4)
        test_metrics_rf["best_params"] = str(rf_params)

        all_metrics_rows.extend([val_metrics_rf, test_metrics_rf])

        plot_confusion_matrix(
            y_test.values, y_prob_test_rf,
            threshold_rf, config_name, "RandomForest"
        )

        plot_feature_importance(
            rf_model, feature_cols,
            config_name, "RandomForest"
        )

        model_path = MODELS_DIR / f"rf_{config_name}.joblib"
        joblib.dump(rf_model, model_path)
        logger.info(f"Model disimpan: {model_path}")

        results_per_config[config_name]["RandomForest"] = {
            "y_prob_test":  y_prob_test_rf,
            "test_metrics": test_metrics_rf,
            "threshold":    threshold_rf
        }

        thresholds_dict[f"{config_name}_RandomForest"] = {
            "threshold":    round(threshold_rf, 4),
            "val_f1":       val_metrics_rf["f1"],
            "val_recall":   val_metrics_rf["recall"],
            "val_precision":val_metrics_rf["precision"],
        }

        logger.info(
            f"[{config_name}] SELESAI | "
            f"XGB → PR-AUC={test_metrics_xgb['pr_auc']:.4f} | F1={test_metrics_xgb['f1']:.4f} | "
            f"RF  → PR-AUC={test_metrics_rf['pr_auc']:.4f} | F1={test_metrics_rf['f1']:.4f}"
        )   

    # ==========================================================
    # MENYIMPAN SEMUA HASIL EVALUASI
    # ==========================================================

    # Tabel metrik lengkap
    df_metrics = pd.DataFrame(all_metrics_rows)
    metrics_path = EVALUATION_DIR / "all_metrics.csv"
    df_metrics.to_csv(metrics_path, index=False)
    logger.info(f"Semua metrik disimpan: {metrics_path}")

    # Thresholds
    thresholds_path = EVALUATION_DIR / "thresholds.json"
    with open(thresholds_path, "w") as f:
        json.dump(thresholds_dict, f, indent=2)
    logger.info(f"Thresholds disimpan: {thresholds_path}")

    # Grafik gabungan PR/ROC
    plot_pr_roc_curves(results_per_config, y_test.values)

    # Grafik perbandingan metrik
    plot_metrics_comparison(df_metrics)

    # ==========================================================
    # PEMILIHAN MODEL TERBAIK
    # ==========================================================

    select_best_model(df_metrics, results_per_config, feature_cols)

    logger.info("=" * 65)
    logger.info("PIPELINE TRAINING FASE 2 SELESAI")
    logger.info("=" * 65)


def select_best_model(
    df_metrics: pd.DataFrame,
    results_per_config: dict,
    feature_cols: list
) -> None:

    logger.info("=== PEMILIHAN MODEL TERBAIK ===")
    logger.info("Primary model: XGBoost | RF: model pembanding")

    df_test = df_metrics[df_metrics["split"] == "test"].copy()
    df_xgb  = df_test[df_test["model"] == "XGBoost"].copy()
    df_rf   = df_test[df_test["model"] == "RandomForest"].copy()

    # --- XGBoost terbaik ---
    candidates_xgb = df_xgb[df_xgb["meets_all_targets"] == True]

    if len(candidates_xgb) > 0:
        best_xgb_row = candidates_xgb.loc[
            candidates_xgb["pr_auc"].idxmax()
        ]
        selection_reason = (
            f"{len(candidates_xgb)} dari {len(df_xgb)} konfigurasi XGBoost "
            f"memenuhi semua target; dipilih berdasarkan PR-AUC tertinggi."
        )
        logger.info(
            f"{len(candidates_xgb)}/4 konfigurasi XGBoost memenuhi "
            f"semua target metrik."
        )
    else:
        best_xgb_row = df_xgb.loc[df_xgb["f1"].idxmax()]
        selection_reason = (
            "Tidak ada konfigurasi XGBoost yang memenuhi semua target; "
            "dipilih berdasarkan F1-Score tertinggi."
        )
        logger.warning(
            "PERHATIAN: Tidak ada konfigurasi XGBoost yang memenuhi "
            "semua target. Pertimbangkan analisis lanjutan."
        )

    best_config = best_xgb_row["config"]

    logger.info(
        f"XGBoost TERBAIK → Config: {best_config} | "
        f"Recall={best_xgb_row['recall']:.4f} | "
        f"Precision={best_xgb_row['precision']:.4f} | "
        f"F1={best_xgb_row['f1']:.4f} | "
        f"PR-AUC={best_xgb_row['pr_auc']:.4f} | "
        f"Threshold={best_xgb_row['threshold']:.4f}"
    )

    # --- Log perbandingan RF sebagai referensi ---
    logger.info("--- Random Forest (Pembanding) ---")
    for _, row in df_rf.sort_values("config").iterrows():
        logger.info(
            f"  RF {row['config']:<12} | "
            f"Recall={row['recall']:.4f} | "
            f"F1={row['f1']:.4f} | "
            f"PR-AUC={row['pr_auc']:.4f}"
        )

    # --- Log semua konfigurasi XGBoost untuk RQ2 ---
    logger.info("--- XGBoost Semua Konfigurasi ---")
    for _, row in df_xgb.sort_values("config").iterrows():
        logger.info(
            f"  XGB {row['config']:<12} | "
            f"PR-AUC={row['pr_auc']:.4f} | "
            f"F1={row['f1']:.4f} | "
            f"Threshold={row['threshold']:.4f} | "
            f"Meets targets: {row['meets_all_targets']}"
        )

    # --- Simpan best pipeline (selalu XGBoost) ---
    best_model_path = MODELS_DIR / f"xgb_{best_config}.joblib"
    best_model = joblib.load(best_model_path)

    best_pipeline = {
        "model":           best_model,
        "model_type":      "XGBoost",
        "config_name":     best_config,
        "feature_columns": feature_cols,
        "threshold":       results_per_config[best_config]["XGBoost"]["threshold"],
        "test_metrics":    best_xgb_row.to_dict(),
    }

    pipeline_path = MODELS_DIR / "best_pipeline.joblib"
    joblib.dump(best_pipeline, pipeline_path)
    logger.info(f"Best pipeline disimpan: {pipeline_path} (XGBoost — {best_config})")

    # --- Simpan best_config.json keseluruhan ---
    xgb_all_configs = {}
    for _, row in df_xgb.sort_values("config").iterrows():
        xgb_all_configs[row["config"]] = {
            "pr_auc":            float(row["pr_auc"]),
            "roc_auc":           float(row["roc_auc"]),
            "f1":                float(row["f1"]),
            "recall":            float(row["recall"]),
            "precision":         float(row["precision"]),
            "threshold":         float(row["threshold"]),
            "meets_all_targets": bool(row["meets_all_targets"]),
        }

    rf_summary = {}
    for _, row in df_rf.sort_values("config").iterrows():
        rf_summary[row["config"]] = {
            "pr_auc":   float(row["pr_auc"]),
            "f1":       float(row["f1"]),
            "recall":   float(row["recall"]),
            "precision":float(row["precision"]),
        }

    best_config_out = {
        "primary_model":         "XGBoost",
        "best_xgb_config":       best_config,
        "selection_reason":      selection_reason,
        "threshold":             float(
            results_per_config[best_config]["XGBoost"]["threshold"]
        ),
        "best_xgb_test_metrics": best_xgb_row.to_dict(),
        "all_xgb_configs":       xgb_all_configs,
        "rf_role":               "model_pembanding_fixed_parameter",
        "rf_summary":            rf_summary,
    }

    best_path = EVALUATION_DIR / "best_config.json"
    with open(best_path, "w") as f:
        json.dump(best_config_out, f, indent=2, default=str)
    logger.info(f"Ringkasan best config disimpan: {best_path}")


if __name__ == "__main__":
    run_training_pipeline()