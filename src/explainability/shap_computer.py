import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import joblib
import shap
import json

from config import (
    DATA_SPLITS_DIR, MODELS_DIR,
    SHAP_DIR, TARGET_COLUMN, SMOTE_CONFIGS
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_test_set() -> tuple[pd.DataFrame, pd.Series, list]:

    with open(DATA_SPLITS_DIR / "split_info.json") as f:
        split_info = json.load(f)

    feature_cols = split_info["feature_columns"]

    test_df = pd.read_csv(DATA_SPLITS_DIR / "test.csv")
    X_test  = test_df[feature_cols]
    y_test  = test_df[TARGET_COLUMN]

    logger.info(
        f"Test set dimuat: {X_test.shape[0]:,} baris | "
        f"{X_test.shape[1]} fitur | "
        f"Fraud: {y_test.sum():,} ({y_test.mean()*100:.4f}%)"
    )
    return X_test, y_test, feature_cols


def compute_shap_for_config(
    config_name: str,
    X_test: pd.DataFrame,
    force_recompute: bool = False
) -> tuple[np.ndarray, float]:

    shap_path      = SHAP_DIR / f"shap_values_{config_name}.npy"
    base_val_path  = SHAP_DIR / f"shap_base_value_{config_name}.npy"

    if shap_path.exists() and base_val_path.exists() and not force_recompute:
        logger.info(f"[{config_name}] SHAP values ditemukan di disk, load ulang.")
        shap_values = np.load(shap_path)
        base_value  = float(np.load(base_val_path))
        logger.info(
            f"[{config_name}] Loaded: shape={shap_values.shape}, "
            f"base_value={base_value:.6f}"
        )
        return shap_values, base_value

    # Load model XGBoost
    model_path = MODELS_DIR / f"xgb_{config_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model tidak ditemukan: {model_path}\n"
            "Pastikan Fase 2 sudah selesai."
        )
    model = joblib.load(model_path)
    logger.info(f"[{config_name}] Model XGBoost dimuat dari: {model_path}")

    # Inisialisasi TreeExplainer
    # check_additivity=False: menonaktifkan pengecekan numerik yang
    # memperlambat komputasi massal tanpa mengubah nilai SHAP
    logger.info(
        f"[{config_name}] Menginisialisasi TreeExplainer dan "
        f"Menghitung SHAP untuk {X_test.shape[0]:,} instance..."
    )
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test, check_additivity=False)
    base_value  = float(explainer.expected_value)

    logger.info(
        f"[{config_name}] Selesai. "
        f"shape={shap_values.shape} | base_value={base_value:.6f}"
    )

    # Simpan ke disk
    np.save(shap_path,     shap_values)
    np.save(base_val_path, np.array([base_value]))
    logger.info(f"[{config_name}] SHAP values disimpan: {shap_path}")

    return shap_values, base_value


def compute_all_configs(
    X_test: pd.DataFrame,
    force_recompute: bool = False
) -> dict:

    results = {}

    for config_name in SMOTE_CONFIGS.keys():
        logger.info(f"{'='*55}")
        logger.info(f"Komputasi SHAP: {config_name.upper()}")
        logger.info(f"{'='*55}")

        shap_values, base_value = compute_shap_for_config(
            config_name, X_test, force_recompute
        )
        results[config_name] = {
            "shap_values": shap_values,
            "base_value":  base_value
        }

    logger.info("Komputasi SHAP selesai untuk semua konfigurasi.")
    return results


def verify_additivity(
    shap_values: np.ndarray,
    base_value: float,
    y_prob: np.ndarray,
    config_name: str,
    tol: float = 1e-3
) -> dict:
    """
    Memverifikasi properti local accuracy SHAP:
    base_value + sum(phi_i) ≈ raw model output (log-odds space)

    Karena XGBoost menggunakan output log-odds dan SHAP TreeExplainer
    bekerja di ruang tersebut, verifikasi dilakukan menggunakan
    transformasi logit dari probabilitas prediksi.

    """
    import scipy.special

    shap_sum  = shap_values.sum(axis=1)
    predicted_logodds = scipy.special.logit(np.clip(y_prob, 1e-7, 1-1e-7))
    reconstructed     = base_value + shap_sum

    mae  = float(np.mean(np.abs(reconstructed - predicted_logodds)))
    pct_pass = float(
        np.mean(np.abs(reconstructed - predicted_logodds) < tol) * 100
    )

    result = {
        "config":           config_name,
        "mean_abs_error":   round(mae, 8),
        "pct_within_tol":   round(pct_pass, 4),
        "tolerance":        tol,
        "passes":           pct_pass >= 99.0
    }

    status = "PASSED" if result["passes"] else "WARNING"
    logger.info(
        f"[{config_name}] Additivity check {status}: "
        f"MAE={mae:.2e} | "
        f"{pct_pass:.2f}% instance dalam toleransi {tol}"
    )
    return result