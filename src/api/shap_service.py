import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import shap
from functools import lru_cache

from config import SHAP_TOP_K_FEATURES, SHAP_TOP_K_AUDIT
from src.api.ml_service import load_pipeline
from src.explainability.shap_local import (
    get_nl_description, build_natural_language_explanation,
    compute_shap_hash
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def load_explainer():
    """
    Inisialisasi TreeExplainer dari model yang sudah dimuat.
    Cache memastikan explainer hanya dibuat sekali per lifecycle API.
    """
    pipeline  = load_pipeline()
    model     = pipeline["model"]
    explainer = shap.TreeExplainer(model)
    logger.info("TreeExplainer berhasil diinisialisasi.")
    return explainer


def compute_local_shap(X_row: pd.DataFrame) -> tuple[np.ndarray, float]:
    """
    Menghitung SHAP values untuk satu instance.
    """
    explainer  = load_explainer()
    shap_vals  = explainer.shap_values(X_row, check_additivity=False)
    shap_row   = shap_vals[0]  # Mengambil baris pertama (satu instance)
    base_value = float(explainer.expected_value)
    return shap_row, base_value


def build_shap_explanation(
    X_row: pd.DataFrame,
    shap_row: np.ndarray,
    base_value: float
) -> dict:
    """
    Membangun struktur penjelasan SHAP 
    """
    feature_names = list(X_row.columns)
    sorted_idx    = np.argsort(np.abs(shap_row))[::-1]

    top_features = []
    for rank, fi in enumerate(sorted_idx[:SHAP_TOP_K_FEATURES]):
        fname     = feature_names[fi]
        sval      = float(shap_row[fi])
        fval      = float(X_row.iloc[0, fi])
        direction = "increases_fraud_risk" if sval > 0 else "decreases_fraud_risk"
        nl_desc   = get_nl_description(fname, sval)

        top_features.append({
            "rank":           rank + 1,
            "feature":        fname,
            "shap_value":     round(sval, 6),
            "feature_value":  round(fval, 4),
            "direction":      direction,
            "nl_description": nl_desc
        })

    nl_summary = build_natural_language_explanation(top_features)
    shap_hash  = compute_shap_hash(top_features, k=SHAP_TOP_K_AUDIT)

    return {
        "base_value":                    round(base_value, 6),
        "top_features":                  top_features,
        "natural_language_explanation":  nl_summary,
        "shap_hash_top3":                shap_hash
    }


def explain(X_row: pd.DataFrame) -> dict:
    """
    Entry point: hitung SHAP dan buat penjelasan untuk satu transaksi.
    """
    shap_row, base_value = compute_local_shap(X_row)
    explanation          = build_shap_explanation(X_row, shap_row, base_value)
    return explanation