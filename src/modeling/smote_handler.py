import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from collections import Counter

from config import (
    TARGET_COLUMN, SMOTE_CONFIGS,
    SMOTE_K_NEIGHBORS, RANDOM_STATE
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config_name: str
) -> tuple[pd.DataFrame, pd.Series]:

    ratio = SMOTE_CONFIGS.get(config_name)

    # Distribusi kelas sebelum resampling
    counter_before = Counter(y_train)
    n_legit  = counter_before[0]
    n_fraud  = counter_before[1]
    fraud_pct_before = n_fraud / len(y_train) * 100

    logger.info(
        f"[{config_name}] Sebelum resampling — "
        f"Legitimate: {n_legit:,} | Fraud: {n_fraud:,} "
        f"({fraud_pct_before:.4f}%)"
    )

    # Baseline: tidak ada SMOTE
    if ratio is None:
        logger.info(f"[{config_name}] Tidak ada resampling (baseline).")
        return X_train.copy(), y_train.copy()

    # Menghitung sampling_strategy:
    # ratio = target proporsi fraud setelah resampling
    # n_fraud_target = ratio * (n_legit + n_fraud_target)
    # n_fraud_target = ratio * n_legit / (1 - ratio)
    n_fraud_target = int(ratio * n_legit / (1 - ratio))

    # Jika n_fraud_target <= n_fraud yang sudah ada, skip SMOTE
    if n_fraud_target <= n_fraud:
        logger.warning(
            f"[{config_name}] Target fraud ({n_fraud_target:,}) <= "
            f"fraud aktual ({n_fraud:,}). Skip SMOTE."
        )
        return X_train.copy(), y_train.copy()

    sampling_strategy = {1: n_fraud_target}

    logger.info(
        f"[{config_name}] Target fraud setelah SMOTE: "
        f"{n_fraud_target:,} ({ratio*100:.0f}% dari total)"
    )

    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=SMOTE_K_NEIGHBORS,
        random_state=RANDOM_STATE
    )

    X_res, y_res = smote.fit_resample(X_train, y_train)

    # Mengonversi kembali ke DataFrame/Series dengan nama kolom
    X_res = pd.DataFrame(X_res, columns=X_train.columns)
    y_res = pd.Series(y_res, name=TARGET_COLUMN)

    # Verifikasi hasil
    counter_after = Counter(y_res)
    fraud_pct_after = counter_after[1] / len(y_res) * 100
    logger.info(
        f"[{config_name}] Setelah resampling — "
        f"Legitimate: {counter_after[0]:,} | "
        f"Fraud: {counter_after[1]:,} ({fraud_pct_after:.2f}%)"
    )

    return X_res, y_res


def get_scale_pos_weight(y_train: pd.Series, config_name: str) -> float:
    """
    Menghitung scale_pos_weight untuk XGBoost.
    Untuk baseline (tanpa SMOTE): n_legit / n_fraud
    Untuk konfigurasi SMOTE: 1.0 (distribusi sudah diseimbangkan)
    """
    if config_name == "baseline":
        n_legit = (y_train == 0).sum()
        n_fraud = (y_train == 1).sum()
        spw = float(n_legit / n_fraud)
        logger.info(
            f"[{config_name}] scale_pos_weight = "
            f"{spw:.2f} ({n_legit:,}/{n_fraud:,})"
        )
        return spw
    else:
        logger.info(
            f"[{config_name}] scale_pos_weight = 1.0 "
            f"(distribusi sudah diseimbangkan SMOTE)"
        )
        return 1.0