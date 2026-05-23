import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import joblib
from functools import lru_cache

from config import MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Mapping OHE
# Diambil dari split_info["feature_columns"] saat pipeline dimuat
OHE_PREFIX = "type_"
TX_TYPES   = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]


@lru_cache(maxsize=1)
def load_pipeline() -> dict:
    """
    Memuat best_pipeline_final.joblib sekali ke memori.
    lru_cache memastikan model tidak dimuat ulang setiap request.
    """
    path = MODELS_DIR / "best_pipeline_final.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"best_pipeline_final.joblib tidak ditemukan: {path}\n"
            "Jalankan: python src/api/build_final_pipeline.py"
        )
    pipeline = joblib.load(path)
    logger.info(
        f"Pipeline dimuat: {pipeline['model_name']} | "
        f"config={pipeline['config_name']} | "
        f"theta_high={pipeline['theta_high']:.4f}"
    )
    return pipeline


def engineer_features(tx: dict) -> pd.DataFrame:
    """
    Rekayasa fitur identik dengan pipeline Fase 1.
    Input: dict dari TransactionInput
    Output: DataFrame 1 baris dengan fitur yang sama dengan training
    """
    pipeline      = load_pipeline()
    feature_cols  = pipeline["feature_columns"]

    # Fitur dasar dari input
    amount          = float(tx["amount"])
    old_balance_org = float(tx["oldbalanceOrg"])
    new_balance_org = float(tx["newbalanceOrig"])
    old_balance_dst = float(tx["oldbalanceDest"])
    new_balance_dst = float(tx["newbalanceDest"])
    step            = int(tx["step"])
    tx_type         = str(tx["type"]).upper()

    # Kelompok 1: Balance Consistency Features 
    error_balance_orig = (old_balance_org - amount) - new_balance_org
    error_balance_dest = (old_balance_dst + amount) - new_balance_dst

    # Kelompok 2: Temporal Features
    transaction_hour = step % 24
    transaction_day  = step // 24

    # Kelompok 3: Transaction Ratio
    amount_to_balance_orig = amount / (old_balance_org + 1)

    # One-Hot Encoding tipe transaksi
    ohe = {f"type_{t}": (1 if tx_type == t else 0) for t in TX_TYPES}

    row = {
        "step":                step,
        "amount":              amount,
        "oldbalanceOrg":       old_balance_org,
        "newbalanceOrig":      new_balance_org,
        "oldbalanceDest":      old_balance_dst,
        "newbalanceDest":      new_balance_dst,
        "errorBalanceOrig":    error_balance_orig,
        "errorBalanceDest":    error_balance_dest,
        "transaction_hour":    transaction_hour,
        "transaction_day":     transaction_day,
        "amountToBalanceOrig": amount_to_balance_orig,
        **ohe
    }

    # Menyusun kolom sesuai urutan training, untuk konsistensi prediksi
    df = pd.DataFrame([row])
    df = df.reindex(columns=feature_cols, fill_value=0)

    return df


def predict(tx: dict) -> tuple[float, pd.DataFrame]:
    """
    Menjalankan inferensi XGBoost untuk satu transaksi.
    """
    pipeline  = load_pipeline()
    model     = pipeline["model"]
    X_row     = engineer_features(tx)
    fraud_prob = float(model.predict_proba(X_row)[0, 1])
    return fraud_prob, X_row