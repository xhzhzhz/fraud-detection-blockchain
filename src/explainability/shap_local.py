import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import hashlib
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from config import (
    DATA_SPLITS_DIR, SHAP_DIR,
    SHAP_TOP_K_FEATURES, SHAP_TOP_K_AUDIT,
    EVALUATION_DIR, TARGET_COLUMN, RANDOM_STATE
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Template penjelasan berbasis bahasa alami
# Setiap fitur memiliki template deskripsi untuk arah positif (→fraud)
# dan negatif (→legitimate).
NL_TEMPLATE = {
    "errorBalanceOrig": {
        "pos": "Terdapat inkonsistensi signifikan antara saldo dan nominal transaksi pengirim",
        "neg": "Saldo pengirim konsisten dengan nominal transaksi"
    },
    "errorBalanceDest": {
        "pos": "Terdapat inkonsistensi saldo pada akun penerima",
        "neg": "Saldo penerima konsisten dengan nominal transaksi"
    },
    "amountToBalanceOrig": {
        "pos": "Nominal transaksi tidak proporsional terhadap saldo pengirim",
        "neg": "Nominal transaksi proporsional terhadap saldo pengirim"
    },
    "transaction_hour": {
        "pos": "Transaksi dilakukan pada jam berisiko tinggi",
        "neg": "Transaksi dilakukan pada jam normal"
    },
    "transaction_day": {
        "pos": "Transaksi terjadi pada hari dengan pola fraud tinggi",
        "neg": "Transaksi terjadi pada hari normal"
    },
    "amount": {
        "pos": "Nominal transaksi tergolong sangat tinggi",
        "neg": "Nominal transaksi dalam rentang normal"
    },
    "oldbalanceOrg": {
        "pos": "Saldo awal pengirim memiliki pola berisiko",
        "neg": "Saldo awal pengirim dalam rentang normal"
    },
    "newbalanceOrig": {
        "pos": "Saldo akhir pengirim menunjukkan pola tidak wajar",
        "neg": "Saldo akhir pengirim normal setelah transaksi"
    },
    "oldbalanceDest": {
        "pos": "Saldo awal penerima memiliki pola berisiko",
        "neg": "Saldo awal penerima dalam rentang normal"
    },
    "newbalanceDest": {
        "pos": "Saldo akhir penerima menunjukkan pola tidak wajar",
        "neg": "Saldo akhir penerima normal setelah transaksi"
    },
}

# Template untuk fitur OHE tipe transaksi
TX_TYPE_TEMPLATE = {
    "pos": "Tipe transaksi {tx_type} memiliki asosiasi tinggi dengan pola kecurangan",
    "neg": "Tipe transaksi {tx_type} tidak mengindikasikan risiko khusus"
}


def get_nl_description(feature_name: str, shap_value: float) -> str:
    """
    Menghasilkan deskripsi bahasa alami untuk satu fitur
    berdasarkan nama fitur dan arah SHAP value.
    """
    direction = "pos" if shap_value > 0 else "neg"

    if feature_name in NL_TEMPLATE:
        return NL_TEMPLATE[feature_name][direction]

    # Fitur OHE tipe transaksi
    if feature_name.startswith("type_"):
        tx_type = feature_name.replace("type_", "")
        return TX_TYPE_TEMPLATE[direction].format(tx_type=tx_type)

    # Fallback generic
    if direction == "pos":
        return f"Fitur '{feature_name}' berkontribusi meningkatkan risiko fraud"
    return f"Fitur '{feature_name}' berkontribusi menurunkan risiko fraud"


def build_natural_language_explanation(
    top_features: list[dict]
) -> str:
    """
    Merangkai penjelasan top features menjadi narasi tunggal
    yang dapat dipahami non-teknis.
    """
    positive_contributors = [
        f["nl_description"]
        for f in top_features if f["shap_value"] > 0
    ]

    if not positive_contributors:
        return (
            "Transaksi ini diidentifikasi berisiko berdasarkan "
            "kombinasi beberapa indikator."
        )

    if len(positive_contributors) == 1:
        return (
            f"Transaksi ini diidentifikasi sebagai fraud karena: "
            f"{positive_contributors[0]}."
        )

    factors = "; ".join(positive_contributors[:-1])
    last    = positive_contributors[-1]
    return (
        f"Transaksi ini diidentifikasi sebagai fraud karena: "
        f"{factors}; dan {last}."
    )


def compute_shap_hash(top_k_features: list[dict], k: int = None) -> str:
    """
    Menghitung hash SHA-256 dari top-k SHAP values.
    Sebagai shapHash yang disimpan on-chain di Fase 4.
    """
    if k is None:
        k = SHAP_TOP_K_AUDIT

    top_k = top_k_features[:k]
    payload = json.dumps(
        [{"feature": f["feature"], "shap_value": round(f["shap_value"], 6)}
         for f in top_k],
        sort_keys=True
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def generate_local_explanation(
    row_idx: int,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    shap_values: np.ndarray,
    base_value: float,
    fraud_prob: float,
    theta_high: float,
    theta_low: float,
    config_name: str
) -> dict:
    """
    Menghasilkan laporan penjelasan lokal lengkap untuk satu
    instance transaksi
    """
    feature_names = list(X_test.columns)
    shap_row      = shap_values[row_idx]

    # Menentukan status keputusan dari dual threshold
    if fraud_prob > theta_high:
        decision = "FRAUD"
    elif fraud_prob >= theta_low:
        decision = "SUSPICIOUS"
    else:
        decision = "LEGITIMATE"

    # Mengurutkan fitur berdasarkan |SHAP| descending
    sorted_idx = np.argsort(np.abs(shap_row))[::-1]

    # Top-k fitur untuk penjelasan (k=SHAP_TOP_K_FEATURES=5)
    top_features = []
    for rank, fi in enumerate(sorted_idx[:SHAP_TOP_K_FEATURES]):
        fname      = feature_names[fi]
        sval       = float(shap_row[fi])
        fval       = float(X_test.iloc[row_idx, fi])
        direction  = "increases_fraud_risk" if sval > 0 else "decreases_fraud_risk"
        nl_desc    = get_nl_description(fname, sval)

        top_features.append({
            "rank":          rank + 1,
            "feature":       fname,
            "shap_value":    round(sval, 6),
            "feature_value": round(fval, 4),
            "direction":     direction,
            "nl_description": nl_desc
        })

    nl_summary = build_natural_language_explanation(top_features)
    shap_hash  = compute_shap_hash(top_features)

    explanation = {
        "transaction_id":    f"TEST-{row_idx:07d}",
        "config_name":       config_name,
        "actual_label":      int(y_test.iloc[row_idx]),
        "fraud_score":       round(float(fraud_prob), 6),
        "decision":          decision,
        "theta_low":         round(theta_low, 4),
        "theta_high":        round(theta_high, 4),
        "shap_base_value":   round(float(base_value), 6),
        "shap_sum":          round(float(shap_row.sum()), 6),
        "top_features":      top_features,
        "natural_language_explanation": nl_summary,
        "shap_hash_top3":    shap_hash,   # shapHash untuk blockchain 
        "generated_at":      datetime.now(timezone.utc).isoformat()
    }

    return explanation


def generate_sample_explanations(
    shap_results: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    dual_thresholds: dict,
    n_fraud_samples: int = 30
) -> dict:
    """
    Menghasilkan laporan penjelasan lokal untuk n_fraud_samples
    transaksi fraud aktual dari test set.

    Hanya fraud aktual yang diambil: penjelasan lokal dihasilkan 
    untuk transaksi yang diprediksi atau aktual sebagai fraud 
    untuk mendukung audit regulasi.
    """
    import joblib

    fraud_indices = y_test[y_test == 1].index.tolist()
    rng           = np.random.default_rng(RANDOM_STATE)
    sample_idx    = rng.choice(
        len(fraud_indices),
        size=min(n_fraud_samples, len(fraud_indices)),
        replace=False
    )
    sampled_fraud_indices = [fraud_indices[i] for i in sample_idx]

    # Posisi dalam array (iloc position)
    fraud_positions = [
        y_test.index.get_loc(i) for i in sampled_fraud_indices
    ]

    all_explanations = {}

    for config_name, data in shap_results.items():
        shap_values = data["shap_values"]
        base_value  = data["base_value"]

        # Load model untuk mendapatkan probabilitas prediksi
        model_path = Path(__file__).resolve().parents[2] / \
                     "models" / "saved" / f"xgb_{config_name}.joblib"
        model      = joblib.load(model_path)
        y_prob     = model.predict_proba(X_test)[:, 1]

        # Mengambil threshold dari dual_thresholds
        key        = f"{config_name}_XGBoost"
        theta_high = dual_thresholds.get(key, {}).get("theta_high", 0.5)
        theta_low  = dual_thresholds.get(key, {}).get("theta_low", 0.3)

        config_explanations = []
        for pos in fraud_positions:
            expl = generate_local_explanation(
                row_idx    = pos,
                X_test     = X_test,
                y_test     = y_test,
                shap_values= shap_values,
                base_value = base_value,
                fraud_prob = float(y_prob[pos]),
                theta_high = theta_high,
                theta_low  = theta_low,
                config_name= config_name
            )
            config_explanations.append(expl)

        all_explanations[config_name] = config_explanations
        logger.info(
            f"[{config_name}] {len(config_explanations)} "
            "penjelasan lokal dihasilkan."
        )

    # Simpan ke JSON
    save_path = SHAP_DIR / "local_explanations_sample.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(all_explanations, f, indent=2, ensure_ascii=False)
    logger.info(f"Penjelasan lokal disimpan: {save_path}")

    return all_explanations