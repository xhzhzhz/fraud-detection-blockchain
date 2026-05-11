import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import numpy as np
import pandas as pd

from config import (
    RAW_DATASET_PATH, DATA_PROCESSED_DIR, DATA_SPLITS_DIR,
    COLUMNS_TO_DROP, TARGET_COLUMN,
    TRAIN_RATIO, VAL_RATIO, RANDOM_STATE
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================
# STEP 1: LOAD DATA
# =============================================================

def load_raw_data(path: Path) -> pd.DataFrame:
    logger.info(f"Memuat dataset dari: {path}")
    df = pd.read_csv(path)
    logger.info(f"Dataset dimuat: {df.shape[0]:,} baris × {df.shape[1]} kolom")
    logger.info(f"Memori awal: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    return df


# =============================================================
# STEP 2: OPTIMASI TIPE DATA
# =============================================================

def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Mengoptimasi tipe data...")
    
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype(np.float32)
    
    for col in df.select_dtypes(include=["int64"]).columns:
        col_min, col_max = df[col].min(), df[col].max()
        if col_min >= 0 and col_max <= 255:
            df[col] = df[col].astype(np.uint8)
        elif col_min >= -128 and col_max <= 127:
            df[col] = df[col].astype(np.int8)
        elif col_min >= -32768 and col_max <= 32767:
            df[col] = df[col].astype(np.int16)
        else:
            df[col] = df[col].astype(np.int32)
    
    logger.info(f"Memori setelah optimasi: "
                f"{df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    return df


# =============================================================
# STEP 3: DROP KOLOM TIDAK RELEVAN
# =============================================================

def drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    logger.info(f"Menghapus kolom: {COLUMNS_TO_DROP}")
    
    cols_exist = [c for c in COLUMNS_TO_DROP if c in df.columns]
    cols_missing = [c for c in COLUMNS_TO_DROP if c not in df.columns]
    
    if cols_missing:
        logger.warning(f"Kolom tidak ditemukan (skip): {cols_missing}")
    
    df = df.drop(columns=cols_exist)
    logger.info(f"Kolom tersisa: {list(df.columns)}")
    return df


# =============================================================
# STEP 4: PEMBERSIHAN DATA
# =============================================================

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Memeriksa missing values...")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        logger.warning(f"Missing values ditemukan:\n{missing[missing > 0]}")
        df = df.dropna()
        logger.info("Baris dengan missing values dihapus.")
    else:
        logger.info("Tidak ada missing values.")

    logger.info("Memeriksa duplikasi...")
    n_before = len(df)
    df = df.drop_duplicates()
    n_removed = n_before - len(df)
    if n_removed > 0:
        logger.info(f"Menghapus {n_removed:,} baris duplikat.")
    else:
        logger.info("Tidak ada baris duplikat.")

    return df.reset_index(drop=True)


# =============================================================
# STEP 5: FEATURE ENGINEERING
# =============================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Memulai feature engineering...")

    # Kelompok 1: Balance Consistency Features
    df["errorBalanceOrig"] = (
        (df["oldbalanceOrg"] - df["amount"]) - df["newbalanceOrig"]
    ).astype(np.float32)

    df["errorBalanceDest"] = (
        (df["oldbalanceDest"] + df["amount"]) - df["newbalanceDest"]
    ).astype(np.float32)

    # Kelompok 2: Temporal Features
    df["transaction_hour"] = (df["step"] % 24).astype(np.int8)
    df["transaction_day"]  = (df["step"] // 24).astype(np.int8)

    # Kelompok 3: Transaction Ratio Feature
    # +1 pada penyebut mencegah pembagian dengan nol
    df["amountToBalanceOrig"] = (
        df["amount"] / (df["oldbalanceOrg"] + 1)
    ).astype(np.float32)

    logger.info(
        "Fitur baru: errorBalanceOrig, errorBalanceDest, "
        "transaction_hour, transaction_day, amountToBalanceOrig"
    )

    # Verifikasi: cek statistik errorBalance pada kelas fraud vs legitimate
    for feat in ["errorBalanceOrig", "errorBalanceDest"]:
        zero_legit = (df[df[TARGET_COLUMN] == 0][feat] == 0).mean() * 100
        zero_fraud = (df[df[TARGET_COLUMN] == 1][feat] == 0).mean() * 100
        logger.info(
            f"{feat} — nilai nol: {zero_legit:.1f}% (legit), "
            f"{zero_fraud:.1f}% (fraud)"
        )

    return df


# =============================================================
# STEP 6: ONE-HOT ENCODING
# =============================================================

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Menerapkan One-Hot Encoding pada kolom 'type'...")

    type_dummies = pd.get_dummies(
        df["type"],
        prefix="type",
        drop_first=False,   # Mempertahankan semua kolom untuk SHAP
        dtype=np.uint8      
    )

    df = pd.concat([df.drop(columns=["type"]), type_dummies], axis=1)
    
    ohe_cols = [c for c in df.columns if c.startswith("type_")]
    logger.info(f"Kolom OHE yang dihasilkan: {ohe_cols}")
    logger.info(f"Total kolom setelah encoding: {df.shape[1]}")

    return df


# =============================================================
# STEP 7: TEMPORAL SPLIT
# =============================================================

def temporal_split(
    df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    
    logger.info("Melakukan stratified temporal split (70/15/15)...")

    max_step = int(df["step"].max())
    min_step = int(df["step"].min())

    # Menghitung cutoff secara dinamis dari data aktual
    step_range = max_step - min_step
    train_cutoff = min_step + int(step_range * TRAIN_RATIO)
    val_cutoff   = min_step + int(step_range * (TRAIN_RATIO + VAL_RATIO))

    logger.info(f"Step range  : {min_step} – {max_step}")
    logger.info(f"Train cutoff: step ≤ {train_cutoff}")
    logger.info(f"Val cutoff  : step ≤ {val_cutoff}")
    logger.info(f"Test range  : step > {val_cutoff}")

    train_df = df[df["step"] <= train_cutoff].copy()
    val_df   = df[(df["step"] > train_cutoff) &
                  (df["step"] <= val_cutoff)].copy()
    test_df  = df[df["step"] > val_cutoff].copy()

    # Verifikasi proporsi fraud di setiap split
    def fraud_pct(d):
        return d[TARGET_COLUMN].mean() * 100

    logger.info(
        f"Train: {len(train_df):,} baris | "
        f"Fraud: {fraud_pct(train_df):.4f}%"
    )
    logger.info(
        f"Val  : {len(val_df):,} baris | "
        f"Fraud: {fraud_pct(val_df):.4f}%"
    )
    logger.info(
        f"Test : {len(test_df):,} baris | "
        f"Fraud: {fraud_pct(test_df):.4f}%"
    )

    # Verifikasi tidak ada overlap
    train_steps = set(train_df["step"].unique())
    val_steps   = set(val_df["step"].unique())
    test_steps  = set(test_df["step"].unique())

    assert len(train_steps & val_steps) == 0, "OVERLAP: Train dan Val!"
    assert len(val_steps & test_steps) == 0,  "OVERLAP: Val dan Test!"
    assert len(train_steps & test_steps) == 0, "OVERLAP: Train dan Test!"
    logger.info("Verifikasi overlap: PASSED — tidak ada overlap antar split.")

    # Metadata split
    split_info = {
        "step_range": {"min": min_step, "max": max_step},
        "train_cutoff": train_cutoff,
        "val_cutoff": val_cutoff,
        "splits": {
            "train": {
                "n_rows": len(train_df),
                "n_fraud": int(train_df[TARGET_COLUMN].sum()),
                "fraud_pct": float(fraud_pct(train_df)),
                "step_range": [int(train_df["step"].min()),
                               int(train_df["step"].max())]
            },
            "val": {
                "n_rows": len(val_df),
                "n_fraud": int(val_df[TARGET_COLUMN].sum()),
                "fraud_pct": float(fraud_pct(val_df)),
                "step_range": [int(val_df["step"].min()),
                               int(val_df["step"].max())]
            },
            "test": {
                "n_rows": len(test_df),
                "n_fraud": int(test_df[TARGET_COLUMN].sum()),
                "fraud_pct": float(fraud_pct(test_df)),
                "step_range": [int(test_df["step"].min()),
                               int(test_df["step"].max())]
            }
        },
        "columns": list(train_df.columns),
        "feature_columns": [c for c in train_df.columns
                            if c != TARGET_COLUMN and c != "step"]
    }

    return train_df, val_df, test_df, split_info


# =============================================================
# STEP 8: SIMPAN OUTPUT
# =============================================================

def save_outputs(
    df_processed: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    split_info: dict
) -> None:

    # Menyimpan dataset processed lengkap (parquet — efisien)
    processed_path = DATA_PROCESSED_DIR / "paysim_processed.parquet"
    df_processed.to_parquet(processed_path, index=False)
    logger.info(f"Dataset processed disimpan: {processed_path}")

    # Menyimpan split data sebagai CSV 
    for name, df_split in [("train", train_df),
                            ("val",   val_df),
                            ("test",  test_df)]:
        path = DATA_SPLITS_DIR / f"{name}.csv"
        df_split.to_csv(path, index=False)
        logger.info(f"Split '{name}' disimpan: {path} "
                    f"({len(df_split):,} baris)")

    # Menyimpan metadata split
    info_path = DATA_SPLITS_DIR / "split_info.json"
    with open(info_path, "w") as f:
        json.dump(split_info, f, indent=2)
    logger.info(f"Metadata split disimpan: {info_path}")

    # Ringkasan ukuran file
    logger.info("=== RINGKASAN OUTPUT ===")
    for path in [processed_path,
                 DATA_SPLITS_DIR / "train.csv",
                 DATA_SPLITS_DIR / "val.csv",
                 DATA_SPLITS_DIR / "test.csv",
                 info_path]:
        size_mb = path.stat().st_size / 1e6
        logger.info(f"  {path.name:<35} {size_mb:>7.1f} MB")


# =============================================================
# MAIN PIPELINE
# =============================================================

def run_preprocessing_pipeline() -> None:
    logger.info("=" * 60)
    logger.info("FASE 1: PREPROCESSING DIMULAI")
    logger.info("=" * 60)

    # Validasi dataset ada
    if not RAW_DATASET_PATH.exists():
        logger.error(
            f"Dataset tidak ditemukan di: {RAW_DATASET_PATH}\n"
            "Pastikan file sudah diunduh ke data/raw/"
        )
        sys.exit(1)

    df = load_raw_data(RAW_DATASET_PATH)
    df = optimize_dtypes(df)
    df = drop_irrelevant_columns(df)
    df = clean_data(df)
    df = engineer_features(df)
    df = encode_features(df)

    train_df, val_df, test_df, split_info = temporal_split(df)

    save_outputs(df, train_df, val_df, test_df, split_info)

    logger.info("=" * 60)
    logger.info("PIPELINE PREPROCESSING SELESAI")
    logger.info(f"Train set : {split_info['splits']['train']['n_rows']:,} baris")
    logger.info(f"Val set   : {split_info['splits']['val']['n_rows']:,} baris")
    logger.info(f"Test set  : {split_info['splits']['test']['n_rows']:,} baris")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_preprocessing_pipeline()