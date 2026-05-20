import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve

from config import (
    DATA_SPLITS_DIR, MODELS_DIR, EVALUATION_DIR, FIGURES_DIR,
    TARGET_COLUMN, SMOTE_CONFIGS, TARGET_RECALL
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def find_theta_low(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_recall: float,
    config_name: str
) -> float:

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    # precision_recall_curve mengembalikan n+1 nilai untuk precisions/recalls
    # dan n nilai untuk thresholds 
    precisions = precisions[:-1]
    recalls    = recalls[:-1]

    # Mencari semua threshold di mana recall >= target
    valid_mask = recalls >= target_recall

    if not valid_mask.any():
        # Fallback: Menggunakan threshold yang menghasilkan recall tertinggi
        best_idx   = np.argmax(recalls)
        theta_low  = float(thresholds[best_idx])
        logger.warning(
            f"[{config_name}] Recall tidak mencapai target {target_recall:.2f} "
            f"di threshold manapun. Fallback ke threshold recall maksimum: "
            f"{theta_low:.4f} (recall={recalls[best_idx]:.4f})"
        )
        return theta_low

    valid_thresholds = thresholds[valid_mask]
    theta_low = float(valid_thresholds.max())

    # Verifikasi recall dan precision di theta_low
    idx = np.where(thresholds == theta_low)[0]
    if len(idx) > 0:
        r = recalls[idx[0]]
        p = precisions[idx[0]]
        logger.info(
            f"[{config_name}] theta_low = {theta_low:.4f} "
            f"| Recall={r:.4f} | Precision={p:.4f}"
        )
    else:
        logger.info(f"[{config_name}] theta_low = {theta_low:.4f}")

    return theta_low


def compute_three_class_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    theta_low: float,
    theta_high: float,
    config_name: str
) -> dict:
    """
    Menghitung distribusi prediksi tiga kelas dan metrik relevan.
    untuk validasi bahwa zona SUSPICIOUS bermakna.
    """
    decisions = np.where(
        y_prob > theta_high, "FRAUD",
        np.where(y_prob >= theta_low, "SUSPICIOUS", "LEGITIMATE")
    )

    n_total      = len(y_true)
    n_fraud_true = int(y_true.sum())

    # Distribusi prediksi
    n_fraud_pred      = int((decisions == "FRAUD").sum())
    n_suspicious_pred = int((decisions == "SUSPICIOUS").sum())
    n_legit_pred      = int((decisions == "LEGITIMATE").sum())

    # Dari fraud aktual, berapa yang masuk FRAUD vs SUSPICIOUS vs LEGIT
    fraud_mask = y_true == 1
    n_fraud_as_fraud      = int((decisions[fraud_mask] == "FRAUD").sum())
    n_fraud_as_suspicious = int((decisions[fraud_mask] == "SUSPICIOUS").sum())
    n_fraud_as_legit      = int((decisions[fraud_mask] == "LEGITIMATE").sum())

    # Coverage: berapa % fraud aktual yang tertangkap
    # (masuk FRAUD atau SUSPICIOUS)
    fraud_coverage = (n_fraud_as_fraud + n_fraud_as_suspicious) / n_fraud_true

    metrics = {
        "config":             config_name,
        "theta_low":          round(theta_low, 4),
        "theta_high":         round(theta_high, 4),
        "suspicious_zone":    round(theta_high - theta_low, 4),
        "n_total":            n_total,
        "n_fraud_true":       n_fraud_true,
        "pred_FRAUD":         n_fraud_pred,
        "pred_SUSPICIOUS":    n_suspicious_pred,
        "pred_LEGITIMATE":    n_legit_pred,
        "pct_FRAUD":          round(n_fraud_pred / n_total * 100, 3),
        "pct_SUSPICIOUS":     round(n_suspicious_pred / n_total * 100, 3),
        "pct_LEGITIMATE":     round(n_legit_pred / n_total * 100, 3),
        "fraud_as_FRAUD":     n_fraud_as_fraud,
        "fraud_as_SUSPICIOUS":n_fraud_as_suspicious,
        "fraud_as_LEGIT":     n_fraud_as_legit,
        "fraud_coverage":     round(fraud_coverage, 4),
    }

    return metrics


def plot_threshold_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    theta_low: float,
    theta_high: float,
    config_name: str
) -> None:

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Analisis Threshold Dua-Kelas → Tiga-Keputusan\n"
        f"XGBoost | {config_name}",
        fontweight="bold"
    )

    # === Panel kiri: PR Curve dengan zona ===
    ax = axes[0]
    ax.plot(recalls[:-1], precisions[:-1],
            color="#2196F3", linewidth=2, label="PR Curve")

    for thresh, label, color in [
        (theta_low,  f"θ_low={theta_low:.3f}",  "#FF9800"),
        (theta_high, f"θ_high={theta_high:.3f}", "#F44336"),
    ]:
        idx = np.argmin(np.abs(thresholds - thresh))
        ax.scatter(recalls[idx], precisions[idx],
                   s=100, color=color, zorder=5,
                   label=f"{label} | R={recalls[idx]:.3f}, P={precisions[idx]:.3f}")

    # Zona suspicious (antara dua threshold)
    idx_low  = np.argmin(np.abs(thresholds - theta_low))
    idx_high = np.argmin(np.abs(thresholds - theta_high))
    if idx_low > idx_high:
        idx_low, idx_high = idx_high, idx_low
    ax.fill_between(
        recalls[idx_low:idx_high+1],
        precisions[idx_low:idx_high+1],
        alpha=0.2, color="#FF9800", label="Zona SUSPICIOUS"
    )

    ax.axhline(y_true.mean(), color="gray", linestyle="--",
               linewidth=1, label=f"Baseline ({y_true.mean():.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

    # === Panel kanan: Distribusi skor fraud per kelas keputusan ===
    ax2 = axes[1]

    # Distribusi skor keseluruhan
    scores_legit = y_prob[y_true == 0]
    scores_fraud = y_prob[y_true == 1]

    ax2.hist(scores_legit, bins=100, alpha=0.5, color="#4CAF50",
             label="Legitimate (aktual)", density=True, log=True)
    ax2.hist(scores_fraud, bins=50, alpha=0.7, color="#F44336",
             label="Fraud (aktual)", density=True, log=True)

    # Garis theta
    ax2.axvline(theta_low, color="#FF9800", linewidth=2,
                linestyle="--", label=f"θ_low={theta_low:.3f}")
    ax2.axvline(theta_high, color="#F44336", linewidth=2,
                linestyle="--", label=f"θ_high={theta_high:.3f}")

    # Annotasi zona
    ax2.axvspan(0, theta_low, alpha=0.05, color="#4CAF50")
    ax2.axvspan(theta_low, theta_high, alpha=0.1, color="#FF9800")
    ax2.axvspan(theta_high, 1, alpha=0.05, color="#F44336")

    ax2.set_xlabel("Fraud Score (Probabilitas)")
    ax2.set_ylabel("Density (log scale)")
    ax2.set_title("Distribusi Skor per Kelas Aktual")
    ax2.legend(fontsize=8)
    ax2.set_xlim([0, 1])

    plt.tight_layout()
    save_path = FIGURES_DIR / f"threshold_analysis_{config_name}.png"
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    logger.info(f"Grafik threshold disimpan: {save_path}")

def plot_three_class_summary(
    three_class_stats: list,
) -> None:
    """
    Bar chart perbandingan distribusi tiga zona keputusan
    (FRAUD / SUSPICIOUS / LEGITIMATE) antar konfigurasi SMOTE.
    """
    df = pd.DataFrame(three_class_stats)

    configs = df["config"].tolist()
    x      = np.arange(len(configs))
    width  = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Distribusi Zona Keputusan Tiga Kelas — XGBoost (Test Set)",
        fontweight="bold"
    )

    # Panel kiri: persentase distribusi prediksi
    ax = axes[0]
    bars_fraud = ax.bar(x - width, df["pct_FRAUD"],
                        width, label="FRAUD", color="#F44336", alpha=0.85)
    bars_susp  = ax.bar(x, df["pct_SUSPICIOUS"],
                        width, label="SUSPICIOUS", color="#FF9800", alpha=0.85)
    bars_legit = ax.bar(x + width, df["pct_LEGITIMATE"],
                        width, label="LEGITIMATE", color="#4CAF50", alpha=0.85)

    for bars in [bars_fraud, bars_susp, bars_legit]:
        for bar in bars:
            h = bar.get_height()
            if h > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                        f"{h:.2f}%", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=15)
    ax.set_ylabel("Persentase dari Total Transaksi (%)")
    ax.set_title("Distribusi Prediksi per Konfigurasi")
    ax.legend(fontsize=9)
    ax.set_ylim([0, max(df["pct_LEGITIMATE"].max() * 1.1, 1)])

    # Panel kanan: fraud coverage per konfigurasi
    ax2 = axes[1]
    colors = ["#F44336", "#FF9800", "#2196F3", "#9C27B0"]
    bars_cov = ax2.bar(configs, df["fraud_coverage"] * 100,
                       color=colors[:len(configs)], alpha=0.85, edgecolor="white")

    for bar in bars_cov:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                 f"{h:.2f}%", ha="center", va="bottom",
                 fontweight="bold", fontsize=10)

    ax2.axhline(100, color="gray", linestyle="--",
                linewidth=1, label="Coverage 100%")
    ax2.set_ylabel("Fraud Coverage — FRAUD + SUSPICIOUS (%)")
    ax2.set_title("Persentase Fraud Aktual yang Tertangkap\n(masuk zona FRAUD atau SUSPICIOUS)")
    ax2.set_ylim([0, 110])
    ax2.legend(fontsize=9)

    plt.tight_layout()
    save_path = FIGURES_DIR / "three_class_summary.png"
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    logger.info(f"Grafik ringkasan tiga kelas disimpan: {save_path}")

def run_threshold_tuning() -> None:
    logger.info("=" * 60)
    logger.info("THRESHOLD TUNING — DUA THRESHOLD (theta_low & theta_high)")
    logger.info("=" * 60)

    # Load validation set
    val_df = pd.read_csv(DATA_SPLITS_DIR / "val.csv")
    test_df = pd.read_csv(DATA_SPLITS_DIR / "test.csv")

    with open(DATA_SPLITS_DIR / "split_info.json") as f:
        split_info = json.load(f)
    feature_cols = split_info["feature_columns"]

    X_val  = val_df[feature_cols]
    y_val  = val_df[TARGET_COLUMN].values
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COLUMN].values

    # Load thresholds yang sudah ada (theta_high dari Fase 2)
    with open(EVALUATION_DIR / "thresholds.json") as f:
        existing_thresholds = json.load(f)

    # Load best_config untuk tahu konfigurasi terbaik
    with open(EVALUATION_DIR / "best_config.json") as f:
        best_config = json.load(f)

    best_xgb_config = best_config["best_xgb_config"]

    # Hasil threshold dua-nilai
    dual_thresholds   = {}
    three_class_stats = []

    for config_name in SMOTE_CONFIGS.keys():
        logger.info(f"\n--- Konfigurasi: {config_name} ---")

        model_path = MODELS_DIR / f"xgb_{config_name}.joblib"
        if not model_path.exists():
            logger.warning(f"Model tidak ditemukan: {model_path}. Skip.")
            continue

        model = joblib.load(model_path)

        # Prediksi pada validation set
        y_prob_val  = model.predict_proba(X_val)[:, 1]
        y_prob_test = model.predict_proba(X_test)[:, 1]

        # theta_high = threshold optimal dari Fase 2
        key_xgb    = f"{config_name}_XGBoost"
        theta_high = existing_thresholds[key_xgb]["threshold"]

        # theta_low = threshold di mana recall >= TARGET_RECALL
        theta_low = find_theta_low(
            y_val, y_prob_val,
            target_recall=TARGET_RECALL,
            config_name=config_name
        )

        # Memastikan theta_low < theta_high (logika bisnis)
        if theta_low >= theta_high:
            logger.warning(
                f"[{config_name}] theta_low ({theta_low:.4f}) >= "
                f"theta_high ({theta_high:.4f}). "
            )
            theta_low = theta_high * 0.7
            logger.warning(
                f"[{config_name}] Fallback theta_low = theta_high * 0.7 = {theta_low:.4f}. "
            )

        logger.info(
            f"[{config_name}] theta_low={theta_low:.4f} | "
            f"theta_high={theta_high:.4f} | "
            f"zona SUSPICIOUS={theta_high - theta_low:.4f}"
        )

        # Menghitung metrik tiga kelas pada test set
        stats = compute_three_class_metrics(
            y_test, y_prob_test,
            theta_low, theta_high, config_name
        )
        three_class_stats.append(stats)

        logger.info(
            f"[{config_name}] Distribusi prediksi test set: "
            f"FRAUD={stats['pct_FRAUD']:.2f}% | "
            f"SUSPICIOUS={stats['pct_SUSPICIOUS']:.2f}% | "
            f"LEGITIMATE={stats['pct_LEGITIMATE']:.2f}%"
        )
        logger.info(
            f"[{config_name}] Dari {stats['n_fraud_true']} fraud aktual: "
            f"{stats['fraud_as_FRAUD']} → FRAUD | "
            f"{stats['fraud_as_SUSPICIOUS']} → SUSPICIOUS | "
            f"{stats['fraud_as_LEGIT']} → LEGITIMATE"
        )
        logger.info(
            f"[{config_name}] Fraud coverage "
            f"(FRAUD+SUSPICIOUS): {stats['fraud_coverage']*100:.2f}%"
        )

        # Visualisasi
        plot_threshold_analysis(
            y_test, y_prob_test,
            theta_low, theta_high, config_name
        )

        # Simpan ke dict
        dual_thresholds[config_name] = {
            "theta_low":       round(theta_low, 4),
            "theta_high":      round(theta_high, 4),
            "suspicious_zone": round(theta_high - theta_low, 4),
            "val_metrics": existing_thresholds[key_xgb],
        }

    # Simpan dual thresholds
    dual_path = EVALUATION_DIR / "dual_thresholds.json"
    with open(dual_path, "w") as f:
        json.dump(dual_thresholds, f, indent=2)
    logger.info(f"\nDual thresholds disimpan: {dual_path}")

    # Simpan statistik tiga kelas
    df_stats = pd.DataFrame(three_class_stats)
    stats_path = EVALUATION_DIR / "three_class_stats.csv"
    df_stats.to_csv(stats_path, index=False)
    logger.info(f"Statistik tiga kelas disimpan: {stats_path}")

    # Update best_config.json dengan threshold terpilih
    best_thresholds = dual_thresholds.get(best_xgb_config, {})
    with open(EVALUATION_DIR / "best_config.json") as f:
        best = json.load(f)

    best["best_theta_low"]  = best_thresholds.get("theta_low")
    best["best_theta_high"] = best_thresholds.get("theta_high")

    with open(EVALUATION_DIR / "best_config.json", "w") as f:
        json.dump(best, f, indent=2, default=str)
    logger.info(
        f"best_config.json diperbarui dengan "
        f"theta_low={best_thresholds.get('theta_low')} dan "
        f"theta_high={best_thresholds.get('theta_high')}"
    )

    # Visualisasi gabungan semua konfigurasi
    plot_three_class_summary(three_class_stats)

    # Ringkasan
    logger.info("\n=== RINGKASAN DUAL THRESHOLD (XGBoost) ===")
    logger.info(f"{'Config':<12} {'θ_low':>8} {'θ_high':>8} "
                f"{'Zona':>8} {'Coverage':>10}")
    logger.info("-" * 55)
    for cfg, vals in dual_thresholds.items():
        stats_row = next(
            (s for s in three_class_stats if s["config"] == cfg), {}
        )
        logger.info(
            f"{cfg:<12} {vals['theta_low']:>8.4f} "
            f"{vals['theta_high']:>8.4f} "
            f"{vals['suspicious_zone']:>8.4f} "
            f"{stats_row.get('fraud_coverage', 0)*100:>9.2f}%"
        )

    logger.info("=" * 60)
    logger.info("THRESHOLD TUNING SELESAI")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_threshold_tuning()