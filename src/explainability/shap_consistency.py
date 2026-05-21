import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from itertools import combinations

from config import (
    SHAP_DIR, FIGURES_DIR, EVALUATION_DIR,
    SHAP_TOP_K_FEATURES,
    SHAP_SPEARMAN_TARGET, SHAP_CV_TARGET, SHAP_JACCARD_TARGET,
    SMOTE_CONFIGS
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================
# METRIK 1 — SPEARMAN RANK CORRELATION
# =============================================================

def compute_spearman_matrix(
    global_importance_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Menghitung koefisien korelasi Spearman antar setiap pasang
    konfigurasi SMOTE berdasarkan peringkat global importance.

    Input: global_importance_df — rows=features, cols=configs
    Output: matriks simetris Spearman ρ
    """
    configs = list(global_importance_df.columns)
    matrix  = pd.DataFrame(
        np.eye(len(configs)),
        index=configs, columns=configs
    )

    for c1, c2 in combinations(configs, 2):
        rank1 = global_importance_df[c1].rank(ascending=False)
        rank2 = global_importance_df[c2].rank(ascending=False)
        rho, pval = spearmanr(rank1, rank2)
        matrix.loc[c1, c2] = round(float(rho), 4)
        matrix.loc[c2, c1] = round(float(rho), 4)
        logger.info(
            f"Spearman ρ [{c1} vs {c2}]: "
            f"{rho:.4f} (p={pval:.4e}) — "
            f"{'PASS' if rho >= SHAP_SPEARMAN_TARGET else 'FAIL'}"
        )

    return matrix


# =============================================================
# METRIK 2 — COEFFICIENT OF VARIATION (CV)
# =============================================================

def compute_cv_per_feature(
    global_importance_df: pd.DataFrame
) -> pd.Series:
    """
    Menghitung CV = std / mean dari global importance
    untuk setiap fitur across semua konfigurasi SMOTE.

    Nilai CV rendah → besaran kontribusi fitur stabil.
    Target: CV < 0.30 untuk semua fitur utama.
    """
    cv = global_importance_df.std(axis=1) / global_importance_df.mean(axis=1)
    cv = cv.fillna(0).sort_values(ascending=True)

    pct_pass = (cv < SHAP_CV_TARGET).mean() * 100
    logger.info(
        f"CV analysis: {pct_pass:.1f}% fitur memenuhi target "
        f"CV < {SHAP_CV_TARGET}"
    )
    for feat, cv_val in cv.items():
        status = "PASS" if cv_val < SHAP_CV_TARGET else "FAIL"
        logger.info(f"  CV[{feat}] = {cv_val:.4f} — {status}")

    return cv


# =============================================================
# METRIK 3 — JACCARD SIMILARITY (TOP-K OVERLAP)
# =============================================================

def compute_jaccard_matrix(
    global_importance_df: pd.DataFrame,
    k: int = None
) -> pd.DataFrame:
    """
    Menghitung Jaccard similarity antara himpunan top-k fitur
    dari setiap pasang konfigurasi SMOTE.

    J(A,B) = |A ∩ B| / |A ∪ B|, k = SHAP_TOP_K_FEATURES (5)
    """
    if k is None:
        k = SHAP_TOP_K_FEATURES

    configs = list(global_importance_df.columns)
    matrix  = pd.DataFrame(
        np.eye(len(configs)),
        index=configs, columns=configs
    )

    # Top-k fitur per konfigurasi
    topk_sets = {
        cfg: set(global_importance_df[cfg].nlargest(k).index)
        for cfg in configs
    }

    for c1, c2 in combinations(configs, 2):
        A = topk_sets[c1]
        B = topk_sets[c2]
        intersection = len(A & B)
        union        = len(A | B)
        jaccard      = intersection / union if union > 0 else 0.0

        matrix.loc[c1, c2] = round(jaccard, 4)
        matrix.loc[c2, c1] = round(jaccard, 4)
        logger.info(
            f"Jaccard [{c1} vs {c2}]: {jaccard:.4f} "
            f"(|A∩B|={intersection}, |A∪B|={union}) — "
            f"{'PASS' if jaccard >= SHAP_JACCARD_TARGET else 'FAIL'}"
        )
        logger.info(f"  Top-{k} {c1}: {sorted(A)}")
        logger.info(f"  Top-{k} {c2}: {sorted(B)}")

    return matrix


# =============================================================
# VISUALISASI
# =============================================================

def plot_spearman_heatmap(spearman_matrix: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    mask = np.zeros_like(spearman_matrix.values, dtype=bool)
    np.fill_diagonal(mask, True)  

    sns.heatmap(
        spearman_matrix,
        annot=True, fmt=".4f",
        cmap="YlOrRd",
        vmin=0.0, vmax=1.0,
        mask=mask,
        ax=ax,
        linewidths=0.5,
        annot_kws={"size": 11},
        cbar_kws={"label": "Spearman ρ", "shrink": 0.8}
    )
    # Menambahkan diagonal = 1.0 manual
    for i in range(len(spearman_matrix)):
        ax.text(i + 0.5, i + 0.5, "1.0000",
                ha="center", va="center",
                fontsize=11, color="gray")

    ax.set_title(
        f"Feature Ranking Consistency — Spearman ρ\n"
        f"(Target: ρ ≥ {SHAP_SPEARMAN_TARGET})",
        fontweight="bold", fontsize=12
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()
    save_path = FIGURES_DIR / "shap_spearman_heatmap.png"
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    logger.info(f"Spearman heatmap disimpan: {save_path}")


def plot_cv_barplot(cv_series: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    colors  = [
        "#4CAF50" if v < SHAP_CV_TARGET else "#F44336"
        for v in cv_series.values
    ]
    bars = ax.bar(
        range(len(cv_series)), cv_series.values,
        color=colors, edgecolor="white", width=0.6
    )
    ax.axhline(
        SHAP_CV_TARGET, color="black",
        linestyle="--", linewidth=1.5,
        label=f"Target CV < {SHAP_CV_TARGET}"
    )
    ax.set_xticks(range(len(cv_series)))
    ax.set_xticklabels(cv_series.index, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Coefficient of Variation (CV)", fontsize=11)
    ax.set_title(
        "SHAP Magnitude Consistency — CV per Fitur\n"
        "(Hijau = memenuhi target, Merah = tidak memenuhi)",
        fontweight="bold", fontsize=12
    )
    ax.legend(fontsize=10)

    for i, (feat, val) in enumerate(cv_series.items()):
        ax.text(i, val + 0.005, f"{val:.4f}",
                ha="center", va="bottom", fontsize=7.5)

    plt.tight_layout()
    save_path = FIGURES_DIR / "shap_cv_barplot.png"
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    logger.info(f"CV barplot disimpan: {save_path}")


def plot_jaccard_heatmap(jaccard_matrix: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    mask = np.zeros_like(jaccard_matrix.values, dtype=bool)
    np.fill_diagonal(mask, True)

    sns.heatmap(
        jaccard_matrix,
        annot=True, fmt=".4f",
        cmap="Blues",
        vmin=0.0, vmax=1.0,
        mask=mask,
        ax=ax,
        linewidths=0.5,
        annot_kws={"size": 11},
        cbar_kws={"label": f"Jaccard (top-{SHAP_TOP_K_FEATURES})", "shrink": 0.8}
    )
    for i in range(len(jaccard_matrix)):
        ax.text(i + 0.5, i + 0.5, "1.0000",
                ha="center", va="center",
                fontsize=11, color="gray")

    ax.set_title(
        f"Top-{SHAP_TOP_K_FEATURES} Feature Overlap — Jaccard Similarity\n"
        f"(Target: J ≥ {SHAP_JACCARD_TARGET})",
        fontweight="bold", fontsize=12
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()
    save_path = FIGURES_DIR / "shap_jaccard_heatmap.png"
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    logger.info(f"Jaccard heatmap disimpan: {save_path}")


# =============================================================
# ANALISIS KONSISTENSI PENUH
# =============================================================

def run_consistency_analysis(
    global_importance_df: pd.DataFrame
) -> dict:

    logger.info("="*55)
    logger.info("ANALISIS KONSISTENSI INTERPRETABILITAS (RQ2)")
    logger.info("="*55)

    # 1. Spearman
    spearman_matrix = compute_spearman_matrix(global_importance_df)
    plot_spearman_heatmap(spearman_matrix)

    # 2. CV
    cv_series = compute_cv_per_feature(global_importance_df)
    plot_cv_barplot(cv_series)

    # 3. Jaccard
    jaccard_matrix = compute_jaccard_matrix(global_importance_df)
    plot_jaccard_heatmap(jaccard_matrix)

    # Ringkasan per konfigurasi (vs baseline)
    configs      = list(global_importance_df.columns)
    baseline     = "baseline"
    summary_rows = []

    for cfg in configs:
        if cfg == baseline:
            rho_vs_base     = 1.0
            jaccard_vs_base = 1.0
        else:
            rho_vs_base     = float(spearman_matrix.loc[baseline, cfg])
            jaccard_vs_base = float(jaccard_matrix.loc[baseline, cfg])

        # CV hanya diambil untuk top-k fitur (paling relevan)
        topk_features  = global_importance_df[cfg].nlargest(
            SHAP_TOP_K_FEATURES
        ).index.tolist()
        cv_topk        = cv_series[topk_features].mean()
        cv_all         = cv_series.mean()

        meets_spearman = rho_vs_base >= SHAP_SPEARMAN_TARGET
        meets_cv       = cv_topk < SHAP_CV_TARGET
        meets_jaccard  = jaccard_vs_base >= SHAP_JACCARD_TARGET
        meets_all      = meets_spearman and meets_cv and meets_jaccard

        summary_rows.append({
            "config":              cfg,
            "spearman_vs_baseline": round(rho_vs_base, 4),
            "meets_spearman":      meets_spearman,
            "cv_top5_features":    round(float(cv_topk), 4),
            "cv_all_features":     round(float(cv_all), 4),
            "meets_cv":            meets_cv,
            "jaccard_vs_baseline": round(jaccard_vs_base, 4),
            "meets_jaccard":       meets_jaccard,
            "meets_all_consistency": meets_all
        })

        logger.info(
            f"[{cfg}] Spearman vs baseline={rho_vs_base:.4f} "
            f"({'PASS' if meets_spearman else 'FAIL'}) | "
            f"CV top-5={cv_topk:.4f} "
            f"({'PASS' if meets_cv else 'FAIL'}) | "
            f"Jaccard vs baseline={jaccard_vs_base:.4f} "
            f"({'PASS' if meets_jaccard else 'FAIL'})"
        )

    # Simpan matriks
    spearman_path = SHAP_DIR / "spearman_matrix.csv"
    jaccard_path  = SHAP_DIR / "jaccard_matrix.csv"
    cv_path       = SHAP_DIR / "cv_per_feature.csv"

    spearman_matrix.to_csv(spearman_path)
    jaccard_matrix.to_csv(jaccard_path)
    cv_series.to_csv(cv_path, header=["cv"])

    logger.info(f"Matriks Spearman disimpan: {spearman_path}")
    logger.info(f"Matriks Jaccard disimpan : {jaccard_path}")
    logger.info(f"CV per fitur disimpan    : {cv_path}")

    results = {
        "metrics": {
            "spearman_matrix": spearman_matrix.to_dict(),
            "jaccard_matrix":  jaccard_matrix.to_dict(),
            "cv_per_feature":  cv_series.to_dict(),
        },
        "summary_per_config": summary_rows,
        "targets": {
            "spearman": SHAP_SPEARMAN_TARGET,
            "cv":       SHAP_CV_TARGET,
            "jaccard":  SHAP_JACCARD_TARGET,
            "top_k":    SHAP_TOP_K_FEATURES
        }
    }

    # Simpan consistency_metrics.json
    metrics_path = SHAP_DIR / "consistency_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Consistency metrics disimpan: {metrics_path}")

    return results