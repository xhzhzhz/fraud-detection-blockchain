import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import shap

from config import (
    SHAP_DIR, FIGURES_DIR, SHAP_TOP_K_FEATURES,
    SMOTE_CONFIGS, RANDOM_STATE
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Subsample untuk beeswarm plot 
# (nilai SHAP tetap dihitung dari full test set
# subsample hanya untuk visualisasi)
BEESWARM_SAMPLE = 5_000


def compute_global_importance(
    shap_values: np.ndarray,
    feature_names: list,
    config_name: str
) -> pd.Series:
    """
    Menghitung rata-rata nilai absolut SHAP per fitur.
    Formula: mean_phi_i = (1/N) * sum_j |phi_i^(j)|
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = pd.Series(mean_abs, index=feature_names, name=config_name)
    importance = importance.sort_values(ascending=False)
    return importance


def plot_beeswarm(
    shap_values: np.ndarray,
    X_test: pd.DataFrame,
    config_name: str,
    top_n: int = 15
) -> None:
    """
    Beeswarm plot: menunjukkan distribusi kontribusi SHAP
    per fitur secara bersamaan (arah dan besaran).

    Subsample diambil secara acak dari test set untuk
    menjaga efisiensi rendering tanpa mengorbankan representasi.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    n   = min(BEESWARM_SAMPLE, len(X_test))
    idx = rng.choice(len(X_test), size=n, replace=False)

    shap_sample = shap_values[idx]
    X_sample    = X_test.iloc[idx].reset_index(drop=True)

    # Memilih top_n fitur berdasarkan mean |SHAP| dari sample
    mean_abs   = np.abs(shap_sample).mean(axis=0)
    top_idx    = np.argsort(mean_abs)[::-1][:top_n]
    top_names  = [X_test.columns[i] for i in top_idx]

    shap_top   = shap_sample[:, top_idx]
    X_top      = X_sample[top_names]

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        shap_top, X_top,
        feature_names=top_names,
        plot_type="dot",
        show=False,
        plot_size=None,
        color_bar_label="Feature Value"
    )
    ax = plt.gca()
    ax.set_title(
        f"SHAP Beeswarm Plot — {config_name}\n"
        f"(Top {top_n} fitur | subsample n={n:,})",
        fontweight="bold", fontsize=12
    )
    plt.tight_layout()

    save_path = FIGURES_DIR / f"shap_beeswarm_{config_name}.png"
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    logger.info(f"[{config_name}] Beeswarm plot disimpan: {save_path}")


def plot_global_comparison(
    global_importance_df: pd.DataFrame,
    top_n: int = 15
) -> None:
    """
    Bar chart perbandingan global importance (mean |SHAP|)
    antar konfigurasi SMOTE, untuk top_n fitur berdasarkan
    rata-rata kepentingan dari semua konfigurasi.
    """
    # Memilih top_n fitur berdasarkan mean across all configs
    mean_across = global_importance_df.mean(axis=1)
    top_features = mean_across.nlargest(top_n).index.tolist()

    df_plot = global_importance_df.loc[top_features]

    n_configs  = len(df_plot.columns)
    bar_width  = 0.18
    x          = np.arange(len(top_features))
    colors     = plt.cm.Set2(np.linspace(0, 0.8, n_configs))

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, (config_name, col_data) in enumerate(df_plot.items()):
        offset = (i - n_configs/2 + 0.5) * bar_width
        bars   = ax.bar(
            x + offset, col_data.values,
            width=bar_width, label=config_name,
            color=colors[i], alpha=0.87, edgecolor="white"
        )

    ax.set_xticks(x)
    ax.set_xticklabels(top_features, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Mean |SHAP Value|", fontsize=11)
    ax.set_title(
        f"Global Feature Importance — Perbandingan Lintas Konfigurasi SMOTE\n"
        f"(Top {top_n} fitur | Mean |SHAP| pada Test Set)",
        fontweight="bold", fontsize=12
    )
    ax.legend(title="Konfigurasi", fontsize=9, title_fontsize=9)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:.4f}")
    )
    plt.tight_layout()

    save_path = FIGURES_DIR / "shap_global_comparison.png"
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    logger.info(f"Global comparison plot disimpan: {save_path}")


def run_global_analysis(
    shap_results: dict,
    X_test: pd.DataFrame
) -> pd.DataFrame:
    """
    Menjalankan seluruh analisis global dan menghasilkan
    DataFrame importance untuk seluruh konfigurasi.
    """
    feature_names = list(X_test.columns)
    importance_dict = {}

    for config_name, data in shap_results.items():
        shap_values = data["shap_values"]

        # Menghitung global importance
        importance = compute_global_importance(
            shap_values, feature_names, config_name
        )
        importance_dict[config_name] = importance

        # Beeswarm plot per konfigurasi
        plot_beeswarm(shap_values, X_test, config_name)

        # Log top-5
        top5 = importance.head(SHAP_TOP_K_FEATURES)
        logger.info(
            f"[{config_name}] Top-5 global features: "
            + " | ".join(f"{f}={v:.5f}" for f, v in top5.items())
        )

    # Menggabungkan ke DataFrame
    global_importance_df = pd.DataFrame(importance_dict)

    # Simpan ke CSV
    save_path = SHAP_DIR / "global_importance_all.csv"
    global_importance_df.to_csv(save_path)
    logger.info(f"Global importance disimpan: {save_path}")

    # Visualisasi perbandingan
    plot_global_comparison(global_importance_df)

    return global_importance_df