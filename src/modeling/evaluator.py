import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json

from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, precision_recall_curve,
    roc_curve, classification_report
)

from config import (
    FIGURES_DIR, EVALUATION_DIR,
    TARGET_RECALL, TARGET_PRECISION,
    TARGET_F1, TARGET_ROC_AUC, TARGET_PR_AUC
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    config_name: str,
    model_name: str
) -> float:
    """
    Menentukan threshold optimal dari validation set
    berdasarkan titik yang memaksimalkan F1-Score
    pada Precision-Recall Curve.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    # Menghitung F1 untuk setiap threshold
    # Menghindari pembagian dengan nol
    f1_scores = np.where(
        (precisions[:-1] + recalls[:-1]) > 0,
        2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1]),
        0
    )

    best_idx       = np.argmax(f1_scores)
    best_threshold = float(thresholds[best_idx])
    best_f1        = float(f1_scores[best_idx])
    best_precision = float(precisions[best_idx])
    best_recall    = float(recalls[best_idx])

    logger.info(
        f"[{config_name}|{model_name}] Threshold optimal = {best_threshold:.4f} "
        f"| F1={best_f1:.4f} | P={best_precision:.4f} | R={best_recall:.4f}"
    )

    return best_threshold

def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    config_name: str,
    model_name: str,
    split_name: str = "test"
) -> dict:
    """
    Menghitung seluruh metrik evaluasi pada threshold tertentu.
    """
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "config":    config_name,
        "model":     model_name,
        "split":     split_name,
        "threshold": round(threshold, 4),
        "precision": round(precision_score(y_true, y_pred,
                                           zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred,
                                        zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred,
                                    zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_true, y_prob), 4),
        "pr_auc":    round(average_precision_score(y_true, y_prob), 4),
        "n_samples": int(len(y_true)),
        "n_fraud":   int(y_true.sum()),
        "n_pred_fraud": int(y_pred.sum()),
        "tp": int(((y_pred == 1) & (y_true == 1)).sum()),
        "fp": int(((y_pred == 1) & (y_true == 0)).sum()),
        "fn": int(((y_pred == 0) & (y_true == 1)).sum()),
        "tn": int(((y_pred == 0) & (y_true == 0)).sum()),
    }

    # Flag apakah memenuhi target
    meets_targets = (
        metrics["recall"]  >= TARGET_RECALL    and
        metrics["precision"] >= TARGET_PRECISION and
        metrics["f1"]      >= TARGET_F1        and
        metrics["roc_auc"] >= TARGET_ROC_AUC   and
        metrics["pr_auc"]  >= TARGET_PR_AUC
    )
    metrics["meets_all_targets"] = meets_targets

    return metrics


def meets_targets(metrics: dict) -> bool:
    return metrics.get("meets_all_targets", False)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    config_name: str,
    model_name: str
) -> None:
    """Membuat dan menyimpan confusion matrix."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)

    labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i,
                    f"{labels[i][j]}\n{cm[i, j]:,}",
                    ha="center", va="center",
                    fontsize=13, fontweight="bold",
                    color="white" if cm[i, j] > cm.max()/2 else "black")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred: Legitimate", "Pred: Fraud"])
    ax.set_yticklabels(["Actual: Legitimate", "Actual: Fraud"])
    ax.set_title(
        f"Confusion Matrix\n{model_name} | {config_name} "
        f"| threshold={threshold:.3f}",
        fontweight="bold"
    )
    plt.tight_layout()

    save_path = FIGURES_DIR / f"cm_{config_name}_{model_name}.png"
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    logger.info(f"Confusion matrix disimpan: {save_path}")


def plot_pr_roc_curves(
    results_per_config: dict,
    y_true_test: np.ndarray
) -> None:
    """
    Membuat grafik PR Curve dan ROC Curve gabungan
    untuk semua konfigurasi dan model pada test set.
    Satu grafik per algoritma, overlay semua konfigurasi SMOTE.
    """
    for model_name in ["XGBoost", "RandomForest"]:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            f"{model_name} — PR Curve & ROC Curve (Test Set)",
            fontweight="bold", fontsize=13
        )

        colors = plt.cm.Set1(np.linspace(0, 0.8, 4))

        for idx, (config_name, data) in enumerate(results_per_config.items()):
            if model_name not in data:
                continue

            y_prob = data[model_name]["y_prob_test"]
            metrics = data[model_name]["test_metrics"]
            label_base = (
                f"{config_name} "
                f"(PR-AUC={metrics['pr_auc']:.3f}, "
                f"ROC-AUC={metrics['roc_auc']:.3f})"
            )
            color = colors[idx]

            # PR Curve
            prec, rec, _ = precision_recall_curve(y_true_test, y_prob)
            axes[0].plot(rec, prec, color=color,
                         linewidth=1.8, label=label_base)

            # ROC Curve
            fpr, tpr, _ = roc_curve(y_true_test, y_prob)
            axes[1].plot(fpr, tpr, color=color,
                         linewidth=1.8, label=label_base)

        # Baseline PR 
        baseline_pr = y_true_test.mean()
        axes[0].axhline(baseline_pr, color="gray",
                        linestyle="--", linewidth=1,
                        label=f"No-skill baseline ({baseline_pr:.4f})")

        axes[0].set_xlabel("Recall")
        axes[0].set_ylabel("Precision")
        axes[0].set_title("Precision-Recall Curve")
        axes[0].legend(fontsize=8, loc="upper right")
        axes[0].set_xlim([0, 1])
        axes[0].set_ylim([0, 1.05])

        axes[1].plot([0, 1], [0, 1], "k--", linewidth=1)
        axes[1].set_xlabel("False Positive Rate")
        axes[1].set_ylabel("True Positive Rate")
        axes[1].set_title("ROC Curve")
        axes[1].legend(fontsize=8, loc="lower right")
        axes[1].set_xlim([0, 1])
        axes[1].set_ylim([0, 1.05])

        plt.tight_layout()
        save_path = FIGURES_DIR / f"pr_roc_{model_name}.png"
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()
        logger.info(f"Grafik PR/ROC disimpan: {save_path}")


def plot_metrics_comparison(all_metrics: pd.DataFrame) -> None:
    """
    Bar chart perbandingan metrik utama
    semua konfigurasi dan model pada test set.
    """
    df_test = all_metrics[all_metrics["split"] == "test"].copy()
    df_test["label"] = df_test["config"] + "\n" + df_test["model"]

    metrics_to_plot = ["recall", "precision", "f1", "pr_auc", "roc_auc"]
    targets = {
        "recall": TARGET_RECALL,
        "precision": TARGET_PRECISION,
        "f1": TARGET_F1,
        "pr_auc": TARGET_PR_AUC,
        "roc_auc": TARGET_ROC_AUC,
    }

    fig, axes = plt.subplots(1, 5, figsize=(20, 5), sharey=False)
    fig.suptitle(
        "Perbandingan Metrik Evaluasi : Test Set\n",
        fontweight="bold"
    )

    colors_model = {"XGBoost": "#2196F3", "RandomForest": "#FF9800"}

    for ax, metric in zip(axes, metrics_to_plot):
        for i, (_, row) in enumerate(df_test.iterrows()):
            color = colors_model.get(row["model"], "#999")
            bar = ax.bar(i, row[metric], color=color,
                         alpha=0.85, width=0.6, edgecolor="white")
            ax.text(i, row[metric] + 0.01,
                    f"{row[metric]:.3f}",
                    ha="center", va="bottom", fontsize=7)

        ax.axhline(targets[metric], color="red",
                   linestyle="--", linewidth=1.2)
        ax.set_title(metric.upper().replace("_", "-"), fontweight="bold")
        ax.set_xticks(range(len(df_test)))
        ax.set_xticklabels(df_test["label"].tolist(),
                            rotation=45, ha="right", fontsize=7)
        ax.set_ylim([0, 1.1])

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2196F3", label="XGBoost"),
        Patch(facecolor="#FF9800", label="RandomForest"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=2, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()
    save_path = FIGURES_DIR / "metrics_comparison_all.png"
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    logger.info(f"Grafik perbandingan metrik disimpan: {save_path}")