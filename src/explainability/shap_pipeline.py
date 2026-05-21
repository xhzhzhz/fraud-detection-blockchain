import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import joblib
import numpy as np
import pandas as pd

from config import (
    DATA_SPLITS_DIR, EVALUATION_DIR, SHAP_DIR,
    MODELS_DIR, TARGET_COLUMN, SMOTE_CONFIGS
)
from src.explainability.shap_computer    import (
    load_test_set, compute_all_configs, verify_additivity
)
from src.explainability.shap_global      import run_global_analysis
from src.explainability.shap_local       import generate_sample_explanations
from src.explainability.shap_consistency import run_consistency_analysis
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_dual_thresholds() -> dict:

    path = EVALUATION_DIR / "dual_thresholds.json"
    if not path.exists():
        raise FileNotFoundError(
            f"dual_thresholds.json tidak ditemukan: {path}\n"
            "Pastikan threshold_tuner.py sudah dijalankan pada Fase 2."
        )
    with open(path) as f:
        return json.load(f)


def determine_optimal_config(
    consistency_results: dict,
    all_metrics_path: Path
) -> dict:
    """
    Menentukan konfigurasi SMOTE optimal berdasarkan kombinasi:
    - Kinerja prediksi (PR-AUC, F1, Recall) dari Fase 2
    - Konsistensi interpretabilitas (Spearman, CV, Jaccard) dari Fase 3
    """
    # Load metrik Fase 2
    df_metrics = pd.read_csv(all_metrics_path)
    df_xgb_test = df_metrics[
        (df_metrics["model"] == "XGBoost") &
        (df_metrics["split"] == "test")
    ].copy()

    # Map konsistensi ke konfigurasi
    consistency_map = {
        row["config"]: row
        for row in consistency_results["summary_per_config"]
    }

    # Menggabungkan
    candidates = []
    for _, perf_row in df_xgb_test.iterrows():
        cfg  = perf_row["config"]
        cons = consistency_map.get(cfg, {})

        candidates.append({
            "config":               cfg,
            "pr_auc":               perf_row["pr_auc"],
            "f1":                   perf_row["f1"],
            "recall":               perf_row["recall"],
            "meets_perf_targets":   bool(perf_row["meets_all_targets"]),
            "meets_consistency":    cons.get("meets_all_consistency", False),
            "spearman_vs_baseline": cons.get("spearman_vs_baseline", 0),
            "cv_top5":              cons.get("cv_top5_features", 999),
            "jaccard_vs_baseline":  cons.get("jaccard_vs_baseline", 0),
        })

    # Kriteria seleksi:
    # 1. Memenuhi target kinerja dan konsistensi: memilih PR-AUC tertinggi
    # 2. Jika tidak ada yang memenuhi keduanya: yang memenuhi kinerja
    #    dengan Spearman+Jaccard tertinggi
    fully_qualified = [
        c for c in candidates
        if c["meets_perf_targets"] and c["meets_consistency"]
    ]

    if fully_qualified:
        best = max(fully_qualified, key=lambda x: x["pr_auc"])
        selection_reason = (
            f"Memenuhi target kinerja dan konsistensi SHAP. "
            f"Dipilih berdasarkan PR-AUC tertinggi ({best['pr_auc']:.4f})."
        )
    else:
        perf_qualified = [c for c in candidates if c["meets_perf_targets"]]
        if perf_qualified:
            best = max(
                perf_qualified,
                key=lambda x: (x["spearman_vs_baseline"] + x["jaccard_vs_baseline"])
            )
            selection_reason = (
                "Tidak ada konfigurasi yang memenuhi semua target kinerja "
                "DAN konsistensi sekaligus. Dipilih konfigurasi yang "
                "memenuhi target kinerja dengan konsistensi SHAP terbaik."
            )
        else:
            best = max(candidates, key=lambda x: x["pr_auc"])
            selection_reason = (
                "Tidak ada konfigurasi yang memenuhi semua target. "
                "Dipilih berdasarkan PR-AUC tertinggi."
            )

    logger.info("="*55)
    logger.info("KEPUTUSAN KONFIGURASI OPTIMAL (RQ2)")
    logger.info("="*55)
    logger.info(f"Konfigurasi terpilih : {best['config']}")
    logger.info(f"Alasan               : {selection_reason}")
    logger.info(
        f"Kinerja test         : "
        f"PR-AUC={best['pr_auc']:.4f} | "
        f"F1={best['f1']:.4f} | "
        f"Recall={best['recall']:.4f}"
    )
    logger.info(
        f"Konsistensi SHAP     : "
        f"Spearman={best['spearman_vs_baseline']:.4f} | "
        f"CV_top5={best['cv_top5']:.4f} | "
        f"Jaccard={best['jaccard_vs_baseline']:.4f}"
    )

    return {
        "optimal_config":     best["config"],
        "selection_reason":   selection_reason,
        "performance":        {k: best[k] for k in ["pr_auc", "f1", "recall"]},
        "consistency":        {
            "spearman_vs_baseline": best["spearman_vs_baseline"],
            "cv_top5":              best["cv_top5"],
            "jaccard_vs_baseline":  best["jaccard_vs_baseline"]
        },
        "all_candidates":     candidates
    }


def run_shap_pipeline(force_recompute: bool = False) -> None:
    logger.info("="*60)
    logger.info("FASE 3: PIPELINE SHAP DIMULAI")
    logger.info("="*60)

    # 1. Load test set
    X_test, y_test, feature_cols = load_test_set()

    # 2. Komputasi SHAP semua konfigurasi
    shap_results = compute_all_configs(X_test, force_recompute)

    # 3. Verifikasi additivity untuk setiap konfigurasi
    additivity_results = []
    for config_name, data in shap_results.items():
        model_path = MODELS_DIR / f"xgb_{config_name}.joblib"
        model      = joblib.load(model_path)
        y_prob     = model.predict_proba(X_test)[:, 1]

        result = verify_additivity(
            data["shap_values"], data["base_value"],
            y_prob, config_name
        )
        additivity_results.append(result)

    # Simpan hasil additivity
    add_path = SHAP_DIR / "additivity_check.json"
    with open(add_path, "w") as f:
        json.dump(additivity_results, f, indent=2)
    logger.info(f"Additivity check disimpan: {add_path}")

    # 4. Analisis global interpretabilitas
    global_importance_df = run_global_analysis(shap_results, X_test)

    # 5. Generate local explanations (sample fraud)
    dual_thresholds = load_dual_thresholds()
    generate_sample_explanations(
        shap_results, X_test, y_test,
        dual_thresholds, n_fraud_samples=30
    )

    # 6. Analisis konsistensi 3 metrik
    consistency_results = run_consistency_analysis(global_importance_df)

    # 7. Menentukan konfigurasi optimal berdasarkan kinerja + konsistensi
    optimal = determine_optimal_config(
        consistency_results,
        EVALUATION_DIR / "all_metrics.csv"
    )

    # 8. Simpan ringkasan akhir konsistensi
    report = {
        "phase":           "Fase 3: SHAP Explainability",
        "optimal_config":  optimal,
        "consistency":     consistency_results["summary_per_config"],
        "targets":         consistency_results["targets"],
        "additivity":      additivity_results
    }
    report_path = SHAP_DIR / "shap_consistency_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Laporan konsistensi disimpan: {report_path}")

    # 9. Update best_config_final.json
    best_config_path = EVALUATION_DIR / "best_config.json"
    with open(best_config_path) as f:
        best_config = json.load(f)

    best_config["shap_analysis"] = {
        "optimal_config_shap":  optimal["optimal_config"],
        "selection_reason":     optimal["selection_reason"],
        "performance_metrics":  optimal["performance"],
        "consistency_metrics":  optimal["consistency"],
        "all_candidates":       optimal["all_candidates"]
    }

    final_path = EVALUATION_DIR / "best_config_final.json"
    with open(final_path, "w") as f:
        json.dump(best_config, f, indent=2, default=str)
    logger.info(f"best_config_final.json disimpan: {final_path}")

    # 10. Ringkasan akhir ke log
    logger.info("="*60)
    logger.info("FASE 3: PIPELINE SHAP SELESAI")
    logger.info(
        f"Konfigurasi optimal : {optimal['optimal_config']}"
    )
    logger.info(
        f"Laporan konsistensi : {report_path}"
    )
    logger.info(
        f"Local explanations  : {SHAP_DIR / 'local_explanations_sample.json'}"
    )
    logger.info("="*60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Pipeline SHAP Fase 3"
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Paksa rekomputasi SHAP meskipun .npy sudah ada"
    )
    args = parser.parse_args()
    run_shap_pipeline(force_recompute=args.force_recompute)