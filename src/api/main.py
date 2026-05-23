import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import time
import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import uvicorn

from config import API_HOST, API_PORT, EVALUATION_DIR
from src.api.schemas import (
    TransactionInput, DetectionResponse, HealthResponse,
    AuditStatsResponse, SHAPExplanation, SHAPFeatureContribution,
    AuditInfo
)
from src.api.ml_service       import load_pipeline, predict
from src.api.shap_service     import load_explainer, explain
from src.api.decision_service import get_decision, get_theta_optimal
from src.blockchain.web3_service  import Web3Service
from src.blockchain.audit_service import should_audit, process_audit
from src.api.background_worker    import retry_pending_records
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ================================================================
# LIFESPAN: inisialisasi saat startup, cleanup saat shutdown
# ================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preload model dan koneksi blockchain saat API start."""
    logger.info("API startup: memuat model dan koneksi blockchain...")

    try:
        pipeline  = load_pipeline()
        _         = load_explainer()
        app.state.pipeline = pipeline
        logger.info(
            f"Model loaded: {pipeline['model_name']} | "
            f"config={pipeline['config_name']}"
        )
    except Exception as e:
        logger.error(f"Gagal memuat model: {e}")
        raise

    try:
        app.state.web3 = Web3Service()
        logger.info("Blockchain service terhubung.")
    except Exception as e:
        logger.warning(
            f"Blockchain service gagal terhubung: {e}\n"
            "API tetap berjalan tanpa blockchain (mode degraded)."
        )
        app.state.web3 = None

    logger.info("API siap menerima request.")
    yield

    logger.info("API shutdown.")


# ================================================================
# APP INSTANCE
# ================================================================

app = FastAPI(
    title       = "Fraud Detection API: XGBoost + SHAP + Blockchain",
    description = (
        "Sistem deteksi fraud transaksi digital dengan explainability SHAP "
        "dan audit trail blockchain. Sesuai POJK 12/2024."
    ),
    version     = "1.0.0",
    lifespan    = lifespan
)


# ================================================================
# HELPER: background task untuk blockchain audit
# ================================================================

def run_blockchain_audit(
    transaction_id:      str,
    transaction_details: dict,
    fraud_score:         float,
    decision:            str,
    theta_high:          float,
    theta_optimal:       float,
    shap_explanation:    dict,
    web3_service,
    audit_reason:        str
):
    """
    Fungsi ini berjalan di background thread setelah respons dikirim.
    Tidak memblokir latency respons API.
    """
    try:
        pipeline      = load_pipeline()
        model_version = pipeline["model_version"]
        validator_addr = web3_service.account.address if web3_service else "N/A"

        process_audit(
            transaction_id      = transaction_id,
            transaction_details = transaction_details,
            fraud_score         = fraud_score,
            decision            = decision,
            theta_high          = theta_high,
            theta_optimal       = theta_optimal,
            shap_explanation    = shap_explanation,
            model_version       = model_version,
            web3_service        = web3_service,
            validator_address   = validator_addr,
            audit_reason        = audit_reason
        )
    except Exception as e:
        logger.error(f"Background audit gagal untuk {transaction_id}: {e}")


# ================================================================
# ENDPOINTS
# ================================================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Memeriksa status sistem: model dan koneksi blockchain.
    Digunakan untuk monitoring dan verifikasi deployment.
    """
    pipeline      = load_pipeline()
    blockchain_ok = app.state.web3 is not None

    with open(EVALUATION_DIR / "deployment_info.json") as f:
        deploy_info = json.load(f)

    return HealthResponse(
        status               = "healthy" if blockchain_ok else "degraded",
        model_loaded         = True,
        blockchain_connected = blockchain_ok,
        contract_address     = deploy_info.get("contractAddress", "N/A"),
        model_version        = pipeline["model_version"]
    )


@app.post("/detect-fraud", response_model=DetectionResponse, tags=["Detection"])
async def detect_fraud(
    transaction: TransactionInput,
    background_tasks: BackgroundTasks
):
    """
    Endpoint utama deteksi fraud.

    Alur:
    1. Validasi input (Pydantic, otomatis)
    2. Feature engineering + inferensi XGBoost
    3. Komputasi SHAP (TreeSHAP)
    4. Penentuan keputusan berdasarkan dual threshold
    5. Kirim respons ke pengguna (< 1 detik)
    6. [Background] Seleksi audit + pencatatan blockchain (asinkron)
    """
    t_start = time.perf_counter()

    tx_dict = transaction.model_dump()

    # ---- Step 2: Inferensi ----
    try:
        fraud_score, X_row = predict(tx_dict)
    except Exception as e:
        logger.error(f"ML inference gagal: {e}")
        raise HTTPException(status_code=500, detail=f"ML Service error: {str(e)}")

    # ---- Step 3: SHAP ----
    try:
        shap_expl = explain(X_row)
    except Exception as e:
        logger.error(f"SHAP computation gagal: {e}")
        raise HTTPException(status_code=500, detail=f"SHAP Service error: {str(e)}")

    # ---- Step 4: Keputusan ----
    decision, theta_low, theta_high = get_decision(fraud_score)
    theta_optimal = get_theta_optimal()

    # ---- Step 5: Menyiapkan respons ----
    elapsed_ms = (time.perf_counter() - t_start) * 1000

    # Konversi top_features ke Pydantic model
    top_features_pydantic = [
        SHAPFeatureContribution(**f)
        for f in shap_expl["top_features"]
    ]
    shap_explanation_pydantic = SHAPExplanation(
        base_value                   = shap_expl["base_value"],
        top_features                 = top_features_pydantic,
        natural_language_explanation = shap_expl["natural_language_explanation"],
        shap_hash_top3               = shap_expl["shap_hash_top3"]
    )

    pipeline      = load_pipeline()
    response      = DetectionResponse(
        transaction_id    = transaction.transaction_id,
        fraud_score       = round(fraud_score, 6),
        decision          = decision,
        theta_low         = theta_low,
        theta_high        = theta_high,
        shap_explanation  = shap_explanation_pydantic,
        model_version     = pipeline["model_version"],
        processing_time_ms= round(elapsed_ms, 2)
    )

    # ---- Step 6: Blockchain audit asinkron ----
    audit_needed, audit_reason = should_audit(
        fraud_score    = fraud_score,
        amount         = transaction.amount,
        theta_high     = theta_high,
        theta_optimal  = theta_optimal
    )

    if audit_needed and app.state.web3 is not None:
        background_tasks.add_task(
            run_blockchain_audit,
            transaction_id      = transaction.transaction_id,
            transaction_details = tx_dict,
            fraud_score         = fraud_score,
            decision            = decision,
            theta_high          = theta_high,
            theta_optimal       = theta_optimal,
            shap_explanation    = shap_expl,
            web3_service        = app.state.web3,
            audit_reason        = audit_reason
        )
        logger.info(
            f"[{transaction.transaction_id}] "
            f"Audit dijadwalkan: reason={audit_reason}"
        )

    logger.info(
        f"[{transaction.transaction_id}] "
        f"decision={decision} | score={fraud_score:.4f} | "
        f"{elapsed_ms:.1f}ms | audit={audit_needed}"
    )

    return response


@app.get("/audit/stats", response_model=AuditStatsResponse, tags=["Audit"])
async def get_audit_stats():
    """
    Mengambil statistik agregat audit trail dari blockchain.
    Fungsi read-only, tidak mengonsumsi gas.
    """
    if app.state.web3 is None:
        raise HTTPException(
            status_code=503,
            detail="Blockchain service tidak tersedia."
        )
    stats = app.state.web3.get_audit_stats()
    if not stats:
        raise HTTPException(status_code=500, detail="Gagal mengambil stats.")
    return AuditStatsResponse(**stats)


@app.get("/audit/verify/{tx_hash_hex}", tags=["Audit"])
async def verify_audit_record(tx_hash_hex: str):
    """
    Memverifikasi integritas catatan audit berdasarkan transaction hash.
    Hash di-decode dari hex string ke bytes32.
    """
    if app.state.web3 is None:
        raise HTTPException(status_code=503, detail="Blockchain tidak tersedia.")

    try:
        tx_hash_bytes = bytes.fromhex(tx_hash_hex)
        record        = app.state.web3.get_transaction(tx_hash_bytes[:32].ljust(32, b'\x00'))
        if record is None:
            return {"exists": False, "message": "Record tidak ditemukan di blockchain."}
        return {"exists": True, "record": record}
    except ValueError:
        raise HTTPException(status_code=400, detail="Format hash tidak valid.")


@app.post("/system/retry-pending", tags=["System"])
async def retry_pending(background_tasks: BackgroundTasks):
    """
    Memicu retry manual untuk pending blockchain records.
    """
    if app.state.web3 is None:
        raise HTTPException(status_code=503, detail="Blockchain tidak tersedia.")

    background_tasks.add_task(retry_pending_records, app.state.web3)
    return {"message": "Retry pending records dijadwalkan."}


# ================================================================
# ENTRYPOINT
# ================================================================

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host    = API_HOST,
        port    = API_PORT,
        reload  = False,
        workers = 1
    )