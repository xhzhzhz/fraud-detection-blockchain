import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path

from config import (
    AUDIT_LOGS_DIR,
    HIGH_VALUE_THRESHOLD,
    UNCERTAINTY_THRESHOLD
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# File fallback untuk transaksi yang gagal dicatat ke blockchain
PENDING_BLOCKCHAIN_PATH = AUDIT_LOGS_DIR / "pending_blockchain.json"


# ================================================================
# SELEKSI AUDIT BERBASIS RISIKO
# ================================================================

def should_audit(
    fraud_score: float,
    amount: float,
    theta_high: float,
    theta_optimal: float,
    uncertainty_threshold: float = None
) -> tuple[bool, str | None]:

    if uncertainty_threshold is None:
        uncertainty_threshold = UNCERTAINTY_THRESHOLD

    # Kriteria 1: Fraud score di atas theta_high
    if fraud_score > theta_high:
        return True, "high_fraud_score"

    # Kriteria 2: Nominal transaksi tinggi (AML)
    if amount > HIGH_VALUE_THRESHOLD:
        return True, "high_value_transaction"

    # Kriteria 3: Model uncertainty tinggi (dekat threshold)
    uncertainty = 1.0 - abs(fraud_score - theta_optimal)
    if uncertainty > (1.0 - uncertainty_threshold):
        return True, "high_model_uncertainty"

    return False, None


# ================================================================
# PEMBUATAN FILE AUDIT OFF-CHAIN
# ================================================================

def build_audit_json(
    transaction_id: str,
    transaction_details: dict,
    fraud_score: float,
    decision: str,
    theta_high: float,
    shap_explanation: dict,
    model_version: str,
    validator_address: str,
    audit_reason: str
) -> dict:

    # Hash identitas (privasi data minimization)
    def hash_id(val: str) -> str:
        return hashlib.sha256(str(val).encode()).hexdigest()[:16] + "..."

    td = transaction_details
    return {
        "transaction_id": transaction_id,
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "audit_reason":   audit_reason,
        "transaction_details": {
            "type":                  td.get("type", "UNKNOWN"),
            "amount":                td.get("amount", 0),
            "sender_id":             hash_id(td.get("nameOrig", "")),
            "receiver_id":           hash_id(td.get("nameDest", "")),
            "sender_balance_before": td.get("oldbalanceOrg", 0),
            "sender_balance_after":  td.get("newbalanceOrig", 0),
            "receiver_balance_before": td.get("oldbalanceDest", 0),
            "receiver_balance_after":  td.get("newbalanceDest", 0),
        },
        "detection_result": {
            "fraud_score":              round(float(fraud_score), 6),
            "decision":                 decision,
            "decision_threshold_high":  round(float(theta_high), 4),
        },
        "shap_explanation":   shap_explanation,
        "audit_metadata": {
            "model_version":    model_version,
            "validator_address": validator_address,
            "blockchain_tx_hash": None   # Diisi setelah on-chain record
        }
    }


def save_audit_json(audit_data: dict, transaction_id: str) -> Path:
    """Menyimpan file JSON audit ke AUDIT_LOGS_DIR."""
    safe_id   = transaction_id.replace("/", "-").replace(":", "-")
    file_path = AUDIT_LOGS_DIR / f"audit_{safe_id}.json"

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2, ensure_ascii=False)

    logger.info(f"File audit disimpan: {file_path}")
    return file_path


# ================================================================
# KOMPUTASI HASH SHA-256
# ================================================================

def compute_audit_hash(audit_data: dict) -> bytes:
    """
    Menghitung SHA-256 dari konten file audit JSON.
    Disimpan on-chain sebagai transactionHash.
    JSON diserialize dengan sort_keys=True untuk determinisme.
    """
    payload = json.dumps(audit_data, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).digest()   # raw bytes


def compute_shap_hash(top_k_shap: list[dict]) -> bytes:
    """
    Menghitung SHA-256 dari top-3 SHAP values.
    Disimpan on-chain sebagai shapHash.
    top_k_shap: list of {"feature": str, "shap_value": float}
    """
    top3    = top_k_shap[:3]
    payload = json.dumps(
        [{"feature": f["feature"],
          "shap_value": round(f["shap_value"], 6)}
         for f in top3],
        sort_keys=True
    )
    return hashlib.sha256(payload.encode()).digest()


def bytes_to_bytes32(b: bytes) -> bytes:
    """Pad/trim bytes ke 32 bytes untuk bytes32 Solidity."""
    return b[:32].ljust(32, b'\x00')


# ================================================================
# KONVERSI NILAI UNTUK SMART CONTRACT
# ================================================================

def fraud_score_to_uint8(fraud_score: float) -> int:
    """Konversi probabilitas fraud (0.0-1.0) ke uint8 (0-100)."""
    return min(100, max(0, int(round(fraud_score * 100))))


def decision_to_status(decision: str) -> int:
    """Konversi string decision ke uint8 status untuk contract."""
    mapping = {"LEGITIMATE": 0, "SUSPICIOUS": 1, "FRAUD": 2}
    return mapping.get(decision.upper(), 0)


# ================================================================
# FALLBACK: PENDING BLOCKCHAIN
# ================================================================

def save_to_pending(pending_record: dict) -> None:
    """
    Menyimpan record yang gagal dicatat ke blockchain ke file pending.
    Background worker akan mencoba ulang secara berkala.
    """
    pending_list = []
    if PENDING_BLOCKCHAIN_PATH.exists():
        with open(PENDING_BLOCKCHAIN_PATH) as f:
            pending_list = json.load(f)

    pending_record["pending_since"] = datetime.now(timezone.utc).isoformat()
    pending_record["retry_count"]   = 0
    pending_list.append(pending_record)

    with open(PENDING_BLOCKCHAIN_PATH, "w") as f:
        json.dump(pending_list, f, indent=2)

    logger.warning(
        f"Record disimpan ke pending_blockchain: "
        f"{pending_record.get('transaction_id')}"
    )


def load_pending_records() -> list:
    if not PENDING_BLOCKCHAIN_PATH.exists():
        return []
    with open(PENDING_BLOCKCHAIN_PATH) as f:
        return json.load(f)


def remove_from_pending(transaction_id: str) -> None:
    pending = load_pending_records()
    pending = [r for r in pending if r.get("transaction_id") != transaction_id]
    with open(PENDING_BLOCKCHAIN_PATH, "w") as f:
        json.dump(pending, f, indent=2)


# ================================================================
# MAIN AUDIT FUNCTION: dipanggil dari API (Fase 5)
# ================================================================

def process_audit(
    transaction_id: str,
    transaction_details: dict,
    fraud_score: float,
    decision: str,
    theta_high: float,
    theta_optimal: float,
    shap_explanation: dict,
    model_version: str,
    web3_service,
    validator_address: str,
    audit_reason: str
) -> dict:

    # 1. Membuat dan simpan file JSON audit
    audit_data = build_audit_json(
        transaction_id=transaction_id,
        transaction_details=transaction_details,
        fraud_score=fraud_score,
        decision=decision,
        theta_high=theta_high,
        shap_explanation=shap_explanation,
        model_version=model_version,
        validator_address=validator_address,
        audit_reason=audit_reason
    )
    audit_file_path = save_audit_json(audit_data, transaction_id)

    # 2. Menghitung hash untuk on-chain
    audit_hash_bytes = compute_audit_hash(audit_data)
    shap_hash_bytes  = compute_shap_hash(
        shap_explanation.get("top_features", [])
    )

    audit_hash_b32 = bytes_to_bytes32(audit_hash_bytes)
    shap_hash_b32  = bytes_to_bytes32(shap_hash_bytes)
    fraud_score_u8 = fraud_score_to_uint8(fraud_score)
    status_u8      = decision_to_status(decision)
    timestamp_int  = int(time.time())

    # 3. Catat ke blockchain
    blockchain_result = web3_service.record_transaction(
        tx_hash_bytes32   = audit_hash_b32,
        fraud_score_int   = fraud_score_u8,
        shap_hash_bytes32 = shap_hash_b32,
        status_int        = status_u8,
        timestamp_int     = timestamp_int
    )

    if blockchain_result.get("success"):
        blockchain_tx_hash = blockchain_result["blockchain_tx_hash"]

        # 4. Update file audit dengan blockchain_tx_hash
        audit_data["audit_metadata"]["blockchain_tx_hash"] = blockchain_tx_hash
        with open(audit_file_path, "w", encoding="utf-8") as f:
            json.dump(audit_data, f, indent=2, ensure_ascii=False)

        logger.info(
            f"Audit selesai: {transaction_id} → "
            f"blockchain tx: {blockchain_tx_hash}"
        )
        return {
            "audited":             True,
            "audit_reason":        audit_reason,
            "audit_file":          str(audit_file_path),
            "audit_hash":          audit_hash_bytes.hex(),
            "shap_hash":           shap_hash_bytes.hex(),
            "blockchain_tx_hash":  blockchain_tx_hash,
            "block_number":        blockchain_result.get("block_number"),
            "gas_used":            blockchain_result.get("gas_used"),
        }
    else:
        # Fallback: simpan ke pending untuk retry
        pending_record = {
            "transaction_id":  transaction_id,
            "audit_file":      str(audit_file_path),
            "audit_hash":      audit_hash_bytes.hex(),
            "shap_hash":       shap_hash_bytes.hex(),
            "audit_hash_b32":  audit_hash_b32.hex(),
            "shap_hash_b32":   shap_hash_b32.hex(),
            "fraud_score_u8":  fraud_score_u8,
            "status_u8":       status_u8,
            "timestamp_int":   timestamp_int,
            "error":           blockchain_result.get("error", "unknown")
        }
        save_to_pending(pending_record)
        return {
            "audited":      False,
            "audit_reason": audit_reason,
            "audit_file":   str(audit_file_path),
            "status":       "pending_blockchain",
            "error":        blockchain_result.get("error")
        }