import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import time
from datetime import datetime, timezone

from src.blockchain.audit_service import (
    load_pending_records, remove_from_pending, save_to_pending
)
from config import BLOCKCHAIN_MAX_RETRIES
from src.utils.logger import get_logger

logger = get_logger(__name__)


def retry_pending_records(web3_service) -> dict:
    """
    Mencoba ulang pencatatan blockchain untuk semua pending records.
    Dipanggil secara periodik dari FastAPI background task.
    """
    pending = load_pending_records()

    if not pending:
        return {"success": 0, "still_pending": 0, "failed": 0}

    logger.info(f"Background worker: {len(pending)} pending record ditemukan.")

    success_count      = 0
    still_pending_list = []
    failed_list        = []

    for record in pending:
        tx_id       = record.get("transaction_id", "unknown")
        retry_count = record.get("retry_count", 0)

        if retry_count >= BLOCKCHAIN_MAX_RETRIES:
            logger.error(
                f"Record {tx_id} melampaui batas retry "
                f"({BLOCKCHAIN_MAX_RETRIES}). Ditandai failed."
            )
            failed_list.append(tx_id)
            continue

        try:
            result = web3_service.record_transaction(
                tx_hash_bytes32   = bytes.fromhex(record["audit_hash_b32"]),
                fraud_score_int   = record["fraud_score_u8"],
                shap_hash_bytes32 = bytes.fromhex(record["shap_hash_b32"]),
                status_int        = record["status_u8"],
                timestamp_int     = record["timestamp_int"]
            )

            if result.get("success"):
                # Update file audit dengan blockchain_tx_hash
                audit_path = Path(record["audit_file"])
                if audit_path.exists():
                    with open(audit_path) as f:
                        audit_data = json.load(f)
                    audit_data["audit_metadata"]["blockchain_tx_hash"] = \
                        result["blockchain_tx_hash"]
                    with open(audit_path, "w") as f:
                        json.dump(audit_data, f, indent=2, ensure_ascii=False)

                remove_from_pending(tx_id)
                success_count += 1
                logger.info(
                    f"Retry berhasil: {tx_id} → "
                    f"block={result.get('block_number')}"
                )
            else:
                record["retry_count"] = retry_count + 1
                record["last_error"]  = result.get("error", "unknown")
                still_pending_list.append(record)
                logger.warning(
                    f"Retry gagal ({retry_count+1}/{BLOCKCHAIN_MAX_RETRIES}): "
                    f"{tx_id}"
                )

        except Exception as e:
            record["retry_count"] = retry_count + 1
            record["last_error"]  = str(e)
            still_pending_list.append(record)
            logger.warning(f"Retry exception: {tx_id} — {e}")

    # Simpan ulang yang masih pending
    if still_pending_list:
        with open(
            Path(__file__).resolve().parents[2] /
            "reports" / "audit_logs" / "pending_blockchain.json", "w"
        ) as f:
            json.dump(still_pending_list, f, indent=2)

    stats = {
        "success":       success_count,
        "still_pending": len(still_pending_list),
        "failed":        len(failed_list)
    }
    logger.info(f"Background worker selesai: {stats}")
    return stats