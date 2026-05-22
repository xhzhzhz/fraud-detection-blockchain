import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import time
import hashlib
from datetime import datetime, timezone

from src.blockchain.web3_service  import Web3Service
from src.blockchain.audit_service import (
    should_audit, compute_audit_hash, compute_shap_hash,
    bytes_to_bytes32, fraud_score_to_uint8, decision_to_status,
    build_audit_json
)
from config import EVALUATION_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Load info deployment
with open(EVALUATION_DIR / "deployment_info.json") as f:
    deploy_info = json.load(f)

# Load dual thresholds (baseline)
with open(EVALUATION_DIR / "dual_thresholds.json") as f:
    dual = json.load(f)

THETA_HIGH    = dual["baseline"]["theta_high"]
THETA_LOW     = dual["baseline"]["theta_low"]
THETA_OPTIMAL = THETA_HIGH  # threshold optimal untuk uncertainty


def run_integration_tests():
    print("="*60)
    print("INTEGRATION TEST : FraudAudit on Sepolia")
    print(f"Contract: {deploy_info['contractAddress']}")
    print(f"Network : {deploy_info['network']}")
    print("="*60)

    w3 = Web3Service()
    results = []

    # --------------------------------------------------
    # TEST 1: getAuditStats awal
    # --------------------------------------------------
    print("\n[T1] getAuditStats: stats awal kontrak...")
    stats = w3.get_audit_stats()
    print(f"     total={stats['total_records']} | fraud={stats['total_fraud']} | "
          f"suspicious={stats['total_suspicious']} | legit={stats['total_legitimate']}")
    results.append(("T1 getAuditStats", True))

    # --------------------------------------------------
    # TEST 2: Catat transaksi FRAUD
    # --------------------------------------------------
    print("\n[T2] recordTransaction: kasus FRAUD...")

    audit_data = build_audit_json(
        transaction_id    = "INTEGRATION-TEST-FRAUD-001",
        transaction_details = {
            "type": "CASH_OUT", "amount": 250000,
            "nameOrig": "C123", "nameDest": "M456",
            "oldbalanceOrg": 300000, "newbalanceOrig": 50000,
            "oldbalanceDest": 0,     "newbalanceDest": 250000
        },
        fraud_score       = 0.92,
        decision          = "FRAUD",
        theta_high        = THETA_HIGH,
        shap_explanation  = {
            "base_value": 0.000214,
            "top_features": [
                {"feature": "errorBalanceOrig",    "shap_value": 0.41,
                 "direction": "increases_fraud_risk", "feature_value": 250000},
                {"feature": "amountToBalanceOrig", "shap_value": 0.28,
                 "direction": "increases_fraud_risk", "feature_value": 0.83},
                {"feature": "type_PAYMENT",        "shap_value": -0.12,
                 "direction": "decreases_fraud_risk", "feature_value": 0}
            ],
            "natural_language_explanation": "Integration test fraud case."
        },
        model_version     = "xgboost-baseline-v1.0",
        validator_address = w3.account.address,
        audit_reason      = "high_fraud_score"
    )

    audit_hash_bytes  = compute_audit_hash(audit_data)
    shap_hash_bytes   = compute_shap_hash(
        audit_data["shap_explanation"]["top_features"]
    )
    audit_hash_b32    = bytes_to_bytes32(audit_hash_bytes)
    shap_hash_b32     = bytes_to_bytes32(shap_hash_bytes)
    ts                = int(time.time())

    result = w3.record_transaction(
        tx_hash_bytes32   = audit_hash_b32,
        fraud_score_int   = fraud_score_to_uint8(0.92),
        shap_hash_bytes32 = shap_hash_b32,
        status_int        = decision_to_status("FRAUD"),
        timestamp_int     = ts
    )
    t2_ok = result.get("success", False)
    print(f"     success={t2_ok} | "
          f"blockchain_tx={result.get('blockchain_tx_hash', 'N/A')[:20]}... | "
          f"gas={result.get('gas_used', 'N/A')}")
    results.append(("T2 recordTransaction FRAUD", t2_ok))

    # --------------------------------------------------
    # TEST 3: getTransaction 
    # --------------------------------------------------
    if t2_ok:
        print("\n[T3] getTransaction: baca data yang baru dicatat...")
        on_chain = w3.get_transaction(audit_hash_b32)
        t3_ok = on_chain is not None and on_chain["fraud_score"] == 92
        print(f"     fraud_score={on_chain['fraud_score'] if on_chain else 'N/A'} "
              f"(exp=92) | status={on_chain['status'] if on_chain else 'N/A'} (exp=2)")
        results.append(("T3 getTransaction", t3_ok))

        # --------------------------------------------------
        # TEST 4: verifyIntegrity
        # --------------------------------------------------
        print("\n[T4] verifyIntegrity: data cocok harus return True...")
        is_valid = w3.verify_integrity(
            audit_hash_b32, 92, shap_hash_b32, 2, ts
        )
        print(f"     result={is_valid} (exp=True)")
        results.append(("T4 verifyIntegrity cocok", is_valid))

        # --------------------------------------------------
        # TEST 5: verifyIntegrity 
        # --------------------------------------------------
        print("\n[T5] verifyIntegrity: fraudScore berbeda harus return False...")
        is_invalid = w3.verify_integrity(
            audit_hash_b32, 50, shap_hash_b32, 2, ts
        )
        print(f"     result={is_invalid} (exp=False)")
        results.append(("T5 verifyIntegrity modifikasi", not is_invalid))

    # --------------------------------------------------
    # TEST 6: should_audit 
    # --------------------------------------------------
    print("\n[T6] should_audit: pengujian tiga kriteria seleksi...")
    cases = [
        (0.95, 100_000,  True,  "high_fraud_score"),     # Kriteria 1
        (0.10, 15_000_000, True,  "high_value_transaction"), # Kriteria 2
        (0.60, 100_000,  True,  "high_model_uncertainty"), # Kriteria 3
        (0.10, 5_000_000, False, None),                    # Tidak diaudit
    ]
    t6_ok = True
    for fs, amt, exp_audit, exp_reason in cases:
        audited, reason = should_audit(fs, amt, THETA_HIGH, THETA_OPTIMAL)
        ok = (audited == exp_audit)
        t6_ok = t6_ok and ok
        print(f"     fs={fs:.2f} amt={amt:>12,} → "
              f"audit={audited} (exp={exp_audit}) reason={reason} {'✓' if ok else '✗'}")
    results.append(("T6 should_audit logic", t6_ok))

    # --------------------------------------------------
    # TEST 7: getAuditStats akhir
    # --------------------------------------------------
    print("\n[T7] getAuditStats: verifikasi stats setelah pencatatan...")
    stats_final = w3.get_audit_stats()
    t7_ok = stats_final["total_records"] >= 1
    print(f"     total={stats_final['total_records']} | "
          f"fraud={stats_final['total_fraud']}")
    results.append(("T7 getAuditStats akhir", t7_ok))

    # --------------------------------------------------
    # RINGKASAN
    # --------------------------------------------------
    print("\n" + "="*60)
    print("RINGKASAN INTEGRATION TEST")
    print("="*60)
    passed = sum(1 for _, ok in results if ok)
    total  = len(results)
    for name, ok in results:
        status = "PASS ✓" if ok else "FAIL ✗"
        print(f"  {status}  {name}")
    print(f"\n{passed}/{total} test lulus")

    if deploy_info["network"] == "sepolia":
        print(f"\nVerifikasi di Etherscan:")
        print(f"  {deploy_info['explorerUrl']}")

    return passed == total


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)