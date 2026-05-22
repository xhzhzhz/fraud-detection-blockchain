import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import time
from web3 import Web3
from web3.exceptions import ContractLogicError
from dotenv import load_dotenv
import os

from config import EVALUATION_DIR
from src.utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

FRAUD_AUDIT_ABI = [
    {
        "inputs": [
            {"name": "txHash",          "type": "bytes32"},
            {"name": "fraudScore",      "type": "uint8"},
            {"name": "shapHash",        "type": "bytes32"},
            {"name": "status",          "type": "uint8"},
            {"name": "recordTimestamp", "type": "uint256"}
        ],
        "name": "recordTransaction",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [{"name": "txHash", "type": "bytes32"}],
        "name": "getTransaction",
        "outputs": [
            {"name": "transactionHash", "type": "bytes32"},
            {"name": "timestamp",       "type": "uint256"},
            {"name": "fraudScore",      "type": "uint8"},
            {"name": "shapHash",        "type": "bytes32"},
            {"name": "status",          "type": "uint8"},
            {"name": "validator",       "type": "address"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [
            {"name": "txHash",          "type": "bytes32"},
            {"name": "fraudScore",      "type": "uint8"},
            {"name": "shapHash",        "type": "bytes32"},
            {"name": "status",          "type": "uint8"},
            {"name": "recordTimestamp", "type": "uint256"}
        ],
        "name": "verifyIntegrity",
        "outputs": [{"name": "", "type": "bool"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [{"name": "txHash", "type": "bytes32"}],
        "name": "recordExists",
        "outputs": [{"name": "", "type": "bool"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "getAuditStats",
        "outputs": [
            {"name": "total",      "type": "uint256"},
            {"name": "fraud",      "type": "uint256"},
            {"name": "suspicious", "type": "uint256"},
            {"name": "legitimate", "type": "uint256"},
            {"name": "firstTs",    "type": "uint256"},
            {"name": "lastTs",     "type": "uint256"}
        ],
        "stateMutability": "view",
        "type": "function"
    }
]


class Web3Service:
    """
    Singleton-style service untuk interaksi dengan FraudAudit contract
    """

    def __init__(self):
        alchemy_url      = os.getenv("ALCHEMY_SEPOLIA_URL", "")
        private_key      = os.getenv("DEPLOYER_PRIVATE_KEY", "")
        contract_address = os.getenv("CONTRACT_ADDRESS", "")

        if not all([alchemy_url, private_key, contract_address]):
            raise EnvironmentError(
                "ALCHEMY_SEPOLIA_URL, DEPLOYER_PRIVATE_KEY, dan "
                "CONTRACT_ADDRESS harus diisi di .env"
            )

        self.w3 = Web3(Web3.HTTPProvider(alchemy_url))
        if not self.w3.is_connected():
            raise ConnectionError(
                f"Tidak dapat terhubung ke Sepolia: {alchemy_url}"
            )

        self.account  = self.w3.eth.account.from_key(private_key)
        self.contract = self.w3.eth.contract(
            address=Web3.to_checksum_address(contract_address),
            abi=FRAUD_AUDIT_ABI
        )

        logger.info(
            f"Web3Service terhubung ke Sepolia | "
            f"Account: {self.account.address} | "
            f"Contract: {contract_address}"
        )

    def record_transaction(
        self,
        tx_hash_bytes32: bytes,
        fraud_score_int: int,
        shap_hash_bytes32: bytes,
        status_int: int,
        timestamp_int: int,
        max_retries: int = 3
    ) -> dict:

        for attempt in range(1, max_retries + 1):
            try:
                # Estimasi gas sebelum kirim
                gas_estimate = self.contract.functions.recordTransaction(
                    tx_hash_bytes32,
                    fraud_score_int,
                    shap_hash_bytes32,
                    status_int,
                    timestamp_int
                ).estimate_gas({"from": self.account.address})

                # Menambahkan 20% buffer pada gas estimate
                gas_limit = int(gas_estimate * 1.2)

                nonce = self.w3.eth.get_transaction_count(self.account.address)

                tx = self.contract.functions.recordTransaction(
                    tx_hash_bytes32,
                    fraud_score_int,
                    shap_hash_bytes32,
                    status_int,
                    timestamp_int
                ).build_transaction({
                    "from":     self.account.address,
                    "nonce":    nonce,
                    "gas":      gas_limit,
                    "gasPrice": self.w3.eth.gas_price,
                })

                signed_tx = self.account.sign_transaction(tx)
                tx_hash   = self.w3.eth.send_raw_transaction(
                    signed_tx.rawTransaction
                )

                logger.info(
                    f"Transaksi blockchain dikirim: {tx_hash.hex()} "
                    f"(gas_limit={gas_limit})"
                )

                # Menunggu konfirmasi (timeout 120 detik)
                receipt = self.w3.eth.wait_for_transaction_receipt(
                    tx_hash, timeout=120
                )

                if receipt.status != 1:
                    raise Exception(
                        f"Transaksi gagal di blockchain: "
                        f"status={receipt.status}"
                    )

                result = {
                    "success":      True,
                    "blockchain_tx_hash": tx_hash.hex(),
                    "block_number": receipt.blockNumber,
                    "gas_used":     receipt.gasUsed,
                    "attempt":      attempt
                }
                logger.info(
                    f"Blockchain confirmed: block={receipt.blockNumber} | "
                    f"gas={receipt.gasUsed}"
                )
                return result

            except ContractLogicError as e:
                # Error dari kontrak (revert)
                logger.error(f"Contract revert: {str(e)}")
                return {"success": False, "error": str(e), "retry": False}

            except Exception as e:
                logger.warning(
                    f"Attempt {attempt}/{max_retries} gagal: {str(e)}"
                )
                if attempt < max_retries:
                    wait = 10 * (2 ** (attempt - 1))  # exponential backoff
                    logger.info(f"Retry dalam {wait} detik...")
                    time.sleep(wait)
                else:
                    return {
                        "success": False,
                        "error":   str(e),
                        "retry":   True   # Menandai untuk background worker
                    }

    def get_transaction(self, tx_hash_bytes32: bytes) -> dict | None:
        """Membaca catatan audit dari blockchain (read-only)"""
        try:
            result = self.contract.functions.getTransaction(
                tx_hash_bytes32
            ).call()
            return {
                "transaction_hash": result[0].hex(),
                "timestamp":        result[1],
                "fraud_score":      result[2],
                "shap_hash":        result[3].hex(),
                "status":           result[4],
                "validator":        result[5]
            }
        except Exception as e:
            logger.error(f"getTransaction gagal: {e}")
            return None

    def verify_integrity(
        self,
        tx_hash_bytes32: bytes,
        fraud_score_int: int,
        shap_hash_bytes32: bytes,
        status_int: int,
        timestamp_int: int
    ) -> bool:
        """Memverifikasi integritas catatan on-chain (read-only)"""
        try:
            return self.contract.functions.verifyIntegrity(
                tx_hash_bytes32,
                fraud_score_int,
                shap_hash_bytes32,
                status_int,
                timestamp_int
            ).call()
        except Exception as e:
            logger.error(f"verifyIntegrity gagal: {e}")
            return False

    def record_exists(self, tx_hash_bytes32: bytes) -> bool:
        try:
            return self.contract.functions.recordExists(tx_hash_bytes32).call()
        except Exception:
            return False

    def get_audit_stats(self) -> dict:
        """Mengambil statistik agregat dari blockchain"""
        try:
            result = self.contract.functions.getAuditStats().call()
            return {
                "total_records":    result[0],
                "total_fraud":      result[1],
                "total_suspicious": result[2],
                "total_legitimate": result[3],
                "first_timestamp":  result[4],
                "last_timestamp":   result[5]
            }
        except Exception as e:
            logger.error(f"getAuditStats gagal: {e}")
            return {}