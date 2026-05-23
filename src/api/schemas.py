from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal
from enum import IntEnum


class TransactionType(str):
    """Tipe transaksi yang valid dalam PaySim."""
    CASH_IN  = "CASH_IN"
    CASH_OUT = "CASH_OUT"
    DEBIT    = "DEBIT"
    PAYMENT  = "PAYMENT"
    TRANSFER = "TRANSFER"
    VALID    = {"CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"}


class TransactionInput(BaseModel):
    """
    Input transaksi untuk endpoint POST /detect-fraud.
    """
    transaction_id:  str   = Field(...,  description="ID unik transaksi")
    step:            int   = Field(...,  ge=1, description="Unit waktu simulasi")
    type:            str   = Field(...,  description="Tipe transaksi")
    amount:          float = Field(...,  gt=0, description="Nominal transaksi")
    oldbalanceOrg:   float = Field(...,  ge=0, description="Saldo pengirim sebelum")
    newbalanceOrig:  float = Field(...,  ge=0, description="Saldo pengirim sesudah")
    oldbalanceDest:  float = Field(...,  ge=0, description="Saldo penerima sebelum")
    newbalanceDest:  float = Field(...,  ge=0, description="Saldo penerima sesudah")

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        if v.upper() not in TransactionType.VALID:
            raise ValueError(
                f"Tipe transaksi tidak valid: '{v}'. "
                f"Pilihan: {sorted(TransactionType.VALID)}"
            )
        return v.upper()


class SHAPFeatureContribution(BaseModel):
    rank:            int
    feature:         str
    shap_value:      float
    feature_value:   float
    direction:       str
    nl_description:  str


class SHAPExplanation(BaseModel):
    base_value:                  float
    top_features:                list[SHAPFeatureContribution]
    natural_language_explanation: str
    shap_hash_top3:              str


class AuditInfo(BaseModel):
    audited:              bool
    audit_reason:         Optional[str]
    audit_file:           Optional[str]
    blockchain_tx_hash:   Optional[str]
    block_number:         Optional[int]
    gas_used:             Optional[int]
    status:               Optional[str]  # "confirmed" | "pending_blockchain"


class DetectionResponse(BaseModel):
    """
    Respons endpoint POST /detect-fraud.
    Dikembalikan ke pengguna sebelum proses blockchain selesai
    (blockchain berjalan asinkron).
    """
    transaction_id:   str
    fraud_score:      float
    decision:         Literal["LEGITIMATE", "SUSPICIOUS", "FRAUD"]
    theta_low:        float
    theta_high:       float
    shap_explanation: SHAPExplanation
    model_version:    str
    audit_info:       Optional[AuditInfo] = None
    processing_time_ms: Optional[float]  = None


class HealthResponse(BaseModel):
    status:           str
    model_loaded:     bool
    blockchain_connected: bool
    contract_address: str
    model_version:    str


class AuditStatsResponse(BaseModel):
    total_records:    int
    total_fraud:      int
    total_suspicious: int
    total_legitimate: int
    first_timestamp:  int
    last_timestamp:   int