import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.api.ml_service import load_pipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_decision(fraud_score: float) -> tuple[str, float, float]:
    """
    Menentukan keputusan berdasarkan fraud_score dan dual threshold.
    """
    pipeline   = load_pipeline()
    theta_low  = pipeline["theta_low"]
    theta_high = pipeline["theta_high"]

    if fraud_score > theta_high:
        decision = "FRAUD"
    elif fraud_score >= theta_low:
        decision = "SUSPICIOUS"
    else:
        decision = "LEGITIMATE"

    return decision, theta_low, theta_high


def get_theta_optimal() -> float:
    """Threshold optimal untuk komputasi uncertainty (= theta_high)."""
    pipeline = load_pipeline()
    return pipeline.get("theta_optimal", pipeline["theta_high"])