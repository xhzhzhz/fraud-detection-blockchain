import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import time
import statistics
import concurrent.futures
import requests
import pandas as pd
import numpy as np

from config import DATA_SPLITS_DIR, EVALUATION_DIR
from src.utils.logger import get_logger

logger  = get_logger(__name__)
API_URL = "http://localhost:8000/detect-fraud"

def make_payload(row: pd.Series, idx: int) -> dict:
    type_cols = [c for c in row.index if c.startswith("type_")]
    tx_type   = next(
        (c.replace("type_", "") for c in type_cols if row[c] == 1),
        "PAYMENT"
    )
    return {
        "transaction_id": f"PERF-{idx:06d}",
        "step":           int(row["step"]),
        "type":           tx_type,
        "amount":         float(row["amount"]),
        "oldbalanceOrg":  float(row["oldbalanceOrg"]),
        "newbalanceOrig": float(row["newbalanceOrig"]),
        "oldbalanceDest": float(row["oldbalanceDest"]),
        "newbalanceDest": float(row["newbalanceDest"])
    }


def single_request(payload: dict) -> float:
    """Mengirim satu request dan mengembalikan latensi dalam ms."""
    t = time.perf_counter()
    try:
        r = requests.post(API_URL, json=payload, timeout=10)
        r.raise_for_status()
        return (time.perf_counter() - t) * 1000
    except Exception as e:
        return -1.0


def run_performance_tests():
    print("="*60)
    print("PERFORMANCE TEST: Latensi & Throughput")
    print("="*60)

    df      = pd.read_csv(DATA_SPLITS_DIR / "test.csv")
    samples = df.sample(n=200, random_state=42)
    payloads = [
        make_payload(row, i)
        for i, (_, row) in enumerate(samples.iterrows())
    ]

    # ---- TEST 1: Single request latency (n=50) ----
    print("\n[T1] Single request latency (n=50 sequential)...")
    latencies = []
    for payload in payloads[:50]:
        lat = single_request(payload)
        if lat > 0:
            latencies.append(lat)
        time.sleep(0.05)  # 50ms jeda

    if latencies:
        p50 = statistics.median(latencies)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        print(f"  Median (P50)  : {p50:.1f} ms")
        print(f"  P95           : {p95:.1f} ms")
        print(f"  P99           : {p99:.1f} ms")
        print(f"  Max           : {max(latencies):.1f} ms")
        print(f"  Target < 5000ms: {'✓' if p95 < 5000 else '✗'}")
        t1_pass = p95 < 5000
    else:
        print("  Tidak ada respons valid.")
        t1_pass = False

    # ---- TEST 2: Concurrent requests (10 simultan) ----
    print("\n[T2] Concurrent requests (10 simultan)...")
    concurrent_payloads = payloads[50:60]
    start_ts = time.perf_counter()
    errors   = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(single_request, p)
            for p in concurrent_payloads
        ]
        conc_latencies = []
        for f in concurrent.futures.as_completed(futures):
            lat = f.result()
            if lat < 0:
                errors += 1
            else:
                conc_latencies.append(lat)

    total_time = (time.perf_counter() - start_ts) * 1000
    print(f"  10 request selesai dalam {total_time:.1f} ms")
    print(f"  Error: {errors}/10")
    if conc_latencies:
        print(f"  Max latency: {max(conc_latencies):.1f} ms")
    t2_pass = errors == 0

    # ---- TEST 3: Concurrent requests (50 simultan) ----
    print("\n[T3] Concurrent requests (50 simultan)...")
    concurrent_payloads_50 = payloads[60:110]
    errors50 = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        futures50 = [
            executor.submit(single_request, p)
            for p in concurrent_payloads_50
        ]
        lat50 = []
        for f in concurrent.futures.as_completed(futures50):
            lat = f.result()
            if lat < 0:
                errors50 += 1
            else:
                lat50.append(lat)

    print(f"  Error: {errors50}/50")
    if lat50:
        print(f"  Max latency   : {max(lat50):.1f} ms")
        print(f"  Median latency: {statistics.median(lat50):.1f} ms")
    t3_pass = errors50 <= 5   # Toleransi ≤10% error pada beban tinggi

    # ---- TEST 4: Invalid input handling ----
    print("\n[T4] Invalid input handling...")
    invalid_cases = [
        {"transaction_id": "INV-001", "step": 1, "type": "INVALID_TYPE",
         "amount": 100, "oldbalanceOrg": 100, "newbalanceOrig": 0,
         "oldbalanceDest": 0, "newbalanceDest": 100},   # tipe tidak valid
        {"transaction_id": "INV-002", "step": 1, "type": "CASH_OUT",
         "amount": -100, "oldbalanceOrg": 100, "newbalanceOrig": 0,
         "oldbalanceDest": 0, "newbalanceDest": 100},   # amount negatif
    ]
    t4_results = []
    for case in invalid_cases:
        try:
            r = requests.post(API_URL, json=case, timeout=10)
            is_400 = r.status_code == 422  # FastAPI validation error
            t4_results.append(is_400)
            print(f"  status={r.status_code} (exp=422) | {'✓' if is_400 else '✗'}")
        except Exception as e:
            print(f"  Error: {e}")
            t4_results.append(False)
    t4_pass = all(t4_results)

    # ---- Simpan hasil ----
    results = {
        "timestamp":        __import__("datetime").datetime.now().isoformat(),
        "T1_latency": {
            "p50_ms": round(p50, 1) if latencies else None,
            "p95_ms": round(p95, 1) if latencies else None,
            "p99_ms": round(p99, 1) if latencies else None,
            "target_ms": 5000,
            "passed": bool(t1_pass)
        },
        "T2_concurrent_10":  {"errors": errors, "passed": bool(t2_pass)},
        "T3_concurrent_50":  {"errors": errors50, "passed": bool(t3_pass)},
        "T4_invalid_input":  {"passed": bool(t4_pass)}
    }

    out_path = EVALUATION_DIR / "performance_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*60)
    print("RINGKASAN PERFORMANCE TEST")
    print("="*60)
    for name, data in results.items():
        if name == "timestamp":
            continue
        ok = data.get("passed", False)
        print(f"  {'PASS ✓' if ok else 'FAIL ✗'}  {name}")
    print(f"\nHasil disimpan: {out_path}")


if __name__ == "__main__":
    run_performance_tests()