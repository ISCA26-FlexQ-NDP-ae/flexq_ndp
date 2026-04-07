#!/usr/bin/env python3
"""
Extract performance data from log files and produce 6 CSV tables for Fig7.

Tables:
  1. w_a + w4s8:   7B MVM (batch=1/4/16/64, q1-q6) + 7B MM (q1-q2)
  2. w_a + w8s16:  7B MVM (batch=1/4/16/64, q1-q6) + 7B MM (q1-q2)
  3. w_a + w4s8 + q1: all models MVM (batch=1..64) + all models MM
  4. w_only + w4s8:   7B MVM (batch=1/4/16/64, q1-q6) + 7B MM (q1-q2)
  5. w_only + w8s16:  7B MVM (batch=1/4/16/64, q1-q6) + 7B MM (q1-q2)
  6. w_only + w4s8 + q1: all models MVM (batch=1..64) + all models MM
"""

import re
import csv
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
LOG_OPTIMAL = SCRIPT_DIR / "log_optimal"
LOG_REBUTTAL = SCRIPT_DIR / "log_rebuttal_mm_new"
OUTPUT_DIR = SCRIPT_DIR / "fig8"
OUTPUT_DIR.mkdir(exist_ok=True)

# Q-config to (group_k, group_n)
Q_CONFIG = {
    "q1": (16, 1),
    "q2": (32, 1),
    "q3": (16, 16),
    "q4": (32, 32),
    "q5": (64, 64),
    "q6": (128, 128),
}

# Precision: user-facing name -> log_optimal header name
PRECISION_TO_HEADER = {
    "w4s8":  "D4S8",
    "w8s16": "D8S16",
}

# Precision: user-facing name -> filename tag in log_rebuttal_mm_new
PRECISION_TO_FILETAG = {
    "w4s8":  "w4s8",
    "w8s16": "w8a16",  # files use w8a16, but content header says D8S16
}

# Method: file/CSV tag -> log_optimal content string
METHOD_TO_HEADER = {
    "wa":    "w_a",
    "wonly": "w_only",
}

# Operator dimensions: (model_size, op_idx) -> (M, K) for MVM
MVM_DIMS = {
    ("7B",  1): (4096,  4096),
    ("7B",  2): (4096,  11008),
    ("7B",  3): (11008, 4096),
    ("13B", 1): (5120,  5120),
    ("13B", 2): (5120,  13824),
    ("13B", 3): (13824, 5120),
    ("34B", 1): (6656,  6656),
    ("34B", 2): (6656,  832),
    ("34B", 3): (6656,  20480),
    ("34B", 4): (20480, 6656),
}

# Operator dimensions: (model_size, op_idx) -> (M, K, N) for MM
MM_DIMS = {
    ("7B",  1): (4096, 4096,  4096),
    ("7B",  2): (4096, 4096,  11008),
    ("7B",  3): (4096, 11008, 4096),
    ("13B", 1): (4096, 5120,  5120),
    ("13B", 2): (4096, 5120,  13824),
    ("13B", 3): (4096, 13824, 5120),
    ("34B", 1): (4096, 6656,  6656),
    ("34B", 2): (4096, 6656,  832),
    ("34B", 3): (4096, 6656,  20480),
    ("34B", 4): (4096, 20480, 6656),
}


# ===========================================================================
# Log parsing
# ===========================================================================

_OPTIMAL_HEADER_RE = re.compile(
    r'=====+\s+(\S+)\s+(\S+)\s+group_k=(\d+)\s+group_n=(\d+)\s+=====+'
)
_OPTIMAL_LATENCY_RE = re.compile(r'==>\s+Optimal Latency:\s+([\d.]+)')

_REBUTTAL_BASELINE_RE = re.compile(r'==>\s+Baseline\s+\d+\s+Latency:\s+([\d.]+)')
_REBUTTAL_BEST_RE     = re.compile(r'==>\s+Best Latency:\s+([\d.]+)')

_optimal_cache: dict = {}


def _load_optimal(path: Path) -> dict:
    """Parse a log_optimal file.
    Returns dict: {(method, precision, group_k, group_n): latency}
    """
    if path in _optimal_cache:
        return _optimal_cache[path]
    result = {}
    if not path.exists():
        _optimal_cache[path] = result
        return result
    current_key = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            m = _OPTIMAL_HEADER_RE.match(line)
            if m:
                method, prec, gk, gn = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
                current_key = (method, prec, gk, gn)
            else:
                m = _OPTIMAL_LATENCY_RE.match(line)
                if m and current_key is not None:
                    result[current_key] = float(m.group(1))
                    current_key = None
    _optimal_cache[path] = result
    return result


def _load_rebuttal(path: Path):
    """Parse a log_rebuttal_mm_new file.
    Returns (baseline_latency, best_latency), either may be None.
    """
    if not path.exists():
        return None, None
    baseline = None
    best = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            m = _REBUTTAL_BASELINE_RE.match(line)
            if m:
                baseline = float(m.group(1))
            m = _REBUTTAL_BEST_RE.match(line)
            if m:
                best = float(m.group(1))
    return baseline, best


# ===========================================================================
# File path builders
# ===========================================================================

def _batch_str(batch: int) -> str:
    """Convert batch size to filename segment (matches both log directories)."""
    if batch == 1:
        return "1"
    return f"B{batch}_{batch}"


def optimal_mvm_path(model: str, idx: int, batch: int) -> Path:
    M, K = MVM_DIMS[(model, idx)]
    bstr = _batch_str(batch)
    return LOG_OPTIMAL / f"{model}_MVM{idx}_{bstr}_{M}_{K}.log"


def optimal_mm_path(model: str, idx: int) -> Path:
    M, K, N = MM_DIMS[(model, idx)]
    return LOG_OPTIMAL / f"{model}_MM{idx}_{M}_{K}_{N}.log"


def rebuttal_mvm_path(model: str, idx: int, batch: int, prec_tag: str, q: str, method_tag: str) -> Path:
    M, K = MVM_DIMS[(model, idx)]
    bstr = _batch_str(batch)
    return LOG_REBUTTAL / f"{model}_MVM{idx}_{bstr}_{M}_{K}_{prec_tag}_{q}_{method_tag}.log"


def rebuttal_mm_path(model: str, idx: int, prec_tag: str, q: str, method_tag: str) -> Path:
    M, K, N = MM_DIMS[(model, idx)]
    return LOG_REBUTTAL / f"{model}_MM{idx}_{M}_{K}_{N}_{prec_tag}_{q}_{method_tag}.log"


# ===========================================================================
# Data fetchers
# ===========================================================================

NA = "N/A"


def get_optimal_mvm(model: str, idx: int, batch: int, method_tag: str, prec: str, q: str):
    path = optimal_mvm_path(model, idx, batch)
    data = _load_optimal(path)
    gk, gn = Q_CONFIG[q]
    key = (METHOD_TO_HEADER[method_tag], PRECISION_TO_HEADER[prec], gk, gn)
    v = data.get(key)
    return v if v is not None else NA


def get_optimal_mm(model: str, idx: int, method_tag: str, prec: str, q: str):
    path = optimal_mm_path(model, idx)
    data = _load_optimal(path)
    gk, gn = Q_CONFIG[q]
    key = (METHOD_TO_HEADER[method_tag], PRECISION_TO_HEADER[prec], gk, gn)
    v = data.get(key)
    return v if v is not None else NA


def get_rebuttal_mvm(model: str, idx: int, batch: int, method_tag: str, prec: str, q: str):
    prec_tag = PRECISION_TO_FILETAG[prec]
    path = rebuttal_mvm_path(model, idx, batch, prec_tag, q, method_tag)
    return _load_rebuttal(path)   # (baseline, best)


def get_rebuttal_mm(model: str, idx: int, method_tag: str, prec: str, q: str):
    prec_tag = PRECISION_TO_FILETAG[prec]
    path = rebuttal_mm_path(model, idx, prec_tag, q, method_tag)
    return _load_rebuttal(path)   # (baseline, best)


def fmt(v):
    return NA if v is None else v


# ===========================================================================
# Table builders
# ===========================================================================

CSV_HEADER = [
    "operator", "batch_size", "q_config",
    "optimal_latency", "baseline_latency", "flexqndp_latency",
]


def build_table_1_2_4_5(method_tag: str, prec: str) -> list:
    """Tables 1, 2 (w_a, w4s8/w8s16) and 4, 5 (w_only, w4s8/w8s16).
    Rows: 7B_MVM_1/2/3 x batch {1,4,16,64} x q1-q6
          7B_MM_1/2/3  x q1-q2
    """
    rows = [CSV_HEADER]
    q_mvm = ["q1", "q2", "q3", "q4", "q5", "q6"]
    q_mm  = ["q1", "q2"]
    batches = [1, 4, 16, 64]
    model = "7B"

    for op_idx in [1, 2, 3]:
        op_name = f"7B_MVM{op_idx}"
        for batch in batches:
            for q in q_mvm:
                opt = get_optimal_mvm(model, op_idx, batch, method_tag, prec, q)
                base, best = get_rebuttal_mvm(model, op_idx, batch, method_tag, prec, q)
                rows.append([op_name, batch, q, opt, fmt(base), fmt(best)])

    for op_idx in [1, 2, 3]:
        op_name = f"7B_MM{op_idx}"
        for q in q_mm:
            opt = get_optimal_mm(model, op_idx, method_tag, prec, q)
            base, best = get_rebuttal_mm(model, op_idx, method_tag, prec, q)
            rows.append([op_name, NA, q, opt, fmt(base), fmt(best)])

    return rows


def build_table_3_6(method_tag: str) -> list:
    """Tables 3, 6 (w_a or w_only, w4s8, q1).
    Rows: 7B/13B/34B MVM x batch {1,2,4,8,16,32,64}
          7B/13B/34B MM
    """
    rows = [CSV_HEADER]
    prec = "w4s8"
    q = "q1"
    batches = [1, 2, 4, 8, 16, 32, 64]

    mvm_ops = [("7B", [1,2,3]), ("13B", [1,2,3]), ("34B", [1,2,3,4])]
    mm_ops  = [("7B", [1,2,3]), ("13B", [1,2,3]), ("34B", [1,2,3,4])]

    for model, indices in mvm_ops:
        for op_idx in indices:
            op_name = f"{model}_MVM{op_idx}"
            for batch in batches:
                opt = get_optimal_mvm(model, op_idx, batch, method_tag, prec, q)
                base, best = get_rebuttal_mvm(model, op_idx, batch, method_tag, prec, q)
                rows.append([op_name, batch, q, opt, fmt(base), fmt(best)])

    for model, indices in mm_ops:
        for op_idx in indices:
            op_name = f"{model}_MM{op_idx}"
            opt = get_optimal_mm(model, op_idx, method_tag, prec, q)
            base, best = get_rebuttal_mm(model, op_idx, method_tag, prec, q)
            rows.append([op_name, NA, q, opt, fmt(base), fmt(best)])

    return rows


# ===========================================================================
# Main
# ===========================================================================

def write_csv(filename: str, rows: list):
    out_path = OUTPUT_DIR / filename
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"  Written: {out_path}")


def main():
    print("Generating Fig7 CSV tables...")

    tables = [
        ("1_wa_w4s8.csv",   build_table_1_2_4_5("wa",    "w4s8")),
        ("2_wa_w8s16.csv",  build_table_1_2_4_5("wa",    "w8s16")),
        ("3_wa_w4s8_q1.csv",build_table_3_6("wa")),
        ("4_wonly_w4s8.csv",   build_table_1_2_4_5("wonly", "w4s8")),
        ("5_wonly_w8s16.csv",  build_table_1_2_4_5("wonly", "w8s16")),
        ("6_wonly_w4s8_q1.csv",build_table_3_6("wonly")),
    ]

    for filename, rows in tables:
        write_csv(filename, rows)

    print("Done.")


if __name__ == "__main__":
    main()
