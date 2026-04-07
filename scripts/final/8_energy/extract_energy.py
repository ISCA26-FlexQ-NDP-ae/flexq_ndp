#!/usr/bin/env python3
"""
Extract baseline and optimized energy from energy_log for all MVM operators.

Log format (new):
  ========== w_a D4S8 group_k=16 group_n=1 ==========
  ==> Baseline Energy:
  0.241748...
  ==> Best Energy:
  0.179229...
  ==> Baseline Latency: 4537.0
  ==> Best Latency: 2410.0, ...

Outputs: energy_mvm.csv  (one row per operator × precision found in log)
"""

import re
import csv
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
LOG_DIR    = SCRIPT_DIR / "energy_log"
OUTPUT     = SCRIPT_DIR / "energy_mvm.csv"

OPERATORS = [
    ("7B",  1), ("7B",  2), ("7B",  3),
    ("13B", 1), ("13B", 2), ("13B", 3),
    ("34B", 1), ("34B", 2), ("34B", 3), ("34B", 4),
]

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

_HEADER_RE   = re.compile(r'=====+\s+w_a\s+(D\d+S\d+)\s+group_k=(\d+)\s+group_n=(\d+)\s+=====+')
_FLOAT_RE    = re.compile(r'^[\d.]+(?:[eE][+-]?\d+)?$')
_BL_LAT_RE   = re.compile(r'==>\s+Baseline Latency:\s+([\d.]+)')
_BEST_LAT_RE = re.compile(r'==>\s+Best Latency:\s+([\d.]+)')

NA = "N/A"


def parse_log(path: Path) -> dict:
    """
    Returns dict keyed by precision string (e.g. 'D4S8'):
      { 'baseline': float, 'best': float,
        'baseline_latency': float, 'best_latency': float }
    """
    result = {}
    current_prec = None
    next_energy  = None   # 'baseline' or 'best'

    with open(path) as f:
        for line in f:
            line_s = line.strip()

            m = _HEADER_RE.match(line_s)
            if m:
                current_prec = m.group(1)
                result.setdefault(current_prec, {})
                next_energy = None
                continue

            if current_prec is None:
                continue

            if "==> Baseline Energy:" in line_s:
                next_energy = "baseline"
                continue
            if "==> Best Energy:" in line_s:
                next_energy = "best"
                continue

            if next_energy and _FLOAT_RE.match(line_s):
                result[current_prec][next_energy] = float(line_s)
                next_energy = None
                continue

            m = _BL_LAT_RE.search(line_s)
            if m:
                result[current_prec]["baseline_latency"] = float(m.group(1))
                continue

            m = _BEST_LAT_RE.search(line_s)
            if m:
                result[current_prec]["best_latency"] = float(m.group(1))

    return result


def main():
    CSV_HEADER = [
        "operator", "precision",
        "baseline_energy", "baseline_latency",
        "best_energy",     "best_latency",
    ]
    rows = [CSV_HEADER]

    for model, idx in OPERATORS:
        M, K      = MVM_DIMS[(model, idx)]
        log_path  = LOG_DIR / f"{model}_MVM{idx}_1_{M}_{K}.log"
        op_name   = f"{model}_MVM{idx}"

        if not log_path.exists():
            print(f"  Missing: {log_path.name}")
            rows.append([op_name, NA, NA, NA, NA, NA])
            continue

        data = parse_log(log_path)
        if not data:
            print(f"  No data parsed: {log_path.name}")
            rows.append([op_name, NA, NA, NA, NA, NA])
            continue

        for prec, entry in data.items():
            rows.append([
                op_name, prec,
                round(entry.get("baseline",         NA)),
                round(entry.get("baseline_latency", NA)),
                round(entry.get("best",             NA)),
                round(entry.get("best_latency",     NA)),
            ])

    with open(OUTPUT, "w", newline="") as f:
        csv.writer(f).writerows(rows)

    print(f"Written: {OUTPUT}  ({len(rows)-1} data rows)")


if __name__ == "__main__":
    main()
