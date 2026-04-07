#!/usr/bin/env python3
"""
Extract speedup under different PU throughput scenarios into a CSV.

Data sources
------------
1x  throughput  → log_rebuttal_mm_new/{stem}_w4s8_q1_wa.log   (Speedup: field)
0.5x throughput → 6_throughput_change/log/{stem}.log           (0.5x section Speedup:)
0.25x throughput→ 6_throughput_change/log/{stem}.log           (0.25x section Speedup:)

Output columns (per operator)
------------------------------
  operator | speedup_1x | speedup_0.5x | speedup_0.25x
"""

import re
import csv
from pathlib import Path

SCRIPT_DIR    = Path(__file__).parent
TPUT_LOG_DIR  = SCRIPT_DIR / "log"
REBUTTAL_DIR  = SCRIPT_DIR.parent / "3_single_op_with_predictor" / "log_rebuttal_mm_new"
OUTPUT        = SCRIPT_DIR / "throughput_speedup.csv"

_TPUT_HDR_RE  = re.compile(r'run pu with ([\d.]+)x fp32 throughput')
_SPEEDUP_RE   = re.compile(r'Speedup:\s*([\d.]+)')

NA = "N/A"


def operator_from_stem(stem: str) -> str:
    parts = stem.split('_')
    return f"{parts[0]}_{parts[1]}"


def parse_rebuttal(path: Path) -> str:
    """Return the Speedup value from a log_rebuttal_mm_new file."""
    if not path.exists():
        return NA
    text = path.read_text()
    m = _SPEEDUP_RE.search(text)
    return m.group(1) if m else NA


def parse_throughput_log(path: Path) -> dict:
    """Return {throughput_label: speedup_str} e.g. {'0.5': '1.28', '0.25': '1.15'}"""
    if not path.exists():
        return {}
    text   = path.read_text()
    result = {}

    # Split into sections by throughput header
    sections = re.split(r'==> run pu with ', text)
    for sec in sections[1:]:
        m_hdr = re.match(r'([\d.]+)x fp32 throughput', sec)
        if not m_hdr:
            continue
        label = m_hdr.group(1)
        m_spd = _SPEEDUP_RE.search(sec)
        result[label] = m_spd.group(1) if m_spd else NA

    return result


def main():
    header = ["operator", "speedup_1x", "speedup_0.5x", "speedup_0.25x"]
    rows   = []

    for tput_path in sorted(TPUT_LOG_DIR.glob("*.log")):
        stem        = tput_path.stem
        op          = operator_from_stem(stem)
        rebuttal_p  = REBUTTAL_DIR / f"{stem}_w4s8_q1_wa.log"

        spd_1x   = parse_rebuttal(rebuttal_p)
        tput_data = parse_throughput_log(tput_path)
        spd_half  = tput_data.get('0.5',  NA)
        spd_qtr   = tput_data.get('0.25', NA)

        rows.append([op, spd_1x, spd_half, spd_qtr])

        if not rebuttal_p.exists():
            print(f"  Warning: rebuttal file not found: {rebuttal_p.name}")

    with open(OUTPUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"Written: {OUTPUT}  ({len(rows)} data rows)")


if __name__ == "__main__":
    main()
