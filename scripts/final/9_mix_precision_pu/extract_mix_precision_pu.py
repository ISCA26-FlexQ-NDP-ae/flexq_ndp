#!/usr/bin/env python3
"""
Extract latency summary for mixed-precision PU experiments.

Data sources
------------
MIX_LOG_DIR  (9_mix_precision_pu/log/)
    Each file contains one config block with:
      Baseline 0 Latency  →  FP16×FP4 PU + Baseline
      Best Latency        →  FP16×FP4 PU + FlexQ-NDP

REBUTTAL_DIR (3_single_op_with_predictor/log_rebuttal_mm_new/)
    Matching *_w4s8_q1_wa.log files (w_a D4S8 group_k=16 group_n=1):
      Baseline 0 Latency  →  FP16×FP16 PU + Baseline

Output columns (per operator, in order)
----------------------------------------
  operator | quant_config
  | fp16x16_pu_baseline | fp16x4_pu_baseline | fp16x4_pu_flexqndp
  | speedup_fp16x4_pu   | further_speedup_flexqndp
"""

import re
import csv
from pathlib import Path

SCRIPT_DIR   = Path(__file__).parent
MIX_LOG_DIR  = SCRIPT_DIR / "log"
REBUTTAL_DIR = SCRIPT_DIR.parent / "3_single_op_with_predictor" / "log_rebuttal_mm_new"
OUTPUT       = SCRIPT_DIR / "mix_precision_pu_summary.csv"

_HEADER_RE  = re.compile(r'=====+\s+(\S+)\s+(D\d+S\d+)\s+group_k=(\d+)\s+group_n=(\d+)\s+=====+')
_BL0_RE     = re.compile(r'==>\s+Baseline 0 Latency:\s+([\d.]+)')
_BEST_RE    = re.compile(r'==>\s+Best Latency:\s+([\d.]+)')

NA = "N/A"


def parse_log(path: Path) -> dict:
    """Return {quant_config_str: {bl0, best}} for each config block in the file."""
    results = {}
    cur_cfg = None
    with open(path) as f:
        for line in f:
            line_s = line.strip()
            m = _HEADER_RE.match(line_s)
            if m:
                cur_cfg = f"{m.group(1)} {m.group(2)} gk={m.group(3)},gn={m.group(4)}"
                results.setdefault(cur_cfg, {})
                continue
            if cur_cfg is None:
                continue
            m = _BL0_RE.search(line_s)
            if m:
                results[cur_cfg]['bl0'] = float(m.group(1))
                continue
            m = _BEST_RE.search(line_s)
            if m:
                results[cur_cfg]['best'] = float(m.group(1))
    return results


def operator_from_stem(stem: str) -> str:
    """'7B_MVM1_1_4096_4096' → '7B_MVM1'"""
    parts = stem.split('_')
    return f"{parts[0]}_{parts[1]}"


def fmt(v):
    return f"{v:.4f}" if isinstance(v, float) else v


def main():
    header = [
        "operator",
        "fp16x16_pu_baseline",
        "fp16x4_pu_baseline",
        "fp16x4_pu_flexqndp",
        "speedup_fp16x4_pu",
        "further_speedup_flexqndp",
    ]
    rows = []

    for mix_path in sorted(MIX_LOG_DIR.glob("*.log")):
        stem     = mix_path.stem                           # e.g. 7B_MVM1_1_4096_4096
        op_name  = operator_from_stem(stem)
        mix_data = parse_log(mix_path)

        # Corresponding FP16×FP16 PU file
        rebuttal_path = REBUTTAL_DIR / f"{stem}_w4s8_q1_wonly.log"
        rebuttal_data = parse_log(rebuttal_path) if rebuttal_path.exists() else {}

        for cfg, mix_entry in mix_data.items():
            fp4_bl   = mix_entry.get('bl0',  NA)
            fp4_best = mix_entry.get('best', NA)

            # Rebuttal file uses same config key; fall back to first entry if present
            reb_entry = (rebuttal_data.get(cfg)
                         or (next(iter(rebuttal_data.values())) if rebuttal_data else {}))
            fp16_bl = reb_entry.get('bl0', NA) if reb_entry else NA

            # Speedups
            if isinstance(fp16_bl, float) and isinstance(fp4_bl, float) and fp4_bl:
                spd_pu = fp16_bl / fp4_bl
            else:
                spd_pu = NA

            if isinstance(fp4_bl, float) and isinstance(fp4_best, float) and fp4_best:
                spd_flexq = fp4_bl / fp4_best
            else:
                spd_flexq = NA

            rows.append([
                op_name,
                fmt(fp16_bl),
                fmt(fp4_bl),
                fmt(fp4_best),
                fmt(spd_pu),
                fmt(spd_flexq),
            ])

            if not rebuttal_path.exists():
                print(f"  Warning: rebuttal file not found: {rebuttal_path.name}")

    with open(OUTPUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"Written: {OUTPUT}  ({len(rows)} data rows)")


if __name__ == "__main__":
    main()
