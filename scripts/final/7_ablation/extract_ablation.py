#!/usr/bin/env python3
"""
Summarise log_ablation logs into a CSV.

Columns per operator
--------------------
  operator | batch_size
  | lat_baseline | lat_mapping | lat_mapping_reorder | lat_dse
  | speedup_mapping_vs_baseline
  | speedup_reorder_vs_mapping
  | speedup_dse_vs_reorder

Final row: geometric mean of each speedup column.
"""

import re
import csv
from math import prod
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
LOG_DIR    = SCRIPT_DIR / "log_ablation"
OUTPUT     = SCRIPT_DIR / "ablation_summary.csv"

_BL_RE     = re.compile(r'\[Baseline\] Latency:\s*(\d+)')
_MAP_RE    = re.compile(r'\[Mapping\] Latency:\s*(\d+)')
_REORD_RE  = re.compile(r'\[Mapping \+ Reorder\] Latency:\s*(\d+)')
_DSE_RE    = re.compile(r'\[DSE \+ Partition \+ Reorder\] Latency:\s*(\d+)')

NA = "N/A"


def parse_log(path: Path) -> dict:
    data = {}
    text = path.read_text()
    for key, pat in (('baseline', _BL_RE), ('mapping', _MAP_RE),
                     ('reorder',  _REORD_RE), ('dse',    _DSE_RE)):
        m = pat.search(text)
        if m:
            data[key] = int(m.group(1))
    return data


def operator_and_batch(stem: str):
    """'13B_MVM1_B4_4_5120_5120' → ('13B_MVM1', 4)
       '13B_MVM1_1_5120_5120'    → ('13B_MVM1', 1)"""
    parts = stem.split('_')
    op    = f"{parts[0]}_{parts[1]}"
    bs_part = parts[2]                         # '1' or 'B4'
    batch = int(bs_part[1:]) if bs_part.startswith('B') else int(bs_part)
    return op, batch


def fmt(v, decimals=4):
    return f"{v:.{decimals}f}" if isinstance(v, float) else v


def geomean(vals):
    n = len(vals)
    return prod(vals) ** (1 / n) if n else 0.0


def main():
    header = [
        "operator", "batch_size",
        "lat_baseline", "lat_mapping", "lat_mapping_reorder", "lat_dse",
        "speedup_mapping_vs_baseline",
        "speedup_reorder_vs_mapping",
        "speedup_dse_vs_reorder",
    ]

    # Collect rows per batch size
    from collections import defaultdict
    by_batch = defaultdict(list)          # batch → list of row dicts
    spd_by_batch = defaultdict(lambda: ([], [], []))  # batch → (map, reord, dse)

    for log_path in sorted(LOG_DIR.glob("*.log")):
        op, batch = operator_and_batch(log_path.stem)
        d = parse_log(log_path)

        bl   = d.get('baseline', None)
        mp   = d.get('mapping',  None)
        ro   = d.get('reorder',  None)
        dse  = d.get('dse',      None)

        spd_map   = bl  / mp   if bl  and mp  else NA
        spd_reord = mp  / ro   if mp  and ro  else NA
        spd_dse   = ro  / dse  if ro  and dse else NA

        m_list, r_list, d_list = spd_by_batch[batch]
        if isinstance(spd_map,   float): m_list.append(spd_map)
        if isinstance(spd_reord, float): r_list.append(spd_reord)
        if isinstance(spd_dse,   float): d_list.append(spd_dse)

        by_batch[batch].append([
            op, batch,
            bl  if bl  is not None else NA,
            mp  if mp  is not None else NA,
            ro  if ro  is not None else NA,
            dse if dse is not None else NA,
            fmt(spd_map),
            fmt(spd_reord),
            fmt(spd_dse),
        ])

    all_rows = [header]
    for batch in sorted(by_batch):
        all_rows.extend(by_batch[batch])
        m_list, r_list, d_list = spd_by_batch[batch]
        all_rows.append([
            f"AVERAGE (BS={batch})", "",
            "", "", "", "",
            fmt(geomean(m_list)),
            fmt(geomean(r_list)),
            fmt(geomean(d_list)),
        ])
        all_rows.append([])   # blank separator between tables

    with open(OUTPUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(all_rows)

    n_data = sum(len(v) for v in by_batch.values())
    print(f"Written: {OUTPUT}  ({n_data} data rows, {len(by_batch)} batch-size tables)")


if __name__ == "__main__":
    main()
