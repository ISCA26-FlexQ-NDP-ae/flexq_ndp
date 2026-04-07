#!/usr/bin/env python3
"""
Extract buffer-change experiment results from 7B_MVM2_1_4096_11008.log into two CSVs:

  fig11a.csv  – latency per buffer size
    columns: buffer_size, baseline(with_dse), layout(with_dse), layout+reorder(with_dse)

  fig11b.csv  – speedup per buffer size
    columns: buffer_size, speedup
"""

import re
import csv
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
LOG_FILE   = SCRIPT_DIR / "log" / "7B_MVM2_1_4096_11008.log"
OUT_A      = SCRIPT_DIR / "fig11a.csv"
OUT_B      = SCRIPT_DIR / "fig11b.csv"

_HDR_RE    = re.compile(r'buffer_size=(\d+)')
_BL_RE     = re.compile(r'Baseline Latency:\s*(\d+)')
_MX_RE     = re.compile(r'With MX Best Latency:\s*(\d+)')
_REORD_RE  = re.compile(r'With Reorder Best Latency:\s*(\d+)')
_SPD_RE    = re.compile(r'^Speedup:\s*([\d.]+)', re.MULTILINE)

NA = "N/A"


def parse_log(path: Path):
    text     = path.read_text()
    sections = re.split(r'(?=={5,}.*buffer_size=)', text)
    rows     = []

    for sec in sections:
        m_hdr = _HDR_RE.search(sec)
        if not m_hdr:
            continue
        buf = int(m_hdr.group(1))

        def get(pat):
            m = pat.search(sec)
            return int(m.group(1)) if m else NA

        def get_f(pat):
            m = pat.search(sec)
            return m.group(1) if m else NA

        rows.append({
            'buffer_size':        buf,
            'baseline':  get(_BL_RE),
            'layout':    get(_MX_RE),
            'layout_reorder': get(_REORD_RE),
            'speedup':            get_f(_SPD_RE),
        })

    return sorted(rows, key=lambda r: r['buffer_size'])


def main():
    rows = parse_log(LOG_FILE)

    with open(OUT_A, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['buffer_size', 'baseline', 'layout', 'layout+reorder'])
        for r in rows:
            w.writerow([r['buffer_size'], r['baseline'],
                        r['layout'], r['layout_reorder']])

    with open(OUT_B, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['buffer_size', 'speedup'])
        for r in rows:
            w.writerow([r['buffer_size'], r['speedup']])

    print(f"Written: {OUT_A}  ({len(rows)} rows)")
    print(f"Written: {OUT_B}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
