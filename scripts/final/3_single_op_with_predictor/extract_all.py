#!/usr/bin/env python3
"""
Extract all data from log_rebuttal_mm_new into the same column format as
operator_latency_detailed.csv, also joining optimal_latency from log_optimal.

Output: rebuttal_latency_detailed.csv  (same directory as this script)
"""

import ast
import re
import csv
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LOG_REBUTTAL = SCRIPT_DIR / "log_rebuttal_mm_new"
LOG_OPTIMAL  = SCRIPT_DIR / "log_optimal"
OUTPUT          = SCRIPT_DIR / "operator_latency_detailed.csv"
OUTPUT_STRATEGY = SCRIPT_DIR / "best_partition_strategy.csv"

# Precision tag in filename  ->  bitwidth string used in headers and CSV
PREC_TAG_TO_BITWIDTH = {
    "w4s8":  "D4S8",
    "w8a16": "D8S16",
}

# Method tag in filename  ->  quant_method string
METHOD_TAG_TO_NAME = {
    "wa":    "w_a",
    "wonly": "w_only",
}

CSV_FIELDNAMES = [
    "model_size", "op_name", "M", "K", "N",
    "quant_method", "bitwidth", "group_k", "group_n",
    "baseline_latency", "best_latency", "optimal_latency", "best_strategy",
]

# ── Filename parsing ──────────────────────────────────────────────────────────

# MVM batch=1:  7B_MVM1_1_4096_4096_w4s8_q1_wa.log
_RE_MVM1 = re.compile(
    r'^(\d+B)_(MVM\d+)_1_(\d+)_(\d+)_(w4s8|w8a16)_(q\d+)_(wa|wonly)\.log$'
)
# MVM batch>1:  7B_MVM1_B4_4_4096_4096_w4s8_q1_wa.log
_RE_MVMB = re.compile(
    r'^(\d+B)_(MVM\d+)_B(\d+)_\d+_(\d+)_(\d+)_(w4s8|w8a16)_(q\d+)_(wa|wonly)\.log$'
)
# MM:           7B_MM1_4096_4096_4096_w4s8_q1_wa.log
_RE_MM = re.compile(
    r'^(\d+B)_(MM\d+)_(\d+)_(\d+)_(\d+)_(w4s8|w8a16)_(q\d+)_(wa|wonly)\.log$'
)


def parse_filename(name: str):
    """
    Returns a dict with keys:
      model_size, op_name, M, K, N, prec_tag, q_tag, method_tag
    or None if the name doesn't match any known pattern.
    """
    m = _RE_MVM1.match(name)
    if m:
        model, op, K, N, prec, q, meth = m.groups()
        return dict(model_size=model, op_name=f"{model}_{op}",
                    M="1", K=K, N=N,
                    prec_tag=prec, q_tag=q, method_tag=meth,
                    op_type="MVM", batch=1)

    m = _RE_MVMB.match(name)
    if m:
        model, op, batch, K, N, prec, q, meth = m.groups()
        return dict(model_size=model, op_name=f"{model}_{op}_B{batch}",
                    M=batch, K=K, N=N,
                    prec_tag=prec, q_tag=q, method_tag=meth,
                    op_type="MVM", batch=int(batch))

    m = _RE_MM.match(name)
    if m:
        model, op, M, K, N, prec, q, meth = m.groups()
        return dict(model_size=model, op_name=f"{model}_{op}",
                    M=M, K=K, N=N,
                    prec_tag=prec, q_tag=q, method_tag=meth,
                    op_type="MM", batch=None)

    return None


# ── Log content parsing ───────────────────────────────────────────────────────

_HEADER_RE   = re.compile(r'=====+\s+(\S+)\s+(\S+)\s+group_k=(\d+)\s+group_n=(\d+)\s+=====+')
_BASELINE_RE = re.compile(r'==>\s+Baseline\s+\d+\s+Latency:\s+([\d.]+)')
_BEST_RE     = re.compile(r'==>\s+Best Latency:\s+([\d.]+),\s*Best Partition Specify:\s*(.*?),\s*Best Buffer Specify:\s*(\[.*?\])')
_OPTIMAL_RE  = re.compile(r'==>\s+Optimal Latency:\s+([\d.]+)')


def parse_rebuttal_log(path: Path) -> dict:
    """
    Returns dict with keys:
      quant_method, bitwidth, group_k, group_n,
      baseline_latency, best_latency, best_strategy
    """
    result = {}
    with open(path) as f:
        content = f.read()

    m = _HEADER_RE.search(content)
    if m:
        result["quant_method"] = m.group(1)
        result["bitwidth"]     = m.group(2)
        result["group_k"]      = m.group(3)
        result["group_n"]      = m.group(4)

    m = _BASELINE_RE.search(content)
    if m:
        result["baseline_latency"] = m.group(1)

    m = _BEST_RE.search(content)
    if m:
        result["best_latency"] = m.group(1)
        partition = m.group(2).strip()
        buf       = m.group(3).strip()
        result["best_strategy"] = f"Partition:{partition}, Buffer:{buf}"

    return result


# ── Optimal latency lookup ────────────────────────────────────────────────────

_optimal_cache: dict = {}


def _load_optimal(path: Path) -> dict:
    """Parse log_optimal file → {(quant_method, bitwidth, group_k, group_n): latency}"""
    if path in _optimal_cache:
        return _optimal_cache[path]
    result = {}
    if not path.exists():
        _optimal_cache[path] = result
        return result
    with open(path) as f:
        content = f.read()
    sections = re.split(r'(=====+\s+\S+\s+\S+\s+group_k=\d+\s+group_n=\d+\s+=====+)', content)
    for i in range(1, len(sections), 2):
        hdr = sections[i]
        body = sections[i + 1] if i + 1 < len(sections) else ""
        hm = _HEADER_RE.search(hdr)
        om = _OPTIMAL_RE.search(body)
        if hm and om:
            key = (hm.group(1), hm.group(2), hm.group(3), hm.group(4))
            result[key] = om.group(1)
    _optimal_cache[path] = result
    return result


def get_optimal_latency(meta: dict, quant_method: str, bitwidth: str,
                        group_k: str, group_n: str) -> str:
    if meta["op_type"] == "MVM":
        batch = meta["batch"]
        bstr  = "1" if batch == 1 else f"B{batch}_{batch}"
        fname = f"{meta['model_size']}_{meta['op_name'].split('_')[1]}_{bstr}_{meta['K']}_{meta['N']}.log"
    else:
        fname = f"{meta['model_size']}_{meta['op_name'].split('_')[1]}_{meta['M']}_{meta['K']}_{meta['N']}.log"

    # op_name may include _B{n} suffix – strip to get just the op part (e.g. MVM1)
    op_bare = re.sub(r'_B\d+$', '', meta['op_name'].split(f"{meta['model_size']}_")[1])
    if meta["op_type"] == "MVM":
        batch = meta["batch"]
        bstr  = "1" if batch == 1 else f"B{batch}_{batch}"
        fname = f"{meta['model_size']}_{op_bare}_{bstr}_{meta['K']}_{meta['N']}.log"
    else:
        fname = f"{meta['model_size']}_{op_bare}_{meta['M']}_{meta['K']}_{meta['N']}.log"

    data = _load_optimal(LOG_OPTIMAL / fname)
    key = (quant_method, bitwidth, group_k, group_n)
    return data.get(key, "")


# ── Strategy CSV helpers ──────────────────────────────────────────────────────

def op_key_from_meta(meta: dict) -> str:
    """Build canonical op key matching the log filename base (no prec/q/method suffix)."""
    model   = meta["model_size"]
    op_bare = re.sub(r'_B\d+$', '', meta["op_name"].split(f"{model}_")[1])
    if meta["op_type"] == "MVM":
        bstr = "1" if meta["batch"] == 1 else f"B{meta['batch']}_{meta['batch']}"
        return f"{model}_{op_bare}_{bstr}_{meta['K']}_{meta['N']}"
    else:
        return f"{model}_{op_bare}_{meta['M']}_{meta['K']}_{meta['N']}"


def partition_to_tuple_str(raw: str) -> str:
    """Convert '[[1,32,2,1],[1,1,1,1],...]' → '((1, 32, 2, 1), (1, 1, 1, 1), ...)'."""
    try:
        parsed = ast.literal_eval(raw.strip())
        inner  = ", ".join(f"({', '.join(str(x) for x in row)})" for row in parsed)
        return f"({inner})"
    except Exception:
        return raw


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    rows = []
    skipped = []

    # strategy_data: op_key -> {"M": .., "K": .., "N": .., "q1_wa": .., "q2_wa": .., "q1_wonly": .., "q2_wonly": ..}
    strategy_data: dict = {}

    for log_path in sorted(LOG_REBUTTAL.glob("*.log")):
        meta = parse_filename(log_path.name)
        if meta is None:
            skipped.append(log_path.name)
            continue

        content = parse_rebuttal_log(log_path)

        quant_method = content.get("quant_method", "")
        bitwidth     = content.get("bitwidth", "")
        group_k      = content.get("group_k", "")
        group_n      = content.get("group_n", "")

        # Prefer values parsed from file content; fall back to filename-derived ones
        if not bitwidth:
            bitwidth = PREC_TAG_TO_BITWIDTH.get(meta["prec_tag"], "")
        if not quant_method:
            quant_method = METHOD_TAG_TO_NAME.get(meta["method_tag"], "")

        optimal = get_optimal_latency(meta, quant_method, bitwidth, group_k, group_n)

        rows.append({
            "model_size":       meta["model_size"],
            "op_name":          meta["op_name"],
            "M":                meta["M"],
            "K":                meta["K"],
            "N":                meta["N"],
            "quant_method":     quant_method,
            "bitwidth":         bitwidth,
            "group_k":          group_k,
            "group_n":          group_n,
            "baseline_latency": content.get("baseline_latency", ""),
            "best_latency":     content.get("best_latency", ""),
            "optimal_latency":  optimal,
            "best_strategy":    content.get("best_strategy", ""),
        })

        # Collect partition strategy for best_partition_strategy.csv
        # Only w4s8, q1 or q2, both methods, 7B and 34B only
        # For MVM: only batch=1 and batch=16
        if meta["prec_tag"] == "w4s8" and meta["q_tag"] in ("q1", "q2") \
                and meta["model_size"] in ("7B", "34B") \
                and (meta["op_type"] == "MM" or meta["batch"] in (1, 16)):
            raw_strategy = content.get("best_strategy", "")
            partition = ""
            if raw_strategy:
                pm = re.search(r'Partition:(.*?),\s*Buffer:', raw_strategy)
                if pm:
                    partition = partition_to_tuple_str(pm.group(1).strip())

            key = op_key_from_meta(meta)
            if key not in strategy_data:
                strategy_data[key] = {"M": meta["M"], "K": meta["K"], "N": meta["N"],
                                      "q1_wa": "", "q2_wa": "", "q1_wonly": "", "q2_wonly": ""}
            slot = f"{meta['q_tag']}_{meta['method_tag']}"  # e.g. "q1_wa"
            strategy_data[key][slot] = partition

    if skipped:
        print(f"Skipped {len(skipped)} non-matching files: {skipped[:5]}{'...' if len(skipped) > 5 else ''}")

    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    has_base    = sum(1 for r in rows if r["baseline_latency"])
    has_best    = sum(1 for r in rows if r["best_latency"])
    has_optimal = sum(1 for r in rows if r["optimal_latency"])
    has_all     = sum(1 for r in rows if r["baseline_latency"] and r["best_latency"] and r["optimal_latency"])

    print(f"Written {len(rows)} rows to {OUTPUT}")
    print(f"  baseline_latency:  {has_base}/{len(rows)}")
    print(f"  best_latency:      {has_best}/{len(rows)}")
    print(f"  optimal_latency:   {has_optimal}/{len(rows)}")
    print(f"  all three present: {has_all}/{len(rows)}")

    # ── Write best_partition_strategy.csv ─────────────────────────────────────
    with open(OUTPUT_STRATEGY, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for op_key, v in sorted(strategy_data.items()):
            writer.writerow([
                op_key, v["M"], v["K"], v["N"],
                v["q1_wa"], v["q2_wa"], v["q1_wonly"], v["q2_wonly"],
            ])

    print(f"Written {len(strategy_data)} rows to {OUTPUT_STRATEGY}")


if __name__ == "__main__":
    main()
