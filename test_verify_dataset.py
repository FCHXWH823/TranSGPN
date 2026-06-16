"""
test_verify_dataset.py

Samples real rows from aggregate.csv files and runs the same verification
logic used in TransistorDataset._load. Lets you confirm the check works
on actual dataset entries before running the full 56-min BFS load.

Run: env/bin/python3 test_verify_dataset.py
"""
import csv
import os
import random
import re
from typing import List, Tuple

from transnet.literal import (
    ALL_VARS, parse_sop_expr, on_off_split,
    covered_patterns, check_safety, decode_literal, extract_vars,
)
from transnet.graph import sorted_g_transistors

DATASET_ROOTS = [
    "dataset/sweep_3input",
    "dataset/sweep_4input_possani",
]
N_SAMPLE = 5
RANDOM_SEED = 42


def _parse_cg(cg_str):
    nc = int(re.search(r"nc=(\d+)", cg_str).group(1))
    tr_str = re.search(r"tr=\[([^\]]*)\]", cg_str).group(1)
    tr = [tuple(int(x) for x in m)
          for m in re.findall(r"\((\d+),\s*(\d+),\s*(\d+)\)", tr_str)]
    return nc, tr


def _read_expr(data_root, fid, subdir):
    path = os.path.join(data_root, fid, subdir, "Booleans.txt")
    with open(path) as f:
        line = f.read().strip()
    return line.split(": ", 1)[1].strip()


def _is_4input_csv(csv_path):
    with open(csv_path) as f:
        return "impl_idx" in f.readline()


def verify_row(fid, expr, vars_in_func, nc, tr):
    s_tr = sorted_g_transistors(tr, nc)
    full_compact = [
        (u, v, *decode_literal(lit, vars_in_func))
        for u, v, lit in s_tr
    ]
    on_patterns  = parse_sop_expr(expr, vars_in_func)
    _, off_patterns = on_off_split(on_patterns)

    covered = covered_patterns(full_compact, nc, on_patterns)
    missing = on_patterns - covered
    safe    = check_safety(full_compact, nc, off_patterns)

    issues = []
    if missing:
        issues.append(f"missing {len(missing)} on-pattern(s)")
    if not safe:
        issues.append("conducts under off-pattern")
    return (not issues), issues


rng = random.Random(RANDOM_SEED)

for root in DATASET_ROOTS:
    csv_path = os.path.join(root, "aggregate.csv")
    if not os.path.exists(csv_path):
        print(f"[SKIP] {csv_path} not found")
        continue

    is_4input = _is_4input_csv(csv_path)
    sat_status = "OK" if is_4input else "SAT"
    dataset_name = os.path.basename(root)

    with open(csv_path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["status"] == sat_status]

    sample = rng.sample(rows, min(N_SAMPLE, len(rows)))

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}  ({len(rows)} rows total, sampling {len(sample)})")
    print(f"{'='*60}")

    for row in sample:
        fid = row["function_id"]
        nc, tr = _parse_cg(row["cg"])

        if is_4input:
            expr = row["sop"]
            vars_in_func = ALL_VARS
            label = f"impl_{row['impl_idx']}"
        else:
            subdir = f"t_{row['t']}"
            expr = _read_expr(root, fid, subdir)
            vars_in_func = extract_vars(expr)
            label = subdir

        ok, issues = verify_row(fid, expr, vars_in_func, nc, tr)
        n_on = len(parse_sop_expr(expr, vars_in_func))
        tag = "\033[32mOK\033[0m" if ok else "\033[31mFAIL\033[0m"
        issue_str = "  ← " + "; ".join(issues) if issues else ""
        print(f"  [{tag}] {fid}/{label}  expr={expr!r}  "
              f"nc={nc} tr={len(tr)} on={n_on}{issue_str}")
