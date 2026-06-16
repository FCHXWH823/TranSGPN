"""
transnet/dataset.py

TransistorDataset: reads sweep_3input/aggregate.csv and pre-expands
each SAT result into BFS prefix training samples.

Each sample is a dict:
  "graph"       : torchdrug.data.Graph  (prefix-k G ∪ full C)
  "node1"       : int  compact source index
  "node2"       : int  compact dest index, or g_num_nodes for a new node
  "var"         : int  global variable index 0..N_VARS-1
  "neg"         : int  0=positive, 1=negative
  "safety_mask" : BoolTensor [N_VARS, 2]  — True where (var, neg) is safe
                  to add for this edge (no SRC→SNK path under any P-)
"""
from __future__ import annotations

import csv
import os
import re
from typing import List

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .graph import MAX_G_NODES, build_gen_graph, build_prefix_graph, sorted_g_transistors
from .literal import (
    ALL_VARS, N_VARS,
    check_safety, covered_patterns, decode_literal, extract_vars, on_off_split, parse_sop_expr,
)


_DATASET_ROOTS = [
    os.path.join(os.path.dirname(__file__), "..", "dataset", "sweep_3input"),
    os.path.join(os.path.dirname(__file__), "..", "dataset", "sweep_4input_possani"),
]


def _parse_cg(cg_str: str):
    """Parse 'CG(nc=N, r=R, tr=[(s,d,l),...])' → (nc, tr_list)."""
    nc = int(re.search(r"nc=(\d+)", cg_str).group(1))
    tr_str = re.search(r"tr=\[([^\]]*)\]", cg_str).group(1)
    tr = [
        tuple(int(x) for x in m)
        for m in re.findall(r"\((\d+),\s*(\d+),\s*(\d+)\)", tr_str)
    ]
    return nc, tr


def _read_expr(data_root: str, function_id: str, subdir: str) -> str:
    """Read SOP expression from Booleans.txt."""
    path = os.path.join(data_root, function_id, subdir, "Booleans.txt")
    with open(path) as f:
        line = f.read().strip()
    # Format: "PF3_XXX(t): <expr>"  or  "PF4_XXX(t): <expr>"
    return line.split(": ", 1)[1].strip()


def _is_4input_csv(csv_path: str) -> bool:
    """Detect format by checking for 'impl_idx' column in header."""
    with open(csv_path) as f:
        header = f.readline()
    return "impl_idx" in header


class TransistorDataset(Dataset):
    """
    Pre-expanded BFS prefix training dataset for transistor-network synthesis.

    Supports sweep_3input (status=SAT, t_N/ subdirs) and
    sweep_4input_possani (status=OK, impl_N/ subdirs, sop column).
    """

    def __init__(self, dataset_roots: List[str] = None, cache_path: str = None,
                 verify: bool = True):
        if dataset_roots is None:
            dataset_roots = _DATASET_ROOTS
        self.samples: List[dict] = []

        if cache_path and os.path.exists(cache_path):
            print(f"Loading dataset from cache: {cache_path}")
            raw = torch.load(cache_path, weights_only=False)
            print(f"  {len(raw)} samples loaded, rebuilding graphs …")
            self.samples = self._build_graphs(raw, desc="Rebuilding graphs")
            return

        # Full build from CSVs (slow — BFS safety checks)
        raw: List[dict] = []
        for root in dataset_roots:
            csv_path = os.path.join(root, "aggregate.csv")
            if not os.path.exists(csv_path):
                continue
            self._load(root, csv_path, raw, verify=verify)

        if cache_path:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            print(f"Saving dataset cache to: {cache_path}")
            torch.save(raw, cache_path)
            print(f"  Saved {len(raw)} raw samples.")

        self.samples = self._build_graphs(raw, desc="Building graphs")

    @staticmethod
    def _build_graphs(raw: List[dict], desc: str = "Building graphs") -> List[dict]:
        """Reconstruct Graph objects from raw data (no BFS, fast)."""
        samples = []
        for entry in tqdm(raw, desc=desc, unit="sample"):
            on_patterns = frozenset(map(tuple, entry["on_patterns"]))
            graph = build_gen_graph(
                entry["compact_tran"], entry["g_num"],
                entry["vars_in_func"], on_patterns,
            )
            samples.append({
                "graph":             graph,
                "node1":             entry["node1"],
                "node2":             entry["node2"],
                "var":               entry["var"],
                "neg":               entry["neg"],
                "node2_safety_mask": entry["node2_safety_mask"],
                "safety_mask":       entry["safety_mask"],
            })
        return samples

    def _load(self, data_root: str, csv_path: str, raw: List[dict],
              verify: bool = True) -> None:
        is_4input = _is_4input_csv(csv_path)
        sat_status = "OK" if is_4input else "SAT"

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = [r for r in reader if r["status"] == sat_status]

        dataset_name = os.path.basename(data_root)
        n_fail = 0
        for row in tqdm(rows, desc=f"BFS {dataset_name}", unit="fn"):
            fid = row["function_id"]
            nc, tr = _parse_cg(row["cg"])
            if is_4input:
                expr = row["sop"]
                subdir = f"impl_{row['impl_idx']}"
                # 4-input uses global literal IDs (a=0..d=3, !a=4..!d=7)
                # so vars_in_func must always be ALL_VARS for correct decoding
                vars_in_func = ALL_VARS
            else:
                subdir = f"t_{row['t']}"
                expr = _read_expr(data_root, fid, subdir)
                vars_in_func = extract_vars(expr)
            on_patterns = parse_sop_expr(expr, vars_in_func)
            _, off_patterns = on_off_split(on_patterns)

            s_tr = sorted_g_transistors(tr, nc)
            R = len(s_tr)

            if verify:
                # Decode full network into compact (u, v, gv, neg) format
                full_compact = [
                    (u, v, *decode_literal(lit, vars_in_func))
                    for u, v, lit in s_tr
                ]
                covered = covered_patterns(full_compact, nc, on_patterns)
                missing  = on_patterns - covered
                safe     = check_safety(full_compact, nc, off_patterns)
                if missing or not safe:
                    n_fail += 1
                    issues = []
                    if missing:
                        issues.append(f"missing on-patterns: {sorted(missing)}")
                    if not safe:
                        issues.append("conducts under off-pattern")
                    print(f"\n  [VERIFY FAIL] {fid} ({expr}): {'; '.join(issues)}")

            for k in range(R):
                _, g2c, g_num = build_prefix_graph(
                    s_tr, k, vars_in_func, on_patterns
                )
                u, v, lit_id = s_tr[k]
                h, lo = max(u, v), min(u, v)
                lo_in = lo in g2c
                h_in  = h  in g2c
                if lo_in and h_in:
                    node1 = g2c[lo]
                    node2 = g2c[h]
                elif lo_in and not h_in:
                    node1 = g2c[lo]
                    node2 = g_num
                elif not lo_in and h_in:
                    node1 = g2c[h]
                    node2 = g_num
                else:
                    raise ValueError(
                        f"Both endpoints ({lo},{h}) of sorted_tr[{k}] are new "
                        f"in function {fid} subdir={subdir}. This should not occur after "
                        f"filtering disconnected transistors."
                    )
                gv, neg = decode_literal(lit_id, vars_in_func)

                compact_k = [
                    (g2c[pu], g2c[pv], *decode_literal(pl, vars_in_func))
                    for pu, pv, pl in s_tr[:k]
                ]

                # node2 safety mask: for each candidate node2, is there ≥1 safe literal?
                node2_safety_mask = torch.zeros(MAX_G_NODES, dtype=torch.bool)
                for n2c in range(g_num + 1):
                    if n2c == node1:
                        continue
                    ng2 = g_num + (1 if n2c == g_num else 0)
                    for vi in range(N_VARS):
                        for ni2 in range(2):
                            if check_safety(compact_k + [(node1, n2c, vi, ni2)], ng2, off_patterns):
                                node2_safety_mask[n2c] = True
                                break
                        if node2_safety_mask[n2c]:
                            break
                node2_safety_mask[node2] = True

                # var/sign safety mask for the GT (node1, node2)
                new_g_num_k = g_num + (1 if node2 == g_num else 0)
                safety_mask = torch.zeros(N_VARS, 2, dtype=torch.bool)
                for vi in range(N_VARS):
                    for ni in range(2):
                        if check_safety(
                            compact_k + [(node1, node2, vi, ni)],
                            new_g_num_k, off_patterns,
                        ):
                            safety_mask[vi, ni] = True

                # Store raw data only — Graph objects are rebuilt from this in _build_graphs
                raw.append({
                    "compact_tran": compact_k,
                    "g_num":        g_num,
                    "vars_in_func": vars_in_func,
                    "on_patterns":  tuple(on_patterns),   # frozenset → serializable tuple
                    "node1":             node1,
                    "node2":             node2,
                    "var":               gv,
                    "neg":               neg,
                    "node2_safety_mask": node2_safety_mask,
                    "safety_mask":       safety_mask,
                })

        if verify:
            status = "OK" if n_fail == 0 else f"{n_fail} FAILURES"
            print(f"  Verification {dataset_name}: {len(rows)} networks checked — {status}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]

    # ── Convenience accessors for the task ──────────────────────────────────

    @property
    def node_feature_dim(self) -> int:
        from .graph import NODE_FEAT_DIM
        return NODE_FEAT_DIM

    @property
    def edge_feature_dim(self) -> int:
        from .literal import EDGE_FEAT_DIM
        return EDGE_FEAT_DIM
