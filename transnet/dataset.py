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

from .graph import MAX_G_NODES, build_prefix_graph, sorted_g_transistors
from .literal import (
    N_VARS,
    check_safety, decode_literal, extract_vars, on_off_split, parse_sop_expr,
)


_CSV_PATH = os.path.join(
    os.path.dirname(__file__), "..", "dataset", "sweep_3input", "aggregate.csv"
)
_DATA_ROOT = os.path.join(
    os.path.dirname(__file__), "..", "dataset", "sweep_3input"
)


def _parse_cg(cg_str: str):
    """Parse 'CG(nc=N, r=R, tr=[(s,d,l),...])' → (nc, tr_list)."""
    nc = int(re.search(r"nc=(\d+)", cg_str).group(1))
    tr_str = re.search(r"tr=\[([^\]]*)\]", cg_str).group(1)
    tr = [
        tuple(int(x) for x in m)
        for m in re.findall(r"\((\d+),\s*(\d+),\s*(\d+)\)", tr_str)
    ]
    return nc, tr


def _read_expr(function_id: str, t: str) -> str:
    """Read SOP expression from Booleans.txt."""
    path = os.path.join(_DATA_ROOT, function_id, f"t_{t}", "Booleans.txt")
    with open(path) as f:
        line = f.read().strip()
    # Format: " PF3_XXX(t): <expr>"
    return line.split(": ", 1)[1].strip()


class TransistorDataset(Dataset):
    """
    Pre-expanded BFS prefix training dataset for transistor-network synthesis.

    Each Boolean function from aggregate.csv produces R samples (one per
    G-transistor in snake-sort order).
    """

    def __init__(self, csv_path: str = _CSV_PATH):
        self.samples: List[dict] = []
        self._load(csv_path)

    def _load(self, csv_path: str) -> None:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = [r for r in reader if r["status"] == "SAT"]

        for row in rows:
            fid = row["function_id"]
            t = row["t"]
            nc, tr = _parse_cg(row["cg"])
            expr = _read_expr(fid, t)
            vars_in_func = extract_vars(expr)
            on_patterns = parse_sop_expr(expr, vars_in_func)
            off_patterns = on_off_split(on_patterns)[1]

            s_tr = sorted_g_transistors(tr, nc)
            R = len(s_tr)

            for k in range(R):
                graph, g2c, g_num = build_prefix_graph(
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
                        f"in function {fid} t={t}. This should not occur after "
                        f"filtering disconnected transistors."
                    )
                gv, neg = decode_literal(lit_id, vars_in_func)

                compact_k = [
                    (g2c[pu], g2c[pv], *decode_literal(pl, vars_in_func))
                    for pu, pv, pl in s_tr[:k]
                ]

                # node2 safety mask: for each candidate node2, is there ≥1 safe literal?
                # Shape [MAX_G_NODES]: index = compact node2 (g_num = virtual new node).
                node2_safety_mask = torch.zeros(MAX_G_NODES, dtype=torch.bool)
                for n2c in range(g_num + 1):      # 0..g_num-1 existing, g_num = virtual
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
                node2_safety_mask[node2] = True    # GT node2 always valid

                # var/sign safety mask: for the GT (node1, node2), which literals are safe?
                new_g_num_k = g_num + (1 if node2 == g_num else 0)
                safety_mask = torch.zeros(N_VARS, 2, dtype=torch.bool)
                for vi in range(N_VARS):
                    for ni in range(2):
                        if check_safety(
                            compact_k + [(node1, node2, vi, ni)],
                            new_g_num_k, off_patterns,
                        ):
                            safety_mask[vi, ni] = True

                self.samples.append({
                    "graph":             graph,
                    "node1":             node1,
                    "node2":             node2,
                    "var":               gv,
                    "neg":               neg,
                    "node2_safety_mask": node2_safety_mask,
                    "safety_mask":       safety_mask,
                })

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
