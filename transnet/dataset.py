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

from .graph import (
    MAX_G_NODES, assemble_graph_from_tensors, build_gen_graph, build_prefix_graph,
    path_sorted_g_transistors, sorted_g_transistors,
)

# Construction-order mode for prefix samples: "bfs" (snake-BFS, historical) or
# "path" (ear-decomposition walks; see path_sorted_g_transistors). Env-gated so
# cache builds pick it up without threading a flag through every call site.
ORDER_MODE = os.environ.get("TRANSGPN_ORDER", "bfs")
from .literal import (
    ALL_VARS, N_VARS,
    check_safety, covered_patterns, decode_literal, extract_vars, on_off_split, parse_sop_expr,
)
from .canonical import canonicalize_expr, canon_lit


# ── Parallel eager graph build (fork + temp chunk files) ─────────────────────────
# Build-in-advance: fork worker processes that each build a contiguous slice of the
# raw samples into Graph objects and torch.save it to a temp file; the parent loads
# the chunks back. Avoids the ~2.4h single-threaded build (which idles the GPU past
# the low-util watchdog) without multiprocessing tensor-IPC/fd pitfalls. The raw list
# is shared to forked workers via copy-on-write (a module global), so only small
# (lo, hi, path) tuples cross the process boundary.
_BUILD_RAW = None


def _build_chunk_to_file(args):
    """Worker: build raw[lo:hi] into samples and torch.save to `path`. Returns path."""
    import gc as _gc
    _gc.disable()                       # short-lived worker; avoid GC-driven CoW on shared raw
    lo, hi, path = args
    out = [TransistorDataset._build_one(_BUILD_RAW[i]) for i in range(lo, hi)]
    torch.save(out, path)
    return path


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
                 verify: bool = True, max_functions: int = None, seed: int = 0,
                 num_shards: int = 1, shard_index: int = 0, raw_only: bool = False,
                 one_impl_per_function: bool = False,
                 one_impl_per_switchcount: bool = False):
        if dataset_roots is None:
            dataset_roots = _DATASET_ROOTS
        self.samples: List[dict] = []
        self._lazy: bool = False        # lazy mode: build Graph in __getitem__
        self._columnar: bool = False    # columnar mode: slice big tensors in __getitem__
        self.raw: List[dict] = None     # raw entries kept for lazy graph build
        self.col: dict = None           # concatenated columnar tensors

        if cache_path and os.path.isdir(cache_path):
            import glob
            col_files = sorted(glob.glob(os.path.join(cache_path, "col_shard*.pt")))
            if col_files:
                # COLUMNAR: the whole dataset stored as a few big contiguous tensors
                # (edge_list/edge_feature/node_feature + per-sample counts/labels).
                # ~75GB of pure tensor data (no per-object overhead), loads fast, and
                # workers share it read-only (no CoW blow-up). __getitem__ slices out
                # one graph's rows and assembles a lightweight Graph — no per-epoch
                # build_gen_graph recomputation. Fixes memory + CoW + epoch speed.
                self.col = self._load_columnar(col_files)
                self._columnar = True
                print(f"  columnar mode: {self.col['total']} samples ready "
                      f"across {len(self.col['shards'])} shards", flush=True)
                return

            # Directory of shard*.pt raw caches → LAZY (build each Graph on demand).
            shard_files = sorted(glob.glob(os.path.join(cache_path, "shard*.pt")))
            print(f"Loading dataset from {len(shard_files)} shards (lazy): {cache_path}")
            raw = []
            for sf in shard_files:
                part = torch.load(sf, weights_only=False)
                raw.extend(part)
                del part
                print(f"  loaded {os.path.basename(sf)} -> {len(raw)} samples", flush=True)
            self.raw = raw
            self._lazy = True
            print(f"  lazy mode: {len(raw)} samples, graphs built on demand in workers",
                  flush=True)
            return

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
            self._load(root, csv_path, raw, verify=verify,
                       max_functions=max_functions, seed=seed,
                       num_shards=num_shards, shard_index=shard_index,
                       one_impl_per_function=one_impl_per_function,
                       one_impl_per_switchcount=one_impl_per_switchcount)

        if cache_path:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            print(f"Saving dataset cache to: {cache_path}")
            torch.save(raw, cache_path)
            print(f"  Saved {len(raw)} raw samples.")

        # raw_only: parallel cache-build worker — skip the in-RAM graph build.
        if raw_only:
            self.samples = []
            return

        self.samples = self._build_graphs(raw, desc="Building graphs")

    @staticmethod
    def _build_one(entry: dict) -> dict:
        """Reconstruct a single training sample (Graph + targets) from raw data."""
        on_patterns = frozenset(map(tuple, entry["on_patterns"]))
        graph = build_gen_graph(
            entry["compact_tran"], entry["g_num"],
            entry["vars_in_func"], on_patterns,
        )
        return {
            "graph":             graph,
            "node1":             entry["node1"],
            "node2":             entry["node2"],
            "var":               entry["var"],
            "neg":               entry["neg"],
            "node2_safety_mask": entry["node2_safety_mask"],
            "safety_mask":       entry["safety_mask"],
        }

    @classmethod
    def _build_graphs(cls, raw: List[dict], desc: str = "Building graphs") -> List[dict]:
        """Reconstruct Graph objects from raw data (no BFS, fast)."""
        return [cls._build_one(entry) for entry in tqdm(raw, desc=desc, unit="sample")]

    # ── Columnar format (few big tensors instead of millions of Graph objects) ──
    @classmethod
    def _columnar_from_raw(cls, raw: List[dict], desc: str = "Columnarizing") -> dict:
        """Build one shard's columnar tensors from raw samples.

        For each sample, build its Graph once and extract the flat tensors
        (edge_list, edge_feature, node_feature); concatenate across the shard with
        per-sample counts, and stack the labels/masks. Result is a handful of big
        contiguous tensors — the memory-efficient, IPC-friendly form.
        """
        from .literal import N_VARS
        n = len(raw)
        ELs, EFs, NFs = [], [], []
        edge_count = torch.empty(n, dtype=torch.long)
        node_count = torch.empty(n, dtype=torch.long)
        g_num_nodes = torch.empty(n, dtype=torch.long)
        node1 = torch.empty(n, dtype=torch.long)
        node2 = torch.empty(n, dtype=torch.long)
        var = torch.empty(n, dtype=torch.long)
        neg = torch.empty(n, dtype=torch.long)
        n2sm = torch.empty(n, MAX_G_NODES, dtype=torch.bool)
        sm = torch.empty(n, N_VARS, 2, dtype=torch.bool)
        for i, entry in enumerate(tqdm(raw, desc=desc, unit="sample")):
            s = cls._build_one(entry)
            g = s["graph"]
            el = g.edge_list.long()
            ELs.append(el)
            EFs.append(g.edge_feature.float())
            NFs.append(g.node_feature.float())
            edge_count[i] = el.shape[0]
            node_count[i] = g.node_feature.shape[0]
            g_num_nodes[i] = int(g.g_num_nodes)
            node1[i] = int(s["node1"]); node2[i] = int(s["node2"])
            var[i] = int(s["var"]); neg[i] = int(s["neg"])
            n2sm[i] = s["node2_safety_mask"]; sm[i] = s["safety_mask"]
        from .literal import EDGE_FEAT_DIM
        from .graph import NODE_FEAT_DIM
        cat_e = torch.cat(ELs) if ELs else torch.zeros((0, 3), dtype=torch.long)
        cat_ef = torch.cat(EFs) if EFs else torch.zeros((0, EDGE_FEAT_DIM))
        cat_nf = torch.cat(NFs) if NFs else torch.zeros((0, NODE_FEAT_DIM))
        return {
            "edge_list": cat_e, "edge_feature": cat_ef, "node_feature": cat_nf,
            "edge_count": edge_count, "node_count": node_count,
            "g_num_nodes": g_num_nodes,
            "node1": node1, "node2": node2, "var": var, "neg": neg,
            "node2_safety_mask": n2sm, "safety_mask": sm,
        }

    @staticmethod
    def _load_columnar(col_files: List[str]) -> dict:
        """Load columnar shards and KEEP them separate (no concat).

        The built data is large (~340GB: the C-edges encoding the function are
        duplicated across every prefix). Concatenating would need ~2x that at peak,
        so instead we keep the shard dicts and index across them: each shard gets
        within-shard cumulative offsets, plus a global cumulative sample count for
        idx -> (shard, local) lookup in __getitem__.
        """
        shards = []
        for f in col_files:
            p = torch.load(f, weights_only=False)
            p["edge_off"] = torch.cat([torch.zeros(1, dtype=torch.long),
                                       p["edge_count"].cumsum(0)])
            p["node_off"] = torch.cat([torch.zeros(1, dtype=torch.long),
                                       p["node_count"].cumsum(0)])
            shards.append(p)
            print(f"  loaded {os.path.basename(f)} ({len(p['node1'])} samples, "
                  f"{p['edge_list'].shape[0]} edges)", flush=True)
        counts = [len(p["node1"]) for p in shards]
        cum = torch.zeros(len(shards) + 1, dtype=torch.long)
        cum[1:] = torch.tensor(counts, dtype=torch.long).cumsum(0)
        return {"shards": shards, "cum": cum, "total": int(cum[-1])}

    @classmethod
    def _build_graphs_parallel(cls, raw: List[dict], nproc: int = None,
                               tmpdir: str = None) -> List[dict]:
        """Build graphs in advance across forked workers writing temp chunk files.

        Falls back to the single-threaded build on any failure. Workers read the
        shared `raw` via copy-on-write (module global _BUILD_RAW); only (lo, hi, path)
        tuples cross the process boundary, and each chunk's built samples are handed
        back through a temp .pt file (avoids multiprocessing tensor-IPC / fd limits).
        """
        import math, multiprocessing as mp, shutil, time
        global _BUILD_RAW
        n = len(raw)
        if n == 0:
            return []
        if nproc is None:
            nproc = min(32, os.cpu_count() or 8)
        nproc = max(1, min(nproc, n))
        tmpdir = tmpdir or os.path.join(".", "_gbuild_tmp")
        try:
            if os.path.isdir(tmpdir):
                shutil.rmtree(tmpdir)
            os.makedirs(tmpdir, exist_ok=True)
            chunk = math.ceil(n / nproc)
            tasks = []
            for k in range(nproc):
                lo, hi = k * chunk, min((k + 1) * chunk, n)
                if lo >= hi:
                    break
                tasks.append((lo, hi, os.path.join(tmpdir, f"chunk{k}.pt")))
            print(f"Building {n} graphs in advance: {len(tasks)} fork workers → {tmpdir}",
                  flush=True)
            t0 = time.time()
            _BUILD_RAW = raw
            ctx = mp.get_context("fork")
            with ctx.Pool(len(tasks)) as pool:
                files = pool.map(_build_chunk_to_file, tasks)
            _BUILD_RAW = None
            print(f"  workers done in {time.time()-t0:.0f}s; loading chunks …", flush=True)
            samples = []
            for f in files:
                samples.extend(torch.load(f, weights_only=False))
            print(f"  built {len(samples)} graphs in {time.time()-t0:.0f}s total",
                  flush=True)
            return samples
        except Exception as e:
            print(f"  parallel build failed ({e!r}); falling back to single-threaded",
                  flush=True)
            _BUILD_RAW = None
            return cls._build_graphs(raw, desc="Building graphs (fallback)")
        finally:
            try:
                if os.path.isdir(tmpdir):
                    shutil.rmtree(tmpdir)
            except Exception:
                pass

    def _load(self, data_root: str, csv_path: str, raw: List[dict],
              verify: bool = True, max_functions: int = None, seed: int = 0,
              num_shards: int = 1, shard_index: int = 0,
              one_impl_per_function: bool = False,
              one_impl_per_switchcount: bool = False) -> None:
        is_4input = _is_4input_csv(csv_path)
        sat_status = "OK" if is_4input else "SAT"

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            # "nc=" guard: a small fraction of rows (e.g. 3,682/299,586 SAT rows
            # in the HF sweep_4input) carry an empty/malformed cg network string.
            rows = [r for r in reader
                    if r["status"] == sat_status and "nc=" in (r.get("cg") or "")]

        # Implementation dedup (done before sharding so every shard agrees):
        #   one_impl_per_switchcount: keep one impl per (function, n_switches) — i.e.
        #     every distinct transistor-count variant of each function (e.g. a 5-tr
        #     and a 6-tr version), dropping only the perm/relabel duplicates that
        #     share the same count. Representative = min literal_count, then impl_idx.
        #     ~656k rows for sweep_6input_all.
        #   one_impl_per_function: keep only the single most compact net (min
        #     n_switches) per function. ~288k rows.
        def _ns(r):
            # new schema: n_switches; old MiniTNtk schema: n_transistors
            v = r.get("n_switches", "") or r.get("n_transistors", "")
            return int(v) if v not in (None, "") else 10**9

        def _fid(r):
            # new schema: truth_index; old MiniTNtk schema: function_id
            return r.get("truth_index") or r["function_id"]
        if one_impl_per_switchcount:
            def _lc(r):
                v = r.get("literal_count", "")
                return int(v) if v not in (None, "") else 10**9
            def _ii(r):
                v = r.get("impl_idx", "")
                return int(v) if v not in (None, "") else 10**9
            best = {}
            for r in rows:
                key = (_fid(r), _ns(r))
                cur = best.get(key)
                if cur is None or (_lc(r), _ii(r)) < (_lc(cur), _ii(cur)):
                    best[key] = r
            rows = sorted(best.values(), key=lambda r: (_fid(r), _ns(r)))
            print(f"  one impl/(function,n_switches): {len(rows)} variants")
        elif one_impl_per_function:
            best = {}
            for r in rows:
                key = _fid(r)
                if key not in best or _ns(r) < _ns(best[key]):
                    best[key] = r
            rows = sorted(best.values(), key=lambda r: _fid(r))
            print(f"  one impl/function: {len(rows)} unique functions "
                  f"(by min n_switches)")

        # Subsample functions (for very large datasets like sweep_6input_possani).
        # Deterministic on `seed`, so every shard selects the SAME subset.
        if max_functions and len(rows) > max_functions:
            import random as _random
            rows = _random.Random(seed).sample(rows, max_functions)
            print(f"  Subsampled to {max_functions} functions (seed={seed}) "
                  f"from {os.path.basename(data_root)}")

        # Sharding: each parallel build worker processes a disjoint stripe.
        if num_shards > 1:
            rows = [r for i, r in enumerate(rows) if i % num_shards == shard_index]

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

            # ── Variable-position canonicalization ───────────────────────────
            # Relabel the function's support to the lowest slots (ALL_VARS[:K])
            # and remap every GT transistor's lit_id into the canonical local
            # scheme. After this the policy only ever trains on low-slot
            # networks, matching the (identically canonicalized) generation step.
            canon_expr, real_vars, perm = canonicalize_expr(expr)
            K = len(real_vars)
            canon_vif = ALL_VARS[:K]
            s_tr = (path_sorted_g_transistors(tr, nc) if ORDER_MODE == "path"
                    else sorted_g_transistors(tr, nc))
            s_tr = [(u, v, canon_lit(lit, vars_in_func, perm, K))
                    for (u, v, lit) in s_tr]
            vars_in_func = canon_vif
            on_patterns = parse_sop_expr(canon_expr, canon_vif)
            _, off_patterns = on_off_split(on_patterns)
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
                if ORDER_MODE == "path":
                    # Walk direction is meaningful: u is the attach point
                    # (always already in the prefix), v extends or closes.
                    if u not in g2c:
                        raise ValueError(
                            f"path-order attach node {u} of sorted_tr[{k}] not in "
                            f"prefix for function {fid} subdir={subdir}."
                        )
                    node1 = g2c[u]
                    node2 = g2c[v] if v in g2c else g_num
                else:
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
        if self._columnar:
            return self.col["total"]
        return len(self.raw) if self._lazy else len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        if self._columnar:
            c = self.col
            si = int(torch.searchsorted(c["cum"], idx, right=True)) - 1
            s = c["shards"][si]
            li = idx - int(c["cum"][si])
            e0 = int(s["edge_off"][li]); e1 = int(s["edge_off"][li + 1])
            n0 = int(s["node_off"][li]); n1 = int(s["node_off"][li + 1])
            # .clone() so each returned tensor owns a small buffer — otherwise the
            # DataLoader would IPC-share the whole multi-GB shard storage per sample.
            graph = assemble_graph_from_tensors(
                s["edge_list"][e0:e1].clone(),
                s["edge_feature"][e0:e1].clone(),
                s["node_feature"][n0:n1].clone(),
                int(s["g_num_nodes"][li]),
            )
            return {
                "graph":             graph,
                "node1":             int(s["node1"][li]),
                "node2":             int(s["node2"][li]),
                "var":               int(s["var"][li]),
                "neg":               int(s["neg"][li]),
                "node2_safety_mask": s["node2_safety_mask"][li].clone(),
                "safety_mask":       s["safety_mask"][li].clone(),
            }
        if self._lazy:
            return self._build_one(self.raw[idx])
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
