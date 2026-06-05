"""
transnet/graph.py

Build torchdrug.data.Graph objects for transistor-network synthesis.

Node layout in every unified graph:
  0           = SOURCE   (shared by G and C)
  1           = SINK     (shared by G and C)
  2 .. g_n-1  = INTERNAL_G
  g_n .. end  = INTERNAL_C

Node feature : 4-dim one-hot  [is_src, is_snk, is_G, is_C]
Edge feature : EDGE_FEAT_DIM  [one_hot(var, N_VARS)…, is_neg, is_compensate]
Graph attr   : g_num_nodes (scalar int64) = |SOURCE ∪ SINK ∪ INTERNAL_G|
"""
from __future__ import annotations

from collections import deque
import torch
from typing import Dict, FrozenSet, List, Set, Tuple

from torchdrug import data as td_data

from .literal import (
    EDGE_FEAT_DIM, N_VARS,
    build_sop_edges, decode_literal, make_edge_feat,
)

NODE_FEAT_DIM = 4
MAX_G_NODES   = 20   # upper bound on compact G-node count (SRC + SNK + INTERNAL_G)
_SRC, _SNK, _G, _C = 0, 1, 2, 3


# ─────────────────────────────────────────────────────────────────────────────
# Snake-pattern ordering
# ─────────────────────────────────────────────────────────────────────────────

def _snake_id(u: int, v: int) -> int:
    hi = max(u, v)
    return hi * (hi + 1) + min(u, v)


def _src_snk_reachable(tr: List[Tuple[int, int, int]], nc: int) -> Set[int]:
    """
    Return the set of dataset node indices that are reachable from BOTH
    dataset node 0 (SOURCE) and dataset node nc-1 (SINK) via the transistors.
    Transistors not in this set are isolated / on dead-ends and are excluded.
    """
    adj: List[List[int]] = [[] for _ in range(nc)]
    for s, d, _ in tr:
        adj[s].append(d)
        adj[d].append(s)

    def bfs(start: int) -> Set[int]:
        visited: Set[int] = {start}
        q: deque = deque([start])
        while q:
            n = q.popleft()
            for nb in adj[n]:
                if nb not in visited:
                    visited.add(nb)
                    q.append(nb)
        return visited

    from_src = bfs(0)
    from_snk = bfs(nc - 1)
    return from_src & from_snk


def sorted_g_transistors(
    tr: List[Tuple[int, int, int]],
    nc: int,
) -> List[Tuple[int, int, int]]:
    """
    Remap CG dataset transistors to unified node indices and sort by
    snake-pattern undirected_id.

    Only transistors where both endpoints can reach SOURCE and SINK are
    kept — isolated / dead-end transistors are excluded.

    Dataset remapping:
      dataset node 0     → 0   (SOURCE)
      dataset node nc-1  → 1   (SINK)
      dataset node k     → k+1 (INTERNAL_G, k in 1..nc-2)
    """
    useful = _src_snk_reachable(tr, nc)

    def remap(k: int) -> int:
        if k == 0:       return 0
        if k == nc - 1:  return 1
        return k + 1

    remapped = [
        (remap(s), remap(d), lit)
        for s, d, lit in tr
        if s in useful and d in useful
    ]
    return sorted(remapped, key=lambda e: _snake_id(e[0], e[1]))


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _node_features(total_nodes: int, g_num_nodes: int) -> torch.Tensor:
    nf = []
    for i in range(total_nodes):
        oh = [0.0] * NODE_FEAT_DIM
        if i == 0:
            oh[_SRC] = 1.0
        elif i == 1:
            oh[_SNK] = 1.0
        elif i < g_num_nodes:
            oh[_G] = 1.0
        else:
            oh[_C] = 1.0
        nf.append(oh)
    return torch.tensor(nf, dtype=torch.float)


def _assemble_graph(
    edges: List[Tuple[int, int]],
    feats: List[List[float]],
    total_nodes: int,
    g_num_nodes: int,
) -> td_data.Graph:
    """Create a torchdrug Graph from pre-built edge/feature lists."""
    if edges:
        edge_arr = torch.tensor(edges, dtype=torch.long)           # [E, 2]
        rel_col = torch.zeros(len(edges), 1, dtype=torch.long)     # [E, 1]
        edge_list = torch.cat([edge_arr, rel_col], dim=1)          # [E, 3]
        edge_feat = torch.tensor(feats, dtype=torch.float)         # [E, EDGE_FEAT_DIM]
    else:
        edge_list = torch.zeros((0, 3), dtype=torch.long)
        edge_feat = torch.zeros((0, EDGE_FEAT_DIM), dtype=torch.float)

    node_feat = _node_features(total_nodes, g_num_nodes)

    graph = td_data.Graph(edge_list=edge_list, num_node=total_nodes, num_relation=1)
    with graph.edge():
        graph.edge_feature = edge_feat
    with graph.node():
        graph.node_feature = node_feat
    with graph.graph():
        # Scalar (0-dim) tensor: pack() does unsqueeze(0) then cat → shape [B]
        graph.g_num_nodes = torch.tensor(g_num_nodes, dtype=torch.long)

    return graph


def _c_parts(
    on_patterns: FrozenSet[Tuple[int, ...]],
    vars_in_func: List[str],
    g_num_nodes: int,
) -> Tuple[List[Tuple[int, int]], List[List[float]], int]:
    """
    Build undirected C-edge list (both directions) with unified node indices.

    C-graph nodes 0=SOURCE and 1=SINK are shared with G-graph.
    C-graph INTERNAL_C node k (k ≥ 2) → unified node g_num_nodes + (k-2).

    Returns (c_edges, c_feats, total_nodes).
    """
    raw, total_c = build_sop_edges(on_patterns, vars_in_func)
    n_internal_c = max(0, total_c - 2)

    def remap(k: int) -> int:
        if k == 0:  return 0
        if k == 1:  return 1
        return g_num_nodes + (k - 2)

    c_edges, c_feats = [], []
    for cs, cd, gv, neg in raw:
        u, v = remap(cs), remap(cd)
        feat = make_edge_feat(gv, neg, is_compensate=1)
        c_edges += [(u, v), (v, u)]
        c_feats += [feat, feat]

    return c_edges, c_feats, g_num_nodes + n_internal_c


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_transnet_graph(
    nc: int,
    tr: List[Tuple[int, int, int]],
    vars_in_func: List[str],
    on_patterns: FrozenSet[Tuple[int, ...]],
) -> td_data.Graph:
    """
    Full G ∪ C unified graph from a dataset CG record.

    nc : number of CG nodes in dataset format (includes SOURCE and SINK)
    tr : list of (src, drn, lit_id) in dataset node numbering
    """
    g_num_nodes = nc

    def remap_g(k: int) -> int:
        if k == 0:       return 0
        if k == nc - 1:  return 1
        return k + 1

    g_edges, g_feats = [], []
    for s, d, lit_id in tr:
        u, v = remap_g(s), remap_g(d)
        gv, neg = decode_literal(lit_id, vars_in_func)
        feat = make_edge_feat(gv, neg, is_compensate=0)
        g_edges += [(u, v), (v, u)]
        g_feats += [feat, feat]

    c_edges, c_feats, total_nodes = _c_parts(on_patterns, vars_in_func, g_num_nodes)
    return _assemble_graph(g_edges + c_edges, g_feats + c_feats, total_nodes, g_num_nodes)


def build_prefix_graph(
    sorted_tr: List[Tuple[int, int, int]],
    k: int,
    vars_in_func: List[str],
    on_patterns: FrozenSet[Tuple[int, ...]],
) -> Tuple[td_data.Graph, Dict[int, int], int]:
    """
    Build the prefix-k G ∪ C graph (first k snake-sorted G-transistors + full C).

    G-node compaction: only nodes that appear in sorted_tr[:k] (plus SOURCE=0,
    SINK=1) are retained; indices are compressed to 0-based sorted order so
    SOURCE=0 and SINK=1 are always preserved.

    Returns:
        graph       : td_data.Graph
        g2compact   : dict  original unified G-node index → compact index
        g_num_nodes : number of compact G-nodes
    """
    prefix_tr = sorted_tr[:k]

    g_nodes = {0, 1}
    for u, v, _ in prefix_tr:
        g_nodes.update((u, v))

    sorted_gn = sorted(g_nodes)
    g2c: Dict[int, int] = {old: new for new, old in enumerate(sorted_gn)}
    g_num_nodes = len(sorted_gn)

    g_edges, g_feats = [], []
    for u, v, lit_id in prefix_tr:
        cu, cv = g2c[u], g2c[v]
        gv, neg = decode_literal(lit_id, vars_in_func)
        feat = make_edge_feat(gv, neg, is_compensate=0)
        g_edges += [(cu, cv), (cv, cu)]
        g_feats += [feat, feat]

    c_edges, c_feats, total_nodes = _c_parts(on_patterns, vars_in_func, g_num_nodes)
    graph = _assemble_graph(
        g_edges + c_edges, g_feats + c_feats, total_nodes, g_num_nodes
    )
    return graph, g2c, g_num_nodes


def build_gen_graph(
    g_tran: List[Tuple[int, int, int, int]],
    g_num: int,
    vars_in_func: List[str],
    on_patterns: FrozenSet[Tuple[int, ...]],
) -> td_data.Graph:
    """
    Build the unified G∪C graph from the current generation state.

    g_tran : list of (u_compact, v_compact, var_idx, is_neg)
    g_num  : number of compact G-nodes (SOURCE=0, SINK=1, INTERNAL_G=2..g_num-1)
    """
    g_edges, g_feats = [], []
    for u, v, gv, neg in g_tran:
        feat = make_edge_feat(gv, neg, is_compensate=0)
        g_edges += [(u, v), (v, u)]
        g_feats += [feat, feat]

    c_edges, c_feats, total_nodes = _c_parts(on_patterns, vars_in_func, g_num)
    return _assemble_graph(g_edges + c_edges, g_feats + c_feats, total_nodes, g_num)


def build_initial_graph(
    vars_in_func: List[str],
    on_patterns: FrozenSet[Tuple[int, ...]],
) -> td_data.Graph:
    """G_0: only SOURCE(0) and SINK(1), no G-transistors, full C-network."""
    g_num_nodes = 2
    c_edges, c_feats, total_nodes = _c_parts(on_patterns, vars_in_func, g_num_nodes)
    return _assemble_graph(c_edges, c_feats, total_nodes, g_num_nodes)
