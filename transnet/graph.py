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

import os

NODE_FEAT_DIM = 4
# Upper bound on compact G-node count (SRC + SNK + INTERNAL_G). Configurable via
# env because 6-input networks can have more internal nodes than 4-input ones.
MAX_G_NODES   = int(os.environ.get("TRANSGPN_MAX_G_NODES", "20"))
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
    Remap CG dataset transistors to unified node indices and sort in
    BFS-connected order from SOURCE/SINK, using snake_id as a tiebreaker.

    Pure snake ordering is not used because it does not guarantee that the
    first transistor touches SOURCE(0) or SINK(1) — this fails for 4-input
    networks where SOURCE connects to high-indexed (high snake_id) nodes.

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

    # BFS-connected ordering: each transistor k has ≥1 endpoint already visited.
    visited: Set[int] = {0, 1}
    pending = sorted(remapped, key=lambda e: _snake_id(e[0], e[1]))
    result: List[Tuple[int, int, int]] = []

    while pending:
        for i, (u, v, lit) in enumerate(pending):
            if u in visited or v in visited:
                result.append((u, v, lit))
                visited.update([u, v])
                pending.pop(i)
                break
        else:
            result.extend(pending)   # disconnected tail (should not happen)
            break

    return result


def path_sorted_g_transistors(
    tr: List[Tuple[int, int, int]],
    nc: int,
) -> List[Tuple[int, int, int]]:
    """
    Path/ear-decomposition ordering (DFS-like): emit transistors as directed
    walks that mirror how conduction paths are actually built.

    The first walk starts at SOURCE and extends through unvisited nodes until
    it closes into visited structure (ideally SNK) — one complete path. Each
    subsequent walk (an "ear") starts at an already-visited node and again
    extends until it reconnects. Unlike snake-BFS ordering, at any prefix
    there is at most ONE dangling stub and it is the next attach point — the
    node1 label becomes structurally determined instead of an index-convention
    artifact, and the order matches the task semantics (coverage accrues by
    completing SRC-SNK paths).

    Returns (attach, other, lit) tuples in construction order, where `attach`
    is ALWAYS already visited at emission time. Walk direction is meaningful:
    dataset labeling must use tuple order (node1=attach), not the lo/hi rule.
    """
    from collections import defaultdict

    useful = _src_snk_reachable(tr, nc)

    def remap(k: int) -> int:
        if k == 0:       return 0
        if k == nc - 1:  return 1
        return k + 1

    edges = [(remap(s), remap(d), lit)
             for s, d, lit in tr if s in useful and d in useful]

    adj: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for i, (u, v, _lit) in enumerate(edges):
        adj[u].append((v, i))
        adj[v].append((u, i))

    used = [False] * len(edges)
    visited: Set[int] = {0, 1}
    result: List[Tuple[int, int, int]] = []

    def walk(start: int) -> None:
        cur = start
        while True:
            nxts = sorted((n, i) for n, i in adj[cur] if not used[i])
            if not nxts:
                return
            unv = [(n, i) for n, i in nxts if n not in visited]
            if unv:
                n, i = unv[0]
            else:
                # closing the walk: prefer completing a conduction path at
                # SNK(1), then interior nodes, and only lastly loop into SRC(0)
                n, i = min(nxts, key=lambda t: (t[0] != 1, t[0] == 0, t))
            used[i] = True
            result.append((cur, n, edges[i][2]))
            closed = n in visited
            visited.add(n)
            if closed:
                return          # walk reconnected into existing structure
            cur = n

    while len(result) < len(edges):
        starts = sorted(n for n in visited
                        if any(not used[i] for _n, i in adj[n]))
        if not starts:
            # disconnected leftovers (should not happen after pruning)
            for i, (u, v, lit) in enumerate(edges):
                if not used[i]:
                    used[i] = True
                    result.append((u, v, lit))
                    visited.update([u, v])
            break
        walk(starts[0])

    assert len(result) == len(edges), "path ordering lost transistors"
    return result


def prune_gtran(g_tran, g_num: int):
    """Keep EXACTLY the transistors lying on >=1 simple SRC(0)-SNK(1) path.

    Everything else (dangling stubs, pendant parallel pairs, dangling cycles,
    disconnected islands) cannot conduct and is physically free to delete, so
    the pruned size is the honest network cost.

    Exact criterion via biconnected decomposition: an edge is on a simple
    s-t path iff its biconnected block lies on the block-cut-tree path
    between s and t (within a 2-connected block, every edge admits a simple
    path through it between any two block vertices).
    """
    edges = [e for e in g_tran if e[0] != e[1]]
    if not edges:
        return []

    adj: List[List[Tuple[int, int]]] = [[] for _ in range(g_num)]
    for i, e in enumerate(edges):
        adj[e[0]].append((e[1], i))
        adj[e[1]].append((e[0], i))

    # ── Tarjan biconnected components (rooted at SRC=0) ────────────────────
    disc = [0] * g_num
    low = [0] * g_num
    comp_of_edge = [-1] * len(edges)
    is_cut = [False] * g_num
    estack: List[int] = []
    timer = [1]
    ncomp = [0]

    import sys
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old_limit, 4 * g_num + 100))

    def _dfs(u: int, parent_eid: int) -> None:
        disc[u] = low[u] = timer[0]
        timer[0] += 1
        children = 0
        for v, i in adj[u]:
            if i == parent_eid:
                continue
            if disc[v] == 0:
                children += 1
                estack.append(i)
                _dfs(v, i)
                low[u] = min(low[u], low[v])
                if low[v] >= disc[u]:
                    if disc[u] > 1 or children > 1:
                        is_cut[u] = True
                    c = ncomp[0]
                    ncomp[0] += 1
                    while True:
                        j = estack.pop()
                        comp_of_edge[j] = c
                        if j == i:
                            break
            elif disc[v] < disc[u]:
                estack.append(i)
                low[u] = min(low[u], disc[v])

    _dfs(0, -1)
    sys.setrecursionlimit(old_limit)
    if disc[1] == 0:
        return []                    # SNK unreachable from SRC: nothing conducts

    # ── Block-cut tree; keep blocks on the tree path SRC -> SNK ────────────
    comp_verts: List[Set[int]] = [set() for _ in range(ncomp[0])]
    for i, e in enumerate(edges):
        if comp_of_edge[i] >= 0:
            comp_verts[comp_of_edge[i]].update((e[0], e[1]))

    # tree nodes: ("c", comp_id) and ("v", cut_vertex); leaves via membership
    tree: Dict[Tuple[str, int], List[Tuple[str, int]]] = {}
    for c in range(ncomp[0]):
        cn = ("c", c)
        tree.setdefault(cn, [])
        for w in comp_verts[c]:
            if is_cut[w]:
                vn = ("v", w)
                tree.setdefault(vn, [])
                tree[cn].append(vn)
                tree[vn].append(cn)

    def _node_of(x: int) -> Tuple[str, int]:
        if is_cut[x]:
            return ("v", x)
        for c in range(ncomp[0]):
            if x in comp_verts[c]:
                return ("c", c)
        return ("v", x)              # isolated terminal (no conducting edges)

    src_n, snk_n = _node_of(0), _node_of(1)
    prev: Dict[Tuple[str, int], Tuple[str, int]] = {src_n: src_n}
    q: deque = deque([src_n])
    while q and snk_n not in prev:
        n = q.popleft()
        for nb in tree.get(n, []):
            if nb not in prev:
                prev[nb] = n
                q.append(nb)
    if snk_n not in prev:
        return []

    path_comps: Set[int] = set()
    n = snk_n
    while True:
        if n[0] == "c":
            path_comps.add(n[1])
        if n == src_n:
            break
        n = prev[n]

    return [e for i, e in enumerate(edges) if comp_of_edge[i] in path_comps]


def prune_gtran_functional(g_tran, g_num: int, on_patterns) -> list:
    """Greedy irredundant reduction: repeatedly remove any transistor whose
    removal leaves every on-pattern covered.

    Removing an edge can only REDUCE conduction, so off-pattern safety is
    automatically preserved — the result computes exactly the same switching
    function with <= transistors. Strictly subsumes prune_gtran (dead edges
    never contribute coverage). Greedy, so the result is a (locally)
    irredundant network, not necessarily the global minimum.
    """
    from .literal import covered_patterns

    on_patterns = frozenset(on_patterns)
    edges = prune_gtran(g_tran, g_num)      # cheap exact-topological pass first
    changed = True
    while changed:
        changed = False
        for i in range(len(edges)):
            cand = edges[:i] + edges[i + 1:]
            if covered_patterns(cand, g_num, on_patterns) == on_patterns:
                # removal may orphan further structure — re-prune topologically
                edges = prune_gtran(cand, g_num)
                changed = True
                break
    return edges


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


def assemble_graph_from_tensors(
    edge_list: torch.Tensor,     # [E, 3] long
    edge_feature: torch.Tensor,  # [E, EDGE_FEAT_DIM] float
    node_feature: torch.Tensor,  # [N, NODE_FEAT_DIM] float
    g_num_nodes,                 # int or 0-dim long tensor
) -> td_data.Graph:
    """Assemble a Graph directly from precomputed tensors (columnar dataset path).

    Same result as `_assemble_graph` but with edge_list / node_feature / edge_feature
    already materialised — no per-edge Python loops, no build_gen_graph recomputation.
    """
    n = int(node_feature.shape[0])
    graph = td_data.Graph(edge_list=edge_list, num_node=n, num_relation=1)
    with graph.edge():
        graph.edge_feature = edge_feature
    with graph.node():
        graph.node_feature = node_feature
    with graph.graph():
        graph.g_num_nodes = (g_num_nodes if torch.is_tensor(g_num_nodes)
                             else torch.tensor(g_num_nodes, dtype=torch.long))
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
