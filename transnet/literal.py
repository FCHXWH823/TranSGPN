"""
Literal encoding, SOP construction, and BFS utilities.

Literal rule (dataset-confirmed):
  vars_in_func = sorted distinct variables appearing in the expression
  K = len(vars_in_func)
  literal_id 0..K-1  -> positive: vars_in_func[id]
  literal_id K..2K-1 -> negative: vars_in_func[id - K]

Edge feature encoding (N_VARS = 3 for sweep_3input):
  [one_hot(global_var_idx, N_VARS)..., is_neg, is_compensate]
  dim = N_VARS + 2 = 5
"""

import os
import re
import itertools
from collections import deque
from typing import List, Tuple, FrozenSet

# Variable set is configurable via the TRANSGPN_VARS env var so the same code
# supports 4-input (default 'abcd') and 6-input ('abcdef') models. Must be set
# BEFORE importing transnet. Old 4-input checkpoints require the default.
ALL_VARS = list(os.environ.get("TRANSGPN_VARS", "abcd"))
N_VARS = len(ALL_VARS)            # 4 by default, 6 for sweep_6input
EDGE_FEAT_DIM = N_VARS + 2        # 6 by default, 8 for 6-input
_VAR_CLASS = "[" + "".join(ALL_VARS) + "]"   # regex char class, e.g. [abcdef]


# ---------------------------------------------------------------------------
# Literal decoding
# ---------------------------------------------------------------------------

def extract_vars(expr: str) -> List[str]:
    """Return sorted variable names that appear in SOP expression."""
    return sorted(set(re.findall(_VAR_CLASS, expr)))


def decode_literal(lit_id: int, vars_in_func: List[str]) -> Tuple[int, int]:
    """
    Decode a literal index from the CG format.

    Returns:
        (global_var_idx, is_neg)
        global_var_idx: index into ALL_VARS  (0=a, 1=b, 2=c)
    """
    K = len(vars_in_func)
    local_idx = lit_id % K
    is_neg = int(lit_id >= K)
    global_idx = ALL_VARS.index(vars_in_func[local_idx])
    return global_idx, is_neg


def make_edge_feat(global_var: int, is_neg: int, is_compensate: int) -> List[float]:
    """Build the EDGE_FEAT_DIM-element edge feature vector."""
    oh = [0.0] * N_VARS
    oh[global_var] = 1.0
    return oh + [float(is_neg), float(is_compensate)]


# ---------------------------------------------------------------------------
# Pattern enumeration
# ---------------------------------------------------------------------------

def all_patterns_n(N: int) -> FrozenSet[Tuple[int, ...]]:
    return frozenset(itertools.product([0, 1], repeat=N))


def parse_sop_expr(expr: str, vars_in_func: List[str]) -> FrozenSet[Tuple[int, ...]]:
    """
    Parse a SOP expression like 'a*!b+!a*b*c' into a frozenset of
    on-patterns as N-tuples over ALL_VARS.

    vars_in_func: sorted K-variable list from extract_vars().
    """
    N = N_VARS
    K = len(vars_in_func)
    on_patterns = set()

    for term in expr.split('+'):
        term = term.strip()
        lits = re.findall(r'!?' + _VAR_CLASS, term)
        # iterate over all K-local assignments
        for local_pat in itertools.product([0, 1], repeat=K):
            ctx = {vars_in_func[i]: local_pat[i] for i in range(K)}
            ok = True
            for lit in lits:
                if lit.startswith('!'):
                    if ctx.get(lit[1:], 1) != 0:
                        ok = False; break
                else:
                    if ctx.get(lit, 0) != 1:
                        ok = False; break
            if not ok:
                continue
            # extend to all N variables (free variables take both values)
            free = [v for v in ALL_VARS if v not in ctx]
            for ext in itertools.product([0, 1], repeat=len(free)):
                full = dict(ctx)
                full.update(dict(zip(free, ext)))
                on_patterns.add(tuple(full[v] for v in ALL_VARS))

    return frozenset(on_patterns)


def on_off_split(on_patterns: FrozenSet) -> Tuple[FrozenSet, FrozenSet]:
    all_p = all_patterns_n(N_VARS)
    return on_patterns, all_p - on_patterns


# ---------------------------------------------------------------------------
# SOP compensate network
# ---------------------------------------------------------------------------

def build_sop_edges(
    on_patterns: FrozenSet[Tuple[int, ...]],
    vars_in_func: List[str],
) -> Tuple[List[Tuple[int, int, int, int]], int]:
    """
    Build the SOP series-parallel network for C_t.

    Node layout: 0=SOURCE, 1=SINK, 2+ = INTERNAL_C
    Returns:
        edges: list of (src, drn, global_var_idx, is_neg)
        total_c_nodes: total node count (including SOURCE/SINK)
    """
    K = len(vars_in_func)
    edges = []
    next_node = 2  # first INTERNAL_C index

    for pi_all in sorted(on_patterns):
        # Project onto vars_in_func (K-local bit pattern)
        pi_k = tuple(pi_all[ALL_VARS.index(v)] for v in vars_in_func)

        # Chain: SOURCE -> c1 -> c2 -> ... -> SINK  (K transistors)
        chain = [0]
        for _ in range(K - 1):
            chain.append(next_node)
            next_node += 1
        chain.append(1)

        for j in range(K):
            gv = ALL_VARS.index(vars_in_func[j])
            neg = int(pi_k[j] == 0)   # SOP: if bit=0 use negated literal
            edges.append((chain[j], chain[j + 1], gv, neg))

    return edges, next_node   # next_node == total C-node count


# ---------------------------------------------------------------------------
# BFS safety and coverage
# ---------------------------------------------------------------------------

def _bfs_reaches_sink(
    g_transistors: List[Tuple[int, int, int, int]],
    num_nodes: int,
    pattern: Tuple[int, ...],
) -> bool:
    """BFS on G-transistors under `pattern`. True if SOURCE->SINK path exists."""
    adj: List[List[int]] = [[] for _ in range(num_nodes)]
    for (u, v, var, neg) in g_transistors:
        val = pattern[var]
        conducts = (val == 0) if neg else (val == 1)
        if conducts:
            adj[u].append(v)
            adj[v].append(u)
    visited = {0}
    q = deque([0])
    while q:
        node = q.popleft()
        if node == 1:
            return True
        for nb in adj[node]:
            if nb not in visited:
                visited.add(nb)
                q.append(nb)
    return False


def check_safety(
    g_transistors: List[Tuple[int, int, int, int]],
    num_nodes: int,
    off_patterns: FrozenSet,
) -> bool:
    """True if no S->T path under any off-pattern."""
    return not any(
        _bfs_reaches_sink(g_transistors, num_nodes, pi) for pi in off_patterns
    )


def covered_patterns(
    g_transistors: List[Tuple[int, int, int, int]],
    num_nodes: int,
    on_patterns: FrozenSet,
) -> FrozenSet:
    """Return the subset of on_patterns already routed in G."""
    return frozenset(
        pi for pi in on_patterns
        if _bfs_reaches_sink(g_transistors, num_nodes, pi)
    )
