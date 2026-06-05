# Graph Representation: Switching Graph and Unified G∪C Graph

**Core files:** `transnet/graph.py`, `transnet/literal.py`

---

## Switching Graph Concept

A **switching graph** G = (V, E) represents a transistor network:
- **Nodes** = circuit terminals: SOURCE (0), SINK (1), internal nodes
- **Edges** = transistors, each labeled with a (variable, polarity) literal
- A path SOURCE→SINK exists under input pattern `x` iff all transistors on the path are active (their gate variable has the right value)

For a PDN (pull-down network): the switching graph conducts (SOURCE→SINK path exists) when the function evaluates to the required value.

---

## Node Layout

Every graph in this codebase uses a **unified G∪C layout**:

```
index 0         = SOURCE         (shared G and C)
index 1         = SINK           (shared G and C)
index 2..g_n-1  = INTERNAL_G    (intermediate G-network nodes)
index g_n..end  = INTERNAL_C    (compensate-network nodes, read-only)
```

`g_num_nodes` (stored as a graph attribute) = count of `{SOURCE, SINK, INTERNAL_G}`.

**Node features** (4-dim one-hot, `NODE_FEAT_DIM=4`):
```
[is_src, is_snk, is_G, is_C]
```

---

## Edge Features

**`EDGE_FEAT_DIM = N_VARS + 2 = 5`** (for sweep_3input with N_VARS=3):

```
[one_hot(global_var_idx, 3)…, is_neg, is_compensate]
```

- `one_hot(global_var_idx, 3)`: which variable (a=0, b=1, c=2) gates this transistor
- `is_neg`: 0=positive literal (transistor on when var=1), 1=negative literal (transistor on when var=0)
- `is_compensate`: 1 for C-network edges, 0 for G-network edges

Both directions of each undirected edge are stored (torchdrug convention requires directed edges).

---

## C-Network (Compensate Network)

The C-network encodes the **sum-of-products (SOP) expansion** of the function's on-patterns. It is a fixed, read-only series-parallel network: one series chain of transistors per on-pattern minterm, all connected in parallel between SOURCE and SINK.

Built by `build_sop_edges` in `literal.py:110`:
```python
for pi_all in sorted(on_patterns):      # one chain per minterm
    chain = [SOURCE, c1, c2, ..., SINK] # K-1 internal C-nodes
    for j in range(K):
        gv  = ALL_VARS.index(vars_in_func[j])
        neg = int(pi_k[j] == 0)          # negative literal if bit=0
        edges.append((chain[j], chain[j+1], gv, neg))
```

**Purpose:** The C-network is always present as context in every graph. It gives the MPNN a "ground truth hint" about which paths need to exist — the G-network transistors are being added relative to this context.

---

## Generation Graph (`build_gen_graph`)

Used during trajectory generation (`graph.py:260`). Built from the current partial G-transistor list `g_tran` plus the full C-network:

```python
def build_gen_graph(g_tran, g_num, vars_in_func, on_patterns) -> td_data.Graph:
    # g_tran: list of (u_compact, v_compact, var_idx, is_neg)
    # g_num:  current number of compact G-nodes
```

Starts with `g_num=2` (only SOURCE and SINK), grows by 1 each time a new internal node is created.

---

## Training Graph (`build_transnet_graph`)

Used during NLL pretraining (`graph.py:186`). Builds the full G∪C graph from a complete dataset transistor network. Remaps dataset node indices to unified layout:
- dataset node 0 → 0 (SOURCE)
- dataset node nc-1 → 1 (SINK)
- dataset node k → k+1 (INTERNAL_G)

---

## Prefix Graph (`build_prefix_graph`)

Used during NLL training to create partial graphs at each snake-pattern step `k` (`graph.py:217`). Snake ordering: `_snake_id(u,v) = max(u,v) * (max(u,v)+1) + min(u,v)`. This gives a canonical order for adding transistors during teacher-forced training.

---

## Literal Encoding

From `transnet/literal.py`:

```python
ALL_VARS = ['a', 'b', 'c']   # for sweep_3input
N_VARS   = 3
EDGE_FEAT_DIM = 5             # N_VARS + 2
```

**Dataset literal format:** `literal_id 0..K-1` = positive, `K..2K-1` = negative, where K = `len(vars_in_func)` (local function variable count). Decoded by `decode_literal`:
```python
local_idx  = lit_id % K
is_neg     = int(lit_id >= K)
global_idx = ALL_VARS.index(vars_in_func[local_idx])
```

---

## BFS Safety and Coverage (`literal.py:149-195`)

**`check_safety(g_tran, num_nodes, off_patterns)`:**  
BFS under each off-pattern. Returns True iff no SOURCE→SINK path exists under any off-pattern.

**`covered_patterns(g_tran, num_nodes, on_patterns)`:**  
BFS under each on-pattern. Returns frozenset of on-patterns for which a SOURCE→SINK path exists.

These are called at every step of trajectory generation. Profiling showed:
- BFS (n2 safety check): ~6% of wall time
- BFS (var/sign safety): ~2% of wall time
- MPNN forward: ~70% of wall time  ← the real bottleneck

---

## Graph Constants

```python
MAX_G_NODES = 20   # upper bound on compact G-node count (graph.py:30)
```

Node-2 safety mask `node2_safety_mask` has shape `[B, MAX_G_NODES]` in NLL batches.
