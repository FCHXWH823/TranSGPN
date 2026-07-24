"""
analyze_node1_ceiling.py

Tier-1 diagnostic: how much of the node1 accuracy gap is a WL-symmetry ceiling?

An L-layer MPNN cannot distinguish nodes with identical L-iteration WL colors.
If the labeled node1 has k WL-equivalent candidate nodes (same color, index <
g_num_nodes), any MPNN's expected top-1 accuracy on that sample is <= 1/k.
Mean over samples = the achievable acc_n1 ceiling for that depth.

Runs on:  6-input onefunc columnar shard0  +  4-input canon cache (validation:
model reached acc_n1=0.87 there, so its ceiling must be >= ~0.87).
"""
import random
import sys
import time

import torch

MAX_ITERS = 10          # "converged" WL (with early stop)
REPORT_ITERS = (1, 2, 3, 5, MAX_ITERS)
N_SAMPLE_6 = 30000
N_SAMPLE_4 = 20000
rng = random.Random(0)


def wl_ceiling(graphs, label_of, title):
    """graphs: list of (edge_list[E,3] long, edge_feature[E,F], node_feature[N,D],
    g_num_nodes int); label_of: list of node1 labels. Prints ceiling table."""
    stats = {it: [] for it in REPORT_ITERS}
    t0 = time.time()
    for gi, (el, ef, nf, gn) in enumerate(graphs):
        n = nf.shape[0]
        label = label_of[gi]
        # initial colors: node feature row identity
        interner = {}
        colors = []
        for v in range(n):
            key = tuple(nf[v].tolist())
            colors.append(interner.setdefault(key, len(interner)))
        # incident (direction, relation, edge-feature, neighbor) per node
        src = el[:, 0].tolist(); dst = el[:, 1].tolist(); rel = el[:, 2].tolist()
        efeat = [tuple(ef[e].tolist()) for e in range(el.shape[0])]
        inc = [[] for _ in range(n)]
        for e in range(el.shape[0]):
            inc[dst[e]].append((0, rel[e], efeat[e], src[e]))
            inc[src[e]].append((1, rel[e], efeat[e], dst[e]))

        prev_parts = None
        for it in range(1, MAX_ITERS + 1):
            interner2 = {}
            new_colors = [0] * n
            for v in range(n):
                sig = (colors[v],
                       tuple(sorted((d, r, f, colors[u]) for d, r, f, u in inc[v])))
                new_colors[v] = interner2.setdefault(sig, len(interner2))
            colors = new_colors
            if it in stats or it == MAX_ITERS:
                k = sum(1 for v in range(min(gn, n))
                        if colors[v] == colors[label])
                if it in stats:
                    stats[it].append(k)
            nparts = len(interner2)
            if prev_parts == nparts:            # converged: fill remaining iters
                k = sum(1 for v in range(min(gn, n)) if colors[v] == colors[label])
                for it2 in REPORT_ITERS:
                    if it2 > it:
                        stats[it2].append(k)
                break
            prev_parts = nparts
        if (gi + 1) % 5000 == 0:
            print(f"  {title}: {gi+1}/{len(graphs)}  ({time.time()-t0:.0f}s)",
                  flush=True)

    print(f"\n=== {title}  (n={len(graphs)}) ===")
    print(f"{'WL iters':>9} {'ceiling':>8} {'k=1':>7} {'k=2':>7} {'k>=3':>7} {'mean_k':>7}")
    for it in REPORT_ITERS:
        ks = stats[it]
        ceiling = sum(1.0 / k for k in ks) / len(ks)
        p1 = sum(1 for k in ks if k == 1) / len(ks)
        p2 = sum(1 for k in ks if k == 2) / len(ks)
        p3 = sum(1 for k in ks if k >= 3) / len(ks)
        mk = sum(ks) / len(ks)
        print(f"{it:>9} {ceiling:>8.3f} {p1:>7.1%} {p2:>7.1%} {p3:>7.1%} {mk:>7.2f}")


# ── 6-input onefunc columnar shard0 ─────────────────────────────────────────
print("Loading 6-input onefunc col_shard0 …", flush=True)
s = torch.load("dataset/cache6_onefunc_columnar_shards/col_shard0.pt",
               weights_only=False)
n_tot = s["node1"].shape[0]
e_off = torch.cat([torch.zeros(1, dtype=torch.long), s["edge_count"].cumsum(0)])
n_off = torch.cat([torch.zeros(1, dtype=torch.long), s["node_count"].cumsum(0)])
idxs = rng.sample(range(n_tot), min(N_SAMPLE_6, n_tot))
graphs, labels = [], []
for i in idxs:
    e0, e1 = int(e_off[i]), int(e_off[i + 1])
    n0, n1 = int(n_off[i]), int(n_off[i + 1])
    graphs.append((s["edge_list"][e0:e1], s["edge_feature"][e0:e1],
                   s["node_feature"][n0:n1], int(s["g_num_nodes"][i])))
    labels.append(int(s["node1"][i]))
del s
wl_ceiling(graphs, labels, "6-input onefunc (clean labels)")
del graphs

# ── 4-input canon cache (validation: observed acc_n1 ~= 0.87) ───────────────
print("\nLoading 4-input canon cache …", flush=True)
sys.path.insert(0, ".")
obj = torch.load("dataset/cache_3_4input_canon.pt", weights_only=False)
samples = obj["samples"] if isinstance(obj, dict) and "samples" in obj else obj
print(f"  type={type(samples)} len={len(samples)}", flush=True)
idxs = rng.sample(range(len(samples)), min(N_SAMPLE_4, len(samples)))
graphs, labels = [], []
for i in idxs:
    smp = samples[i]
    g = smp["graph"]
    graphs.append((g.edge_list.long(), g.edge_feature.float(),
                   g.node_feature.float(), int(g.g_num_nodes)))
    labels.append(int(smp["node1"]))
wl_ceiling(graphs, labels, "4-input canon (observed acc 0.87)")

print("\nCEILING_ANALYSIS_DONE")
