"""
analyze_node1_errors.py

Error anatomy for the node1 head of the final 6-input checkpoint. Tests four
hypotheses for the ~0.50 top-1 wall:

  H1 frontier/recency: GT node1 concentrates on recently-created nodes
     (compact index ≈ creation order), which the model cannot see directly.
     -> report: GT/pred index-recency distributions; acc when GT is newest.
  H2 near-ambiguity: wrong predictions are structural neighbors of the GT.
     -> report: fraction of errors where pred is 1-hop from GT (G-G edges).
  H3 late-step difficulty: accuracy vs graph size (g_num_nodes buckets).
  H4 head noise: mean softmax prob of GT vs pred on errors (confidence gap).

Usage: python analyze_node1_errors.py <checkpoint>
"""
import random
import sys
from collections import defaultdict

import torch
from torchdrug import data as td_data, models

from transnet import GCPNTransNet
from transnet.graph import NODE_FEAT_DIM, assemble_graph_from_tensors
from transnet.literal import EDGE_FEAT_DIM

CKPT = sys.argv[1] if len(sys.argv) > 1 else \
    "checkpoints/transnet_pretrain_6input_onefunc.pt"
N_EVAL = 20000
BATCH = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
a = ck.get("args", {})
mpnn = models.MPNN(input_dim=NODE_FEAT_DIM, hidden_dim=a.get("hidden", 128),
                   edge_input_dim=EDGE_FEAT_DIM, num_layer=a.get("num_layer", 3),
                   batch_norm=False)
task = GCPNTransNet(mpnn, hidden_dim_mlp=a.get("mlp_hidden", 256)).to(device)
task.load_state_dict(ck["model_state"], strict=False)
task.eval()
print(f"ckpt {CKPT} epoch {ck.get('epoch')}  device {device}", flush=True)

s = torch.load("dataset/cache6_onefunc_columnar_shards/col_shard0.pt",
               weights_only=False)
n_tot = s["node1"].shape[0]
e_off = torch.cat([torch.zeros(1, dtype=torch.long), s["edge_count"].cumsum(0)])
n_off = torch.cat([torch.zeros(1, dtype=torch.long), s["node_count"].cumsum(0)])
idxs = random.Random(0).sample(range(n_tot), min(N_EVAL, n_tot))

# aggregates
acc_by_gsize   = defaultdict(lambda: [0, 0])     # g_num bucket -> [hit, n]
acc_by_recency = defaultdict(lambda: [0, 0])     # GT recency class -> [hit, n]
gt_recency_n   = defaultdict(int)                # GT recency class -> count
pred_recency_n = defaultdict(int)
err_pred_neighbor = 0                            # errors where pred ~ 1-hop of GT
err_pred_newest   = 0                            # errors where pred is newest node
n_err = 0
gt_prob_sum = pred_prob_sum = 0.0                # on errors
n_hit = n_all = 0


def recency_class(idx, g_num):
    """0/1 = SRC/SNK; 'new-1'/'new-2' = newest/2nd-newest internal; 'old'."""
    if idx <= 1:
        return "src/snk"
    if idx == g_num - 1:
        return "newest"
    if idx == g_num - 2:
        return "2nd-newest"
    return "older"


with torch.no_grad():
    for b0 in range(0, len(idxs), BATCH):
        chunk = idxs[b0:b0 + BATCH]
        graphs, labels, gnums, adjs = [], [], [], []
        for i in chunk:
            e0, e1 = int(e_off[i]), int(e_off[i + 1])
            n0, n1 = int(n_off[i]), int(n_off[i + 1])
            el = s["edge_list"][e0:e1]
            gn = int(s["g_num_nodes"][i])
            graphs.append(assemble_graph_from_tensors(
                el, s["edge_feature"][e0:e1].float(),
                s["node_feature"][n0:n1].float(), gn))
            labels.append(int(s["node1"][i]))
            gnums.append(gn)
            # adjacency among G-nodes only (transistor edges)
            adj = set()
            for u, v in el[:, :2].tolist():
                if u < gn and v < gn:
                    adj.add((u, v)); adj.add((v, u))
            adjs.append(adj)
        packed = td_data.Graph.pack(graphs).to(device)
        B = packed.batch_size
        total_N = packed.num_node

        node_feat, graph_feat = task._encode(packed)
        virt = task.new_node_emb.unsqueeze(0).expand(B, -1)
        ext_feat = torch.cat([node_feat, virt], dim=0)
        n2g = packed.node2graph
        ext_n2g = torch.cat([n2g, torch.arange(B, device=device)])
        ext_gfeat = graph_feat[ext_n2g]

        starts = packed.num_cum_nodes - packed.num_nodes
        local_idx = torch.arange(total_N, device=device) - starts[n2g]
        is_g = torch.cat([local_idx < packed.g_num_nodes[n2g],
                          torch.ones(B, dtype=torch.bool, device=device)])
        logits = task._n1_logits(ext_feat, ext_gfeat).masked_fill(~is_g, -1e9)

        from torch_scatter.composite import scatter_log_softmax
        logprob = scatter_log_softmax(logits, ext_n2g)

        for b in range(B):
            s_i = int(starts[b]); N_i = int(packed.num_nodes[b]); gn = gnums[b]
            lg = logits[s_i:s_i + N_i]
            lp = logprob[s_i:s_i + N_i]
            pred = int(lg.argmax())
            gt = labels[b]
            hit = pred == gt
            n_all += 1; n_hit += hit

            gb = min(gn // 8 * 8, 40)          # g_num buckets of 8
            acc_by_gsize[gb][0] += hit; acc_by_gsize[gb][1] += 1
            rc = recency_class(gt, gn)
            gt_recency_n[rc] += 1
            acc_by_recency[rc][0] += hit; acc_by_recency[rc][1] += 1
            pred_recency_n[recency_class(pred, gn)] += 1

            if not hit:
                n_err += 1
                if (pred, gt) in adjs[b]:
                    err_pred_neighbor += 1
                if pred == gn - 1:
                    err_pred_newest += 1
                gt_prob_sum += float(lp[gt].exp())
                pred_prob_sum += float(lp[pred].exp())
        if n_all % 5120 == 0:
            print(f"  {n_all}/{len(idxs)}  acc={n_hit/n_all:.3f}", flush=True)

print(f"\n=== overall top-1 acc = {n_hit/n_all:.4f}  (n={n_all}) ===")
print("\nH3: accuracy by graph size (g_num_nodes bucket)")
for k in sorted(acc_by_gsize):
    h, n = acc_by_gsize[k]
    print(f"  g_num {k:>2d}-{k+7:<2d}: acc={h/n:.3f}  (n={n})")
print("\nH1: GT node1 recency distribution + accuracy per class")
for k in ("src/snk", "newest", "2nd-newest", "older"):
    h, n = acc_by_recency.get(k, (0, 0))
    print(f"  GT={k:<11s}: {gt_recency_n[k]/n_all:6.1%} of samples, "
          f"acc={h/max(1,n):.3f}   (pred goes here {pred_recency_n[k]/n_all:.1%})")
print(f"\nH2: errors where pred is 1-hop neighbor of GT: "
      f"{err_pred_neighbor/max(1,n_err):.1%} of {n_err} errors")
print(f"    errors where pred is the newest node:      "
      f"{err_pred_newest/max(1,n_err):.1%}")
print(f"H4: on errors, mean P(GT)={gt_prob_sum/max(1,n_err):.3f}  "
      f"mean P(pred)={pred_prob_sum/max(1,n_err):.3f}")
print("ERR_ANATOMY_DONE")
