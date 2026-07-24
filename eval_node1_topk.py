"""
eval_node1_topk.py

Measure node1 top-k accuracy (k=1,2,3,5,10) of a pretrained checkpoint on a
sample of the onefunc columnar cache. Rationale: RL samples from the policy
distribution rather than taking argmax, so if the GT node1 is reliably in the
top few candidates, the pretrained prior may already be good enough for RL
even though top-1 accuracy looks low.

Usage: python eval_node1_topk.py <checkpoint> [--global-attn]
"""
import sys

import torch
from torchdrug import data as td_data, models

from transnet import GCPNTransNet
from transnet.graph import NODE_FEAT_DIM, assemble_graph_from_tensors
from transnet.literal import EDGE_FEAT_DIM

CKPT = sys.argv[1]
GLOBAL_ATTN = "--global-attn" in sys.argv
N_EVAL = 20000
BATCH = 256
KS = (1, 2, 3, 5, 10)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"ckpt: {CKPT}  global_attn: {GLOBAL_ATTN}  device: {device}")

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
a = ck.get("args", {})
mpnn = models.MPNN(
    input_dim      = NODE_FEAT_DIM,
    hidden_dim     = a.get("hidden", 128),
    edge_input_dim = EDGE_FEAT_DIM,
    num_layer      = a.get("num_layer", 3),
    batch_norm     = False,
)
task = GCPNTransNet(mpnn, hidden_dim_mlp=a.get("mlp_hidden", 256),
                    global_attn=GLOBAL_ATTN).to(device)
task.load_state_dict(ck["model_state"])
task.eval()
print(f"loaded epoch {ck.get('epoch')}  loss {ck.get('loss'):.4f}  "
      f"(hidden={a.get('hidden')}, layers={a.get('num_layer')})")

s = torch.load("dataset/cache6_onefunc_columnar_shards/col_shard0.pt",
               weights_only=False)
n_tot = s["node1"].shape[0]
e_off = torch.cat([torch.zeros(1, dtype=torch.long), s["edge_count"].cumsum(0)])
n_off = torch.cat([torch.zeros(1, dtype=torch.long), s["node_count"].cumsum(0)])
import random
idxs = random.Random(0).sample(range(n_tot), min(N_EVAL, n_tot))

hits = {k: 0 for k in KS}
ranks_sum = 0
n_done = 0
with torch.no_grad():
    for b0 in range(0, len(idxs), BATCH):
        chunk = idxs[b0:b0 + BATCH]
        graphs, labels = [], []
        for i in chunk:
            e0, e1 = int(e_off[i]), int(e_off[i + 1])
            n0, n1 = int(n_off[i]), int(n_off[i + 1])
            graphs.append(assemble_graph_from_tensors(
                s["edge_list"][e0:e1], s["edge_feature"][e0:e1].float(),
                s["node_feature"][n0:n1].float(), int(s["g_num_nodes"][i])))
            labels.append(int(s["node1"][i]))
        packed = td_data.Graph.pack(graphs).to(device)
        B = packed.batch_size
        total_N = packed.num_node

        node_feat, graph_feat = task._encode(packed)
        virt_feat = task.new_node_emb.unsqueeze(0).expand(B, -1)
        ext_feat  = torch.cat([node_feat, virt_feat], dim=0)
        n2g       = packed.node2graph
        ext_n2g   = torch.cat([n2g, torch.arange(B, device=device)])
        ext_gfeat = graph_feat[ext_n2g]

        starts      = packed.num_cum_nodes - packed.num_nodes
        local_idx   = torch.arange(total_N, device=device) - starts[n2g]
        is_g_actual = local_idx < packed.g_num_nodes[n2g]
        is_g_ext    = torch.cat([is_g_actual,
                                 torch.ones(B, dtype=torch.bool, device=device)])

        n1_logits = task._n1_logits(ext_feat, ext_gfeat)
        n1_logits = n1_logits.masked_fill(~is_g_ext, -1e9)

        gt_global = starts + torch.tensor(labels, device=device)
        gt_logit  = n1_logits[gt_global]                       # [B]
        # rank of GT within its own graph (0 = best)
        rank = torch.zeros(B, dtype=torch.long, device=device)
        better = (n1_logits > gt_logit[ext_n2g])
        rank.scatter_add_(0, ext_n2g, better.long())
        for k in KS:
            hits[k] += int((rank < k).sum())
        ranks_sum += int(rank.sum())
        n_done += B
        if n_done % 5120 == 0:
            print(f"  {n_done}/{len(idxs)}", flush=True)

print(f"\n=== node1 top-k accuracy  (n={n_done}) ===")
for k in KS:
    print(f"  acc@{k:<2d} = {hits[k]/n_done:.4f}")
print(f"  mean GT rank = {ranks_sum/n_done:.2f}")
print("TOPK_DONE")
