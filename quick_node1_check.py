"""
quick_node1_check.py

Overfit sanity check for the node1 head: can the CURRENT model config
(hidden=128, num_layer=3, mlp_hidden=256) reduce node1 loss when labels are
UNIQUE (one_impl_per_function), on a small 2k-row subset of sweep_6input_all?

Full-data runs plateau at acc_n1~48% with ~2.3 conflicting impls/function.
- acc_n1 climbs high here  -> model can learn node1; plateau = label conflict.
- acc_n1 stalls ~48% here  -> architecture problem (node1 head / oversmoothing).
"""
import sys
import time

import torch
from torchdrug import data as td_data, models

from transnet import GCPNTransNet, TransistorDataset
from transnet.graph import NODE_FEAT_DIM
from transnet.literal import EDGE_FEAT_DIM

GLOBAL_ATTN  = "--global-attn" in sys.argv
POINTER_HEAD = "--pointer-head" in sys.argv
print(f"global_attn: {GLOBAL_ATTN}  pointer_head: {POINTER_HEAD}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

import os
_order = os.environ.get("TRANSGPN_ORDER", "bfs")
_cache = ("dataset/cache6_quickcheck_onefunc2k.pt" if _order == "bfs"
          else f"dataset/cache6_quickcheck_onefunc2k_{_order}.pt")
print(f"Building unique-label subset (one_impl_per_function, 2000 rows, "
      f"order={_order}) …")
t0 = time.time()
dataset = TransistorDataset(
    dataset_roots=["dataset/sweep_6input_all"],
    cache_path=_cache,
    max_functions=2000, seed=0,
    verify=False,
    one_impl_per_function=True,
)
print(f"  {len(dataset)} prefix samples  ({time.time()-t0:.1f}s)")

loader = td_data.DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)

mpnn = models.MPNN(
    input_dim      = NODE_FEAT_DIM,
    hidden_dim     = 128,
    edge_input_dim = EDGE_FEAT_DIM,
    num_layer      = 3,
    batch_norm     = False,
).to(device)
task = GCPNTransNet(mpnn, hidden_dim_mlp=256, global_attn=GLOBAL_ATTN,
                    pointer_head=POINTER_HEAD).to(device)
optimizer = torch.optim.Adam(task.parameters(), lr=1e-3)

for epoch in range(1, 31):
    task.train()
    agg = {"loss": 0.0, "n1": 0.0, "acc_n1": 0.0, "acc_n2": 0.0}
    n_batch = 0
    for batch in loader:
        graph = batch["graph"].to(device)
        labels = {k: v.to(device) for k, v in batch.items() if k != "graph"}
        labels["graph"] = graph

        optimizer.zero_grad()
        loss, metric = task(labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(task.parameters(), 1.0)
        optimizer.step()

        agg["loss"]   += metric["loss/total"]
        agg["n1"]     += metric["loss/node1"]
        agg["acc_n1"] += metric["acc/node1"]
        agg["acc_n2"] += metric["acc/node2"]
        n_batch += 1

    print(f"Epoch {epoch:3d}  loss={agg['loss']/n_batch:.4f}"
          f"  n1loss={agg['n1']/n_batch:.4f}"
          f"  acc_n1={agg['acc_n1']/n_batch:.4f}"
          f"  acc_n2={agg['acc_n2']/n_batch:.4f}")

print("QUICK_CHECK_DONE")
