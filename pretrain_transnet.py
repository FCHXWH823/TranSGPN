"""
pretrain_transnet.py

Phase 1: NLL pretraining for transistor-network synthesis.

Usage:
    python pretrain_transnet.py [--epochs N] [--batch-size B] [--lr LR]
"""
import argparse
import os
import time

import torch
from torchdrug import data as td_data, models

from transnet import GCPNTransNet, TransistorDataset

# ── CLI args ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--epochs",     type=int,   default=50)
parser.add_argument("--batch-size", type=int,   default=64)
parser.add_argument("--lr",         type=float, default=1e-3)
parser.add_argument("--hidden",     type=int,   default=64,  help="MPNN hidden dim")
parser.add_argument("--num-layer",  type=int,   default=3,   help="MPNN message-passing layers")
parser.add_argument("--mlp-hidden", type=int,   default=128, help="Policy MLP hidden dim")
parser.add_argument("--checkpoint", type=str,   default="checkpoints/transnet_pretrain.pt")
parser.add_argument("--log-interval", type=int, default=20)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Dataset ──────────────────────────────────────────────────────────────────
print("Loading TransistorDataset …")
t0 = time.time()
dataset = TransistorDataset()
print(f"  {len(dataset)} prefix samples  ({time.time()-t0:.1f}s)")

loader = td_data.DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=0,
)

# ── Model ────────────────────────────────────────────────────────────────────
from transnet.graph import NODE_FEAT_DIM
from transnet.literal import EDGE_FEAT_DIM

mpnn = models.MPNN(
    input_dim      = NODE_FEAT_DIM,
    hidden_dim     = args.hidden,
    edge_input_dim = EDGE_FEAT_DIM,
    num_layer      = args.num_layer,
    batch_norm     = False,
).to(device)

task = GCPNTransNet(mpnn, hidden_dim_mlp=args.mlp_hidden).to(device)
optimizer = torch.optim.Adam(task.parameters(), lr=args.lr)

# ── Training loop ─────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)
best_loss = float("inf")

for epoch in range(1, args.epochs + 1):
    task.train()
    epoch_loss = 0.0
    n_batch = 0

    for step, batch in enumerate(loader):
        # Move graph to device; move scalar label tensors to device
        graph = batch["graph"].to(device)
        labels = {
            k: v.to(device)
            for k, v in batch.items()
            if k != "graph"
        }
        labels["graph"] = graph

        optimizer.zero_grad()
        loss, metric = task(labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(task.parameters(), 1.0)
        optimizer.step()

        epoch_loss += loss.item()
        n_batch += 1

        if (step + 1) % args.log_interval == 0:
            print(
                f"  epoch {epoch:3d}  step {step+1:4d}/{len(loader)}"
                f"  loss={metric['loss/total']:.4f}"
                f"  n1={metric['loss/node1']:.3f}"
                f"  n2={metric['loss/node2']:.3f}"
                f"  var={metric['loss/var']:.3f}"
                f"  sign={metric['loss/sign']:.3f}"
                f"  acc_n1={metric['acc/node1']:.3f}"
                f"  acc_n2={metric['acc/node2']:.3f}"
            )

    avg_loss = epoch_loss / n_batch
    print(f"Epoch {epoch:3d}  avg_loss={avg_loss:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save({
            "epoch": epoch,
            "model_state": task.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "loss": best_loss,
            "args": vars(args),
        }, args.checkpoint)
        print(f"  → checkpoint saved  (loss={best_loss:.4f})")

print("Done.")
