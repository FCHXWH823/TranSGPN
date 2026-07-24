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
parser.add_argument("--batch-size",   type=int,   default=64)
parser.add_argument("--num-workers",  type=int,   default=0,
                    help="DataLoader worker processes. Set to 4-8 when training on GPU.")
parser.add_argument("--lr",           type=float, default=1e-3)
parser.add_argument("--hidden",     type=int,   default=64,  help="MPNN hidden dim")
parser.add_argument("--num-layer",  type=int,   default=3,   help="MPNN message-passing layers")
parser.add_argument("--mlp-hidden", type=int,   default=128, help="Policy MLP hidden dim")
parser.add_argument("--global-attn", action="store_true",
                    help="Add a global self-attention block between the MPNN "
                         "and the policy heads (full-graph receptive field).")
parser.add_argument("--pointer-head", action="store_true",
                    help="Pointer-network node1 scoring (graph-query · node-key) "
                         "instead of the per-node concat-MLP.")
parser.add_argument("--w-node1",  type=float, default=1.0,
                    help="Weight of the node1 loss term in the total loss.")
parser.add_argument("--cosine-lr", action="store_true",
                    help="Cosine-decay LR from --lr to --min-lr over the full "
                         "epoch range (deterministic in epoch → resume-safe).")
parser.add_argument("--min-lr",   type=float, default=None,
                    help="Final LR for --cosine-lr (default: --lr / 20).")
parser.add_argument("--checkpoint",   type=str,   default="checkpoints/transnet_pretrain_3_4input_v1.pt")
parser.add_argument("--cache",        type=str,   default="dataset/cache_3_4input.pt",
                    help="Path to save/load preprocessed dataset cache (saves ~56 min on reload).")
parser.add_argument("--dataset-root", type=str,   action="append", default=None,
                    help="Dataset root dir(s) to load (repeatable). Defaults to "
                         "sweep_3input + sweep_4input_possani. Use for 6-input.")
parser.add_argument("--max-functions", type=int,  default=None,
                    help="Subsample at most this many functions per dataset root "
                         "(for very large sets like sweep_6input_possani).")
parser.add_argument("--seed",          type=int,  default=0,
                    help="Random seed for --max-functions subsampling.")
parser.add_argument("--log-interval", type=int,   default=20)
parser.add_argument("--resume",       type=str,   default=None,
                    help="Resume training from this checkpoint (continues model, "
                         "optimizer, best_loss and epoch counter). Train --epochs MORE epochs.")
parser.add_argument("--until-epoch",  type=int,   default=None,
                    help="Train until this TOTAL epoch number (overrides --epochs). "
                         "Resumable toward a fixed total across multiple jobs.")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Dataset ──────────────────────────────────────────────────────────────────
print("Loading TransistorDataset …")
t0 = time.time()
dataset = TransistorDataset(dataset_roots=args.dataset_root, cache_path=args.cache,
                            max_functions=args.max_functions, seed=args.seed)
print(f"  {len(dataset)} prefix samples  ({time.time()-t0:.1f}s)")

# Prevent copy-on-write RAM blow-up with persistent workers: the dataset is millions
# of Python objects shared via fork. Each worker's GC traversal would dirty every
# shared page (CoW) and replicate the whole dataset per worker -> OOM. gc.freeze()
# moves all current objects to a permanent generation the GC never scans, and each
# worker disables GC entirely, so the shared pages stay shared.
import gc
gc.collect()
gc.freeze()

def _worker_init(_wid):
    import gc as _gc
    _gc.disable()

loader = td_data.DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    worker_init_fn=_worker_init,
    # Keep workers alive across epochs (avoid respawn stalls that idle the GPU
    # at every epoch boundary), and prefetch deeper so collation overlaps compute.
    persistent_workers=(args.num_workers > 0),
    prefetch_factor=(2 if args.num_workers > 0 else None),
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

task = GCPNTransNet(mpnn, hidden_dim_mlp=args.mlp_hidden,
                    global_attn=args.global_attn,
                    pointer_head=args.pointer_head,
                    w_node1=args.w_node1).to(device)
optimizer = torch.optim.Adam(task.parameters(), lr=args.lr)

# ── Resume from checkpoint (optional) ─────────────────────────────────────────
best_loss   = float("inf")
start_epoch = 1
if args.resume:
    print(f"Resuming from checkpoint: {args.resume}")
    ck = torch.load(args.resume, map_location=device)
    task.load_state_dict(ck["model_state"])
    if "optimizer_state" in ck:
        optimizer.load_state_dict(ck["optimizer_state"])
    # "best_loss" is carried in .latest checkpoints so resuming from a
    # non-best epoch never lets a worse model overwrite the best file.
    best_loss   = ck.get("best_loss", ck.get("loss", float("inf")))
    start_epoch = ck.get("epoch", 0) + 1
    print(f"  loaded epoch {ck.get('epoch')}  best_loss={best_loss:.4f}  "
          f"→ continuing for {args.epochs} more epochs (epochs "
          f"{start_epoch}..{start_epoch + args.epochs - 1})")

# ── Training loop ─────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)

end_epoch = args.until_epoch if args.until_epoch else (start_epoch + args.epochs - 1)
if start_epoch > end_epoch:
    print(f"Already at epoch {start_epoch-1} ≥ target {end_epoch}; nothing to do.")
print(f"Training epochs {start_epoch}..{end_epoch}")

import math

def _set_cosine_lr(epoch: int) -> None:
    """LR as a pure function of epoch — safe across resumes (overrides the
    LR restored from the checkpoint's optimizer state)."""
    lo = args.min_lr if args.min_lr is not None else args.lr / 20
    t  = (epoch - 1) / max(1, end_epoch - 1)
    lr = lo + 0.5 * (args.lr - lo) * (1 + math.cos(math.pi * t))
    for g in optimizer.param_groups:
        g["lr"] = lr

for epoch in range(start_epoch, end_epoch + 1):
    if args.cosine_lr:
        _set_cosine_lr(epoch)
        print(f"  lr = {optimizer.param_groups[0]['lr']:.2e}")
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

    # Always save a rolling "latest" checkpoint so multi-window runs make
    # epoch progress even when avg_loss is not improving (best stays separate).
    torch.save({
        "epoch": epoch,
        "model_state": task.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "loss": avg_loss,
        "best_loss": best_loss,
        "args": vars(args),
    }, args.checkpoint + ".latest")

print("Done.")
