"""
finetune_rl_transnet.py

Phase 2: RL finetuning to minimize transistor count.

Reward:
  complete network  →  t_ref / len(g_tran)   (1.0 = matches reference best)
  partial network   →  coverage_ratio * partial_coeff

Function selection (mutually exclusive, pick one):
  --func EXPR            one specific Boolean expression  (e.g. "!a*b+a*!b")
  --func_ids ID [ID...]  subset by dataset function_id
  --hard_threshold T     all functions with eval success_rate < T
  (no flag)              all SAT functions from dataset

Usage examples:
  # single XOR function
  python finetune_rl_transnet.py --func "!a*!b*c+!a*b*!c+a*!b*!c+a*b*c" --t_ref 5

  # two specific dataset functions
  python finetune_rl_transnet.py --func_ids PF3_001 PF3_007

  # hard functions only (pre-evaluation pass first)
  python finetune_rl_transnet.py --hard_threshold 0.7

  # all functions
  python finetune_rl_transnet.py
"""
import argparse
import csv
import math
import os
import random
import shutil
import time

import torch
from torchdrug import data as td_data, models

import transnet
from transnet import GCPNTransNet, TransistorDataset
from transnet.graph import NODE_FEAT_DIM
from transnet.literal import EDGE_FEAT_DIM, extract_vars, on_off_split, parse_sop_expr

_DATA_ROOT = os.path.join(os.path.dirname(__file__), "dataset", "sweep_3input")
_CSV_PATH  = os.path.join(_DATA_ROOT, "aggregate.csv")


# ── CLI ────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

# Function selection (use at most one)
grp = parser.add_mutually_exclusive_group()
grp.add_argument("--func",       type=str, default=None,
                 metavar="EXPR", help="Single Boolean expression to finetune on")
grp.add_argument("--func_ids",   type=str, nargs="+", default=None,
                 metavar="ID",   help="Dataset function_id(s) to finetune on")
grp.add_argument("--hard_threshold", type=float, default=None,
                 metavar="T",    help="Finetune only on functions with eval success_rate < T")

parser.add_argument("--t_ref",         type=int,   default=None,
                    help="Reference transistor count for reward = t_ref/T "
                         "(default: max_steps; used for --func mode or overrides t_opt)")
parser.add_argument("--eval_samples",  type=int,   default=20,
                    help="Trials per function during pre-eval (only with --hard_threshold)")

# Checkpoints
parser.add_argument("--pretrain_ck",   default="checkpoints/transnet_pretrain_v6.pt")
parser.add_argument("--out_ck",        default="checkpoints/transnet_rl_v1.pt")

# Training
parser.add_argument("--epochs",          type=int,   default=100)
parser.add_argument("--funcs_per_batch", type=int,   default=4,
                    help="Functions sampled per PPO update step")
parser.add_argument("--num_traj",        type=int,   default=4,
                    help="Trajectories per function per PPO step")
parser.add_argument("--temperature",     type=float, default=0.8)
parser.add_argument("--max_steps",       type=int,   default=30)
parser.add_argument("--lr",              type=float, default=1e-4)
parser.add_argument("--clip_eps",        type=float, default=0.2)
parser.add_argument("--lambda_entropy",  type=float, default=0.01,
                    help="Entropy bonus coefficient — prevents mode collapse "
                         "(0 = disable)")
parser.add_argument("--agent_sync_every",type=int,   default=10)
parser.add_argument("--nll_every",       type=int,   default=5,
                    help="Mix in one NLL batch every N PPO steps (0 = disable)")
parser.add_argument("--lambda_nll",      type=float, default=0.1)
parser.add_argument("--nll_batch_size",  type=int,   default=32)
parser.add_argument("--log_interval",    type=int,   default=10)
parser.add_argument("--keep_ckpts",      action="store_true",
                    help="Keep all per-interval checkpoints (default: delete after run)")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Helper: load SAT functions from CSV ───────────────────────────────────────

def _read_expr(fid, t):
    path = os.path.join(_DATA_ROOT, fid, f"t_{t}", "Booleans.txt")
    with open(path) as f:
        return f.read().strip().split(": ", 1)[1].strip()


def load_all_functions(csv_path=_CSV_PATH):
    """Return list of (fid, t_opt, vars_in_func, on_patterns, off_patterns)."""
    funcs = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["status"] != "SAT":
                continue
            fid   = row["function_id"]
            t_opt = int(row["t"])
            expr  = _read_expr(fid, row["t"])
            vif   = extract_vars(expr)
            on_p  = parse_sop_expr(expr, vif)
            off_p = on_off_split(on_p)[1]
            funcs.append((fid, t_opt, vif, on_p, off_p))
    return funcs


# ── Helper: build function list based on CLI flags ────────────────────────────

def build_func_list(args, all_funcs, task):
    t_ref_override = args.t_ref  # None means: use per-function t_opt or max_steps

    if args.func is not None:
        # Single user-specified expression
        expr = args.func
        vif  = extract_vars(expr)
        on_p = parse_sop_expr(expr, vif)
        off_p = on_off_split(on_p)[1]
        fid   = f"user:{expr[:40]}"   # synthetic ID for baseline tracking
        funcs = [(fid, 0, vif, on_p, off_p)]   # t_opt unused (reward = 1/T)
        print(f"  Mode: single expression")
        return funcs

    if args.func_ids is not None:
        # Subset by dataset function_id
        id_set = set(args.func_ids)
        funcs  = [f for f in all_funcs if f[0] in id_set]
        if t_ref_override is not None:
            funcs = [(fid, t_ref_override, vif, on_p, off_p)
                     for fid, _, vif, on_p, off_p in funcs]
        missing = id_set - {f[0] for f in funcs}
        if missing:
            print(f"  WARNING: function_ids not found in dataset: {missing}")
        print(f"  Mode: func_ids={args.func_ids}  ({len(funcs)} functions)")
        return funcs

    if args.hard_threshold is not None:
        # Pre-evaluate and keep hard functions
        print(f"  Pre-evaluating {len(all_funcs)} functions "
              f"(n={args.eval_samples}, T={args.temperature}) …")
        t0 = time.time()
        rates = _pre_evaluate(task, all_funcs, args.eval_samples,
                              args.temperature, args.max_steps)
        funcs = [f for f in all_funcs if rates[f[0]] < args.hard_threshold]
        if t_ref_override is not None:
            funcs = [(fid, t_ref_override, vif, on_p, off_p)
                     for fid, _, vif, on_p, off_p in funcs]
        easy_n = len(all_funcs) - len(funcs)
        print(f"  Done in {time.time()-t0:.1f}s  "
              f"hard={len(funcs)} easy={easy_n} (threshold={args.hard_threshold})")
        if not funcs:
            print("  No hard functions found; lower --hard_threshold.")
            raise SystemExit(0)
        return funcs

    # Default: all SAT functions
    funcs = all_funcs
    if t_ref_override is not None:
        funcs = [(fid, t_ref_override, vif, on_p, off_p)
                 for fid, _, vif, on_p, off_p in funcs]
    print(f"  Mode: all SAT functions  ({len(funcs)} functions)")
    return funcs


def _pre_evaluate(task, funcs, num_sample, temperature, max_steps):
    rates = {}
    task.eval()
    with torch.no_grad():
        for fid, t_opt, vif, on_p, off_p in funcs:
            results = task.generate(
                vif, on_p, off_p,
                num_sample=num_sample, max_steps=max_steps,
                temperature=temperature, verbose=0,
            )
            rates[fid] = sum(r["correct"] for r in results) / num_sample
    task.train()
    return rates


# ── Build model from checkpoint ────────────────────────────────────────────────

def build_model_from_checkpoint(ck_path):
    ck = torch.load(ck_path, map_location="cpu")
    # arch_args: dedicated key written by this script; falls back to pretrain "args"
    a  = ck.get("arch_args") or ck.get("args", {})
    mpnn = models.MPNN(
        input_dim      = NODE_FEAT_DIM,
        hidden_dim     = a.get("hidden",     64),
        edge_input_dim = EDGE_FEAT_DIM,
        num_layer      = a.get("num_layer",  3),
        batch_norm     = False,
    )
    task = GCPNTransNet(mpnn, hidden_dim_mlp=a.get("mlp_hidden", 128))
    task.load_state_dict(ck["model_state"])
    return task, ck, a   # also return arch_args for re-saving


# ── Main ───────────────────────────────────────────────────────────────────────

print("Building model from checkpoint …")
task, _, arch_args = build_model_from_checkpoint(args.pretrain_ck)
task = task.to(device)
task.sync_agent()
print(f"  Loaded: {args.pretrain_ck}")

print("Loading function data …")
all_funcs = load_all_functions()
print(f"  {len(all_funcs)} SAT functions in dataset")

print("Selecting functions for RL …")
func_list = build_func_list(args, all_funcs, task)

# NLL DataLoader for regularization (skipped if --nll_every 0 or --func mode)
use_nll = (args.nll_every > 0) and (args.func is None)
nll_loader_it = None
if use_nll:
    nll_dataset  = TransistorDataset()
    nll_loader_it = iter(td_data.DataLoader(
        nll_dataset, batch_size=args.nll_batch_size, shuffle=True
    ))

def next_nll_batch():
    global nll_loader_it
    try:
        return next(nll_loader_it)
    except StopIteration:
        nll_loader_it = iter(td_data.DataLoader(
            nll_dataset, batch_size=args.nll_batch_size, shuffle=True
        ))
        return next(nll_loader_it)


optimizer = torch.optim.Adam(task.parameters(), lr=args.lr)
out_dir = os.path.dirname(args.out_ck) or "."
os.makedirs(out_dir, exist_ok=True)

# Intermediate checkpoint stem: e.g. "checkpoints/transnet_rl_v1" (no .pt)
_stem = args.out_ck[:-3] if args.out_ck.endswith(".pt") else args.out_ck

ppo_step = 0
interval_ckpts: list[str] = []          # paths of saved per-interval checkpoints
best_success  = -1.0
best_avg_t    = float("inf")
best_ck_path  = None

print(f"\nStarting RL finetuning: {len(func_list)} function(s), {args.epochs} epochs\n")
task.train()

for epoch in range(1, args.epochs + 1):
    random.shuffle(func_list)
    epoch_ppo_losses  = []
    epoch_rewards_all = []

    for i in range(0, max(len(func_list), 1), args.funcs_per_batch):
        func_batch = func_list[i : i + args.funcs_per_batch]
        if not func_batch:
            continue
        # reinforce_forward expects: (fid, vars_in_func, on_patterns, off_patterns, t_opt)
        rl_input = [(fid, vif, on_p, off_p, t_opt)
                    for fid, t_opt, vif, on_p, off_p in func_batch]

        optimizer.zero_grad()

        ppo_loss = task.reinforce_forward(
            rl_input,
            num_traj       = args.num_traj,
            max_steps      = args.max_steps,
            temperature    = args.temperature,
            clip_eps       = args.clip_eps,
            lambda_entropy = args.lambda_entropy,
        )

        if ppo_loss is None:
            ppo_step += 1
            continue

        total_loss = ppo_loss

        if use_nll and ppo_step % args.nll_every == 0:
            b     = next_nll_batch()
            graph = b["graph"].to(device)
            lbls  = {k: v.to(device) for k, v in b.items() if k != "graph"}
            lbls["graph"] = graph
            nll_loss, _ = task.MLE_forward(lbls)
            total_loss  = total_loss + args.lambda_nll * nll_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(task.parameters(), 1.0)
        optimizer.step()

        epoch_ppo_losses.append(ppo_loss.item())
        ppo_step += 1

        if ppo_step % args.agent_sync_every == 0:
            task.sync_agent()

    if not epoch_ppo_losses:
        continue

    avg_ppo = sum(epoch_ppo_losses) / len(epoch_ppo_losses)

    if epoch % args.log_interval == 0 or epoch == 1:
        # Eval
        task.eval()
        n_correct = n_total = 0
        t_counts = []
        with torch.no_grad():
            for fid, t_opt, vif, on_p, off_p in func_list:
                results = task.generate(
                    vif, on_p, off_p,
                    num_sample=20, max_steps=args.max_steps,
                    temperature=args.temperature, verbose=0,
                )
                n_correct += sum(r["correct"] for r in results)
                n_total   += 20
                t_counts  += [len(r["g_tran"]) for r in results if r["correct"]]
        success = n_correct / n_total
        avg_t   = sum(t_counts) / len(t_counts) if t_counts else float("nan")
        task.train()

        # Save interval checkpoint
        ck_path = f"{_stem}_epoch{epoch:04d}.pt"
        ck_data = {
            "epoch":           epoch,
            "model_state":     task.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "ppo_loss":        avg_ppo,
            "success":         success,
            "avg_t":           avg_t,
            "arch_args":       arch_args,
            "rl_args":         vars(args),
            "pretrain_ck":     args.pretrain_ck,
        }
        torch.save(ck_data, ck_path)
        interval_ckpts.append(ck_path)

        # Track best: higher success wins; break ties with lower avg_t
        _t = avg_t if not math.isnan(avg_t) else float("inf")
        is_best = (success > best_success or
                   (success == best_success and _t < best_avg_t))
        marker = " ← best" if is_best else ""
        if is_best:
            best_success = success
            best_avg_t   = _t
            best_ck_path = ck_path

        # best_T discovered by RL (self-tightening target)
        best_ts = [v for v in task._best_t.values() if v is not None]
        best_t_str = f"  best_T={min(best_ts)}" if best_ts else ""

        print(f"epoch={epoch:4d}  ppo_loss={avg_ppo:.4f}  "
              f"success={success:.2f}  avg_t={avg_t:.1f}  "
              f"ppo_steps={ppo_step}{best_t_str}{marker}")

# ── After training: copy best checkpoint to out_ck ───────────────────────────
if best_ck_path is not None:
    shutil.copy(best_ck_path, args.out_ck)
    print(f"\nBest checkpoint: epoch {best_success:.0%} success, "
          f"avg_t={best_avg_t:.1f}  →  {args.out_ck}")
else:
    print("\nNo eval checkpoint recorded.")

if not args.keep_ckpts:
    for p in interval_ckpts:
        if p != args.out_ck:
            os.remove(p)
else:
    print(f"Interval checkpoints kept: {interval_ckpts}")

print(f"Done. Final checkpoint: {args.out_ck}")
