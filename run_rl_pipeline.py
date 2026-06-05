"""
run_rl_pipeline.py

Evaluates transnet_pretrain_v6.pt on every 3-input SAT function.
A function NEEDS fine-tuning if either:
  (A) success_rate < 0.50  across EVAL_TRIALS generation trials, OR
  (B) no trial found a transistor network with the minimum known count
      (t_opt = min SAT t in aggregate.csv).

Functions that pass both criteria are skipped.
Failing functions are fine-tuned one by one via finetune_rl_transnet.py.
Fine-tuned checkpoints are saved to checkpoints/rl_{fid}.pt.
A completed-function log (pipeline_done.txt) makes the run resumable.

Usage:
  python run_rl_pipeline.py                  # full run
  python run_rl_pipeline.py --dry_run        # show which functions fail, no training
  python run_rl_pipeline.py --eval_only      # evaluate and report, skip training
  python run_rl_pipeline.py --skip_eval      # load cached eval, train failing only
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from collections import defaultdict

import torch
from torchdrug import models

from transnet.graph import NODE_FEAT_DIM
from transnet.literal import EDGE_FEAT_DIM, extract_vars, on_off_split, parse_sop_expr
from transnet.task import GCPNTransNet

# ── Config ────────────────────────────────────────────────────────────────────
_DATA_ROOT   = os.path.join(os.path.dirname(__file__), "dataset", "sweep_3input")
_CSV_PATH    = os.path.join(_DATA_ROOT, "aggregate.csv")
_PRETRAIN_CK = "checkpoints/transnet_pretrain_v6.pt"
_CK_DIR      = "checkpoints"
_DONE_LOG    = "pipeline_done.txt"       # fids that have been fine-tuned
_EVAL_CACHE  = "pipeline_eval_cache.json"  # cached evaluation results

# Evaluation parameters (from user spec)
EVAL_TRIALS = 50
EVAL_TEMP   = 0.8
EVAL_STEPS  = 20

# Pass criteria
SUCCESS_THRESH = 0.50   # success rate must be >= this

# Fine-tuning parameters (from user spec)
FT_EPOCHS   = 300
FT_NUM_TRAJ = 8
FT_TEMP     = 0.8
FT_STEPS    = 20
FT_LR       = 3e-5
FT_LOG_INT  = 10

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--dry_run",   action="store_true",
                    help="Show failing functions but do not fine-tune")
parser.add_argument("--eval_only", action="store_true",
                    help="Evaluate all functions and report, skip fine-tuning")
parser.add_argument("--skip_eval", action="store_true",
                    help="Load cached evaluation results, skip re-evaluation")
parser.add_argument("--pretrain_ck", default=_PRETRAIN_CK)
args = parser.parse_args()

os.makedirs(_CK_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Load aggregate.csv ────────────────────────────────────────────────────────

def load_optimal_t(csv_path=_CSV_PATH):
    """
    For each function_id, find the minimum t among all SAT rows.
    Returns dict: {fid: t_opt}.
    """
    best = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["status"] != "SAT":
                continue
            fid = row["function_id"]
            t   = int(row["t"])
            if fid not in best or t < best[fid]:
                best[fid] = t
    return best   # {fid: t_opt}


def read_expr(fid, t):
    path = os.path.join(_DATA_ROOT, fid, f"t_{t}", "Booleans.txt")
    with open(path) as f:
        return f.read().strip().split(": ", 1)[1].strip()


# ── Load model ────────────────────────────────────────────────────────────────

def load_model(ck_path):
    ck  = torch.load(ck_path, map_location="cpu")
    a   = ck.get("arch_args") or ck.get("args", {})
    mpnn = models.MPNN(
        input_dim      = NODE_FEAT_DIM,
        hidden_dim     = a.get("hidden",    64),
        edge_input_dim = EDGE_FEAT_DIM,
        num_layer      = a.get("num_layer", 3),
        batch_norm     = False,
    )
    task = GCPNTransNet(mpnn, hidden_dim_mlp=a.get("mlp_hidden", 128))
    task.load_state_dict(ck["model_state"], strict=False)
    return task.to(device).eval()


# ── Evaluate one function ─────────────────────────────────────────────────────

def evaluate_function(task, fid, t_opt):
    """
    Generate EVAL_TRIALS networks. Returns dict:
      success_rate : float  (fraction of correct trials)
      best_t       : int    (fewest transistors in any correct trial, or None)
      found_opt    : bool   (any trial achieved exactly t_opt transistors)
    """
    expr = read_expr(fid, t_opt)
    vif  = extract_vars(expr)
    on_p = parse_sop_expr(expr, vif)
    on_p, off_p = on_off_split(on_p)

    with torch.no_grad():
        results = task.generate(
            vif, on_p, off_p,
            num_sample  = EVAL_TRIALS,
            max_steps   = EVAL_STEPS,
            temperature = EVAL_TEMP,
            verbose     = 0,
        )

    n_correct = sum(r["correct"] for r in results)
    t_counts  = [len(r["g_tran"]) for r in results if r["correct"]]
    best_t    = min(t_counts) if t_counts else None
    found_opt = (best_t is not None and best_t <= t_opt)

    return {
        "success_rate": n_correct / EVAL_TRIALS,
        "best_t":       best_t,
        "t_opt":        t_opt,
        "found_opt":    found_opt,
    }


def needs_finetuning(ev):
    """Return True if the function fails either criterion."""
    return ev["success_rate"] < SUCCESS_THRESH or not ev["found_opt"]


# ── Fine-tune one function ────────────────────────────────────────────────────

def finetune(fid, t_opt):
    """Call finetune_rl_transnet.py for a single function, return True on success."""
    out_ck = os.path.join(_CK_DIR, f"rl_{fid}.pt")
    cmd = [
        sys.executable, "finetune_rl_transnet.py",
        "--func_ids",       fid,
        "--pretrain_ck",    args.pretrain_ck,
        "--out_ck",         out_ck,
        "--epochs",         str(FT_EPOCHS),
        "--num_traj",       str(FT_NUM_TRAJ),
        "--temperature",    str(FT_TEMP),
        "--max_steps",      str(FT_STEPS),
        "--lr",             str(FT_LR),
        "--log_interval",   str(FT_LOG_INT),
        "--nll_every",      "0",
    ]
    print(f"  → {' '.join(cmd)}")
    ret = subprocess.run(cmd)
    return ret.returncode == 0


# ── Load/save done log ────────────────────────────────────────────────────────

def load_done():
    if os.path.exists(_DONE_LOG):
        with open(_DONE_LOG) as f:
            return set(l.strip() for l in f if l.strip())
    return set()


def mark_done(fid):
    with open(_DONE_LOG, "a") as f:
        f.write(fid + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

print("Loading optimal transistor counts from aggregate.csv …")
opt_t = load_optimal_t()
print(f"  {len(opt_t)} SAT functions  "
      f"(t_opt range: {min(opt_t.values())}–{max(opt_t.values())})")

# ── Evaluation phase ──────────────────────────────────────────────────────────
if args.skip_eval and os.path.exists(_EVAL_CACHE):
    print(f"\nLoading cached evaluation from {_EVAL_CACHE} …")
    with open(_EVAL_CACHE) as f:
        eval_results = json.load(f)
    print(f"  {len(eval_results)} functions loaded from cache")
else:
    print(f"\nLoading pretrain model: {args.pretrain_ck} …")
    task = load_model(args.pretrain_ck)
    print(f"  Evaluating {len(opt_t)} functions "
          f"({EVAL_TRIALS} trials each, T={EVAL_TEMP}, max_steps={EVAL_STEPS}) …\n")

    eval_results = {}
    t0 = time.time()
    for i, (fid, t_opt) in enumerate(sorted(opt_t.items()), 1):
        ev = evaluate_function(task, fid, t_opt)
        eval_results[fid] = ev
        status = "PASS" if not needs_finetuning(ev) else "FAIL"
        flag_a = f"succ={ev['success_rate']:.0%}"
        flag_b = f"best_t={ev['best_t']} opt={t_opt} {'✓' if ev['found_opt'] else '✗'}"
        print(f"  [{i:3d}/{len(opt_t)}] {fid:<8} {status}  {flag_a}  {flag_b}")

    elapsed = time.time() - t0
    print(f"\nEvaluation done in {elapsed:.1f}s")

    # Cache results for resumability
    with open(_EVAL_CACHE, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"Evaluation results saved to {_EVAL_CACHE}")

# ── Summarise ─────────────────────────────────────────────────────────────────
fail_low_succ = [fid for fid, ev in eval_results.items()
                 if ev["success_rate"] < SUCCESS_THRESH]
fail_no_opt   = [fid for fid, ev in eval_results.items()
                 if not ev["found_opt"]]
fail_any      = sorted(set(fail_low_succ) | set(fail_no_opt))

print(f"\n{'='*55}")
print(f"Evaluation summary ({len(eval_results)} functions):")
print(f"  Pass (both criteria)  : {len(eval_results) - len(fail_any)}")
print(f"  Fail (low success <50%): {len(fail_low_succ)}")
print(f"  Fail (no opt found)   : {len(fail_no_opt)}")
print(f"  Fail (either)         : {len(fail_any)}  → need fine-tuning")
print(f"{'='*55}\n")

if args.eval_only or args.dry_run:
    print("Failing functions:")
    for fid in fail_any:
        ev = eval_results[fid]
        reasons = []
        if ev["success_rate"] < SUCCESS_THRESH:
            reasons.append(f"succ={ev['success_rate']:.0%}<50%")
        if not ev["found_opt"]:
            reasons.append(f"best_t={ev['best_t']} > t_opt={ev['t_opt']}")
        print(f"  {fid:<8}  {', '.join(reasons)}")
    if args.dry_run:
        print("\nDry run — no fine-tuning performed.")
    sys.exit(0)

# ── Fine-tuning phase ─────────────────────────────────────────────────────────
done = load_done()
todo = [fid for fid in fail_any if fid not in done]
print(f"Fine-tuning {len(todo)} functions "
      f"({len(fail_any) - len(todo)} already done, skipped) …\n")

for idx, fid in enumerate(todo, 1):
    t_opt = opt_t[fid]
    ev    = eval_results[fid]
    print(f"[{idx}/{len(todo)}] Fine-tuning {fid}  "
          f"(t_opt={t_opt}, succ={ev['success_rate']:.0%}, "
          f"best_t={ev['best_t']})")

    ok = finetune(fid, t_opt)
    if ok:
        mark_done(fid)
        print(f"  ✓ {fid} done  → checkpoints/rl_{fid}.pt\n")
    else:
        print(f"  ✗ {fid} FAILED (non-zero exit code)\n")

print("\nPipeline complete.")
print(f"Fine-tuned checkpoints in: {_CK_DIR}/rl_*.pt")
