"""
physical_finetune_rl_transnet.py

Phase 3: RL finetuning of joint PDN+PUN synthesis optimized for physical
placement quality (ASTRAN CPP metric).

For a Boolean function f(x):
  PDN switching function : !f(x)      → on-patterns = off-patterns of f
  PUN switching function : f(!x)      → on-patterns = complement(on-patterns of f)

Both networks are synthesized by the same TransSyn policy, conditioned on
their respective on/off pattern sets. ASTRAN evaluates the combined cell
and returns Width / Nr. Gaps (CPP). A joint reward drives both trajectories.

Usage:
  # Single XOR function
  python physical_finetune_rl_transnet.py \\
      --func '!a*!b*c+!a*b*!c+a*!b*!c+a*b*c' \\
      --pretrain_ck checkpoints/transnet_pretrain_v6.pt \\
      --out_ck checkpoints/transnet_physical_v1.pt

  # All SAT functions in the dataset
  python physical_finetune_rl_transnet.py
"""
import argparse
import csv
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
import random
from concurrent.futures import ThreadPoolExecutor

import torch
from torchdrug import models

from transnet.graph import NODE_FEAT_DIM
from transnet.literal import (
    EDGE_FEAT_DIM, extract_vars, on_off_split, parse_sop_expr,
    covered_patterns,
)
from transnet.task import GCPNTransNet

_DATA_ROOT  = os.path.join(os.path.dirname(__file__), "dataset", "sweep_3input")
_CSV_PATH   = os.path.join(_DATA_ROOT, "aggregate.csv")
_ASTRAN_BIN = os.path.join(os.path.dirname(__file__),
              "astran/Astran/build/bin/Astran.app/Contents/MacOS/Astran")
_TECH_FILE  = os.path.join(os.path.dirname(__file__),
              "astran/Astran/build/Work/tech_freePDK45.rul")


# ── CLI ────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                 description=__doc__)

grp = parser.add_mutually_exclusive_group()
grp.add_argument("--func",     type=str, default=None, metavar="EXPR")
grp.add_argument("--func_ids", type=str, nargs="+", default=None, metavar="ID")

parser.add_argument("--pretrain_ck",    default="checkpoints/transnet_pretrain_v6.pt")
parser.add_argument("--out_ck",         default="checkpoints/transnet_physical_v1.pt")

# ASTRAN
parser.add_argument("--astran_bin",     default=_ASTRAN_BIN)
parser.add_argument("--tech_file",      default=_TECH_FILE)
parser.add_argument("--astran_timeout", type=int,   default=30,
                    help="Max seconds per ASTRAN call")
parser.add_argument("--astran_workers", type=int,   default=4,
                    help="Parallel ASTRAN processes per PPO step")

# Training
parser.add_argument("--epochs",          type=int,   default=200)
parser.add_argument("--funcs_per_batch", type=int,   default=2)
parser.add_argument("--num_traj",        type=int,   default=4,
                    help="PDN+PUN pairs per function per PPO step")
parser.add_argument("--temperature",     type=float, default=0.8)
parser.add_argument("--max_steps",       type=int,   default=20)
parser.add_argument("--lr",              type=float, default=3e-5)
parser.add_argument("--clip_eps",        type=float, default=0.2)
parser.add_argument("--lambda_entropy",  type=float, default=0.01)
parser.add_argument("--agent_sync_every",type=int,   default=5)
parser.add_argument("--log_interval",    type=int,   default=10)
parser.add_argument("--keep_ckpts",      action="store_true")

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Function derivation ────────────────────────────────────────────────────────

def derive_pdn_patterns(on_p, off_p):
    """PDN implements !f(x): conducts when f(x)=0. Swap on and off."""
    return off_p, on_p          # pdn_on=off_p, pdn_off=on_p


def derive_pun_patterns(on_p, off_p):
    """PUN implements f(!x): conducts when f(complement(x))=1."""
    comp = lambda pat: tuple(1 - b for b in pat)
    pun_on  = frozenset(comp(p) for p in on_p)
    pun_off = frozenset(comp(p) for p in off_p)
    return pun_on, pun_off


# ── SPICE netlist generator ────────────────────────────────────────────────────

_W_N = "0.090000U"   # minimum NMOS width  (freePDK45, from library45.sp)
_W_P = "0.090000U"   # minimum PMOS width
_L   = "0.050000U"   # gate length = 50 nm

def _node(n, prefix, src_net, snk_net):
    if n == 0: return src_net
    if n == 1: return snk_net
    return f"{prefix}{n}"

def _gate(var_idx, is_neg, vif):
    return vif[var_idx] if is_neg == 0 else f"{vif[var_idx]}_N"

def build_spice(cell_name, vif,
                pdn_tran, pdn_gnum,
                pun_tran, pun_gnum):
    """
    Build a combined SPICE netlist for one CMOS cell.

    PDN (NMOS_VTL): SRC=GND, SNK=ZN, internal nodes nd{k}
    PUN (PMOS_VTL): SRC=VCC, SNK=ZN, internal nodes np{k}
    Gate signals  : var_name (is_neg=0) or var_name_N (is_neg=1)
    Power nets    : VCC / GND  (ASTRAN default names — do NOT use VDD/VSS)
    """
    ports = " ".join(vif) + " " + " ".join(v + "_N" for v in vif) + " ZN VCC GND"
    lines = [f".SUBCKT {cell_name} {ports}",
             "* PDN — NMOS_VTL — implements !f(x)"]
    for i, (u, v, vi, neg) in enumerate(pdn_tran):
        d = _node(v, "nd", "GND", "ZN")
        s = _node(u, "nd", "GND", "ZN")
        g = _gate(vi, neg, vif)
        lines.append(f"MN{i:<2} {d:<6} {g:<5} {s:<6} GND  NMOS_VTL  W={_W_N}  L={_L}")
    lines.append("* PUN — PMOS_VTL — implements f(!x)")
    for i, (u, v, vi, neg) in enumerate(pun_tran):
        d = _node(v, "np", "VCC", "ZN")
        s = _node(u, "np", "VCC", "ZN")
        g = _gate(vi, neg, vif)
        lines.append(f"MP{i:<2} {d:<6} {g:<5} {s:<6} VCC  PMOS_VTL  W={_W_P}  L={_L}")
    lines.append(".ENDS")
    return "\n".join(lines)


# ── ASTRAN placement wrapper ───────────────────────────────────────────────────

# Objective weights — must match the cellgen place parameters below.
# RoutingCost and RtDensityCost are set to 0 so ASTRAN only optimises
# Place weights — must match the cellgen place command below.
# These are the original balanced ASTRAN defaults.
_OBJ_CPP      = 3   # WidthCost
_OBJ_MISMATCH = 4   # GateMismatchCost
_OBJ_ROUTING  = 1   # RoutingCost (WL)
_OBJ_DENSITY  = 4   # RtDensityCost (congestion peak)
_OBJ_GAPS     = 2   # GapsCost

def astran_objective(metrics):
    """
    Weighted sum matching ASTRAN's internal SA objective.

    ASTRAN prints all five components on the Final cost line, so we can
    reproduce the exact objective used by the placer:
      obj = CPP*width + MISMATCH*gate_mismatches + ROUTING*wl
          + DENSITY*rt_density + GAPS*nr_gaps
    """
    return (
        _OBJ_CPP      * metrics["width"]
      + _OBJ_MISMATCH * metrics["gate_mismatches"]
      + _OBJ_ROUTING  * metrics["wl"]
      + _OBJ_DENSITY  * metrics["rt_density"]
      + _OBJ_GAPS     * metrics["nr_gaps"]
    )

# Parse all five fields from: "Final cost: Width=19; Gate Mismatches=0; WL=57; Rt. Density=6; Nr. Gaps=4"
_COST_RE = re.compile(
    r"Final cost: Width=(\d+).*?Gate Mismatches=(\d+).*?WL=(\d+).*?Rt\. Density=(\d+).*?Nr\. Gaps=(\d+)"
)


def run_astran(pdn_tran, pdn_gnum, pun_tran, pun_gnum, vif,
               cell_name="TRANSSYN",
               astran_bin=None, tech_file=None, timeout=30):
    """
    Write SPICE + script to temp files, call ASTRAN --shell <script>, parse stdout.

    Correct invocation: Astran --shell script.run  (not Astran script.run, which
    launches the GUI and hangs).  All placement output goes to stdout; SA progress
    dots go to stderr (discarded).

    Returns dict with keys: width, gate_mismatches, wl, rt_density, nr_gaps
    Returns None on failure or timeout.
    """
    astran_bin = astran_bin or args.astran_bin
    tech_file  = tech_file  or args.tech_file

    spice_text = build_spice(cell_name, vif, pdn_tran, pdn_gnum, pun_tran, pun_gnum)

    sp_path = sc_path = None
    try:
        # Write SPICE netlist
        sp_fd, sp_path = tempfile.mkstemp(suffix=".sp", prefix="transsyn_")
        os.write(sp_fd, spice_text.encode())
        os.close(sp_fd)

        # Build script after sp_path is known, write it.
        # cellgen place <Saquality> <#Attempts>
        #   <WidthCost> <GateMismatchCost> <RoutingCost> <RtDensityCost> <GapsCost>
        # Weights match _OBJ_* constants so the RL reward = ASTRAN's own objective.
        script_text = (
            f'load technology "{tech_file}"\n'
            f'load netlist "{sp_path}"\n'
            f'cellgen select {cell_name}\n'
            f'cellgen fold 2 0\n'
            f'cellgen place 1 1 {_OBJ_CPP} {_OBJ_MISMATCH} {_OBJ_ROUTING} {_OBJ_DENSITY} {_OBJ_GAPS}\n'
        )
        sc_fd, sc_path = tempfile.mkstemp(suffix=".run", prefix="transsyn_")
        os.write(sc_fd, script_text.encode())
        os.close(sc_fd)

        # Run: Astran --shell script.run  (shell+script = batch mode, exits when done)
        result = subprocess.run(
            [astran_bin, "--shell", sc_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,   # SA progress dots → discard
            timeout=timeout,
        )
        output = result.stdout.decode(errors="replace")
        m = _COST_RE.search(output)
        if m:
            return {
                "width":           int(m.group(1)),
                "gate_mismatches": int(m.group(2)),
                "wl":              int(m.group(3)),
                "rt_density":      int(m.group(4)),
                "nr_gaps":         int(m.group(5)),
            }
        return None

    except (subprocess.TimeoutExpired, Exception):
        return None

    finally:
        for p in (sp_path, sc_path):
            if p and os.path.exists(p):
                try:
                    os.unlink(p)
                except OSError:
                    pass


# ── Reward ─────────────────────────────────────────────────────────────────────

def physical_reward(metrics, best_obj, fid):
    """
    Adaptive self-tightening reward using ASTRAN's weighted placement objective.

      objective = _OBJ_CPP * width + _OBJ_MISMATCH * gate_mismatches + _OBJ_GAPS * nr_gaps

      reward = best_obj[fid] / objective
               1.0 when this pair matches the best objective ever seen for fid,
               < 1.0 for worse (higher cost) placements.

    As training discovers lower-cost placements, best_obj[fid] tightens and
    subsequent worse solutions receive a lower reward.
    """
    obj = astran_objective(metrics)
    if obj == 0:
        obj = 1   # guard against degenerate zero-cost case

    if fid not in best_obj or obj < best_obj[fid]:
        best_obj[fid] = obj

    return best_obj[fid] / obj


# ── Physical PPO forward ───────────────────────────────────────────────────────

def physical_reinforce_forward(
    task,
    func_list,        # [(fid, vif, pdn_on, pdn_off, pun_on, pun_off), ...]
    best_obj,         # dict fid → best ASTRAN objective found
    num_traj    = 4,
    max_steps   = 20,
    temperature = 0.8,
    clip_eps    = 0.2,
    lambda_entropy = 0.01,
    astran_workers = 4,
):
    """
    Off-policy PPO with joint PDN+PUN ASTRAN reward and GRPO baseline.

    Returns (loss, stats) where stats is a dict of lists:
      {"cpp": [...], "mismatch": [...], "gaps": [...], "obj": [...]}
    collected from all complete pairs evaluated this step.
    Returns (None, stats) when no gradient step was taken.
    """
    all_steps:      list = []
    all_advantages: list = []
    stats = {"cpp": [], "mismatch": [], "wl": [], "density": [], "gaps": [], "obj": []}

    # ── Trajectory collection (no_grad) ───────────────────────────────────────
    pending_evals = []   # items to send to ASTRAN

    with torch.no_grad():
        for fid, vif, pdn_on, pdn_off, pun_on, pun_off in func_list:
            group = []   # (pdn_steps, pun_steps, pdn_tran, pdn_gnum, pun_tran, pun_gnum, T_pdn, T_pun, complete)

            for _ in range(num_traj):
                pdn_tran, pdn_gnum, _, pdn_steps = task._generate_traj(
                    vif, pdn_on, pdn_off, max_steps, temperature
                )
                pun_tran, pun_gnum, _, pun_steps = task._generate_traj(
                    vif, pun_on, pun_off, max_steps, temperature
                )

                if not pdn_steps or not pun_steps:
                    group.append(None)
                    continue

                pdn_complete = (covered_patterns(pdn_tran, pdn_gnum, pdn_on) == pdn_on)
                pun_complete = (covered_patterns(pun_tran, pun_gnum, pun_on) == pun_on)
                complete = pdn_complete and pun_complete

                group.append({
                    "fid":       fid,
                    "vif":       vif,
                    "pdn_steps": pdn_steps,
                    "pun_steps": pun_steps,
                    "pdn_tran":  pdn_tran,
                    "pdn_gnum":  pdn_gnum,
                    "pun_tran":  pun_tran,
                    "pun_gnum":  pun_gnum,
                    "T_pdn":     len(pdn_steps),
                    "T_pun":     len(pun_steps),
                    "complete":  complete,
                    "metrics":   None,   # filled after ASTRAN
                })

            pending_evals.append((fid, vif, group))

    # ── ASTRAN evaluation (parallel) ──────────────────────────────────────────
    def _eval_item(item):
        """Call ASTRAN for one complete pair; mutates item['metrics']."""
        if not item["complete"]:
            return
        metrics = run_astran(
            item["pdn_tran"], item["pdn_gnum"],
            item["pun_tran"], item["pun_gnum"],
            item["vif"],
        )
        item["metrics"] = metrics

    eval_items = [
        item
        for _, _, group in pending_evals
        for item in group
        if item is not None and item["complete"]
    ]
    with ThreadPoolExecutor(max_workers=astran_workers) as pool:
        list(pool.map(_eval_item, eval_items))

    # ── Compute rewards, GRPO, collect steps ──────────────────────────────────
    for fid, vif, group in pending_evals:
        # Compute scalar reward for each trajectory pair
        rewards = []
        for item in group:
            if item is None or not item["complete"] or item["metrics"] is None:
                rewards.append(0.0)
            else:
                r = physical_reward(item["metrics"], best_obj, fid)
                rewards.append(r)
                # accumulate per-pair metrics for logging
                m = item["metrics"]
                stats["cpp"].append(m["width"])
                stats["mismatch"].append(m["gate_mismatches"])
                stats["wl"].append(m["wl"])
                stats["density"].append(m["rt_density"])
                stats["gaps"].append(m["nr_gaps"])
                stats["obj"].append(astran_objective(m))

        # GRPO: normalize within this function's group
        mean_r = sum(rewards) / len(rewards)
        sq     = sum((r - mean_r) ** 2 for r in rewards)
        std_r  = (sq / len(rewards)) ** 0.5

        for item, r in zip(group, rewards):
            if item is None:
                continue
            adv = (r - mean_r) / (std_r + 1e-8)
            steps = item["pdn_steps"] + item["pun_steps"]
            all_steps.extend(steps)
            all_advantages.extend([adv] * len(steps))

    if not all_steps:
        return None, stats

    # Global normalization across all functions in the batch
    adv_t = torch.tensor(all_advantages, dtype=torch.float)
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
    all_advantages = adv_t.tolist()

    return task._ppo_loss_batched(all_steps, all_advantages, clip_eps, lambda_entropy), stats


# ── Data loading ───────────────────────────────────────────────────────────────

def _read_expr(fid, t):
    path = os.path.join(_DATA_ROOT, fid, f"t_{t}", "Booleans.txt")
    with open(path) as f:
        return f.read().strip().split(": ", 1)[1].strip()


def load_all_functions(csv_path=_CSV_PATH):
    funcs = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["status"] != "SAT":
                continue
            fid   = row["function_id"]
            expr  = _read_expr(fid, row["t"])
            vif   = extract_vars(expr)
            on_p  = parse_sop_expr(expr, vif)
            on_p, off_p = on_off_split(on_p)
            funcs.append((fid, vif, on_p, off_p))
    return funcs


def build_func_list(args, all_funcs):
    if args.func is not None:
        expr  = args.func
        vif   = extract_vars(expr)
        on_p  = parse_sop_expr(expr, vif)
        on_p, off_p = on_off_split(on_p)
        fid   = f"user:{expr[:40]}"
        return [(fid, vif, on_p, off_p)]

    if args.func_ids is not None:
        id_set = set(args.func_ids)
        funcs  = [f for f in all_funcs if f[0] in id_set]
        missing = id_set - {f[0] for f in funcs}
        if missing:
            print(f"  WARNING: not found: {missing}")
        return funcs

    return all_funcs


# ── Model loading ──────────────────────────────────────────────────────────────

def build_model_from_checkpoint(ck_path):
    ck  = torch.load(ck_path, map_location="cpu")
    a   = ck.get("arch_args") or ck.get("args", {})
    mpnn = models.MPNN(
        input_dim      = NODE_FEAT_DIM,
        hidden_dim     = a.get("hidden",     64),
        edge_input_dim = EDGE_FEAT_DIM,
        num_layer      = a.get("num_layer",  3),
        batch_norm     = False,
    )
    task = GCPNTransNet(mpnn, hidden_dim_mlp=a.get("mlp_hidden", 128))
    task.load_state_dict(ck["model_state"], strict=False)
    return task, a


# ── Eval helper ────────────────────────────────────────────────────────────────

def evaluate(task, func_list_with_pdnpun, n_trials=10, max_steps=20, temperature=0.8):
    """
    For each function, generate n_trials PDN trajectories and n_trials PUN
    trajectories. Report per-network success rate and average transistor count.
    Does NOT call ASTRAN (fast).
    """
    task.eval()
    pdn_ok = pun_ok = n_total = 0
    pdn_ts = []
    pun_ts = []

    with torch.no_grad():
        for fid, vif, pdn_on, pdn_off, pun_on, pun_off in func_list_with_pdnpun:
            for _ in range(n_trials):
                pdn_tran, pdn_gnum, _, _ = task._generate_traj(
                    vif, pdn_on, pdn_off, max_steps, temperature)
                pun_tran, pun_gnum, _, _ = task._generate_traj(
                    vif, pun_on, pun_off, max_steps, temperature)

                pc = covered_patterns(pdn_tran, pdn_gnum, pdn_on) == pdn_on
                uc = covered_patterns(pun_tran, pun_gnum, pun_on) == pun_on
                pdn_ok += int(pc)
                pun_ok += int(uc)
                n_total += 1
                if pc: pdn_ts.append(len(pdn_tran))
                if uc: pun_ts.append(len(pun_tran))

    task.train()
    return {
        "pdn_success": pdn_ok / n_total,
        "pun_success": pun_ok / n_total,
        "pdn_avg_t":  sum(pdn_ts) / len(pdn_ts) if pdn_ts else float("nan"),
        "pun_avg_t":  sum(pun_ts) / len(pun_ts) if pun_ts else float("nan"),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

print("Building model from pretrain checkpoint …")
task, arch_args = build_model_from_checkpoint(args.pretrain_ck)
task = task.to(device)
task.sync_agent()
print(f"  Loaded: {args.pretrain_ck}")

print("Loading function data …")
all_funcs = load_all_functions()
print(f"  {len(all_funcs)} SAT functions in dataset")

print("Selecting functions …")
base_funcs = build_func_list(args, all_funcs)  # [(fid, vif, on_p, off_p), ...]

# Build PDN/PUN pattern sets for each function
func_list = []
for fid, vif, on_p, off_p in base_funcs:
    pdn_on, pdn_off = derive_pdn_patterns(on_p, off_p)
    pun_on, pun_off = derive_pun_patterns(on_p, off_p)
    func_list.append((fid, vif, pdn_on, pdn_off, pun_on, pun_off))

print(f"  {len(func_list)} function(s) selected")
print(f"  PDN: !f(x)   PUN: f(!x)")

optimizer  = torch.optim.Adam(task.parameters(), lr=args.lr)
out_dir    = os.path.dirname(args.out_ck) or "."
os.makedirs(out_dir, exist_ok=True)
_stem      = args.out_ck[:-3] if args.out_ck.endswith(".pt") else args.out_ck

# Adaptive tracking dict: best ASTRAN objective per function
best_obj = {}   # fid → best (lowest) weighted placement objective

# Per-epoch ASTRAN metric accumulators (reset each log interval)
_epoch_cpp      = []   # Width (CPP)
_epoch_mismatch = []   # Gate Mismatches
_epoch_wl       = []   # WL (wire length)
_epoch_density  = []   # Rt. Density (congestion peak)
_epoch_gaps     = []   # Nr. Gaps
_epoch_obj      = []   # astran_objective values

ppo_step       = 0
interval_ckpts = []
best_score     = -1.0   # higher pdn_success + pun_success = better
best_ck_path   = None

print(f"\nStarting physical RL: {len(func_list)} function(s), {args.epochs} epochs\n")
print(f"Objective = {_OBJ_CPP}*CPP + {_OBJ_MISMATCH}*Mismatch + "
      f"{_OBJ_ROUTING}*WL + {_OBJ_DENSITY}*Density + {_OBJ_GAPS}*Gaps\n")
task.train()

for epoch in range(1, args.epochs + 1):
    random.shuffle(func_list)

    epoch_losses = []

    for i in range(0, max(len(func_list), 1), args.funcs_per_batch):
        batch = func_list[i : i + args.funcs_per_batch]
        if not batch:
            continue

        optimizer.zero_grad()

        loss, step_stats = physical_reinforce_forward(
            task, batch,
            best_obj,
            num_traj       = args.num_traj,
            max_steps      = args.max_steps,
            temperature    = args.temperature,
            clip_eps       = args.clip_eps,
            lambda_entropy = args.lambda_entropy,
            astran_workers = args.astran_workers,
        )
        # accumulate ASTRAN metrics for logging
        _epoch_cpp.extend(step_stats["cpp"])
        _epoch_mismatch.extend(step_stats["mismatch"])
        _epoch_wl.extend(step_stats["wl"])
        _epoch_density.extend(step_stats["density"])
        _epoch_gaps.extend(step_stats["gaps"])
        _epoch_obj.extend(step_stats["obj"])

        if loss is None:
            ppo_step += 1
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(task.parameters(), 1.0)
        optimizer.step()

        epoch_losses.append(loss.item())
        ppo_step += 1

        if ppo_step % args.agent_sync_every == 0:
            task.sync_agent()

    if not epoch_losses:
        continue

    avg_loss = sum(epoch_losses) / len(epoch_losses)

    if epoch % args.log_interval == 0 or epoch == 1:
        ev = evaluate(task, func_list,
                      n_trials=10, max_steps=args.max_steps,
                      temperature=args.temperature)

        # Summarise ASTRAN metrics seen since last log
        def _avg(lst): return sum(lst) / len(lst) if lst else float("nan")
        cpp_str      = f"{_avg(_epoch_cpp):.1f}"      if _epoch_cpp      else "?"
        mismatch_str = f"{_avg(_epoch_mismatch):.1f}" if _epoch_mismatch else "?"
        wl_str       = f"{_avg(_epoch_wl):.1f}"       if _epoch_wl       else "?"
        density_str  = f"{_avg(_epoch_density):.1f}"  if _epoch_density  else "?"
        gaps_str     = f"{_avg(_epoch_gaps):.1f}"     if _epoch_gaps     else "?"
        obj_str      = f"{_avg(_epoch_obj):.0f}"      if _epoch_obj      else "?"
        best_obj_str = (f"{min(best_obj.values())}" if best_obj else "?")
        _epoch_cpp.clear();  _epoch_mismatch.clear()
        _epoch_wl.clear();   _epoch_density.clear()
        _epoch_gaps.clear(); _epoch_obj.clear()

        score   = ev["pdn_success"] + ev["pun_success"]
        is_best = score > best_score
        marker  = " ← best" if is_best else ""
        if is_best:
            best_score   = score
            best_ck_path = f"{_stem}_epoch{epoch:04d}.pt"

        ck_path = f"{_stem}_epoch{epoch:04d}.pt"
        torch.save({
            "epoch":           epoch,
            "model_state":     task.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "ppo_loss":        avg_loss,
            "pdn_success":     ev["pdn_success"],
            "pun_success":     ev["pun_success"],
            "pdn_avg_t":       ev["pdn_avg_t"],
            "pun_avg_t":       ev["pun_avg_t"],
            "best_obj":        dict(best_obj),
            "arch_args":       arch_args,
            "rl_args":         vars(args),
            "pretrain_ck":     args.pretrain_ck,
        }, ck_path)
        interval_ckpts.append(ck_path)

        print(
            f"epoch={epoch:4d}  loss={avg_loss:.4f}  "
            f"pdn={ev['pdn_success']:.2f}(T={ev['pdn_avg_t']:.1f})  "
            f"pun={ev['pun_success']:.2f}(T={ev['pun_avg_t']:.1f})  "
            f"obj={obj_str}(best={best_obj_str})  "
            f"cpp={cpp_str}  mismatch={mismatch_str}  "
            f"wl={wl_str}  density={density_str}  gaps={gaps_str}  "
            f"ppo_steps={ppo_step}{marker}"
        )

# ── Save best checkpoint ───────────────────────────────────────────────────────
if best_ck_path is not None and os.path.exists(best_ck_path):
    shutil.copy(best_ck_path, args.out_ck)
    print(f"\nBest checkpoint → {args.out_ck}")
else:
    print("\nNo eval checkpoint recorded.")

if not args.keep_ckpts:
    for p in interval_ckpts:
        if p != args.out_ck and os.path.exists(p):
            os.remove(p)
else:
    print(f"Interval checkpoints: {interval_ckpts}")

print(f"Done. Final checkpoint: {args.out_ck}")
