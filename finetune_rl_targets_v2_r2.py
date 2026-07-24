"""
finetune_rl_targets.py

Per-target Phase-2 RL fine-tuning (run_rl_pipeline style) for every PDN/PUN
sub-network that the pretrained policy could NOT synthesize at or below the
PClassResults SAT-optimal transistor count.

Each target is fine-tuned INDEPENDENTLY from the same pretrain base (not chained),
producing one checkpoint per target:  checkpoints/rl_targets/rl_<name>_<type>.pt

The job is sharded across many single-threaded CPU processes (--num-shards /
--shard-index) so thousands of independent finetunes run in parallel without a
GPU (and thus without the GPU-utilization watchdog). It is resumable: a target
whose checkpoint already exists is skipped.

Reward (unchanged from finetune_rl_transnet.py): terminal-only best_T/T via
reinforce_forward — self-tightens toward the shortest network the policy finds.
The reference t_ref is the SUCCESS criterion (did RL reach a correct network with
T <= t_ref?), logged per target, matching run_rl_pipeline's Criterion B.

Targets file (results/rl_targets.csv): columns name,type,expr,t_ref,reason,our_count
"""
import argparse
import copy
import csv
import math
import os
import time

import torch
from torchdrug import models

from transnet.graph import NODE_FEAT_DIM
from transnet.literal import EDGE_FEAT_DIM, extract_vars, on_off_split, parse_sop_expr
from transnet.canonical import canonicalize_expr
from transnet.task import GCPNTransNet


def _comp(p):
    return tuple(1 - b for b in p)


def derive_patterns(expr, net_type):
    """Return (vif, on_p, off_p) for the PDN (f) or PUN (!f(!x)) sub-network.

    Variable-position canonicalization (consistent with pretraining + generation):
    relabel the support to the lowest slots before deriving patterns so the RL
    rollouts run in the same low-slot space the canonical policy was trained on.
    f and !f(!x) share the same support, so canonicalizing f covers both; the PUN
    comp-trick preserves slot positions (it only flips bit values).
    """
    c_expr, _real_vars, _perm = canonicalize_expr(expr)
    vif = extract_vars(c_expr)                # = ALL_VARS[:K]
    on_f = parse_sop_expr(c_expr, vif)
    on_f, off_f = on_off_split(on_f)
    if net_type == "pdn":
        return vif, on_f, off_f
    # PUN = !f(!x): on = comp(off_f), off = comp(on_f)
    return vif, frozenset(_comp(p) for p in off_f), frozenset(_comp(p) for p in on_f)


def build_model(ck_path, device):
    ck = torch.load(ck_path, map_location="cpu")
    a = ck.get("arch_args") or ck.get("args", {})
    mpnn = models.MPNN(
        input_dim=NODE_FEAT_DIM, hidden_dim=a.get("hidden", 64),
        edge_input_dim=EDGE_FEAT_DIM, num_layer=a.get("num_layer", 3),
        batch_norm=False,
    )
    task = GCPNTransNet(mpnn, hidden_dim_mlp=a.get("mlp_hidden", 128)).to(device)
    base_state = copy.deepcopy(ck["model_state"])
    return task, base_state, a


def finetune_one(task, base_state, arch_args, vif, on_p, off_p, t_ref, fid, args, device):
    """Independent RL finetune from the pretrain base. Returns (best_state, info).

    Early stop (args.early_stop): as soon as a correct network with
    T <= t_ref (the PClassResults reference count) is found, stop fine-tuning
    this target — the objective for it is already met.
    """
    task.load_state_dict(base_state)            # fresh start from pretrain
    task.sync_agent()
    object.__setattr__(task, "_best_t", {})     # reset self-tightening target
    opt = torch.optim.Adam(task.parameters(), lr=args.lr)

    rl_input = [(fid, vif, on_p, off_p, t_ref)]
    ppo_step = 0
    best_success, best_t, best_state = -1.0, float("inf"), None
    stop_epoch = args.epochs

    task.train()
    for epoch in range(1, args.epochs + 1):
        opt.zero_grad()
        ppo_loss = task.reinforce_forward_shaped(
            rl_input, num_traj=args.num_traj, max_steps=args.max_steps,
            temperature=args.temperature, clip_eps=args.clip_eps,
            lambda_entropy=args.lambda_entropy,
        )
        if ppo_loss is not None:
            ppo_loss.backward()
            torch.nn.utils.clip_grad_norm_(task.parameters(), 1.0)
            opt.step()
            ppo_step += 1
            if ppo_step % args.agent_sync_every == 0:
                task.sync_agent()

        # Cheap trigger: a training rollout already produced a network <= t_ref.
        # Force an eval this epoch to capture/verify that model, then early-stop.
        rollout_best = min((v for v in task._best_t.values() if v is not None),
                           default=float("inf"))
        do_eval = (epoch % args.eval_interval == 0 or epoch == args.epochs
                   or (args.early_stop and rollout_best <= t_ref))

        if do_eval:
            task.eval()
            with torch.no_grad():
                res = task.generate(vif, on_p, off_p, num_sample=args.eval_samples,
                                    max_steps=args.max_steps, temperature=args.temperature,
                                    verbose=0)
            task.train()
            correct = [r for r in res if r["correct"]]
            success = len(correct) / args.eval_samples
            min_t = min((len(r["g_tran"]) for r in correct), default=float("inf"))
            # best = higher success, tie-break shorter min_t
            if success > best_success or (success == best_success and min_t < best_t):
                best_success, best_t = success, min_t
                best_state = copy.deepcopy(task.state_dict())

            # Early stop once a correct network reaches the reference count.
            if args.early_stop and best_t <= t_ref:
                stop_epoch = epoch
                break

    reached = best_t <= t_ref
    info = {"success": best_success,
            "min_t": (best_t if best_t != float("inf") else ""),
            "t_ref": t_ref, "reached_ref": reached, "stop_epoch": stop_epoch}
    return best_state, info, arch_args


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--targets",     default="results/rl_targets.csv")
    p.add_argument("--booleans",    default="dataset/Booleans.txt")
    p.add_argument("--pretrain-ck", default="checkpoints/transnet_pretrain_3_4input_v1.pt")
    p.add_argument("--warm-dir",    default="checkpoints/rl_targets_v2",
                   help="Round-1 checkpoint dir: each target warm-starts from its "
                        "rl_<name>_<type>.pt here (continue pushing from the already-"
                        "good policy) instead of the cold pretrain base.")
    p.add_argument("--out-dir",     default="checkpoints/rl_targets")
    p.add_argument("--results-dir", default="results/rl_finetune")
    p.add_argument("--num-shards",  type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    # documented per-function pipeline hyperparameters
    p.add_argument("--epochs",          type=int,   default=300)
    p.add_argument("--num-traj",        type=int,   default=8)
    p.add_argument("--temperature",     type=float, default=0.8)
    p.add_argument("--max-steps",       type=int,   default=20)
    p.add_argument("--lr",              type=float, default=3e-5)
    p.add_argument("--clip-eps",        type=float, default=0.2)
    p.add_argument("--lambda-entropy",  type=float, default=0.01)
    p.add_argument("--agent-sync-every",type=int,   default=10)
    p.add_argument("--eval-interval",   type=int,   default=50)
    p.add_argument("--eval-samples",    type=int,   default=20)
    p.add_argument("--early-stop", action=argparse.BooleanOptionalAction, default=True,
                   help="Stop a target's finetune as soon as a correct network with "
                        "T <= its reference t_ref is found (use --no-early-stop to disable).")
    p.add_argument("--limit",           type=int,   default=0)
    args = p.parse_args()

    torch.set_num_threads(1)   # one core per shard process
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    task, base_state, arch_args = build_model(args.pretrain_ck, device)
    print(f"shard {args.shard_index}/{args.num_shards}  device={device}  base={args.pretrain_ck}",
          flush=True)

    expr = {}
    import re
    LINE = re.compile(r"^\s*(\S+?)\s*\(\d+\)\s*:\s*(.+?)\s*$")
    for raw in open(args.booleans):
        m = LINE.match(raw)
        if m:
            expr[m.group(1)] = m.group(2)

    targets = list(csv.DictReader(open(args.targets)))
    if args.limit:
        targets = targets[: args.limit]

    res_path = os.path.join(args.results_dir, f"rl_results.shard{args.shard_index}.csv")
    res_exists = os.path.exists(res_path)
    res_f = open(res_path, "a", newline="")
    res_w = csv.DictWriter(res_f, fieldnames=["name", "type", "t_ref", "success",
                                              "min_t", "reached_ref", "stop_epoch"])
    if not res_exists:
        res_w.writeheader(); res_f.flush()

    t0 = time.time()
    done = 0
    for i, tg in enumerate(targets):
        if args.num_shards > 1 and (i % args.num_shards) != args.shard_index:
            continue
        name, net_type, t_ref = tg["name"], tg["type"], int(tg["t_ref"])
        ck_out = os.path.join(args.out_dir, f"rl_{name}_{net_type}.pt")
        if os.path.exists(ck_out):       # resume: already finetuned
            continue
        # Prefer the target row's own expr column (e.g. NSP catalog); fall back to
        # the Booleans.txt lookup by name (4-input targets).
        e = (tg.get("expr") or "").strip() or expr.get(name)
        if not e:
            continue

        vif, on_p, off_p = derive_patterns(e, net_type)
        fid = f"{name}_{net_type}"
        # Round-2 warm start: continue from this target's round-1 policy if present.
        warm_ck = os.path.join(args.warm_dir, f"rl_{name}_{net_type}.pt")
        if os.path.exists(warm_ck):
            start_state = torch.load(warm_ck, map_location="cpu")["model_state"]
        else:
            start_state = base_state
        best_state, info, a = finetune_one(task, start_state, arch_args, vif, on_p, off_p,
                                           t_ref, fid, args, device)
        if best_state is not None:
            torch.save({
                "model_state": best_state, "arch_args": a,
                "name": name, "type": net_type, "t_ref": t_ref,
                "success": info["success"], "min_t": info["min_t"],
                "reached_ref": info["reached_ref"], "stop_epoch": info["stop_epoch"],
                "pretrain_ck": args.pretrain_ck, "rl_args": vars(args),
            }, ck_out)
        res_w.writerow({"name": name, "type": net_type, "t_ref": t_ref,
                        "success": f"{info['success']:.2f}", "min_t": info["min_t"],
                        "reached_ref": info["reached_ref"], "stop_epoch": info["stop_epoch"]})
        res_f.flush()
        done += 1
        if done % 10 == 0:
            rate = done / (time.time() - t0)
            print(f"  shard{args.shard_index}: done={done}  last={fid} "
                  f"success={info['success']:.2f} min_t={info['min_t']} ref={info['t_ref']} "
                  f"reached={info['reached_ref']} stop@{info['stop_epoch']}  "
                  f"({rate:.3f} tgt/s)", flush=True)

    res_f.close()
    print(f"shard {args.shard_index} finished: {done} targets in {(time.time()-t0)/60:.1f} min",
          flush=True)


if __name__ == "__main__":
    main()
