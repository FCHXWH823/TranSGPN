"""
finetune_rl_targets_explore.py

Exploration-hardened variant of finetune_rl_targets.py (original untouched).
Adds four techniques against the converged-suboptimal trap (e.g. P153 pun:
policy locks onto one 6-T net incl. a dangling stub, group advantage -> 0,
learning stops while the true optimum is 5-T):

  1. PRUNED SIZE ACCOUNTING: transistors not on any SRC<->SNK path (dangling
     stubs abandoned during construction) are physically removable; count
     network size AFTER pruning. (P153's 6-T net is the optimal 5-T net plus
     one dead stub — pruning alone reaches the reference.)
  2. ANCHORED ADVANTAGE (--anchor-beta): reinforce_forward_shaped adds
     beta*(r - 1.0) so a uniformly-suboptimal rollout group gets uniformly
     negative advantage instead of zero — PPO pushes off the entrenched habit.
  3. HOTTER EXPLORATION: use --temperature 1.2-1.5 / --lambda-entropy 0.05-0.1.
  4. RESTARTS (--restarts N): up to N independent finetunes from the pretrain
     base (fresh optimizer + sampling), keeping the best; stops at the first
     run that reaches t_ref.
"""
import argparse
import copy
import csv
import os
import time
from collections import deque

import torch

from finetune_rl_targets import build_model, derive_patterns
from transnet.graph import prune_gtran_functional


def _finetune_single_run(task, base_state, arch_args, vif, on_p, off_p,
                         t_ref, fid, args, device):
    """One independent RL finetune from the pretrain base (mirrors the
    original finetune_one, plus pruned-size accounting + anchored advantage)."""
    task.load_state_dict(base_state)
    task.sync_agent()
    object.__setattr__(task, "_best_t", {})
    opt = torch.optim.Adam(task.parameters(), lr=args.lr)

    rl_input = [(fid, vif, on_p, off_p, t_ref)]
    ppo_step = 0
    best_success, best_t, best_state = -1.0, float("inf"), None
    stop_epoch = args.epochs

    task.train()
    for epoch in range(1, args.epochs + 1):
        opt.zero_grad()
        if args.shaped:
            ppo_loss = task.reinforce_forward_shaped(
                rl_input, num_traj=args.num_traj, max_steps=args.max_steps,
                temperature=args.temperature, clip_eps=args.clip_eps,
                lambda_entropy=args.lambda_entropy,
                anchor_beta=args.anchor_beta,
            )
        else:
            ppo_loss = task.reinforce_forward(
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

        rollout_best = min((v for v in task._best_t.values() if v is not None),
                           default=float("inf"))
        do_eval = (epoch % args.eval_interval == 0 or epoch == args.epochs
                   or (args.early_stop and rollout_best <= t_ref))

        if do_eval:
            task.eval()
            with torch.no_grad():
                res = task.generate(vif, on_p, off_p, num_sample=args.eval_samples,
                                    max_steps=args.max_steps,
                                    temperature=args.temperature, verbose=0)
            task.train()
            correct = [r for r in res if r["correct"]]
            success = len(correct) / args.eval_samples
            # pruned size: dangling stubs are free to remove
            min_t = min((len(prune_gtran_functional(r["g_tran"], r["g_num"], on_p))
                         for r in correct), default=float("inf"))
            if success > best_success or (success == best_success and min_t < best_t):
                best_success, best_t = success, min_t
                best_state = copy.deepcopy(task.state_dict())
            if args.early_stop and best_t <= t_ref:
                stop_epoch = epoch
                break

    reached = best_t <= t_ref
    info = {"success": best_success,
            "min_t": (best_t if best_t != float("inf") else ""),
            "t_ref": t_ref, "reached_ref": reached, "stop_epoch": stop_epoch}
    return best_state, info, arch_args


def finetune_one(task, base_state, arch_args, vif, on_p, off_p,
                 t_ref, fid, args, device):
    """Restart wrapper: up to args.restarts independent runs, best kept."""
    def _key(i):
        mt = i["min_t"] if i["min_t"] != "" else float("inf")
        return (bool(i["reached_ref"]), -mt, i["success"])

    best_state, best_info = None, None
    for ri in range(max(1, args.restarts)):
        state, info, _ = _finetune_single_run(task, base_state, arch_args, vif,
                                              on_p, off_p, t_ref, fid, args, device)
        print(f"    [{fid}] restart {ri+1}/{args.restarts}: success={info['success']:.2f} "
              f"min_t={info['min_t']} reached={info['reached_ref']} "
              f"stop@{info['stop_epoch']}", flush=True)
        if best_info is None or _key(info) > _key(best_info):
            best_state, best_info = state, info
        if best_info["reached_ref"]:
            break
    return best_state, best_info, arch_args


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--targets",     default="results/rl_targets.csv")
    p.add_argument("--booleans",    default="dataset/Booleans.txt")
    p.add_argument("--pretrain-ck", default="checkpoints/transnet_pretrain_3_4input_v1.pt")
    p.add_argument("--out-dir",     default="checkpoints/rl_targets")
    p.add_argument("--results-dir", default="results/rl_finetune")
    p.add_argument("--num-shards",  type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--epochs",          type=int,   default=300)
    p.add_argument("--num-traj",        type=int,   default=8)
    p.add_argument("--temperature",     type=float, default=0.8)
    p.add_argument("--max-steps",       type=int,   default=20)
    p.add_argument("--lr",              type=float, default=3e-5)
    p.add_argument("--clip-eps",        type=float, default=0.2)
    p.add_argument("--lambda-entropy",  type=float, default=0.01)
    p.add_argument("--shaped", action="store_true")
    p.add_argument("--anchor-beta",     type=float, default=0.0,
                   help="Absolute-anchor advantage term beta*(r-1.0); breaks the "
                        "zero-gradient trap of converged-suboptimal groups.")
    p.add_argument("--restarts",        type=int,   default=1,
                   help="Independent finetune attempts from the pretrain base.")
    p.add_argument("--agent-sync-every",type=int,   default=10)
    p.add_argument("--eval-interval",   type=int,   default=50)
    p.add_argument("--eval-samples",    type=int,   default=20)
    p.add_argument("--early-stop", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--limit",           type=int,   default=0)
    args = p.parse_args()

    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    task, base_state, arch_args = build_model(args.pretrain_ck, device)
    print(f"shard {args.shard_index}/{args.num_shards}  device={device}  "
          f"base={args.pretrain_ck}  anchor_beta={args.anchor_beta} "
          f"restarts={args.restarts}", flush=True)

    import re
    expr = {}
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
        if os.path.exists(ck_out):
            continue
        e = (tg.get("expr") or "").strip() or expr.get(name)
        if not e:
            continue

        vif, on_p, off_p = derive_patterns(e, net_type)
        fid = f"{name}_{net_type}"
        best_state, info, a = finetune_one(task, base_state, arch_args, vif, on_p,
                                           off_p, t_ref, fid, args, device)
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

    res_f.close()
    print(f"shard {args.shard_index} finished: {done} targets in "
          f"{(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
