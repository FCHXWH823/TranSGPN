"""
finetune_rl_targets_v3.py

TEMPLATE v3: core-RL improvements (no external search — the RL itself learns
better), via task.reinforce_forward_dense:

  * dense per-step reward: Delta-coverage - per-transistor cost, plus a
    completion bonus t_ref / T_eff (functionally pruned size)
  * learned critic V(s) -> per-step advantages R_t - V(s_t); credit is
    assigned to individual transistor placements, and the baseline never
    degenerates when all rollouts agree (the GRPO group-std failure mode)
  * self-imitation (SIL): the best complete trajectory per target is replayed
    every update with positive-part advantage, consolidating rare discoveries

Accounting matches the explore versions: eval min_t = functionally pruned
size; early-stop on reaching t_ref. No rewire polish here by design.
"""
import argparse
import copy
import csv
import os
import time

import torch
from torchdrug import models

from finetune_rl_targets import derive_patterns
from transnet import GCPNTransNet
from transnet.graph import NODE_FEAT_DIM, prune_gtran_functional
from transnet.literal import EDGE_FEAT_DIM


def build_model_v3(ck_path, device):
    """Like finetune_rl_targets.build_model but with a critic head (fresh —
    absent from pretrain checkpoints; strict=False leaves it random-init)."""
    ck = torch.load(ck_path, map_location="cpu")
    a = ck.get("arch_args") or ck.get("args", {})
    mpnn = models.MPNN(
        input_dim=NODE_FEAT_DIM, hidden_dim=a.get("hidden", 64),
        edge_input_dim=EDGE_FEAT_DIM, num_layer=a.get("num_layer", 3),
        batch_norm=False,
    )
    task = GCPNTransNet(mpnn, hidden_dim_mlp=a.get("mlp_hidden", 128),
                        global_attn=a.get("global_attn", False),
                        pointer_head=a.get("pointer_head", False),
                        value_head=True).to(device)
    task.load_state_dict(ck["model_state"], strict=False)
    base_state = copy.deepcopy(task.state_dict())   # includes fresh critic
    return task, base_state, a


def _finetune_single_run(task, base_state, arch_args, vif, on_p, off_p,
                         t_ref, fid, args, device):
    task.load_state_dict(base_state)
    task.sync_agent()
    object.__setattr__(task, "_best_t", {})
    object.__setattr__(task, "_best_traj", {})
    object.__setattr__(task, "_seen_structs", {})
    opt = torch.optim.Adam(task.parameters(), lr=args.lr)

    # v4 plateau re-heating state
    base_temp, base_ent = args.temperature, args.lambda_entropy
    cur_temp,  cur_ent  = base_temp, base_ent
    best_eval_t = float("inf")
    evals_no_improve = 0

    rl_input = [(fid, vif, on_p, off_p, t_ref)]
    ppo_step = 0
    best_success, best_t, best_state = -1.0, float("inf"), None
    stop_epoch = args.epochs

    task.train()
    for epoch in range(1, args.epochs + 1):
        opt.zero_grad()
        ppo_loss = task.reinforce_forward_dense(
            rl_input, num_traj=args.num_traj, max_steps=args.max_steps,
            temperature=cur_temp, clip_eps=args.clip_eps,
            lambda_entropy=cur_ent,
            cov_bonus=args.cov_bonus, step_cost=args.step_cost,
            complete_bonus=args.complete_bonus, sil_weight=args.sil_weight,
            novelty_bonus=args.novelty_bonus, beat_bonus=args.beat_bonus,
            sil_subopt_scale=args.sil_subopt_scale,
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
            min_t = min((len(prune_gtran_functional(r["g_tran"], r["g_num"], on_p))
                         for r in correct), default=float("inf"))
            if success > best_success or (success == best_success and min_t < best_t):
                best_success, best_t = success, min_t
                best_state = copy.deepcopy(task.state_dict())
            if args.early_stop and best_t <= t_ref:
                stop_epoch = epoch
                break

            # v4 plateau re-heating: if the pruned min_t stopped improving while
            # still above the reference, heat up temperature+entropy to escape;
            # cool back to base the moment a shorter net appears.
            if min_t < best_eval_t:
                best_eval_t = min_t
                evals_no_improve = 0
                cur_temp, cur_ent = base_temp, base_ent
            else:
                evals_no_improve += 1
                if (evals_no_improve >= args.reheat_patience
                        and best_t > t_ref):
                    cur_temp = min(cur_temp * args.reheat_factor,
                                   base_temp * args.reheat_max)
                    cur_ent  = min(cur_ent  * args.reheat_factor,
                                   base_ent  * args.reheat_max)
                    evals_no_improve = 0

    reached = best_t <= t_ref
    info = {"success": best_success,
            "min_t": (best_t if best_t != float("inf") else ""),
            "t_ref": t_ref, "reached_ref": reached, "stop_epoch": stop_epoch}
    return best_state, info, arch_args


def finetune_one(task, base_state, arch_args, vif, on_p, off_p,
                 t_ref, fid, args, device):
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
    p.add_argument("--num-traj",        type=int,   default=16)
    p.add_argument("--temperature",     type=float, default=1.0)
    p.add_argument("--max-steps",       type=int,   default=20)
    p.add_argument("--lr",              type=float, default=1e-4)
    p.add_argument("--clip-eps",        type=float, default=0.2)
    p.add_argument("--lambda-entropy",  type=float, default=0.05)
    p.add_argument("--cov-bonus",       type=float, default=1.0)
    p.add_argument("--step-cost",       type=float, default=0.03)
    p.add_argument("--complete-bonus",  type=float, default=1.0)
    p.add_argument("--sil-weight",      type=float, default=0.2)
    # v4 anti-premature-convergence knobs
    p.add_argument("--novelty-bonus",   type=float, default=0.15,
                   help="Reward for a complete net with a not-yet-seen pruned structure.")
    p.add_argument("--beat-bonus",      type=float, default=0.5,
                   help="Jackpot added when a rollout strictly beats the running best T_eff.")
    p.add_argument("--sil-subopt-scale",type=float, default=0.3,
                   help="Multiply sil-weight while the buffered best is still > t_ref.")
    p.add_argument("--reheat-patience", type=int,   default=2,
                   help="Evals with no min_t improvement (and best>ref) before re-heating.")
    p.add_argument("--reheat-factor",   type=float, default=1.4,
                   help="Multiply temperature+entropy per re-heat trigger.")
    p.add_argument("--reheat-max",      type=float, default=3.0,
                   help="Cap re-heated temperature/entropy at this multiple of base.")
    p.add_argument("--restarts",        type=int,   default=1)
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

    task, base_state, arch_args = build_model_v3(args.pretrain_ck, device)
    print(f"shard {args.shard_index}/{args.num_shards}  device={device}  "
          f"base={args.pretrain_ck}  [v3 dense+critic+SIL] "
          f"cov={args.cov_bonus} cost={args.step_cost} sil={args.sil_weight}",
          flush=True)

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
