"""
finetune_rl_targets_explore2.py

TEMPLATE v2 of the exploration-hardened per-target RL finetuner.
Everything from finetune_rl_targets_explore.py (pruned accounting, anchored
advantage, restarts, hot exploration) PLUS:

  5. REWIRE POLISH (--polish-states): a local plateau search applied to every
     correct evaluated network. Move = delete one transistor + re-add one
     anywhere (size-neutral); after each move, try plain deletions. This finds
     optima that are unreachable by delete-only pruning — e.g. P1687 pun:
     the RL policy's 9-T net has two load-bearing `a` transistors; ONE rewire
     (re-anchoring `a` between two interior nodes) unlocks a deletion -> 8-T
     (= the reference), found after exploring just 2 plateau states.

     Rationale: RL discovers good global structure but converges before
     sampling the coordinated multi-step trajectory change that separates a
     near-optimal net from the optimum. The polish makes that final step a
     deterministic combinatorial search instead of a sampling miracle.

Polish is applied at EVAL time only (reward-side accounting keeps the cheaper
functional prune), and the polished size decides reached_ref/early-stop.
"""
import argparse
import copy
import csv
import os
import time
from collections import deque

import torch

from finetune_rl_targets import build_model, derive_patterns
from finetune_rl_targets_explore import finetune_one as _explore_finetune_one  # noqa: F401 (kept for reference)
from transnet.graph import prune_gtran_functional
from transnet.literal import check_safety, covered_patterns


# ─────────────────────────────────────────────────────────────────────────────
# Technique 5: rewire polish
# ─────────────────────────────────────────────────────────────────────────────

def _canon(net):
    return tuple(sorted((min(u, v), max(u, v), gv, ng) for u, v, gv, ng in net))


def _implements(net, g_num, on_p, off_p):
    """Exact functionality: covers every on-pattern, conducts no off-pattern."""
    return (covered_patterns(net, g_num, on_p) == on_p
            and check_safety(net, g_num, off_p))


def rewire_polish(g_tran, g_num, on_p, off_p, n_vars, max_states=500):
    """Plateau local search: size-neutral (delete-1 + add-1) rewires, trying a
    plain deletion at every visited state. Returns the smallest functionally
    identical network found. BFS over unseen plateau states, budgeted."""
    on_p = frozenset(on_p)
    net = prune_gtran_functional(list(g_tran), g_num, on_p)
    if max_states <= 0:
        return net

    nodes = sorted({n for e in net for n in e[:2]})
    pairs = [(u, v) for i, u in enumerate(nodes) for v in nodes[i + 1:]]
    lits = [(gv, ng) for gv in range(n_vars) for ng in range(2)]

    def _try_delete(state):
        for i in range(len(state)):
            cand = state[:i] + state[i + 1:]
            if _implements(cand, g_num, on_p, off_p):
                return cand
        return None

    frontier = deque([net])
    seen = {_canon(net)}
    explored = 0
    while frontier and explored < max_states:
        state = frontier.popleft()
        explored += 1
        smaller = _try_delete(state)
        if smaller is not None:
            # descend: re-prune and restart the plateau search one size down
            return rewire_polish(smaller, g_num, on_p, off_p, n_vars,
                                 max_states - explored)
        for i in range(len(state)):
            rest = state[:i] + state[i + 1:]
            for (u, v) in pairs:
                for (gv, ng) in lits:
                    cand = rest + [(u, v, gv, ng)]
                    key = _canon(cand)
                    if key in seen:
                        continue
                    if _implements(cand, g_num, on_p, off_p):
                        seen.add(key)
                        frontier.append(cand)
    return net


# ─────────────────────────────────────────────────────────────────────────────
# Finetune (explore v1 + polish at eval)
# ─────────────────────────────────────────────────────────────────────────────

def _finetune_single_run(task, base_state, arch_args, vif, on_p, off_p,
                         t_ref, fid, args, device):
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
            # polish each correct net (functional prune + budgeted rewiring)
            min_t = float("inf")
            for r in correct:
                p = rewire_polish(r["g_tran"], r["g_num"], on_p, off_p,
                                  len(vif), max_states=args.polish_states)
                min_t = min(min_t, len(p))
                if min_t <= t_ref:
                    break
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
    p.add_argument("--anchor-beta",     type=float, default=0.0)
    p.add_argument("--restarts",        type=int,   default=1)
    p.add_argument("--polish-states",   type=int,   default=500,
                   help="Plateau-state budget of the rewire polish per network "
                        "(0 disables; polish runs at eval time only).")
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
          f"restarts={args.restarts} polish_states={args.polish_states}", flush=True)

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
