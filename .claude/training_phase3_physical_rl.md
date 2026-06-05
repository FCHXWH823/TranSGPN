# Training Phase 3: Physical RL (ASTRAN Placement)

**Script:** `physical_finetune_rl_transnet.py`  
**Checkpoint (partial):** `checkpoints/transnet_physical_xor_v1.pt`

---

## Goal

Close the RL loop through a real placement tool (ASTRAN) so the agent learns to synthesize transistor network topologies that are not just logically correct but also **physically efficient** — fewer CPP (Cell Pitch), lower wire length, fewer diffusion breaks (gaps).

---

## Dual Network Synthesis

CMOS requires two complementary networks for each Boolean function `f(x)`:

| Network | Implements | On-patterns |
|---------|-----------|-------------|
| PDN (pull-down, NMOS) | `¬f(x)` | = off-patterns of f |
| PUN (pull-up, PMOS) | `f(¬x)` | = bit-complement of on-patterns |

Pattern derivation in `physical_finetune_rl_transnet.py:95-105`:

```python
def derive_pdn_patterns(on_p, off_p):
    return off_p, on_p              # PDN on = f off; PDN off = f on

def derive_pun_patterns(on_p, off_p):
    comp = lambda pat: tuple(1-b for b in pat)
    pun_on  = frozenset(comp(p) for p in on_p)
    pun_off = frozenset(comp(p) for p in off_p)
    return pun_on, pun_off
```

The **same agent** synthesizes both networks, conditioned on their respective on/off pattern sets.

---

## ASTRAN Objective

```python
# physical_finetune_rl_transnet.py:157-177
_OBJ_CPP      = 3   # WidthCost     (CPP = Contacted Poly Pitch)
_OBJ_MISMATCH = 4   # GateMismatchCost
_OBJ_ROUTING  = 1   # RoutingCost   (wire length)
_OBJ_DENSITY  = 4   # RtDensityCost (routing congestion peak)
_OBJ_GAPS     = 2   # GapsCost      (diffusion breaks)

def astran_objective(metrics):
    return (  _OBJ_CPP      * metrics["width"]
            + _OBJ_MISMATCH * metrics["gate_mismatches"]
            + _OBJ_ROUTING  * metrics["wl"]
            + _OBJ_DENSITY  * metrics["rt_density"]
            + _OBJ_GAPS     * metrics["nr_gaps"]  )
```

**Critical alignment:** These weights MUST match the `cellgen place` command parameters — so the Python reward function reproduces ASTRAN's exact internal objective.

---

## Adaptive Self-Tightening Reward

```python
# physical_finetune_rl_transnet.py:259-279
def physical_reward(metrics, best_obj, fid):
    obj = astran_objective(metrics)
    if obj == 0: obj = 1          # guard degenerate case
    if fid not in best_obj or obj < best_obj[fid]:
        best_obj[fid] = obj       # tighten as training improves
    return best_obj[fid] / obj
```

- `r = 1.0` when this placement matches the best ever found for this function
- `r < 1.0` for worse placements
- As training finds better topologies, `best_obj[fid]` decreases, making future worse solutions receive lower rewards automatically

---

## PPO Forward: `physical_reinforce_forward`

`physical_finetune_rl_transnet.py:284`

1. **Trajectory generation** (no_grad): for each function in batch, generate `num_traj` PDN+PUN pairs using `task._generate_traj`
2. **ASTRAN evaluation** (parallel, ThreadPoolExecutor): run ASTRAN on all complete pairs
3. **GRPO**: normalize rewards within each function's group
4. **PPO loss**: call `task._ppo_loss_batched` on combined PDN+PUN steps

```python
steps = item["pdn_steps"] + item["pun_steps"]   # all steps get the joint reward
all_steps.extend(steps)
all_advantages.extend([adv] * len(steps))
```

Note: Phase 3 uses `_generate_traj` (sequential, not K-batched) because PDN and PUN are separate trajectories that need independent safety-mask computation.

---

## Parallel ASTRAN Evaluation

```python
with ThreadPoolExecutor(max_workers=astran_workers) as pool:
    list(pool.map(_eval_item, eval_items))
```

Default `astran_workers=4`. ASTRAN is a separate process (subprocess), so threading works here — each thread waits on I/O (subprocess.run), not Python-GIL-bound computation. This is the opposite of why threading fails for MPNN.

---

## Training Loop

```python
for epoch in range(1, args.epochs + 1):
    random.shuffle(func_list)
    for i in range(0, len(func_list), args.funcs_per_batch):
        batch = func_list[i : i + args.funcs_per_batch]
        loss, step_stats = physical_reinforce_forward(task, batch, best_obj, ...)
        loss.backward()
        optimizer.step()
    if epoch % log_interval == 0:
        ev = evaluate(task, func_list, ...)   # fast eval without ASTRAN
        print(f"obj={avg_obj}(best={best_obj})  cpp={avg_cpp}  ...")
```

**Logging format:**
```
epoch= 10  loss=0.0123  pdn=0.80(T=6.2)  pun=0.78(T=6.8)
           obj=87(best=72)  cpp=19.3  mismatch=0.1  wl=45.2  density=5.1  gaps=1.8
```

---

## Key Hyperparameters (defaults)

| Parameter | Value |
|-----------|-------|
| `--epochs` | 200 |
| `--funcs_per_batch` | 2 |
| `--num_traj` | 4 (PDN+PUN pairs per function) |
| `--temperature` | 0.8 |
| `--max_steps` | 20 |
| `--lr` | 3e-5 |
| `--clip_eps` | 0.2 |
| `--lambda_entropy` | 0.01 |
| `--agent_sync_every` | 5 |
| `--astran_workers` | 4 |
| `--astran_timeout` | 30s |

---

## Run Command

```bash
# Single XOR function
env/bin/python3 physical_finetune_rl_transnet.py \
    --func '!a*!b*c+!a*b*!c+a*!b*!c+a*b*c' \
    --pretrain_ck checkpoints/transnet_pretrain_v6.pt \
    --out_ck checkpoints/transnet_physical_xor_v2.pt \
    --epochs 200 --num_traj 4

# All SAT functions
env/bin/python3 physical_finetune_rl_transnet.py \
    --pretrain_ck checkpoints/transnet_pretrain_v6.pt \
    --out_ck checkpoints/transnet_physical_all_v1.pt
```

---

## Eval During Training (No ASTRAN)

`evaluate()` in `physical_finetune_rl_transnet.py:473` runs fast: generates PDN and PUN trajectories, checks correctness, reports `pdn_success`, `pun_success`, `pdn_avg_t`, `pun_avg_t`. Does NOT call ASTRAN (for speed). ASTRAN metrics are only collected during the PPO steps.

---

## Current State

- `transnet_physical_xor_v1.pt` exists (partial XOR run)
- Full Phase 3 training on all functions has NOT been run
- Next step: run Phase 3 after Phase 2 pipeline completes and produces per-function checkpoints
