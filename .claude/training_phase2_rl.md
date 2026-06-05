# Training Phase 2: RL Fine-tuning (Transistor Count)

**Script:** `finetune_rl_transnet.py`  
**Key checkpoints:** `transnet_rl_xor_v5.pt` (best XOR), `transnet_rl_all_v1.pt` (all functions)

---

## Algorithm: Off-Policy PPO with GRPO Baseline

Implemented in `task.py:242` (`reinforce_forward`) and `task.py:319` (`_ppo_loss_batched`).

### Reward (terminal-only)

```python
# task.py:293-297
if complete:
    r = (best_T / T) if best_T else 1.0   # best_T = self._best_t[fid]
else:
    r = 0.0
```

- **No step-level reward** — only terminal reward at episode end
- `best_T[fid]` = best transistor count ever found for this function (auto-tightens)
- `r = 1.0` when the current trajectory matches the best known; `r < 1.0` for longer solutions
- Zero reward for incomplete/incorrect networks

**Why terminal-only:** Step-cost rewards caused degenerate policies where the agent stopped immediately to avoid accumulating costs. GRPO needs clear differentiation between trajectories — sparse rewards at the terminal step provide this.

### GRPO Baseline

```python
# task.py:299-306
group_mean = sum(rewards) / len(rewards)
group_std  = (sum((r - group_mean)**2 for r in rewards) / len(rewards)) ** 0.5
adv = (r - group_mean) / (group_std + 1e-8)
```

Group = all `num_traj` trajectories for one function in one PPO step. Every step in a trajectory gets the same advantage as its terminal reward's normalized value.

**Global normalization** (across all functions in batch) applied on top:
```python
adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)   # task.py:314
```

### PPO Clipping (task.py:404-408)

```python
ratio = (lp - old_logp).exp()          # current vs agent (frozen) log-probs
loss  = -min(ratio * adv, clamp(ratio, 1-ε, 1+ε) * adv)
clip_eps = 0.2   # default
```

### Entropy Bonus (task.py:410-418)

```python
lambda_entropy = 0.01   # default
# Added to each step: H(n1) + H(n2) + H(var) + H(sign)
ppo_loss = ppo_loss - lambda_entropy * mean(entropy_terms)
```

Prevents mode collapse. Computed over the masked logit distributions (only valid actions contribute).

---

## Batched PPO Loss (`_ppo_loss_batched`)

All trajectory steps from one PPO update are packed into a single `PackedGraph` and processed with **one MPNN forward pass** (task.py:338):

```python
packed = td_data.Graph.pack([s.graph for s in steps]).to(device)
out    = self.model(packed, packed.node_feature.float())
nf_all = out["node_feature"]   # [total_N_all, h]
gf_all = out["graph_feature"]  # [S, 2h]
```

`mlp_var` and `mlp_sign` are also batched together (task.py:349-360). Only `mlp_node1` and `mlp_node2` must be per-step (variable graph size).

This reduces PPO update MPNN calls from `S` → `1`.

---

## K-Trajectory Batching (`_generate_K_trajs`)

`task.py:531`. Batches all K concurrent trajectory graphs at each generation step:

```python
packed  = td_data.Graph.pack(graphs).to(device)   # all K active graphs
out     = agent.model(packed, packed.node_feature.float())
# → process K trajectories' node actions from one MPNN call
```

**Profiling results** (from `profile_rl.py`):
| Component | % of wall time |
|-----------|----------------|
| MPNN forward | 69.9% |
| MLP heads | 16.2% |
| n2 BFS safety | 5.5% |
| build_graph | 5.0% |
| var BFS safety | 2.2% |

K-batching reduces MPNN calls from `K × max_steps` to `max_steps`. At K=8, max_steps=20: ~8× fewer MPNN calls during trajectory generation.

**Combined speedup:** `_generate_K_trajs` (K=4) + batched PPO loss = **3.44× total speedup** vs. original sequential loop.

**Threading was tried and REJECTED:** Python GIL makes thread-based parallelism WORSE (0.75× slowdown) for this workload.

**`torch.compile` was tried and REJECTED:** torchdrug's `PackedGraph` uses custom Python objects incompatible with the `inductor` backend. Caused runtime crash (not compile-time), so `try/except` around `torch.compile(...)` didn't help. Removed entirely.

---

## NLL Regularization (Optional)

```python
# finetune_rl_transnet.py:294-300
if use_nll and ppo_step % args.nll_every == 0:
    nll_loss, _ = task.MLE_forward(next_nll_batch())
    total_loss  = ppo_loss + lambda_nll * nll_loss
```

Mixes in one NLL batch every `nll_every` PPO steps to prevent catastrophic forgetting of pretrained behaviors. Default: `nll_every=5`, `lambda_nll=0.1`. Disabled in pipeline runs (`--nll_every 0`).

---

## Key Hyperparameters

### For per-function pipeline fine-tuning (`run_rl_pipeline.py`)

| Parameter | Value |
|-----------|-------|
| `--epochs` | 300 |
| `--num_traj` | 8 |
| `--temperature` | 0.8 |
| `--max_steps` | 20 |
| `--lr` | 3e-5 |
| `--lambda_entropy` | 0.01 |
| `--nll_every` | 0 (disabled) |

### For all-function training (command used)

```bash
env/bin/python3 finetune_rl_transnet.py \
  --pretrain_ck checkpoints/transnet_pretrain_v6.pt \
  --out_ck checkpoints/transnet_rl_all_v1.pt \
  --epochs 100 --num_traj 4 --lr 1e-4 --max_steps 30
```

### For XOR-specific training

```bash
env/bin/python3 finetune_rl_transnet.py \
  --func '!a*!b*c+!a*b*!c+a*!b*!c+a*b*c' \
  --pretrain_ck checkpoints/transnet_pretrain_v6.pt \
  --out_ck checkpoints/transnet_rl_xor_v5.pt \
  --epochs 500 --num_traj 8 --lr 3e-5
```

---

## Why All-Function Model Is Worse Than XOR-Specific on XOR

With 242 functions and `funcs_per_batch=4`, XOR gets only ~1.7% of gradient updates. The 9-transistor solution for XOR is a stable local optimum sufficient for many other functions too, so the model doesn't specialize. XOR's optimal 8T topology requires sustained, XOR-focused pressure to converge.

---

## Agent Sync Schedule

`sync_agent()` called every `agent_sync_every=10` PPO steps. The frozen agent (trajectory generator) gradually catches up to the current policy. Too-frequent sync: reduces off-policy correction benefit. Too-infrequent: stale behavior policy diverges from current policy, PPO ratio becomes large and unstable.

---

## Checkpoint Format

```python
{
    "epoch":           epoch,
    "model_state":     task.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "ppo_loss":        avg_ppo,
    "success":         success,     # fraction of correct trials in eval
    "avg_t":           avg_t,       # mean transistors among correct trials
    "arch_args":       arch_args,   # {"hidden":64, "num_layer":3, "mlp_hidden":128}
    "rl_args":         vars(args),
    "pretrain_ck":     args.pretrain_ck,
}
```

Best checkpoint (highest success, tie-break: lowest avg_t) is copied to `--out_ck`. Intermediate per-epoch checkpoints deleted by default (`--keep_ckpts` to retain).
