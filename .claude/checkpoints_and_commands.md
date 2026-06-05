# Checkpoints and Training Commands

---

## Checkpoint Registry

| Checkpoint | Phase | Notes |
|-----------|-------|-------|
| `transnet_pretrain.pt` | 1 | First pretrain |
| `transnet_pretrain_v2.pt` | 1 | |
| `transnet_pretrain_v3.pt` | 1 | |
| `transnet_pretrain_v4.pt` | 1 | |
| `transnet_pretrain_v5.pt` | 1 | |
| `transnet_pretrain_v6.pt` | 1 | **Best pretrain — base for all RL** |
| `transnet_rl_xor.pt` | 2 | Early XOR RL |
| `transnet_rl_xor_v2.pt` | 2 | |
| `transnet_rl_xor_v3.pt` | 2 | |
| `transnet_rl_xor_v4.pt` | 2 | |
| `transnet_rl_xor_v5.pt` | 2 | **Best XOR-specific RL** |
| `transnet_rl_all_v1.pt` | 2 | All-function RL (worse on XOR than v5) |
| `transnet_physical_xor_v1.pt` | 3 | XOR physical RL, partial run |
| `rl_3_NNN.pt` (33 files) | 2 | Per-function pipeline checkpoints |

All checkpoints stored in `checkpoints/`.

---

## Checkpoint Internal Format

All RL checkpoints share this structure:
```python
{
    "epoch":           int,
    "model_state":     OrderedDict,   # task.state_dict()
    "optimizer_state": OrderedDict,
    "ppo_loss":        float,
    "success":         float,         # eval success rate
    "avg_t":           float,         # mean transistors (correct trials)
    "arch_args":       dict,          # {"hidden":64, "num_layer":3, "mlp_hidden":128}
    "rl_args":         dict,          # CLI args used during training
    "pretrain_ck":     str,           # path to pretrain checkpoint used
}
```

Physical RL checkpoints add:
```python
    "pdn_success": float,
    "pun_success": float,
    "pdn_avg_t":   float,
    "pun_avg_t":   float,
    "best_obj":    dict,   # fid → best ASTRAN objective found
```

Pretrain checkpoints use `"args"` instead of `"arch_args"` (older key) — all loading code handles both:
```python
a = ck.get("arch_args") or ck.get("args", {})
```

---

## Phase 1: Pretrain Commands

```bash
# Standard pretrain (generates transnet_pretrain_v6.pt style)
env/bin/python3 pretrain_transnet.py \
  --hidden     64  \
  --num-layer  3   \
  --mlp-hidden 128 \
  --epochs     200 \
  --lr         1e-3 \
  --out_ck     checkpoints/transnet_pretrain_v6.pt
```

---

## Phase 2: RL Fine-tuning Commands

### All functions (baseline)
```bash
env/bin/python3 finetune_rl_transnet.py \
  --pretrain_ck checkpoints/transnet_pretrain_v6.pt \
  --out_ck      checkpoints/transnet_rl_all_v1.pt \
  --epochs      100  \
  --num_traj    4    \
  --lr          1e-4 \
  --max_steps   30
```

### XOR-specific (best result)
```bash
env/bin/python3 finetune_rl_transnet.py \
  --func '!a*!b*c+!a*b*!c+a*!b*!c+a*b*c' \
  --pretrain_ck checkpoints/transnet_pretrain_v6.pt \
  --out_ck      checkpoints/transnet_rl_xor_v5.pt \
  --epochs      500  \
  --num_traj    8    \
  --lr          3e-5 \
  --temperature 0.8
```

### Hard functions only (pre-eval threshold)
```bash
env/bin/python3 finetune_rl_transnet.py \
  --hard_threshold 0.7 \
  --pretrain_ck checkpoints/transnet_pretrain_v6.pt \
  --out_ck      checkpoints/transnet_rl_hard_v1.pt \
  --epochs      300 --num_traj 8 --lr 3e-5
```

### Per-function pipeline (run_rl_pipeline.py handles this automatically)
```bash
env/bin/python3 run_rl_pipeline.py          # full: eval + finetune
env/bin/python3 run_rl_pipeline.py --dry_run  # identify failing functions
```

---

## Phase 3: Physical RL Commands

### Single XOR function
```bash
env/bin/python3 physical_finetune_rl_transnet.py \
  --func '!a*!b*c+!a*b*!c+a*!b*!c+a*b*c' \
  --pretrain_ck checkpoints/transnet_pretrain_v6.pt \
  --out_ck      checkpoints/transnet_physical_xor_v2.pt \
  --epochs      200 --num_traj 4 --astran_workers 4
```

### All SAT functions
```bash
env/bin/python3 physical_finetune_rl_transnet.py \
  --pretrain_ck checkpoints/transnet_pretrain_v6.pt \
  --out_ck      checkpoints/transnet_physical_all_v1.pt \
  --epochs      200 --funcs_per_batch 2
```

---

## Inference Commands

### Generate networks for a function
```bash
env/bin/python3 generate_transnet.py \
  --checkpoint checkpoints/transnet_rl_xor_v5.pt \
  --expr '!a*!b*c+!a*b*!c+a*!b*!c+a*b*c' \
  --trials 50
```

### Generate PDN+PUN SPICE + ASTRAN placement
```bash
env/bin/python3 generate_physical_transnet.py \
  --checkpoint checkpoints/transnet_rl_xor_v5.pt \
  --func '!a*!b*c+!a*b*!c+a*!b*!c+a*b*c' \
  --num_sample 20 \
  --out best_xor.sp
```

### Place an external SPICE file
```bash
env/bin/python3 place_spice.py /path/to/NET.sp CELL_NAME
```

---

## Environment

```bash
# Python environment
/Users/fch/Python/TranSGPN/env/bin/python3

# Key dependencies
# torchdrug  (MPNN, PackedGraph, DataLoader)
# torch      (≥1.12, no torch.compile on torchdrug models)
# torch_scatter
```

---

## Model Reconstruction (for inference or fine-tuning)

```python
import torch
from torchdrug import models
from transnet.graph import NODE_FEAT_DIM
from transnet.literal import EDGE_FEAT_DIM
from transnet.task import GCPNTransNet

ck   = torch.load("checkpoints/transnet_pretrain_v6.pt", map_location="cpu")
a    = ck.get("arch_args") or ck.get("args", {})
mpnn = models.MPNN(
    input_dim      = NODE_FEAT_DIM,            # 4
    hidden_dim     = a.get("hidden",    64),
    edge_input_dim = EDGE_FEAT_DIM,            # 5
    num_layer      = a.get("num_layer", 3),
    batch_norm     = False,
)
task = GCPNTransNet(mpnn, hidden_dim_mlp=a.get("mlp_hidden", 128))
task.load_state_dict(ck["model_state"], strict=False)
task.eval()
```
