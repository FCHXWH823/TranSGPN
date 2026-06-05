# Training Phase 1: Supervised Pretraining (NLL)

**Script:** `pretrain_transnet.py`  
**Best checkpoint:** `checkpoints/transnet_pretrain_v6.pt`

---

## Objective

Learn to imitate known correct transistor networks from the dataset via maximum likelihood (teacher forcing). The model is trained to predict each transistor (edge) in snake-pattern order given the partial graph built from all previous transistors.

Loss = sum of cross-entropy over 4 heads per step:
```
L = L_node1 + L_node2 + L_var + L_sign
```

Each head uses logits masked with the safety mask before computing NLL, so the model learns within the feasible action space.

---

## Dataset

**Path:** `dataset/sweep_3input/`  
**Structure:** 244 directories; 242 contain SAT functions with solved transistor networks.  
**Format:** Each `3_NNN/t_K/` contains `Booleans.txt` (SOP expression) and transistor network files.

**`TransistorDataset`** (in `transnet/dataset.py`) reads all networks, converts each into prefix graphs (one per transistor step), and serves them as training samples.

---

## Model Config

```bash
env/bin/python3 pretrain_transnet.py \
  --hidden     64    \   # MPNN hidden_dim
  --num-layer  3     \   # MPNN message-passing layers
  --mlp-hidden 128   \   # policy head hidden width
  --epochs     200   \
  --lr         1e-3
```

Checkpoint keys saved:
```python
{
  "model_state": ...,
  "args": {"hidden": 64, "num_layer": 3, "mlp_hidden": 128},
  ...
}
```

The `args` dict is read back by all downstream scripts as `arch_args` to reconstruct the identical MPNN architecture.

---

## Checkpoint Progression

| Checkpoint | Notes |
|-----------|-------|
| `transnet_pretrain.pt` | First successful pretrain |
| `transnet_pretrain_v2.pt` | Longer training |
| `transnet_pretrain_v3.pt` | Hyperparameter search |
| `transnet_pretrain_v4.pt` | |
| `transnet_pretrain_v5.pt` | |
| `transnet_pretrain_v6.pt` | **Best — used as base for all RL** |

---

## NLL Forward Details

In `task.py:111` (`MLE_forward`):

1. MPNN encodes the partial G∪C graph → `node_feat [total_N, h]`, `graph_feat [B, 2h]`
2. Virtual new-node embedding appended: `ext_feat [total_N+B, h]`
3. **Node-1 loss:** softmax over extended G-nodes (masked by G-membership), NLL at GT node1
4. **Node-2 loss:** softmax conditioned on n1 embedding, masked by safety + n2_safety_mask, NLL at GT node2
5. **Var loss:** softmax over N_VARS=3, masked by safety_mask.any(dim=2), NLL at GT var
6. **Sign loss:** softmax over 2, masked by safety_mask[GT_var], NLL at GT sign

**Key detail:** GT action is force-set as valid in the safety mask before loss computation (guards against floating-point / dataset edge cases):
```python
safety_mask[arange_B, var_lb, neg_lb] = True   # task.py:194
```

---

## How Downstream Scripts Load Pretrain

```python
ck = torch.load(ck_path, map_location="cpu")
a  = ck.get("arch_args") or ck.get("args", {})   # arch_args key added by RL scripts
mpnn = models.MPNN(
    input_dim      = NODE_FEAT_DIM,           # 4
    hidden_dim     = a.get("hidden",    64),
    edge_input_dim = EDGE_FEAT_DIM,           # 5
    num_layer      = a.get("num_layer", 3),
    batch_norm     = False,
)
task = GCPNTransNet(mpnn, hidden_dim_mlp=a.get("mlp_hidden", 128))
task.load_state_dict(ck["model_state"])
```
