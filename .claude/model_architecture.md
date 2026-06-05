# Model Architecture: SGPN (Switching Graph Policy Network)

**System name:** TransSGPN (`\method` in paper)  
**Core class:** `GCPNTransNet` in `transnet/task.py:69`

---

## Backbone: MPNN

From `torchdrug.models.MPNN`. Configured in `pretrain_transnet.py:49` and reconstructed in all finetune scripts via `build_model_from_checkpoint`.

```python
mpnn = models.MPNN(
    input_dim      = 4,    # NODE_FEAT_DIM: [is_src, is_snk, is_G, is_C]
    hidden_dim     = 64,   # --hidden default; node_output_dim = h = 64
    edge_input_dim = 5,    # EDGE_FEAT_DIM = N_VARS + 2 = 3 + 2
    num_layer      = 3,    # --num-layer default; message-passing rounds
    batch_norm     = False,
)
# node_output_dim = 64  (h)
# output_dim      = 128 (2h, graph-level readout via mean pooling + MLP)
```

**MPNN output:**  
- `node_feature`: `[total_N, h=64]` — per-node embeddings after 3 rounds of message passing  
- `graph_feature`: `[B, 2h=128]` — graph-level embedding (mean pool → linear)

---

## Virtual Node Embedding

```python
self.new_node_emb = nn.Parameter(torch.empty(h))   # task.py:86
nn.init.normal_(self.new_node_emb, std=0.1)
```

At each generation step the agent can add a **new internal node** (creating an intermediate circuit node). This learnable embedding represents that virtual "new node" candidate in the node selection heads.

**Extended node space:** `ext_feat = [node_feat; new_node_emb]` → shape `[total_N + 1, h]`

---

## Five Policy Heads

All are `torchdrug.layers.MultiLayerPerceptron` with `activation="tanh"`. Defined in `task.py:90-93`.

```python
MLP = layers.MultiLayerPerceptron
self.mlp_node1 = MLP(h + gh,     [hidden_dim_mlp, 1],      activation="tanh")
self.mlp_node2 = MLP(h + h + gh, [hidden_dim_mlp, 1],      activation="tanh")
self.mlp_var   = MLP(h + h,      [hidden_dim_mlp, N_VARS], activation="tanh")
self.mlp_sign  = MLP(h + h,      [hidden_dim_mlp, 2],      activation="tanh")
# mlp_stop is implicit (not used; stop is when safety mask is empty)
```

With `h=64`, `gh=128`, `hidden_dim_mlp=128`:

| Head | Input shape | Output | Decision |
|------|------------|--------|----------|
| `mlp_node1` | `[total_N+1, 192]` = `[ext_feat ‖ graph_feat]` | scalar per node | Source node of new transistor |
| `mlp_node2` | `[total_N+1, 256]` = `[n1_feat ‖ ext_feat ‖ graph_feat]` | scalar per node | Destination node |
| `mlp_var` | `[B, 128]` = `[n1_feat ‖ ext_feat[n2]]` | `[B, N_VARS=3]` | Gate variable index |
| `mlp_sign` | `[B, 128]` = `[n1_feat ‖ ext_feat[n2]]` | `[B, 2]` | Polarity: 0=positive, 1=negated |

**Why graph_feat is NOT used in mlp_var / mlp_sign:** The variable and polarity are local decisions given the edge endpoints — the graph-level summary adds noise without useful signal once the endpoint embeddings are known.

---

## Safety Masks

Computed by BFS in `transnet/literal.py` before each head's logits are evaluated.

**Node-2 mask** (`n2_has_safe`): For each candidate node-2, check if ∃ (var, polarity) such that adding that transistor doesn't create any path from SOURCE→SINK under an off-pattern. Done in `_generate_once` / `_generate_K_trajs`.

**Var/sign mask** (`safety_m [N_VARS, 2]`): After fixing (node1, node2), enumerate all (var, polarity) pairs; keep only those safe under all off-patterns.

```python
# task.py:485-490
safety_m = torch.zeros(N_VARS, 2, dtype=torch.bool, device=device)
for vi in range(N_VARS):
    for ni in range(2):
        if check_safety(g_tran + [(node1, node2_c, vi, ni)], new_g_num, off_patterns):
            safety_m[vi, ni] = True
```

Masks fill invalid logits with `-1e9` before softmax. **Proven never to remove all valid solutions** (Completeness Theorem in paper §3.1).

---

## Reachability Mask (Node-1)

`_reachability_mask` in `task.py:898`. Only nodes reachable from SOURCE or SINK under at least one uncovered on-pattern are eligible as node-1. This prunes dead-end node choices that cannot possibly cover any remaining required path.

```python
n1_mask = torch.zeros(total_N + 1, dtype=torch.bool, device=device)
n1_mask[:g_num] = reach   # [g_num] bool from BFS
```

SOURCE (0) and SINK (1) are always `True`.

---

## Agent / Model Split for Off-Policy RL

```python
object.__setattr__(self, '_agent', None)   # task.py:97
```

`_agent` is a **frozen deepcopy** of the model used for trajectory generation. It is NOT registered as an `nn.Module` submodule (uses `object.__setattr__` to bypass PyTorch's module registration), so its weights don't appear in `state_dict()` and don't receive gradients.

`sync_agent()` refreshes the frozen copy after every `--agent_sync_every` PPO steps.

**`_best_t` dict** (`task.py:98`): tracks the best (minimum) transistor count found so far per function ID. Used by GRPO reward to automatically tighten the target as training improves.

---

## NLL Forward (Pretraining)

`MLE_forward` in `task.py:111`. Processes a batch of labeled training graphs from `TransistorDataset`. Computes cross-entropy loss over all 4 action heads (node1, node2, var, sign) simultaneously. Safety masks are applied to logits before computing NLL.

Total loss: `loss_node1 + loss_node2 + loss_var + loss_sign`

Also logged: `acc/node1`, `acc/node2` (fraction of greedy-argmax correct).
