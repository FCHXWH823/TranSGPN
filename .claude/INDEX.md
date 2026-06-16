# TransSGPN Project — Context Index

**Project root:** `/Users/fch/Python/TranSGPN/`  
**Last updated:** 2026-06-05

---

## Context Files

| File | Covers |
|------|--------|
| [model_architecture.md](model_architecture.md) | MPNN backbone, 5 policy heads (dims, inputs, outputs), virtual node, safety masks, reachability mask, agent/model split, NLL forward |
| [graph_representation.md](graph_representation.md) | Switching graph concept, G∪C unified node layout, node/edge features, C-network (SOP compensate), literal encoding, BFS safety & coverage, graph builder functions |
| [training_phase1_pretrain.md](training_phase1_pretrain.md) | NLL supervised pretraining, dataset, loss formulation, commands, checkpoint progression, how downstream scripts load the pretrain |
| [training_combined_datasets.md](training_combined_datasets.md) | Combined 3-input+4-input training: literal indexing differences, N_VARS=4 model, cache system, GPU commands, profiling results |
| [training_phase2_rl.md](training_phase2_rl.md) | PPO with GRPO baseline, terminal-only reward, entropy bonus, K-trajectory batching, batched PPO loss, NLL regularization, hyperparameters, XOR vs all-function comparison |
| [training_phase3_physical_rl.md](training_phase3_physical_rl.md) | Physical RL via ASTRAN, PDN/PUN dual synthesis, ASTRAN objective, adaptive reward, parallel ASTRAN eval, current status |
| [astran_integration.md](astran_integration.md) | ASTRAN binary path, correct invocation (`--shell`), script format, SPICE requirements (VCC/GND, NMOS_VTL/PMOS_VTL), output parsing, CPP formula, all bugs fixed |
| [pipeline_finetuning.md](pipeline_finetuning.md) | run_rl_pipeline.py, failure criteria, dry-run results (36 fail), resumability files, current state, commands |
| [checkpoints_and_commands.md](checkpoints_and_commands.md) | All checkpoint names and meanings, checkpoint format, all training/inference commands, model reconstruction snippet |
| [aaai_paper.md](aaai_paper.md) | Title, LaTeX macros, paper structure, theorem list, figures, TODO items, build, Overleaf import, architecture name history |
| [key_decisions_and_lessons.md](key_decisions_and_lessons.md) | torch.compile (removed), threading (rejected for MPNN, used for ASTRAN), K-traj batching, terminal reward, VCC/GND, NMOS_VTL, --shell flag, GRPO rationale, `object.__setattr__` pattern |

---

## Quick Reference

### Best Checkpoints
- **Pretrain base:** `checkpoints/transnet_pretrain_v6.pt`
- **Best XOR RL:** `checkpoints/transnet_rl_xor_v5.pt`
- **All-function RL:** `checkpoints/transnet_rl_all_v1.pt` (worse on XOR)
- **Physical XOR (partial):** `checkpoints/transnet_physical_xor_v1.pt`

### Pending Work
1. Run Phase 2 pipeline: `env/bin/python3 run_rl_pipeline.py` (will re-evaluate all 242 functions then fine-tune ~36)
2. Run Phase 3 physical RL on best Phase 2 checkpoint
3. Fill `\TODO{}` markers in paper experiments section

### Critical Constants
```python
N_VARS       = 3        # variables in sweep_3input dataset
EDGE_FEAT_DIM = 5       # N_VARS + 2
NODE_FEAT_DIM = 4       # [src, snk, G, C]
hidden_dim    = 64      # MPNN hidden
num_layer     = 3       # MPNN layers
hidden_dim_mlp = 128    # policy head hidden
```

### ASTRAN Objective Weights (must match in both Python and ASTRAN script)
```python
_OBJ_CPP=3, _OBJ_MISMATCH=4, _OBJ_ROUTING=1, _OBJ_DENSITY=4, _OBJ_GAPS=2
# cellgen place 1 1 3 4 1 4 2
```
