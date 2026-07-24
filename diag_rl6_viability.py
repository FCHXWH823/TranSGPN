import os
os.environ.setdefault("TRANSGPN_VARS","abcdef"); os.environ.setdefault("TRANSGPN_MAX_G_NODES","70")
import torch, types, time
from finetune_rl_targets import build_model, derive_patterns, finetune_one
torch.set_num_threads(4)
task, base, arch = build_model("checkpoints/transnet_pretrain_6input_canon.pt", torch.device("cpu"))
args = types.SimpleNamespace(epochs=400, num_traj=16, temperature=1.0, max_steps=40,
                             lr=1e-4, clip_eps=0.2, lambda_entropy=0.02,
                             agent_sync_every=10, eval_interval=50, eval_samples=64,
                             early_stop=True)
tests = [("ab","a*b",2),("abc","a*b*c",3),("nsp1","a*b+a*c+a*d+b*c*d",5)]
for fid,expr,tref in tests:
    vif,on,off = derive_patterns(expr,"pdn")
    t0=time.time()
    _,info,_ = finetune_one(task, base, arch, vif, on, off, tref, fid, args, torch.device("cpu"))
    print(f"{fid:6} expr={expr:22} t_ref={tref} -> success={info['success']:.2f} "
          f"min_t={info['min_t']} reached_ref={info['reached_ref']} stop@{info['stop_epoch']} "
          f"({time.time()-t0:.0f}s)", flush=True)
