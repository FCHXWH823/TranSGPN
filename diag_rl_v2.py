import os
os.environ.setdefault("TRANSGPN_VARS","abcd")
import torch, types, time, csv
from finetune_rl_targets import build_model, derive_patterns, finetune_one
torch.set_num_threads(4)
CK="checkpoints/transnet_pretrain_3_4input_canon.pt"
# load a few hard targets
want={("P686","pdn"),("P173","pun"),("P414","pdn"),("P317","pun")}
tg={}
for r in csv.DictReader(open("results/rl_targets_canon.csv")):
    if (r["name"],r["type"]) in want: tg[(r["name"],r["type"])]=(r["expr"],int(r["t_ref"]))
def run(mode):
    task,base,arch=build_model(CK,torch.device("cpu"))
    args=types.SimpleNamespace(epochs=400,num_traj=16,temperature=1.0,max_steps=20,
        lr=1e-4,clip_eps=0.2,lambda_entropy=0.05,agent_sync_every=10,
        eval_interval=50,eval_samples=64,early_stop=True)
    # monkeypatch reward selection
    import transnet.task as T
    if mode=="old":
        orig=T.GCPNTransNet.reinforce_forward_shaped
        T.GCPNTransNet.reinforce_forward_shaped=T.GCPNTransNet.reinforce_forward
    print(f"--- reward={mode} ---")
    for (name,typ),(expr,tref) in tg.items():
        vif,on,off=derive_patterns(expr,typ)
        t0=time.time()
        _,info,_=finetune_one(task,base,arch,vif,on,off,tref,f"{name}_{typ}",args,torch.device("cpu"))
        print(f"  {name}_{typ:3} ref={tref} -> min_t={info['min_t']} reached={info['reached_ref']} stop@{info['stop_epoch']} ({time.time()-t0:.0f}s)",flush=True)
    if mode=="old":
        T.GCPNTransNet.reinforce_forward_shaped=orig
run("old")
run("new")
