import argparse, csv, os, time
os.environ.setdefault("TRANSGPN_VARS","abcdef")
import torch
from finetune_rl_targets import build_model, derive_patterns
from generate_all_transnets import eval_network
p=argparse.ArgumentParser()
p.add_argument("--targets"); p.add_argument("--pretrain-ck"); p.add_argument("--out")
p.add_argument("--n",type=int,default=800); p.add_argument("--temp",type=float,default=0.8)
a=p.parse_args()
torch.set_num_threads(4)
dev="cuda" if torch.cuda.is_available() else "cpu"
task,_,_=build_model(a.pretrain_ck,dev); task.eval()
print(f"best-of-{a.n} temp={a.temp} dev={dev}",flush=True)
rows=list(csv.DictReader(open(a.targets)))
w=csv.DictWriter(open(a.out,"w",newline=""),fieldnames=["name","type","t_ref","min_t","reached_ref","sec"]); w.writeheader()
for r in rows:
    e=(r.get("expr") or "").strip()
    if not e: continue
    tref=int(r["t_ref"]); vif,on,off=derive_patterns(e,r["type"])
    t0=time.time()
    net,succ,best=eval_network(task,vif,on,off,a.n,20,a.temp)
    dt=time.time()-t0
    reached = best is not None and best<=tref
    w.writerow({"name":r["name"],"type":r["type"],"t_ref":tref,"min_t":(best if best else ""),"reached_ref":reached,"sec":f"{dt:.1f}"})
    print(f"  {r['name']}_{r['type']}: min_t={best} ref={tref} reached={reached} succ={succ:.2f} ({dt:.1f}s)",flush=True)
print("BESTOFN_DONE",flush=True)
