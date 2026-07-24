import os
os.environ.setdefault("TRANSGPN_VARS","abcdef"); os.environ.setdefault("TRANSGPN_MAX_G_NODES","70")
import torch
from transnet.literal import extract_vars, on_off_split, parse_sop_expr
from transnet.canonical import canonicalize_expr
from generate_all_transnets import load_model
torch.set_num_threads(4)
task,ep=load_model("checkpoints/transnet_pretrain_6input_canon.pt", torch.device("cpu"))
print("epoch",ep)
def run(expr, ms=40, tr=256):
    ce,rv,_=canonicalize_expr(expr); cvif=extract_vars(ce)
    on,off=on_off_split(parse_sop_expr(ce,cvif))
    with torch.no_grad():
        res=task.generate(cvif,on,off,num_sample=tr,max_steps=ms,temperature=0.8,verbose=0)
    c=[r for r in res if r["correct"]]; best=min((len(r["g_tran"]) for r in c),default=None)
    print(f"  {expr:28} vars={len(rv)} succ={len(c)/tr:.3f} best={best}")
print("simple sanity:")
for e in ["a","a+b","a*b","a+b+c","a*b*c","a+b+c+d+e+f","a*b*c*d*e*f"]:
    run(e)
print("NSP samples:")
for e in ["a*b+a*c+a*d+b*c*d","a*b","a*b+c*d"]:
    run(e)
