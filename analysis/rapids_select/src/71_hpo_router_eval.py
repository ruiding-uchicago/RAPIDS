#!/usr/bin/env python3
"""71 — Does HPO's +5.6% LOBO-MAE / +0.066 AUC translate to ROUTING? HPO router vs V6,
dual battlefield + paired bootstrap CI over the 18 in-dist benchmarks (robustness, not aggregate noise)."""
import pickle
from pathlib import Path
import numpy as np

OUT=Path(__file__).resolve().parents[1]; CACHE=OUT/'cache'
V6I=pickle.load(open(CACHE/'v6_indist.pkl','rb')); V6C=pickle.load(open(CACHE/'v6_charged.pkl','rb'))
HI=pickle.load(open(CACHE/'vhpo_indist.pkl','rb')); HC=pickle.load(open(CACHE/'vhpo_charged.pkl','rb'))
arms=V6I['arms']; FIXED=V6I['fixed_cost']; idx={a:i for i,a in enumerate(arms)}

def lam(cache,l): return np.argmin(FIXED[:,None]+l*cache['preds'].astype(float),0)
def fm(cache,ch):
    n=cache['n']; e=cache['true_err'][ch,np.arange(n)]; c=cache['cost'][ch,np.arange(n)]
    return np.nanmean(c),np.nanmean(e),np.nanmean(e>10)
def agg_i(cache_dict,l): return np.array([fm(c,lam(c,l)) for c in cache_dict.values()]).mean(0)
def agg_c(cache_list,l): return np.array([fm(c,lam(c,l)) for c in cache_list]).mean(0)

print("=== lambda sweep: V6 vs HPO, both battlefields ===")
print(f"{'lam':>5} | {'V6 in-dist':^22} {'HPO in-dist':^22} | {'V6 charged':^20} {'HPO charged':^20}")
for l in [200,400,800,1500,3000]:
    v6i=agg_i(V6I['caches'],l); hi=agg_i(HI['caches'],l); v6c=agg_c(V6C['caches'],l); hc=agg_c(HC['caches'],l)
    print(f"{l:>5} | {v6i[0]:>5.0f}/{v6i[1]:.3f}/{v6i[2]*100:4.1f}% {hi[0]:>5.0f}/{hi[1]:.3f}/{hi[2]*100:4.1f}% | "
          f"{v6c[0]:>5.0f}/{v6c[1]:.3f}/{v6c[2]*100:4.1f}% {hc[0]:>5.0f}/{hc[1]:.3f}/{hc[2]*100:4.1f}%")

# paired bootstrap over 18 benchmarks at lam=800: HPO - V6
L=800
benches=list(V6I['caches'].keys())
R_v6=np.array([fm(V6I['caches'][b],lam(V6I['caches'][b],L)) for b in benches])
R_hp=np.array([fm(HI['caches'][b],lam(HI['caches'][b],L)) for b in benches])
D=R_hp-R_v6  # [18,3]
rng=np.random.default_rng(0)
def boot(M,B=5000):
    n=len(M); out=np.empty((B,3))
    for b in range(B): out[b]=M[rng.integers(0,n,n)].mean(0)
    return M.mean(0),np.percentile(out,2.5,0),np.percentile(out,97.5,0)
mu,lo,hi=boot(D)
names=['cost','err','cat']
print(f"\n=== HPO − V6 router, paired over 18 benchmarks @lam={L} (negative=HPO better) ===")
for k in range(3):
    sig='SIGNIFICANT' if (hi[k]<0 or lo[k]>0) else 'n.s.'
    sc=100 if k==2 else 1; u='%' if k==2 else ''
    print(f"  Δ{names[k]:<5} {mu[k]*sc:+7.3f}{u}  95%CI [{lo[k]*sc:+.3f},{hi[k]*sc:+.3f}]{u}  {sig}")
win=((R_hp[:,1]<R_v6[:,1])).mean(); winc=((R_hp[:,2]<=R_v6[:,2])).mean()
print(f"  per-benchmark: HPO err<V6 {win*100:.0f}%, HPO cat<=V6 {winc*100:.0f}%")

# charged paired over 5 folds
Rc6=np.array([fm(c,lam(c,L)) for c in V6C['caches']]); Rch=np.array([fm(c,lam(c,L)) for c in HC['caches']])
print(f"\ncharged @lam={L}: V6 {Rc6.mean(0)[0]:.0f}/{Rc6.mean(0)[1]:.3f}/{Rc6.mean(0)[2]*100:.2f}%  "
      f"HPO {Rch.mean(0)[0]:.0f}/{Rch.mean(0)[1]:.3f}/{Rch.mean(0)[2]*100:.2f}%")
