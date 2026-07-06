#!/usr/bin/env python3
"""68 — Paper-level in-dist error bars: bootstrap CI over the 18 LOBO benchmarks.
LOBO is deterministic (no shuffle), so rigor = bootstrap the per-benchmark metrics."""
import pickle
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parents[1]; CACHE=OUT/'cache'
I = pickle.load(open(CACHE/'v6_indist.pkl','rb'))
arms=I['arms']; FIXED=I['fixed_cost']; idx={a:i for i,a in enumerate(arms)}
folds=list(I['caches'].items())

def router(cache, lam=800):
    return np.argmin(FIXED[:,None]+lam*cache['preds'].astype(float),0)
def always(cache,a): return np.full(cache['n'],idx[a],int)
def fold_metric(cache,ch):
    n=cache['n']; e=cache['true_err'][ch,np.arange(n)]; c=cache['cost'][ch,np.arange(n)]
    return np.nanmean(c),np.nanmean(e),np.nanmean(e>10)

# per-benchmark metrics for router + baselines
def per_bench(sel):
    return np.array([fold_metric(c,sel(c)) for _,c in folds])  # [18,3] cost,err,cat

R=per_bench(lambda c: router(c))
G=per_bench(lambda c: always(c,'PBE-D3BJ_GeoSP'))

# bootstrap over the 18 benchmarks (resample benchmarks with replacement)
rng=np.random.default_rng(0)
def boot_ci(M, B=5000):
    n=len(M); means=np.empty((B,3))
    for b in range(B):
        idxs=rng.integers(0,n,n); means[b]=M[idxs].mean(0)
    lo=np.percentile(means,2.5,0); hi=np.percentile(means,97.5,0); mu=M.mean(0)
    return mu,lo,hi

names=['cost(s)','err(kcal/mol)','catastrophic']
for label,M in [('V11/V6 router (λ=800)',R),('Always-GeoSP',G)]:
    mu,lo,hi=boot_ci(M)
    print(f"\n{label}  (macro-avg over 18 LOBO benchmarks, 95% bootstrap CI):")
    for k in range(3):
        v=mu[k]*(100 if k==2 else 1); l=lo[k]*(100 if k==2 else 1); h=hi[k]*(100 if k==2 else 1)
        unit='%' if k==2 else ''
        print(f"    {names[k]:<16} {v:7.3f}{unit}  [{l:.3f}, {h:.3f}]{unit}")

# paired bootstrap: router - GeoSP difference (does router robustly beat GeoSP?)
D=R-G  # [18,3]
print("\nRouter − GeoSP (paired, per-benchmark differences; negative = router better):")
mu,lo,hi=boot_ci(D)
for k in range(3):
    sig = 'SIGNIFICANT' if (hi[k]<0 or lo[k]>0) else 'n.s.'
    v=mu[k]*(100 if k==2 else 1); l=lo[k]*(100 if k==2 else 1); h=hi[k]*(100 if k==2 else 1)
    unit='%' if k==2 else ''
    print(f"    Δ{names[k]:<15} {v:+7.3f}{unit}  95%CI [{l:+.3f}, {h:+.3f}]{unit}  {sig}")

# per-fold win rate
win_cost=(R[:,0]<G[:,0]).mean(); win_err=(R[:,1]<=G[:,1]).mean(); win_cat=(R[:,2]<=G[:,2]).mean()
alldom=((R[:,0]<G[:,0])&(R[:,1]<=G[:,1])&(R[:,2]<=G[:,2])).mean()
print(f"\nPer-benchmark win rate vs GeoSP: cheaper {win_cost*100:.0f}%, err≤ {win_err*100:.0f}%, cat≤ {win_cat*100:.0f}%, ALL-3 {alldom*100:.0f}%")
