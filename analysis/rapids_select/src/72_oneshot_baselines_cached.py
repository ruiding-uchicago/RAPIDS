#!/usr/bin/env python3
"""
72 — One-shot literature baselines (same setting as our selector), on cached preds.

Implements published one-shot selection paradigms directly on the 5-arm caches
(v8_*.pkl: preds=arm err regressors, pcat10/pcat5, true_err, cost):

  - Cost-blind / SATzilla-core (Xu-Hutter-Hoos-Leyton-Brown 2008; Rice 1976):
      arm* = argmin_a pred_err_a(x)   [per-instance empirical-hardness selection, NO cost]
  - FrugalML-style budget cascade (Chen-Zaharia-Zou, NeurIPS 2020):
      cheapest-first; escalate arm a→next when its predicted quality is low (pcat10>t);
      sweep t = budget knob → cost-accuracy curve.
  - Learning-to-defer (Mozannar-Sontag 2020; Cortes-DeSalvo-Mohri 2016):
      predict with RAPIDS unless a defer signal fires (pcat10_RAPIDS>t), then defer to the
      argmin predicted-error expensive arm. One-shot 2-stage.
  - Global-best-arm / majority-oracle (meta static): the single arm most often oracle-best
      on the training folds, applied to all (a per-instance-selection-free meta baseline).

Reference rows: V6 router, Oracle, Always-arms. Both battlefields + paired bootstrap vs V6.
"""
import pickle
from pathlib import Path
import numpy as np

OUT=Path(__file__).resolve().parents[1]; CACHE=OUT/'cache'
I=pickle.load(open(CACHE/'v8_indist.pkl','rb')); C=pickle.load(open(CACHE/'v8_charged.pkl','rb'))
arms=I['arms']; FIXED=I['fixed_cost']; idx={a:i for i,a in enumerate(arms)}; K=len(arms)
Ic=list(I['caches'].values()); Cc=C['caches']

def fm(cache,ch):
    n=cache['n']; e=cache['true_err'][ch,np.arange(n)]; c=cache['cost'][ch,np.arange(n)]
    return np.nanmean(c),np.nanmean(e),np.nanmean(e>10)
def agg(cs,sel): return np.array([fm(c,sel(c)) for c in cs]).mean(0)

# --- selection rules (one-shot, from cache) ---
def v6(l=800): return lambda c: np.argmin(FIXED[:,None]+l*c['preds'].astype(float),0)
def costblind(): return lambda c: np.argmin(c['preds'].astype(float),0)   # SATzilla core
def always(a): return lambda c: np.full(c['n'],idx[a],int)
def oracle(): return lambda c: np.nanargmin(c['true_err'],0)

def frugalml(t):
    """cheapest-first; use arm j if pcat10_j<=t else try next; last arm = fallback."""
    def f(c):
        pc=c['pcat10']; n=c['n']; ok=pc<=t
        return np.where(ok.any(0),ok.argmax(0),pc.argmin(0))
    return f

def learn_to_defer(t):
    """RAPIDS unless defer (pcat10_RAPIDS>t); then defer to argmin pred_err among arms>=GeoSP."""
    rj=idx['RAPIDS']; exp=[j for j,a in enumerate(arms) if FIXED[j]>=FIXED[idx['PBE-D3BJ_GeoSP']]]
    def f(c):
        n=c['n']; out=np.full(n,rj,int); defer=c['pcat10'][rj]>t
        if defer.any():
            sub=c['preds'][np.ix_(exp,np.where(defer)[0])]
            out[defer]=np.array(exp)[sub.argmin(0)]
        return out
    return f

def global_best_arm(train_folds):
    """meta static: arm most often oracle-best across training folds."""
    counts=np.zeros(K)
    for c in train_folds:
        ba=np.nanargmin(c['true_err'],0)
        for j in range(K): counts[j]+=(ba==j).sum()
    best=int(counts.argmax())
    return lambda c: np.full(c['n'],best,int), arms[best]

print("="*88)
print(f"{'baseline':<34}{'IN-DIST (18 LOBO)':^26}{'CHARGED (5 CV)':^26}")
print(f"{'':<34}{'cost':>8}{'err':>9}{'cat%':>7}{'':2}{'cost':>8}{'err':>9}{'cat%':>7}")
print("="*88)
def row(name,sel_i,sel_c=None):
    iv=agg(Ic,sel_i); cv=agg(Cc,sel_c or sel_i)
    print(f"{name:<34}{iv[0]:>8.0f}{iv[1]:>9.3f}{iv[2]*100:>7.2f}  {cv[0]:>8.0f}{cv[1]:>9.3f}{cv[2]*100:>7.2f}")
    return iv,cv

row("V6 router (ours, λ=800)", v6())
row("Cost-blind / SATzilla (argmin ê)", costblind())
# FrugalML: pick the budget point closest to V6's in-dist cost (~375)
print("  -- FrugalML-style budget sweep (t) --")
for t in [0.05,0.1,0.2,0.3,0.5]: row(f"   FrugalML t={t}", frugalml(t))
print("  -- Learning-to-defer sweep (t) --")
for t in [0.1,0.2,0.3,0.5]: row(f"   L2D t={t}", learn_to_defer(t))
gba_i,name_i=global_best_arm(Ic); gba_c,_=global_best_arm(Cc)
row(f"Global-best-arm ({name_i}, meta)", gba_i, gba_c)
print("  -- references --")
row("Oracle (per-system best)", oracle())
for a in arms: row(f"Always-{a}", always(a))

# paired bootstrap: cost-blind vs V6 (does the cost term matter?) on in-dist
benches=list(I['caches'].keys())
Rv=np.array([fm(I['caches'][b],v6()(I['caches'][b])) for b in benches])
Rcb=np.array([fm(I['caches'][b],costblind()(I['caches'][b])) for b in benches])
rng=np.random.default_rng(0)
def boot(M,B=5000):
    n=len(M); o=np.empty((B,3))
    for b in range(B): o[b]=M[rng.integers(0,n,n)].mean(0)
    return M.mean(0),np.percentile(o,2.5,0),np.percentile(o,97.5,0)
D=Rcb-Rv; mu,lo,hi=boot(D)
print("\n=== Cost-blind − V6 (paired 18 benchmarks; shows the cost term's effect) ===")
for k,nm in enumerate(['cost','err','cat']):
    sc=100 if k==2 else 1; u='%' if k==2 else ''; sig='SIG' if (hi[k]<0 or lo[k]>0) else 'n.s.'
    print(f"  Δ{nm:<5}{mu[k]*sc:+8.2f}{u}  95%CI[{lo[k]*sc:+.2f},{hi[k]*sc:+.2f}]{u}  {sig}")
