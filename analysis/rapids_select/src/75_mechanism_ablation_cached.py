#!/usr/bin/env python3
"""75 — Mechanism ablations (cache-based, fast): each row toggles ONE V6 design choice."""
import json, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import importlib.util
spec = importlib.util.spec_from_file_location("b25c", Path(__file__).parent / "25c_baselines.py")
b25c = importlib.util.module_from_spec(spec); spec.loader.exec_module(b25c)

OUT=Path(__file__).resolve().parents[1]; CACHE=OUT/'cache'
I=pickle.load(open(CACHE/'v8_indist.pkl','rb')); C=pickle.load(open(CACHE/'v8_charged.pkl','rb'))
arms=I['arms']; FIXED=I['fixed_cost']; idx={a:i for i,a in enumerate(arms)}; K=len(arms)
feature_cols=json.load(open(OUT/'models'/'rapids_select_v5_final'/'manifest.json'))['feature_cols']

# recover per-system charge for in-dist folds
ind=pd.read_csv(OUT/'data'/'selector_feature_matrix.csv',low_memory=False); ind=ind[ind['Reference'].notna()].reset_index(drop=True)
def gc(r):
    for c in ('oracle_complex_charge','complex_charge','monA_charge','monB_charge'):
        v=r.get(c)
        if v is not None and not pd.isna(v):
            try:
                if abs(int(v))>=1: return 1
            except: pass
    return 0
ind['_q']=ind.apply(gc,axis=1)
q_by_bench={b:ind[ind['benchmark']==b]['_q'].values for b in ind['benchmark'].unique()}
benches=list(I['caches'].keys())

def fm(cache,ch):
    n=cache['n']; e=cache['true_err'][ch,np.arange(n)]; c=cache['cost'][ch,np.arange(n)]
    return np.nanmean(c),np.nanmean(e),np.nanmean(e>10)
def agg_i(sel): return np.array([fm(I['caches'][b],sel(I['caches'][b],b)) for b in benches]).mean(0)
def agg_c(sel): return np.array([fm(c,sel(c,None)) for c in C['caches']]).mean(0)

def v6(l=800): return lambda c,b=None: np.argmin(FIXED[:,None]+l*c['preds'].astype(float),0)
def costblind(): return lambda c,b=None: np.argmin(c['preds'].astype(float),0)
def v8(l=800,M=2000): return lambda c,b=None: np.argmin(FIXED[:,None]+l*c['preds'].astype(float)+M*c['pcat10'].astype(float),0)
def v6_shortcircuit(l=800):
    def f(c,b):
        ch=np.argmin(FIXED[:,None]+l*c['preds'].astype(float),0)
        if b is not None:  # in-dist: use recovered charge
            q=q_by_bench[b]; ch[q>0]=idx['PBE-D3BJ_GeoSP']
        else:              # charged CV: all |q|>=1
            ch[:]=idx['PBE-D3BJ_GeoSP']
        return ch
    return f

print(f"{'mechanism variant':<32}{'IN-DIST':^22}{'CHARGED':^22}")
print(f"{'':<32}{'cost/err/cat':^22}{'cost/err/cat':^22}")
print("-"*76)
def row(name,sel_i,sel_c):
    iv=agg_i(sel_i); cv=agg_c(sel_c)
    print(f"{name:<32}{iv[0]:>7.0f}/{iv[1]:.3f}/{iv[2]*100:>4.1f}%   {cv[0]:>7.0f}/{cv[1]:.3f}/{cv[2]*100:>4.1f}%")

row("V6 (cost + λ·err)  [reference]", v6(), v6())
row("  drop cost term (SATzilla)", costblind(), costblind())
row("  + re-add charge short-circuit", v6_shortcircuit(), v6_shortcircuit())
row("  + catastrophe term (V8)", v8(), v8())
print("  -- λ sensitivity --")
for l in [200,400,800,1500,3000]:
    row(f"  λ={l}", v6(l), v6(l))
