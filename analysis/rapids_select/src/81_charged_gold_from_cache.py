#!/usr/bin/env python3
"""81 — FAST gold-only charged metrics from the existing cache (no retraining).
Reconstruct the exact 5-fold charged split, tag each system gold/silver, filter test
sets to gold, recompute selector + Always-GeoSP metrics. Corrects the pooled-tier error."""
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import importlib.util
spec=importlib.util.spec_from_file_location("b25c",Path(__file__).parent/"25c_baselines.py"); b25c=importlib.util.module_from_spec(spec); spec.loader.exec_module(b25c)

OUT=Path(__file__).resolve().parents[1]
chg=pd.read_csv(OUT/'data'/'p02_selector_feature_matrix.csv',low_memory=False)
chg=chg[chg['Reference'].notna()&chg['RAPIDS'].notna()].reset_index(drop=True)
tier=chg['reference_tier'].values
print(f"charged pool {len(chg)}  gold {(tier=='gold').sum()}  silver {(tier=='silver').sum()}")

I=pickle.load(open(OUT/'cache'/'v6_charged.pkl','rb')); arms=I['arms']; FIXED=I['fixed_cost']; idx={a:i for i,a in enumerate(arms)}
caches=I['caches']
kf=KFold(5,shuffle=True,random_state=0); folds=[te for _,te in kf.split(np.arange(len(chg)))]
print("fold sizes cache vs reconstructed:", [c['n'] for c in caches], [len(f) for f in folds])

def sel(c,l=800): return np.argmin(FIXED[:,None]+l*c['preds'].astype(float),0)
def metrics(c,ch,mask):
    n=c['n']; e=c['true_err'][ch,np.arange(n)][mask]; co=c['cost'][ch,np.arange(n)][mask]
    return np.nanmean(co),np.nanmean(e),np.nanmean(e>10)

def agg(want):
    S=[];G=[];D=[]
    for fi,c in enumerate(caches):
        msk=want(tier[folds[fi]])
        if msk.sum()==0: continue
        S.append(metrics(c,sel(c),msk))
        G.append(metrics(c,np.full(c['n'],idx['PBE-D3BJ_GeoSP']),msk))
        D.append(metrics(c,np.full(c['n'],idx['CREST_xTB_DFT']),msk))
    return np.array(S).mean(0),np.array(G).mean(0),np.array(D).mean(0)

print("\nSELECTOR (V6) / Always-GeoSP / Always-CREST-DFT on charged, from cache:")
for label,want in [('GOLD (DES370K)',lambda t:t=='gold'),('silver (IL174)',lambda t:t=='silver'),('POOLED (old, WRONG)',lambda t:np.ones(len(t),bool))]:
    s,g,d=agg(want)
    print(f"  [{label:<20}] V6 {s[0]:.0f}/{s[1]:.3f}/{s[2]*100:.2f}%   GeoSP {g[0]:.0f}/{g[1]:.3f}/{g[2]*100:.2f}%   CREST-DFT {d[0]:.0f}/{d[1]:.3f}/{d[2]*100:.2f}%")
