#!/usr/bin/env python3
"""
66 — V10: Group-conditional (Mondrian) Conformal Risk Control.

Round-3 critique challenge: the naive-CRC LOBO failure (4/18 violations, worst =
charged in-dist benchmarks SSI_charged 82%, IHB100 18%) is a TEXTBOOK split-conformal
failure under covariate shift. A reviewer will say "use group-conditional / Mondrian
conformal conditioned on the known shift axis (charge)." This script tests exactly
that: is the failure PROCEDURAL (fixed by conditioning on charge) or FUNDAMENTAL
(persists for genuinely novel neutral chemistry)?

Method: recover per-system |charge| from the feature matrix (deterministic fold
indexing matches script 62). Mondrian-CRC: calibrate a SEPARATE threshold t_g per
group g ∈ {neutral (|q|=0), charged (|q|≥1)}; apply per group at test.
Compare naive-CRC vs group-CRC per-benchmark coverage at α.
No retraining — uses cache/v8_indist.pkl + charge recovered from features.
"""
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

import importlib.util
spec = importlib.util.spec_from_file_location("b25c", Path(__file__).parent / "25c_baselines.py")
b25c = importlib.util.module_from_spec(spec); spec.loader.exec_module(b25c)

OUT = Path(__file__).resolve().parents[1]; CACHE=OUT/'cache'
I = pickle.load(open(CACHE/'v8_indist.pkl','rb'))
arms=I['arms']; K=len(arms); Ic=I['caches']
TGRID=np.linspace(0,1,201)

# recover |charge| per in-dist system, in the SAME per-benchmark row order the cache used
ind = pd.read_csv(OUT/'data'/'selector_feature_matrix.csv', low_memory=False)
ind = ind[ind['Reference'].notna()].reset_index(drop=True)
def gc(row):
    for c in ('oracle_complex_charge','complex_charge','monA_charge','monB_charge'):
        v=row.get(c)
        if v is not None and not pd.isna(v):
            try:
                if abs(int(v))>=1: return 1
            except: pass
    return 0
ind['_q']=ind.apply(gc,axis=1)
charge_by_bench={b: ind[ind['benchmark']==b]['_q'].values for b in ind['benchmark'].unique()}
# sanity: lengths match cache
for b,f in Ic.items():
    assert len(charge_by_bench[b])==f['n'], f"len mismatch {b}: {len(charge_by_bench[b])} vs {f['n']}"
print("charge alignment OK for all 18 benchmarks")
print("charged in-dist benchmarks:", [b for b in Ic if charge_by_bench[b].mean()>0.5])

def sel(fold, t):
    pc=fold['pcat10']; ok=pc<=t
    return np.where(ok.any(0),ok.argmax(0),pc.argmin(0))
def sel_group(fold, tn, tc, q):
    pc=fold['pcat10']; n=fold['n']; out=np.empty(n,int)
    for g,t in [(0,tn),(1,tc)]:
        m=(q==g)
        if not m.any(): continue
        ok=pc[:,m]<=t
        out[np.where(m)[0]]=np.where(ok.any(0),ok.argmax(0),pc[:,m].argmin(0))
    return out
def risk(fold,ch):
    n=fold['n']; e=fold['true_err'][ch,np.arange(n)]; return np.nanmean(e>10)
def cost(fold,ch):
    n=fold['n']; return np.nanmean(fold['cost'][ch,np.arange(n)])

def crc_t_pooled(folds, alpha, mask_fn=None, B=1.0):
    """largest grid t with (n*Rhat+B)/(n+1)<=alpha, risk pooled over folds (optionally masked to a group)."""
    pcs=[];tes=[]
    for f in folds:
        m=mask_fn(f) if mask_fn else np.ones(f['n'],bool)
        if m.any(): pcs.append(f['pcat10'][:,m]); tes.append(f['true_err'][:,m])
    pc=np.concatenate(pcs,1); te=np.concatenate(tes,1); n=pc.shape[1]
    best=0.0
    for t in TGRID:
        ok=pc<=t; first=np.where(ok.any(0),ok.argmax(0),pc.argmin(0))
        Rhat=np.nanmean(te[first,np.arange(n)]>10)
        if (n*Rhat+B)/(n+1)<=alpha: best=t
        else: break
    return best

benches=sorted(Ic.keys())
for alpha in [0.05,0.07]:
    print(f"\n{'='*74}\nalpha={alpha}\n{'='*74}")
    naive_v=[]; grp_v=[]; naive_cost=[]; grp_cost=[]
    print(f"{'benchmark':<22}{'charged?':>9}{'naive risk':>12}{'group risk':>12}{'nv':>4}{'gp':>4}")
    for held in benches:
        cal=[Ic[b] for b in benches if b!=held]; test=Ic[held]; q=charge_by_bench[held]
        # naive: one pooled threshold
        tn_all=crc_t_pooled(cal,alpha)
        r_naive=risk(test,sel(test,tn_all)); naive_cost.append(cost(test,sel(test,tn_all)))
        # group-conditional: separate thresholds for neutral/charged, calibrated on same-group cal systems
        def mk_group_cal(g):
            pcs=[];tes=[]
            for b in benches:
                if b==held: continue
                qc=charge_by_bench[b]; m=(qc==g)
                if m.any(): pcs.append(Ic[b]['pcat10'][:,m]); tes.append(Ic[b]['true_err'][:,m])
            if not pcs: return None
            return np.concatenate(pcs,1),np.concatenate(tes,1)
        def crc_from(arrs,alpha,B=1.0):
            if arrs is None: return 0.0
            pc,te=arrs; n=pc.shape[1]; best=0.0
            for t in TGRID:
                ok=pc<=t; first=np.where(ok.any(0),ok.argmax(0),pc.argmin(0))
                Rhat=np.nanmean(te[first,np.arange(n)]>10)
                if (n*Rhat+B)/(n+1)<=alpha: best=t
                else: break
            return best
        tneu=crc_from(mk_group_cal(0),alpha); tchg=crc_from(mk_group_cal(1),alpha)
        r_grp=risk(test,sel_group(test,tneu,tchg,q)); grp_cost.append(cost(test,sel_group(test,tneu,tchg,q)))
        naive_v.append(r_naive>alpha+1e-9); grp_v.append(r_grp>alpha+1e-9)
        ch='YES' if q.mean()>0.5 else ''
        if r_naive>alpha+1e-9 or r_grp>alpha+1e-9 or ch:
            print(f"{held:<22}{ch:>9}{r_naive*100:>11.1f}%{r_grp*100:>11.1f}%{'X' if naive_v[-1] else '.':>4}{'X' if grp_v[-1] else '.':>4}")
    print(f"\n  naive-CRC violations: {sum(naive_v)}/18   mean cost {np.mean(naive_cost):.0f}s")
    print(f"  group-CRC violations: {sum(grp_v)}/18   mean cost {np.mean(grp_cost):.0f}s")
    print(f"  --> conditioning on charge {'FIXES' if sum(grp_v)<sum(naive_v) else 'does NOT fix'} the charged-benchmark violations")
