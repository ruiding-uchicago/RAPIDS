#!/usr/bin/env python3
"""
80 — CORRECTED charged evaluation on DES370K GOLD tier ONLY.
Earlier charged numbers pooled DES370K(gold) + IL174(silver); IL174's silver reference
(DLPNO correlation-only, different energy scale) gives ~50 MAE for ALL methods (README
warned "absolute MAE not comparable"), inflating pooled charged MAE 3x. Absolute MAE must
be reported on gold only. IL174 handled separately (relative/catastrophe-avoidance only).
V6 recipe, charged-aware training, 5-fold CV over gold charged, multi-seed.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
import importlib.util
spec=importlib.util.spec_from_file_location("b25c",Path(__file__).parent/"25c_baselines.py"); b25c=importlib.util.module_from_spec(spec); spec.loader.exec_module(b25c)

OUT=Path(__file__).resolve().parents[1]; RES=OUT/'results'
m=json.load(open(OUT/'models'/'rapids_select_v5_final'/'manifest.json')); feature_cols=m['feature_cols']; arm_costs=m['arm_costs']; ARMS=m['arms']
AE=b25c.ARM_ENERGY_COL; arms=sorted(ARMS,key=lambda a:arm_costs[a]); FIXED=np.array([arm_costs[a] for a in arms]); K=len(arms); ERR=50.0; LAM=800

ind=pd.read_csv(OUT/'data'/'selector_feature_matrix.csv',low_memory=False); ind=ind[ind['Reference'].notna()].reset_index(drop=True)
chg=pd.read_csv(OUT/'data'/'p02_selector_feature_matrix.csv',low_memory=False)
gold=chg[(chg['reference_tier']=='gold')&(chg['Reference'].notna())&(chg['RAPIDS'].notna())].reset_index(drop=True)
print(f"in-dist {len(ind)}  DES370K-gold charged {len(gold)}", flush=True)
Xind=b25c.encode_features(ind,feature_cols).astype(np.float32); Xg=b25c.encode_features(gold,feature_cols).astype(np.float32)
refi=ind['Reference'].values.astype(float); refg=gold['Reference'].values.astype(float)
te_g=np.array([np.minimum(np.abs(gold[AE[a]].values.astype(float)-refg),ERR) for a in arms])
def earm(df,a,r): return np.minimum(np.abs(df[AE[a]].values.astype(float)-r),ERR)

# Always-arm baselines on gold (no training)
print("\nGold Always-arm (cost / MAE / cat>10):")
for a in arms:
    j=arms.index(a); e=te_g[j]; print(f"  Always-{a:<16} {FIXED[j]:>5.0f} / {np.nanmean(e):.3f} / {np.nanmean(e>10)*100:.2f}%")
ba=np.nanargmin(te_g,0); print(f"  {'Oracle':<23} {np.nanmean(FIXED[ba]):>5.0f} / {np.nanmean(te_g[ba,np.arange(len(gold))]):.3f} / {np.nanmean(te_g[ba,np.arange(len(gold))]>10)*100:.2f}%")

# selector 5-fold CV, multi-seed, charged-aware (train in-dist + gold-charged-train)
def run(seed):
    kf=KFold(5,shuffle=True,random_state=seed); fold=[]
    for tr_rel,te_rel in kf.split(np.arange(len(gold))):
        Xtr=np.vstack([Xind,Xg[tr_rel]]); preds=np.full((K,len(te_rel)),ERR)
        for j,a in enumerate(arms):
            ytr=np.concatenate([earm(ind,a,refi),earm(gold.iloc[tr_rel],a,refg[tr_rel])]); mk=~np.isnan(ytr)
            mod=xgb.XGBRegressor(n_estimators=120,max_depth=4,learning_rate=0.1,subsample=0.85,colsample_bytree=0.7,random_state=0,objective='reg:absoluteerror',n_jobs=-1,verbosity=0)
            mod.fit(Xtr[mk],ytr[mk]); preds[j]=mod.predict(Xg[te_rel])
        ch=np.argmin(FIXED[:,None]+LAM*preds.astype(float),0)
        e=te_g[ch,te_rel]; fold.append((np.nanmean(FIXED[ch]),np.nanmean(e),np.nanmean(e>10)))
    return np.array(fold).mean(0)
P=np.array([run(s) for s in range(5)])
print(f"\nSELECTOR (V6) on DES370K-gold charged, 5 seeds:")
print(f"  cost {P[:,0].mean():.0f}±{P[:,0].std():.0f}  MAE {P[:,1].mean():.3f}±{P[:,1].std():.3f}  cat {P[:,2].mean()*100:.2f}±{P[:,2].std()*100:.2f}%")
json.dump({'selector_gold':{'cost':float(P[:,0].mean()),'mae':float(P[:,1].mean()),'cat':float(P[:,2].mean()),
           'mae_std':float(P[:,1].std())}},open(RES/'charged_gold_eval.json','w'),indent=2)
print("Saved -> results/charged_gold_eval.json")
