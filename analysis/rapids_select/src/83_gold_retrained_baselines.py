#!/usr/bin/env python3
"""83 — Gold-only (DES370K) charged for the RETRAINED baselines/ablation that can't be
cache-filtered: (1) ALORS, (2) multiclass selector, (3) drop-charged-aware ablation.
Closes the last charged-tier-correction loose ends."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import importlib.util
spec=importlib.util.spec_from_file_location("b25c",Path(__file__).parent/"25c_baselines.py"); b25c=importlib.util.module_from_spec(spec); spec.loader.exec_module(b25c)

OUT=Path(__file__).resolve().parents[1]; RES=OUT/'results'
m=json.load(open(OUT/'models'/'rapids_select_v5_final'/'manifest.json')); feature_cols=m['feature_cols']; arm_costs=m['arm_costs']; ARMS=m['arms']
AE=b25c.ARM_ENERGY_COL; arms=sorted(ARMS,key=lambda a:arm_costs[a]); FIXED=np.array([arm_costs[a] for a in arms]); K=len(arms); ERR=50.0; LAM=800

ind=pd.read_csv(OUT/'data'/'selector_feature_matrix.csv',low_memory=False); ind=ind[ind['Reference'].notna()].reset_index(drop=True)
chg=pd.read_csv(OUT/'data'/'p02_selector_feature_matrix.csv',low_memory=False)
gold=chg[(chg['reference_tier']=='gold')&(chg['Reference'].notna())&(chg['RAPIDS'].notna())].reset_index(drop=True)
Xind=b25c.encode_features(ind,feature_cols).astype(np.float32); Xg=b25c.encode_features(gold,feature_cols).astype(np.float32)
refi=ind['Reference'].values.astype(float); refg=gold['Reference'].values.astype(float)
teg=np.array([np.minimum(np.abs(gold[AE[a]].values.astype(float)-refg),ERR) for a in arms])
def earm(df,a,r): return np.minimum(np.abs(df[AE[a]].values.astype(float)-r),ERR)
best_ind=np.nanargmin(np.array([earm(ind,a,refi) for a in arms]),0)
print(f"in-dist {len(ind)}  gold charged {len(gold)}", flush=True)
def M(ch): e=teg[ch,np.arange(len(gold))]; return np.nanmean(FIXED[ch]),np.nanmean(e),np.nanmean(e>10)

# (3) drop-charged-aware: train arm regressors on IN-DIST ONLY, eval gold charged
preds=np.full((K,len(gold)),ERR)
for j,a in enumerate(arms):
    y=earm(ind,a,refi); mk=~np.isnan(y)
    mod=xgb.XGBRegressor(n_estimators=120,max_depth=4,learning_rate=0.1,subsample=0.85,colsample_bytree=0.7,random_state=0,objective='reg:absoluteerror',n_jobs=-1,verbosity=0); mod.fit(Xind[mk],y[mk]); preds[j]=mod.predict(Xg)
dca=M(np.argmin(FIXED[:,None]+LAM*preds.astype(float),0))
print(f"drop-charged-aware (train in-dist only) gold charged: {dca[0]:.0f} / {dca[1]:.2f} / {dca[2]*100:.2f}%", flush=True)

# CV over gold for ALORS + multiclass (charged-aware: train in-dist + gold-train)
def M_sub(ch,te): e=teg[ch,te]; return np.nanmean(FIXED[ch]),np.nanmean(e),np.nanmean(e>10)
kf=KFold(5,shuffle=True,random_state=0); AL=[]; MC=[]
for tr,te in kf.split(np.arange(len(gold))):
    Xtr=np.vstack([Xind,Xg[tr]])
    # ALORS: SVD on train error matrix + Ridge cold-start
    E=np.array([np.concatenate([earm(ind,a,refi),earm(gold.iloc[tr],a,refg[tr])]) for a in arms]).T
    cm=np.nanmean(E,0); ii=np.where(np.isnan(E)); E[ii]=np.take(cm,ii[1])
    svd=TruncatedSVD(3,random_state=0); U=svd.fit_transform(E); V=svd.components_
    sc=StandardScaler().fit(np.nan_to_num(Xtr)); rg=Ridge(alpha=10.0).fit(sc.transform(np.nan_to_num(Xtr)),U)
    pa=(rg.predict(sc.transform(np.nan_to_num(Xg[te])))@V).T
    AL.append(M_sub(np.argmin(FIXED[:,None]+LAM*pa,0),te))
    # multiclass: predict oracle-best arm
    ytr=np.concatenate([best_ind, np.nanargmin(np.array([earm(gold.iloc[tr],a,refg[tr]) for a in arms]),0)])
    clf=xgb.XGBClassifier(objective='multi:softprob',num_class=K,n_estimators=200,max_depth=4,learning_rate=0.1,subsample=0.85,colsample_bytree=0.7,random_state=0,n_jobs=-1,verbosity=0); clf.fit(Xtr,ytr)
    MC.append(M_sub(clf.predict_proba(Xg[te]).argmax(1),te))
AL=np.array(AL).mean(0); MC=np.array(MC).mean(0)
print(f"ALORS +cost gold charged:      {AL[0]:.0f} / {AL[1]:.2f} / {AL[2]*100:.2f}%", flush=True)
print(f"multiclass argmax gold charged: {MC[0]:.0f} / {MC[1]:.2f} / {MC[2]*100:.2f}%", flush=True)
json.dump({'drop_charged_aware':list(dca),'alors':list(AL),'multiclass':list(MC),
           'defer':[1636,2.78,0.0522],'global_best':[2041,2.42,0.0521]},open(RES/'gold_retrained_baselines.json','w'),indent=2)
print("Saved -> results/gold_retrained_baselines.json", flush=True)
