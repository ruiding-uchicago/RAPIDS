#!/usr/bin/env python3
"""90 — Clean 3-panel composite main figure (self-contained, paper style).
(a) transfer asymmetry  (b) in-distribution cost-accuracy  (c) charged OOD (DES370K gold)."""
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
import matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter
from adjustText import adjust_text

ARM_COLOR = {"RAPIDS":"#2196F3","PBE-D3BJ_SP":"#4CAF50","PBE-D3BJ_GeoSP":"#FF9800",
             "CREST_xTB":"#9C27B0","CREST_xTB_DFT":"#673AB7","Oracle":"#2ca02c"}
ARM_MARKER= {"RAPIDS":"o","PBE-D3BJ_SP":"s","PBE-D3BJ_GeoSP":"D",
             "CREST_xTB":"^","CREST_xTB_DFT":"P","Oracle":"*"}
ARM_LABEL = {"RAPIDS":"RAPIDS","PBE-D3BJ_SP":"PBE-D3BJ SP","PBE-D3BJ_GeoSP":"PBE-D3BJ GeoSP",
             "CREST_xTB":"CREST xTB","CREST_xTB_DFT":"CREST xTB+DFT"}
SEL_C="#d62728"; SEL_LW=2.5; DPI=200; LAM=800
BAR_A="#2ca02c"; BAR_B="#d62728"

OUT=Path(__file__).resolve().parents[1]; CACHE=OUT/'cache'; FIG=OUT/'figures'; DATA=OUT/'data'
# match offline_replay/plot_sequential.py (neutral_replay_frontier.png) aesthetic, larger fonts
plt.rcParams.update({'figure.dpi':DPI,'savefig.dpi':DPI,'font.size':12,'axes.titlesize':15,
    'axes.titleweight':'bold','axes.labelsize':14,'xtick.labelsize':12,'ytick.labelsize':12,'legend.fontsize':10})

def _auc(p,y):
    m=~np.isnan(y); p,y=p[m],y[m]; pos=y.sum(); neg=len(y)-pos
    if pos==0 or neg==0: return np.nan
    o=np.argsort(p); r=np.empty(len(p)); r[o]=np.arange(1,len(p)+1)
    return (r[y==1].sum()-pos*(pos+1)/2)/(pos*neg)

# ---- panel (a): transfer asymmetry ----
I8=pickle.load(open(CACHE/'v8_indist.pkl','rb')); arms=I8['arms']
AUCs=[];RHOs=[]
for arm in arms:
    j=arms.index(arm)
    pe=np.concatenate([c['preds'][j] for c in I8['caches'].values()])
    pc=np.concatenate([c['pcat10'][j] for c in I8['caches'].values()])
    te=np.concatenate([c['true_err'][j] for c in I8['caches'].values()])
    m=~np.isnan(te)
    AUCs.append(_auc(pc,(te>10).astype(float))); RHOs.append(spearmanr(pe[m],te[m]).correlation)

# ---- panels (b)(c): cost-accuracy ----
I=pickle.load(open(CACHE/'v6_indist.pkl','rb')); C=pickle.load(open(CACHE/'v6_charged.pkl','rb'))
FIXED=I['fixed_cost']; idx={a:i for i,a in enumerate(arms)}
Ic=list(I['caches'].values()); Cc=C['caches']

# gold mask for charged
_chg=pd.read_csv(DATA/'p02_selector_feature_matrix.csv',low_memory=False)
_chg=_chg[_chg['Reference'].notna()&_chg['RAPIDS'].notna()].reset_index(drop=True)
_tier=_chg['reference_tier'].values
GOLD=[(_tier[te]=='gold') for _,te in KFold(5,shuffle=True,random_state=0).split(np.arange(len(_chg)))]
N_GOLD=int(sum(m.sum() for m in GOLD))

def lam(l): return lambda c: np.argmin(FIXED[:,None]+l*c['preds'].astype(float),0)
def always(a): return lambda c: np.full(c['n'],idx[a],int)
def oracle(): return lambda c: np.nanargmin(c['true_err'],0)
def m_on(c,ch,mask=None):
    n=c['n']; e=c['true_err'][ch,np.arange(n)]; co=c['cost'][ch,np.arange(n)]
    if mask is not None: e,co=e[mask],co[mask]
    return np.nanmean(co),np.nanmean(e),np.nanmean(e>10)
def agg(caches,fn): return np.array([m_on(c,fn(c)) for c in caches]).mean(0)
def agg_g(caches,fn):
    r=[m_on(c,fn(c),GOLD[fi]) for fi,c in enumerate(caches) if GOLD[fi].sum()>0]
    return np.array(r).mean(0)

LAMS=[300,400,500,600,700,800,1000,1200,1500,2000,2500,3000]
i_front=np.array([agg(Ic,lam(l)) for l in LAMS])
c_front=np.array([agg_g(Cc,lam(l)) for l in LAMS])
iv=agg(Ic,lam(LAM)); cv=agg_g(Cc,lam(LAM))

# ================= figure (neutral_replay_frontier.png aesthetic) =================
# top row: (a) transfer asymmetry, wide+short.  bottom row: (b)(c) cost-accuracy 1x2.
fig=plt.figure(figsize=(16,7.2),dpi=DPI)
gs=fig.add_gridspec(2,2,height_ratios=[1.0,1.05],hspace=0.42,wspace=0.16)
ax_a=fig.add_subplot(gs[0,:]); ax_b=fig.add_subplot(gs[1,0]); ax_c=fig.add_subplot(gs[1,1])

# (a) grouped bars — light, matches bar panels (edgecolor black lw0.5, alpha0.85)
x=np.arange(len(arms))
ax_a.bar(x-0.2,AUCs,0.4,color=BAR_A,edgecolor='black',linewidth=0.5,alpha=0.85,
         label='catastrophe occurrence  —  AUC, |err|>10 kcal/mol')
ax_a.bar(x+0.2,RHOs,0.4,color=BAR_B,edgecolor='black',linewidth=0.5,alpha=0.85,
         label='error magnitude  —  Spearman rho')
ax_a.axhline(0.5,color='gray',ls=':',lw=0.8)
for i,(au,rh) in enumerate(zip(AUCs,RHOs)):
    ax_a.text(i-0.2,au+0.012,f'{au:.2f}',ha='center',fontsize=10)
    ax_a.text(i+0.2,rh+0.012,f'{rh:.2f}',ha='center',fontsize=10)
ax_a.set_xticks(x); ax_a.set_xticklabels([ARM_LABEL[a] for a in arms],fontsize=12)
ax_a.set_ylim(0,1.05); ax_a.set_ylabel('pooled out-of-fold transfer quality',fontsize=14)
ax_a.set_title('a.  Transfer asymmetry across chemical families',fontsize=15,fontweight='bold')
ax_a.legend(fontsize=11,loc='lower right',framealpha=0.85,edgecolor='gray',fancybox=False)
ax_a.grid(axis='y',alpha=0.25,which='both',linewidth=0.5)
ax_a.tick_params(labelsize=12)

# literature selector baselines on OUR benchmarks (cost, mean|err|). in-dist=72/73/78, charged gold=82/83
LIT={  # panel: (cost_s, mean_err); 's': method label; 'y': citation year
    'SATzilla-style (cost-blind)':{'in':(1027,2.51),'ch':(1861,2.41),'s':'SATzilla','y':2008},
    'FrugalML cascade':           {'in':(417,2.43), 'ch':(676,3.07), 's':'FrugalML','y':2020},
    'reject/defer rule':          {'in':(495,2.42), 'ch':(1636,2.78),'s':'defer','y':2020},
    'ALORS +cost':                {'in':(769,2.55), 'ch':(847,3.53), 's':'ALORS','y':2017},
    'Multiclass (Rice)':          {'in':(1066,2.91),'ch':(1903,2.35),'s':'Rice','y':1976},
}
LIT_C='#555555'
SHORT={'RAPIDS':'RAPIDS','PBE-D3BJ_SP':'PBE SP','PBE-D3BJ_GeoSP':'PBE GeoSP','CREST_xTB':'CREST','CREST_xTB_DFT':'CREST+DFT'}
# (b)(c) — every point labelled in-plot (name + cost, s); adjustText prevents overlap; markers semi-transparent
for ax,caches,front,vp,title,gold,litkey in [
    (ax_b,Ic,i_front,iv,'b.  In-distribution, 18-benchmark',False,'in'),
    (ax_c,Cc,c_front,cv,f'c.  Charged OOD, DES370K  n={N_GOLD}',True,'ch')]:
    A=agg_g if gold else agg
    xs=[]; texts=[]
    ax.plot(front[:,0],front[:,1],'-',color=SEL_C,lw=2.0,alpha=0.7,zorder=3)
    for arm in ['RAPIDS','PBE-D3BJ_SP','PBE-D3BJ_GeoSP','CREST_xTB','CREST_xTB_DFT']:
        c,e,k=A(caches,always(arm))
        ax.plot(c,e,marker=ARM_MARKER[arm],color=ARM_COLOR[arm],ms=11,ls='',
                markeredgecolor='black',markeredgewidth=0.5,alpha=0.75,zorder=4)
        texts.append(ax.text(c,e,f'{SHORT[arm]}',fontsize=8.5,color=ARM_COLOR[arm],fontweight='bold'))
        xs.append(c)
    for name,d in LIT.items():
        lc,le=d[litkey]
        ax.plot(lc,le,marker='X',color=LIT_C,ms=11,ls='',markeredgecolor='black',markeredgewidth=0.5,alpha=0.75,zorder=4)
        texts.append(ax.text(lc,le,f"{d['s']} ({d['y']})",fontsize=8.5,color=LIT_C))
        xs.append(lc)
    oc,oe,ok=A(caches,oracle())
    ax.plot(oc,oe,marker='*',color=ARM_COLOR['Oracle'],ms=19,ls='',
            markeredgecolor='black',markeredgewidth=0.5,alpha=0.85,zorder=6)
    texts.append(ax.text(oc,oe,f'Oracle',fontsize=8.5,color=ARM_COLOR['Oracle'],fontweight='bold'))
    ax.plot(vp[0],vp[1],marker='*',color=SEL_C,ms=22,ls='',
            markeredgecolor='black',markeredgewidth=0.8,alpha=0.9,zorder=7)
    texts.append(ax.text(vp[0],vp[1],f'RAPIDS-Select',fontsize=9.5,color=SEL_C,fontweight='bold'))
    xs+=[oc,vp[0]]
    ax.set_xscale('log'); ax.set_xlim(min(xs)*0.55,max(xs)*1.9)
    ax.xaxis.set_major_locator(LogLocator(base=10,subs=(1,2,3,5)))
    ax.xaxis.set_minor_locator(LogLocator(base=10,subs=np.arange(1,10)*0.1))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v,_:(f'{v:.0f}' if v>=1 else f'{v:g}')))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda v,_:''))
    ax.set_xlabel('mean per-system cost, s',fontsize=14)
    ax.set_ylabel('mean |error|, kcal/mol',fontsize=14)
    ax.set_title(title,fontsize=15,fontweight='bold')
    ax.grid(True,alpha=0.25,which='both',linewidth=0.5); ax.tick_params(labelsize=12)
    adjust_text(texts,ax=ax,expand=(1.15,1.4),force_text=(0.4,0.6),
                arrowprops=dict(arrowstyle='-',color='0.6',lw=0.5))

out=FIG/'composite_main_figure.png'; plt.savefig(out,dpi=DPI,bbox_inches='tight')
print(f"Saved -> {out}")
print(f"(b) in-dist RAPIDS-Select: {iv[0]:.0f}s / {iv[1]:.2f} / cat {iv[2]*100:.1f}%")
print(f"(c) charged RAPIDS-Select: {cv[0]:.0f}s / {cv[1]:.2f} / cat {cv[2]*100:.1f}%  (gold n={N_GOLD})")
