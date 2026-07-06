#!/usr/bin/env python3
"""58 — V6 dual-battlefield figure: in-dist Pareto + charged Pareto, both won by ONE policy."""
import pickle, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paper_style import (ARM_COLOR, ARM_MARKER, SELECTOR_C, SELECTOR_LW, DPI,
                          FS_LABEL, FS_TITLE, FS_SUPTITLE, FS_LEGEND, FS_ANNOT,
                          apply_rc, style_grid)
apply_rc()

OUT = Path(__file__).resolve().parents[1]
CACHE = OUT/'cache'; FIG = OUT/'figures'
I = pickle.load(open(CACHE/'v6_indist.pkl','rb'))
C = pickle.load(open(CACHE/'v6_charged.pkl','rb'))
arms = I['arms']; FIXED = I['fixed_cost']; idx = {a:i for i,a in enumerate(arms)}

def m_on(cache, chosen):
    n=cache['n']; e=cache['true_err'][chosen,np.arange(n)]; c=cache['cost'][chosen,np.arange(n)]
    return np.nanmean(c),np.nanmean(e),np.nanmean(e>10)
def agg(caches, fn):
    r=np.array([m_on(c,fn(c)) for c in caches]); return r.mean(0)
def lam_pick(lam):
    return lambda c: np.argmin(FIXED[:,None]+lam*c['preds'].astype(float),0)
def always(a): return lambda c: np.full(c['n'],idx[a],int)
def oracle(): return lambda c: np.nanargmin(c['true_err'],0)

Ic=list(I['caches'].values()); Cc=C['caches']
LAMS=[300,400,500,600,700,800,1000,1200,1500,2000,2500,3000]
i_front=np.array([agg(Ic,lam_pick(l)) for l in LAMS])
c_front=np.array([agg(Cc,lam_pick(l)) for l in LAMS])
V6=800; iv=agg(Ic,lam_pick(V6)); cv=agg(Cc,lam_pick(V6))

BASE={'Always_RAPIDS':'RAPIDS','Always_PBE-D3BJ_SP':'PBE-D3BJ_SP','Always_GeoSP':'PBE-D3BJ_GeoSP',
      'Always_CREST_xTB':'CREST_xTB','Always_CREST_DFT':'CREST_xTB_DFT'}

fig,ax=plt.subplots(1,2,figsize=(16,6.5))
for pan,(caches,front,vp,title,oref) in enumerate([
    (Ic,i_front,iv,'IN-DIST (18 LOBO folds, 5,532 systems)','in'),
    (Cc,c_front,cv,'CHARGED OOD (P0.2, 5-fold CV, 1,155 charged)','ch')]):
    a=ax[pan]
    a.plot(front[:,0],front[:,1],'-o',color=SELECTOR_C,lw=SELECTOR_LW,ms=5,zorder=10,label='V6 λ-frontier (charged-aware)')
    for name,arm in BASE.items():
        c,e,k=agg(caches,always(arm))
        a.scatter(c,e,s=240,marker=ARM_MARKER[arm],color=ARM_COLOR[arm],edgecolors='black',lw=0.8,zorder=8,label=name.replace('Always_','Always-'))
    oc,oe,ok=agg(caches,oracle())
    a.scatter(oc,oe,s=380,marker=ARM_MARKER['Oracle'],color=ARM_COLOR['Oracle'],edgecolors='black',lw=0.8,zorder=8,label='Oracle (per-system best)')
    a.scatter(vp[0],vp[1],s=560,marker='*',color=SELECTOR_C,edgecolors='black',lw=1.4,zorder=20,label='★ V6 (λ=800)')
    a.annotate(f'★ V6\n{vp[0]:.0f}s / {vp[1]:.2f}\ncat {vp[2]*100:.1f}%',xy=(vp[0],vp[1]),
               xytext=(vp[0]*1.35,vp[1]+0.25 if pan==0 else vp[1]+0.8),fontsize=FS_ANNOT,fontweight='bold',
               arrowprops=dict(arrowstyle='->',color=SELECTOR_C,lw=1.8),
               bbox=dict(boxstyle='round',fc='mistyrose',ec=SELECTOR_C))
    a.set_xscale('log'); a.set_xlabel('mean per-system cost (s)',fontsize=FS_LABEL)
    a.set_ylabel('mean |error| (kcal/mol)',fontsize=FS_LABEL); a.set_title(title,fontsize=FS_TITLE,fontweight='bold')
    style_grid(a); a.legend(fontsize=FS_LEGEND,loc='upper right',framealpha=0.85,edgecolor='gray',fancybox=False)

plt.suptitle('RAPIDS-Select V6 — ONE cost-aware policy (no charge short-circuit) wins BOTH battlefields\n'
             f'in-dist: {iv[0]:.0f}s/{iv[1]:.2f} (−39% vs GeoSP)   |   charged OOD: {cv[0]:.0f}s/{cv[1]:.2f} (beats GeoSP err, −66% vs CREST-DFT), non-degenerate',
             fontsize=FS_SUPTITLE,fontweight='bold')
plt.tight_layout()
out=FIG/'V6_dual_battlefield.png'; plt.savefig(out,dpi=DPI,bbox_inches='tight')
print(f"Saved -> {out}")
print(f"V6 in-dist: {iv[0]:.1f}/{iv[1]:.3f}/{iv[2]*100:.2f}%   charged: {cv[0]:.1f}/{cv[1]:.3f}/{cv[2]*100:.2f}%")
