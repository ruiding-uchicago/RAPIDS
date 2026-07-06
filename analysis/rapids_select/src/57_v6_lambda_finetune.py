#!/usr/bin/env python3
"""57 — Fine lambda sweep to lock V6 operating point. Reports joint Pareto."""
import pickle
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parents[1]
CACHE = OUT/'cache'
I = pickle.load(open(CACHE/'v6_indist.pkl','rb'))
C = pickle.load(open(CACHE/'v6_charged.pkl','rb'))
arms = I['arms']; FIXED = I['fixed_cost']; idx = {a:i for i,a in enumerate(arms)}

def m_on(cache, chosen):
    n=cache['n']; e=cache['true_err'][chosen,np.arange(n)]; c=cache['cost'][chosen,np.arange(n)]
    return np.nanmean(c),np.nanmean(e),np.nanmedian(e),np.nanmean(e>10)
def agg(caches, fn):
    r=np.array([m_on(c,fn(c)) for c in caches]); return r.mean(0)
def lam_pick(lam):
    def f(cache):
        return np.argmin(FIXED[:,None]+lam*cache['preds'].astype(float),0)
    return f
def always(a): return lambda c: np.full(c['n'],idx[a],int)
def oracle(): return lambda c: np.nanargmin(c['true_err'],0)

Ic=list(I['caches'].values()); Cc=C['caches']
gd_i=agg(Ic,always('PBE-D3BJ_GeoSP')); dft_c=agg(Cc,always('CREST_xTB_DFT')); geo_c=agg(Cc,always('PBE-D3BJ_GeoSP'))

print("baselines:")
print(f"  in-dist GeoSP {gd_i[0]:.0f}/{gd_i[1]:.3f}/{gd_i[3]*100:.2f}%   charged GeoSP {geo_c[0]:.0f}/{geo_c[1]:.3f}/{geo_c[3]*100:.2f}%   charged DFT {dft_c[0]:.0f}/{dft_c[1]:.3f}/{dft_c[3]*100:.2f}%")
print()
print(f"{'lambda':>7} | {'IN-DIST cost/err/cat':^28} | {'CHARGED cost/err/cat':^28} | dual?")
best=None
for lam in [300,400,500,600,700,800,1000,1200,1500,2000,2500,3000,4000]:
    ic,ie,im,ik=agg(Ic,lam_pick(lam)); cc,ce,cm,ck=agg(Cc,lam_pick(lam))
    # dual-win: in-dist beats GeoSP (cheaper & err<=GeoSP), charged beats GeoSP err & cheaper than DFT
    win_i = ic<gd_i[0] and ie<=gd_i[1]+0.01
    win_c = ce<=geo_c[1]+0.05 and cc<dft_c[0]
    dual = "★DUAL" if (win_i and win_c) else ("i" if win_i else "")+("c" if win_c else "")
    print(f"{lam:>7} | {ic:>7.0f}/{ie:>6.3f}/{ik*100:>5.2f}% | {cc:>7.0f}/{ce:>6.3f}/{ck*100:>5.2f}% | {dual}")
    if win_i and win_c:
        score = ie + ce*0.1  # prefer low in-dist err, tie-break charged
        if best is None or score<best[0]: best=(score,lam,ic,ie,ik,cc,ce,ck)

print()
if best:
    _,lam,ic,ie,ik,cc,ce,ck=best
    print(f"RECOMMENDED V6: lambda={lam}")
    print(f"  in-dist: {ic:.0f}s / {ie:.3f} / {ik*100:.2f}%   (vs GeoSP {gd_i[0]:.0f}/{gd_i[1]:.3f}/{gd_i[3]*100:.2f}%)")
    print(f"  charged: {cc:.0f}s / {ce:.3f} / {ck*100:.2f}%   (vs GeoSP {geo_c[0]:.0f}/{geo_c[1]:.3f}, DFT {dft_c[0]:.0f}/{dft_c[1]:.3f})")
else:
    print("No strict dual-win; inspect table for best tradeoff.")
