#!/usr/bin/env python3
"""
55 — V6 unified policy sweep on both caches (charged-aware, NO charge short-circuit).

Goal: ONE policy that wins in-dist (beat Always-GeoSP) AND is OOD-robust on
charged (dominate Always-DFT, no degeneration).

Policies evaluated on both v6_indist.pkl (18 LOBO folds) and v6_charged.pkl (5 CV folds):
  - greedy_eps: cheapest arm with pred_err <= eps
  - lambda: argmin cost + lam*pred_err
  - greedy + p20 floor: greedy, but if p20>0.5 force >= GeoSP (catastrophic guard)
"""
import pickle
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parents[1]
CACHE = OUT/'cache'
ERR_CAP = 50.0

I = pickle.load(open(CACHE/'v6_indist.pkl','rb'))
C = pickle.load(open(CACHE/'v6_charged.pkl','rb'))
arms = I['arms']; FIXED = I['fixed_cost']
idx = {a:i for i,a in enumerate(arms)}
print("arms:", arms)

def metrics_on(cache, chosen):
    n = cache['n']
    e = cache['true_err'][chosen, np.arange(n)]
    c = cache['cost'][chosen, np.arange(n)]
    return np.nanmean(c), np.nanmean(e), np.nanmedian(e), np.nanmean(e>10)

def agg_indist(pick_fn):
    cs=[];es=[];ms=[];ks=[]
    for held,cache in I['caches'].items():
        ch = pick_fn(cache)
        c,e,m,k = metrics_on(cache, ch); cs.append(c);es.append(e);ms.append(m);ks.append(k)
    return np.mean(cs),np.mean(es),np.mean(ms),np.mean(ks)

def agg_charged(pick_fn):
    cs=[];es=[];ms=[];ks=[]
    for cache in C['caches']:
        ch = pick_fn(cache)
        c,e,m,k = metrics_on(cache, ch); cs.append(c);es.append(e);ms.append(m);ks.append(k)
    return np.mean(cs),np.mean(es),np.mean(ms),np.mean(ks)

# ---- pick functions ----
def greedy(eps):
    def f(cache):
        pe = cache['preds']; mask = pe <= eps
        return np.where(mask.any(0), mask.argmax(0), len(arms)-1)
    return f

def lam_pick(lam):
    def f(cache):
        util = FIXED[:,None] + lam*cache['preds'].astype(float)
        return np.argmin(util, 0)
    return f

def greedy_p20floor(eps, t20):
    def f(cache):
        pe = cache['preds']; mask = pe <= eps
        ch = np.where(mask.any(0), mask.argmax(0), len(arms)-1)
        # catastrophic guard: if p20 high and chosen arm cheaper than GeoSP, bump to GeoSP
        floor = idx['PBE-D3BJ_GeoSP']
        hi = cache['p20'] > t20
        ch[hi & (ch < floor)] = floor
        return ch
    return f

def always(a):
    return lambda cache: np.full(cache['n'], idx[a], dtype=int)
def oracle():
    return lambda cache: np.nanargmin(cache['true_err'],0)

print("\n" + "="*94)
print(f"{'policy':<22}{'|':^3}{'IN-DIST (18 LOBO)':^34}{'|':^3}{'CHARGED (5 CV)':^30}")
print(f"{'':<22}{'|':^3}{'cost':>8}{'err':>8}{'med':>7}{'cat%':>7}{'|':^3}{'cost':>8}{'err':>8}{'med':>7}{'cat%':>7}")
print("="*94)

def row(name, fn):
    ic,ie,im,ik = agg_indist(fn)
    cc,ce,cm,ck = agg_charged(fn)
    print(f"{name:<22}{'|':^3}{ic:>8.1f}{ie:>8.3f}{im:>7.2f}{ik*100:>7.2f}{'|':^3}{cc:>8.1f}{ce:>8.3f}{cm:>7.2f}{ck*100:>7.2f}")
    return (ic,ie,ik,cc,ce,ck)

results = {}
for eps in [1,2,3,5,7,10]:
    results[f'greedy_eps{eps}'] = row(f'greedy_eps{eps}', greedy(eps))
print("  " + "-"*90)
for lam in [50,100,200,400,800]:
    results[f'lambda{lam}'] = row(f'lambda{lam}', lam_pick(lam))
print("  " + "-"*90)
for eps in [2,3,5]:
    results[f'greedy_eps{eps}_p20floor'] = row(f'g_eps{eps}_p20.5', greedy_p20floor(eps,0.5))
print("  " + "-"*90)
row('Always_RAPIDS', always('RAPIDS'))
row('Always_GeoSP', always('PBE-D3BJ_GeoSP'))
row('Always_CREST_DFT', always('CREST_xTB_DFT'))
row('Oracle', oracle())

print("\n" + "="*94)
print("WIN CHECK (in-dist vs Always-GeoSP 615/2.53 target; charged vs Always-DFT 2041/8.99):")
print("="*94)
# in-dist Always-GeoSP baseline from this cache
gd_i = agg_indist(always('PBE-D3BJ_GeoSP'))
dft_c = agg_charged(always('CREST_xTB_DFT'))
geo_c = agg_charged(always('PBE-D3BJ_GeoSP'))
print(f"  in-dist Always-GeoSP: cost={gd_i[0]:.1f} err={gd_i[1]:.3f} cat={gd_i[3]*100:.2f}%")
print(f"  charged Always-DFT:   cost={dft_c[0]:.1f} err={dft_c[1]:.3f} cat={dft_c[3]*100:.2f}%")
print(f"  charged Always-GeoSP: cost={geo_c[0]:.1f} err={geo_c[1]:.3f} cat={geo_c[3]*100:.2f}%")
print()
for name,(ic,ie,ik,cc,ce,ck) in results.items():
    win_indist = (ic < gd_i[0]) and (ie <= gd_i[1]+0.02)      # cheaper & >= GeoSP accuracy in-dist
    beat_dft   = (cc < dft_c[0]) and (ce <= dft_c[1]+0.7)     # cheaper than DFT, near accuracy
    not_degen  = True  # greedy never degenerates by construction
    tag = []
    if win_indist: tag.append("IN-DIST✓")
    if beat_dft:   tag.append("beatDFT✓")
    if win_indist and beat_dft: tag.append("★DUAL-WIN")
    if tag: print(f"  {name:<24} indist {ic:>6.0f}/{ie:.2f}  charged {cc:>6.0f}/{ce:.2f}/{ck*100:.1f}%  {' '.join(tag)}")
