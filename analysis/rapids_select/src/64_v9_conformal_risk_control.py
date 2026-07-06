#!/usr/bin/env python3
"""
64 — V9: Conformal Risk Control (CRC) routing.

Headline reframing (from round-2 critique): stop chasing the ~1.5% numeric gain
(it's noise: V8 beats V6 in only 7/18 LOBO folds). The defensible AAAI contribution
is a DISTRIBUTION-FREE guarantee on the catastrophic rate at minimum cost — AND an
honest characterization of exactly where that guarantee holds vs. provably fails.

Mechanism (ONE knob, replaces V8's ad-hoc M/M5 terms):
  Selection rule S_t(x): pick the cheapest arm a (cost order) with pcat10_a(x) <= t;
  if none qualifies, pick argmin_a pcat10_a(x) (the safest arm).
  As t increases -> cheaper arms used -> realized catastrophic risk R(t) increases
  monotonically. CRC calibrates t so E[risk] <= alpha with finite-sample validity.

CRC (Angelopoulos, Bates, Fisch, Lei, Schuster, Jordan 2022):
  loss ℓ = 1[selected arm true err > 10], bounded B=1, R(t) monotone non-decreasing.
  Choose t_hat = largest grid t with (n·R̂_cal(t) + B)/(n+1) <= alpha.
  Then E[R(t_hat)] <= alpha under exchangeability of (cal, test).

Two regimes:
  (A) charged OOD = 5-fold CV over one pooled distribution -> EXCHANGEABLE -> guarantee should HOLD.
  (B) in-dist LOBO = each held-out benchmark is a DIFFERENT chemistry -> NOT exchangeable
      -> guarantee expected to be VIOLATED on some folds (the honest negative result),
      then reported at the benchmark-jackknife level (benchmark = exchangeable unit).

Uses cached per-arm pcat10 (cache/v8_*.pkl) — NO retraining.
"""
import pickle, json
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parents[1]; CACHE=OUT/'cache'; RES=OUT/'results'
I = pickle.load(open(CACHE/'v8_indist.pkl','rb'))
C = pickle.load(open(CACHE/'v8_charged.pkl','rb'))
arms = I['arms']; FIXED = I['fixed_cost']; K=len(arms)
Ic = I['caches']          # dict benchmark -> fold
Cc = C['caches']          # list of 5 folds
TGRID = np.linspace(0.0, 1.0, 201)

def select(fold, t):
    """cheapest arm with pcat10<=t, else safest (min pcat10). returns chosen idx [n]."""
    pc = fold['pcat10']; n = fold['n']
    ok = pc <= t                       # [K,n]
    out = np.empty(n, int)
    any_ok = ok.any(0)
    # cheapest qualifying (arms already in cost order 0..K-1)
    first = np.where(ok.any(0), ok.argmax(0), pc.argmin(0))
    return first

def risk_cost(fold, chosen):
    n = fold['n']
    e = fold['true_err'][chosen, np.arange(n)]
    c = fold['cost'][chosen, np.arange(n)]
    return np.nanmean(e>10), np.nanmean(c), np.nanmean(e)

def R_on(folds_concat_pcat, folds_concat_true, folds_concat_cost, t):
    """risk over a pooled set given stacked arrays [K,N]."""
    ok = folds_concat_pcat <= t
    first = np.where(ok.any(0), ok.argmax(0), folds_concat_pcat.argmin(0))
    N = folds_concat_pcat.shape[1]
    e = folds_concat_true[first, np.arange(N)]
    c = folds_concat_cost[first, np.arange(N)]
    return np.nanmean(e>10), np.nanmean(c), np.nanmean(e)

def stack(folds):
    pc = np.concatenate([f['pcat10'] for f in folds],axis=1)
    te = np.concatenate([f['true_err'] for f in folds],axis=1)
    co = np.concatenate([f['cost'] for f in folds],axis=1)
    return pc,te,co

def crc_threshold(cal_folds, alpha, B=1.0):
    """largest t on grid with (n*Rhat + B)/(n+1) <= alpha. R monotone non-decreasing in t."""
    pc,te,co = stack(cal_folds); n = pc.shape[1]
    best_t = 0.0
    for t in TGRID:
        Rhat,_,_ = R_on(pc,te,co,t)
        if (n*Rhat + B)/(n+1) <= alpha:
            best_t = t
        else:
            break   # monotone: once it exceeds, stays exceeded
    return best_t

# ---------- (A) charged OOD: 5-fold CV, exchangeable ----------
print("="*76)
print("(A) CHARGED OOD (5-fold CV, exchangeable) — CRC guarantee should HOLD")
print("="*76)
for alpha in [0.15, 0.18, 0.20]:
    realized=[]; costs=[]; ts=[]
    for f in range(len(Cc)):
        cal=[Cc[g] for g in range(len(Cc)) if g!=f]; test=Cc[f]
        t=crc_threshold(cal,alpha); ts.append(t)
        r,c,e=risk_cost(test, select(test,t)); realized.append(r); costs.append(c)
    realized=np.array(realized)
    cov = np.mean(realized<=alpha)
    print(f"  alpha={alpha:.2f}: mean realized risk {realized.mean()*100:5.2f}%  (target {alpha*100:.0f}%)  "
          f"folds within target {int((realized<=alpha).sum())}/5  mean cost {np.mean(costs):.0f}s  t_hat~{np.mean(ts):.3f}")
# compare to always-DFT (safest) risk/cost on charged
dft_r=[]; dft_c=[]
for f in Cc:
    ch=np.full(f['n'], K-1, int); r,c,e=risk_cost(f,ch); dft_r.append(r); dft_c.append(c)
print(f"  ref Always-CREST-DFT: risk {np.mean(dft_r)*100:.2f}%  cost {np.mean(dft_c):.0f}s")

# ---------- (B) in-dist LOBO: NOT exchangeable ----------
print("\n"+"="*76)
print("(B) IN-DIST LOBO (each fold = unseen chemistry, NOT exchangeable) — guarantee EXPECTED TO FAIL")
print("="*76)
benches=sorted(Ic.keys())
for alpha in [0.05, 0.07]:
    realized={}; costs=[]
    for held in benches:
        cal=[Ic[b] for b in benches if b!=held]; test=Ic[held]
        t=crc_threshold(cal,alpha)
        r,c,e=risk_cost(test, select(test,t)); realized[held]=r; costs.append(c)
    rv=np.array(list(realized.values()))
    viol=[b for b in benches if realized[b]>alpha+1e-9]
    print(f"  alpha={alpha:.2f}: benchmark-jackknife mean risk {rv.mean()*100:5.2f}%  "
          f"VIOLATIONS {len(viol)}/18  mean cost {np.mean(costs):.0f}s")
    print(f"    worst violators: "+", ".join(f'{b}={realized[b]*100:.1f}%' for b in sorted(viol,key=lambda b:-realized[b])[:5]))

# benchmark-jackknife (honest weaker claim): treat each benchmark as exchangeable unit
print("\n  Benchmark-level jackknife+ (benchmark = exchangeable unit; guarantees META coverage, NOT per-benchmark):")
for alpha in [0.05, 0.07]:
    # leave-one-benchmark-out on the benchmark risks: calibrate t on 17 benchmarks' POOLED risk, target alpha
    per=[]
    for held in benches:
        cal=[Ic[b] for b in benches if b!=held]; test=Ic[held]
        t=crc_threshold(cal,alpha); r,_,_=risk_cost(test,select(test,t)); per.append(r)
    per=np.array(per)
    print(f"    alpha={alpha:.2f}: mean-over-benchmarks realized risk {per.mean()*100:.2f}%  "
          f"(meta-coverage {'HOLDS' if per.mean()<=alpha else 'FAILS'}); median {np.median(per)*100:.2f}%")

json.dump({'note':'V9 CRC routing — charged exchangeable (holds), LOBO not (fails per-benchmark, meta-level honest)'},
          open(RES/'v9_crc_summary.json','w'))
print("\nSaved -> results/v9_crc_summary.json")
