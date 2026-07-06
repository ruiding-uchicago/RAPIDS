#!/usr/bin/env python3
"""
49 — Build the P0.2 (charged-rescue prospective) selector feature matrix.

Applies the SAME feature-extraction recipe as `25a_build_feature_matrix.py`
to the 1,290 charged-rescue systems and emits a CSV with columns EXACTLY
matching `data/selector_feature_matrix.csv` (5,567 in-distribution).

Sources for P0.2:
  1. scan_summary.json  — committed_charged_rescue/<sid>/  (UMA, 1155 sys)
                          committed_mace/<sid>/            (MACE, 1291 sys)
                          committed_orb/<sid>/             (ORB, 1097 sys)
  2. geometry descriptors — recomputed from
        RAPIDS_NCI_charged_rescue_2026-06-30/charged/<bench>/systems/<sid>/RAPIDS_scan_x9/{probe,target,complex}.xyz
        (falls back to charged_rescue_ground_truth_2026-06-14/systems/<sid>/*.xyz)
  3. Ground-truth 9-arm energies + timings — RAPIDS_NCI_charged_rescue_2026-06-30/charged/<bench>/<bench>.csv
     (adds Reference, RAPIDS, PBE-D3BJ_SP, ..., CREST_xTB_DFT + _time + _status)
  4. MACE_L2 / ORB_L2 relax — deployment/L2_local/results_cr_{mace,orb}_relax_shard{00,01}.csv
  5. Frozen-geom SP (xtb/uma/mace/orb) — NOT AVAILABLE for P0.2 → NaN
  6. Chem descriptors — RDKit-computed from meta.json SMILES
        (PubChem not queried; probe__pc_* / target__pc_* left NaN except
         formal_charge/molecular_weight/heavy_atom_count/h_bond_donor_count/
         h_bond_acceptor_count/rotatable_bond_count/tpsa/xlogp/complexity
         which we compute via RDKit into the `pc_` slot as a proxy —
         these are RDKit surrogates of PubChem values; XGBoost sees them
         the same way it saw the pc_ column during training.)

Naming convention:
  - benchmark = 'DES370K_charged'  or  'IL174_charged'
  - system_id = row's `System` column in the ground-truth CSV
                (== manifest.system_id  == committed_*/ dir name)
  - reference_tier = 'gold' (DES370K, CCSD(T)/CBS)  or 'silver' (IL174, DLPNO-CCSD(T))
"""
from __future__ import annotations
import os, sys, json, math, glob, csv, re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import numpy as np

# ============================================================================
# Paths
# ============================================================================
OUT_ROOT   = Path(__file__).resolve().parents[1]
P02_ROOT   = Path('/home/ruiding/benchmarking/charged_rescue_ground_truth_2026-06-14')
GT_ROOT    = P02_ROOT / 'RAPIDS_NCI_charged_rescue_2026-06-30' / 'charged'
UMA_DIR    = P02_ROOT / 'committed_charged_rescue'
MACE_DIR   = P02_ROOT / 'committed_mace'
ORB_DIR    = P02_ROOT / 'committed_orb'
MACE_L2_SHARDS = sorted((P02_ROOT / 'deployment' / 'L2_local').glob('results_cr_mace_relax_shard*.csv'))
ORB_L2_SHARDS  = sorted((P02_ROOT / 'deployment' / 'L2_local').glob('results_cr_orb_relax_shard*.csv'))
MANIFEST   = P02_ROOT / 'manifest.csv'

REF_MATRIX = OUT_ROOT / 'data' / 'selector_feature_matrix.csv'
OUT_CSV    = OUT_ROOT / 'data' / 'p02_selector_feature_matrix.csv'
FEAT_JSON  = OUT_ROOT / 'models' / 'rapids_select_v1' / 'feature_columns.json'

# ============================================================================
# 1. Reference column ordering (must match 5,567 pipeline EXACTLY)
# ============================================================================
print('[0/7] Loading reference schema...')
ref_head = pd.read_csv(REF_MATRIX, nrows=1)
REF_COLS = list(ref_head.columns)
print(f'      reference has {len(REF_COLS)} cols')
V5_FEATS = json.load(open(FEAT_JSON))
print(f'      V5 uses {len(V5_FEATS)} features')

# ============================================================================
# 2. Enumerate all P0.2 systems + assign benchmark
# ============================================================================
print('[1/7] Building canonical P0.2 system catalogue...')
mani = pd.read_csv(MANIFEST)
print(f'      manifest: {len(mani)} rows')

def bench_of(sid, source):
    if source == 'DES370K': return 'DES370K_charged'
    if source in ('IL128','IL174'): return 'IL174_charged'
    # fallback by prefix
    if sid.startswith('DES_'): return 'DES370K_charged'
    if sid.startswith('IL_'):  return 'IL174_charged'
    return 'UNKNOWN_charged'

mani['benchmark'] = mani.apply(lambda r: bench_of(r['system_id'], r.get('source','')), axis=1)
mani['reference_tier'] = mani['reference_tier'].fillna('unknown')

# Restrict to non-UNKNOWN
sys_df = mani[['system_id','benchmark','probe_smiles','target_smiles','probe_charge','target_charge','complex_charge','E_ref_kcal_mol','reference_method','reference_tier']].copy()
print(f'      benchmarks: {sys_df.benchmark.value_counts().to_dict()}')

# ============================================================================
# 3. Extract scan_summary per backbone
# ============================================================================
print('[2/7] Extracting scan_summary per backbone...')

SCAN_KEYS_TOP = ('task_name','n_anchors','num_orientations','total_configs',
                 'eligible_configs','flagged_configs','wall_time_seconds',
                 'best_energy_eV','best_solution_eV')
VG_KEYS = ('mean_eV','std_eV','iqr_eV','range_eV',
           'best_is_outlier','sol_best_is_outlier','top2_within_1_kcal',
           'low_contact_success','overall_recommendation')
ST_KEYS = ('tier_used','auto_upgrade','tier_1_configs','tier_2_configs','tier_3_configs')

def extract_scan(bb, d, source_path):
    row = {}
    for k in SCAN_KEYS_TOP:
        row[f'{bb}__{k}'] = d.get(k)
    vg = d.get('variance_guard') or {}
    for k in VG_KEYS:
        row[f'{bb}__variance_guard__{k}'] = vg.get(k)
    st = d.get('sampling_tier') or {}
    for k in ST_KEYS:
        row[f'{bb}__sampling_tier__{k}'] = st.get(k)
    ec = d.get('energy_consistency') or {}
    if isinstance(ec, dict):
        row[f'{bb}__energy_consistency__status'] = ec.get('status','')
    else:
        row[f'{bb}__energy_consistency__status'] = str(ec)
    row[f'{bb}__source_path'] = source_path
    return row

def harvest_backbone(bb, base_dir):
    found = {}
    if not base_dir.exists():
        print(f'      [{bb}] {base_dir} missing')
        return found
    for sdir in base_dir.iterdir():
        if not sdir.is_dir(): continue
        sp = sdir / 'scan_summary.json'
        if not sp.exists(): continue
        try:
            d = json.load(open(sp))
        except Exception:
            continue
        found[sdir.name] = extract_scan(bb, d, str(sp))
    print(f'      [{bb}] {len(found)} systems from {base_dir.name}')
    return found

uma_scan  = harvest_backbone('UMA',  UMA_DIR)
mace_scan = harvest_backbone('MACE', MACE_DIR)
orb_scan  = harvest_backbone('ORB',  ORB_DIR)

# ============================================================================
# 4. Geometry descriptors (recompute from XYZs)
# ============================================================================
print('[3/7] Computing geometry descriptors from XYZs...')
sys.path.insert(0, '/home/ruiding/benchmarking/scan_summary_harvest_2026-06-15/_scratch')
try:
    from compute_geometry_descriptors import compute_row as geo_compute, finalize_row as geo_finalize, COLUMNS as GEO_COLS
except Exception as e:
    print('      failed to import geometry compute:', e)
    raise

# The upstream compute_row expects (bench, sysid, group) with ROOT-relative layout
# under /collection_finished_all_fidelity/{group}/{bench}/systems/{sysid}/RAPIDS_scan_x9/.
# We monkey-patch by writing our own compute_row_p02 that reads P0.2 xyz paths.
import compute_geometry_descriptors as _gd

def compute_geo_p02(args):
    bench, sysid, group_row = args  # group_row is a manifest row dict
    row = {'benchmark': bench, 'system_id': sysid, 'warn': ''}
    # Preferred: ground-truth systems dir with RAPIDS_scan_x9/*.xyz
    scan_dir1 = GT_ROOT / bench / 'systems' / sysid / 'RAPIDS_scan_x9'
    p1 = scan_dir1 / 'probe.xyz'
    t1 = scan_dir1 / 'target.xyz'
    c1 = scan_dir1 / 'complex.xyz'
    # Fallback: manifest points to systems/<sid>/*.xyz under P02_ROOT
    fallback = P02_ROOT / 'systems' / sysid
    p2 = fallback / 'probe.xyz'
    t2 = fallback / 'target.xyz'
    c2 = fallback / 'complex.xyz'
    if p1.exists() and t1.exists():
        pxyz, txyz, cxyz = str(p1), str(t1), str(c1) if c1.exists() else None
    elif p2.exists() and t2.exists():
        pxyz, txyz, cxyz = str(p2), str(t2), str(c2) if c2.exists() else None
    else:
        row['warn'] = 'xyz_missing'
        return row
    try:
        pe, pc = _gd.read_xyz(pxyz)
        te, tc = _gd.read_xyz(txyz)
    except Exception as e:
        row['warn'] = f'read_pt_failed:{e.__class__.__name__}'
        return row
    # Reuse the interior of compute_row by directly copying the logic
    row['n_atoms_probe']  = len(pe)
    row['n_atoms_target'] = len(te)
    row['heavy_atoms_probe']  = sum(1 for e in pe if e != 'H')
    row['heavy_atoms_target'] = sum(1 for e in te if e != 'H')
    row.update(_gd.element_hist(pe, 'probe'))
    row.update(_gd.element_hist(te, 'target'))
    row['has_halogen_probe']  = int(any(e in _gd.HALOGENS for e in pe))
    row['has_halogen_target'] = int(any(e in _gd.HALOGENS for e in te))
    row['rg_probe']         = _gd.radius_of_gyration(pe, pc)
    row['rg_target']        = _gd.radius_of_gyration(te, tc)
    row['axis_ratio_probe'] = _gd.axis_ratio(pe, pc)
    row['axis_ratio_target']= _gd.axis_ratio(te, tc)
    use_complex = cxyz is not None
    n_p, n_t = len(pe), len(te)
    ce = cc = None
    if use_complex:
        try:
            ce, cc = _gd.read_xyz(cxyz)
        except Exception as e:
            row['warn'] = (row['warn']+f';complex_read:{e.__class__.__name__}').strip(';')
            use_complex = False
    if use_complex:
        row['n_atoms_complex'] = len(ce)
        if len(ce) != n_p + n_t:
            row['warn'] = (row['warn']+';complex_size_mismatch').strip(';')
            use_complex = False
        else:
            from collections import Counter
            if ce[:n_p] != pe or ce[n_p:] != te:
                if Counter(ce[:n_p]) != Counter(pe) or Counter(ce[n_p:]) != Counter(te):
                    row['warn'] = (row['warn']+';complex_order_mismatch').strip(';')
                    use_complex = False
    else:
        row['n_atoms_complex'] = n_p + n_t
    if use_complex:
        A_coords = cc[:n_p]; B_coords = cc[n_p:]
        A_elems  = ce[:n_p]; B_elems  = ce[n_p:]
    else:
        A_coords = pc; B_coords = tc
        A_elems  = pe; B_elems  = te
    cmA = _gd.com(A_elems, A_coords)
    cmB = _gd.com(B_elems, B_coords)
    row['com_distance_AB_angstrom'] = float(np.linalg.norm(cmA - cmB))
    D_AB = _gd.pairwise_dist(A_coords, B_coords)
    row['min_contact_AB_angstrom'] = float(D_AB.min())
    vdwA = np.array([_gd.VDW.get(e, 1.7) for e in A_elems])
    vdwB = np.array([_gd.VDW.get(e, 1.7) for e in B_elems])
    sumVdW = vdwA[:,None] + vdwB[None,:]
    row['n_close_contacts_AB_within_vdw_x1.0'] = int((D_AB < sumVdW * 1.0).sum())
    row['n_close_contacts_AB_within_vdw_x1.2'] = int((D_AB < sumVdW * 1.2).sum())
    overlap = np.maximum(0.0, sumVdW - D_AB)
    row['vdw_overlap_score'] = float(overlap.sum())
    # H-bonds
    donors_A = _gd.find_h_donors(A_elems, A_coords)
    donors_B = _gd.find_h_donors(B_elems, B_coords)
    n_hb = 0; min_dha = float('inf')
    HB_DA_MAX = 3.5; HB_ANG_MIN = 120.0
    for ih, idn in donors_A:
        for ja, ea in enumerate(B_elems):
            if ea not in _gd.HBD_HBA: continue
            d_DA = np.linalg.norm(A_coords[idn] - B_coords[ja])
            if d_DA >= HB_DA_MAX: continue
            ang = _gd.angle_deg(A_coords[idn], A_coords[ih], B_coords[ja])
            if ang > HB_ANG_MIN:
                n_hb += 1
                if d_DA < min_dha: min_dha = d_DA
    for ih, idn in donors_B:
        for ja, ea in enumerate(A_elems):
            if ea not in _gd.HBD_HBA: continue
            d_DA = np.linalg.norm(B_coords[idn] - A_coords[ja])
            if d_DA >= HB_DA_MAX: continue
            ang = _gd.angle_deg(B_coords[idn], B_coords[ih], A_coords[ja])
            if ang > HB_ANG_MIN:
                n_hb += 1
                if d_DA < min_dha: min_dha = d_DA
    row['n_hbonds_AB'] = int(n_hb)
    row['min_dha_distance_AB'] = float(min_dha) if min_dha < float('inf') else float('nan')
    # halogen bond
    cx_A = _gd.find_C_X_neighbors(A_elems, A_coords)
    cx_B = _gd.find_C_X_neighbors(B_elems, B_coords)
    n_xb = 0; min_xa = float('inf')
    XB_ANG = 140.0
    for ix, ic in cx_A.items():
        for ja, ea in enumerate(B_elems):
            if ea not in _gd.LP_ACCEPTORS: continue
            d = np.linalg.norm(A_coords[ix] - B_coords[ja])
            if d >= 4.0: continue
            ang = _gd.angle_deg(A_coords[ic], A_coords[ix], B_coords[ja])
            if ang > XB_ANG:
                n_xb += 1
                if d < min_xa: min_xa = d
    for ix, ic in cx_B.items():
        for ja, ea in enumerate(A_elems):
            if ea not in _gd.LP_ACCEPTORS: continue
            d = np.linalg.norm(B_coords[ix] - A_coords[ja])
            if d >= 4.0: continue
            ang = _gd.angle_deg(B_coords[ic], B_coords[ix], A_coords[ja])
            if ang > XB_ANG:
                n_xb += 1
                if d < min_xa: min_xa = d
    row['n_halogen_bond_AB'] = int(n_xb)
    row['min_halX_distance_AB'] = float(min_xa) if min_xa < float('inf') else float('nan')
    # salt-bridge O/N cross counts
    row['_OA_NB_within5'] = 0
    row['_OB_NA_within5'] = 0
    for ia, ea in enumerate(A_elems):
        if ea != 'O': continue
        for jb, eb in enumerate(B_elems):
            if eb != 'N': continue
            if D_AB[ia, jb] < 5.0:
                row['_OA_NB_within5'] += 1
    for ia, ea in enumerate(A_elems):
        if ea != 'N': continue
        for jb, eb in enumerate(B_elems):
            if eb != 'O': continue
            if D_AB[ia, jb] < 5.0:
                row['_OB_NA_within5'] += 1
    row['n_aromatic_rings_probe']  = _gd.count_aromatic_rings(A_elems, A_coords)
    row['n_aromatic_rings_target'] = _gd.count_aromatic_rings(B_elems, B_coords)
    if row['n_aromatic_rings_probe'] > 0 and row['n_aromatic_rings_target'] > 0:
        rA = _gd.ring_centroids(A_elems, A_coords)
        rB = _gd.ring_centroids(B_elems, B_coords)
        best = float('inf')
        for ca in rA:
            for cb in rB:
                d = float(np.linalg.norm(ca - cb))
                if d < best: best = d
        row['nearest_ring_pair_distance_angstrom'] = best
    else:
        row['nearest_ring_pair_distance_angstrom'] = float('nan')
    # charges from manifest row
    monA = float(group_row.get('probe_charge', 0) or 0)
    monB = float(group_row.get('target_charge', 0) or 0)
    row = geo_finalize(row, (monA, monB))
    return row

print(f'      computing geometry for {len(sys_df)} systems (may take a moment)...')
geo_rows = []
n_ok = 0; n_warn = 0
items = [(r['benchmark'], r['system_id'], r.to_dict()) for _, r in sys_df.iterrows()]
with ProcessPoolExecutor(max_workers=12) as ex:
    futs = {ex.submit(compute_geo_p02, it): it for it in items}
    for fut in as_completed(futs):
        it = futs[fut]
        try:
            r = fut.result()
            geo_rows.append(r)
            if r.get('warn',''): n_warn += 1
            else: n_ok += 1
        except Exception as e:
            geo_rows.append({'benchmark': it[0], 'system_id': it[1], 'warn': f'compute_failed:{e.__class__.__name__}'})
            n_warn += 1
geo_df = pd.DataFrame(geo_rows)
print(f'      geometry: {n_ok} clean, {n_warn} with warn')

# ============================================================================
# 5. Chem descriptors from SMILES via RDKit
# ============================================================================
print('[4/7] Computing chem descriptors from SMILES (RDKit)...')
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors, QED, Crippen
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def rdkit_desc(smi):
    """Return a dict of {pc_*, rd_*, formal_charge} matching extract_chem() from 25a."""
    out = {}
    if not isinstance(smi, str) or not smi.strip():
        return out
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return out
    try:
        mh = Chem.AddHs(m)
    except Exception:
        mh = m
    try:
        out['pc_formal_charge']       = float(Chem.GetFormalCharge(m))
        out['pc_molecular_weight']    = float(Descriptors.MolWt(m))
        out['pc_heavy_atom_count']    = float(Lipinski.HeavyAtomCount(m))
        out['pc_xlogp']               = float(Crippen.MolLogP(m))
        out['pc_tpsa']                = float(Descriptors.TPSA(m))
        out['pc_h_bond_donor_count']  = float(Lipinski.NumHDonors(m))
        out['pc_h_bond_acceptor_count']= float(Lipinski.NumHAcceptors(m))
        out['pc_rotatable_bond_count']= float(Lipinski.NumRotatableBonds(m))
        # PubChem 'complexity' is Bertz-like; use BertzCT as a numeric surrogate
        out['pc_complexity']          = float(Descriptors.BertzCT(m))
        out['rd_aromatic_atom_count'] = float(sum(1 for a in m.GetAtoms() if a.GetIsAromatic()))
        out['rd_ring_count']          = float(rdMolDescriptors.CalcNumRings(m))
        try:
            out['rd_qed']             = float(QED.qed(m))
        except Exception:
            out['rd_qed']             = float('nan')
        out['rd_fsp3']                = float(Lipinski.FractionCSP3(m))
        out['formal_charge']          = out['pc_formal_charge']
    except Exception:
        pass
    return out

chem_rows = []
for _, r in sys_df.iterrows():
    row = {'benchmark': r['benchmark'], 'system_id': r['system_id']}
    p = rdkit_desc(r.get('probe_smiles',''))
    t = rdkit_desc(r.get('target_smiles',''))
    for k, v in p.items(): row[f'probe__{k}']  = v
    for k, v in t.items(): row[f'target__{k}'] = v
    chem_rows.append(row)
chem_df = pd.DataFrame(chem_rows)
print(f'      chem: {len(chem_df)} rows, {chem_df.shape[1]-2} cols')

# ============================================================================
# 6. MACE_L2 / ORB_L2 relax
# ============================================================================
print('[5/7] Loading MACE_L2 / ORB_L2 relax shards...')

def load_l2_shards(shards, prefix):
    if not shards: return pd.DataFrame(columns=['benchmark','system_id'])
    dfs = []
    for p in shards:
        try:
            dfs.append(pd.read_csv(p))
        except Exception as e:
            print(f'      failed to read {p}: {e}')
    if not dfs: return pd.DataFrame(columns=['benchmark','system_id'])
    df = pd.concat(dfs, ignore_index=True)
    df = df.rename(columns={'system':'system_id'})
    # drop extra P0.2-only cols not in canonical schema
    keep_cols = ['atoms_complex','atoms_probe','atoms_target',
                 'E_complex_eV','E_probe_eV','E_target_eV','dE_int_kcal_mol',
                 'steps_complex','steps_probe','steps_target',
                 'fmax_complex','fmax_probe','fmax_target',
                 'time_s','success','error']
    have = [c for c in keep_cols if c in df.columns]
    df = df[['system_id'] + have].copy()
    # rename with prefix
    df = df.rename(columns={c: f'{prefix}__{c}' for c in have})
    return df

mace_l2 = load_l2_shards(MACE_L2_SHARDS, 'mace_l2')
orb_l2  = load_l2_shards(ORB_L2_SHARDS,  'orb_l2')
print(f'      mace_l2: {len(mace_l2)} rows')
print(f'      orb_l2:  {len(orb_l2)} rows')

# ============================================================================
# 7. Ground-truth 9-arm energies + timings
# ============================================================================
print('[6/7] Loading ground-truth 9-arm CSVs...')
gt_frames = []
for bench, csv_path in [('DES370K_charged', GT_ROOT / 'DES370K_charged' / 'DES370K_charged.csv'),
                        ('IL174_charged',   GT_ROOT / 'IL174_charged'   / 'IL174_charged.csv')]:
    if not csv_path.exists():
        print(f'      MISSING {csv_path}')
        continue
    df = pd.read_csv(csv_path)
    df['benchmark'] = bench
    df = df.rename(columns={'System':'system_id',
                            'Molecule_A':'oracle_Molecule_A',
                            'Molecule_B':'oracle_Molecule_B',
                            'Atoms':'oracle_Atoms',
                            'Charge':'oracle_complex_charge'})
    # duplicate to keep both `system_id` join-key AND oracle_System echo
    df['oracle_System'] = df['system_id']
    df = df.drop(columns=['ID'])
    gt_frames.append(df)
oracle = pd.concat(gt_frames, ignore_index=True) if gt_frames else pd.DataFrame()
print(f'      oracle: {len(oracle)} rows, {oracle.shape[1]} cols')

# ============================================================================
# 8. Assemble the master matrix
# ============================================================================
print('[7/7] Assembling master matrix...')

base = sys_df[['benchmark','system_id']].copy()
base['canonical_system_key'] = base['system_id']

# Attach scan_summary per backbone
def attach_scan(base, scan_dict, bb):
    rows = []
    for sid, r in scan_dict.items():
        rows.append({'system_id': sid, **r})
    if not rows:
        return base
    df = pd.DataFrame(rows)
    return base.merge(df, on='system_id', how='left')

base = attach_scan(base, uma_scan,  'UMA')
base = attach_scan(base, mace_scan, 'MACE')
base = attach_scan(base, orb_scan,  'ORB')

# Cross-backbone derived
def n_backbones(r):
    return int(pd.notna(r.get('UMA__source_path'))) + int(pd.notna(r.get('MACE__source_path'))) + int(pd.notna(r.get('ORB__source_path')))
base['n_backbones_present'] = base.apply(n_backbones, axis=1)
std_cols = [f'{bb}__variance_guard__std_eV' for bb in ('UMA','MACE','ORB')]
have = [c for c in std_cols if c in base.columns]
base['cross_backbone_std_eV_mean'] = base[have].mean(axis=1, skipna=True) if have else np.nan
be_cols = [f'{bb}__best_energy_eV' for bb in ('UMA','MACE','ORB')]
have_be = [c for c in be_cols if c in base.columns]
if len(have_be) == 3:
    all3 = base[have_be].notna().all(axis=1)
    base['cross_backbone_best_E_disagreement_kcal'] = np.where(
        all3, (base[have_be].max(axis=1) - base[have_be].min(axis=1)) * 23.06, np.nan)
else:
    base['cross_backbone_best_E_disagreement_kcal'] = np.nan
rec_cols = [f'{bb}__variance_guard__overall_recommendation' for bb in ('UMA','MACE','ORB')]
def agree_count(r):
    n = 0
    for c in rec_cols:
        v = r.get(c)
        if isinstance(v, str) and 'confident' in v.lower(): n += 1
    return n / 3.0
base['cross_backbone_recommendation_agreement'] = base.apply(agree_count, axis=1)

# Geometry
base = base.merge(geo_df, on=['benchmark','system_id'], how='left', suffixes=('','_geo'))

# Oracle 9-arm
base = base.merge(oracle, on=['benchmark','system_id'], how='left', suffixes=('','_oracle'))

# MACE_L2 / ORB_L2 (join by system_id only; benchmark not present in shards)
if len(mace_l2):
    base = base.merge(mace_l2, on='system_id', how='left')
if len(orb_l2):
    base = base.merge(orb_l2, on='system_id', how='left')

# Chem
base = base.merge(chem_df, on=['benchmark','system_id'], how='left', suffixes=('','_chem'))

# Frozen-geom (not available for P0.2)
for c in ('frozen__xtb_kcal','frozen__uma_kcal','frozen__mace_kcal','frozen__orb_kcal'):
    if c not in base.columns:
        base[c] = np.nan

# Reference tier annotation
tier_map = mani.set_index('system_id')['reference_tier'].to_dict()
base['reference_tier'] = base['system_id'].map(tier_map)

# ============================================================================
# 9. Reorder to match 5567 schema EXACTLY (+ append P0.2-only columns)
# ============================================================================
extra_p02 = ['reference_tier']  # keep P0.2-only annotations at end
missing_cols = [c for c in REF_COLS if c not in base.columns]
for c in missing_cols:
    base[c] = np.nan
extra_cols = [c for c in base.columns if c not in REF_COLS and c not in extra_p02]
final_cols = REF_COLS + extra_p02
# Drop any accidentally-created suffixed columns
base = base[final_cols].copy()

# ============================================================================
# 10. Write CSV
# ============================================================================
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
base.to_csv(OUT_CSV, index=False)
print(f'\nWROTE {OUT_CSV}  rows={len(base)}  cols={base.shape[1]}')

# ============================================================================
# 11. Sanity audit
# ============================================================================
print('\n' + '='*70)
print('AUDIT SUMMARY')
print('='*70)
print(f'Rows: {len(base)}   Cols: {base.shape[1]}   (ref has {len(REF_COLS)})')

# schema drift
missing_in_p02 = [c for c in REF_COLS if c not in base.columns]
extra_in_p02   = [c for c in base.columns if c not in REF_COLS and c not in extra_p02]
print(f'\nSchema drift vs 5,567 pipeline:')
print(f'  cols missing from P0.2 output: {len(missing_in_p02)}  → {missing_in_p02[:10]}')
print(f'  extra cols beyond ref:         {len(extra_in_p02)}   → {extra_in_p02[:10]}')

# Reference coverage
ref_cov = base['Reference'].notna().sum() / len(base) if len(base) else 0
print(f'\nReference coverage: {base["Reference"].notna().sum()}/{len(base)} = {100*ref_cov:.1f}%')

# 5 V5 arms
V5_ARMS = ['RAPIDS','CREST_xTB','PBE-D3BJ_SP','PBE-D3BJ_GeoSP','CREST_xTB_DFT']
print(f'\n5 V5 arms coverage:')
for a in V5_ARMS:
    if a in base.columns:
        c = base[a].notna().sum()
        print(f'  {a:20s}  {c}/{len(base)} = {100*c/len(base):.1f}%')
    else:
        print(f'  {a:20s}  MISSING COLUMN')

# 156 V5 features presence + coverage
print(f'\n156 V5 feature presence in output:')
missing_v5 = [f for f in V5_FEATS if f not in base.columns]
print(f'  V5 feats missing entirely: {len(missing_v5)}  → {missing_v5[:10]}')
cov_v5 = []
for f in V5_FEATS:
    if f in base.columns:
        cov_v5.append(base[f].notna().sum() / len(base))
    else:
        cov_v5.append(0.0)
cov_v5 = np.array(cov_v5)
print(f'  V5 feat mean coverage:      {100*cov_v5.mean():.1f}%')
print(f'  V5 feats >=90% coverage:    {int((cov_v5>=0.90).sum())}/156')
print(f'  V5 feats >=50% coverage:    {int((cov_v5>=0.50).sum())}/156')
print(f'  V5 feats  =0% coverage:     {int((cov_v5==0.0).sum())}/156')

# per-backbone coverage
print(f'\nPer-backbone scan_summary coverage:')
for bb in ('UMA','MACE','ORB'):
    c = base[f'{bb}__source_path'].notna().sum() if f'{bb}__source_path' in base.columns else 0
    print(f'  {bb}: {c}/{len(base)} = {100*c/len(base):.1f}%')
c3 = (base['n_backbones_present']==3).sum()
c2 = (base['n_backbones_present']==2).sum()
c1 = (base['n_backbones_present']==1).sum()
c0 = (base['n_backbones_present']==0).sum()
print(f'  3-of-3: {c3}   2-of-3: {c2}   1-of-3: {c1}   0-of-3: {c0}')

# systems with all 156 present
present_mask = np.ones(len(base), dtype=bool)
for f in V5_FEATS:
    if f in base.columns:
        present_mask &= base[f].notna().values
n_all_156 = int(present_mask.sum())
print(f'\nSystems with ALL 156 V5 features present: {n_all_156}/{len(base)}')

# near-present: >=140/156 features
per_row_cov = np.zeros(len(base))
for f in V5_FEATS:
    if f in base.columns:
        per_row_cov += base[f].notna().values.astype(int)
print(f'Systems with ≥140/156 V5 features present: {int((per_row_cov>=140).sum())}/{len(base)}')
print(f'Systems with ≥120/156 V5 features present: {int((per_row_cov>=120).sum())}/{len(base)}')
print(f'Systems with ≥100/156 V5 features present: {int((per_row_cov>=100).sum())}/{len(base)}')

# per-benchmark
print(f'\nPer-benchmark row counts:')
for b, n in base['benchmark'].value_counts().items():
    ref_c = base[base.benchmark==b]['Reference'].notna().sum() if 'Reference' in base.columns else 0
    print(f'  {b}: {n} rows, Reference={ref_c}')

# Reference tier
print(f'\nReference tier distribution:')
for t, n in base['reference_tier'].value_counts(dropna=False).items():
    print(f'  {t}: {n}')

print(f'\nOUT: {OUT_CSV}')
print('DONE.')
