#!/usr/bin/env python3
"""
25a — Build the unified per-system selector feature matrix.

Joins 7 sources on (benchmark, system_id):
  1. scan_summary_per_system_wide.csv   — 3-backbone ECG + cross-backbone (5567 × 82)
  2. geometry_descriptors.csv           — XYZ-derived geometry features (5567 × 51)
  3. master_table.csv (crossMLIP supp)  — 15-method oracle energies + walltimes (5567 × 43)
  4. MACE relax raw_L2                  — UMA-fixed-geom MACE relax steps/fmax (5567 × 18)
  5. ORB relax raw_L2                   — same for ORB (5567 × 18)
  6. frozen-geom per-bench CSV          — xtb/uma/mace/orb SP at gold geometry (per-bench)
  7. probe.meta.json + target.meta.json — per-fragment PubChem/RDKit descriptors

Output:
  data/selector_feature_matrix.csv    — 5567 rows × ~200 cols
  _scratch/feature_matrix_audit.json  — schema, NaN per col, coverage
"""
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import numpy as np

OUT_ROOT = Path(__file__).resolve().parents[1]

PATHS = {
    'scan_summary': '/home/ruiding/benchmarking/scan_summary_harvest_2026-06-15/scan_summary_per_system_wide.csv',
    'geometry': '/home/ruiding/benchmarking/scan_summary_harvest_2026-06-15/geometry_descriptors.csv',
    'master_table': '/home/ruiding/benchmarking/RAPIDS_NCI_crossMLIP_L1_L2_L3_supp_2026-06-03/analysis_L1_L2_L3_ladder/outputs/tables/master_table.csv',
    'mace_l2': '/home/ruiding/benchmarking/RAPIDS_NCI_crossMLIP_L1_L2_L3_supp_2026-06-03/raw_L2/results_mace_omol_relax.csv',
    'orb_l2': '/home/ruiding/benchmarking/RAPIDS_NCI_crossMLIP_L1_L2_L3_supp_2026-06-03/raw_L2/results_orb_omol_relax.csv',
}
FROZEN_GEOM_ROOT = Path('/home/ruiding/benchmarking/RAPIDS_NCI_frozen_geom_benchmark_2026-06-01')
COLLECTION_ROOT = Path('/home/ruiding/benchmarking/collection_finished_all_fidelity')


# -------- 1. base table: scan_summary_wide --------
print("[1/7] Loading scan_summary (per-system 3-backbone)")
base = pd.read_csv(PATHS['scan_summary'])
print(f"      rows={len(base)}  cols={base.shape[1]}")


# -------- 2. geometry descriptors --------
print("[2/7] Loading geometry descriptors")
geo = pd.read_csv(PATHS['geometry'])
print(f"      rows={len(geo)}  cols={geo.shape[1]}")
base = base.merge(geo, on=['benchmark','system_id'], how='left', suffixes=('','_geo'))
print(f"      after merge: rows={len(base)}  cols={base.shape[1]}")


# -------- 3. per-bench CSVs (Reference + 9-arm oracle energies) --------
# Bypass master_table entirely (its system_id is in a different naming convention).
# Read the 18 per-bench CSVs directly; map their integer ID to scan_summary canonical name.
print("[3/7] Loading per-bench CSVs (Reference + 9-arm oracle energies)")
import re

def id_from_canonical(name):
    """Extract ID token from scan_summary canonical name. Returns string for stable joins.
    'D1200_HBCNO_001_hydrogen_hydrogen' → '1'
    'BFDb_BBI_001'                       → '1'
    'A24_01_water_ammonia'               → '1'
    'HB375_1.001_acetic_acid_acetaldehyde' → '1.001'
    Pattern: <bench>_<digits or X.YYY>(_ or end)
    """
    m = re.search(r'_(\d+(?:\.\d+)?)(?:_|$)', name)
    if not m: return None
    raw = m.group(1)
    # if integer-like, strip leading zeros for stable string compare
    if '.' not in raw:
        return str(int(raw))
    return raw

base['_join_id'] = base['system_id'].apply(id_from_canonical)
id_lookup = base[['benchmark','system_id','_join_id']].copy()

# Read per-bench CSVs
oracle_rows = []
for chargeclass in ('neutral','charged'):
    d = COLLECTION_ROOT / chargeclass
    if not d.exists(): continue
    for bdir in d.iterdir():
        if not bdir.is_dir(): continue
        bench = bdir.name
        csv_path = bdir / f"{bench}.csv"
        if not csv_path.exists(): continue
        bdf = pd.read_csv(csv_path)
        if 'ID' not in bdf.columns:
            # synthesize ID = 1-based row index (matches canonical scan_summary naming)
            bdf = bdf.reset_index(drop=True)
            bdf.insert(0, 'ID', range(1, len(bdf)+1))
        bdf['benchmark'] = bench
        oracle_rows.append(bdf)
oracle = pd.concat(oracle_rows, ignore_index=True)
print(f"      raw oracle rows: {len(oracle)}, columns sample: {list(oracle.columns)[:15]}")

# normalize ID to same string format as _join_id
def _id_to_str(v):
    try:
        f = float(v)
        if f.is_integer(): return str(int(f))
        return f"{f:g}"
    except: return str(v)
oracle['_join_id'] = oracle['ID'].apply(_id_to_str)
oracle = oracle.drop(columns=['ID'])
# rename feature-overlap cols
rename_map = {'Atoms':'oracle_Atoms', 'System':'oracle_System',
              'Molecule_A':'oracle_Molecule_A','Molecule_B':'oracle_Molecule_B',
              'Reference':'Reference', 'Charge':'oracle_complex_charge',
              'Group':'oracle_Group','Selection_A':'oracle_Selection_A','Selection_B':'oracle_Selection_B',
              'monA_charge':'oracle_monA_charge','monB_charge':'oracle_monB_charge'}
oracle = oracle.rename(columns={k:v for k,v in rename_map.items() if k in oracle.columns})

base = base.merge(oracle, on=['benchmark','_join_id'], how='left', suffixes=('','_oracle'))
base = base.drop(columns=['_join_id'])
print(f"      after merge: rows={len(base)}  cols={base.shape[1]}  Reference coverage={base['Reference'].notna().sum()}/{len(base)}")


# -------- 4. MACE L2 relax --------
print("[4/7] Loading MACE L2 (steps + fmax)")
mace_l2 = pd.read_csv(PATHS['mace_l2'])
mace_l2 = mace_l2.rename(columns={'system': 'system_id'})
mace_cols = [c for c in mace_l2.columns if c not in ('benchmark','system_id')]
mace_l2 = mace_l2[['benchmark','system_id'] + mace_cols].add_prefix('mace_l2__')
mace_l2 = mace_l2.rename(columns={'mace_l2__benchmark':'benchmark','mace_l2__system_id':'system_id'})
print(f"      rows={len(mace_l2)}  feature cols: {[c for c in mace_l2.columns if c.startswith('mace_l2__')][:6]}")
base = base.merge(mace_l2, on=['benchmark','system_id'], how='left')
print(f"      after merge: rows={len(base)}  cols={base.shape[1]}")


# -------- 5. ORB L2 relax --------
print("[5/7] Loading ORB L2 (steps + fmax)")
orb_l2 = pd.read_csv(PATHS['orb_l2'])
orb_l2 = orb_l2.rename(columns={'system': 'system_id'})
orb_cols = [c for c in orb_l2.columns if c not in ('benchmark','system_id')]
orb_l2 = orb_l2[['benchmark','system_id'] + orb_cols].add_prefix('orb_l2__')
orb_l2 = orb_l2.rename(columns={'orb_l2__benchmark':'benchmark','orb_l2__system_id':'system_id'})
base = base.merge(orb_l2, on=['benchmark','system_id'], how='left')
print(f"      after merge: rows={len(base)}  cols={base.shape[1]}")


# -------- 6. frozen-geom per-bench (xtb/uma/mace/orb fixed-geom SP) --------
print("[6/7] Loading frozen-geom per-bench (xtb/uma/mace/orb)")
fg_rows = []
for chargeclass in ('neutral','charged'):
    d = FROZEN_GEOM_ROOT / chargeclass
    if not d.exists(): continue
    for bdir in d.iterdir():
        if not bdir.is_dir(): continue
        bench = bdir.name
        # the bench csv has columns: ID, Molecule_A, Molecule_B, Reference, Atoms, + method columns
        csv_path = bdir / f"{bench}.csv"
        if not csv_path.exists(): continue
        df = pd.read_csv(csv_path)
        if 'ID' not in df.columns: continue  # skip statistics csvs
        # method cols vary; common are xtb, uma, mace, orb
        method_cols = [c for c in df.columns if c.lower() in ('xtb','uma','mace','orb')]
        if not method_cols: continue
        df = df[['ID'] + method_cols].copy()
        # convert ID → system_id by replicating per-bench naming convention
        # ID is integer 1..N; system_id is e.g. BFDb_SSI_charged_001
        # look up via master_table benchmark systems for canonical mapping
        df['benchmark'] = bench
        df = df.rename(columns={c: f'frozen__{c}_kcal' for c in method_cols})
        fg_rows.append(df)

if fg_rows:
    fg = pd.concat(fg_rows, ignore_index=True)
    # map ID to system_id via master_table
    id_map = mt.set_index(['benchmark','master_Atoms']).reset_index() if False else None
    # quick mapping: use master_table's per-benchmark sorted index assumption
    # Actually simpler — load per-bench CSV from collection_finished and use its ID + System columns
    sys_lookup = []
    for chargeclass in ('neutral','charged'):
        d = COLLECTION_ROOT / chargeclass
        if not d.exists(): continue
        for bdir in d.iterdir():
            if not bdir.is_dir(): continue
            bench = bdir.name
            csv_path = bdir / f"{bench}.csv"
            if not csv_path.exists(): continue
            try:
                bdf = pd.read_csv(csv_path)
                if 'ID' in bdf.columns and 'System' in bdf.columns:
                    sys_lookup.append(bdf[['ID','System']].rename(columns={'System':'system_id'}).assign(benchmark=bench))
            except: pass
    if sys_lookup:
        lookup = pd.concat(sys_lookup, ignore_index=True)
        fg = fg.merge(lookup, on=['benchmark','ID'], how='left')
        fg = fg.drop(columns=['ID'])
        # keep only relevant cols
        feat_cols = [c for c in fg.columns if c.startswith('frozen__')]
        fg = fg[['benchmark','system_id'] + feat_cols]
        print(f"      frozen-geom merged rows={len(fg)}  feature cols={feat_cols[:4]}")
        base = base.merge(fg, on=['benchmark','system_id'], how='left')
print(f"      after merge: rows={len(base)}  cols={base.shape[1]}")


# -------- 7. per-fragment chem descriptors (probe + target meta.json) --------
print("[7/7] Loading per-fragment chem descriptors from meta.json")

def extract_chem(meta):
    """Pull cheap descriptors out of probe/target meta.json safely."""
    out = {}
    if not meta: return out
    pc = (meta.get('descriptors_pubchem') or {})
    rd = (meta.get('descriptors_rdkit') or {})
    ident = (meta.get('identity') or {})
    for src, prefix in [(pc, 'pc'), (rd, 'rd')]:
        for k in ('formal_charge','molecular_weight','heavy_atom_count','xlogp','tpsa',
                  'h_bond_donor_count','h_bond_acceptor_count','rotatable_bond_count',
                  'aromatic_atom_count','ring_count','complexity','qed','fsp3'):
            v = src.get(k)
            if v is not None:
                try: out[f"{prefix}_{k}"] = float(v)
                except: pass
    # formal_charge: prefer rdkit
    out['formal_charge'] = out.get('rd_formal_charge', out.get('pc_formal_charge'))
    return out

chem_rows = []
n_seen = 0; n_with_meta = 0
for chargeclass in ('neutral','charged'):
    d = COLLECTION_ROOT / chargeclass
    if not d.exists(): continue
    for bdir in d.iterdir():
        if not bdir.is_dir(): continue
        bench = bdir.name
        sysdir = bdir / 'systems'
        if not sysdir.exists(): continue
        for sdir in sysdir.iterdir():
            if not sdir.is_dir(): continue
            sys_id = sdir.name
            n_seen += 1
            row = {'benchmark': bench, 'system_id': sys_id}
            scan_dir = sdir / 'RAPIDS_scan_x9'
            if not scan_dir.exists(): continue
            for side in ('probe','target'):
                p = scan_dir / f"{side}.meta.json"
                if not p.exists(): continue
                try:
                    m = json.load(open(p))
                    feats = extract_chem(m)
                    for k,v in feats.items():
                        row[f'{side}__{k}'] = v
                    n_with_meta += 1
                except: pass
            if len(row) > 2:
                chem_rows.append(row)
print(f"      systems seen: {n_seen}  with meta: {n_with_meta}")
chem_df = pd.DataFrame(chem_rows)
print(f"      chem feature cols: {chem_df.shape[1]-2}")
base = base.merge(chem_df, on=['benchmark','system_id'], how='left')
print(f"      after merge: rows={len(base)}  cols={base.shape[1]}")


# -------- save --------
out_csv = OUT_ROOT / 'data' / 'selector_feature_matrix.csv'
base.to_csv(out_csv, index=False)
print(f"\nFINAL: {out_csv}")
print(f"  rows = {len(base)}")
print(f"  cols = {base.shape[1]}")
print(f"  size = {out_csv.stat().st_size/1e6:.1f} MB")

# Audit
audit = {
    'rows': len(base),
    'cols': base.shape[1],
    'nan_per_col_top20': base.isna().sum().sort_values(ascending=False).head(20).to_dict(),
    'has_reference': 'Reference' in base.columns,
    'has_uma_rapids': any(c for c in base.columns if 'UMA_RAPIDS' in c or 'RAPIDS' in c),
    'per_benchmark_rows': base['benchmark'].value_counts().to_dict(),
}
audit_path = OUT_ROOT / '_scratch' / 'feature_matrix_audit.json'
json.dump(audit, open(audit_path,'w'), indent=2, default=str)
print(f"  audit  = {audit_path}")
