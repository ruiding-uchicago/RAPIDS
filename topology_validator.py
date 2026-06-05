#!/usr/bin/env python3
"""
Structural Integrity Guardrails for RAPIDS.

Three layers of validation after ML potential geometry optimization:
  1. Topology Guard — detect spurious bond breaking/formation (intramolecular only)
  2. Geometry Guard — detect atom clashes and severe bond strain
  3. Energy Guard   — detect non-physical binding energies

Each guard is independent and produces its own result dict.
"""

import numpy as np
from ase import Atoms
from ase.data import vdw_radii, atomic_numbers
from ase.neighborlist import natural_cutoffs, NeighborList
from typing import Dict, List, Optional, Set, Tuple


# Element-pair specific bond thresholds (Angstroms).
# These override the default covalent_radius * multiplier when set.
# Format: frozenset({element1, element2}): max_distance
BOND_THRESHOLDS: Dict[frozenset, float] = {
    frozenset({"C", "F"}): 1.80,
    frozenset({"C", "Cl"}): 2.00,
    frozenset({"C", "Br"}): 2.20,
}

# Task fallback order for multi-task consensus
TASK_FALLBACK = {
    "omol": ["oc20", "omat"],
    "oc20": ["omol", "omat"],
    "omat": ["oc20", "omol"],
}

# Tasks that do NOT support charged systems
NEUTRAL_ONLY_TASKS = {"oc20", "omat"}


def get_fallback_tasks(primary_task: str, charge: int = 0) -> List[str]:
    """Return ordered list of fallback tasks for a given primary task.

    Args:
        primary_task: The primary task that failed topology check.
        charge: Net system charge. When non-zero, oc20 and omat are
                excluded because they do not support charged systems.
    """
    tasks = TASK_FALLBACK.get(primary_task, [])
    if charge != 0:
        tasks = [t for t in tasks if t not in NEUTRAL_ONLY_TASKS]
    return tasks


def get_bond_graph(
    atoms: Atoms,
    mult: float = 1.2,
    custom_thresholds: Optional[Dict[frozenset, float]] = None,
) -> Set[Tuple[int, int]]:
    """
    Build set of bonded atom pairs using covalent radii.

    Args:
        atoms: ASE Atoms object
        mult: Multiplier on covalent radii for bond detection (default 1.2)
        custom_thresholds: Element-pair specific distance thresholds.
                           If None, uses module-level BOND_THRESHOLDS.

    Returns:
        Set of (i, j) tuples where i < j, representing bonded pairs.
    """
    if custom_thresholds is None:
        custom_thresholds = BOND_THRESHOLDS

    cutoffs = natural_cutoffs(atoms, mult=mult)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)

    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    bonds: Set[Tuple[int, int]] = set()

    for i in range(len(atoms)):
        indices, offsets = nl.get_neighbors(i)
        for j in indices:
            a, b = min(i, j), max(i, j)
            if (a, b) in bonds:
                continue

            # Check element-pair specific threshold
            pair_key = frozenset({symbols[a], symbols[b]})
            if pair_key in custom_thresholds:
                dist = np.linalg.norm(positions[a] - positions[b])
                if dist <= custom_thresholds[pair_key]:
                    bonds.add((a, b))
                # If dist > threshold, this bond is considered broken
                # even though neighborlist found it
            else:
                bonds.add((a, b))

    # Also check custom thresholds for pairs NOT found by neighborlist
    # (in case neighborlist cutoff is tighter than custom threshold)
    for pair_key, threshold in custom_thresholds.items():
        elems = list(pair_key)
        if len(elems) != 2:
            continue
        e1, e2 = elems
        for i in range(len(atoms)):
            if symbols[i] != e1:
                continue
            for j in range(i + 1, len(atoms)):
                if symbols[j] != e2:
                    if not (symbols[j] == e1 and symbols[i] == e2):
                        continue
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist <= threshold:
                    bonds.add((i, j))

    return bonds


def split_bonds(
    bonds: Set[Tuple[int, int]],
    n_probe: int,
    n_total: int,
) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]], Set[Tuple[int, int]]]:
    """
    Split bonds into probe-internal, target-internal, and intermolecular.

    Atom indexing convention:
        probe atoms:  0 .. n_probe-1
        target atoms: n_probe .. n_total-1

    Returns:
        (probe_bonds, target_bonds, inter_bonds)
    """
    probe_bonds = set()
    target_bonds = set()
    inter_bonds = set()

    for a, b in bonds:
        a_is_probe = a < n_probe
        b_is_probe = b < n_probe

        if a_is_probe and b_is_probe:
            probe_bonds.add((a, b))
        elif (not a_is_probe) and (not b_is_probe):
            target_bonds.add((a, b))
        else:
            inter_bonds.add((a, b))

    return probe_bonds, target_bonds, inter_bonds


def _filter_by_distance_change(
    changed_bonds: Set[Tuple[int, int]],
    pos_init: np.ndarray,
    pos_final: np.ndarray,
    change_type: str,
    min_pct: float = 15.0,
) -> Set[Tuple[int, int]]:
    """
    Filter out bond changes where the distance moved less than min_pct%.

    For 'broke': reference distance is the initial (bonded) distance.
    For 'formed': reference distance is the final (bonded) distance.

    Args:
        changed_bonds: Set of (i, j) pairs flagged as changed.
        pos_init: Initial positions array.
        pos_final: Final positions array.
        change_type: 'broke' or 'formed'.
        min_pct: Minimum percentage change to count as real. Default 15%.

    Returns:
        Filtered set with only significant changes.
    """
    filtered = set()
    for a, b in changed_bonds:
        d_init = np.linalg.norm(pos_init[a] - pos_init[b])
        d_final = np.linalg.norm(pos_final[a] - pos_final[b])
        ref = d_init if change_type == "broke" else d_final
        if ref < 0.1:
            # Degenerate case, keep it
            filtered.add((a, b))
            continue
        pct = abs(d_final - d_init) / ref * 100
        if pct >= min_pct:
            filtered.add((a, b))
    return filtered


def validate_topology(
    initial_atoms: Atoms,
    final_atoms: Atoms,
    n_probe: int,
    mult: float = 1.2,
    min_pct: float = 15.0,
) -> dict:
    """
    Compare intramolecular bond topology between initial and final structures.

    Only flags changes WITHIN probe or WITHIN target molecules.
    Intermolecular bond changes (probe-target) are expected and not flagged.
    Bond changes where the distance moved less than min_pct% of the reference
    bond length are filtered out as threshold noise.

    Args:
        initial_atoms: Structure before optimization
        final_atoms: Structure after optimization
        n_probe: Number of probe atoms (first n_probe atoms are probe)
        mult: Covalent radius multiplier for bond detection
        min_pct: Minimum percentage distance change to count as real (default 15%)

    Returns:
        dict with keys:
            - topology_preserved (bool): True if no intramolecular changes
            - confidence (str): "high" if preserved, "needs_verification" if not
            - probe_broken (list): Bonds broken within probe
            - probe_formed (list): New bonds formed within probe
            - target_broken (list): Bonds broken within target
            - target_formed (list): New bonds formed within target
            - inter_new (list): New intermolecular bonds (informational)
            - details (str): Human-readable summary
    """
    n_total = len(initial_atoms)

    initial_bonds = get_bond_graph(initial_atoms, mult=mult)
    final_bonds = get_bond_graph(final_atoms, mult=mult)

    # Split into probe/target/inter for both states
    init_probe, init_target, init_inter = split_bonds(initial_bonds, n_probe, n_total)
    final_probe, final_target, final_inter = split_bonds(final_bonds, n_probe, n_total)

    # Compute raw changes per category
    raw_probe_broken = init_probe - final_probe
    raw_probe_formed = final_probe - init_probe
    raw_target_broken = init_target - final_target
    raw_target_formed = final_target - init_target
    inter_new = final_inter - init_inter

    # Filter out threshold noise using percentage-based criterion
    pos_init = initial_atoms.get_positions()
    pos_final = final_atoms.get_positions()

    probe_broken = _filter_by_distance_change(raw_probe_broken, pos_init, pos_final, "broke", min_pct)
    probe_formed = _filter_by_distance_change(raw_probe_formed, pos_init, pos_final, "formed", min_pct)
    target_broken = _filter_by_distance_change(raw_target_broken, pos_init, pos_final, "broke", min_pct)
    target_formed = _filter_by_distance_change(raw_target_formed, pos_init, pos_final, "formed", min_pct)

    n_filtered = (
        (len(raw_probe_broken) - len(probe_broken))
        + (len(raw_probe_formed) - len(probe_formed))
        + (len(raw_target_broken) - len(target_broken))
        + (len(raw_target_formed) - len(target_formed))
    )

    has_intramolecular_change = bool(probe_broken or probe_formed or target_broken or target_formed)

    # Build human-readable details
    symbols_init = initial_atoms.get_chemical_symbols()
    symbols_final = final_atoms.get_chemical_symbols()

    details_parts = []

    if probe_broken:
        for a, b in sorted(probe_broken):
            d_init = np.linalg.norm(pos_init[a] - pos_init[b])
            d_final = np.linalg.norm(pos_final[a] - pos_final[b])
            pct = abs(d_final - d_init) / d_init * 100
            details_parts.append(
                f"PROBE broke {symbols_init[a]}[{a}]-{symbols_init[b]}[{b}]: "
                f"{d_init:.2f}A -> {d_final:.2f}A ({pct:.0f}%)"
            )
    if probe_formed:
        for a, b in sorted(probe_formed):
            d_init = np.linalg.norm(pos_init[a] - pos_init[b])
            d_final = np.linalg.norm(pos_final[a] - pos_final[b])
            pct = abs(d_final - d_init) / d_final * 100
            details_parts.append(
                f"PROBE formed {symbols_final[a]}[{a}]-{symbols_final[b]}[{b}]: "
                f"{d_init:.2f}A -> {d_final:.2f}A ({pct:.0f}%)"
            )
    if target_broken:
        for a, b in sorted(target_broken):
            d_init = np.linalg.norm(pos_init[a] - pos_init[b])
            d_final = np.linalg.norm(pos_final[a] - pos_final[b])
            pct = abs(d_final - d_init) / d_init * 100
            details_parts.append(
                f"TARGET broke {symbols_init[a]}[{a}]-{symbols_init[b]}[{b}]: "
                f"{d_init:.2f}A -> {d_final:.2f}A ({pct:.0f}%)"
            )
    if target_formed:
        for a, b in sorted(target_formed):
            d_init = np.linalg.norm(pos_init[a] - pos_init[b])
            d_final = np.linalg.norm(pos_final[a] - pos_final[b])
            pct = abs(d_final - d_init) / d_final * 100
            details_parts.append(
                f"TARGET formed {symbols_final[a]}[{a}]-{symbols_final[b]}[{b}]: "
                f"{d_init:.2f}A -> {d_final:.2f}A ({pct:.0f}%)"
            )

    # --- Intermolecular bond detection (B3: semantic classifier) ---
    # New covalent bonds between probe and target indicate either:
    # 1. ML artifact (spurious bond formation)
    # 2. Real chemical reaction / chemisorption
    # Either way, the binding energy formula E = E(P+T) - E(P) - E(T) loses meaning.
    inter_new_details = []
    for a, b in sorted(inter_new):
        d_init = np.linalg.norm(pos_init[a] - pos_init[b])
        d_final = np.linalg.norm(pos_final[a] - pos_final[b])
        # Determine which molecule each atom belongs to
        mol_a = "probe" if a < n_probe else "target"
        mol_b = "probe" if b < n_probe else "target"
        inter_new_details.append({
            "atoms": [a, b],
            "symbols": [symbols_final[a], symbols_final[b]],
            "molecules": [mol_a, mol_b],
            "d_init": round(float(d_init), 3),
            "d_final": round(float(d_final), 3),
        })
        details_parts.append(
            f"INTER formed {symbols_final[a]}[{a}]({mol_a})-{symbols_final[b]}[{b}]({mol_b}): "
            f"{d_init:.2f}A -> {d_final:.2f}A"
        )

    # Classify interaction type based on intermolecular bond formation
    # This is a semantic classifier, not a "problem" flag
    has_covalent_interaction = len(inter_new) > 0
    interaction_type = "covalent" if has_covalent_interaction else "non_covalent"
    energy_interpretation = "reaction_energy" if has_covalent_interaction else "binding_energy"

    if not details_parts:
        details = "Topology preserved"
    else:
        details = "; ".join(details_parts)

    return {
        "topology_preserved": not has_intramolecular_change,
        "interaction_type": interaction_type,
        "energy_interpretation": energy_interpretation,
        "confidence": "high" if not has_intramolecular_change else "needs_verification",
        "probe_broken": [list(b) for b in sorted(probe_broken)],
        "probe_formed": [list(b) for b in sorted(probe_formed)],
        "target_broken": [list(b) for b in sorted(target_broken)],
        "target_formed": [list(b) for b in sorted(target_formed)],
        "inter_new": [list(b) for b in sorted(inter_new)],
        "inter_new_details": inter_new_details,
        "n_intramolecular_changes": len(probe_broken) + len(probe_formed) + len(target_broken) + len(target_formed),
        "n_intermolecular_new": len(inter_new),
        "n_filtered_noise": n_filtered,
        "details": details,
    }


# ==================== Geometry Guard ====================

# Van der Waals clash overlap threshold (Angstroms).
# Basis: MolProbity defines "severe clash" as overlap > 0.4 Å beyond vdW contact.
# Atoms closer than (vdW_sum - VDW_CLASH_OVERLAP) are in severe steric clash.
# Reference: https://grade.globalphasing.org (MolProbity documentation)
VDW_CLASH_OVERLAP = 0.4  # Angstroms

# Bond strain upper limit (percentage).
# Basis: Longest known stable C-C bond is ~1.80 A (17% above 1.54 A).
# 30% is a conservative upper bound covering all known stable bond
# deformations. Topology guard already catches >15% as "broken".
# This guard flags the 15-30% range as "strained but intact".
STRAIN_UPPER_PCT = 30.0


def _get_vdw_radius(symbol: str) -> float:
    """Get van der Waals radius for an element, with fallback."""
    z = atomic_numbers.get(symbol, 0)
    r = vdw_radii[z] if z < len(vdw_radii) else 0.0
    if r < 0.1:
        # Fallback: use 2.0 A for elements missing vdW data
        return 2.0
    return r


def validate_geometry(
    initial_atoms: Atoms,
    final_atoms: Atoms,
    n_probe: int,
    final_bonds: Optional[Set[Tuple[int, int]]] = None,
    mult: float = 1.2,
) -> dict:
    """
    Check optimized structure for atom clashes and severe bond strain.

    Atom clashes: non-bonded atom pairs closer than (vdW_sum - 0.4Å), consistent with
                  MolProbity's "severe clash" definition (overlap > 0.4Å beyond vdW contact).
    Bond strain:  intramolecular bonds with 15-30% length change (strained but not broken).

    Args:
        initial_atoms: Structure before optimization.
        final_atoms: Structure after optimization.
        n_probe: Number of probe atoms (first n_probe atoms).
        final_bonds: Pre-computed bond graph of final structure. If None,
                     computed internally.
        mult: Covalent radius multiplier for bond detection.

    Returns:
        dict with geometry validation results.
    """
    if final_bonds is None:
        final_bonds = get_bond_graph(final_atoms, mult=mult)

    symbols = final_atoms.get_chemical_symbols()
    pos_final = final_atoms.get_positions()
    pos_init = initial_atoms.get_positions()
    n_atoms = len(final_atoms)

    # --- Check A: Atom clashes (non-bonded pairs) ---
    # MolProbity-consistent: clash if dist < vdw_sum - 0.4 Å (overlap > 0.4 Å)
    atom_clashes = []
    for i in range(n_atoms):
        r_i = _get_vdw_radius(symbols[i])
        for j in range(i + 1, n_atoms):
            if (i, j) in final_bonds:
                continue  # bonded pair, skip
            dist = np.linalg.norm(pos_final[i] - pos_final[j])
            r_j = _get_vdw_radius(symbols[j])
            vdw_sum = r_i + r_j
            clash_threshold = vdw_sum - VDW_CLASH_OVERLAP
            if dist < clash_threshold:
                overlap = vdw_sum - dist  # How much the vdW spheres overlap
                atom_clashes.append({
                    "atoms": [i, j],
                    "symbols": [symbols[i], symbols[j]],
                    "distance": round(float(dist), 3),
                    "vdw_sum": round(float(vdw_sum), 3),
                    "overlap": round(float(overlap), 3),  # MolProbity-style overlap
                })

    # --- Check B: Bond strain (intramolecular, 15-30% range) ---
    # Get initial bond graph to compare
    initial_bonds = get_bond_graph(initial_atoms, mult=mult)
    init_probe, init_target, _ = split_bonds(initial_bonds, n_probe, n_atoms)

    # Only check intramolecular bonds that exist in BOTH initial and final
    final_probe, final_target, _ = split_bonds(final_bonds, n_probe, n_atoms)
    preserved_probe = init_probe & final_probe
    preserved_target = init_target & final_target

    strained_bonds = []
    for a, b in preserved_probe | preserved_target:
        d_init = np.linalg.norm(pos_init[a] - pos_init[b])
        d_final = np.linalg.norm(pos_final[a] - pos_final[b])
        if d_init < 0.1:
            continue
        strain_pct = abs(d_final - d_init) / d_init * 100
        if 15.0 < strain_pct <= STRAIN_UPPER_PCT:
            mol = "probe" if a < n_probe else "target"
            strained_bonds.append({
                "atoms": [a, b],
                "symbols": [symbols[a], symbols[b]],
                "molecule": mol,
                "d_init": round(float(d_init), 3),
                "d_final": round(float(d_final), 3),
                "strain_pct": round(float(strain_pct), 1),
            })

    geometry_ok = len(atom_clashes) == 0

    # Build details string
    details_parts = []
    for c in atom_clashes:
        details_parts.append(
            f"CLASH {c['symbols'][0]}[{c['atoms'][0]}]-{c['symbols'][1]}[{c['atoms'][1]}]: "
            f"{c['distance']:.2f}A (vdW sum {c['vdw_sum']:.2f}A, overlap {c['overlap']:.2f}A)"
        )
    for s in strained_bonds:
        details_parts.append(
            f"STRAIN {s['molecule'].upper()} {s['symbols'][0]}[{s['atoms'][0]}]-"
            f"{s['symbols'][1]}[{s['atoms'][1]}]: "
            f"{s['d_init']:.2f}A -> {s['d_final']:.2f}A ({s['strain_pct']:.0f}%)"
        )

    return {
        "geometry_ok": geometry_ok,
        "atom_clashes": atom_clashes,
        "strained_bonds": strained_bonds,
        "n_clashes": len(atom_clashes),
        "n_strained": len(strained_bonds),
        "details": "; ".join(details_parts) if details_parts else "Geometry OK",
    }


# ==================== Energy Guard ====================

# Maximum binding energy per atom of the smaller molecule (eV/atom).
# Basis: [FHF]⁻ bifluoride ion is one of the strongest hydrogen bond systems,
# with significant 3-center-4-electron (3c-4e) delocalized character.
# Its interaction energy is ~160-192 kJ/mol for 3 atoms ≈ 0.55-0.66 eV/atom.
# Values exceeding this are stronger than [FHF]⁻, indicating ML potential failure.
# Reference: Crystals 2016, 6(1), 3 (MDPI)
MAX_BINDING_PER_ATOM = 0.65  # eV/atom

# Maximum total non-covalent binding energy (eV).
# Basis: Cucurbit[7]uril (CB[7]) host-guest complexes exhibit ultrahigh binding
# affinities with Ka ~ 10^15-10^16 M⁻¹, corresponding to |ΔG°| ~ 21-22 kcal/mol
# at 298K. We use 2× this range (~50 kcal/mol, 2.2 eV) as a generous upper bound.
# Any small-molecule complex exceeding this is non-physical for non-covalent binding.
# Reference: Phys. Chem. Chem. Phys. 2019, RSC Publishing
MAX_BINDING_TOTAL = 2.2  # eV


def validate_energy(
    binding_eV: float,
    n_atoms_smaller: int,
) -> dict:
    """
    Check if a binding energy is physically reasonable.

    Uses two independent criteria based on established domain knowledge:
    1. Per-atom limit from [FHF]⁻ (strongest H-bond system with 3c-4e character).
    2. Total limit from CB[7] (ultrahigh-affinity host-guest, Ka ~ 10^15-10^16 M⁻¹).

    Args:
        binding_eV: Binding energy in eV (negative = favorable).
        n_atoms_smaller: Atom count of the smaller molecule in the pair.

    Returns:
        dict with energy validation results.
    """
    abs_binding = abs(binding_eV)
    binding_per_atom = abs_binding / max(n_atoms_smaller, 1)

    flags = []
    if binding_per_atom > MAX_BINDING_PER_ATOM:
        flags.append(
            f"exceeds_strongest_known_per_atom "
            f"({binding_per_atom:.3f} > {MAX_BINDING_PER_ATOM} eV/atom)"
        )
    if abs_binding > MAX_BINDING_TOTAL:
        flags.append(
            f"exceeds_strongest_known_total "
            f"({abs_binding:.3f} > {MAX_BINDING_TOTAL} eV)"
        )

    return {
        "energy_ok": len(flags) == 0,
        "binding_eV": round(float(binding_eV), 4),
        "binding_per_atom_eV": round(float(binding_per_atom), 4),
        "n_atoms_smaller": n_atoms_smaller,
        "flags": flags,
        "details": "; ".join(flags) if flags else "Energy OK",
    }


# ==================== Enhanced Energy Guard ====================

# Suspicious and extreme thresholds (eV).
# Basis: Empirical analysis of 1187 molecular pairs from RAPIDS scans.
# - Median energy range: 0.19 eV (stable ML predictions)
# - 75th percentile: 3.2 eV (transition to suspicious)
# - 90th percentile: 13.3 eV (clearly problematic)
# Using 3 eV as suspicious (~75th percentile) and 10 eV as extreme (~90th percentile).
ENERGY_SUSPICIOUS = 3.0   # eV - ~75th percentile, recommend DFT verification
ENERGY_EXTREME = 10.0     # eV - ~90th percentile, strongly recommend DFT


def validate_energy_enhanced(
    binding_eV: float,
    n_atoms_smaller: int,
) -> dict:
    """
    Enhanced energy validation with tiered severity levels.

    Three tiers:
    1. Normal: |E| <= 2.2 eV (original MAX_BINDING_TOTAL)
    2. Suspicious: 2.2 < |E| <= 5 eV (warn, recommend DFT)
    3. Extreme: |E| > 5 eV (confidence="low", strongly recommend DFT)
    4. Non-physical: |E| > 10 eV (almost certainly ML artifact)

    Args:
        binding_eV: Binding energy in eV (negative = favorable).
        n_atoms_smaller: Atom count of the smaller molecule.

    Returns:
        dict with enhanced energy validation results.
    """
    abs_binding = abs(binding_eV)
    binding_per_atom = abs_binding / max(n_atoms_smaller, 1)

    # Determine severity level
    if abs_binding > ENERGY_EXTREME:
        severity = "non_physical"
        energy_ok = False
        confidence_impact = "low"
        dft_recommendation = "strongly_recommended"
    elif abs_binding > ENERGY_SUSPICIOUS:
        severity = "extreme"
        energy_ok = False
        confidence_impact = "low"
        dft_recommendation = "strongly_recommended"
    elif abs_binding > MAX_BINDING_TOTAL:
        severity = "suspicious"
        energy_ok = False
        confidence_impact = "medium"
        dft_recommendation = "recommended"
    else:
        severity = "normal"
        energy_ok = True
        confidence_impact = None
        dft_recommendation = None

    # Also check per-atom limit
    per_atom_flags = []
    if binding_per_atom > MAX_BINDING_PER_ATOM:
        per_atom_flags.append(
            f"exceeds_strongest_known_per_atom "
            f"({binding_per_atom:.3f} > {MAX_BINDING_PER_ATOM} eV/atom)"
        )
        if energy_ok:  # Only downgrade if not already flagged
            energy_ok = False
            confidence_impact = "medium"

    # Build details
    details_parts = []
    if severity == "non_physical":
        details_parts.append(
            f"NON-PHYSICAL: |E|={abs_binding:.2f} eV > {ENERGY_EXTREME} eV extreme limit"
        )
    elif severity == "extreme":
        details_parts.append(
            f"EXTREME: |E|={abs_binding:.2f} eV > {ENERGY_SUSPICIOUS} eV suspicious limit"
        )
    elif severity == "suspicious":
        details_parts.append(
            f"SUSPICIOUS: |E|={abs_binding:.2f} eV > {MAX_BINDING_TOTAL} eV (CB7 record)"
        )
    details_parts.extend(per_atom_flags)

    return {
        "energy_ok": energy_ok,
        "severity": severity,
        "binding_eV": round(float(binding_eV), 4),
        "binding_per_atom_eV": round(float(binding_per_atom), 4),
        "n_atoms_smaller": n_atoms_smaller,
        "confidence_impact": confidence_impact,
        "dft_recommendation": dft_recommendation,
        "per_atom_flags": per_atom_flags,
        "details": "; ".join(details_parts) if details_parts else "Energy OK",
    }


# ==================== Energy Consistency Guard ====================

# Geometry similarity thresholds (Angstroms).
# Basis: Empirical analysis of 1187 molecular pairs from RAPIDS scans.
# - 90th percentile COM range: 1.41 Å
# - 90th percentile contact range: 0.48 Å
# Configurations with geometry variation below these thresholds are considered "similar".
GEOM_COM_TOLERANCE = 1.5      # Å - ~90th percentile COM range
GEOM_CONTACT_TOLERANCE = 0.5  # Å - ~90th percentile contact range


def validate_energy_consistency(
    results: List[Dict],
    energy_variance_limit: float = ENERGY_SUSPICIOUS,
    extreme_limit: float = ENERGY_EXTREME,
) -> dict:
    """
    Detect ML energy prediction instability across scan configurations.

    Core principle: Similar geometries should have similar energies.
    Large energy variance with small geometric variance indicates ML instability.

    Args:
        results: List of scan results, each containing:
            - 'energy_eV': Binding energy in eV
            - 'com_distance': Center-of-mass distance (Å), optional
            - 'min_distance': Minimum atomic distance (Å), optional
            - 'run_name': Configuration identifier, optional
        energy_variance_limit: Energy range threshold for "unstable" (default 5 eV)
        extreme_limit: Energy range threshold for "extreme_instability" (default 10 eV)

    Returns:
        dict with:
            - stable (bool): True if energies consistent with geometry
            - status: 'stable' | 'unstable' | 'extreme_instability'
            - selection_mode: 'minimum' | 'median' | 'manual_review'
            - recommended_value_eV: Robust energy estimate
            - dft_recommended (bool): Whether DFT verification is recommended
            - extreme_outliers: List of configuration names flagged as extreme
            - statistics: Energy and geometry statistics
            - details: Human-readable explanation
    """
    if len(results) < 2:
        return {
            "stable": True,
            "status": "stable",
            "selection_mode": "minimum",
            "recommended_value_eV": results[0]["energy_eV"] if results else None,
            "dft_recommended": False,
            "extreme_outliers": [],
            "statistics": {},
            "details": "Insufficient data for consistency check (N < 2)",
        }

    # Extract energies
    energies = np.array([r["energy_eV"] for r in results])
    e_min, e_max = float(np.min(energies)), float(np.max(energies))
    e_range = e_max - e_min
    e_median = float(np.median(energies))
    e_mean = float(np.mean(energies))
    e_std = float(np.std(energies))

    # IQR-based outlier detection
    if len(energies) >= 4:
        e_q1, e_q3 = np.percentile(energies, [25, 75])
        e_iqr = float(e_q3 - e_q1)
        extreme_low = e_q1 - 3 * e_iqr
        extreme_high = e_q3 + 3 * e_iqr
    else:
        e_q1, e_q3, e_iqr = e_min, e_max, e_range
        extreme_low = e_min - 3 * e_std
        extreme_high = e_max + 3 * e_std

    # Extract geometry metrics (optional)
    com_distances = [r.get("com_distance") for r in results if r.get("com_distance") is not None]
    min_distances = [r.get("min_distance") for r in results if r.get("min_distance") is not None]

    # Calculate geometry variance
    if com_distances:
        com_range = max(com_distances) - min(com_distances)
    else:
        com_range = None

    if min_distances:
        contact_range = max(min_distances) - min(min_distances)
    else:
        contact_range = None

    # Determine if geometries are similar
    geom_similar = True
    if com_range is not None and com_range > GEOM_COM_TOLERANCE:
        geom_similar = False
    if contact_range is not None and contact_range > GEOM_CONTACT_TOLERANCE:
        geom_similar = False

    # If no geometry data, assume similar (conservative)
    if com_range is None and contact_range is None:
        geom_similar = True  # Assume similar, rely on energy stats alone

    # Identify extreme outliers
    extreme_outliers = []
    for r in results:
        e = r["energy_eV"]
        if e < extreme_low or e > extreme_high:
            name = r.get("run_name", f"E={e:.2f}")
            extreme_outliers.append(name)

    # Also flag the minimum if it's suspiciously low compared to median
    best_is_extreme = e_min < extreme_low

    # Determine stability status
    if e_range > extreme_limit and geom_similar:
        status = "extreme_instability"
        selection_mode = "manual_review"
        dft_recommended = True
        stable = False
    elif (e_range > energy_variance_limit and geom_similar) or best_is_extreme:
        status = "unstable"
        selection_mode = "median"
        dft_recommended = True
        stable = False
    else:
        status = "stable"
        selection_mode = "minimum"
        dft_recommended = False
        stable = True

    # Recommended value
    if selection_mode == "minimum":
        recommended_value = e_min
    else:
        recommended_value = e_median

    # Build statistics dict
    statistics = {
        "n_configs": len(results),
        "energy_min_eV": round(e_min, 4),
        "energy_max_eV": round(e_max, 4),
        "energy_range_eV": round(e_range, 4),
        "energy_median_eV": round(e_median, 4),
        "energy_mean_eV": round(e_mean, 4),
        "energy_std_eV": round(e_std, 4),
        "energy_iqr_eV": round(e_iqr, 4) if e_iqr else None,
        "geometry_similar": geom_similar,
    }
    if com_range is not None:
        statistics["com_distance_range_A"] = round(com_range, 3)
    if contact_range is not None:
        statistics["min_distance_range_A"] = round(contact_range, 3)

    # Build details message
    if status == "extreme_instability":
        details = (
            f"EXTREME ML INSTABILITY: {e_range:.1f} eV energy range for "
            f"geometrically similar structures (COM range {com_range:.2f} Å, "
            f"contact range {contact_range:.2f} Å). "
            f"Best energy ({e_min:.2f} eV) is likely an artifact. "
            f"DFT verification required."
        ) if com_range is not None else (
            f"EXTREME ML INSTABILITY: {e_range:.1f} eV energy range. "
            f"Best energy ({e_min:.2f} eV) is likely an artifact. "
            f"DFT verification required."
        )
    elif status == "unstable":
        details = (
            f"ML INSTABILITY DETECTED: {e_range:.1f} eV energy range. "
            f"Reporting median ({e_median:.2f} eV) instead of minimum ({e_min:.2f} eV). "
            f"DFT verification recommended."
        )
    else:
        details = "Energy distribution consistent with geometric variation."

    return {
        "stable": stable,
        "status": status,
        "selection_mode": selection_mode,
        "recommended_value_eV": round(recommended_value, 4),
        "dft_recommended": dft_recommended,
        "extreme_outliers": extreme_outliers,
        "best_is_extreme_outlier": best_is_extreme,
        "statistics": statistics,
        "details": details,
    }
