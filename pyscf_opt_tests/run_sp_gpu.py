#!/usr/bin/env python3
"""GPU4PySCF single-point DFT — the paper's three **SP** intermediate arms.

This script realizes the *single-point (SP)* half of the 9-arm multi-fidelity
matrix described in the RAPIDS paper
(``ICML2025/sections/appendix_methods_details.tex``, "Intermediate DFT arms used
in the 9-arm matrix"). Each SP arm evaluates the RAPIDS-committed complex
*together with* the corresponding RAPIDS-optimized isolated probe and target
monomers at one explicit single-point level, **in vacuum, no BSSE/counterpoise,
no implicit solvent**, and reports the binding energy

    dE_bind = E_complex - E_probe - E_target.

The three functionals (dispersion is per-functional):

    * ``pbe-d3bj``    : PBE          + D3BJ            / def2-TZVP
    * ``wb97x-d3bj``  : wB97X        + D3BJ            / def2-TZVP
    * ``wb97m-v``     : wB97M-V (VV10 built in; NO D3BJ) / def2-TZVP

omega-B97M-V carries VV10 nonlocal correlation internally, so it is run with no
D3BJ dispersion (adding D3BJ would double-count). PySCF activates the VV10 NLC
automatically from the ``wb97m-v`` xc string.

Geometries
----------
The SP arms run on the RAPIDS-committed geometries as-is (no DFT relaxation; that
is what the companion ``run_geomopt_gpu.py`` / GeoSP arms do). Provide the three
species explicitly::

    --complex complex.vasp --probe probe.xyz --target target.xyz

or rely on the env-var defaults (``RAPIDS_SP_COMPLEX`` / ``RAPIDS_SP_PROBE`` /
``RAPIDS_SP_TARGET``; ``RAPIDS_SP_STRUCTURE`` is a back-compat alias for the
complex). If only the complex is given, only its absolute energy is reported
(no binding energy).

Charge & spin
-------------
Fragment charges are carried independently (``--charge-complex/-probe/-target``);
the paper assumes closed-shell singlets unless the dataset requires otherwise, so
spin defaults to a singlet for even-electron systems and a doublet otherwise (or
set ``--spin*`` explicitly, as a 2S+1 multiplicity).

Usage
-----
    # all three SP arms on a probe/target/complex triple
    python run_sp_gpu.py --functional all \
        --complex complex.vasp --probe probe.xyz --target target.xyz

    # a single arm
    python run_sp_gpu.py --functional wb97m-v \
        --complex complex.vasp --probe probe.xyz --target target.xyz

    # a charged fragment (e.g. anionic probe, neutral target, anionic complex)
    python run_sp_gpu.py --functional all --complex c.vasp --probe p.xyz --target t.xyz \
        --charge-complex -1 --charge-probe -1 --charge-target 0

Requires ``gpu4pyscf`` + a GPU. The DFT layer is GPU4PySCF; vacuum only.
"""
import argparse
import json
import os
import time
import traceback
from pathlib import Path

from ase.io import read

try:
    from pyscf import gto
    from pyscf.data import elements
except Exception as exc:  # pragma: no cover - depends on env (GPU host only)
    raise SystemExit(f"pyscf import failed: {exc} (install pyscf + gpu4pyscf)")

try:
    from gpu4pyscf.dft import rks, uks
except Exception as exc:  # pragma: no cover - depends on env (GPU host only)
    raise SystemExit(f"gpu4pyscf import failed: {exc} (install gpu4pyscf; GPU required)")


HARTREE_TO_EV = 27.211386245988
HARTREE_TO_KCAL = 627.5094740631

# ---------------------------------------------------------------------------
# The three paper functionals. Dispersion is per-functional:
#   * PBE  -> D3BJ          * wB97X -> D3BJ
#   * wB97M-V -> none (VV10 nonlocal correlation is built into the functional).
# ``xc``  : libxc/GPU4PySCF functional string (passed to mf.xc).
# ``disp``: value for mf.disp ("d3bj" or None). None => no DFT-D correction.
# ``nlc`` : explicit nonlocal-correlation override for mf.nlc; None lets PySCF
#           decide from the xc string (wb97m-v auto-enables VV10).
# ``basis``: SP basis for these arms is def2-TZVP (vacuum SP on the given pose).
# ---------------------------------------------------------------------------
# wB97X-D3BJ: use PySCF's recognized combined xc string ``wb97x-d3bj``. PySCF's
# dispersion whitelist maps it to base ``wb97x-v`` with the VV10 NLC turned OFF
# plus D3BJ -- NOT the bare 2008 ``wb97x`` (a different parameterization that
# would give wrong energies). We also set ``disp="d3bj"`` explicitly (mf.disp is
# the dispersion source in GPU4PySCF; the whitelist handles base/NLC, no double-count).
FUNCTIONALS = {
    "pbe-d3bj": {"xc": "PBE", "disp": "d3bj", "nlc": None},
    "wb97x-d3bj": {"xc": "wb97x-d3bj", "disp": "d3bj", "nlc": None},
    "wb97m-v": {"xc": "wb97m-v", "disp": None, "nlc": None},
}
SP_BASIS = "def2-tzvp"

# Back-compat single-structure default; the preferred inputs are the three
# species below.
DEFAULT_STRUCTURE = os.environ.get("RAPIDS_SP_STRUCTURE", "probe_target_vacuum_optimized.vasp")
DEFAULT_COMPLEX = os.environ.get("RAPIDS_SP_COMPLEX", DEFAULT_STRUCTURE)
DEFAULT_PROBE = os.environ.get("RAPIDS_SP_PROBE")
DEFAULT_TARGET = os.environ.get("RAPIDS_SP_TARGET")
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "sp_results"


def _count_electrons(symbols, charge):
    return sum(elements.charge(sym) for sym in symbols) - charge


def _select_spin(spin_arg, nelec):
    """Return a 2S+1 multiplicity (singlet for even electrons unless overridden)."""
    if spin_arg is not None:
        return spin_arg
    return 1 if nelec % 2 == 0 else 2


def _attach_scf_counter(mf):
    state = {"count": 0}
    existing_callback = getattr(mf, "callback", None)

    def _callback(envs):
        cycle = envs.get("cycle")
        if isinstance(cycle, int) and cycle >= 0:
            state["count"] = max(state["count"], cycle + 1)
        else:
            state["count"] += 1
        if callable(existing_callback):
            existing_callback(envs)

    mf.callback = _callback
    return state


def run_single_point(structure_path, basis, xc, disp, nlc, charge, spin,
                     max_cycle, conv_tol, conv_tol_grad, grids_level):
    """Vacuum single point on one structure; returns a result dict."""
    atoms = read(structure_path)
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    atom_list = [(sym, pos) for sym, pos in zip(symbols, positions)]

    nelec = _count_electrons(symbols, charge)
    spin_multiplicity = _select_spin(spin, nelec)

    mol = gto.M(
        atom=atom_list,
        basis=basis,
        charge=charge,
        spin=spin_multiplicity - 1,
        unit="Angstrom",
        verbose=4,
    )

    if spin_multiplicity == 1 and nelec % 2 == 0:
        mf = rks.RKS(mol)
    else:
        mf = uks.UKS(mol)
    mf.xc = xc
    # Per-functional dispersion: D3BJ for PBE/wB97X, nothing for wB97M-V (VV10).
    if disp:
        mf.disp = disp
    # Only override the nonlocal correlation if explicitly requested; for
    # wb97m-v PySCF enables VV10 automatically from the xc string.
    if nlc:
        mf.nlc = nlc
    # Vacuum only: NO implicit solvent (the paper DFT arms are strictly vacuum).
    if hasattr(mf, "grids") and grids_level is not None:
        mf.grids.level = grids_level
    mf.max_cycle = max_cycle
    mf.conv_tol = conv_tol
    if conv_tol_grad is not None:
        mf.conv_tol_grad = conv_tol_grad

    scf_state = _attach_scf_counter(mf)

    start = time.time()
    energy = mf.kernel()
    elapsed = time.time() - start

    return {
        "structure": str(structure_path),
        "basis": basis,
        "xc": xc,
        "dispersion": disp,
        "nlc": nlc if nlc else ("auto" if "-v" in xc.lower() else None),
        "charge": charge,
        "spin_multiplicity": spin_multiplicity,
        "electrons": nelec,
        "natoms": len(symbols),
        "scf_max_cycle": max_cycle,
        "scf_conv_tol": conv_tol,
        "scf_conv_tol_grad": conv_tol_grad,
        "scf_cycles": scf_state["count"] if scf_state["count"] > 0 else None,
        "solvent": None,
        "energy_hartree": float(energy),
        "energy_eV": float(energy) * HARTREE_TO_EV,
        "time_seconds": round(elapsed, 1),
        "converged": bool(getattr(mf, "converged", False)),
    }


def run_arm(functional, species, max_cycle, conv_tol, conv_tol_grad,
            grids_level, out_dir):
    """Run one SP arm over the provided species and compute the binding energy.

    ``species`` is an ordered dict-like mapping label -> (path, charge, spin).
    Returns the per-arm result dict.
    """
    spec = FUNCTIONALS[functional]
    arm_dir = out_dir / functional
    arm_dir.mkdir(parents=True, exist_ok=True)

    per_species = {}
    for label, (path, charge, spin) in species.items():
        if path is None:
            continue
        res = run_single_point(
            structure_path=path,
            basis=SP_BASIS,
            xc=spec["xc"],
            disp=spec["disp"],
            nlc=spec["nlc"],
            charge=charge,
            spin=spin,
            max_cycle=max_cycle,
            conv_tol=conv_tol,
            conv_tol_grad=conv_tol_grad,
            grids_level=grids_level,
        )
        per_species[label] = res
        print(f"  [{functional}/{label}] E={res['energy_hartree']:.8f} Ha "
              f"converged={res['converged']} ({res['time_seconds']}s)")

    arm = {
        "arm": f"{functional}_SP",
        "functional": functional,
        "xc": spec["xc"],
        "dispersion": spec["disp"],
        "basis": SP_BASIS,
        "solvent": None,
        "species": per_species,
    }

    have = {"complex", "probe", "target"} <= set(per_species)
    if have:
        e_c = per_species["complex"]["energy_hartree"]
        e_p = per_species["probe"]["energy_hartree"]
        e_t = per_species["target"]["energy_hartree"]
        de = e_c - e_p - e_t
        arm["binding_energy_hartree"] = de
        arm["binding_energy_eV"] = de * HARTREE_TO_EV
        arm["binding_energy_kcal_mol"] = de * HARTREE_TO_KCAL
        print(f"  [{functional}_SP] dE_bind = {de * HARTREE_TO_KCAL:.3f} kcal/mol")
    else:
        arm["binding_energy_hartree"] = None
        arm["note"] = "binding energy needs complex+probe+target; only available species reported"

    with open(arm_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(arm, f, indent=2, ensure_ascii=True)
    return arm


def main():
    parser = argparse.ArgumentParser(
        description="GPU4PySCF single-point DFT — paper SP arms (vacuum, def2-TZVP)")
    parser.add_argument("--functional", default="all",
                        choices=list(FUNCTIONALS) + ["all"],
                        help="Which SP arm(s) to run (default: all three).")
    # Species inputs (preferred). --structure is a back-compat alias for --complex.
    parser.add_argument("--complex", dest="complex_path", default=None,
                        help="Complex geometry (default: $RAPIDS_SP_COMPLEX).")
    parser.add_argument("--probe", dest="probe_path", default=None,
                        help="Isolated probe geometry (default: $RAPIDS_SP_PROBE).")
    parser.add_argument("--target", dest="target_path", default=None,
                        help="Isolated target geometry (default: $RAPIDS_SP_TARGET).")
    parser.add_argument("--structure", default=None,
                        help="Back-compat alias for --complex (single-structure SP).")
    # Per-fragment charge (q_PT, q_P, q_T carried independently).
    parser.add_argument("--charge-complex", type=int, default=0)
    parser.add_argument("--charge-probe", type=int, default=0)
    parser.add_argument("--charge-target", type=int, default=0)
    # Per-fragment spin multiplicity (2S+1); None => singlet for even electrons.
    parser.add_argument("--spin-complex", type=int, default=None)
    parser.add_argument("--spin-probe", type=int, default=None)
    parser.add_argument("--spin-target", type=int, default=None)
    parser.add_argument("--max-cycle", type=int, default=200)
    parser.add_argument("--conv-tol", type=float, default=1e-9)
    parser.add_argument("--conv-tol-grad", type=float, default=None)
    parser.add_argument("--grids-level", type=int, default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    def _resolve(p):
        return Path(p).expanduser().resolve() if p else None

    complex_path = _resolve(args.complex_path or args.structure or DEFAULT_COMPLEX)
    probe_path = _resolve(args.probe_path or DEFAULT_PROBE)
    target_path = _resolve(args.target_path or DEFAULT_TARGET)
    out_dir = _resolve(args.out_dir) if args.out_dir else DEFAULT_OUT_DIR

    species = {
        "complex": (complex_path, args.charge_complex, args.spin_complex),
        "probe": (probe_path, args.charge_probe, args.spin_probe),
        "target": (target_path, args.charge_target, args.spin_target),
    }
    if probe_path is None or target_path is None:
        print("NOTE: probe and/or target not provided -> reporting complex "
              "absolute energy only (no binding energy). Pass --probe/--target "
              "(or $RAPIDS_SP_PROBE/$RAPIDS_SP_TARGET) for the full SP arm.")

    functionals = list(FUNCTIONALS) if args.functional == "all" else [args.functional]

    summary = {}
    for functional in functionals:
        try:
            arm = run_arm(
                functional=functional,
                species=species,
                max_cycle=args.max_cycle,
                conv_tol=args.conv_tol,
                conv_tol_grad=args.conv_tol_grad,
                grids_level=args.grids_level,
                out_dir=out_dir,
            )
            summary[f"{functional}_SP"] = arm
        except Exception as exc:
            err = traceback.format_exc()
            (out_dir / functional).mkdir(parents=True, exist_ok=True)
            (out_dir / functional / "error.txt").write_text(err, encoding="utf-8")
            summary[f"{functional}_SP"] = {"error": str(exc), "error_type": type(exc).__name__}
            print(f"[{functional}_SP] failed: {type(exc).__name__}: {exc}")

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)


if __name__ == "__main__":
    main()
