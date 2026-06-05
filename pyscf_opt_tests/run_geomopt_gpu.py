#!/usr/bin/env python3
"""GPU4PySCF geometry-reopt + single-point — the paper's three **GeoSP** arms.

This script realizes the *geometry-reoptimization + single-point (GeoSP)* half of
the 9-arm multi-fidelity matrix in the RAPIDS paper
(``ICML2025/sections/appendix_methods_details.tex``, "Intermediate DFT arms used
in the 9-arm matrix"). Each GeoSP arm is a **two-step**, vacuum-only,
no-BSSE/no-solvent protocol applied to the complex *and* the two isolated
monomers:

    Step 1 (Geo): reoptimize the geometry at ``functional/def2-TZVP`` with
                  TightOpt / TightSCF convergence.
    Step 2 (SP) : a *same-functional* ``def2-TZVPD`` single point on that
                  optimized geometry.

The **reported** per-species energy is the def2-TZVPD single-point energy (NOT
the optimization-basis energy), and the binding energy is

    dE_bind = E_complex - E_probe - E_target

from the three def2-TZVPD single points.

The three functionals (dispersion is per-functional, identical to the SP arms):

    * ``pbe-d3bj``    : PBE          + D3BJ              (Geo: def2-TZVP -> SP: def2-TZVPD)
    * ``wb97x-d3bj``  : wB97X        + D3BJ              (Geo: def2-TZVP -> SP: def2-TZVPD)
    * ``wb97m-v``     : wB97M-V (VV10 built in; NO D3BJ) (Geo: def2-TZVP -> SP: def2-TZVPD)

omega-B97M-V carries VV10 nonlocal correlation internally, so it is run with no
D3BJ dispersion in both steps.

Inputs
------
Provide the three RAPIDS-committed species explicitly (each is independently
reoptimized)::

    --complex complex.vasp --probe probe.xyz --target target.xyz

or via env-var defaults (``RAPIDS_GEOMOPT_COMPLEX`` / ``RAPIDS_GEOMOPT_PROBE`` /
``RAPIDS_GEOMOPT_TARGET``; ``RAPIDS_GEOMOPT_STRUCTURE`` is a back-compat alias for
the complex). If only the complex is provided, only its optimized def2-TZVPD
energy is reported (no binding energy).

Charge & spin
-------------
Fragment charges are carried independently (``--charge-complex/-probe/-target``);
closed-shell singlets are assumed for even-electron systems unless ``--spin*``
(a 2S+1 multiplicity) is set.

Usage
-----
    # all three GeoSP arms over a probe/target/complex triple
    python run_geomopt_gpu.py --functional all \
        --complex complex.vasp --probe probe.xyz --target target.xyz

    # a single arm
    python run_geomopt_gpu.py --functional pbe-d3bj \
        --complex complex.vasp --probe probe.xyz --target target.xyz

Requires ``gpu4pyscf`` + a GPU. Vacuum only.
"""
import argparse
import json
import os
import time
import traceback
from pathlib import Path

from ase import Atoms
from ase.io import read, write

try:
    from pyscf import gto
    from pyscf.data import elements
    from pyscf.geomopt import geometric_solver
except Exception as exc:  # pragma: no cover - depends on env (GPU host only)
    raise SystemExit(f"pyscf import failed: {exc} (install pyscf + gpu4pyscf)")

try:
    from gpu4pyscf.dft import rks, uks
except Exception as exc:  # pragma: no cover - depends on env (GPU host only)
    raise SystemExit(f"gpu4pyscf import failed: {exc} (install gpu4pyscf; GPU required)")


HARTREE_TO_EV = 27.211386245988
HARTREE_TO_KCAL = 627.5094740631

# ---------------------------------------------------------------------------
# The three paper functionals, with the GeoSP two-step basis pair. Dispersion is
# per-functional (D3BJ for PBE/wB97X; none for wB97M-V which carries VV10).
#   ``xc``       : libxc/GPU4PySCF functional string (mf.xc).
#   ``disp``     : mf.disp value ("d3bj" or None).
#   ``nlc``      : explicit mf.nlc override; None lets PySCF auto-enable VV10
#                  from a "-v" xc string (wb97m-v).
#   ``opt_basis``: Step-1 geometry-optimization basis (def2-TZVP for all arms).
#   ``sp_basis`` : Step-2 single-point basis (def2-TZVPD for all arms).
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
OPT_BASIS = "def2-tzvp"
SP_BASIS = "def2-tzvpd"

# TightSCF: tight SCF convergence for both the optimization driver and the
# final def2-TZVPD single point.
TIGHT_SCF_PARAMS = {
    "conv_tol": 1e-10,
    "conv_tol_grad": 1e-6,
}

# TightOpt: Gaussian "tight" geometry-convergence thresholds for geomeTRIC
# (energy in Eh, gradients in Eh/Bohr, displacements in Angstrom).
TIGHT_OPT_PARAMS = {
    "convergence_energy": 1e-6,   # Eh
    "convergence_grms": 1e-5,     # Eh/Bohr
    "convergence_gmax": 1.5e-5,   # Eh/Bohr
    "convergence_drms": 4e-5,     # Angstrom
    "convergence_dmax": 6e-5,     # Angstrom
}

# Back-compat single-structure default; the preferred inputs are the three
# species below.
DEFAULT_STRUCTURE = os.environ.get("RAPIDS_GEOMOPT_STRUCTURE", "complex.vasp")
DEFAULT_COMPLEX = os.environ.get("RAPIDS_GEOMOPT_COMPLEX", DEFAULT_STRUCTURE)
DEFAULT_PROBE = os.environ.get("RAPIDS_GEOMOPT_PROBE")
DEFAULT_TARGET = os.environ.get("RAPIDS_GEOMOPT_TARGET")
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "geomopt_results"


def _count_electrons(symbols, charge):
    return sum(elements.charge(sym) for sym in symbols) - charge


def _select_spin(spin_arg, nelec):
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


def _build_mf(mol, xc, disp, nlc, spin_multiplicity, nelec):
    if spin_multiplicity == 1 and nelec % 2 == 0:
        mf = rks.RKS(mol)
    else:
        mf = uks.UKS(mol)
    mf.xc = xc
    # Per-functional dispersion: D3BJ for PBE/wB97X, none for wB97M-V (VV10).
    if disp:
        mf.disp = disp
    if nlc:
        mf.nlc = nlc
    return mf


def _mol_from_symbols_positions(symbols, positions, basis, charge, spin_mult):
    atom_list = [(sym, pos) for sym, pos in zip(symbols, positions)]
    return gto.M(
        atom=atom_list,
        basis=basis,
        charge=charge,
        spin=spin_mult - 1,
        unit="Angstrom",
        verbose=4,
    )


def _save_geometry(symbols, positions, out_path):
    atoms = Atoms(symbols=symbols, positions=positions)
    write(str(out_path), atoms)


def run_geo_sp(structure_path, xc, disp, nlc, charge, spin,
               max_cycle, max_steps, species_dir):
    """Two-step GeoSP on one structure.

    Step 1: TightOpt/TightSCF geometry optimization at ``OPT_BASIS``.
    Step 2: same-functional single point at ``SP_BASIS`` on the optimized
            geometry. The reported energy is the Step-2 (def2-TZVPD) energy.
    """
    atoms = read(structure_path)
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()

    nelec = _count_electrons(symbols, charge)
    spin_multiplicity = _select_spin(spin, nelec)

    # ---- Step 1: geometry optimization at def2-TZVP (TightOpt / TightSCF) ----
    opt_mol = _mol_from_symbols_positions(symbols, positions, OPT_BASIS,
                                          charge, spin_multiplicity)
    opt_mf = _build_mf(opt_mol, xc=xc, disp=disp, nlc=nlc,
                       spin_multiplicity=spin_multiplicity, nelec=nelec)
    opt_mf.max_cycle = max_cycle
    opt_mf.conv_tol = TIGHT_SCF_PARAMS["conv_tol"]
    opt_mf.conv_tol_grad = TIGHT_SCF_PARAMS["conv_tol_grad"]
    opt_scf_state = _attach_scf_counter(opt_mf)

    t0 = time.time()
    relaxed_mol = geometric_solver.optimize(
        opt_mf,
        maxsteps=max_steps,
        assert_convergence=False,
        **TIGHT_OPT_PARAMS,
    )
    opt_seconds = time.time() - t0

    opt_symbols = [relaxed_mol.atom_symbol(i) for i in range(relaxed_mol.natm)]
    opt_coords = relaxed_mol.atom_coords(unit="Angstrom")
    species_dir.mkdir(parents=True, exist_ok=True)
    _save_geometry(opt_symbols, opt_coords, species_dir / "optimized.xyz")

    # ---- Step 2: same-functional single point at def2-TZVPD on opt geometry --
    sp_mol = _mol_from_symbols_positions(opt_symbols, opt_coords, SP_BASIS,
                                         charge, spin_multiplicity)
    sp_mf = _build_mf(sp_mol, xc=xc, disp=disp, nlc=nlc,
                      spin_multiplicity=spin_multiplicity, nelec=nelec)
    sp_mf.max_cycle = max_cycle
    sp_mf.conv_tol = TIGHT_SCF_PARAMS["conv_tol"]
    sp_mf.conv_tol_grad = TIGHT_SCF_PARAMS["conv_tol_grad"]
    sp_scf_state = _attach_scf_counter(sp_mf)

    t1 = time.time()
    sp_energy = sp_mf.kernel()
    sp_seconds = time.time() - t1

    return {
        "structure": str(structure_path),
        "xc": xc,
        "dispersion": disp,
        "nlc": nlc if nlc else ("auto" if "-v" in xc.lower() else None),
        "opt_basis": OPT_BASIS,
        "sp_basis": SP_BASIS,
        "charge": charge,
        "spin_multiplicity": spin_multiplicity,
        "electrons": nelec,
        "natoms": len(symbols),
        "solvent": None,
        # Step-1 (optimization) diagnostics:
        "opt_energy_hartree": float(opt_mf.e_tot),
        "opt_converged": bool(getattr(opt_mf, "converged", False)),
        "opt_scf_cycles_last": opt_scf_state["count"] if opt_scf_state["count"] > 0 else None,
        "opt_time_seconds": round(opt_seconds, 1),
        "opt_convergence": TIGHT_OPT_PARAMS,
        "scf_conv_tol": TIGHT_SCF_PARAMS["conv_tol"],
        "scf_conv_tol_grad": TIGHT_SCF_PARAMS["conv_tol_grad"],
        # Step-2 (def2-TZVPD single point) — the REPORTED energy:
        "energy_hartree": float(sp_energy),
        "energy_eV": float(sp_energy) * HARTREE_TO_EV,
        "sp_converged": bool(getattr(sp_mf, "converged", False)),
        "sp_scf_cycles": sp_scf_state["count"] if sp_scf_state["count"] > 0 else None,
        "sp_time_seconds": round(sp_seconds, 1),
        "time_seconds": round(opt_seconds + sp_seconds, 1),
    }


def run_arm(functional, species, max_cycle, max_steps, out_dir):
    """Run one GeoSP arm over the provided species and compute binding energy."""
    spec = FUNCTIONALS[functional]
    arm_dir = out_dir / functional
    arm_dir.mkdir(parents=True, exist_ok=True)

    per_species = {}
    for label, (path, charge, spin) in species.items():
        if path is None:
            continue
        res = run_geo_sp(
            structure_path=path,
            xc=spec["xc"],
            disp=spec["disp"],
            nlc=spec["nlc"],
            charge=charge,
            spin=spin,
            max_cycle=max_cycle,
            max_steps=max_steps,
            species_dir=arm_dir / label,
        )
        per_species[label] = res
        print(f"  [{functional}/{label}] E(def2-TZVPD)={res['energy_hartree']:.8f} Ha "
              f"opt_conv={res['opt_converged']} sp_conv={res['sp_converged']} "
              f"({res['time_seconds']}s)")

    arm = {
        "arm": f"{functional}_GeoSP",
        "functional": functional,
        "xc": spec["xc"],
        "dispersion": spec["disp"],
        "opt_basis": OPT_BASIS,
        "sp_basis": SP_BASIS,
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
        print(f"  [{functional}_GeoSP] dE_bind = {de * HARTREE_TO_KCAL:.3f} kcal/mol")
    else:
        arm["binding_energy_hartree"] = None
        arm["note"] = "binding energy needs complex+probe+target; only available species reported"

    with open(arm_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(arm, f, indent=2, ensure_ascii=True)
    return arm


def main():
    parser = argparse.ArgumentParser(
        description="GPU4PySCF GeoSP arms (TightOpt def2-TZVP -> def2-TZVPD SP, vacuum)")
    parser.add_argument("--functional", default="all",
                        choices=list(FUNCTIONALS) + ["all"],
                        help="Which GeoSP arm(s) to run (default: all three).")
    parser.add_argument("--complex", dest="complex_path", default=None,
                        help="Complex geometry (default: $RAPIDS_GEOMOPT_COMPLEX).")
    parser.add_argument("--probe", dest="probe_path", default=None,
                        help="Isolated probe geometry (default: $RAPIDS_GEOMOPT_PROBE).")
    parser.add_argument("--target", dest="target_path", default=None,
                        help="Isolated target geometry (default: $RAPIDS_GEOMOPT_TARGET).")
    parser.add_argument("--structure", default=None,
                        help="Back-compat alias for --complex (single-structure GeoSP).")
    parser.add_argument("--charge-complex", type=int, default=0)
    parser.add_argument("--charge-probe", type=int, default=0)
    parser.add_argument("--charge-target", type=int, default=0)
    parser.add_argument("--spin-complex", type=int, default=None)
    parser.add_argument("--spin-probe", type=int, default=None)
    parser.add_argument("--spin-target", type=int, default=None)
    parser.add_argument("--max-cycle", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    def _resolve(p):
        return Path(p).expanduser().resolve() if p else None

    complex_path = _resolve(args.complex_path or args.structure or DEFAULT_COMPLEX)
    probe_path = _resolve(args.probe_path or DEFAULT_PROBE)
    target_path = _resolve(args.target_path or DEFAULT_TARGET)
    out_base = _resolve(args.out_dir) if args.out_dir else DEFAULT_OUT_DIR

    species = {
        "complex": (complex_path, args.charge_complex, args.spin_complex),
        "probe": (probe_path, args.charge_probe, args.spin_probe),
        "target": (target_path, args.charge_target, args.spin_target),
    }
    if probe_path is None or target_path is None:
        print("NOTE: probe and/or target not provided -> reporting complex "
              "optimized def2-TZVPD energy only (no binding energy). Pass "
              "--probe/--target (or $RAPIDS_GEOMOPT_PROBE/$RAPIDS_GEOMOPT_TARGET) "
              "for the full GeoSP arm.")

    functionals = list(FUNCTIONALS) if args.functional == "all" else [args.functional]

    summary = {}
    for functional in functionals:
        try:
            arm = run_arm(
                functional=functional,
                species=species,
                max_cycle=args.max_cycle,
                max_steps=args.max_steps,
                out_dir=out_base,
            )
            summary[f"{functional}_GeoSP"] = arm
            print(f"[{functional}_GeoSP] done")
        except Exception as exc:
            err = traceback.format_exc()
            (out_base / functional).mkdir(parents=True, exist_ok=True)
            (out_base / functional / "error.txt").write_text(err, encoding="utf-8")
            summary[f"{functional}_GeoSP"] = {"error": str(exc), "error_type": type(exc).__name__}
            print(f"[{functional}_GeoSP] failed: {type(exc).__name__}: {exc}")

    out_base.mkdir(parents=True, exist_ok=True)
    with open(out_base / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)


if __name__ == "__main__":
    main()
