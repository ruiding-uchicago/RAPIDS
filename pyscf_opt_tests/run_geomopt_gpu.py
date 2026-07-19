#!/usr/bin/env python3
"""Run the RAPIDS geometry-optimization + single-point DFT arms.

ORCA is the default backend and GPU4PySCF remains selectable.  The historical
filename is retained for command compatibility.  Each vacuum GeoSP arm does:

1. TightOpt/TightSCF optimization at functional/def2-TZVP.
2. Same-functional def2-TZVPD single point on the optimized geometry.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

from ase import Atoms
from ase.data import atomic_numbers
from ase.io import read, write

try:  # Supports both direct execution and package-style imports.
    from .orca_backend import (
        DEFAULT_OPENMPI_ROOT,
        DEFAULT_ORCA_EXECUTABLE,
        OrcaSettings,
        run_orca,
        write_orca_input,
    )
except ImportError:  # pragma: no cover - direct script execution
    from orca_backend import (
        DEFAULT_OPENMPI_ROOT,
        DEFAULT_ORCA_EXECUTABLE,
        OrcaSettings,
        run_orca,
        write_orca_input,
    )


HARTREE_TO_EV = 27.211386245988
HARTREE_TO_KCAL = 627.5094740631

FUNCTIONALS = {
    "pbe-d3bj": {
        "xc": "PBE",
        "disp": "d3bj",
        "nlc": None,
        "orca_method": "PBE D3BJ",
    },
    "wb97x-d3bj": {
        "xc": "wb97x-d3bj",
        "disp": "d3bj",
        "nlc": None,
        "orca_method": "WB97X-D3BJ",
    },
    "wb97m-v": {
        "xc": "wb97m-v",
        "disp": None,
        "nlc": None,
        "orca_method": "WB97M-V SCNL",
    },
}
OPT_BASIS = "def2-tzvp"
SP_BASIS = "def2-tzvpd"

TIGHT_SCF_PARAMS = {"conv_tol": 1e-10, "conv_tol_grad": 1e-6}
TIGHT_OPT_PARAMS = {
    "convergence_energy": 1e-6,
    "convergence_grms": 1e-5,
    "convergence_gmax": 1.5e-5,
    "convergence_drms": 4e-5,
    "convergence_dmax": 6e-5,
}

DEFAULT_STRUCTURE = os.environ.get("RAPIDS_GEOMOPT_STRUCTURE", "complex.vasp")
DEFAULT_COMPLEX = os.environ.get("RAPIDS_GEOMOPT_COMPLEX", DEFAULT_STRUCTURE)
DEFAULT_PROBE = os.environ.get("RAPIDS_GEOMOPT_PROBE")
DEFAULT_TARGET = os.environ.get("RAPIDS_GEOMOPT_TARGET")
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "geomopt_results"


def _count_electrons(symbols: list[str], charge: int) -> int:
    try:
        electrons = sum(atomic_numbers[symbol] for symbol in symbols) - charge
    except KeyError as exc:
        raise ValueError(f"Unknown chemical element: {exc.args[0]}") from exc
    if electrons < 1:
        raise ValueError(f"Invalid electron count ({electrons}) for charge {charge}")
    return electrons


def _select_spin(spin_arg: Optional[int], nelec: int) -> int:
    multiplicity = spin_arg if spin_arg is not None else (1 if nelec % 2 == 0 else 2)
    if multiplicity < 1:
        raise ValueError("Spin multiplicity must be at least 1")
    unpaired = multiplicity - 1
    if unpaired > nelec or (nelec - unpaired) % 2:
        raise ValueError(
            f"Multiplicity {multiplicity} is incompatible with {nelec} electrons"
        )
    return multiplicity


def _load_gpu4pyscf():
    """Import the optional GPU stack only when that backend is selected."""
    try:
        from pyscf import gto
        from pyscf.geomopt import geometric_solver
        from gpu4pyscf.dft import rks, uks
    except Exception as exc:  # pragma: no cover - requires a CUDA host
        raise RuntimeError(
            "GPU4PySCF backend unavailable; install pyscf + gpu4pyscf + geometric "
            "on a CUDA host"
        ) from exc
    return gto, geometric_solver, rks, uks


def _attach_scf_counter(mf: Any) -> Dict[str, int]:
    state = {"count": 0}
    existing_callback = getattr(mf, "callback", None)

    def _callback(envs: Dict[str, Any]) -> None:
        cycle = envs.get("cycle")
        if isinstance(cycle, int) and cycle >= 0:
            state["count"] = max(state["count"], cycle + 1)
        else:
            state["count"] += 1
        if callable(existing_callback):
            existing_callback(envs)

    mf.callback = _callback
    return state


def _build_mf(
    mol: Any,
    xc: str,
    disp: Optional[str],
    nlc: Optional[str],
    spin_multiplicity: int,
    nelec: int,
    rks: Any,
    uks: Any,
) -> Any:
    mf = rks.RKS(mol) if spin_multiplicity == 1 and nelec % 2 == 0 else uks.UKS(mol)
    mf.xc = xc
    if disp:
        mf.disp = disp
    if nlc:
        mf.nlc = nlc
    return mf


def _mol_from_symbols_positions(
    symbols: list[str],
    positions: Any,
    basis: str,
    charge: int,
    spin_multiplicity: int,
    gto: Any,
) -> Any:
    return gto.M(
        atom=list(zip(symbols, positions)),
        basis=basis,
        charge=charge,
        spin=spin_multiplicity - 1,
        unit="Angstrom",
        verbose=4,
    )


def run_geo_sp_gpu4pyscf(
    structure_path: Path,
    xc: str,
    disp: Optional[str],
    nlc: Optional[str],
    charge: int,
    spin: Optional[int],
    max_cycle: int,
    max_steps: int,
    species_dir: Path,
) -> Dict[str, Any]:
    """Run the existing two-step GeoSP protocol with GPU4PySCF."""
    gto, geometric_solver, rks, uks = _load_gpu4pyscf()
    atoms = read(structure_path)
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    nelec = _count_electrons(symbols, charge)
    spin_multiplicity = _select_spin(spin, nelec)

    opt_mol = _mol_from_symbols_positions(
        symbols, positions, OPT_BASIS, charge, spin_multiplicity, gto
    )
    opt_mf = _build_mf(
        opt_mol, xc, disp, nlc, spin_multiplicity, nelec, rks, uks
    )
    opt_mf.max_cycle = max_cycle
    opt_mf.conv_tol = TIGHT_SCF_PARAMS["conv_tol"]
    opt_mf.conv_tol_grad = TIGHT_SCF_PARAMS["conv_tol_grad"]
    opt_scf_state = _attach_scf_counter(opt_mf)

    started = time.monotonic()
    relaxed_mol = geometric_solver.optimize(
        opt_mf,
        maxsteps=max_steps,
        assert_convergence=False,
        **TIGHT_OPT_PARAMS,
    )
    opt_seconds = time.monotonic() - started

    opt_symbols = [relaxed_mol.atom_symbol(index) for index in range(relaxed_mol.natm)]
    opt_coords = relaxed_mol.atom_coords(unit="Angstrom")
    species_dir.mkdir(parents=True, exist_ok=True)
    write(str(species_dir / "optimized.xyz"), Atoms(symbols=opt_symbols, positions=opt_coords))

    sp_mol = _mol_from_symbols_positions(
        opt_symbols, opt_coords, SP_BASIS, charge, spin_multiplicity, gto
    )
    sp_mf = _build_mf(sp_mol, xc, disp, nlc, spin_multiplicity, nelec, rks, uks)
    sp_mf.max_cycle = max_cycle
    sp_mf.conv_tol = TIGHT_SCF_PARAMS["conv_tol"]
    sp_mf.conv_tol_grad = TIGHT_SCF_PARAMS["conv_tol_grad"]
    sp_scf_state = _attach_scf_counter(sp_mf)

    started = time.monotonic()
    sp_energy = sp_mf.kernel()
    sp_seconds = time.monotonic() - started

    return {
        "backend": "gpu4pyscf",
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
        "opt_energy_hartree": float(opt_mf.e_tot),
        "opt_converged": bool(getattr(opt_mf, "converged", False)),
        "opt_scf_cycles_last": opt_scf_state["count"] or None,
        "opt_time_seconds": round(opt_seconds, 1),
        "opt_convergence": TIGHT_OPT_PARAMS,
        "scf_conv_tol": TIGHT_SCF_PARAMS["conv_tol"],
        "scf_conv_tol_grad": TIGHT_SCF_PARAMS["conv_tol_grad"],
        "energy_hartree": float(sp_energy),
        "energy_eV": float(sp_energy) * HARTREE_TO_EV,
        "sp_converged": bool(getattr(sp_mf, "converged", False)),
        "sp_scf_cycles": sp_scf_state["count"] or None,
        "sp_time_seconds": round(sp_seconds, 1),
        "time_seconds": round(opt_seconds + sp_seconds, 1),
    }


# Backward-compatible Python entry point for callers that imported this helper.
run_geo_sp = run_geo_sp_gpu4pyscf


def run_geo_sp_orca(
    structure_path: Path,
    functional: str,
    spec: Dict[str, Any],
    charge: int,
    spin: Optional[int],
    max_cycle: int,
    max_steps: int,
    species_dir: Path,
    settings: OrcaSettings,
) -> Dict[str, Any]:
    """Run TightOpt/def2-TZVP then def2-TZVPD SP with ORCA."""
    atoms = read(structure_path)
    symbols = atoms.get_chemical_symbols()
    nelec = _count_electrons(symbols, charge)
    spin_multiplicity = _select_spin(spin, nelec)
    species_dir.mkdir(parents=True, exist_ok=True)

    opt_input = species_dir / "geometry_optimization.inp"
    opt_output = species_dir / "geometry_optimization.out"
    orca_xyz = species_dir / "geometry_optimization.xyz"
    optimized_path = species_dir / "optimized.xyz"
    sp_input = species_dir / "single_point.inp"
    sp_output = species_dir / "single_point.out"
    for stale_path in (orca_xyz, optimized_path, sp_input, sp_output):
        if stale_path.is_file():
            stale_path.unlink()
    write_orca_input(
        atoms,
        opt_input,
        method=spec["orca_method"],
        basis=OPT_BASIS,
        charge=charge,
        multiplicity=spin_multiplicity,
        max_scf_cycles=max_cycle,
        nprocs=settings.nprocs,
        optimize=True,
        max_opt_steps=max_steps,
    )
    opt_run = run_orca(opt_input, opt_output, settings)
    if not opt_run["optimization_converged"]:
        raise RuntimeError(
            "ORCA geometry optimization terminated without convergence; "
            f"increase --max-steps and inspect {opt_output}"
        )

    if not orca_xyz.is_file():
        raise RuntimeError(
            f"ORCA optimization produced no final geometry: expected {orca_xyz}"
        )
    optimized_atoms = read(orca_xyz, index=-1)
    write(str(optimized_path), optimized_atoms)

    write_orca_input(
        optimized_atoms,
        sp_input,
        method=spec["orca_method"],
        basis=SP_BASIS,
        charge=charge,
        multiplicity=spin_multiplicity,
        max_scf_cycles=max_cycle,
        nprocs=settings.nprocs,
    )
    sp_run = run_orca(sp_input, sp_output, settings)
    sp_energy = sp_run["energy_hartree"]

    return {
        "backend": "orca",
        "structure": str(structure_path),
        "xc": spec["xc"],
        "orca_method": spec["orca_method"],
        "dispersion": spec["disp"],
        "nlc": "SCNL" if functional == "wb97m-v" else None,
        "opt_basis": OPT_BASIS,
        "sp_basis": SP_BASIS,
        "charge": charge,
        "spin_multiplicity": spin_multiplicity,
        "electrons": nelec,
        "natoms": len(symbols),
        "solvent": None,
        "opt_energy_hartree": opt_run["energy_hartree"],
        "opt_converged": opt_run["optimization_converged"],
        "opt_scf_cycles_last": opt_run["scf_cycles"],
        "opt_time_seconds": round(opt_run["time_seconds"], 1),
        "opt_convergence": {"preset": "TightOpt"},
        "scf_conv_tol": None,
        "scf_conv_tol_grad": None,
        "scf_convergence": "TightSCF",
        "optimized_structure": str(optimized_path.resolve()),
        "energy_hartree": sp_energy,
        "energy_eV": sp_energy * HARTREE_TO_EV,
        "sp_converged": sp_run["converged"],
        "sp_scf_cycles": sp_run["scf_cycles"],
        "sp_time_seconds": round(sp_run["time_seconds"], 1),
        "time_seconds": round(
            opt_run["time_seconds"] + sp_run["time_seconds"], 1
        ),
        "orca_nprocs": settings.nprocs,
        "orca_opt_input": opt_run["input"],
        "orca_opt_output": opt_run["output"],
        "orca_sp_input": sp_run["input"],
        "orca_sp_output": sp_run["output"],
    }


def run_arm(
    functional: str,
    species: Dict[str, tuple[Optional[Path], int, Optional[int]]],
    max_cycle: int,
    max_steps: int,
    out_dir: Path,
    backend: str = "orca",
    orca_settings: Optional[OrcaSettings] = None,
) -> Dict[str, Any]:
    """Run one GeoSP arm over available species and compute binding energy."""
    if backend not in {"orca", "gpu4pyscf"}:
        raise ValueError(f"Unknown DFT backend: {backend}")
    if backend == "orca" and orca_settings is None:
        orca_settings = OrcaSettings().validated()

    spec = FUNCTIONALS[functional]
    arm_dir = out_dir / functional
    arm_dir.mkdir(parents=True, exist_ok=True)
    result_path = arm_dir / "result.json"
    if result_path.is_file():
        result_path.unlink()

    per_species: Dict[str, Dict[str, Any]] = {}
    for label, (path, charge, spin) in species.items():
        if path is None:
            continue
        if backend == "orca":
            if orca_settings is None:
                raise ValueError("ORCA settings are required for the ORCA backend")
            result = run_geo_sp_orca(
                structure_path=path,
                functional=functional,
                spec=spec,
                charge=charge,
                spin=spin,
                max_cycle=max_cycle,
                max_steps=max_steps,
                species_dir=arm_dir / label,
                settings=orca_settings,
            )
        elif backend == "gpu4pyscf":
            result = run_geo_sp_gpu4pyscf(
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
        else:  # Defensive guard for programmatic callers.
            raise ValueError(f"Unknown DFT backend: {backend}")
        per_species[label] = result
        print(
            f"  [{functional}/{label}] E(def2-TZVPD)="
            f"{result['energy_hartree']:.8f} Ha opt_conv={result['opt_converged']} "
            f"sp_conv={result['sp_converged']} ({result['time_seconds']}s)"
        )

    arm: Dict[str, Any] = {
        "arm": f"{functional}_GeoSP",
        "backend": backend,
        "functional": functional,
        "xc": spec["xc"],
        "orca_method": spec["orca_method"] if backend == "orca" else None,
        "dispersion": spec["disp"],
        "opt_basis": OPT_BASIS,
        "sp_basis": SP_BASIS,
        "solvent": None,
        "species": per_species,
    }

    if {"complex", "probe", "target"} <= set(per_species):
        energy = (
            per_species["complex"]["energy_hartree"]
            - per_species["probe"]["energy_hartree"]
            - per_species["target"]["energy_hartree"]
        )
        arm["binding_energy_hartree"] = energy
        arm["binding_energy_eV"] = energy * HARTREE_TO_EV
        arm["binding_energy_kcal_mol"] = energy * HARTREE_TO_KCAL
        print(
            f"  [{functional}_GeoSP] dE_bind = "
            f"{energy * HARTREE_TO_KCAL:.3f} kcal/mol"
        )
    else:
        arm["binding_energy_hartree"] = None
        arm["note"] = (
            "binding energy needs complex+probe+target; only available species reported"
        )

    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(arm, handle, indent=2, ensure_ascii=True)
    return arm


def _optional_env_float(name: str) -> Optional[float]:
    value = os.environ.get(name)
    return float(value) if value else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAPIDS GeoSP DFT arms (ORCA default; GPU4PySCF optional)"
    )
    parser.add_argument(
        "--backend",
        choices=("orca", "gpu4pyscf"),
        default="orca",
        help="DFT engine (default: orca).",
    )
    parser.add_argument(
        "--functional",
        default="all",
        choices=list(FUNCTIONALS) + ["all"],
        help="Which GeoSP arm(s) to run (default: all three).",
    )
    parser.add_argument("--complex", dest="complex_path", default=None)
    parser.add_argument("--probe", dest="probe_path", default=None)
    parser.add_argument("--target", dest="target_path", default=None)
    parser.add_argument("--structure", default=None, help="Alias for --complex.")
    parser.add_argument("--charge-complex", type=int, default=0)
    parser.add_argument("--charge-probe", type=int, default=0)
    parser.add_argument("--charge-target", type=int, default=0)
    parser.add_argument("--spin-complex", type=int, default=None)
    parser.add_argument("--spin-probe", type=int, default=None)
    parser.add_argument("--spin-target", type=int, default=None)
    parser.add_argument("--max-cycle", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument(
        "--orca-executable",
        default=str(DEFAULT_ORCA_EXECUTABLE),
        help="ORCA executable (default: $RAPIDS_ORCA_EXE or ~/Library/orca_6_1_1/orca).",
    )
    parser.add_argument(
        "--orca-openmpi-root",
        default=str(DEFAULT_OPENMPI_ROOT),
        help="OpenMPI prefix (default: $RAPIDS_OPENMPI_ROOT or ~/Library/openmpi-4.1.1).",
    )
    parser.add_argument(
        "--orca-nprocs",
        "--nprocs",
        type=int,
        default=int(os.environ.get("RAPIDS_ORCA_NPROCS", "1")),
        help="ORCA MPI process count (default: 1).",
    )
    parser.add_argument(
        "--orca-timeout",
        type=float,
        default=_optional_env_float("RAPIDS_ORCA_TIMEOUT"),
        help="Optional per-calculation timeout in seconds.",
    )
    args = parser.parse_args()

    def _resolve(value: Optional[str]) -> Optional[Path]:
        return Path(value).expanduser().resolve() if value else None

    complex_path = _resolve(args.complex_path or args.structure or DEFAULT_COMPLEX)
    probe_path = _resolve(args.probe_path or DEFAULT_PROBE)
    target_path = _resolve(args.target_path or DEFAULT_TARGET)
    out_dir = _resolve(args.out_dir) if args.out_dir else DEFAULT_OUT_DIR
    assert out_dir is not None

    species = {
        "complex": (complex_path, args.charge_complex, args.spin_complex),
        "probe": (probe_path, args.charge_probe, args.spin_probe),
        "target": (target_path, args.charge_target, args.spin_target),
    }
    if probe_path is None or target_path is None:
        print(
            "NOTE: probe and/or target not provided; reporting only available "
            "optimized energies (no binding energy)."
        )

    orca_settings = None
    if args.backend == "orca":
        orca_settings = OrcaSettings(
            executable=Path(args.orca_executable),
            openmpi_root=Path(args.orca_openmpi_root),
            nprocs=args.orca_nprocs,
            timeout_seconds=args.orca_timeout,
        ).validated()
        print(
            f"Backend: ORCA ({orca_settings.executable}; "
            f"nprocs={orca_settings.nprocs})"
        )
    else:
        print("Backend: GPU4PySCF")

    functionals = list(FUNCTIONALS) if args.functional == "all" else [args.functional]
    summary: Dict[str, Any] = {}
    failed = False
    for functional in functionals:
        try:
            arm = run_arm(
                functional=functional,
                species=species,
                max_cycle=args.max_cycle,
                max_steps=args.max_steps,
                out_dir=out_dir,
                backend=args.backend,
                orca_settings=orca_settings,
            )
            summary[f"{functional}_GeoSP"] = arm
            error_path = out_dir / functional / "error.txt"
            if error_path.is_file():
                error_path.unlink()
            print(f"[{functional}_GeoSP] done")
        except Exception as exc:
            failed = True
            error = traceback.format_exc()
            (out_dir / functional).mkdir(parents=True, exist_ok=True)
            (out_dir / functional / "error.txt").write_text(error, encoding="utf-8")
            summary[f"{functional}_GeoSP"] = {
                "backend": args.backend,
                "error": str(exc),
                "error_type": type(exc).__name__,
            }
            print(f"[{functional}_GeoSP] failed: {type(exc).__name__}: {exc}")

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
