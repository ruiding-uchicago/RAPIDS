#!/usr/bin/env python3
"""Run the RAPIDS single-point DFT arms with ORCA or GPU4PySCF.

ORCA is the default backend.  The historical filename is retained so existing
RAPIDS commands keep working.  Every arm is a vacuum calculation without BSSE:

    dE_bind = E_complex - E_probe - E_target

The three protocols are PBE-D3BJ, wB97X-D3BJ, and wB97M-V with def2-TZVP.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

from ase.data import atomic_numbers
from ase.io import read

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

# xc/disp/nlc are the GPU4PySCF settings. orca_method is the equivalent ORCA
# simple-input keyword sequence. wB97M-V already includes VV10, so no D3 term is
# added to that arm.
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
SP_BASIS = "def2-tzvp"

DEFAULT_STRUCTURE = os.environ.get(
    "RAPIDS_SP_STRUCTURE", "probe_target_vacuum_optimized.vasp"
)
DEFAULT_COMPLEX = os.environ.get("RAPIDS_SP_COMPLEX", DEFAULT_STRUCTURE)
DEFAULT_PROBE = os.environ.get("RAPIDS_SP_PROBE")
DEFAULT_TARGET = os.environ.get("RAPIDS_SP_TARGET")
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "sp_results"


def _count_electrons(symbols: list[str], charge: int) -> int:
    try:
        electrons = sum(atomic_numbers[symbol] for symbol in symbols) - charge
    except KeyError as exc:
        raise ValueError(f"Unknown chemical element: {exc.args[0]}") from exc
    if electrons < 1:
        raise ValueError(f"Invalid electron count ({electrons}) for charge {charge}")
    return electrons


def _select_spin(spin_arg: Optional[int], nelec: int) -> int:
    """Return and validate a 2S+1 multiplicity."""
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
        from gpu4pyscf.dft import rks, uks
    except Exception as exc:  # pragma: no cover - requires a CUDA host
        raise RuntimeError(
            "GPU4PySCF backend unavailable; install pyscf + gpu4pyscf on a CUDA host"
        ) from exc
    return gto, rks, uks


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


def run_single_point_gpu4pyscf(
    structure_path: Path,
    basis: str,
    xc: str,
    disp: Optional[str],
    nlc: Optional[str],
    charge: int,
    spin: Optional[int],
    max_cycle: int,
    conv_tol: float,
    conv_tol_grad: Optional[float],
    grids_level: Optional[int],
) -> Dict[str, Any]:
    """Run one vacuum single point with the optional GPU4PySCF backend."""
    gto, rks, uks = _load_gpu4pyscf()
    atoms = read(structure_path)
    symbols = atoms.get_chemical_symbols()
    atom_list = list(zip(symbols, atoms.get_positions()))

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
    mf = rks.RKS(mol) if spin_multiplicity == 1 and nelec % 2 == 0 else uks.UKS(mol)
    mf.xc = xc
    if disp:
        mf.disp = disp
    if nlc:
        mf.nlc = nlc
    if hasattr(mf, "grids") and grids_level is not None:
        mf.grids.level = grids_level
    mf.max_cycle = max_cycle
    mf.conv_tol = conv_tol
    if conv_tol_grad is not None:
        mf.conv_tol_grad = conv_tol_grad

    scf_state = _attach_scf_counter(mf)
    started = time.monotonic()
    energy = mf.kernel()
    elapsed = time.monotonic() - started

    return {
        "backend": "gpu4pyscf",
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
        "scf_cycles": scf_state["count"] or None,
        "solvent": None,
        "energy_hartree": float(energy),
        "energy_eV": float(energy) * HARTREE_TO_EV,
        "time_seconds": round(elapsed, 1),
        "converged": bool(getattr(mf, "converged", False)),
    }


# Backward-compatible Python entry point for callers that imported this helper.
run_single_point = run_single_point_gpu4pyscf


def run_single_point_orca(
    structure_path: Path,
    functional: str,
    spec: Dict[str, Any],
    charge: int,
    spin: Optional[int],
    max_cycle: int,
    species_dir: Path,
    settings: OrcaSettings,
) -> Dict[str, Any]:
    """Run one vacuum single point with ORCA."""
    atoms = read(structure_path)
    symbols = atoms.get_chemical_symbols()
    nelec = _count_electrons(symbols, charge)
    spin_multiplicity = _select_spin(spin, nelec)

    input_path = species_dir / "single_point.inp"
    output_path = species_dir / "single_point.out"
    write_orca_input(
        atoms,
        input_path,
        method=spec["orca_method"],
        basis=SP_BASIS,
        charge=charge,
        multiplicity=spin_multiplicity,
        max_scf_cycles=max_cycle,
        nprocs=settings.nprocs,
    )
    run = run_orca(input_path, output_path, settings)
    energy = run["energy_hartree"]

    return {
        "backend": "orca",
        "structure": str(structure_path),
        "basis": SP_BASIS,
        "xc": spec["xc"],
        "orca_method": spec["orca_method"],
        "dispersion": spec["disp"],
        "nlc": "SCNL" if functional == "wb97m-v" else None,
        "charge": charge,
        "spin_multiplicity": spin_multiplicity,
        "electrons": nelec,
        "natoms": len(symbols),
        "scf_max_cycle": max_cycle,
        "scf_conv_tol": None,
        "scf_conv_tol_grad": None,
        "scf_convergence": "TightSCF",
        "scf_cycles": run["scf_cycles"],
        "solvent": None,
        "energy_hartree": energy,
        "energy_eV": energy * HARTREE_TO_EV,
        "time_seconds": round(run["time_seconds"], 1),
        "converged": run["converged"],
        "orca_nprocs": settings.nprocs,
        "orca_input": run["input"],
        "orca_output": run["output"],
    }


def run_arm(
    functional: str,
    species: Dict[str, tuple[Optional[Path], int, Optional[int]]],
    max_cycle: int,
    conv_tol: float,
    conv_tol_grad: Optional[float],
    grids_level: Optional[int],
    out_dir: Path,
    backend: str = "orca",
    orca_settings: Optional[OrcaSettings] = None,
) -> Dict[str, Any]:
    """Run one SP arm over available species and compute binding energy."""
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
            result = run_single_point_orca(
                structure_path=path,
                functional=functional,
                spec=spec,
                charge=charge,
                spin=spin,
                max_cycle=max_cycle,
                species_dir=arm_dir / label,
                settings=orca_settings,
            )
        elif backend == "gpu4pyscf":
            result = run_single_point_gpu4pyscf(
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
        else:  # Defensive guard for programmatic callers.
            raise ValueError(f"Unknown DFT backend: {backend}")
        per_species[label] = result
        print(
            f"  [{functional}/{label}] E={result['energy_hartree']:.8f} Ha "
            f"converged={result['converged']} ({result['time_seconds']}s)"
        )

    arm: Dict[str, Any] = {
        "arm": f"{functional}_SP",
        "backend": backend,
        "functional": functional,
        "xc": spec["xc"],
        "orca_method": spec["orca_method"] if backend == "orca" else None,
        "dispersion": spec["disp"],
        "basis": SP_BASIS,
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
        print(f"  [{functional}_SP] dE_bind = {energy * HARTREE_TO_KCAL:.3f} kcal/mol")
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
        description="RAPIDS SP DFT arms (ORCA default; GPU4PySCF optional)"
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
        help="Which SP arm(s) to run (default: all three).",
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
    parser.add_argument("--conv-tol", type=float, default=1e-9)
    parser.add_argument("--conv-tol-grad", type=float, default=None)
    parser.add_argument("--grids-level", type=int, default=None)
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
            "absolute energies (no binding energy)."
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
                conv_tol=args.conv_tol,
                conv_tol_grad=args.conv_tol_grad,
                grids_level=args.grids_level,
                out_dir=out_dir,
                backend=args.backend,
                orca_settings=orca_settings,
            )
            summary[f"{functional}_SP"] = arm
            error_path = out_dir / functional / "error.txt"
            if error_path.is_file():
                error_path.unlink()
        except Exception as exc:
            failed = True
            error = traceback.format_exc()
            (out_dir / functional).mkdir(parents=True, exist_ok=True)
            (out_dir / functional / "error.txt").write_text(error, encoding="utf-8")
            summary[f"{functional}_SP"] = {
                "backend": args.backend,
                "error": str(exc),
                "error_type": type(exc).__name__,
            }
            print(f"[{functional}_SP] failed: {type(exc).__name__}: {exc}")

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
