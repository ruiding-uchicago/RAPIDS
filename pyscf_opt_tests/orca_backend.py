#!/usr/bin/env python3
"""Small ORCA runner shared by the RAPIDS DFT command-line scripts."""

from __future__ import annotations

import os
import re
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_ORCA_EXECUTABLE = Path(
    os.environ.get(
        "RAPIDS_ORCA_EXE",
        str(Path.home() / "Library" / "orca_6_1_1" / "orca"),
    )
).expanduser()

DEFAULT_OPENMPI_ROOT = Path(
    os.environ.get(
        "RAPIDS_OPENMPI_ROOT",
        str(Path.home() / "Library" / "openmpi-4.1.1"),
    )
).expanduser()

_ENERGY_RE = re.compile(
    r"^\s*FINAL SINGLE POINT ENERGY\s+"
    r"([-+]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)\s*$",
    re.MULTILINE,
)
_SCF_CYCLES_RE = re.compile(
    r"SCF\s+CONVERGED\s+AFTER\s+(\d+)\s+CYCLES?", re.IGNORECASE
)


@dataclass(frozen=True)
class OrcaSettings:
    executable: Path = DEFAULT_ORCA_EXECUTABLE
    openmpi_root: Path = DEFAULT_OPENMPI_ROOT
    nprocs: int = 1
    timeout_seconds: Optional[float] = None

    def validated(self) -> "OrcaSettings":
        executable = self.executable.expanduser().resolve()
        openmpi_root = self.openmpi_root.expanduser().resolve()

        if self.nprocs < 1:
            raise ValueError("ORCA nprocs must be at least 1")
        if not executable.is_file():
            raise FileNotFoundError(f"ORCA executable not found: {executable}")
        if not os.access(executable, os.X_OK):
            raise PermissionError(f"ORCA executable is not executable: {executable}")

        if self.nprocs > 1:
            mpirun = openmpi_root / "bin" / "mpirun"
            mpi_lib = openmpi_root / "lib"
            if not mpirun.is_file():
                raise FileNotFoundError(f"OpenMPI mpirun not found: {mpirun}")
            if not mpi_lib.is_dir():
                raise FileNotFoundError(f"OpenMPI library directory not found: {mpi_lib}")

        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("ORCA timeout must be positive")

        return OrcaSettings(
            executable=executable,
            openmpi_root=openmpi_root,
            nprocs=self.nprocs,
            timeout_seconds=self.timeout_seconds,
        )


def write_orca_input(
    atoms: Any,
    path: Path,
    *,
    method: str,
    basis: str,
    charge: int,
    multiplicity: int,
    max_scf_cycles: int,
    nprocs: int,
    optimize: bool = False,
    max_opt_steps: Optional[int] = None,
) -> None:
    """Write an ORCA input file for a single point or geometry optimization."""
    if multiplicity < 1:
        raise ValueError("Multiplicity must be at least 1")
    if max_scf_cycles < 1:
        raise ValueError("Maximum SCF cycles must be at least 1")
    if optimize and (max_opt_steps is None or max_opt_steps < 1):
        raise ValueError("Maximum optimization steps must be at least 1")

    # Result directories are intentionally reusable. Disable ORCA's default
    # single-point AutoStart so an old GBW cannot seed a changed structure.
    keywords = [method, basis, "TightSCF", "NoAutoStart"]
    if optimize:
        keywords.extend(["Opt", "TightOpt"])

    lines = [f"! {' '.join(keywords)}", "", "%scf", f"  MaxIter {max_scf_cycles}", "end"]

    if optimize:
        lines.extend(["", "%geom", f"  MaxIter {max_opt_steps}", "end"])

    if nprocs > 1:
        lines.extend(["", "%pal", f"  nprocs {nprocs}", "end"])

    lines.extend(["", f"* xyz {charge} {multiplicity}"])
    for atom in atoms:
        x, y, z = atom.position
        lines.append(f"  {atom.symbol:<2s} {x: .12f} {y: .12f} {z: .12f}")
    lines.extend(["*", ""])

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _terminate_process_group(process: subprocess.Popen[Any]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    try:
        process.wait(timeout=2)
        return
    except subprocess.TimeoutExpired:
        pass

    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def _failure_tail(text: str, lines: int = 25) -> str:
    return "\n".join(text.splitlines()[-lines:])


def run_orca(input_path: Path, output_path: Path, settings: OrcaSettings) -> Dict[str, Any]:
    """Run ORCA and validate its output rather than trusting its exit status alone."""
    settings = settings.validated()
    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"ORCA input not found: {input_path}")
    if input_path.parent != output_path.parent:
        raise ValueError("ORCA input and output must use the same working directory")

    command = [str(settings.executable), input_path.name]
    env = os.environ.copy()
    path_prefix = [str(settings.executable.parent)]

    if settings.nprocs > 1:
        mpi_bin = settings.openmpi_root / "bin"
        mpi_lib = settings.openmpi_root / "lib"
        path_prefix.insert(0, str(mpi_bin))

        # ORCA 6.1.1 forwards this single argument to its MPI launcher. This is
        # the invocation tested with the local Apple Silicon installation.
        command.append(f"-x DYLD_LIBRARY_PATH={mpi_lib}")
        old_dyld = env.get("DYLD_LIBRARY_PATH")
        env["DYLD_LIBRARY_PATH"] = (
            f"{mpi_lib}{os.pathsep}{old_dyld}" if old_dyld else str(mpi_lib)
        )

    old_path = env.get("PATH")
    env["PATH"] = os.pathsep.join(path_prefix + ([old_path] if old_path else []))
    env["OMP_NUM_THREADS"] = "1"

    started = time.monotonic()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_handle:
        process = subprocess.Popen(
            command,
            cwd=str(input_path.parent),
            env=env,
            stdout=output_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
        try:
            returncode = process.wait(timeout=settings.timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            _terminate_process_group(process)
            elapsed = time.monotonic() - started
            raise TimeoutError(
                f"ORCA timed out after {elapsed:.1f} seconds; output: {output_path}"
            ) from exc

    elapsed = time.monotonic() - started
    output_text = output_path.read_text(encoding="utf-8", errors="replace")
    lowered = output_text.lower()

    # Some ORCA error paths still return status 0, so the termination markers
    # are the authoritative success criterion.
    normal = "orca terminated normally" in lowered
    error_termination = "orca finished by error termination" in lowered
    if returncode != 0 or not normal or error_termination:
        raise RuntimeError(
            "ORCA did not terminate normally "
            f"(exit status {returncode}, output: {output_path}).\n"
            f"Last output lines:\n{_failure_tail(output_text)}"
        )

    energies = _ENERGY_RE.findall(output_text)
    if not energies:
        raise RuntimeError(
            f"ORCA terminated normally but no final energy was found: {output_path}"
        )

    cycle_matches = _SCF_CYCLES_RE.findall(output_text)
    return {
        "command": command,
        "returncode": returncode,
        "energy_hartree": float(energies[-1]),
        "scf_cycles": int(cycle_matches[-1]) if cycle_matches else None,
        "converged": True,
        "optimization_converged": "the optimization has converged" in lowered,
        "time_seconds": elapsed,
        "input": str(input_path),
        "output": str(output_path),
    }
