#!/usr/bin/env python3
"""
RAPIDS MCP Server (v1.10.0)
===========================
Model Context Protocol server for RAPIDS molecular simulation toolkit.

Exposes comprehensive tools for:
- Setting workspace directory for project isolation
- Listing available substrates, molecules, and rare molecules
- Building molecular simulation structures with full positioning control
- Running geometry optimization
- Calculating potential energies
- Batch screening multiple molecules

Usage:
    python mcp_server.py

Or configure in Claude Desktop config.json:
{
    "mcpServers": {
        "rapids": {
            "command": "python",
            "args": ["/path/to/mcp_server.py"]
        }
    }
}

IMPORTANT: Agents must call set_workspace() before running simulations.
This ensures project isolation - each agent's results are saved to their
own project directory, not the MCP server's directory.
"""

import asyncio
import concurrent.futures
import json
import multiprocessing as mp
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional, List, Dict

# MCP imports
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    Resource,
)

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# RAPIDS imports (lazy loaded to speed up server start)
_simulation_builder = None
_smart_flow = None
_batch_comparison = None

def get_simulation_builder():
    """Lazy load SimulationBuilder"""
    global _simulation_builder
    if _simulation_builder is None:
        from simulation_builder import SimulationBuilder
        _simulation_builder = SimulationBuilder
    return _simulation_builder

def get_smart_flow():
    """Lazy load SmartFAIRChemFlow"""
    global _smart_flow
    if _smart_flow is None:
        from smart_fairchem_flow import SmartFAIRChemFlow
        _smart_flow = SmartFAIRChemFlow
    return _smart_flow

def get_batch_comparison():
    """Lazy load BatchComparison"""
    global _batch_comparison
    if _batch_comparison is None:
        from batch_comparison import BatchComparison
        _batch_comparison = BatchComparison
    return _batch_comparison

# Constants - MCP server's own directories (read-only resources)
MCP_SERVER_DIR = Path(__file__).parent
SUBSTRATE_DIR = MCP_SERVER_DIR / "substrate"
RARE_MOLECULES_DIR = MCP_SERVER_DIR / "rare_molecules"

# Workspace state - set by agent via set_workspace tool
_current_workspace: Optional[Path] = None

# Default backend device for FAIRChem MLIP ("cuda" or "cpu")
_DEFAULT_DEVICE = "cuda"
_GPU_POOL: Optional["_GpuPool"] = None
_GPU_POOL_LOCK = threading.Lock()


def _default_device_backend() -> str:
    """Return the default backend device."""
    if os.environ.get("RAPIDS_FORCE_CPU", "").strip() == "1":
        return "cpu"
    return _DEFAULT_DEVICE


def _get_gpu_pool_devices() -> list[str]:
    """Determine GPU indices for the internal pool (physical indices)."""
    raw = os.environ.get("RAPIDS_GPU_POOL")
    if raw is None:
        raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not raw:
        return ["0", "1", "2", "3"]
    return [item.strip() for item in raw.split(",") if item.strip()]


def _init_worker_device(device_idx: str) -> None:
    """Initializer for GPU worker processes."""
    os.environ["CUDA_VISIBLE_DEVICES"] = device_idx


def _run_flow_worker(config: dict, workspace: str) -> dict:
    """Run build + optimize workflow inside a worker process.

    Includes structural integrity guardrails:
    1. After optimization, check intramolecular bond topology preservation.
    2. If topology changed, re-run with up to 2 fallback ML task heads.
    3. Majority vote: 2/3 or 3/3 broken → confidence=low; 1/3 → confidence=medium.
    4. If 3/3 ML tasks agree on breakage, escalate to DFT (gpu4pyscf) for
       definitive classification.
    """
    import gc

    try:
        simulations_dir = Path(workspace) / "simulations"
        run_name = config["run_name"]
        primary_task = config.get("task_name", "omol")
        validate_topology_flag = config.get("validate_topology", True)

        # --- Phase 1: Build structures ---
        config_path = simulations_dir / f"{run_name}_config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        old_stdout = sys.stdout
        sys.stdout = sys.stderr
        try:
            SimulationBuilder = get_simulation_builder()
            builder = SimulationBuilder(str(config_path), workspace=workspace)
            structures = builder.build_simulation()
            builder.save_structures(structures)
            config_path.unlink(missing_ok=True)

            # --- Phase 2: Run primary optimization ---
            opt_config_path = simulations_dir / run_name / "config_opt.json"
            with open(opt_config_path, "w") as f:
                json.dump(config, f, indent=2)

            SmartFlow = get_smart_flow()
            flow = SmartFlow(str(opt_config_path), workspace=workspace)
            flow.run_workflow()
            del flow
            gc.collect()
        finally:
            sys.stdout = old_stdout

        # --- Phase 3: Topology guardrail ---
        validation = {"confidence": "high", "topology_checked": False, "alerts": []}
        result = {"run_name": run_name, "validation": validation}

        # --- Supp-1: Substrate mode flag ---
        # When substrate != "vacuum", absolute adsorption energies are not reliable
        # (UMA lacks training data for 2D materials). Only relative rankings are useful.
        substrate = config.get("substrate", "vacuum")
        if substrate.lower() not in ("vacuum", "vac"):
            validation["rank_only"] = True
            validation["rank_only_reason"] = (
                f"Substrate mode ({substrate}): ML potential not trained on 2D material adsorption. "
                "Absolute energies are unreliable; use results only for relative ranking."
            )
        else:
            validation["rank_only"] = False

        if not validate_topology_flag:
            return result

        # Check if this is a probe-target system (has both molecules)
        sim_dir = simulations_dir / run_name
        initial_xyz = sim_dir / "probe_target_vacuum.xyz"
        final_vasp = sim_dir / "probe_target_vacuum_optimized.vasp"
        probe_xyz = sim_dir / "probe_vacuum.xyz"

        if not (initial_xyz.exists() and final_vasp.exists() and probe_xyz.exists()):
            # Not a probe-target system or files missing, skip validation
            return result

        from ase.io import read as ase_read
        from topology_validator import (
            validate_topology, get_fallback_tasks,
            validate_geometry, validate_energy, validate_energy_consistency,
        )

        initial_atoms = ase_read(str(initial_xyz))
        final_atoms = ase_read(str(final_vasp))
        n_probe = len(ase_read(str(probe_xyz)))

        topo_result = validate_topology(initial_atoms, final_atoms, n_probe)
        validation["topology_checked"] = True
        validation["primary_task"] = primary_task
        validation["primary_topology"] = topo_result

        # --- Phase 3.4: Interaction type classifier (B3: semantic classifier) ---
        # Classifies whether probe-target interaction is covalent or non-covalent.
        # This is neutral information, not a "problem" — user decides relevance.
        interaction_type = topo_result.get("interaction_type", "non_covalent")
        energy_interpretation = topo_result.get("energy_interpretation", "binding_energy")
        validation["interaction_type"] = interaction_type
        validation["energy_interpretation"] = energy_interpretation

        if interaction_type == "covalent":
            n_inter = topo_result.get("n_intermolecular_new", 0)
            validation["info"] = validation.get("info", [])
            validation["info"].append(
                f"Covalent interaction: {n_inter} new bond(s) formed between probe and target. "
                f"Energy value represents {energy_interpretation}."
            )

        # --- Phase 3.5: Geometry guard (runs regardless of topology result) ---
        try:
            geom_result = validate_geometry(initial_atoms, final_atoms, n_probe)
            validation["geometry"] = geom_result
            if not geom_result["geometry_ok"]:
                validation["alerts"].append(f"Geometry: {geom_result['details']}")
            if geom_result["n_strained"] > 0:
                validation["alerts"].append(
                    f"Geometry: {geom_result['n_strained']} strained bond(s)"
                )
        except Exception as e:
            validation["geometry"] = {"error": str(e)}

        # --- Phase 3.6: Energy guard (needs interactions.json) ---
        try:
            interactions_path = sim_dir / "interactions.json"
            if interactions_path.exists():
                with open(interactions_path) as f:
                    interactions = json.load(f)
                binding_eV = interactions.get("probe_target_vacuum")
                if binding_eV is not None:
                    n_target = len(initial_atoms) - n_probe
                    n_smaller = min(n_probe, n_target)
                    energy_result = validate_energy(binding_eV, n_smaller)
                    validation["energy"] = energy_result
                    if not energy_result["energy_ok"]:
                        validation["alerts"].append(
                            f"Energy: {energy_result['details']}"
                        )
        except Exception as e:
            validation["energy"] = {"error": str(e)}

        # --- Phase 3.7: Convergence guard (B2) ---
        # Check if optimization converged; limit_reached/diverged affects confidence
        convergence_status = None
        try:
            conv_path = sim_dir / "convergence_status.json"
            if conv_path.exists():
                with open(conv_path) as f:
                    conv_data = json.load(f)
                # Get status for probe_target_vacuum (the main structure we validate)
                pt_status = conv_data.get("probe_target_vacuum", {})
                if isinstance(pt_status, dict):
                    convergence_status = pt_status.get("status")
                    max_force = pt_status.get("max_force")
                    validation["convergence"] = {
                        "status": convergence_status,
                        "max_force": max_force,
                    }
                    if convergence_status == "diverged":
                        validation["alerts"].append(
                            f"Convergence: optimization diverged (max_force={max_force:.2f} eV/Å). "
                            "Structure may be unreliable."
                        )
                    elif convergence_status == "limit_reached":
                        validation["alerts"].append(
                            f"Convergence: optimization did not fully converge "
                            f"(max_force={max_force:.4f} eV/Å)."
                        )
        except Exception as e:
            validation["convergence"] = {"error": str(e)}

        # --- Confidence assignment (topology + geometry + energy + convergence) ---
        # Note: interaction_type (covalent/non_covalent) does NOT affect confidence.
        # It's a semantic classifier, not a quality indicator.
        if topo_result["topology_preserved"]:
            confidence = "high"
            # Diverged optimization: structure unreliable, downgrade to low
            if convergence_status == "diverged":
                confidence = "low"
            # Limit reached (not fully converged): downgrade to medium at most
            elif convergence_status == "limit_reached" and confidence == "high":
                confidence = "medium"
            # Geometry clashes downgrade to medium (if not already low)
            geom = validation.get("geometry", {})
            if not geom.get("geometry_ok", True) and confidence == "high":
                confidence = "medium"
            # Non-physical energy downgrades to low
            eng = validation.get("energy", {})
            if not eng.get("energy_ok", True):
                confidence = "low"
            validation["confidence"] = confidence
            # Save validation to file before returning
            val_path = simulations_dir / run_name / "validation.json"
            with open(val_path, "w") as f:
                json.dump(validation, f, indent=2, default=str)
            return result

        # --- Phase 4: Multi-task consensus (topology changed) ---
        validation["alerts"].append(
            f"Primary ({primary_task}): {topo_result['details']}"
        )

        charge = config.get("charge", 0)
        fallback_tasks = get_fallback_tasks(primary_task, charge=charge)
        fallback_results = {}  # task_name -> topo_result
        n_broken = 1  # primary is broken

        old_stdout = sys.stdout
        sys.stdout = sys.stderr
        try:
            for fb_task in fallback_tasks:
                # Create fallback config
                fb_config = config.copy()
                fb_config["task_name"] = fb_task
                fb_run_name = f"{run_name}_fb_{fb_task}"
                fb_config["run_name"] = fb_run_name

                fb_config_path = simulations_dir / f"{fb_run_name}_config.json"
                fb_config_path.parent.mkdir(parents=True, exist_ok=True)
                with open(fb_config_path, "w") as f:
                    json.dump(fb_config, f, indent=2)

                # Build from same initial geometry
                builder = SimulationBuilder(str(fb_config_path), workspace=workspace)
                structures = builder.build_simulation()
                builder.save_structures(structures)
                fb_config_path.unlink(missing_ok=True)

                # Optimize with fallback task
                fb_opt_path = simulations_dir / fb_run_name / "config_opt.json"
                with open(fb_opt_path, "w") as f:
                    json.dump(fb_config, f, indent=2)

                flow = SmartFlow(str(fb_opt_path), workspace=workspace)
                flow.run_workflow()
                del flow
                gc.collect()

                # Check fallback topology
                fb_final = simulations_dir / fb_run_name / "probe_target_vacuum_optimized.vasp"
                if fb_final.exists():
                    fb_atoms = ase_read(str(fb_final))
                    fb_topo = validate_topology(initial_atoms, fb_atoms, n_probe)
                    fallback_results[fb_task] = fb_topo

                    if not fb_topo["topology_preserved"]:
                        n_broken += 1
                        validation["alerts"].append(
                            f"Fallback ({fb_task}): {fb_topo['details']}"
                        )
                    else:
                        validation["alerts"].append(
                            f"Fallback ({fb_task}): topology preserved"
                        )
                else:
                    validation["alerts"].append(
                        f"Fallback ({fb_task}): optimization failed"
                    )
        finally:
            sys.stdout = old_stdout

        validation["fallback_results"] = {
            task: {"topology_preserved": r["topology_preserved"], "details": r["details"]}
            for task, r in fallback_results.items()
        }
        validation["n_tasks_broken"] = n_broken
        validation["n_tasks_total"] = 1 + len(fallback_results)

        # --- Phase 5: Consensus decision ---
        if n_broken == 1:
            # Only primary broke — artifact. Find best intact fallback.
            validation["confidence"] = "medium"
            # Use the first intact fallback result
            for fb_task, fb_topo in fallback_results.items():
                if fb_topo["topology_preserved"]:
                    validation["adopted_task"] = fb_task
                    validation["adopted_run"] = f"{run_name}_fb_{fb_task}"
                    # Copy fallback results to main run directory
                    _copy_fallback_to_primary(
                        simulations_dir, run_name, f"{run_name}_fb_{fb_task}"
                    )
                    break
        elif n_broken >= 2 and n_broken < validation["n_tasks_total"]:
            # Majority broken — likely real but one intact
            validation["confidence"] = "low"
        else:
            # All ML tasks broken — mark for user review, no automatic DFT
            validation["confidence"] = "low"
            validation["all_ml_broken"] = True
            validation["dft_recommended"] = True
            validation["alerts"].append(
                "All ML tasks (omol/oc20/omat) show topology change. "
                "Consider DFT verification with gpu4pyscf if this molecular pair is important."
            )

        result["validation"] = validation
        # Save validation to file
        val_path = simulations_dir / run_name / "validation.json"
        with open(val_path, "w") as f:
            json.dump(validation, f, indent=2, default=str)

        return result
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        return {"error": str(e)}


def _copy_fallback_to_primary(simulations_dir: Path, primary_name: str, fallback_name: str):
    """Copy key result files from fallback run to primary run directory."""
    import shutil
    primary_dir = simulations_dir / primary_name
    fallback_dir = simulations_dir / fallback_name

    for filename in [
        "probe_target_vacuum_optimized.vasp",
        "interactions.json",
        "solvation.json",
        "smart_report.txt",
    ]:
        src = fallback_dir / filename
        dst = primary_dir / filename
        if src.exists():
            # Backup original
            if dst.exists():
                backup = primary_dir / f"original_{filename}"
                shutil.copy2(str(dst), str(backup))
            shutil.copy2(str(src), str(dst))


class _GpuPool:
    """Simple round-robin GPU process pool."""

    def __init__(self, devices: list[str]) -> None:
        self.devices = devices
        self.executors: list[concurrent.futures.ProcessPoolExecutor] = []
        ctx = mp.get_context("spawn")
        for device_idx in devices:
            self.executors.append(
                concurrent.futures.ProcessPoolExecutor(
                    max_workers=1,
                    mp_context=ctx,
                    initializer=_init_worker_device,
                    initargs=(device_idx,),
                )
            )
        self._next = 0
        self._lock = threading.Lock()

    def next_executor(self) -> concurrent.futures.ProcessPoolExecutor:
        with self._lock:
            executor = self.executors[self._next]
            self._next = (self._next + 1) % len(self.executors)
        return executor


def _get_gpu_pool() -> Optional[_GpuPool]:
    """Return a lazily initialized GPU pool."""
    global _GPU_POOL
    with _GPU_POOL_LOCK:
        if _GPU_POOL is None:
            devices = _get_gpu_pool_devices()
            if not devices:
                return None
            _GPU_POOL = _GpuPool(devices)
    return _GPU_POOL


async def _run_multi_config_tasks(task_entries: list[dict], workspace: Path) -> None:
    """Run multi-config tasks using the internal GPU pool when available."""
    if not task_entries:
        return

    if _default_device_backend() != "cuda":
        for entry in task_entries:
            entry["worker_result"] = await asyncio.to_thread(
                _run_flow_worker, entry["config"], str(workspace)
            )
        return

    pool = _get_gpu_pool()
    if not pool or not pool.executors:
        for entry in task_entries:
            entry["worker_result"] = await asyncio.to_thread(
                _run_flow_worker, entry["config"], str(workspace)
            )
        return

    loop = asyncio.get_running_loop()
    futures: list[tuple[dict, asyncio.Future]] = []
    for entry in task_entries:
        executor = pool.next_executor()
        future = loop.run_in_executor(
            executor, _run_flow_worker, entry["config"], str(workspace)
        )
        futures.append((entry, future))

    for entry, future in futures:
        entry["worker_result"] = await future


# ============================================================
# Anchor Sampling Helper Functions
# ============================================================

def get_principal_axis(atoms) -> tuple:
    """
    Calculate the principal axis of a molecule using PCA.
    Returns the principal axis (longest direction) as a unit vector.

    Args:
        atoms: ASE Atoms object

    Returns:
        (principal_axis, extent): Unit vector along longest direction, and extent in Å
    """
    import numpy as np

    positions = atoms.get_positions()
    centered = positions - positions.mean(axis=0)

    # Covariance matrix
    cov = np.cov(centered.T)

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Principal axis is eigenvector with largest eigenvalue
    principal_idx = np.argmax(eigenvalues)
    principal_axis = eigenvectors[:, principal_idx]

    # Calculate extent along principal axis
    projections = centered @ principal_axis
    extent = projections.max() - projections.min()

    return principal_axis, extent


def generate_anchor_positions(atoms, target_atoms, n_anchors: int = 3, base_distance: float = 4.0) -> list:
    """
    Generate anchor positions for probe molecule relative to target.
    Places anchors along the probe's principal axis at different distances from target.

    Args:
        atoms: Probe molecule (ASE Atoms)
        target_atoms: Target molecule (ASE Atoms)
        n_anchors: Number of anchor positions (1=center only, 3=near/mid/far)
        base_distance: Base distance from target in Å

    Returns:
        List of position offsets [(dx, dy, dz), ...] relative to target center
    """
    import numpy as np

    if n_anchors == 1:
        # Single anchor at center (current behavior)
        return [(0, 0, base_distance)]

    # Get probe's principal axis
    principal_axis, extent = get_principal_axis(atoms)

    # Generate anchor offsets along principal axis
    # For n_anchors=3: near (0.25), mid (0.5), far (0.75) along extent
    anchors = []

    for i in range(n_anchors):
        # Position along principal axis: from 0.25 to 0.75
        frac = 0.25 + 0.5 * i / (n_anchors - 1) if n_anchors > 1 else 0.5

        # Offset along principal axis from center
        axis_offset = (frac - 0.5) * extent * principal_axis

        # Position above target (z-direction) plus axis offset
        position = np.array([0, 0, base_distance]) + axis_offset
        anchors.append(tuple(position))

    return anchors


def check_contact_state(atoms, probe_indices: list, target_indices: list,
                        min_distance_threshold: float = 3.8,
                        com_distance_threshold: float = 10.0) -> dict:
    """
    Check if probe and target are in proper contact state after optimization.

    Args:
        atoms: Optimized ASE Atoms object
        probe_indices: Indices of probe atoms
        target_indices: Indices of target atoms
        min_distance_threshold: Maximum allowed minimum distance for contact (Å)
        com_distance_threshold: Maximum allowed COM distance (Å)

    Returns:
        dict with 'is_contact', 'min_distance', 'com_distance'
    """
    import numpy as np

    positions = atoms.get_positions()

    probe_pos = positions[probe_indices]
    target_pos = positions[target_indices]

    # Calculate minimum interatomic distance
    min_dist = float('inf')
    for p in probe_pos:
        for t in target_pos:
            dist = np.linalg.norm(p - t)
            if dist < min_dist:
                min_dist = dist

    # Calculate COM distance
    probe_com = probe_pos.mean(axis=0)
    target_com = target_pos.mean(axis=0)
    com_dist = np.linalg.norm(probe_com - target_com)

    is_contact = (min_dist < min_distance_threshold) and (com_dist < com_distance_threshold)

    return {
        'is_contact': is_contact,
        'min_distance': min_dist,
        'com_distance': com_dist
    }


def get_workspace() -> Optional[Path]:
    """Get current workspace directory"""
    return _current_workspace


def get_simulations_dir() -> Optional[Path]:
    """Get simulations directory in current workspace"""
    if _current_workspace is None:
        return None
    return _current_workspace / "simulations"


def get_molecules_dir() -> Optional[Path]:
    """Get molecules directory in current workspace"""
    if _current_workspace is None:
        return None
    return _current_workspace / "molecules"



def require_workspace() -> tuple[bool, str]:
    """Check if workspace is set, return (ok, error_message)"""
    if _current_workspace is None:
        return False, (
            "Error: No workspace set.\n\n"
            "You must call set_workspace(path) first to specify where simulation "
            "files should be saved.\n\n"
            "Example: set_workspace('/path/to/your/project')\n\n"
            "This ensures your results are saved to your project directory, "
            "not the MCP server's directory."
        )
    return True, ""


def resolve_workspace(args: dict, require: bool = True) -> tuple[Optional[Path], str]:
    """
    Resolve workspace from args['workspace'] or fall back to global _current_workspace.

    This function enables parallel-safe operation by allowing explicit workspace
    parameter to override the global state. Essential for scenarios where multiple
    workers share the same MCP connection.

    Args:
        args: Tool arguments dictionary, may contain 'workspace' key
        require: If True, return error message when no workspace is available

    Returns:
        (workspace_path, error_msg) - error_msg is empty string if OK
    """
    # Priority 1: Explicit workspace parameter (parallel-safe)
    workspace_arg = args.get("workspace")
    if workspace_arg:
        workspace = Path(workspace_arg).resolve()
        # Create directory if doesn't exist
        workspace.mkdir(parents=True, exist_ok=True)
        return workspace, ""

    # Priority 2: Global workspace (backward compatible)
    if _current_workspace is not None:
        return _current_workspace, ""

    # No workspace available
    if require:
        return None, (
            "Error: No workspace set.\n\n"
            "Either:\n"
            "1. Pass 'workspace' parameter to this tool call, or\n"
            "2. Call set_workspace(path) first to set a global workspace.\n\n"
            "For parallel workers: always pass explicit 'workspace' parameter to avoid conflicts.\n\n"
            "Example: workspace='/path/to/your/project'"
        )

    return None, ""

# Available substrates (pre-built crystal structures)
AVAILABLE_SUBSTRATES = [
    "BP",        # Black Phosphorus
    "Co_HHTP",   # Cobalt-HHTP MOF
    "Cu_HHTP",   # Copper-HHTP MOF
    "Graphene",  # Graphene sheet
    "MoS2",      # Molybdenum disulfide
    "Ni_HHTP",   # Nickel-HHTP MOF
    "Si",        # Silicon
    "ZnO",       # Zinc Oxide
]

# Create server instance
server = Server(
    name="rapids",
    version="1.10.0",
    instructions="""RAPIDS (Rapid Atomistic Probe-target Interaction Discovery Scaffold) is a molecular simulation toolkit.

=== IMPORTANT: WORKSPACE SETUP ===

Before running ANY simulation, you MUST call set_workspace() with the user's project directory:

    set_workspace("/path/to/user/project")

This ensures:
- Your simulation results are saved to YOUR project directory
- The MCP server's directory remains clean (it only contains shared resources)
- Different projects don't pollute each other

The MCP server directory contains READ-ONLY shared resources:
- substrate/: Pre-built crystal structures (Graphene, MoS2, etc.)
- rare_molecules/: Complex molecules not available on PubChem

Your workspace will contain:
- simulations/: Your simulation results
- molecules/: Downloaded molecules for your project

=== TYPICAL WORKFLOW ===

1. set_workspace("/path/to/project")  ← MUST do this first!
2. build_simulation(probe="benzene", substrate="Graphene", ...)
3. optimize_structure(run_name="...")
4. calculate_adsorption_energy(run_name="...")

=== FEATURES ===

- Automatic molecule download from PubChem (just use the molecule name)
- 9 pre-built substrates: Graphene, MoS2, BP, Si, ZnO, Co/Cu/Ni_HHTP MOFs
- xTB implicit solvation (automatic with optimize_structure)
- Van der Waals contact distance mode
- Relative positioning between molecules

=== SOLVATION - IMPORTANT ===

TWO solvation methods are available:

1. xTB IMPLICIT SOLVATION (RECOMMENDED) - runs AUTOMATICALLY with optimize_structure()
   - Uses GFN2-xTB + ALPB water model
   - Fast (~seconds per molecule)
   - Gives "Solution binding" energy in solvation.json
   - NO configuration needed - just run optimize_structure()

2. EXPLICIT SOLVATION (NOT RECOMMENDED) - explicit_solvation parameter in build_simulation
   - Adds REAL water molecules to the simulation box
   - MUCH slower (many more atoms to optimize)
   - Only use for: MD simulations, specific solvent structure studies
   - DO NOT use for normal screening tasks!

For binding energy screening: Just use optimize_structure() → get Solution binding from results

=== TASK SELECTION GUIDE ===

task_name options and their accuracy:

1. 'omol' (default) - Best for VACUUM molecular interactions
   - Trained on ωB97M-V/def2-TZVPD with VV10 dispersion
   - ACCURATE: molecule-molecule interactions in vacuum (dimers, complexes)
   - INACCURATE: substrate adsorption (gives unrealistic -20 to -30 eV values)
   - Use for: hydrogen bonding, π-π stacking, drug-receptor interactions

2. 'oc20' - Better for SUBSTRATE systems (qualitative only)
   - Trained on RPBE for catalysis surfaces
   - Gives reasonable magnitude (~0.1 eV) but may have wrong sign
   - Use for: qualitative ranking of molecules on surfaces
   - NOT accurate for absolute adsorption energies

3. 'omat' - For inorganic materials
   - Trained on PBE/PBE+U for bulk materials
   - Use for: materials properties, not recommended for molecular adsorption

IMPORTANT: Use the SAME task_name across ALL tools in a workflow!

=== ACCURACY LIMITATIONS ===

VACUUM MODE (substrate='vacuum'):
✓ RELIABLE for molecule-molecule interactions
✓ Quantitative results comparable to literature
✓ Examples: water dimer (-5.0 kcal/mol), benzene dimer (-2.0 kcal/mol)

SUBSTRATE MODE (Graphene, MoS2, etc.):
✗ NOT RELIABLE for absolute adsorption energies
✗ omol: gives -20 to -30 eV (should be ~-0.5 eV) - 50x overestimate
✗ oc20: gives +0.1 eV (should be ~-0.5 eV) - wrong sign
- Root cause: UMA training data lacks 2D material adsorption
- USE ONLY FOR: qualitative ranking/comparison between molecules
- DO NOT TRUST: absolute energy values

=== RECOMMENDATIONS ===

For QUANTITATIVE analysis:
→ Use vacuum mode (probe-target without substrate)
→ Use omol task for best dispersion description

For QUALITATIVE substrate screening:
→ Use oc20 task (more reasonable magnitudes)
→ Trust relative rankings, not absolute values
→ Validate important results with DFT

=== THREE-TIER SAMPLING STRATEGY ===

scan_orientations now supports multi-anchor sampling for more reliable results:

TIER 1 - Initial (Quick check):
  optimize_structure() → 1 optimization
  Use for: Quick sanity check, obvious cases

TIER 2 - Middle (RECOMMENDED DEFAULT):
  scan_orientations(n_anchors=3, num_orientations=3) → 9 optimizations
  Use for: All production screening tasks
  Features: 3 anchor positions (near/mid/far) × 3 orientations

TIER 3 - High (Confirmation):
  Middle tier + additional random sampling → 15-20 optimizations
  Trigger when: Uncertainty detected (see below)

=== UPGRADE TRIGGERS (Middle → High) ===

Consider high-tier sampling when:
1. Top2 candidates ΔE < 1 kcal/mol (too close to distinguish)
2. Same molecule, different anchors ΔE > 2 kcal/mol (position-sensitive)
3. Contact state success rate < 50% (geometry issues)

=== scan_orientations PARAMETERS ===

n_anchors: Number of anchor positions along probe's principal axis
  - 1: Center only (legacy mode, fast but may miss optimal binding site)
  - 3: Near/Mid/Far (RECOMMENDED - explores different binding sites)

num_orientations: Rotations per anchor (default: 3)
  - Total optimizations = n_anchors × num_orientations

=== WHEN TO USE scan_orientations ===

ALWAYS use scan_orientations for reliable binding energy ranking.
The default (n_anchors=3, num_orientations=3) is recommended for all screening.

Use n_anchors=1 only for:
- Quick preliminary checks
- Very small/symmetric molecules (e.g., water, methane)

=== PARALLEL EXECUTION ===

Multiple optimize_structure calls can run in parallel.
scan_orientations dispatches multi-config optimizations through an internal GPU pool
(one worker per GPU), while preserving SMART incremental caching.

Typical workflow:
1. scan_orientations(probe, target) → Get reliable binding energies
2. Review results → Check for uncertainty warnings
3. If uncertain → Add random sampling for top candidates
""",
)


# ============================================================
# Tool Definitions
# ============================================================

@server.list_tools()
async def list_tools() -> list[Tool]:
    """List all available tools"""
    return [
        # ==================== Workspace Tools ====================
        Tool(
            name="set_workspace",
            description="Set the working directory for this session. MUST be called before running any simulations. "
                       "All simulation results and downloaded molecules will be saved to this directory. "
                       "This ensures project isolation - different agents/projects don't pollute each other.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to your project directory (e.g., '/Users/name/my_project')"
                    }
                },
                "required": ["path"]
            }
        ),
        Tool(
            name="get_workspace",
            description="Get the current workspace directory. Returns None if not set.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),

        # ==================== Listing Tools ====================
        Tool(
            name="list_substrates",
            description="List all available substrate materials for molecular simulations. "
                       "Substrates are pre-built crystal structures that cannot be downloaded from PubChem.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="list_local_molecules",
            description="List molecules already downloaded in the local library. "
                       "Note: Any molecule can be used by name - it will be auto-downloaded from PubChem if not local.",
            inputSchema={
                "type": "object",
                "properties": {
                    "workspace": {
                        "type": "string",
                        "description": "Optional: Explicit workspace path for parallel-safe operation. "
                                      "Takes priority over global workspace."
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="list_rare_molecules",
            description="List pre-optimized rare/complex molecules that are difficult to obtain from PubChem. "
                       "These include MOF linkers, complex ligands, and other specialized molecules.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="list_simulations",
            description="List all completed simulation runs with their status.",
            inputSchema={
                "type": "object",
                "properties": {
                    "workspace": {
                        "type": "string",
                        "description": "Optional: Explicit workspace path for parallel-safe operation. "
                                      "Takes priority over global workspace."
                    }
                },
                "required": []
            }
        ),

        # ==================== Build Tool ====================
        Tool(
            name="build_simulation",
            description="Build molecular simulation structures with full control over positioning, orientation, and solvation. "
                       "Molecules are automatically downloaded from PubChem if not found locally. "
                       "Returns paths to generated structure files (VASP, XYZ formats).",
            inputSchema={
                "type": "object",
                "properties": {
                    "run_name": {
                        "type": "string",
                        "description": "Name for this simulation run (used for output folder)"
                    },
                    "probe": {
                        "type": "string",
                        "description": "Probe molecule name (e.g., 'benzene', 'glucose', 'ibuprofen'). Auto-downloaded from PubChem if not local."
                    },
                    "target": {
                        "type": "string",
                        "description": "Optional target molecule name for probe-target interaction studies."
                    },
                    "substrate": {
                        "type": "string",
                        "description": "Substrate material. Use 'vacuum' for gas-phase or one of: BP, Co_HHTP, Cu_HHTP, Graphene, MoS2, Ni_HHTP, Si, ZnO"
                    },
                    # Height parameters
                    "probe_height": {
                        "type": "number",
                        "description": "Height of probe above substrate in Angstroms (default: 2.5)"
                    },
                    "target_height": {
                        "type": "number",
                        "description": "Height of target above substrate in Angstroms (default: 6.0)"
                    },
                    "probe_target_distance": {
                        "type": ["number", "string"],
                        "description": "Distance between probe and target. Use number for fixed distance (Å), or 'contact' for van der Waals contact distance."
                    },
                    # Position parameters
                    "probe_position": {
                        "type": "object",
                        "description": "Custom probe position. Options: "
                                      "1) Cartesian: {\"x\": 10.0, \"y\": 10.0, \"z\": 5.0} "
                                      "2) Fractional: {\"frac\": [0.5, 0.5, 0.3]} "
                                      "3) Cylindrical: {\"cylindrical\": {\"r\": 5.0, \"theta\": 45, \"z\": 3.0}}"
                    },
                    "target_position": {
                        "type": "object",
                        "description": "Custom target position. Options: "
                                      "1) Cartesian: {\"x\": 10.0, \"y\": 10.0, \"z\": 8.0} "
                                      "2) Fractional: {\"frac\": [0.5, 0.5, 0.5]} "
                                      "3) Cylindrical: {\"cylindrical\": {\"r\": 5.0, \"theta\": 45, \"z\": 6.0}} "
                                      "4) Relative: {\"relative_to\": \"probe\", \"lateral_offset\": 4.0, \"vertical_offset\": 1.0, \"direction\": \"x\"}"
                    },
                    # Orientation parameters
                    "probe_orientation": {
                        "type": "object",
                        "description": "Probe molecule orientation. Options: "
                                      "1) Euler angles: {\"euler\": [0, 0, 45]} (degrees, ZYZ convention) "
                                      "2) Axis-angle: {\"axis\": [0, 0, 1], \"angle\": 45} "
                                      "3) Quaternion: {\"quaternion\": [1, 0, 0, 0]} "
                                      "4) Align vector: {\"align\": {\"from\": [0, 0, 1], \"to\": [1, 0, 0]}}"
                    },
                    "target_orientation": {
                        "type": "object",
                        "description": "Target molecule orientation. Same options as probe_orientation."
                    },
                    # Box and substrate parameters
                    "box_size": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Custom simulation box size [x, y, z] in Angstroms. Default: [30, 30, 40] for vacuum."
                    },
                    "fix_substrate_layers": {
                        "type": "integer",
                        "description": "Number of substrate layers to fix during optimization (default: 1). Set to 0 to allow all atoms to move."
                    },
                    # Explicit solvation parameters (NOT RECOMMENDED for most tasks)
                    "explicit_solvation": {
                        "type": "object",
                        "description": "⚠️ EXPLICIT SOLVATION - NOT RECOMMENDED for most tasks! "
                                      "This adds REAL water molecules to the simulation box, making calculations MUCH slower. "
                                      "For binding energy calculations, use optimize_structure() instead - it automatically "
                                      "runs xTB implicit solvation (ALPB) which is fast and gives Solution binding energies. "
                                      "Only use explicit solvation for: (1) MD simulations, (2) studying specific solvent effects. "
                                      "Examples if you really need it: "
                                      "1) Auto mode: {\"enabled\": true, \"mode\": \"auto\", \"shell_thickness\": 5.0} "
                                      "2) Manual mode: {\"enabled\": true, \"mode\": \"manual\", \"count\": 50}",
                        "properties": {
                            "enabled": {"type": "boolean", "description": "Enable explicit solvation (NOT RECOMMENDED)"},
                            "mode": {"type": "string", "enum": ["auto", "manual"], "description": "auto: calculate count from cluster size; manual: specify count"},
                            "solvent": {"type": "string", "description": "Solvent molecule name (default: water)"},
                            "count": {"type": "integer", "description": "Number of solvent molecules (manual mode)"},
                            "shell_thickness": {"type": "number", "description": "Solvation shell thickness in Å (default: 5.0)"}
                        }
                    },
                    # Charge and spin parameters (omol task only)
                    "charge": {
                        "type": "integer",
                        "description": "Net charge of the system (omol task only). "
                                      "Examples: 0 (neutral), +1 (cation), -1 (anion). Default: 0"
                    },
                    "spin": {
                        "type": "integer",
                        "description": "Spin multiplicity 2S+1 (omol task only). "
                                      "1=singlet (default, no unpaired electrons), "
                                      "2=doublet (1 unpaired electron, e.g., radicals), "
                                      "3=triplet (2 unpaired electrons). Default: 1"
                    },
                    # Workspace parameter for parallel-safe operation
                    "workspace": {
                        "type": "string",
                        "description": "Optional: Explicit workspace path for parallel-safe operation. "
                                      "Takes priority over global workspace set by set_workspace(). "
                                      "Required when running multiple workers in parallel."
                    }
                },
                "required": ["run_name", "probe"]
            }
        ),

        # ==================== Computation Tools ====================
        Tool(
            name="optimize_structure",
            description="Run geometry optimization on a simulation using FAIRChem UMA ML potential. "
                       "This may take several minutes depending on system size. "
                       "Returns optimized energies and structure paths. "
                       "Includes automatic topology guardrail: checks intramolecular bond preservation after optimization.",
            inputSchema={
                "type": "object",
                "properties": {
                    "run_name": {
                        "type": "string",
                        "description": "Name of existing simulation run to optimize"
                    },
                    "fmax": {
                        "type": "number",
                        "description": "Force convergence criterion in eV/Å (default: 0.05)"
                    },
                    "max_steps": {
                        "type": "integer",
                        "description": "Maximum optimization steps (default: 200)"
                    },
                    "task_name": {
                        "type": "string",
                        "description": "FAIRChem task type: 'omol' (default, has VV10 dispersion), 'oc20' (catalysis), 'omat' (materials/surfaces). Must be consistent across all tools in workflow.",
                        "enum": ["omol", "oc20", "omat"]
                    },
                    "workspace": {
                        "type": "string",
                        "description": "Optional: Explicit workspace path for parallel-safe operation. "
                                      "Takes priority over global workspace set by set_workspace()."
                    }
                },
                "required": ["run_name"]
            }
        ),
        Tool(
            name="calculate_energy",
            description="Calculate potential energy of a structure using FAIRChem UMA ML potential. "
                       "Fast single-point energy calculation without optimization.",
            inputSchema={
                "type": "object",
                "properties": {
                    "structure_path": {
                        "type": "string",
                        "description": "Path to structure file (VASP or XYZ format). Can be relative to RAPIDS directory."
                    },
                    "task_name": {
                        "type": "string",
                        "description": "FAIRChem task type: 'omol' (default, has VV10 dispersion), 'oc20' (catalysis), 'omat' (materials/surfaces)",
                        "enum": ["omol", "oc20", "omat"],
                        "default": "omol"
                    },
                    "workspace": {
                        "type": "string",
                        "description": "Optional: Explicit workspace path for resolving relative structure paths. "
                                      "Takes priority over global workspace."
                    }
                },
                "required": ["structure_path"]
            }
        ),
        Tool(
            name="calculate_adsorption_energy",
            description="Calculate adsorption/interaction energies from optimized structures. "
                       "Computes: E_ads = E_complex - E_probe - E_target - E_substrate. "
                       "Negative values indicate favorable adsorption.",
            inputSchema={
                "type": "object",
                "properties": {
                    "run_name": {
                        "type": "string",
                        "description": "Name of optimized simulation run"
                    },
                    "workspace": {
                        "type": "string",
                        "description": "Optional: Explicit workspace path for parallel-safe operation. "
                                      "Takes priority over global workspace."
                    }
                },
                "required": ["run_name"]
            }
        ),

        # ==================== Batch Screening Tool ====================
        Tool(
            name="batch_screening",
            description="Screen multiple probe molecules against a target and/or substrate. "
                       "By default uses multi-anchor scanning (9 optimizations per probe) for reliable results. "
                       "Set use_scan=false for faster but less reliable single-point screening. "
                       "Includes topology guardrail: probes with intramolecular bond artifacts are flagged and separated from main ranking.",
            inputSchema={
                "type": "object",
                "properties": {
                    "run_name": {
                        "type": "string",
                        "description": "Name for this batch screening run"
                    },
                    "probes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of probe molecule names to screen (e.g., ['benzene', 'toluene', 'naphthalene'])"
                    },
                    "target": {
                        "type": "string",
                        "description": "Target molecule for probe-target interaction screening (required for use_scan=true)"
                    },
                    "substrate": {
                        "type": "string",
                        "description": "Substrate material (default: 'vacuum')"
                    },
                    "use_scan": {
                        "type": "boolean",
                        "description": "Use multi-anchor scan (default: true, 9 optimizations per probe). "
                                      "Set to false for fast single-optimization mode (less reliable).",
                        "default": True
                    },
                    "n_anchors": {
                        "type": "integer",
                        "description": "Number of anchor positions when use_scan=true (default: 3)",
                        "default": 3
                    },
                    "num_orientations": {
                        "type": "integer",
                        "description": "Number of orientations per anchor when use_scan=true (default: 3)",
                        "default": 3
                    },
                    "fmax": {
                        "type": "number",
                        "description": "Force convergence criterion in eV/Å (default: 0.05)"
                    },
                    "task_name": {
                        "type": "string",
                        "description": "FAIRChem task type: 'omol' (default, has VV10 dispersion), 'oc20' (catalysis), 'omat' (materials/surfaces). Must be consistent across all tools in workflow.",
                        "enum": ["omol", "oc20", "omat"]
                    },
                    "charge": {
                        "type": "integer",
                        "description": "Net charge of the system (omol task only). Default: 0"
                    },
                    "spin": {
                        "type": "integer",
                        "description": "Spin multiplicity 2S+1 (omol task only). 1=singlet, 2=doublet, 3=triplet. Default: 1"
                    },
                    "workspace": {
                        "type": "string",
                        "description": "Optional: Explicit workspace path for parallel-safe operation. "
                                      "Takes priority over global workspace. Required for parallel workers."
                    }
                },
                "required": ["run_name", "probes"]
            }
        ),

        # ==================== Orientation Scanning Tool ====================
        Tool(
            name="scan_orientations",
            description="Scan molecular positions and orientations to find the most stable configuration. "
                       "TIERED AUTO-UPGRADE: Automatically upgrades sampling based on result quality. "
                       "Tier 1: Grid Scan (3×3=9 opts, fast baseline). "
                       "Tier 2: +Basin Hopping (6 opts, exploration). "
                       "Tier 3: +Random Sampling (9 opts, confirmation). "
                       "Upgrade triggers: HIGH_VARIANCE, TOP2_CLOSE, LOW_VALID_RATE, LARGE_SYSTEM, BH_FOUND_BETTER. "
                       "SMART INCREMENTAL: Skips already-computed configurations. "
                       "Includes topology guardrail with 3-task ML consensus. "
                       "Results are confidence-annotated. Flagged results excluded from ranking.",
            inputSchema={
                "type": "object",
                "properties": {
                    "run_name": {
                        "type": "string",
                        "description": "Base name for this scan. Use SAME name across tiers for incremental computation."
                    },
                    "probe": {
                        "type": "string",
                        "description": "Probe molecule name"
                    },
                    "target": {
                        "type": "string",
                        "description": "Target molecule name (required for probe-target interaction)"
                    },
                    "substrate": {
                        "type": "string",
                        "description": "Substrate material (default: 'vacuum')"
                    },
                    "n_anchors": {
                        "type": "integer",
                        "description": "Number of anchor positions (default: 3). "
                                      "1=center only (initial tier), 3=near/mid/far (middle tier).",
                        "default": 3
                    },
                    "num_orientations": {
                        "type": "integer",
                        "description": "Number of orientations per anchor (default: 3). "
                                      "3=middle tier, 6=high tier. Already-computed orientations are skipped.",
                        "default": 3
                    },
                    "random_samples": {
                        "type": "integer",
                        "description": "Additional random samples near best configuration (default: 0). "
                                      "Use 5-10 for high-tier confirmation sampling.",
                        "default": 0
                    },
                    "rotate_molecule": {
                        "type": "string",
                        "description": "Which molecule to rotate: 'probe' (default) or 'target'",
                        "enum": ["probe", "target"]
                    },
                    "rotation_axis": {
                        "type": "string",
                        "description": "Axis to rotate around: 'x' (tilt - RECOMMENDED), 'y', 'z', or 'all'.",
                        "enum": ["x", "y", "z", "all"]
                    },
                    "task_name": {
                        "type": "string",
                        "description": "FAIRChem task type: 'omol' (default), 'oc20', 'omat'",
                        "enum": ["omol", "oc20", "omat"]
                    },
                    "fmax": {
                        "type": "number",
                        "description": "Force convergence criterion in eV/Å (default: 0.05)"
                    },
                    "charge": {
                        "type": "integer",
                        "description": "Net charge of the system (omol task only). Default: 0"
                    },
                    "spin": {
                        "type": "integer",
                        "description": "Spin multiplicity 2S+1 (omol task only). 1=singlet, 2=doublet, 3=triplet. Default: 1"
                    },
                    "workspace": {
                        "type": "string",
                        "description": "Optional: Explicit workspace path for parallel-safe operation. "
                                      "Takes priority over global workspace. Required for parallel workers."
                    },
                    "auto_upgrade": {
                        "type": "boolean",
                        "description": "Enable automatic tier upgrade based on result quality (default: true). "
                                      "Tier 1=Grid(9), Tier 2=+BH(6), Tier 3=+Random(9).",
                        "default": True
                    },
                    "max_tier": {
                        "type": "integer",
                        "description": "Maximum tier to upgrade to (1, 2, or 3). Default: 3",
                        "default": 3
                    },
                    "min_tier": {
                        "type": "integer",
                        "description": "Minimum tier to run (1, 2, or 3). Default: 1",
                        "default": 1
                    }
                },
                "required": ["run_name", "probe", "target"]
            }
        ),

        # ==================== Results Tool ====================
        Tool(
            name="get_simulation_results",
            description="Get detailed results and summary from a completed simulation run, including energies, structures, and files.",
            inputSchema={
                "type": "object",
                "properties": {
                    "run_name": {
                        "type": "string",
                        "description": "Name of the simulation run"
                    },
                    "workspace": {
                        "type": "string",
                        "description": "Optional: Explicit workspace path for parallel-safe operation. "
                                      "Takes priority over global workspace."
                    }
                },
                "required": ["run_name"]
            }
        ),

        # ==================== Structure Analysis Tool ====================
        Tool(
            name="analyze_structure",
            description="Analyze a molecular structure - get atom count, formula, dimensions, and basic properties.",
            inputSchema={
                "type": "object",
                "properties": {
                    "structure_path": {
                        "type": "string",
                        "description": "Path to structure file (VASP, XYZ, or SDF format)"
                    },
                    "workspace": {
                        "type": "string",
                        "description": "Optional: Explicit workspace path for resolving relative structure paths. "
                                      "Takes priority over global workspace."
                    }
                },
                "required": ["structure_path"]
            }
        ),
    ]


# ============================================================
# Tool Implementations
# ============================================================

@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls"""

    if name == "set_workspace":
        return await handle_set_workspace(arguments)

    elif name == "get_workspace":
        return await handle_get_workspace()

    elif name == "list_substrates":
        return await handle_list_substrates()

    elif name == "list_local_molecules":
        return await handle_list_local_molecules(arguments)

    elif name == "list_rare_molecules":
        return await handle_list_rare_molecules()

    elif name == "list_simulations":
        return await handle_list_simulations(arguments)

    elif name == "build_simulation":
        return await handle_build_simulation(arguments)

    elif name == "optimize_structure":
        return await handle_optimize_structure(arguments)

    elif name == "calculate_energy":
        return await handle_calculate_energy(arguments)

    elif name == "get_simulation_results":
        return await handle_get_simulation_results(arguments)

    elif name == "calculate_adsorption_energy":
        return await handle_calculate_adsorption_energy(arguments)

    elif name == "batch_screening":
        return await handle_batch_screening(arguments)

    elif name == "scan_orientations":
        return await handle_scan_orientations(arguments)

    elif name == "analyze_structure":
        return await handle_analyze_structure(arguments)

    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


# ==================== Workspace Handlers ====================

async def handle_set_workspace(args: dict) -> list[TextContent]:
    """Set the workspace directory for this session"""
    global _current_workspace

    path = args.get("path", "").strip()
    if not path:
        return [TextContent(type="text", text="Error: path is required")]

    workspace_path = Path(path).resolve()

    # Create workspace directory if it doesn't exist
    try:
        workspace_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return [TextContent(type="text", text=f"Error creating workspace directory: {e}")]

    # Create subdirectories
    simulations_dir = workspace_path / "simulations"
    molecules_dir = workspace_path / "molecules"

    try:
        simulations_dir.mkdir(exist_ok=True)
        molecules_dir.mkdir(exist_ok=True)
    except Exception as e:
        return [TextContent(type="text", text=f"Error creating subdirectories: {e}")]

    _current_workspace = workspace_path

    result = f"Workspace set successfully!\n\n"
    result += f"Workspace: {workspace_path}\n"
    result += f"  └── simulations/  (simulation results will be saved here)\n"
    result += f"  └── molecules/    (downloaded molecules will be saved here)\n\n"
    result += "You can now run simulations. All results will be saved to this workspace."

    return [TextContent(type="text", text=result)]


async def handle_get_workspace() -> list[TextContent]:
    """Get the current workspace directory"""
    if _current_workspace is None:
        return [TextContent(type="text", text="No workspace set. Call set_workspace(path) first.")]

    result = f"Current workspace: {_current_workspace}\n"
    result += f"  └── simulations/: {get_simulations_dir()}\n"
    result += f"  └── molecules/:   {get_molecules_dir()}"

    return [TextContent(type="text", text=result)]


# ==================== Listing Handlers ====================

async def handle_list_substrates() -> list[TextContent]:
    """List available substrates with details"""
    substrate_info = []
    for sub in AVAILABLE_SUBSTRATES:
        sub_path = SUBSTRATE_DIR / sub
        if sub_path.exists():
            vasp_files = list(sub_path.glob("*.vasp"))
            info = f"- {sub}"
            if vasp_files:
                try:
                    from ase.io import read
                    atoms = read(vasp_files[0])
                    formula = atoms.get_chemical_formula()
                    info += f" ({len(atoms)} atoms, {formula})"
                except:
                    pass
            substrate_info.append(info)

    result = "Available Substrates (9 total):\n" + "\n".join(substrate_info)
    result += "\n\nUse 'vacuum' for gas-phase calculations without substrate."
    result += "\n\nSubstrate descriptions:"
    result += "\n- BP: Black Phosphorus (2D material)"
    result += "\n- Graphene: Graphene sheet (2D carbon)"
    result += "\n- MoS2: Molybdenum disulfide (2D semiconductor)"
    result += "\n- Co/Cu/Ni_HHTP: Metal-HHTP MOF structures"
    result += "\n- Si: Silicon surface"
    result += "\n- ZnO: Zinc Oxide surface"
    return [TextContent(type="text", text=result)]


async def handle_list_local_molecules(args: dict) -> list[TextContent]:
    """List locally available molecules in current workspace and shared directories"""
    # Resolve workspace (supports explicit parameter, not required for this tool)
    workspace, _ = resolve_workspace(args, require=False)
    molecules_dir = workspace / "molecules" if workspace else None

    # Collect molecules with full paths
    molecules = {}  # name -> path

    # 1. Workspace molecules
    if molecules_dir and molecules_dir.exists():
        for f in molecules_dir.glob("*.sdf"):
            if f.stem and not f.name.startswith('.'):
                molecules[f.stem] = str(f)

    # 2. Global shared molecules (MCP_SERVER_DIR/molecules)
    global_molecules_dir = MCP_SERVER_DIR / "molecules"
    if global_molecules_dir.exists():
        for f in global_molecules_dir.glob("*.sdf"):
            if f.stem and not f.name.startswith('.'):
                if f.stem not in molecules:  # workspace takes priority
                    molecules[f.stem] = str(f)

    # 3. Rare molecules (MCP_SERVER_DIR/rare_molecules)
    if RARE_MOLECULES_DIR.exists():
        for f in RARE_MOLECULES_DIR.glob("*.sdf"):
            if f.stem and not f.name.startswith('.'):
                if f.stem not in molecules:
                    molecules[f.stem] = str(f)

    result = f"Workspace: {workspace or '(not set)'}\n\n"
    result += f"Local Molecules ({len(molecules)} available):\n\n"

    if molecules:
        for name in sorted(molecules.keys()):
            result += f"  {name}: {molecules[name]}\n"
    else:
        result += "(none yet - molecules will be downloaded when you run simulations)\n"

    result += "\nNote: Any molecule name can be used - if not found locally, it will be auto-downloaded from PubChem."
    result += "\nTo view a molecule's 3D structure, use Read tool on the .sdf file path above."
    return [TextContent(type="text", text=result)]


async def handle_list_rare_molecules() -> list[TextContent]:
    """List rare/complex molecules"""
    if not RARE_MOLECULES_DIR.exists():
        return [TextContent(type="text", text="No rare_molecules directory found. All molecules will be fetched from PubChem.")]

    molecules = []
    for f in RARE_MOLECULES_DIR.glob("*.sdf"):
        if f.stem and not f.name.startswith('.'):
            molecules.append(f.stem)
    for f in RARE_MOLECULES_DIR.glob("*.xyz"):
        if f.stem and not f.name.startswith('.'):
            molecules.append(f.stem)

    molecules = sorted(set(molecules))

    if not molecules:
        return [TextContent(type="text", text="No rare molecules found in rare_molecules directory.")]

    result = f"Rare/Complex Molecules ({len(molecules)} available):\n"
    result += ", ".join(molecules)
    result += "\n\nThese are pre-optimized 3D structures for molecules that may be difficult to obtain from PubChem."
    result += "\nThey are automatically used when you specify the molecule name."
    return [TextContent(type="text", text=result)]


async def handle_list_simulations(args: dict) -> list[TextContent]:
    """List all simulation runs in current workspace"""
    # Resolve workspace (supports explicit parameter for parallel-safe operation)
    workspace, error = resolve_workspace(args)
    if error:
        return [TextContent(type="text", text=error)]

    simulations_dir = workspace / "simulations"

    if not simulations_dir.exists():
        return [TextContent(type="text", text=f"Workspace: {workspace}\n\nNo simulations found yet.")]

    simulations = []
    for sim_dir in sorted(simulations_dir.iterdir()):
        if sim_dir.is_dir() and not sim_dir.name.startswith('.'):
            status = "built"
            if (sim_dir / "results.json").exists():
                status = "optimized"
            elif (sim_dir / "summary.txt").exists():
                status = "built"
            else:
                status = "incomplete"

            # Get atom count if possible
            info = f"- {sim_dir.name} [{status}]"
            config_path = sim_dir / "config.json"
            if config_path.exists():
                try:
                    with open(config_path) as f:
                        config = json.load(f)
                    probe = config.get("probe", "?")
                    target = config.get("target", "")
                    substrate = config.get("substrate", "vacuum")
                    info += f" (probe={probe}"
                    if target:
                        info += f", target={target}"
                    info += f", substrate={substrate})"
                except:
                    pass
            simulations.append(info)

    if not simulations:
        return [TextContent(type="text", text=f"Workspace: {workspace}\n\nNo simulations found yet.")]

    result = f"Workspace: {workspace}\n\n"
    result += f"Simulations ({len(simulations)} total):\n" + "\n".join(simulations)
    return [TextContent(type="text", text=result)]


# ==================== Build Handler ====================

async def handle_build_simulation(args: dict) -> list[TextContent]:
    """Build simulation structures with full configuration options"""
    try:
        # Resolve workspace (supports explicit parameter for parallel-safe operation)
        workspace, error = resolve_workspace(args)
        if error:
            return [TextContent(type="text", text=error)]

        simulations_dir = workspace / "simulations"

        # Prepare config with all possible parameters
        config = {
            "run_name": args["run_name"],
            "probe": args["probe"],
        }

        # Basic parameters
        if "target" in args and args["target"]:
            config["target"] = args["target"]

        config["substrate"] = args.get("substrate", "vacuum")

        # Height parameters
        if "probe_height" in args:
            config["probe_height"] = args["probe_height"]
        if "target_height" in args:
            config["target_height"] = args["target_height"]
        if "probe_target_distance" in args:
            config["probe_target_distance"] = args["probe_target_distance"]

        # Position parameters
        if "probe_position" in args:
            config["probe_position"] = args["probe_position"]
        if "target_position" in args:
            config["target_position"] = args["target_position"]

        # Orientation parameters
        if "probe_orientation" in args:
            config["probe_orientation"] = args["probe_orientation"]
        if "target_orientation" in args:
            config["target_orientation"] = args["target_orientation"]

        # Box and substrate parameters
        if "box_size" in args:
            config["box_size"] = args["box_size"]
        if "fix_substrate_layers" in args:
            config["fix_substrate_layers"] = args["fix_substrate_layers"]

        # Explicit solvation parameters (support both old and new parameter names)
        if "explicit_solvation" in args:
            config["solvation"] = args["explicit_solvation"]  # Map to internal 'solvation' key
        elif "solvation" in args:
            config["solvation"] = args["solvation"]  # Backward compatibility

        # Charge and spin parameters (omol task only)
        if "charge" in args:
            config["charge"] = args["charge"]
        if "spin" in args:
            config["spin"] = args["spin"]

        # Save config to temp file in workspace
        config_path = simulations_dir / f"{args['run_name']}_config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        # Build simulation (redirect stdout to avoid MCP pollution)
        old_stdout = sys.stdout
        sys.stdout = sys.stderr
        try:
            SimulationBuilder = get_simulation_builder()
            # Pass workspace to builder so it knows where to save molecules
            builder = SimulationBuilder(str(config_path), workspace=str(workspace))
            structures = builder.build_simulation()
            builder.save_structures(structures)
        finally:
            sys.stdout = old_stdout

        # Clean up temp config
        config_path.unlink(missing_ok=True)

        # Prepare result
        output_dir = simulations_dir / args["run_name"]
        result = f"Simulation built successfully!\n\n"
        result += f"Output directory: {output_dir}\n\n"
        result += "Configuration:\n"
        result += f"  Probe: {args['probe']}\n"
        if "target" in args and args["target"]:
            result += f"  Target: {args['target']}\n"
        result += f"  Substrate: {config['substrate']}\n"
        # Check for explicit solvation (either parameter name)
        solvation_args = args.get("explicit_solvation") or args.get("solvation")
        if solvation_args and solvation_args.get("enabled"):
            result += f"  ⚠️ Explicit solvation: enabled ({solvation_args.get('mode', 'auto')} mode)\n"
            result += f"     Note: This adds real water molecules. For most tasks, xTB implicit solvation is sufficient.\n"

        result += "\nGenerated structures:\n"
        for name, atoms in structures.items():
            result += f"  - {name}: {len(atoms)} atoms ({atoms.get_chemical_formula()})\n"

        result += f"\nFiles saved in VASP and XYZ formats."
        result += f"\nUse optimize_structure(run_name='{args['run_name']}') to run geometry optimization."

        return [TextContent(type="text", text=result)]

    except Exception as e:
        import traceback
        return [TextContent(type="text", text=f"Error building simulation: {str(e)}\n\n{traceback.format_exc()}")]


# ==================== Computation Handlers ====================

async def handle_optimize_structure(args: dict) -> list[TextContent]:
    """Run geometry optimization"""
    try:
        _opt_wall_start = time.time()

        # Resolve workspace (supports explicit parameter for parallel-safe operation)
        workspace, error = resolve_workspace(args)
        if error:
            return [TextContent(type="text", text=error)]

        simulations_dir = workspace / "simulations"

        run_name = args["run_name"]
        fmax = args.get("fmax", 0.05)
        max_steps = args.get("max_steps", 200)
        task_name = args.get("task_name", "omol")

        sim_dir = simulations_dir / run_name
        config_path = sim_dir / "config.json"

        if not config_path.exists():
            return [TextContent(type="text", text=f"Simulation '{run_name}' not found. Run build_simulation first.")]

        with open(config_path) as f:
            config = json.load(f)

        config["fmax"] = fmax
        config["max_steps"] = max_steps
        config["task_name"] = task_name
        config["device"] = _default_device_backend()

        # Save updated config to temp file (SmartFlow expects file path)
        temp_config_path = sim_dir / "config_opt.json"
        with open(temp_config_path, 'w') as f:
            json.dump(config, f, indent=2)

        SmartFlow = get_smart_flow()
        flow = SmartFlow(str(temp_config_path), workspace=str(workspace))

        result_text = f"Starting optimization for '{run_name}'...\n"
        result_text += f"Parameters: fmax={fmax} eV/Å, max_steps={max_steps}, task={task_name}\n\n"

        # Redirect stdout to stderr to avoid interfering with MCP stdio protocol
        # MCP uses stdout for JSON-RPC communication
        import io
        old_stdout = sys.stdout
        sys.stdout = sys.stderr  # Redirect prints to stderr
        try:
            await asyncio.to_thread(flow.run_workflow)
        finally:
            sys.stdout = old_stdout  # Restore stdout

        # Clean up
        del flow
        import gc
        gc.collect()

        _opt_wall_seconds = time.time() - _opt_wall_start
        result_text += f"Optimization completed! (wall time: {_opt_wall_seconds:.1f} s)\n\n"

        # Save wall-clock timing to optimization_timing.json
        timing_path = sim_dir / "optimization_timing.json"
        with open(timing_path, 'w') as f:
            json.dump({"wall_time_seconds": round(_opt_wall_seconds, 2)}, f, indent=2)

        # Read results from saved files (run_workflow doesn't return results)
        interactions_path = sim_dir / "interactions.json"
        if interactions_path.exists():
            with open(interactions_path) as f:
                interactions = json.load(f)
            result_text += "Adsorption/Interaction Energies:\n"
            for name, energy in interactions.items():
                kcal = energy * 23.061
                result_text += f"  {name}: {energy:.4f} eV ({kcal:.2f} kcal/mol)\n"
                if energy < 0:
                    result_text += f"    → Favorable (exothermic)\n"
                else:
                    result_text += f"    → Unfavorable (endothermic)\n"

        # Read solvation results if available (vacuum mode only)
        solvation_path = sim_dir / "solvation.json"
        if solvation_path.exists():
            with open(solvation_path) as f:
                solvation = json.load(f)
            result_text += "\nSolvation Analysis (xTB GFN2-xTB + ALPB water):\n"
            for name in ["probe_vacuum", "target_vacuum", "probe_target_vacuum"]:
                if name in solvation:
                    G_solv = solvation[name]
                    result_text += f"  G_solv({name}): {G_solv:.4f} eV ({G_solv * 23.061:.2f} kcal/mol)\n"
            if "delta_G_solvation" in solvation:
                dG_solv = solvation["delta_G_solvation"]
                result_text += f"\n  ΔG_solvation: {dG_solv:.4f} eV ({dG_solv * 23.061:.2f} kcal/mol)\n"
            if "delta_G_solution" in solvation:
                dG_sol = solvation["delta_G_solution"]
                result_text += f"  Solution binding: {dG_sol:.4f} eV ({dG_sol * 23.061:.2f} kcal/mol)\n"

        # Topology validation for probe-target systems
        init_xyz = sim_dir / "probe_target_vacuum.xyz"
        final_vasp = sim_dir / "probe_target_vacuum_optimized.vasp"
        probe_xyz = sim_dir / "probe_vacuum.xyz"
        if init_xyz.exists() and final_vasp.exists() and probe_xyz.exists():
            try:
                from ase.io import read as ase_read
                from topology_validator import validate_topology
                init_atoms = ase_read(str(init_xyz))
                final_atoms = ase_read(str(final_vasp))
                probe_atoms = ase_read(str(probe_xyz))
                n_probe = len(probe_atoms)
                topo = validate_topology(init_atoms, final_atoms, n_probe)
                if topo["topology_preserved"]:
                    result_text += "\nTopology Check: PASSED (intramolecular bonds preserved)\n"
                else:
                    result_text += f"\nTopology Check: WARNING — intramolecular bond changes detected\n"
                    result_text += f"  {topo['details']}\n"
                    result_text += f"  Confidence: needs_verification\n"
                    result_text += f"  Consider re-running with a different task_name for verification.\n"
            except Exception as topo_err:
                result_text += f"\nTopology Check: Could not validate ({topo_err})\n"

        # Also show the report if available
        report_path = sim_dir / "smart_report.txt"
        if report_path.exists():
            result_text += f"\nFull report saved to: {report_path}\n"

        return [TextContent(type="text", text=result_text)]

    except Exception as e:
        import traceback
        return [TextContent(type="text", text=f"Error during optimization: {str(e)}\n\n{traceback.format_exc()}")]


async def handle_calculate_energy(args: dict) -> list[TextContent]:
    """Calculate single-point energy"""
    try:
        from ase.io import read
        from fairchem.core import pretrained_mlip, FAIRChemCalculator

        structure_path = args["structure_path"]

        if not os.path.isabs(structure_path):
            # Resolve workspace (supports explicit parameter, not required)
            workspace, _ = resolve_workspace(args, require=False)
            if workspace:
                structure_path = workspace / structure_path
            else:
                structure_path = Path(__file__).parent / structure_path

        if not Path(structure_path).exists():
            return [TextContent(type="text", text=f"Structure file not found: {structure_path}")]

        # Redirect stdout to avoid MCP pollution from fairchem
        old_stdout = sys.stdout
        sys.stdout = sys.stderr
        try:
            atoms = read(structure_path)

            # Load model with specified task (turbo mode for ~1.7x speedup)
            task_name = args.get("task_name", "omol")
            device = _default_device_backend()
            predictor = pretrained_mlip.get_predict_unit(
                "uma-s-1p2",
                device=device,
                inference_settings="turbo"
            )
            calc = FAIRChemCalculator(predictor, task_name=task_name)
            atoms.calc = calc

            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
            max_force = (forces**2).sum(axis=1).max()**0.5
        finally:
            sys.stdout = old_stdout

        result = f"Energy Calculation Results:\n"
        result += f"  Structure: {Path(structure_path).name}\n"
        result += f"  Task: {task_name}\n"
        result += f"  Atoms: {len(atoms)}\n"
        result += f"  Formula: {atoms.get_chemical_formula()}\n"
        result += f"  Potential Energy: {energy:.4f} eV\n"
        result += f"  Max Force: {max_force:.4f} eV/Å\n"

        return [TextContent(type="text", text=result)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error calculating energy: {str(e)}")]


async def handle_calculate_adsorption_energy(args: dict) -> list[TextContent]:
    """Calculate adsorption/interaction energies from optimized structures"""
    try:
        # Resolve workspace (supports explicit parameter for parallel-safe operation)
        workspace, error = resolve_workspace(args)
        if error:
            return [TextContent(type="text", text=error)]

        simulations_dir = workspace / "simulations"
        run_name = args["run_name"]
        sim_dir = simulations_dir / run_name

        if not sim_dir.exists():
            return [TextContent(type="text", text=f"Simulation '{run_name}' not found.")]

        # Check for interactions.json (output from SmartFAIRChemFlow)
        interactions_path = sim_dir / "interactions.json"
        smart_report_path = sim_dir / "smart_report.txt"

        result = f"Adsorption/Interaction Energy Analysis: {run_name}\n\n"

        if interactions_path.exists():
            with open(interactions_path) as f:
                interactions = json.load(f)

            result += "Adsorption/Interaction Energies:\n"
            for name, energy in interactions.items():
                kcal = energy * 23.061
                result += f"  {name}:\n"
                result += f"    {energy:.4f} eV\n"
                result += f"    {kcal:.2f} kcal/mol\n"
                if energy < 0:
                    result += f"    → Favorable (exothermic)\n"
                else:
                    result += f"    → Unfavorable/weak (endothermic)\n"
                result += "\n"
        elif smart_report_path.exists():
            # Fall back to parsing smart_report.txt
            with open(smart_report_path) as f:
                result += "From smart_report.txt:\n"
                result += f.read()
        else:
            return [TextContent(type="text", text=f"No results found for '{run_name}'. Run optimize_structure first.")]

        # Add solvation data if available
        solvation_path = sim_dir / "solvation.json"
        if solvation_path.exists():
            with open(solvation_path) as f:
                solvation = json.load(f)
            result += "\nSolvation Analysis (xTB GFN2-xTB + ALPB water):\n"
            for name in ["probe_vacuum", "target_vacuum", "probe_target_vacuum"]:
                if name in solvation:
                    G_solv = solvation[name]
                    result += f"  G_solv({name}): {G_solv:.4f} eV ({G_solv * 23.061:.2f} kcal/mol)\n"
            if "delta_G_solvation" in solvation:
                dG_solv = solvation["delta_G_solvation"]
                result += f"\n  ΔG_solvation: {dG_solv:.4f} eV ({dG_solv * 23.061:.2f} kcal/mol)\n"
            if "delta_G_solution" in solvation:
                dG_sol = solvation["delta_G_solution"]
                result += f"  Solution binding: {dG_sol:.4f} eV ({dG_sol * 23.061:.2f} kcal/mol)\n"

        return [TextContent(type="text", text=result)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error calculating adsorption energy: {str(e)}")]


# ==================== Batch Screening Handler ====================

async def handle_batch_screening(args: dict) -> list[TextContent]:
    """Run batch screening of multiple probes with optional multi-anchor scanning"""
    try:
        # Resolve workspace (supports explicit parameter for parallel-safe operation)
        workspace, error = resolve_workspace(args)
        if error:
            return [TextContent(type="text", text=error)]

        simulations_dir = workspace / "simulations"

        run_name = args["run_name"]
        probes = args["probes"]
        target = args.get("target")
        substrate = args.get("substrate", "vacuum")
        use_scan = args.get("use_scan", True)  # NEW: default to scan mode
        n_anchors = args.get("n_anchors", 3)
        num_orientations = args.get("num_orientations", 3)
        fmax = args.get("fmax", 0.05)
        task_name = args.get("task_name", "omol")
        charge = args.get("charge", 0)
        spin = args.get("spin", 1)

        # Validate: scan mode requires target
        if use_scan and not target:
            return [TextContent(type="text", text="Error: use_scan=true requires a target molecule. Either provide target or set use_scan=false.")]

        total_opts = n_anchors * num_orientations if use_scan else 1
        result_text = f"Batch Screening: {run_name}\n"
        result_text += f"Workspace: {workspace}\n"
        result_text += f"Probes: {', '.join(probes)}\n"
        if target:
            result_text += f"Target: {target}\n"
        result_text += f"Substrate: {substrate}\n"
        result_text += f"Task: {task_name}\n"
        result_text += f"Mode: {'Multi-anchor scan (' + str(n_anchors) + '×' + str(num_orientations) + '=' + str(total_opts) + ' opts/probe)' if use_scan else 'Single optimization (fast mode)'}\n\n"

        all_results = {}

        for i, probe in enumerate(probes):
            result_text += f"\n[{i+1}/{len(probes)}] Processing {probe}...\n"

            if use_scan:
                # Use scan_orientations for reliable results
                sub_run_name = f"{run_name}_{probe}"
                scan_args = {
                    "run_name": sub_run_name,
                    "probe": probe,
                    "target": target,
                    "substrate": substrate,
                    "n_anchors": n_anchors,
                    "num_orientations": num_orientations,
                    "task_name": task_name,
                    "fmax": fmax,
                    "charge": charge,
                    "spin": spin,
                    "workspace": str(workspace),  # Pass workspace for parallel-safe operation
                }

                # Call scan_orientations handler directly
                scan_result = await handle_scan_orientations(scan_args)
                scan_text = scan_result[0].text

                # Parse best result from scan output
                # Look for the best energy line (from eligible ranking)
                best_energy = None
                best_solution = None
                best_confidence = "high"
                n_flagged = 0
                for line in scan_text.split('\n'):
                    if '← BEST' in line:
                        # Parse energy from line like:
                        # "1. [near] x-axis 0°: -0.0854 eV (-1.97 kcal/mol) | Sol: -0.1003 eV [medium] ← BEST"
                        try:
                            parts = line.split(':')[1].split('eV')[0].strip()
                            best_energy = float(parts)
                            if '| Sol:' in line:
                                sol_part = line.split('| Sol:')[1].split('eV')[0].strip()
                                best_solution = float(sol_part)
                            # Parse confidence tag [high], [medium], etc.
                            import re
                            conf_match = re.search(r'\[(high|medium|low|confirmed_reactive|ml_artifact)\]', line)
                            if conf_match:
                                best_confidence = conf_match.group(1)
                        except:
                            pass
                        break
                    # Count flagged configurations
                    if 'Topology Guardrail:' in line:
                        try:
                            n_flagged = int(line.split(':')[1].split('/')[0].strip())
                        except:
                            pass

                if best_energy is not None:
                    all_results[probe] = {
                        "probe_target_vacuum": best_energy,
                        "scan_mode": True,
                        "n_configs": total_opts,
                        "confidence": best_confidence,
                        "n_flagged": n_flagged,
                    }
                    if best_solution is not None:
                        all_results[probe]["solvation"] = {"delta_G_solution": best_solution}
                    result_text += f"  Best: {best_energy:.4f} eV"
                    if best_solution:
                        result_text += f" | Solution: {best_solution:.4f} eV"
                    if best_confidence != "high":
                        result_text += f" [{best_confidence}]"
                    result_text += "\n"
                else:
                    all_results[probe] = {"error": "Could not parse scan results"}
                    result_text += f"  Error parsing results\n"

            else:
                # Legacy single-optimization mode
                sub_run_name = f"{run_name}_{probe}"
                config = {
                    "run_name": sub_run_name,
                    "probe": probe,
                    "substrate": substrate,
                    "task_name": task_name,
                    "fmax": fmax,
                    "charge": charge,
                    "spin": spin,
                    "device": _default_device_backend(),
                }
                if target:
                    config["target"] = target

                # Build
                config_path = simulations_dir / f"{sub_run_name}_config.json"
                config_path.parent.mkdir(parents=True, exist_ok=True)
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)

                # Redirect stdout for entire build+optimize to avoid MCP pollution
                old_stdout = sys.stdout
                sys.stdout = sys.stderr
                try:
                    SimulationBuilder = get_simulation_builder()
                    builder = SimulationBuilder(str(config_path), workspace=str(workspace))
                    structures = builder.build_simulation()
                    builder.save_structures(structures)
                    config_path.unlink(missing_ok=True)

                    config["fmax"] = fmax
                    opt_config_path = simulations_dir / f"{sub_run_name}" / "config_opt.json"
                    with open(opt_config_path, 'w') as f:
                        json.dump(config, f, indent=2)
                    SmartFlow = get_smart_flow()
                    flow = SmartFlow(str(opt_config_path), workspace=str(workspace))
                    await asyncio.to_thread(flow.run_workflow)
                    del flow
                    import gc
                    gc.collect()

                    interactions_path = simulations_dir / sub_run_name / "interactions.json"
                    if interactions_path.exists():
                        with open(interactions_path) as f:
                            all_results[probe] = json.load(f)
                    else:
                        all_results[probe] = {"status": "optimized but no interactions found"}

                    solvation_path = simulations_dir / sub_run_name / "solvation.json"
                    if solvation_path.exists():
                        with open(solvation_path) as f:
                            all_results[probe]["solvation"] = json.load(f)

                    # Check for topology validation results (from _run_flow_worker guardrail)
                    validation_path = simulations_dir / sub_run_name / "validation.json"
                    if validation_path.exists():
                        with open(validation_path) as f:
                            val_data = json.load(f)
                        all_results[probe]["confidence"] = val_data.get("confidence", "high")
                    else:
                        # Perform inline topology check for non-worker path
                        sim_dir = simulations_dir / sub_run_name
                        init_xyz = sim_dir / "probe_target_vacuum.xyz"
                        final_vasp = sim_dir / "probe_target_vacuum_optimized.vasp"
                        probe_xyz = sim_dir / "probe_vacuum.xyz"
                        if init_xyz.exists() and final_vasp.exists() and probe_xyz.exists():
                            try:
                                from ase.io import read as ase_read
                                from topology_validator import validate_topology
                                init_atoms = ase_read(str(init_xyz))
                                final_atoms = ase_read(str(final_vasp))
                                probe_atoms = ase_read(str(probe_xyz))
                                n_probe = len(probe_atoms)
                                topo = validate_topology(init_atoms, final_atoms, n_probe)
                                if topo["topology_preserved"]:
                                    all_results[probe]["confidence"] = "high"
                                else:
                                    all_results[probe]["confidence"] = "low"
                            except Exception:
                                all_results[probe]["confidence"] = "high"  # default if check fails

                except Exception as e:
                    all_results[probe] = {"error": str(e)}
                finally:
                    sys.stdout = old_stdout

        # Summary
        result_text += "\n" + "="*60 + "\n"
        result_text += "BATCH SCREENING RESULTS\n"
        result_text += "="*60 + "\n\n"

        # Sort by adsorption energy with confidence-aware ranking
        CONF_ELIGIBLE = {"high", "medium"}
        eligible_probes = []
        flagged_probes = []
        error_probes = []

        for probe, data in all_results.items():
            if isinstance(data, dict) and "error" not in data:
                # Determine energy value
                if "probe_target" in data:
                    energy = data["probe_target"]
                elif "probe_substrate" in data:
                    energy = data["probe_substrate"]
                elif "probe_target_vacuum" in data:
                    energy = data["probe_target_vacuum"]
                else:
                    energy = None

                if energy is None:
                    error_probes.append((probe, "no energy"))
                    continue

                confidence = data.get("confidence", "high")
                entry = (probe, energy, confidence)
                if confidence in CONF_ELIGIBLE:
                    eligible_probes.append(entry)
                else:
                    flagged_probes.append(entry)
            else:
                error_probes.append((probe, data.get("error", "unknown") if isinstance(data, dict) else "unknown"))

        eligible_probes.sort(key=lambda x: x[1])
        flagged_probes.sort(key=lambda x: x[1])

        result_text += "Ranked by binding energy (most favorable/negative first):\n\n"
        for rank, (probe, energy, confidence) in enumerate(eligible_probes, 1):
            kcal = energy * 23.061
            conf_tag = f" [{confidence}]" if confidence != "high" else ""
            line = f"{rank}. {probe}: {energy:.4f} eV ({kcal:.2f} kcal/mol)"
            if "solvation" in all_results[probe] and "delta_G_solution" in all_results[probe]["solvation"]:
                sol_energy = all_results[probe]["solvation"]["delta_G_solution"]
                sol_kcal = sol_energy * 23.061
                line += f" | Solution: {sol_energy:.4f} eV ({sol_kcal:.2f} kcal/mol)"
            if all_results[probe].get("scan_mode"):
                line += f" [scanned {all_results[probe].get('n_configs', '?')} configs]"
            line += conf_tag
            result_text += line + "\n"

        # Show flagged probes separately
        if flagged_probes:
            result_text += "\nFlagged probes (topology concerns, excluded from main ranking):\n"
            for probe, energy, confidence in flagged_probes:
                kcal = energy * 23.061
                line = f"  {probe}: {energy:.4f} eV ({kcal:.2f} kcal/mol) [{confidence}]"
                if "solvation" in all_results[probe] and "delta_G_solution" in all_results[probe]["solvation"]:
                    sol_energy = all_results[probe]["solvation"]["delta_G_solution"]
                    sol_kcal = sol_energy * 23.061
                    line += f" | Solution: {sol_energy:.4f} eV ({sol_kcal:.2f} kcal/mol)"
                result_text += line + "\n"

        # Show errors
        if error_probes:
            result_text += "\nErrors:\n"
            for probe, err in error_probes:
                result_text += f"  {probe}: {err}\n"

        if use_scan:
            result_text += f"\nNote: Each probe was screened with {total_opts} configurations (multi-anchor scan).\n"
        else:
            result_text += f"\nNote: Fast mode (single optimization). For more reliable results, use use_scan=true.\n"

        # Topology guardrail summary across all probes
        total_flagged = sum(all_results[p].get("n_flagged", 0) for p in all_results if isinstance(all_results[p], dict) and "error" not in all_results[p])
        if total_flagged > 0 or flagged_probes:
            result_text += f"\nTopology Guardrail Summary: {len(flagged_probes)} probe(s) had only flagged results as best.\n"

        return [TextContent(type="text", text=result_text)]

    except Exception as e:
        import traceback
        return [TextContent(type="text", text=f"Error in batch screening: {str(e)}\n\n{traceback.format_exc()}")]


# ==================== Orientation Scanning Handler ====================


def _check_upgrade_triggers(results: list, n_atoms: int, tier: int) -> tuple[bool, list[str]]:
    """
    Check if upgrade to higher tier is needed.

    Tier 1 → Tier 2 triggers:
    - HIGH_VARIANCE: energy std > 0.3 eV
    - TOP2_CLOSE: top 2 within 0.02 eV (~0.5 kcal/mol)
    - LOW_VALID_RATE: valid rate < 80%
    - LARGE_SYSTEM: n_atoms > 50

    Tier 2 → Tier 3 triggers:
    - BH_FOUND_BETTER: BH found energy > 0.1 eV lower than GS

    Returns: (should_upgrade, list of trigger reasons)
    """
    import numpy as np

    triggers = []

    if not results:
        return False, triggers

    # Get valid results with energies
    valid_results = [r for r in results if r.get("confidence", "high") in ("high", "medium")]
    if not valid_results:
        return False, triggers

    energies = [r["energy_eV"] for r in valid_results]

    if tier == 1:
        # HIGH_VARIANCE
        if len(energies) >= 3 and np.std(energies) > 0.3:
            triggers.append("HIGH_VARIANCE")

        # TOP2_CLOSE
        if len(energies) >= 2:
            sorted_e = sorted(energies)
            if sorted_e[1] - sorted_e[0] < 0.02:
                triggers.append("TOP2_CLOSE")

        # LOW_VALID_RATE
        valid_rate = len(valid_results) / len(results) if results else 1.0
        if valid_rate < 0.8:
            triggers.append("LOW_VALID_RATE")

        # LARGE_SYSTEM
        if n_atoms > 50:
            triggers.append("LARGE_SYSTEM")

    elif tier == 2:
        # Check if exploration found significantly better energy
        # This is checked after BH exploration
        pass

    return len(triggers) > 0, triggers


def _generate_bh_configs(
    base_config: dict,
    run_name: str,
    best_result: dict,
    n_samples: int = 6,
) -> list[dict]:
    """
    Generate Basin Hopping exploration configurations.

    Starts from the best result and applies random perturbations.
    """
    import numpy as np

    configs = []

    for i in range(n_samples):
        config = base_config.copy()
        config["run_name"] = f"{run_name}_bh_{i}"

        # Random perturbation from best orientation
        if "probe_orientation" in config:
            base_euler = config["probe_orientation"].get("euler", [0, 0, 0])
            perturbation = [np.random.uniform(-45, 45) for _ in range(3)]
            config["probe_orientation"] = {"euler": [b + p for b, p in zip(base_euler, perturbation)]}

        # Random position perturbation
        if "probe_position" in config:
            base_offset = config["probe_position"].get("lateral_offset", 0)
            config["probe_position"] = {
                "relative_to": "target",
                "lateral_offset": base_offset + np.random.uniform(-1.5, 1.5),
                "vertical_offset": np.random.uniform(-0.5, 0.5),
                "direction": config["probe_position"].get("direction", "x"),
            }
        else:
            config["probe_position"] = {
                "relative_to": "target",
                "lateral_offset": np.random.uniform(-2.0, 2.0),
                "vertical_offset": np.random.uniform(-0.5, 0.5),
                "direction": "x",
            }

        configs.append(config)

    return configs


async def handle_scan_orientations(args: dict) -> list[TextContent]:
    """Scan multiple positions (anchors) and orientations with SMART INCREMENTAL computation"""
    try:
        _scan_wall_start = time.time()

        # Resolve workspace (supports explicit parameter for parallel-safe operation)
        workspace, error = resolve_workspace(args)
        if error:
            return [TextContent(type="text", text=error)]

        simulations_dir = workspace / "simulations"

        import numpy as np
        from ase.io import read

        run_name = args["run_name"]
        probe = args["probe"]
        target = args["target"]
        substrate = args.get("substrate", "vacuum")
        n_anchors = args.get("n_anchors", 3)
        num_orientations = args.get("num_orientations", 3)
        random_samples = args.get("random_samples", 0)
        rotate_molecule = args.get("rotate_molecule", "probe")
        rotation_axis = args.get("rotation_axis", "x")
        task_name = args.get("task_name", "omol")
        fmax = args.get("fmax", 0.05)
        charge = args.get("charge", 0)
        spin = args.get("spin", 1)
        device_backend = _default_device_backend()

        # Auto-upgrade parameters
        auto_upgrade = args.get("auto_upgrade", True)  # Enable automatic tier upgrade
        max_tier = args.get("max_tier", 3)  # Maximum tier to upgrade to (1, 2, or 3)
        min_tier = args.get("min_tier", 1)  # Minimum tier to run

        total_grid_configs = n_anchors * num_orientations
        current_tier = 1
        upgrade_triggers = []
        tier_results = {1: [], 2: [], 3: []}  # Track results by tier

        result_text = f"Anchor-Orientation Scan: {run_name}\n"
        result_text += f"Workspace: {workspace}\n"
        result_text += f"Probe: {probe}, Target: {target}\n"
        result_text += f"Grid: {n_anchors} anchors × {num_orientations} orientations = {total_grid_configs} configs\n"
        if random_samples > 0:
            result_text += f"Random samples: {random_samples} (near best configuration)\n"
        result_text += f"Rotating: {rotate_molecule} around {rotation_axis}-axis\n"
        result_text += f"Task: {task_name}\n"
        if auto_upgrade:
            result_text += f"Auto-upgrade: enabled (max_tier={max_tier})\n"
        result_text += "Mode: SMART INCREMENTAL (skips existing results)\n\n"

        if n_anchors == 1:
            anchor_labels = ["center"]
        elif n_anchors == 3:
            anchor_labels = ["near", "mid", "far"]
        else:
            anchor_labels = [f"anchor_{i}" for i in range(n_anchors)]

        if rotation_axis == "all":
            angles_list = []
            for axis in ["x", "y", "z"]:
                for angle in np.linspace(0, 300, num_orientations // 3 + 1)[:-1]:
                    angles_list.append((axis, float(angle)))
            if len(angles_list) < num_orientations:
                for angle in np.linspace(0, 300, num_orientations - len(angles_list) + 1)[1:]:
                    angles_list.append(("z", float(angle)))
        else:
            angles_list = [
                (rotation_axis, float(a))
                for a in np.linspace(0, 360 - 360 / num_orientations, num_orientations)
            ]

        all_results = []
        contact_stats = {"success": 0, "failed": 0}
        skipped_count = 0
        computed_count = 0

        def load_result_entry(sub_run_name, anchor_label, anchor_idx, axis, angle, from_cache, is_random):
            interactions_path = simulations_dir / sub_run_name / "interactions.json"
            if not interactions_path.exists():
                return None

            with open(interactions_path) as f:
                interactions = json.load(f)

            energy = interactions.get("probe_target_vacuum", interactions.get("probe_target", None))
            if energy is None:
                return None

            result_entry = {
                "run_name": sub_run_name,
                "anchor": anchor_label,
                "anchor_idx": anchor_idx,
                "axis": axis,
                "angle": angle,
                "energy_eV": energy,
                "energy_kcal": energy * 23.061,
                "from_cache": from_cache,
                "is_random": is_random,
            }

            opt_xyz = simulations_dir / sub_run_name / "optimized_probe_target_vacuum.xyz"
            if opt_xyz.exists():
                opt_atoms = read(opt_xyz)
                n_atoms = len(opt_atoms)
                mid = n_atoms // 2
                contact_info = check_contact_state(
                    opt_atoms,
                    probe_indices=list(range(mid, n_atoms)),
                    target_indices=list(range(0, mid)),
                )
                result_entry["is_contact"] = contact_info["is_contact"]
                result_entry["min_distance"] = contact_info["min_distance"]
                result_entry["com_distance"] = contact_info["com_distance"]

            solvation_path = simulations_dir / sub_run_name / "solvation.json"
            if solvation_path.exists():
                with open(solvation_path) as f:
                    solvation = json.load(f)
                if "delta_G_solution" in solvation:
                    result_entry["solution_eV"] = solvation["delta_G_solution"]
                    result_entry["solution_kcal"] = solvation["delta_G_solution"] * 23.061

            # Load topology validation results if available
            validation_path = simulations_dir / sub_run_name / "validation.json"
            if validation_path.exists():
                with open(validation_path) as f:
                    val_data = json.load(f)
                result_entry["confidence"] = val_data.get("confidence", "high")
                result_entry["validation_alerts"] = val_data.get("alerts", [])
            else:
                result_entry["confidence"] = "high"  # No validation = assume OK

            return result_entry

        def build_config(sub_run_name, axis, angle, anchor_frac, is_random=False, random_perturbation=None):
            if axis == "x":
                euler = [angle, 0, 0]
            elif axis == "y":
                euler = [0, angle, 0]
            else:
                euler = [0, 0, angle]

            if random_perturbation is not None:
                euler = [e + p for e, p in zip(euler, random_perturbation)]

            config = {
                "run_name": sub_run_name,
                "probe": probe,
                "target": target,
                "substrate": substrate,
                "task_name": task_name,
                "fmax": fmax,
                "charge": charge,
                "spin": spin,
                "device": device_backend,
            }

            if rotate_molecule == "probe":
                config["probe_orientation"] = {"euler": euler}
            else:
                config["target_orientation"] = {"euler": euler}

            if n_anchors > 1 or is_random:
                offset = (anchor_frac - 0.5) * 4.0
                if random_perturbation is not None:
                    offset += np.random.uniform(-1.0, 1.0)
                config["probe_position"] = {
                    "relative_to": "target",
                    "lateral_offset": offset,
                    "vertical_offset": 0,
                    "direction": "x",
                }

            return config

        result_text += "--- Grid Sampling ---\n"
        grid_sequence = []
        grid_tasks = []
        config_idx = 0

        for anchor_idx in range(n_anchors):
            anchor_label = anchor_labels[anchor_idx]
            anchor_frac = 0.5 if n_anchors == 1 else 0.25 + 0.5 * anchor_idx / (n_anchors - 1)

            for angle_idx, (axis, angle) in enumerate(angles_list):
                config_idx += 1
                sub_run_name = f"{run_name}_a{anchor_idx}_o{angle_idx}"
                entry = {
                    "config_idx": config_idx,
                    "sub_run_name": sub_run_name,
                    "anchor_label": anchor_label,
                    "anchor_idx": anchor_idx,
                    "axis": axis,
                    "angle": angle,
                    "anchor_frac": anchor_frac,
                }

                cached = load_result_entry(sub_run_name, anchor_label, anchor_idx, axis, angle, True, False)
                if cached:
                    all_results.append(cached)
                    skipped_count += 1
                    entry["cached"] = cached
                else:
                    entry["config"] = build_config(sub_run_name, axis, angle, anchor_frac)
                    grid_tasks.append(entry)
                grid_sequence.append(entry)

        await _run_multi_config_tasks(grid_tasks, workspace)

        for entry in grid_sequence:
            prefix = (
                f"[{entry['config_idx']}/{total_grid_configs}] {entry['anchor_label']}, "
                f"{entry['axis']}-axis {entry['angle']:.0f}°... "
            )
            if "cached" in entry:
                cached = entry["cached"]
                result_text += f"{prefix}CACHED {cached['energy_eV']:.4f} eV\n"
                continue

            worker_result = entry.get("worker_result", {})
            if worker_result.get("error"):
                result_text += f"{prefix}Error: {worker_result['error']}\n"
                continue

            result_entry = load_result_entry(
                entry["sub_run_name"],
                entry["anchor_label"],
                entry["anchor_idx"],
                entry["axis"],
                entry["angle"],
                False,
                False,
            )
            if result_entry:
                all_results.append(result_entry)
                computed_count += 1
                result_text += (
                    f"{prefix}{result_entry['energy_eV']:.4f} eV "
                    f"({result_entry['energy_kcal']:.2f} kcal/mol)\n"
                )
            else:
                result_text += f"{prefix}No result\n"

        # --- Tier 1 complete: Check auto-upgrade triggers ---
        tier_results[1] = list(all_results)  # Save Tier 1 results
        n_atoms_total = 0
        if all_results:
            # Estimate n_atoms from first result
            first_result = all_results[0]
            opt_xyz = simulations_dir / first_result["run_name"] / "optimized_probe_target_vacuum.xyz"
            if opt_xyz.exists():
                n_atoms_total = len(read(opt_xyz))

        gs_best_energy = min((r["energy_eV"] for r in all_results if r.get("confidence", "high") in ("high", "medium")), default=float('inf'))

        if auto_upgrade and max_tier >= 2 and all_results:
            should_upgrade, triggers = _check_upgrade_triggers(all_results, n_atoms_total, tier=1)

            if should_upgrade or min_tier >= 2:
                current_tier = 2
                upgrade_triggers.extend(triggers)
                result_text += f"\n--- Tier 2: Basin Hopping Exploration ---\n"
                result_text += f"Upgrade triggers: {', '.join(triggers) if triggers else 'min_tier=2'}\n"

                # Get best result from Tier 1 as starting point
                best_t1 = min(all_results, key=lambda x: x["energy_eV"])

                # Generate BH exploration configs
                base_config = {
                    "probe": probe,
                    "target": target,
                    "substrate": substrate,
                    "task_name": task_name,
                    "fmax": fmax,
                    "charge": charge,
                    "spin": spin,
                    "device": device_backend,
                }
                if rotate_molecule == "probe":
                    base_config["probe_orientation"] = {"euler": [best_t1["angle"], 0, 0]}
                else:
                    base_config["target_orientation"] = {"euler": [best_t1["angle"], 0, 0]}

                bh_configs = _generate_bh_configs(base_config, run_name, best_t1, n_samples=6)

                # Prepare BH tasks
                bh_tasks = []
                for bh_idx, bh_config in enumerate(bh_configs):
                    sub_run_name = bh_config["run_name"]
                    cached = load_result_entry(sub_run_name, "bh", -1, "bh", bh_idx, True, False)
                    entry = {
                        "bh_idx": bh_idx,
                        "sub_run_name": sub_run_name,
                    }
                    if cached:
                        all_results.append(cached)
                        tier_results[2].append(cached)
                        skipped_count += 1
                        entry["cached"] = cached
                    else:
                        entry["config"] = bh_config
                        bh_tasks.append(entry)

                # Run BH tasks
                if bh_tasks:
                    await _run_multi_config_tasks(bh_tasks, workspace)

                    for entry in bh_tasks:
                        bh_idx = entry["bh_idx"]
                        prefix = f"[BH{bh_idx+1}/6] "

                        if "cached" in entry:
                            cached = entry["cached"]
                            result_text += f"{prefix}CACHED {cached['energy_eV']:.4f} eV\n"
                            continue

                        worker_result = entry.get("worker_result", {})
                        if worker_result.get("error"):
                            result_text += f"{prefix}Error: {worker_result['error']}\n"
                            continue

                        result_entry = load_result_entry(
                            entry["sub_run_name"], "bh", -1, "bh", bh_idx, False, False
                        )
                        if result_entry:
                            result_entry["tier"] = 2
                            all_results.append(result_entry)
                            tier_results[2].append(result_entry)
                            computed_count += 1
                            result_text += f"{prefix}{result_entry['energy_eV']:.4f} eV ({result_entry['energy_kcal']:.2f} kcal/mol)\n"
                        else:
                            result_text += f"{prefix}No result\n"

                # Check if BH found better energy
                bh_best_energy = min((r["energy_eV"] for r in tier_results[2] if r.get("confidence", "high") in ("high", "medium")), default=float('inf'))
                if bh_best_energy < gs_best_energy - 0.1:
                    upgrade_triggers.append("BH_FOUND_BETTER")
                    result_text += f"\nBH found better energy: {bh_best_energy:.4f} eV vs GS {gs_best_energy:.4f} eV\n"

                # Check Tier 2 → Tier 3 upgrade
                if auto_upgrade and max_tier >= 3:
                    should_upgrade_t3, triggers_t3 = _check_upgrade_triggers(all_results, n_atoms_total, tier=2)
                    if (should_upgrade_t3 or "BH_FOUND_BETTER" in upgrade_triggers or min_tier >= 3):
                        current_tier = 3
                        upgrade_triggers.extend(triggers_t3)
                        # Tier 3 will be handled by the existing random_samples logic below
                        if random_samples == 0:
                            random_samples = 9  # Auto-add random samples for Tier 3
                            result_text += f"\n--- Tier 3: Extended Random Exploration (auto-enabled) ---\n"

        if random_samples > 0 and all_results:
            result_text += f"\n--- Random Sampling ({random_samples} samples) ---\n"
            best_so_far = min(all_results, key=lambda x: x["energy_eV"])
            best_anchor_idx = best_so_far["anchor_idx"]
            best_anchor_frac = 0.5 if n_anchors == 1 else 0.25 + 0.5 * best_anchor_idx / (n_anchors - 1)

            random_sequence = []
            random_tasks = []
            for rand_idx in range(random_samples):
                sub_run_name = f"{run_name}_random_{rand_idx}"
                entry = {
                    "rand_idx": rand_idx,
                    "sub_run_name": sub_run_name,
                    "axis": best_so_far["axis"],
                    "angle": best_so_far["angle"],
                    "anchor_idx": best_anchor_idx,
                }
                cached = load_result_entry(sub_run_name, "random", -1, "random", rand_idx, True, True)
                if cached:
                    all_results.append(cached)
                    skipped_count += 1
                    entry["cached"] = cached
                else:
                    perturbation = [np.random.uniform(-30, 30) for _ in range(3)]
                    entry["config"] = build_config(
                        sub_run_name,
                        best_so_far["axis"],
                        best_so_far["angle"],
                        best_anchor_frac,
                        is_random=True,
                        random_perturbation=perturbation,
                    )
                    random_tasks.append(entry)
                random_sequence.append(entry)

            await _run_multi_config_tasks(random_tasks, workspace)

            for entry in random_sequence:
                prefix = f"[R{entry['rand_idx']+1}/{random_samples}] random... "
                if "cached" in entry:
                    cached = entry["cached"]
                    result_text += f"{prefix}CACHED {cached['energy_eV']:.4f} eV\n"
                    continue

                worker_result = entry.get("worker_result", {})
                if worker_result.get("error"):
                    result_text += f"{prefix}Error: {worker_result['error']}\n"
                    continue

                result_entry = load_result_entry(
                    entry["sub_run_name"],
                    "random",
                    -1,
                    entry["axis"],
                    entry["angle"],
                    False,
                    True,
                )
                if result_entry:
                    all_results.append(result_entry)
                    computed_count += 1
                    result_text += (
                        f"{prefix}{result_entry['energy_eV']:.4f} eV "
                        f"({result_entry['energy_kcal']:.2f} kcal/mol)\n"
                    )
                else:
                    result_text += f"{prefix}No result\n"

        for r in all_results:
            if "is_contact" in r:
                if r["is_contact"]:
                    contact_stats["success"] += 1
                else:
                    contact_stats["failed"] += 1

        if all_results:
            result_text += "\n" + "=" * 60 + "\n"
            result_text += "SCAN RESULTS\n"
            result_text += "=" * 60 + "\n\n"

            # Tier information
            result_text += f"Sampling Tier: {current_tier}"
            if upgrade_triggers:
                result_text += f" (upgraded due to: {', '.join(upgrade_triggers)})"
            result_text += "\n"
            result_text += f"  Tier 1 (Grid): {len(tier_results[1])} configs\n"
            if tier_results[2]:
                result_text += f"  Tier 2 (BH): {len(tier_results[2])} configs\n"
            if current_tier == 3:
                result_text += f"  Tier 3 (Random): {random_samples} configs\n"
            result_text += "\n"

            result_text += (
                f"Computed: {computed_count}, Cached: {skipped_count}, Total: {len(all_results)}\n\n"
            )

            # Confidence-aware ranking:
            # Eligible results: high and medium confidence
            # Low / confirmed_reactive / ml_artifact are listed separately
            CONF_ELIGIBLE = {"high", "medium"}
            CONF_ORDER = {"high": 0, "medium": 1, "low": 2, "confirmed_reactive": 3, "ml_artifact": 4}

            eligible = [r for r in all_results if r.get("confidence", "high") in CONF_ELIGIBLE]
            flagged = [r for r in all_results if r.get("confidence", "high") not in CONF_ELIGIBLE]

            # Sort eligible by energy (primary ranking criterion)
            sorted_eligible = sorted(eligible, key=lambda x: x["energy_eV"])
            sorted_flagged = sorted(flagged, key=lambda x: x["energy_eV"])

            # --- Energy Consistency Guard ---
            # Detect ML energy prediction instability (large energy variance + small geometry variance)
            from topology_validator import validate_energy_consistency
            ecg_result = validate_energy_consistency(sorted_eligible)

            # Add ECG alert if instability detected
            if ecg_result["status"] != "stable":
                result_text += "=" * 60 + "\n"
                result_text += f"ENERGY CONSISTENCY GUARD: {ecg_result['status'].upper()}\n"
                result_text += "=" * 60 + "\n"
                result_text += f"{ecg_result['details']}\n\n"

                if ecg_result["selection_mode"] == "median":
                    result_text += f"RECOMMENDED: Use median energy ({ecg_result['recommended_value_eV']:.4f} eV) "
                    result_text += "instead of minimum.\n\n"
                elif ecg_result["selection_mode"] == "manual_review":
                    result_text += "RECOMMENDED: Manual review required. Do NOT trust minimum energy.\n\n"

                if ecg_result["dft_recommended"]:
                    result_text += "DFT VERIFICATION STRONGLY RECOMMENDED for this molecular pair.\n\n"

                # Mark extreme outliers in sorted_eligible
                extreme_outliers = set(ecg_result.get("extreme_outliers", []))
                for r in sorted_eligible:
                    if r["run_name"] in extreme_outliers:
                        r["is_extreme_outlier"] = True

            result_text += "Ranked by interaction energy (most favorable first):\n\n"
            for rank, r in enumerate(sorted_eligible, 1):
                # Modified marker based on ECG status
                if ecg_result["status"] == "stable":
                    marker = " ← BEST" if rank == 1 else ""
                elif r.get("is_extreme_outlier"):
                    marker = " [EXTREME OUTLIER - EXCLUDED]"
                elif rank == 1 and ecg_result["selection_mode"] != "minimum":
                    marker = " [minimum - NOT RECOMMENDED]"
                else:
                    marker = ""
                label = "random" if r.get("is_random") else f"{r['anchor']}"
                conf = r.get("confidence", "high")
                conf_tag = f" [{conf}]" if conf != "high" else ""
                line = f"{rank}. [{label}] {r['axis']}-axis {r['angle']:.0f}°: "
                line += f"{r['energy_eV']:.4f} eV ({r['energy_kcal']:.2f} kcal/mol)"
                if "solution_eV" in r:
                    line += f" | Sol: {r['solution_eV']:.4f} eV ({r['solution_kcal']:.2f} kcal/mol)"
                if r.get("from_cache"):
                    line += " [cached]"
                line += conf_tag + marker
                result_text += line + "\n"

            # Show flagged results separately
            if sorted_flagged:
                result_text += "\nFlagged results (excluded from ranking):\n"
                for r in sorted_flagged:
                    label = "random" if r.get("is_random") else f"{r['anchor']}"
                    conf = r.get("confidence", "unknown")
                    line = f"  [{label}] {r['axis']}-axis {r['angle']:.0f}°: "
                    line += f"{r['energy_eV']:.4f} eV ({r['energy_kcal']:.2f} kcal/mol)"
                    line += f" [{conf}]"
                    alerts = r.get("validation_alerts", [])
                    if alerts:
                        line += f" — {alerts[0]}"
                    result_text += line + "\n"

            if contact_stats["success"] + contact_stats["failed"] > 0:
                total = contact_stats["success"] + contact_stats["failed"]
                success_rate = contact_stats["success"] / total * 100
                result_text += (
                    f"\nContact State Success Rate: {contact_stats['success']}/{total} ({success_rate:.0f}%)\n"
                )
                if success_rate < 50:
                    result_text += "⚠️ Warning: Low contact success rate. Results may be unreliable.\n"

            # Use eligible results for uncertainty check
            if len(sorted_eligible) >= 2:
                delta_kcal = sorted_eligible[1]["energy_kcal"] - sorted_eligible[0]["energy_kcal"]
                if abs(delta_kcal) < 1.0:
                    result_text += (
                        f"\n⚠️ Uncertainty: Top 2 candidates within {abs(delta_kcal):.2f} kcal/mol\n"
                    )
                    result_text += "Consider high-tier sampling (random_samples=5-10) for confirmation.\n"

            # Topology guardrail summary
            n_flagged = len(sorted_flagged)
            if n_flagged > 0:
                result_text += (
                    f"\nTopology Guardrail: {n_flagged}/{len(all_results)} configurations flagged "
                    f"(intramolecular bond changes detected)\n"
                )

            # --- Variance guard (adaptive statistics, no hardcoded thresholds) ---
            if len(sorted_eligible) >= 4:
                energies_arr = np.array([r["energy_eV"] for r in sorted_eligible])
                e_mean = float(np.mean(energies_arr))
                e_std = float(np.std(energies_arr))
                e_q1, e_q3 = float(np.percentile(energies_arr, 25)), float(np.percentile(energies_arr, 75))
                e_iqr = e_q3 - e_q1
                e_range = float(np.max(energies_arr) - np.min(energies_arr))
                e_best = float(np.min(energies_arr))
                best_is_outlier = e_best < e_q1 - 1.5 * e_iqr if e_iqr > 1e-6 else False

                result_text += (
                    f"\nEnergy Statistics (N={len(sorted_eligible)}): "
                    f"mean={e_mean:.4f}, σ={e_std:.4f}, "
                    f"IQR={e_iqr:.4f}, range={e_range:.4f} eV\n"
                )
                if best_is_outlier:
                    result_text += (
                        "⚠️ Best result is a statistical outlier (below Q1-1.5×IQR). "
                        "Consider higher-tier sampling for confirmation.\n"
                    )

                # Also report solution energy stats if available
                sol_energies = [r["solution_eV"] for r in sorted_eligible if "solution_eV" in r]
                sol_outlier = False
                if len(sol_energies) >= 4:
                    sol_arr = np.array(sol_energies)
                    s_mean = float(np.mean(sol_arr))
                    s_std = float(np.std(sol_arr))
                    s_q1, s_q3 = float(np.percentile(sol_arr, 25)), float(np.percentile(sol_arr, 75))
                    s_iqr = s_q3 - s_q1
                    s_best = float(np.min(sol_arr))
                    sol_outlier = s_best < s_q1 - 1.5 * s_iqr if s_iqr > 1e-6 else False
                    result_text += (
                        f"Solution Energy Stats (N={len(sol_energies)}): "
                        f"mean={s_mean:.4f}, σ={s_std:.4f}, IQR={s_iqr:.4f} eV\n"
                    )
                    if sol_outlier:
                        result_text += (
                            "⚠️ Best solution energy is a statistical outlier.\n"
                        )

                # --- B8: Structured variance guard summary with recommendations ---
                # Check uncertainty (top 2 within 1 kcal/mol)
                top2_uncertain = False
                if len(sorted_eligible) >= 2:
                    delta_kcal = sorted_eligible[1]["energy_kcal"] - sorted_eligible[0]["energy_kcal"]
                    if abs(delta_kcal) < 1.0:
                        top2_uncertain = True

                # Check contact success rate
                low_success_rate = False
                total_contacts = contact_stats["success"] + contact_stats["failed"]
                if total_contacts > 0:
                    success_rate_pct = contact_stats["success"] / total_contacts * 100
                    if success_rate_pct < 50:
                        low_success_rate = True

                # Build machine-actionable recommendations
                recommendations = []
                if best_is_outlier or sol_outlier:
                    recommendations.append({
                        "action": "increase_sampling",
                        "reason": "best_result_is_statistical_outlier",
                        "suggestion": "Add random_samples=5-10 near best configuration"
                    })
                if top2_uncertain:
                    recommendations.append({
                        "action": "increase_sampling",
                        "reason": "top2_within_1_kcal_mol",
                        "suggestion": "Results too close to distinguish; add random_samples=5-10"
                    })
                if low_success_rate:
                    recommendations.append({
                        "action": "check_geometry",
                        "reason": "low_contact_success_rate",
                        "suggestion": "Many configurations failed contact distance check; review initial geometry"
                    })

                # Determine overall recommendation
                # First check ECG status (takes priority)
                if ecg_result["status"] == "extreme_instability":
                    overall = "dft_required"
                elif ecg_result["status"] == "unstable":
                    overall = "dft_recommended"
                elif not recommendations:
                    # Higher tier = higher confidence
                    if current_tier >= 2:
                        overall = "high_confidence"
                    else:
                        overall = "confident"
                elif any(r["action"] == "increase_sampling" for r in recommendations):
                    # If we already upgraded but still seeing issues, recommend DFT
                    if current_tier >= 2:
                        overall = "dft_recommended"
                    else:
                        overall = "increase_sampling"
                else:
                    overall = "review_needed"

                variance_guard = {
                    "n_configs": len(sorted_eligible),
                    "mean_eV": e_mean,
                    "std_eV": e_std,
                    "iqr_eV": e_iqr,
                    "range_eV": e_range,
                    "best_is_outlier": best_is_outlier,
                    "sol_best_is_outlier": sol_outlier,
                    "top2_within_1_kcal": top2_uncertain,
                    "low_contact_success": low_success_rate,
                    "recommendations": recommendations,
                    "overall_recommendation": overall,
                }

                # Save scan summary JSON
                # Supp-1: Add rank_only flag for substrate mode
                is_substrate_mode = substrate.lower() not in ("vacuum", "vac")

                scan_summary = {
                    "run_name": run_name,
                    "probe": probe,
                    "target": target,
                    "substrate": substrate,
                    "task_name": task_name,
                    "n_anchors": n_anchors,
                    "num_orientations": num_orientations,
                    "random_samples": random_samples,
                    "total_configs": len(all_results),
                    "eligible_configs": len(sorted_eligible),
                    "flagged_configs": len(sorted_flagged),
                    "variance_guard": variance_guard,
                    "rank_only": is_substrate_mode,
                    # Tiered sampling information
                    "sampling_tier": {
                        "tier_used": current_tier,
                        "auto_upgrade": auto_upgrade,
                        "upgrade_triggers": upgrade_triggers,
                        "tier_1_configs": len(tier_results[1]),
                        "tier_2_configs": len(tier_results[2]),
                        "tier_3_configs": len(tier_results[3]) if tier_results[3] else 0,
                    },
                    # Energy Consistency Guard results
                    "energy_consistency": {
                        "status": ecg_result["status"],
                        "selection_mode": ecg_result["selection_mode"],
                        "recommended_value_eV": ecg_result["recommended_value_eV"],
                        "dft_recommended": ecg_result["dft_recommended"],
                        "statistics": ecg_result.get("statistics", {}),
                        "extreme_outliers": ecg_result.get("extreme_outliers", []),
                    },
                    # Wall-clock timing
                    "wall_time_seconds": round(time.time() - _scan_wall_start, 2),
                }
                # Report best energy based on ECG selection mode
                if ecg_result["selection_mode"] == "minimum":
                    scan_summary["best_energy_eV"] = sorted_eligible[0]["energy_eV"] if sorted_eligible else None
                    scan_summary["best_config"] = sorted_eligible[0]["run_name"] if sorted_eligible else None
                else:
                    # For unstable/extreme_instability, report recommended value instead
                    scan_summary["recommended_energy_eV"] = ecg_result["recommended_value_eV"]
                    scan_summary["minimum_energy_eV"] = sorted_eligible[0]["energy_eV"] if sorted_eligible else None
                    scan_summary["minimum_config"] = sorted_eligible[0]["run_name"] if sorted_eligible else None
                    scan_summary["warning"] = ecg_result["details"]
                if is_substrate_mode:
                    scan_summary["rank_only_reason"] = (
                        f"Substrate mode ({substrate}): ML potential not trained on 2D material adsorption. "
                        "Absolute energies are unreliable; use results only for relative ranking."
                    )
                if sorted_eligible and "solution_eV" in sorted_eligible[0]:
                    scan_summary["best_solution_eV"] = sorted_eligible[0]["solution_eV"]

                # Save to base run directory
                base_run_dir = simulations_dir / run_name
                base_run_dir.mkdir(parents=True, exist_ok=True)
                summary_path = base_run_dir / "scan_summary.json"
                with open(summary_path, 'w', encoding='utf-8') as f:
                    json.dump(scan_summary, f, indent=2)
                result_text += f"\nScan summary saved: {summary_path}\n"
                result_text += f"Overall recommendation: {overall}\n"

        else:
            result_text += "\nNo valid results obtained.\n"

        _scan_wall_seconds = time.time() - _scan_wall_start
        result_text += f"\nTotal wall time: {_scan_wall_seconds:.1f} s\n"

        return [TextContent(type="text", text=result_text)]

    except Exception as e:
        import traceback
        return [TextContent(type="text", text=f"Error in scan orientations: {str(e)}\n\n{traceback.format_exc()}")]


# ==================== Results Handler ====================

async def handle_get_simulation_results(args: dict) -> list[TextContent]:
    """Get simulation results"""
    try:
        # Resolve workspace (supports explicit parameter for parallel-safe operation)
        workspace, error = resolve_workspace(args)
        if error:
            return [TextContent(type="text", text=error)]

        simulations_dir = workspace / "simulations"
        run_name = args["run_name"]
        sim_dir = simulations_dir / run_name

        if not sim_dir.exists():
            return [TextContent(type="text", text=f"Simulation '{run_name}' not found.")]

        result = f"Workspace: {workspace}\n"
        result += f"Simulation: {run_name}\n"
        result += f"Directory: {sim_dir}\n\n"

        # Read config
        config_path = sim_dir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            result += "Configuration:\n"
            result += f"  Probe: {config.get('probe', 'N/A')}\n"
            if config.get('target'):
                result += f"  Target: {config.get('target')}\n"
            result += f"  Substrate: {config.get('substrate', 'vacuum')}\n\n"

        # Read summary
        summary_path = sim_dir / "summary.txt"
        if summary_path.exists():
            result += "Summary:\n"
            result += summary_path.read_text()
            result += "\n"

        # Read results.json
        results_path = sim_dir / "results.json"
        if results_path.exists():
            with open(results_path) as f:
                results = json.load(f)

            if "energies" in results:
                result += "Energies (eV):\n"
                for name, energy in results["energies"].items():
                    result += f"  {name}: {energy:.4f}\n"

            if "binding_energies" in results:
                result += "\nAdsorption/Interaction Energies:\n"
                for name, energy in results["binding_energies"].items():
                    kcal = energy * 23.061
                    result += f"  {name}: {energy:.4f} eV ({kcal:.2f} kcal/mol)\n"

        # Read solvation data if available
        solvation_path = sim_dir / "solvation.json"
        if solvation_path.exists():
            with open(solvation_path) as f:
                solvation = json.load(f)
            result += "\nSolvation Analysis (xTB GFN2-xTB + ALPB water):\n"
            for name in ["probe_vacuum", "target_vacuum", "probe_target_vacuum"]:
                if name in solvation:
                    G_solv = solvation[name]
                    result += f"  G_solv({name}): {G_solv:.4f} eV ({G_solv * 23.061:.2f} kcal/mol)\n"
            if "delta_G_solvation" in solvation:
                dG_solv = solvation["delta_G_solvation"]
                result += f"\n  ΔG_solvation: {dG_solv:.4f} eV ({dG_solv * 23.061:.2f} kcal/mol)\n"
            if "delta_G_solution" in solvation:
                dG_sol = solvation["delta_G_solution"]
                result += f"  Solution binding: {dG_sol:.4f} eV ({dG_sol * 23.061:.2f} kcal/mol)\n"

        # Read topology validation if available
        validation_path = sim_dir / "validation.json"
        if validation_path.exists():
            with open(validation_path) as f:
                val_data = json.load(f)
            result += "\nTopology Validation:\n"
            result += f"  Confidence: {val_data.get('confidence', 'N/A')}\n"
            if val_data.get("primary_task"):
                result += f"  Primary task: {val_data['primary_task']}\n"
            if val_data.get("adopted_task"):
                result += f"  Adopted task: {val_data['adopted_task']}\n"
            if val_data.get("alerts"):
                for alert in val_data["alerts"]:
                    result += f"  Alert: {alert}\n"
            if val_data.get("details"):
                result += f"  Details: {val_data['details']}\n"

        # List files
        result += "\nAvailable files:\n"
        for f in sorted(sim_dir.glob("*")):
            if f.is_file():
                size = f.stat().st_size
                if size > 1024*1024:
                    size_str = f"{size/1024/1024:.1f} MB"
                elif size > 1024:
                    size_str = f"{size/1024:.1f} KB"
                else:
                    size_str = f"{size} B"
                result += f"  - {f.name} ({size_str})\n"

        return [TextContent(type="text", text=result)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error getting results: {str(e)}")]


# ==================== Structure Analysis Handler ====================

async def handle_analyze_structure(args: dict) -> list[TextContent]:
    """Analyze a molecular structure"""
    try:
        from ase.io import read
        import numpy as np

        structure_path = args["structure_path"]

        if not os.path.isabs(structure_path):
            # Resolve workspace (supports explicit parameter, not required)
            workspace, _ = resolve_workspace(args, require=False)
            if workspace:
                structure_path = workspace / structure_path
            else:
                structure_path = Path(__file__).parent / structure_path

        if not Path(structure_path).exists():
            return [TextContent(type="text", text=f"Structure file not found: {structure_path}")]

        # Redirect stdout to avoid MCP pollution
        old_stdout = sys.stdout
        sys.stdout = sys.stderr
        try:
            atoms = read(structure_path)
        finally:
            sys.stdout = old_stdout

        # Basic info
        result = f"Structure Analysis: {Path(structure_path).name}\n\n"
        result += f"Atoms: {len(atoms)}\n"
        result += f"Formula: {atoms.get_chemical_formula()}\n"

        # Element composition
        symbols = atoms.get_chemical_symbols()
        unique_elements = sorted(set(symbols))
        result += f"Elements: {', '.join(unique_elements)}\n"
        result += "Composition:\n"
        for elem in unique_elements:
            count = symbols.count(elem)
            result += f"  {elem}: {count}\n"

        # Cell info
        cell = atoms.get_cell()
        if cell.any():
            result += f"\nCell dimensions:\n"
            result += f"  a = {cell[0, 0]:.2f} Å\n"
            result += f"  b = {cell[1, 1]:.2f} Å\n"
            result += f"  c = {cell[2, 2]:.2f} Å\n"
            result += f"  Volume = {atoms.get_volume():.2f} Å³\n"

        # Position range
        positions = atoms.get_positions()
        result += f"\nPosition range:\n"
        result += f"  x: {positions[:, 0].min():.2f} to {positions[:, 0].max():.2f} Å\n"
        result += f"  y: {positions[:, 1].min():.2f} to {positions[:, 1].max():.2f} Å\n"
        result += f"  z: {positions[:, 2].min():.2f} to {positions[:, 2].max():.2f} Å\n"

        # Center of mass
        com = atoms.get_center_of_mass()
        result += f"\nCenter of mass: ({com[0]:.2f}, {com[1]:.2f}, {com[2]:.2f}) Å\n"

        return [TextContent(type="text", text=result)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error analyzing structure: {str(e)}")]


# ============================================================
# Resource Definitions
# ============================================================

@server.list_resources()
async def list_resources() -> list[Resource]:
    """List available resources"""
    resources = []

    # Shared molecules from MCP server directory
    molecules_dir = MCP_SERVER_DIR / "molecules"
    if molecules_dir.exists():
        for f in sorted(molecules_dir.glob("*.sdf")):
            if not f.name.startswith('.'):
                resources.append(Resource(
                    uri=f"rapids://molecules/{f.stem}",
                    name=f"Molecule: {f.stem}",
                    description=f"Cached molecule structure",
                    mimeType="chemical/x-mdl-sdfile"
                ))

    # Rare molecules from MCP server directory
    if RARE_MOLECULES_DIR.exists():
        for f in sorted(RARE_MOLECULES_DIR.glob("*.sdf")):
            if not f.name.startswith('.'):
                resources.append(Resource(
                    uri=f"rapids://rare_molecules/{f.stem}",
                    name=f"Rare: {f.stem}",
                    description=f"Pre-optimized complex molecule",
                    mimeType="chemical/x-mdl-sdfile"
                ))

    # Simulations from current workspace
    simulations_dir = get_simulations_dir()
    if simulations_dir and simulations_dir.exists():
        for sim_dir in sorted(simulations_dir.iterdir()):
            if sim_dir.is_dir() and (sim_dir / "summary.txt").exists():
                resources.append(Resource(
                    uri=f"rapids://simulations/{sim_dir.name}",
                    name=f"Simulation: {sim_dir.name}",
                    description=f"Results from simulation run '{sim_dir.name}'",
                    mimeType="text/plain"
                ))

    return resources


@server.read_resource()
async def read_resource(uri) -> str:
    """Read a resource"""
    uri_str = str(uri)

    # Shared molecules
    if uri_str.startswith("rapids://molecules/"):
        mol_name = uri_str.replace("rapids://molecules/", "")
        mol_path = MCP_SERVER_DIR / "molecules" / f"{mol_name}.sdf"
        if mol_path.exists():
            return mol_path.read_text()
        return f"Molecule not found: {mol_name}"

    # Rare molecules
    if uri_str.startswith("rapids://rare_molecules/"):
        mol_name = uri_str.replace("rapids://rare_molecules/", "")
        mol_path = RARE_MOLECULES_DIR / f"{mol_name}.sdf"
        if mol_path.exists():
            return mol_path.read_text()
        return f"Rare molecule not found: {mol_name}"

    # Simulations (require workspace)
    if uri_str.startswith("rapids://simulations/"):
        simulations_dir = get_simulations_dir()
        if simulations_dir is None:
            return "No workspace set. Call set_workspace(path) first."

        run_name = uri_str.replace("rapids://simulations/", "")
        sim_dir = simulations_dir / run_name

        if sim_dir.exists():
            summary_path = sim_dir / "summary.txt"
            if summary_path.exists():
                return summary_path.read_text()

    return f"Resource not found: {uri_str}"


# ============================================================
# Main Entry Point
# ============================================================

async def main():
    """Run the MCP server"""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
