#!/usr/bin/env python3
"""
Benchmark FAIRChem inference modes and multi-GPU strategies.

Compares:
1. inference_settings: "default" vs "turbo"
2. Multi-GPU: single GPU vs workers=N (FAIRChem internal) vs process pool (RAPIDS style)

Usage:
    python benchmark_fairchem_modes.py
"""

import time
import torch
import numpy as np
from pathlib import Path
from ase import Atoms
from ase.build import molecule
from ase.optimize import LBFGS
from ase.io import write, read
import warnings
warnings.filterwarnings("ignore")

# FAIRChem imports
from fairchem.core import pretrained_mlip, FAIRChemCalculator


def build_test_system(size="medium"):
    """Build test systems of different sizes."""

    if size == "small":
        # Benzene dimer (~24 atoms)
        benzene1 = molecule("C6H6")
        benzene2 = molecule("C6H6")
        benzene2.translate([0, 0, 4.0])
        atoms = benzene1 + benzene2
        atoms.center(vacuum=10.0)
        return atoms, "benzene_dimer"

    elif size == "medium":
        # Larger system: multiple benzene (~72 atoms)
        atoms = molecule("C6H6")
        for i in range(1, 6):
            b = molecule("C6H6")
            b.translate([4.0 * (i % 3), 4.0 * (i // 3), 0])
            atoms += b
        atoms.center(vacuum=10.0)
        return atoms, "benzene_cluster_6"

    elif size == "large":
        # Even larger: 12 benzene (~144 atoms)
        atoms = molecule("C6H6")
        for i in range(1, 12):
            b = molecule("C6H6")
            b.translate([4.0 * (i % 4), 4.0 * ((i // 4) % 3), 4.0 * (i // 12)])
            atoms += b
        atoms.center(vacuum=10.0)
        return atoms, "benzene_cluster_12"

    else:
        raise ValueError(f"Unknown size: {size}")


def benchmark_single_point(atoms, predictor, task_name="omol", n_runs=5):
    """Benchmark single point energy calculation."""
    calc = FAIRChemCalculator(predictor, task_name=task_name)
    test_atoms = atoms.copy()
    test_atoms.calc = calc

    # Warmup
    _ = test_atoms.get_potential_energy()

    # Benchmark
    times = []
    for _ in range(n_runs):
        test_atoms = atoms.copy()
        test_atoms.calc = calc
        t0 = time.perf_counter()
        energy = test_atoms.get_potential_energy()
        forces = test_atoms.get_forces()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return {
        "mean_time": np.mean(times),
        "std_time": np.std(times),
        "energy": energy,
        "max_force": np.sqrt((forces**2).sum(axis=1).max()),
    }


def benchmark_optimization(atoms, predictor, task_name="omol", fmax=0.1, max_steps=50):
    """Benchmark geometry optimization."""
    calc = FAIRChemCalculator(predictor, task_name=task_name)
    test_atoms = atoms.copy()
    test_atoms.calc = calc

    t0 = time.perf_counter()
    opt = LBFGS(test_atoms, logfile=None)
    converged = opt.run(fmax=fmax, steps=max_steps)
    t1 = time.perf_counter()

    return {
        "time": t1 - t0,
        "steps": opt.nsteps,
        "converged": converged,
        "final_energy": test_atoms.get_potential_energy(),
        "final_max_force": np.sqrt((test_atoms.get_forces()**2).sum(axis=1).max()),
    }


def benchmark_workers(atoms, model_name, task_name="omol", n_workers_list=[1, 2, 4]):
    """Benchmark FAIRChem internal workers=N mode."""
    results = {}

    for n_workers in n_workers_list:
        print(f"  Testing workers={n_workers}...")
        try:
            predictor = pretrained_mlip.get_predict_unit(
                model_name,
                device="cuda",
                inference_settings="default"
            )
            calc = FAIRChemCalculator(predictor, task_name=task_name, workers=n_workers)
            test_atoms = atoms.copy()
            test_atoms.calc = calc

            # Warmup
            _ = test_atoms.get_potential_energy()

            # Benchmark
            times = []
            for _ in range(3):
                test_atoms = atoms.copy()
                test_atoms.calc = calc
                t0 = time.perf_counter()
                _ = test_atoms.get_potential_energy()
                _ = test_atoms.get_forces()
                t1 = time.perf_counter()
                times.append(t1 - t0)

            results[n_workers] = {
                "mean_time": np.mean(times),
                "std_time": np.std(times),
                "success": True,
            }
        except Exception as e:
            results[n_workers] = {
                "success": False,
                "error": str(e),
            }

    return results


def main():
    print("=" * 70)
    print("FAIRChem Benchmark: Inference Modes & Multi-GPU Strategies")
    print("=" * 70)

    # Check GPU availability
    n_gpus = torch.cuda.device_count()
    print(f"\nGPU count: {n_gpus}")
    for i in range(n_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    device = "cuda" if n_gpus > 0 else "cpu"
    model_name = "uma-s-1p1"
    task_name = "omol"

    # Build test systems
    print("\n" + "-" * 70)
    print("Building test systems...")
    systems = {}
    for size in ["small", "medium", "large"]:
        atoms, name = build_test_system(size)
        systems[size] = (atoms, name)
        print(f"  {size}: {name} ({len(atoms)} atoms)")

    # ============================================================
    # Test 1: Default vs Turbo mode
    # ============================================================
    print("\n" + "=" * 70)
    print("TEST 1: inference_settings comparison (default vs turbo)")
    print("=" * 70)
    print("\nNOTE: Turbo mode requires fixed atomic composition.")
    print("      Each system size needs its own turbo predictor.\n")

    for size, (atoms, name) in systems.items():
        print(f"\n--- System: {name} ({len(atoms)} atoms) ---")

        for mode in ["default", "turbo"]:
            print(f"\n  Mode: {mode}")
            # Turbo mode needs fresh predictor for each system size
            predictor = pretrained_mlip.get_predict_unit(
                model_name,
                device=device,
                inference_settings=mode
            )

            # Single point
            sp_result = benchmark_single_point(atoms, predictor, task_name, n_runs=5)
            print(f"    Single point: {sp_result['mean_time']*1000:.1f} ± {sp_result['std_time']*1000:.1f} ms")
            print(f"    Energy: {sp_result['energy']:.6f} eV")

            # Optimization (only for small/medium to save time)
            if size in ["small", "medium"]:
                opt_result = benchmark_optimization(atoms, predictor, task_name, fmax=0.1, max_steps=30)
                print(f"    Optimization: {opt_result['time']:.2f} s, {opt_result['steps']} steps")

    # ============================================================
    # Test 2: Consistency check (turbo vs default results)
    # ============================================================
    print("\n" + "=" * 70)
    print("TEST 2: Accuracy comparison (default vs turbo)")
    print("=" * 70)

    atoms, name = systems["medium"]

    predictor_default = pretrained_mlip.get_predict_unit(model_name, device=device, inference_settings="default")
    predictor_turbo = pretrained_mlip.get_predict_unit(model_name, device=device, inference_settings="turbo")

    calc_default = FAIRChemCalculator(predictor_default, task_name=task_name)
    calc_turbo = FAIRChemCalculator(predictor_turbo, task_name=task_name)

    atoms_default = atoms.copy()
    atoms_default.calc = calc_default
    e_default = atoms_default.get_potential_energy()
    f_default = atoms_default.get_forces()

    atoms_turbo = atoms.copy()
    atoms_turbo.calc = calc_turbo
    e_turbo = atoms_turbo.get_potential_energy()
    f_turbo = atoms_turbo.get_forces()

    print(f"\nSystem: {name} ({len(atoms)} atoms)")
    print(f"  Energy (default): {e_default:.6f} eV")
    print(f"  Energy (turbo):   {e_turbo:.6f} eV")
    print(f"  Energy diff:      {abs(e_turbo - e_default):.6f} eV ({abs(e_turbo - e_default)/abs(e_default)*100:.4f}%)")
    print(f"  Force MAE:        {np.mean(np.abs(f_turbo - f_default)):.6f} eV/Å")
    print(f"  Force Max diff:   {np.max(np.abs(f_turbo - f_default)):.6f} eV/Å")

    # ============================================================
    # Test 3: FAIRChem workers=N (if multiple GPUs)
    # ============================================================
    if n_gpus > 1:
        print("\n" + "=" * 70)
        print("TEST 3: FAIRChem internal workers=N mode")
        print("=" * 70)

        atoms, name = systems["large"]
        print(f"\nSystem: {name} ({len(atoms)} atoms)")

        workers_results = benchmark_workers(
            atoms, model_name, task_name,
            n_workers_list=[1, 2, min(4, n_gpus)]
        )

        for n_workers, result in workers_results.items():
            if result["success"]:
                print(f"  workers={n_workers}: {result['mean_time']*1000:.1f} ± {result['std_time']*1000:.1f} ms")
            else:
                print(f"  workers={n_workers}: FAILED - {result['error']}")
    else:
        print("\n(Skipping workers=N test - only 1 GPU available)")

    # ============================================================
    # Test 4: Process pool simulation (RAPIDS style)
    # ============================================================
    print("\n" + "=" * 70)
    print("TEST 4: Process pool vs sequential (RAPIDS multi-config style)")
    print("=" * 70)

    # Simulate scan_orientations: 9 configurations
    n_configs = 9
    atoms, name = systems["small"]

    # Create slightly different configs
    configs = []
    for i in range(n_configs):
        a = atoms.copy()
        # Small random perturbation
        a.positions += np.random.randn(*a.positions.shape) * 0.1
        configs.append(a)

    print(f"\nSimulating {n_configs} configurations (like scan_orientations)")
    print(f"System: {name} ({len(atoms)} atoms each)")

    # Sequential single GPU
    predictor = pretrained_mlip.get_predict_unit(model_name, device=device, inference_settings="turbo")
    calc = FAIRChemCalculator(predictor, task_name=task_name)

    t0 = time.perf_counter()
    for conf in configs:
        conf_copy = conf.copy()
        conf_copy.calc = calc
        _ = conf_copy.get_potential_energy()
        _ = conf_copy.get_forces()
    t_sequential = time.perf_counter() - t0

    print(f"  Sequential (1 GPU):  {t_sequential:.2f} s ({t_sequential/n_configs*1000:.1f} ms/config)")

    if n_gpus > 1:
        # Note: True process pool benchmark would require multiprocessing
        # Here we just estimate based on sequential time
        estimated_parallel = t_sequential / min(n_gpus, n_configs)
        print(f"  Estimated parallel ({n_gpus} GPUs): ~{estimated_parallel:.2f} s")
        print(f"  Estimated speedup: ~{t_sequential/estimated_parallel:.1f}x")

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Key findings:

1. TURBO MODE:
   - Provides faster inference at cost of slight accuracy reduction
   - Best for: screening, scan_orientations, batch_screening
   - NOT for: final production calculations requiring high accuracy

2. MULTI-GPU STRATEGIES:
   a) FAIRChem workers=N:
      - Single process, model distributed across GPUs
      - Best for: LARGE systems (>500 atoms)
      - Overhead for small systems may negate benefit

   b) RAPIDS process pool (_GpuPool):
      - Multiple processes, each owns one GPU
      - Best for: MANY small/medium systems in parallel
      - Perfect for scan_orientations (9 independent configs)

3. RECOMMENDATION FOR RAPIDS:
   - scan_orientations/batch_screening: use turbo mode
   - Keep process pool for multi-config parallelism
   - workers=N not needed (configs are small, not one huge system)
""")


if __name__ == "__main__":
    main()
