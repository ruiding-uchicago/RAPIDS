#!/usr/bin/env python3
"""
Test FAIRChem Ray-based InferenceBatcher for multi-GPU parallel inference.

Key findings:
- InferenceBatcher uses Ray Serve to deploy multiple replicas
- num_replicas controls how many GPU workers (each gets 1 GPU by default)
- batch_predict_unit is used with FAIRChemCalculator for actual calculations
"""

import time
import logging
logging.basicConfig(level=logging.WARNING)

import numpy as np
from ase.build import molecule
from ase.optimize import LBFGS

print("=" * 70)
print("FAIRChem Ray InferenceBatcher Test")
print("=" * 70)

# Build test system
benzene = molecule('C6H6')
benzene.center(vacuum=10.0)
benzene.info['charge'] = 0
benzene.info['spin'] = 1

n_structures = 20

# Create multiple test structures
structures = []
for i in range(n_structures):
    b = benzene.copy()
    b.positions += np.random.randn(*b.positions.shape) * 0.05
    b.info['charge'] = 0
    b.info['spin'] = 1
    structures.append(b)

print(f"\nTest: {n_structures} benzene structures")

# ============================================================
# Test 1: Sequential baseline (single GPU)
# ============================================================
print("\n" + "-" * 70)
print("Test 1: Sequential baseline (FAIRChemCalculator, turbo mode)")
print("-" * 70)

from fairchem.core import pretrained_mlip, FAIRChemCalculator

predictor = pretrained_mlip.get_predict_unit(
    'uma-s-1p1',
    device='cuda',  # New API: only 'cuda' or 'cpu', not 'cuda:0'
    inference_settings='turbo'
)
calc = FAIRChemCalculator(predictor, task_name='omol')

# Warmup
structures[0].calc = calc
_ = structures[0].get_potential_energy()

t0 = time.perf_counter()
energies_seq = []
for s in structures:
    s_copy = s.copy()
    s_copy.calc = calc
    e = s_copy.get_potential_energy()
    energies_seq.append(e)
t_sequential = time.perf_counter() - t0

print(f"  Time: {t_sequential:.2f} s ({t_sequential/n_structures*1000:.1f} ms/structure)")

# ============================================================
# Test 2: Ray InferenceBatcher (multi-GPU)
# ============================================================
print("\n" + "-" * 70)
print("Test 2: Ray InferenceBatcher (num_replicas=4, 4 GPUs)")
print("-" * 70)

try:
    from fairchem.core.calculate import InferenceBatcher

    # Create fresh predictor for Ray
    predictor_ray = pretrained_mlip.get_predict_unit(
        'uma-s-1p1',
        device='cuda',  # Ray will manage GPU assignment
        inference_settings='turbo'
    )

    print("  Starting Ray Serve (this may take a moment)...")
    t_init_start = time.perf_counter()

    batcher = InferenceBatcher(
        predict_unit=predictor_ray,
        max_batch_size=16,
        batch_wait_timeout_s=0.1,
        num_replicas=4,  # Use all 4 GPUs
        ray_actor_options={'num_gpus': 1}  # Each replica gets 1 GPU
    )

    t_init = time.perf_counter() - t_init_start
    print(f"  Ray initialization: {t_init:.1f} s")

    # Get the batch predict unit for use with calculator
    batch_pred_unit = batcher.batch_predict_unit

    # Create calculator using the batched predictor
    calc_ray = FAIRChemCalculator(batch_pred_unit, task_name='omol')

    # Warmup
    structures[0].calc = calc_ray
    _ = structures[0].get_potential_energy()

    # Benchmark
    t0 = time.perf_counter()
    energies_ray = []
    for s in structures:
        s_copy = s.copy()
        s_copy.calc = calc_ray
        e = s_copy.get_potential_energy()
        energies_ray.append(e)
    t_ray = time.perf_counter() - t0

    print(f"  Inference time: {t_ray:.2f} s ({t_ray/n_structures*1000:.1f} ms/structure)")
    print(f"  Speedup vs sequential: {t_sequential/t_ray:.2f}x")

    # Verify results match
    max_diff = max(abs(e1 - e2) for e1, e2 in zip(energies_seq, energies_ray))
    print(f"  Max energy difference: {max_diff:.6f} eV")

    # Cleanup
    batcher.shutdown()

except Exception as e:
    import traceback
    print(f"  Failed: {e}")
    traceback.print_exc()

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
Ray InferenceBatcher:
- Uses Ray Serve to deploy multiple model replicas
- Each replica gets 1 GPU (configurable via ray_actor_options)
- Automatic request batching and load balancing
- Best for: high-throughput serving, many concurrent requests

RAPIDS use case consideration:
- scan_orientations: 9 configs, each needs geometry optimization
- Optimization = many sequential energy+force calls per config
- Ray batching helps if configs can share the same calculator
- But RAPIDS already uses process pool for config-level parallelism

Recommendation:
- For RAPIDS: Keep process pool (_GpuPool) for config parallelism
- Ray InferenceBatcher: better for serving/API scenarios
- Turbo mode: immediate 2x speedup with minimal code change
""")
