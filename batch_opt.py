#!/usr/bin/env python3
"""
Simple batch geometry optimization using FAIRChem UMA model
Searches for .vasp files and optimizes them one by one
Supports cell optimization for 2D materials
"""

import sys
import argparse
import numpy as np
from pathlib import Path
from ase.io import read, write
from ase.optimize import LBFGS
from ase.constraints import ExpCellFilter, StrainFilter
from fairchem.core import pretrained_mlip, FAIRChemCalculator

def optimize_structure(vasp_file, relax_cell=False, isotropic_2d=False):
    """
    Optimize a single structure
    
    Args:
        vasp_file: Path to VASP structure file
        relax_cell: Whether to relax cell parameters
        isotropic_2d: If True, maintain x/y ratio for 2D materials
    """
    print(f"Optimizing: {vasp_file}")
    if relax_cell:
        if isotropic_2d:
            print("  Mode: Cell + atoms optimization (isotropic 2D - maintaining x/y ratio)")
        else:
            print("  Mode: Cell + atoms optimization (all degrees of freedom)")
    else:
        print("  Mode: Atoms only optimization")
    
    # Read structure
    atoms = read(vasp_file)
    
    # Store initial cell parameters
    initial_cell = atoms.cell.copy()
    print(f"  Initial cell parameters:")
    print(f"    a = {np.linalg.norm(initial_cell[0]):.4f} Å")
    print(f"    b = {np.linalg.norm(initial_cell[1]):.4f} Å")
    print(f"    c = {np.linalg.norm(initial_cell[2]):.4f} Å")
    
    # Setup calculator
    predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cpu")
    calc = FAIRChemCalculator(predictor, task_name="omat")
    atoms.calc = calc
    
    # Get initial energy
    e_initial = atoms.get_potential_energy()
    forces_initial = atoms.get_forces()
    max_force_initial = np.max(np.abs(forces_initial))
    print(f"  Initial energy: {e_initial:.4f} eV")
    print(f"  Initial max force: {max_force_initial:.4f} eV/Å")
    
    # Setup optimization based on mode
    if relax_cell:
        if isotropic_2d:
            # For 2D materials: use StrainFilter for isotropic scaling
            # mask = [1, 1, 0, 0, 0, 0] allows independent x,y but we want same scaling
            # Using mask = [1, 0, 0, 0, 0, 0] and letting the filter handle isotropy
            # Or use ExpCellFilter with hydrostatic_strain=True for isotropic scaling
            mask = [True, True, False, False, False, False]  # Allow x, y scaling, no z, no shear
            atoms_filtered = ExpCellFilter(atoms, mask=mask, hydrostatic_strain=True)
            print("  Using ExpCellFilter with hydrostatic_strain=True for isotropic 2D scaling")
            print("  This enforces same scaling factor for x and y dimensions")
        else:
            # Full cell relaxation
            atoms_filtered = ExpCellFilter(atoms)
            print("  Using ExpCellFilter for full cell relaxation")
        
        opt = LBFGS(atoms_filtered, logfile="-")
    else:
        # Atoms only optimization
        opt = LBFGS(atoms, logfile="-")
    
    # Run optimization
    opt.run(fmax=0.05, steps=500)
    
    # Get final energy and forces
    e_final = atoms.get_potential_energy()
    forces_final = atoms.get_forces()
    max_force_final = np.max(np.abs(forces_final))
    
    # Report final cell parameters if cell was relaxed
    if relax_cell:
        final_cell = atoms.cell.copy()
        print(f"  Final cell parameters:")
        print(f"    a = {np.linalg.norm(final_cell[0]):.4f} Å (Δ = {np.linalg.norm(final_cell[0]) - np.linalg.norm(initial_cell[0]):.4f})")
        print(f"    b = {np.linalg.norm(final_cell[1]):.4f} Å (Δ = {np.linalg.norm(final_cell[1]) - np.linalg.norm(initial_cell[1]):.4f})")
        print(f"    c = {np.linalg.norm(final_cell[2]):.4f} Å (Δ = {np.linalg.norm(final_cell[2]) - np.linalg.norm(initial_cell[2]):.4f})")
        
        if isotropic_2d:
            # Check that x/y ratio is maintained
            initial_ratio = np.linalg.norm(initial_cell[0]) / np.linalg.norm(initial_cell[1])
            final_ratio = np.linalg.norm(final_cell[0]) / np.linalg.norm(final_cell[1])
            print(f"    Initial x/y ratio: {initial_ratio:.6f}")
            print(f"    Final x/y ratio: {final_ratio:.6f}")
            print(f"    Ratio change: {abs(final_ratio - initial_ratio):.6f}")
    
    print(f"  Final energy: {e_final:.4f} eV")
    print(f"  Final max force: {max_force_final:.4f} eV/Å")
    print(f"  Energy change: {e_final - e_initial:.4f} eV")
    
    # Save optimized structure
    suffix = "_opted"
    if relax_cell:
        suffix += "_cell"
        if isotropic_2d:
            suffix += "_iso2d"
    output_file = vasp_file.parent / f"{vasp_file.stem}{suffix}.vasp"
    write(output_file, atoms)
    print(f"Saved: {output_file}\n")
    
    # Return results for summary
    return {
        'file': vasp_file.name,
        'e_final': e_final,
        'max_force': max_force_final,
        'e_change': e_final - e_initial,
        'cell_relaxed': relax_cell,
        'isotropic_2d': isotropic_2d
    }

def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(
        description='Batch geometry optimization using FAIRChem UMA model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Optimize only atomic positions (default)
  python batch_opt.py /path/to/folder
  
  # Optimize atoms and cell (full relaxation)
  python batch_opt.py /path/to/folder --relax-cell
  
  # Optimize atoms and cell for 2D materials (maintain x/y ratio)
  python batch_opt.py /path/to/folder --relax-cell --iso-2d
        """
    )
    
    parser.add_argument('target_folder', type=str, 
                       help='Path to folder containing .vasp files')
    parser.add_argument('--relax-cell', action='store_true',
                       help='Also optimize cell parameters along with atomic positions')
    parser.add_argument('--iso-2d', action='store_true',
                       help='For 2D materials: maintain x/y ratio during cell optimization (requires --relax-cell)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.iso_2d and not args.relax_cell:
        print("Error: --iso-2d requires --relax-cell to be set")
        sys.exit(1)
    
    target_folder = Path(args.target_folder)
    if not target_folder.exists():
        print(f"Error: Folder {target_folder} does not exist")
        sys.exit(1)
    
    # Find all .vasp files
    vasp_files = list(target_folder.glob("*.vasp"))
    
    # Exclude already optimized files
    vasp_files = [f for f in vasp_files if "_opted" not in f.stem]
    
    if not vasp_files:
        print(f"No .vasp files found in {target_folder} (excluding already optimized files)")
        sys.exit(1)
    
    print(f"Found {len(vasp_files)} VASP files to optimize")
    if args.relax_cell:
        if args.iso_2d:
            print("Mode: Cell + atoms optimization (isotropic 2D - maintaining x/y ratio)")
        else:
            print("Mode: Cell + atoms optimization (all degrees of freedom)")
    else:
        print("Mode: Atoms only optimization")
    print()
    
    # Store results
    results = []
    
    # Optimize each file
    for vasp_file in vasp_files:
        try:
            result = optimize_structure(vasp_file, 
                                       relax_cell=args.relax_cell,
                                       isotropic_2d=args.iso_2d)
            if result:
                results.append(result)
        except Exception as e:
            print(f"Error optimizing {vasp_file}: {e}\n")
            continue
    
    # Save summary to file
    if results:
        summary_file = target_folder / "optimization_summary.txt"
        with open(summary_file, "w", encoding='utf-8') as f:
            f.write("File\tFinal_Energy(eV)\tMax_Force(eV/Å)\tEnergy_Change(eV)\tCell_Relaxed\tIsotropic_2D\n")
            for r in results:
                f.write(f"{r['file']}\t{r['e_final']:.6f}\t{r['max_force']:.6f}\t{r['e_change']:.6f}\t{r['cell_relaxed']}\t{r.get('isotropic_2d', False)}\n")
        print(f"\nResults saved to: {summary_file}")
    
    print("Batch optimization complete!")

if __name__ == "__main__":
    main()
