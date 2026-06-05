#!/usr/bin/env python3
"""
Smart FAIRChem Flow with intelligent features:
- Auto box sizing
- Smart continuation
- Minimal input mode
- Structure validation
"""

import json
import os
import sys
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from version import __version__

from ase import Atoms
from ase.io import read, write
from ase.optimize import LBFGS
from ase.constraints import FixAtoms

try:
    from fairchem.core import pretrained_mlip, FAIRChemCalculator
    FAIRCHEM_AVAILABLE = True
except ImportError:
    FAIRCHEM_AVAILABLE = False

# xTB for solvation free energy calculations
try:
    from xtb.ase.calculator import XTB
    XTB_AVAILABLE = True
except ImportError:
    XTB_AVAILABLE = False

from simulation_builder import SimulationBuilder
from molecule_downloader import MoleculeDownloader
from backbone_factory import get_backbone_calculator, normalize_backbone


class SmartFAIRChemFlow:
    """Enhanced FAIRChem workflow with intelligent features"""

    # Base directory for all relative paths (directory containing this script)
    BASE_DIR = Path(__file__).parent.resolve()

    def __init__(self, config_file: str = None, config_dict: dict = None, workspace: str = None):
        """Initialize with smart defaults

        Args:
            config_file: Path to JSON configuration file
            config_dict: Configuration dictionary (alternative to file)
            workspace: Optional workspace directory for output. If provided,
                       simulations are saved here instead of BASE_DIR.
        """
        self.workspace = Path(workspace).resolve() if workspace else None

        # Load configuration
        if config_file:
            with open(config_file, 'r') as f:
                self.config = json.load(f)
        elif config_dict:
            self.config = config_dict
        else:
            raise ValueError("Must provide either config_file or config_dict")

        # Apply smart defaults
        self.apply_smart_defaults()

        # Setup - use workspace if provided, otherwise BASE_DIR
        self.run_name = self.config.get("run_name", self.generate_run_name())
        if self.workspace:
            self.output_dir = self.workspace / "simulations" / self.run_name
        else:
            self.output_dir = self.BASE_DIR / self.config.get("output_dir", "simulations") / self.run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Model settings
        self.model_name = self.config.get("model_name", "uma-s-1p2")
        self.device = self.config.get("device", "cuda")

        # Backbone selection (default "uma" => unchanged behavior).
        # Precedence: config["backbone"] > RAPIDS_BACKBONE env var > "uma".
        # Supported: "uma", "mace-omol", "orb-omol" (see backbone_factory.py).
        backbone = self.config.get("backbone") or os.environ.get("RAPIDS_BACKBONE") or "uma"
        self.backbone = normalize_backbone(backbone)
        # For non-UMA backbones, model_name="uma-s-1p2" is not meaningful; let the
        # factory pick the backbone-appropriate default unless the user set one
        # explicitly in the config.
        self._explicit_model_name = "model_name" in self.config

        # Task selection - default to omol (has VV10 dispersion for molecular interactions)
        self.task_name = self.config.get("task_name", "omol")
        # Optional MACE head override (only used by the "mace-omol" backbone;
        # defaults to the omol head inside backbone_factory when None).
        self.head = self.config.get("head")

        # Charge and spin for omol task (stored in atoms.info)
        self.charge = self.config.get("charge", 0)
        self.spin = self.config.get("spin", 1)

        # Optimization settings
        self.fmax = self.config.get("fmax", 0.05)
        self.max_steps = self.config.get("max_steps", 100)
        self.max_continuations = self.config.get("max_continuations", 3)
        
        # Calculator cache - one per unique composition (for turbo mode compatibility)
        self._calculator_cache = {}  # {composition_key: calculator}
        self.calculator = None  # Legacy - kept for compatibility

        self.energies = {}
        self.warnings = []
        self.solvation_energies = {}  # xTB solvation free energies
        
    def apply_smart_defaults(self):
        """Apply intelligent defaults based on system"""
        
        # Essential parameters only
        if "probe" not in self.config:
            raise ValueError("Must specify 'probe' molecule")
            
        # Smart defaults for optional parameters
        if "probe_height" not in self.config:
            self.config["probe_height"] = 2.5  # Reasonable default
            
        if "target" in self.config and self.config["target"]:
            if "target_height" not in self.config:
                self.config["target_height"] = self.config["probe_height"] + 3.5
            if "probe_target_distance" not in self.config:
                self.config["probe_target_distance"] = 4.0
                
        # Auto box size will be calculated dynamically
        if "box_size" not in self.config:
            self.config["box_size"] = "auto"
            
    def generate_run_name(self):
        """Generate descriptive run name from config"""
        probe = self.config.get("probe", "mol")
        target = self.config.get("target", "")
        substrate = self.config.get("substrate", "vac")
        
        if target:
            return f"{probe}_{target}_{substrate}"
        return f"{probe}_{substrate}"
        
            
    def calculate_optimal_box_size(self, molecules: list) -> list:
        """Calculate optimal box size based on molecular extent.

        For charged systems, uses 15Å padding (vs 10Å for neutral) to reduce
        spurious periodic Coulomb interactions in the ML potential.
        """

        if not molecules:
            return [30.0, 30.0, 40.0]  # Default

        # Get combined extent of all molecules
        all_positions = []
        for mol in molecules:
            if isinstance(mol, Atoms):
                all_positions.extend(mol.positions)

        if not all_positions:
            return [30.0, 30.0, 40.0]

        positions = np.array(all_positions)
        mol_min = positions.min(axis=0)
        mol_max = positions.max(axis=0)
        extent = mol_max - mol_min

        # B9: Use larger padding for charged systems to reduce periodic Coulomb artifacts
        charge = self.config.get("charge", 0)
        if charge != 0:
            padding = 15.0  # Charged systems need more vacuum
            print(f"Charged system (charge={charge}): using 15Å padding")
        else:
            padding = 10.0  # Neutral systems

        box_size = extent + 2 * padding

        # Ensure minimum size (larger for charged systems)
        if charge != 0:
            min_size = np.array([35.0, 35.0, 45.0])
        else:
            min_size = np.array([25.0, 25.0, 35.0])
        box_size = np.maximum(box_size, min_size)

        print(f"Auto-calculated box size: [{box_size[0]:.1f}, {box_size[1]:.1f}, {box_size[2]:.1f}] Å")
        return box_size.tolist()
        
    def init_calculator(self):
        """Initialize FAIRChem calculator (legacy - creates default calculator)"""
        # Note: With turbo mode, we now use per-composition calculators
        # This is kept for compatibility but not actively used
        pass

    def _get_composition_key(self, atoms: Atoms) -> str:
        """Get a unique key for the atomic composition (sorted element counts)."""
        from collections import Counter
        counts = Counter(atoms.get_chemical_symbols())
        # Sort by element symbol for consistency
        return "_".join(f"{k}{v}" for k, v in sorted(counts.items()))

    def _get_calculator_for_composition(self, atoms: Atoms):
        """Get or create a turbo-mode calculator for the given composition.

        Turbo mode (merge_mole=True) locks the model to a specific composition.
        We cache calculators per composition to enable turbo speedup while
        supporting multiple different molecules in the same workflow.

        The backbone is selected via ``self.backbone`` (default "uma"). UMA is
        routed through the FAIRChem turbo per-composition path as before;
        alternative backbones ("mace-omol", "orb-omol") are built through the
        shared backbone factory and cached the same way.
        """
        # For the default UMA backbone, keep the original availability gate so a
        # missing fairchem install degrades gracefully (returns None -> skip).
        if self.backbone == "uma" and not FAIRCHEM_AVAILABLE:
            return None

        comp_key = self._get_composition_key(atoms)

        if comp_key not in self._calculator_cache:
            try:
                print(f"  Creating {self.backbone} calculator for composition: {comp_key}")
                # Only forward an explicit model_name; otherwise let the factory
                # choose the backbone-appropriate default checkpoint.
                model_name = self.model_name if self._explicit_model_name else None
                calc = get_backbone_calculator(
                    backbone=self.backbone,
                    device=self.device,
                    task_name=self.task_name,
                    model_name=model_name,
                    inference_settings="turbo",
                    head=self.head,
                )
                self._calculator_cache[comp_key] = calc
            except Exception as e:
                print(f"  Error creating calculator for {comp_key}: {e}")
                return None

        return self._calculator_cache[comp_key]
            
    def smart_optimize(self, atoms: Atoms, name: str,
                      fix_substrate: bool = False) -> Tuple[Atoms, dict]:
        """
        Smart optimization with automatic continuation

        Returns:
            (optimized_atoms, status_dict)
            status_dict contains:
                - status: "converged" | "limit_reached" | "diverged" | "error"
                - max_force: final max force in eV/Å
                - total_steps: total optimization steps taken
        """
        print(f"\n🔧 Optimizing {name}...")

        opt_atoms = atoms.copy()

        # Ensure charge and spin are set in atoms.info (required for omol task)
        opt_atoms.info["charge"] = self.charge
        opt_atoms.info["spin"] = self.spin

        # Get calculator for this specific composition (turbo mode compatible)
        calc = self._get_calculator_for_composition(opt_atoms)
        if calc is None:
            print(f"  Calculator not available, skipping {name}")
            return atoms, {"status": "error", "max_force": None, "total_steps": 0}

        opt_atoms.calc = calc

        # Get initial energy
        try:
            e_initial = opt_atoms.get_potential_energy()
            print(f"  Initial energy: {e_initial:.4f} eV")
        except Exception as e:
            print(f"  Error getting initial energy: {e}")
            return atoms, {"status": "error", "max_force": None, "total_steps": 0}

        status = "limit_reached"  # Default: ran out of steps
        max_force = None
        total_steps = 0
        continuation = 0

        while continuation <= self.max_continuations and status == "limit_reached":

            # Setup optimizer
            logfile = str(self.output_dir / f"{name}_opt_{continuation}.log")
            opt = LBFGS(opt_atoms, logfile=logfile)

            # Run optimization
            try:
                print(f"  Attempt {continuation+1}: Running up to {self.max_steps} steps...")
                opt.run(fmax=self.fmax, steps=self.max_steps)

                # Check convergence
                forces = opt_atoms.get_forces()
                max_force = float(np.max(np.abs(forces)))
                total_steps += opt.nsteps

                if max_force <= self.fmax:
                    status = "converged"
                    print(f"  ✓ Converged! Max force: {max_force:.4f} eV/Å")
                else:
                    print(f"  Max force: {max_force:.4f} eV/Å (target: {self.fmax})")

                    # Smart decision: continue or stop?
                    if max_force <= self.fmax * 2:  # Close to convergence
                        print(f"  📍 Close to convergence, continuing...")
                        continuation += 1
                    elif max_force > 0.5:  # Too far from convergence
                        print(f"  ⚠️ Forces too large ({max_force:.2f} eV/Å), structure may be unreasonable")
                        print(f"  Stopping optimization to avoid distorted structure")
                        self.warnings.append(f"{name}: Stopped due to large forces")
                        status = "diverged"
                        break
                    else:
                        print(f"  Continuing optimization...")
                        continuation += 1

            except Exception as e:
                print(f"  Optimization error: {e}")
                status = "error"
                break
                
        # Final energy
        try:
            e_final = opt_atoms.get_potential_energy()
            print(f"  Final energy: {e_final:.4f} eV")
            print(f"  Energy change: {e_final - e_initial:.4f} eV")
            print(f"  Total steps: {total_steps}")

            self.energies[name] = e_final

            # Save structure
            opt_path = self.output_dir / f"{name}_optimized.vasp"
            write(opt_path, opt_atoms, format='vasp')

            # Check for structural issues
            self.validate_structure(atoms, opt_atoms, name)

        except Exception as e:
            print(f"  Error in final processing: {e}")
            return atoms, {"status": "error", "max_force": max_force, "total_steps": total_steps}

        status_dict = {
            "status": status,
            "max_force": max_force,
            "total_steps": total_steps,
        }
        return opt_atoms, status_dict
        
    def validate_structure(self, initial: Atoms, final: Atoms, name: str):
        """Validate structural integrity"""
        
        # Check for large geometric changes
        if len(initial) == len(final):
            # Calculate RMSD for same atoms
            initial_pos = initial.positions
            final_pos = final.positions
            
            # Simple RMSD (could be improved with alignment)
            rmsd = np.sqrt(np.mean((final_pos - initial_pos)**2))
            
            if rmsd > 3.0:
                warning = f"⚠️ {name}: Large structural change (RMSD: {rmsd:.2f} Å)"
                print(f"  {warning}")
                self.warnings.append(warning)
                
        # Check for unreasonable bond lengths (simple check)
        distances = final.get_all_distances()
        min_dist = np.min(distances[distances > 0])
        
        if min_dist < 0.8:  # Å
            warning = f"⚠️ {name}: Unusually short distance detected ({min_dist:.2f} Å)"
            print(f"  {warning}")
            self.warnings.append(warning)
            
    def build_structures_with_auto_box(self) -> Dict[str, Atoms]:
        """Build structures with automatic box sizing"""
        
        print("\n" + "="*60)
        print("BUILDING STRUCTURES (with smart box sizing)")
        print("="*60)
        
        # If box_size is "auto", calculate it
        if self.config.get("box_size") == "auto":
            # First load molecules to determine size
            # Create a temporary builder just to access download_and_load_molecule
            # Pass workspace to ensure molecules are saved to correct location
            temp_builder = SimulationBuilder(
                config_dict=self.config,
                workspace=str(self.workspace) if self.workspace else None
            )

            molecules = []
            if self.config.get("probe"):
                mol = temp_builder.download_and_load_molecule(self.config["probe"])
                if mol:
                    molecules.append(mol)
            if self.config.get("target"):
                mol = temp_builder.download_and_load_molecule(self.config["target"])
                if mol:
                    molecules.append(mol)

            # Calculate optimal box
            self.config["box_size"] = self.calculate_optimal_box_size(molecules)

        # Now build with optimized settings
        # Pass workspace to ensure outputs are saved to correct location
        builder = SimulationBuilder(
            config_dict=self.config,
            workspace=str(self.workspace) if self.workspace else None
        )
        structures = builder.build_simulation()
        builder.save_structures(structures)
        
        return structures
        
    def run_workflow(self):
        """Execute smart workflow"""
        
        print("\n" + "="*60)
        print("SMART FAIRCHEM WORKFLOW")
        print("="*60)
        print(f"Configuration: {self.run_name}")
        print(f"Backbone: {self.backbone}")
        print(f"Model: {self.model_name}")
        print(f"Task: {self.task_name}")
        print(f"Smart continuation: Enabled (max {self.max_continuations} attempts)")
        
        # Build structures
        structures = self.build_structures_with_auto_box()

        if not FAIRCHEM_AVAILABLE:
            print("\n⚠️ FAIRChem not available")
            return
            
        # Optimize with smart continuation
        print("\n" + "="*60)
        print("SMART OPTIMIZATION")
        print("="*60)
        
        optimized = {}
        convergence_status = {}
        
        # Optimize each structure (including solvated systems)
        for struct_name in ["substrate_only", "probe_vacuum", "target_vacuum",
                           "probe_substrate", "probe_target_vacuum",
                           "probe_solvated", "target_solvated", "probe_target_solvated"]:
            if struct_name in structures:
                # Solvated systems don't have substrate to fix
                fix_sub = (struct_name in ["substrate_only", "probe_substrate"])
                opt_struct, converged = self.smart_optimize(
                    structures[struct_name],
                    struct_name,
                    fix_substrate=fix_sub
                )
                optimized[struct_name] = opt_struct
                convergence_status[struct_name] = converged
        
        # Special handling for three-component system
        # Use optimized probe_substrate as starting point
        if (self.config.get("probe") and self.config.get("target") and self.config.get("substrate") 
            and "probe_substrate" in optimized and "target_vacuum" in structures and "substrate_only" in structures):
            print("\n" + "="*60)
            print("Building three-component system from optimized probe_substrate")
            print("="*60)
            
            # Start with optimized probe_substrate
            system_three = optimized["probe_substrate"].copy()
            
            # Get the current positions
            substrate_atoms = len(structures["substrate_only"])
            substrate_top = system_three.positions[:substrate_atoms, 2].max()
            
            # Add target above the optimized probe position
            probe_positions = system_three.positions[substrate_atoms:]
            probe_center = probe_positions.mean(axis=0)
            
            # Load target and position it above probe
            target_height = self.config.get("target_height", probe_center[2] + 4.0)
            target_position = np.array([probe_center[0], probe_center[1], target_height])
            
            # Add target molecule
            target_mol = structures["target_vacuum"].copy()
            target_mol.positions = target_mol.positions - target_mol.positions.mean(axis=0) + target_position
            
            # Check for overlaps and adjust if needed
            min_dist = 2.5
            for i in range(len(system_three)):
                for j in range(len(target_mol)):
                    dist = np.linalg.norm(system_three.positions[i] - target_mol.positions[j])
                    if dist < min_dist:
                        # Move target up slightly
                        target_mol.positions[:, 2] += (min_dist - dist) + 0.5
                        break
            
            system_three.extend(target_mol)
            
            # Now optimize the three-component system
            # Fix both substrate and probe (only optimize target position)
            substrate_and_probe_atoms = len(system_three) - len(structures["target_vacuum"])
            fix_indices = list(range(substrate_and_probe_atoms))
            system_three.set_constraint(FixAtoms(indices=fix_indices))
            
            opt_three, converged_three = self.smart_optimize(
                system_three,
                "probe_target_substrate",
                fix_substrate=False  # Already set constraints above
            )
            optimized["probe_target_substrate"] = opt_three
            convergence_status["probe_target_substrate"] = converged_three
                
        # Calculate interaction energies
        self.calculate_and_report_interactions()

        # Calculate xTB solvation (vacuum mode only)
        self.calculate_all_solvation()

        # Final report
        self.generate_smart_report(convergence_status)
        
    def calculate_and_report_interactions(self):
        """Calculate and report interaction energies"""
        
        print("\n" + "="*60)
        print("INTERACTION ANALYSIS")
        print("="*60)
        
        results = {}
        
        # Probe-target in vacuum
        if all(k in self.energies for k in ["probe_target_vacuum", "probe_vacuum", "target_vacuum"]):
            e_int = self.energies["probe_target_vacuum"] - self.energies["probe_vacuum"] - self.energies["target_vacuum"]
            results["probe_target_vacuum"] = e_int
            print(f"Probe-Target interaction (vacuum): {e_int:.4f} eV")
            
        # Probe on substrate
        if all(k in self.energies for k in ["probe_substrate", "probe_vacuum", "substrate_only"]):
            e_ads = self.energies["probe_substrate"] - self.energies["probe_vacuum"] - self.energies["substrate_only"]
            results["probe_adsorption"] = e_ads
            print(f"Probe adsorption energy: {e_ads:.4f} eV")
            
            if e_ads > 0.1:
                print("  📍 Note: Positive adsorption energy may indicate:")
                print("     - Local minimum (not global)")
                print("     - Constrained geometry")
        
        # Three-component system interactions
        if all(k in self.energies for k in ["probe_target_substrate", "probe_substrate", "target_vacuum"]):
            # Target adsorption to probe on substrate
            e_target_ads = self.energies["probe_target_substrate"] - self.energies["probe_substrate"] - self.energies["target_vacuum"]
            results["target_adsorption_to_adsorbed_probe"] = e_target_ads
            print(f"Target adsorption to adsorbed probe: {e_target_ads:.4f} eV")

            # Total interaction energy (all three components)
            if all(k in self.energies for k in ["substrate_only", "probe_vacuum"]):
                e_total = self.energies["probe_target_substrate"] - self.energies["substrate_only"] - self.energies["probe_vacuum"] - self.energies["target_vacuum"]
                results["total_three_component_interaction"] = e_total
                print(f"Total three-component interaction: {e_total:.4f} eV")

            # Compare with vacuum interaction
            if "probe_target_vacuum" in results:
                e_substrate_effect = e_target_ads - results["probe_target_vacuum"]
                results["substrate_effect_on_interaction"] = e_substrate_effect
                print(f"Substrate effect on probe-target interaction: {e_substrate_effect:.4f} eV")
                if e_substrate_effect < 0:
                    print("  → Substrate enhances probe-target interaction")
                elif e_substrate_effect > 0:
                    print("  → Substrate weakens probe-target interaction")

        # Solvated probe-target interaction
        # Note: This is an approximation since water molecule counts may differ between systems
        if all(k in self.energies for k in ["probe_target_solvated", "probe_solvated", "target_solvated"]):
            # Raw energy difference (includes water molecule count differences)
            e_int_solv_raw = (self.energies["probe_target_solvated"]
                              - self.energies["probe_solvated"]
                              - self.energies["target_solvated"])
            results["probe_target_solvated_raw"] = e_int_solv_raw
            print(f"\nSolvated Probe-Target interaction (raw): {e_int_solv_raw:.4f} eV")
            print("  ⚠️  Note: Raw value includes water count differences between systems")

            # Compare with vacuum if available
            if "probe_target_vacuum" in results:
                print(f"  Vacuum interaction for reference: {results['probe_target_vacuum']:.4f} eV")
                solvation_effect = e_int_solv_raw - results["probe_target_vacuum"]
                results["solvation_effect"] = solvation_effect
                print(f"  Solvation effect (qualitative): {solvation_effect:.4f} eV")

        # Save results
        with open(self.output_dir / "interactions.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
            
    def generate_smart_report(self, convergence_status: dict):
        """Generate comprehensive report"""
        
        report_path = self.output_dir / "smart_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("SMART WORKFLOW REPORT\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"RAPIDS Version: {__version__}\n")
            f.write(f"System: {self.config.get('probe')} + {self.config.get('target', 'none')}\n")
            f.write(f"Substrate: {self.config.get('substrate', 'vacuum')}\n")
            f.write(f"Task: {self.task_name}\n")
            f.write(f"Box size: {self.config.get('box_size')}\n\n")
            
            f.write("CONVERGENCE STATUS:\n")
            f.write("-"*40 + "\n")
            for name, status_info in convergence_status.items():
                # Handle both old bool format and new dict format for backwards compatibility
                if isinstance(status_info, dict):
                    status_str = status_info.get("status", "unknown")
                    max_force = status_info.get("max_force")
                    if status_str == "converged":
                        status_text = "✓ Converged"
                    elif status_str == "limit_reached":
                        status_text = f"⚠️ Limit reached (max_force={max_force:.4f} eV/Å)" if max_force else "⚠️ Limit reached"
                    elif status_str == "diverged":
                        status_text = f"❌ Diverged (max_force={max_force:.2f} eV/Å)" if max_force else "❌ Diverged"
                    else:
                        status_text = f"❓ {status_str}"
                else:
                    # Old boolean format
                    status_text = "✓ Converged" if status_info else "⚠️ Not fully converged"
                f.write(f"{name}: {status_text}\n")
                
            if self.warnings:
                f.write("\nWARNINGS:\n")
                f.write("-"*40 + "\n")
                for warning in self.warnings:
                    f.write(f"{warning}\n")
                    
            f.write("\nENERGIES (eV):\n")
            f.write("-"*40 + "\n")
            for name, energy in self.energies.items():
                f.write(f"{name}: {energy:.4f}\n")

            # Solvation analysis (if available)
            if self.solvation_energies:
                f.write("\nSOLVATION ANALYSIS (xTB GFN2-xTB + ALPB water):\n")
                f.write("-"*40 + "\n")
                for name in ["probe_vacuum", "target_vacuum", "probe_target_vacuum"]:
                    if name in self.solvation_energies:
                        g_solv = self.solvation_energies[name]
                        kcal = g_solv * 23.061
                        f.write(f"G_solv({name}): {g_solv:.4f} eV ({kcal:.2f} kcal/mol)\n")

                if "delta_G_solvation" in self.solvation_energies:
                    delta_g = self.solvation_energies["delta_G_solvation"]
                    kcal = delta_g * 23.061
                    f.write(f"\nΔG_solvation (desolvation penalty): {delta_g:.4f} eV ({kcal:.2f} kcal/mol)\n")

                if "delta_G_solution" in self.solvation_energies:
                    # Calculate vacuum binding for comparison
                    if all(k in self.energies for k in ["probe_target_vacuum", "probe_vacuum", "target_vacuum"]):
                        e_vac = (self.energies["probe_target_vacuum"]
                                - self.energies["probe_vacuum"]
                                - self.energies["target_vacuum"])
                        e_sol = self.solvation_energies["delta_G_solution"]
                        f.write(f"\nBINDING ENERGY COMPARISON:\n")
                        f.write(f"  Vacuum:   {e_vac:.4f} eV ({e_vac * 23.061:.2f} kcal/mol)\n")
                        f.write(f"  Solution: {e_sol:.4f} eV ({e_sol * 23.061:.2f} kcal/mol)\n")

        print(f"\n📊 Report saved: {report_path}")

        # Save convergence status as JSON for downstream validation
        conv_path = self.output_dir / "convergence_status.json"
        with open(conv_path, 'w', encoding='utf-8') as f:
            json.dump(convergence_status, f, indent=2)

    def calculate_xtb_solvation(self, atoms: Atoms, name: str) -> Optional[float]:
        """Calculate solvation free energy using xTB GFN2-xTB + ALPB water

        Args:
            atoms: ASE Atoms object (optimized structure)
            name: Name for logging

        Returns:
            Solvation free energy in eV, or None if calculation fails
        """
        if not XTB_AVAILABLE:
            return None

        try:
            # Gas phase calculation (xTB requires pbc=False)
            atoms_gas = atoms.copy()
            atoms_gas.pbc = False  # xTB doesn't support periodic systems
            atoms_gas.calc = XTB(method="GFN2-xTB")
            e_gas = atoms_gas.get_potential_energy()

            # Solvated calculation (ALPB implicit water)
            atoms_solv = atoms.copy()
            atoms_solv.pbc = False  # xTB doesn't support periodic systems
            atoms_solv.calc = XTB(method="GFN2-xTB", solvent="water")
            e_solv = atoms_solv.get_potential_energy()

            # Solvation free energy
            g_solv = e_solv - e_gas
            return g_solv

        except Exception as e:
            print(f"  ⚠️ xTB solvation failed for {name}: {e}")
            return None

    def calculate_all_solvation(self):
        """Calculate solvation free energies for all vacuum structures"""

        # Only for vacuum mode
        substrate = self.config.get("substrate", "vacuum")
        if substrate != "vacuum":
            print("\n⚠️ xTB solvation only available for vacuum mode")
            return

        if not XTB_AVAILABLE:
            print("\n⚠️ xTB not available - skipping solvation calculations")
            return

        print("\n" + "="*60)
        print("SOLVATION ANALYSIS (xTB GFN2-xTB + ALPB water)")
        print("="*60)

        # Calculate for each optimized vacuum structure
        structures_to_calc = ["probe_vacuum", "target_vacuum", "probe_target_vacuum"]

        for name in structures_to_calc:
            struct_path = self.output_dir / f"{name}_optimized.vasp"
            if struct_path.exists():
                print(f"\n  Calculating G_solv for {name}...")
                atoms = read(str(struct_path))
                g_solv = self.calculate_xtb_solvation(atoms, name)
                if g_solv is not None:
                    self.solvation_energies[name] = g_solv
                    kcal = g_solv * 23.061
                    print(f"  G_solv({name}): {g_solv:.4f} eV ({kcal:.2f} kcal/mol)")

        # Calculate delta G solvation if all components available
        if all(k in self.solvation_energies for k in ["probe_vacuum", "target_vacuum", "probe_target_vacuum"]):
            delta_g = (self.solvation_energies["probe_target_vacuum"]
                      - self.solvation_energies["probe_vacuum"]
                      - self.solvation_energies["target_vacuum"])
            self.solvation_energies["delta_G_solvation"] = delta_g
            kcal = delta_g * 23.061
            print(f"\n  ΔG_solvation (desolvation penalty): {delta_g:.4f} eV ({kcal:.2f} kcal/mol)")

            # Calculate solution binding energy
            if "probe_target_vacuum" in self.energies and "probe_vacuum" in self.energies and "target_vacuum" in self.energies:
                e_bind_vacuum = (self.energies["probe_target_vacuum"]
                                - self.energies["probe_vacuum"]
                                - self.energies["target_vacuum"])
                e_bind_solution = e_bind_vacuum + delta_g
                self.solvation_energies["delta_G_solution"] = e_bind_solution
                print(f"\n  Vacuum binding:   {e_bind_vacuum:.4f} eV ({e_bind_vacuum * 23.061:.2f} kcal/mol)")
                print(f"  Solution binding: {e_bind_solution:.4f} eV ({e_bind_solution * 23.061:.2f} kcal/mol)")

        # Save solvation data
        if self.solvation_energies:
            solvation_path = self.output_dir / "solvation.json"
            with open(solvation_path, 'w', encoding='utf-8') as f:
                json.dump(self.solvation_energies, f, indent=2)
            print(f"\n  💾 Solvation data saved: {solvation_path}")


def main():
    """Enhanced command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description=f"RAPIDS Smart FAIRChem workflow v{__version__}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Minimal configuration example:
{
  "probe": "glucose",
  "target": "caffeine",
  "substrate": "Graphene"
}

Everything else is automatically determined!
        """
    )
    
    parser.add_argument("config", help="JSON configuration file")
    parser.add_argument("--version", action="version", version=f"RAPIDS v{__version__}")
    args = parser.parse_args()
    
    # Run smart workflow
    flow = SmartFAIRChemFlow(config_file=args.config)
    flow.run_workflow()
    

if __name__ == "__main__":
    main()