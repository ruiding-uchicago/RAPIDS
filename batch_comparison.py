#!/usr/bin/env python3
"""
Batch Comparison Tool for Multiple Probe Screening
Compare adsorption energies of multiple probe molecules with the same target
"""

import json
import os
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

from smart_fairchem_flow import SmartFAIRChemFlow
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend


class BatchComparison:
    """Compare multiple probe molecules against the same target"""

    # Base directory for all relative paths (directory containing this script)
    BASE_DIR = Path(__file__).parent.resolve()

    def __init__(self, config_file: str = None, config_dict: dict = None, workspace: str = None):
        """Initialize batch comparison

        Args:
            config_file: Path to JSON configuration file
            config_dict: Configuration dictionary (alternative to file)
            workspace: Optional workspace directory. If provided, outputs are saved here.
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

        # Validate configuration
        if "probes" not in self.config:
            # Backward compatibility: convert single probe to list
            if "probe" in self.config:
                self.config["probes"] = [self.config["probe"]]
                print("Note: Converting single probe to batch format")
            else:
                raise ValueError("Must specify 'probes' (list) or 'probe' (single)")

        self.probes = self.config["probes"]

        # Support multiple targets (optional)
        self.targets = []
        if "targets" in self.config:
            raw_targets = self.config["targets"]
            if isinstance(raw_targets, list):
                self.targets = [t for t in raw_targets if t]
            elif raw_targets:
                self.targets = [raw_targets]
        elif "target" in self.config:
            target_value = self.config.get("target")
            if target_value:
                self.targets = [target_value]

        # Backward compatibility helpers
        self.target = self.targets[0] if self.targets else None

        # Support both single substrate and multiple substrates
        if "substrates" in self.config:
            self.substrates = self.config["substrates"]
        elif "substrate" in self.config:
            self.substrates = [self.config["substrate"]]
        else:
            self.substrates = ["vacuum"]

        # Output directory for comparison results
        # Use workspace if provided, otherwise BASE_DIR
        if self.workspace:
            self.output_dir = self.workspace / "simulations"
        else:
            self.output_dir = self.BASE_DIR / self.config.get("output_dir", "simulations")
        self.comparison_name = self.config.get("comparison_name", self.generate_comparison_name())
        self.comparison_dir = self.output_dir / f"comparison_{self.comparison_name}"
        self.comparison_dir.mkdir(parents=True, exist_ok=True)
        
        # Store results
        self.results = {}
        self.energies = {}
        self.rankings = {}
        
    def generate_comparison_name(self) -> str:
        """Generate descriptive name for comparison run"""
        if len(self.targets) > 1:
            target_str = f"{len(self.targets)}targets"
        else:
            target_str = self.target if self.target else "nosol"
        if len(self.substrates) == 1:
            substrate_str = self.substrates[0].lower().replace(" ", "")
        else:
            substrate_str = f"{len(self.substrates)}substrates"
        n_probes = len(self.probes)
        return f"{target_str}_{substrate_str}_{n_probes}probes"
        
    def run_single_probe(self, probe: str, substrate: str = None, target: str = None) -> Dict:
        """Run simulation for a single probe on a specific substrate"""
        
        if substrate is None:
            substrate = self.substrates[0] if self.substrates else "vacuum"
        
        print("\n" + "="*60)
        if target:
            print(f"PROCESSING: {probe} with target {target} on {substrate}")
        else:
            print(f"PROCESSING: {probe} (no target) on {substrate}")
        print("="*60)
        
        # Create configuration for this probe-substrate combination
        single_config = self.config.copy()
        single_config["probe"] = probe
        single_config["substrate"] = substrate
        single_config.pop("probes", None)  # Remove probes list
        single_config.pop("substrates", None)  # Remove substrates list
        single_config.pop("targets", None)

        if target:
            single_config["target"] = target
            single_config["run_name"] = f"{probe}_{target}_{substrate}"
        else:
            single_config.pop("target", None)
            single_config["run_name"] = f"{probe}_{substrate}"
        
        # Run FAIRChem workflow
        try:
            flow = SmartFAIRChemFlow(
                config_dict=single_config,
                workspace=str(self.workspace) if self.workspace else None
            )
            flow.run_workflow()
            
            # Extract results
            result = {
                "probe": probe,
                "target": target,
                "energies": flow.energies.copy(),
                "warnings": flow.warnings.copy(),
                "converged": True
            }
            
            # Calculate interaction energies
            if target:
                # Probe-Target interaction in vacuum
                if all(k in flow.energies for k in ["probe_target_vacuum", "probe_vacuum", "target_vacuum"]):
                    result["probe_target_interaction_vacuum"] = (
                        flow.energies["probe_target_vacuum"] - 
                        flow.energies["probe_vacuum"] - 
                        flow.energies["target_vacuum"]
                    )
                    # Keep backward compatibility
                    result["probe_target_interaction"] = result["probe_target_interaction_vacuum"]
                
                # Three-component system: Target adsorption to adsorbed probe
                if substrate != "vacuum":
                    if all(k in flow.energies for k in ["probe_target_substrate", "probe_substrate", "target_vacuum"]):
                        result["target_adsorption_on_substrate"] = (
                            flow.energies["probe_target_substrate"] -
                            flow.energies["probe_substrate"] -
                            flow.energies["target_vacuum"]
                        )
                        # This is the most relevant interaction for substrate systems
                        result["probe_target_interaction"] = result["target_adsorption_on_substrate"]

                        # Calculate substrate effect
                        if "probe_target_interaction_vacuum" in result:
                            result["substrate_effect"] = (
                                result["target_adsorption_on_substrate"] -
                                result["probe_target_interaction_vacuum"]
                            )
                    
            # Probe adsorption on substrate
            if substrate != "vacuum":
                if all(k in flow.energies for k in ["probe_substrate", "probe_vacuum", "substrate_only"]):
                    result["probe_adsorption"] = (
                        flow.energies["probe_substrate"] - 
                        flow.energies["probe_vacuum"] - 
                        flow.energies["substrate_only"]
                    )
                    
            return result
            
        except Exception as e:
            print(f"Error processing {probe}: {e}")
            return {
                "probe": probe,
                "target": target,
                "energies": {},
                "warnings": [f"Failed: {str(e)}"],
                "converged": False
            }
            
    def run_comparison(self):
        """Run comparison for all probes on all substrates"""
        
        target_desc = ', '.join(self.targets) if self.targets else 'None (self-interaction)'

        print("\n" + "="*60)
        print("BATCH COMPARISON WORKFLOW")
        print("="*60)
        print(f"Targets: {target_desc}")
        print(f"Substrates: {', '.join(self.substrates)}")
        print(f"Number of probes: {len(self.probes)}")
        print(f"Probes: {', '.join(self.probes)}")

        n_targets = len(self.targets) if self.targets else 1
        print(f"Total simulations: {len(self.probes) * len(self.substrates) * n_targets}")
        
        # Process each probe-substrate combination
        sim_count = 0
        total_sims = len(self.probes) * len(self.substrates) * n_targets

        for substrate in self.substrates:
            print(f"\n{'='*60}")
            print(f"SUBSTRATE: {substrate}")
            print(f"{'='*60}")

            target_loop = self.targets if self.targets else [None]

            for target in target_loop:
                target_label = target if target else "None"
                print(f"\n--- TARGET: {target_label} ---")

                for probe in self.probes:
                    sim_count += 1
                    print(f"\n[{sim_count}/{total_sims}] Processing {probe} on {substrate} (target: {target_label})...")
                    result = self.run_single_probe(probe, substrate, target)

                    # Store results with substrate and target as keys
                    if substrate not in self.results:
                        self.results[substrate] = {}
                    target_key = target if target else "__no_target__"
                    if target_key not in self.results[substrate]:
                        self.results[substrate][target_key] = {}
                    self.results[substrate][target_key][probe] = result
            
        # Analyze and rank results
        self.analyze_results()
        
        # Generate reports
        self.generate_comparison_report()
        self.generate_ranking_plot()
        
        print("\n" + "="*60)
        print("COMPARISON COMPLETE")
        print("="*60)
        print(f"Results saved to: {self.comparison_dir}")
        
    def analyze_results(self):
        """Analyze and rank probe molecules for each substrate"""
        self.rankings = {}

        for substrate, target_map in self.results.items():
            if substrate not in self.rankings:
                self.rankings[substrate] = {}

            if not target_map:
                continue

            # Backward compatibility: handle flat probe->result mapping
            first_value = next(iter(target_map.values())) if isinstance(target_map, dict) and target_map else None
            if first_value and isinstance(first_value, dict) and "probe" in first_value:
                wrapped_map = {"__no_target__": target_map}
            else:
                wrapped_map = target_map

            for target_key, probe_results in wrapped_map.items():
                interaction_energies = {}
                adsorption_energies = {}

                for probe, result in probe_results.items():
                    if "probe_target_interaction" in result:
                        interaction_energies[probe] = result["probe_target_interaction"]
                    elif "probe_target_interaction_vacuum" in result:
                        interaction_energies[probe] = result["probe_target_interaction_vacuum"]

                    if "probe_adsorption" in result:
                        adsorption_energies[probe] = result["probe_adsorption"]

                rankings_entry = {}
                if interaction_energies:
                    rankings_entry["interaction"] = sorted(interaction_energies.items(), key=lambda x: x[1])
                if adsorption_energies:
                    rankings_entry["adsorption"] = sorted(adsorption_energies.items(), key=lambda x: x[1])

                if rankings_entry:
                    self.rankings[substrate][target_key] = rankings_entry
    
    def _write_result_details(self, f, result):
        """Helper method to write result details to file"""
        if result["converged"]:
            f.write("Status: ✓ Converged\n")
        else:
            f.write("Status: ⚠️ Not converged\n")
            
        if result.get("target"):
            f.write(f"Target: {result['target']}\n")
        else:
            f.write("Target: None (self-interaction)\n")

        if "probe_target_interaction_vacuum" in result:
            f.write(f"Probe-Target interaction (vacuum): {result['probe_target_interaction_vacuum']:.4f} eV\n")
            
        if "target_adsorption_on_substrate" in result:
            f.write(f"Target adsorption to adsorbed probe: {result['target_adsorption_on_substrate']:.4f} eV\n")

        if "substrate_effect" in result:
            effect = result["substrate_effect"]
            f.write(f"Substrate effect on interaction: {effect:.4f} eV ")
            if effect < 0:
                f.write("(enhances interaction)\n")
            else:
                f.write("(weakens interaction)\n")
                
        if "probe_adsorption" in result:
            f.write(f"Probe adsorption energy: {result['probe_adsorption']:.4f} eV\n")
            
        if result["warnings"]:
            f.write("Warnings:\n")
            for warning in result["warnings"]:
                f.write(f"  - {warning}\n")
            
    def generate_comparison_report(self):
        """Generate detailed comparison report"""
        
        report_path = self.comparison_dir / "comparison_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("BATCH COMPARISON REPORT\n")
            f.write("="*70 + "\n\n")
            
            target_desc = ', '.join(self.targets) if self.targets else 'None (self-interaction)'

            f.write(f"Targets: {target_desc}\n")
            f.write(f"Substrates: {', '.join(self.substrates)}\n")
            f.write(f"Number of probes tested: {len(self.probes)}\n")
            f.write(f"Probes: {', '.join(self.probes)}\n")

            n_targets = len(self.targets) if self.targets else 1
            f.write(f"Total simulations: {len(self.probes) * len(self.substrates) * n_targets}\n\n")

            def format_target_label(key: str) -> str:
                if key in (None, "__no_target__", "null"):
                    return "None (self-interaction)"
                return key

            # Rankings organized by substrate and target
            for substrate, rankings_per_target in self.rankings.items():
                if not rankings_per_target:
                    continue

                f.write(f"{'='*70}\n")
                f.write(f"SUBSTRATE: {substrate}\n")
                f.write(f"{'='*70}\n")

                for target_key, metrics in rankings_per_target.items():
                    target_label = format_target_label(target_key)
                    f.write(f"\nTARGET: {target_label}\n")
                    f.write("-"*70 + "\n")

                    if "interaction" in metrics and metrics["interaction"]:
                        f.write("Probe-Target Interaction Energies (eV)\n")
                        f.write(f"{'Rank':<6} {'Probe':<20} {'Energy':>12} {'Δ vs Best':>14}\n")
                        f.write("-"*60 + "\n")
                        best_energy = metrics["interaction"][0][1]
                        for i, (probe, energy) in enumerate(metrics["interaction"], 1):
                            relative = energy - best_energy
                            status = " ✓" if i == 1 else ""
                            f.write(f"{i:<6} {probe:<20} {energy:>12.4f} {relative:>14.4f}{status}\n")

                    if "adsorption" in metrics and metrics["adsorption"]:
                        f.write("\nAdsorption Energies (eV)\n")
                        f.write(f"{'Rank':<6} {'Probe':<20} {'Energy':>12} {'Δ vs Best':>14}\n")
                        f.write("-"*60 + "\n")
                        best_energy = metrics["adsorption"][0][1]
                        for i, (probe, energy) in enumerate(metrics["adsorption"], 1):
                            relative = energy - best_energy
                            status = " ✓" if i == 1 else ""
                            f.write(f"{i:<6} {probe:<20} {energy:>12.4f} {relative:>14.4f}{status}\n")

                    f.write("\n")

            # Detailed results across all combinations
            f.write("="*70 + "\n")
            f.write("DETAILED RESULTS\n")
            f.write("="*70 + "\n")

            for substrate, stored_results in self.results.items():
                if not stored_results:
                    continue

                first_value = next(iter(stored_results.values())) if stored_results else None
                if first_value and isinstance(first_value, dict) and "probe" in first_value:
                    target_wrapped = {"__no_target__": stored_results}
                else:
                    target_wrapped = stored_results

                for target_key, probe_map in target_wrapped.items():
                    target_label = format_target_label(target_key)
                    f.write(f"\n--- Substrate: {substrate} | Target: {target_label} ---\n")

                    for probe, result in probe_map.items():
                        f.write(f"\n>>> {probe}\n")
                        f.write("-"*40 + "\n")
                        self._write_result_details(f, result)

            # Summary statistics per combination
            f.write("\n" + "="*70 + "\n")
            f.write("SUMMARY STATISTICS\n")
            f.write("-"*70 + "\n")

            for substrate, rankings_per_target in self.rankings.items():
                for target_key, metrics in rankings_per_target.items():
                    if ("interaction" in metrics and len(metrics["interaction"]) > 1) or (
                        "adsorption" in metrics and len(metrics["adsorption"]) > 1):
                        target_label = format_target_label(target_key)
                        f.write(f"\nSubstrate: {substrate} | Target: {target_label}\n")

                    if "interaction" in metrics and len(metrics["interaction"]) > 1:
                        energies = [e for _, e in metrics["interaction"]]
                        f.write(f"  Interaction energy range: {min(energies):.4f} to {max(energies):.4f} eV\n")
                        f.write(f"  Mean interaction energy: {np.mean(energies):.4f} eV\n")
                        f.write(f"  Std deviation: {np.std(energies):.4f} eV\n")

                    if "adsorption" in metrics and len(metrics["adsorption"]) > 1:
                        energies = [e for _, e in metrics["adsorption"]]
                        f.write(f"  Adsorption energy range: {min(energies):.4f} to {max(energies):.4f} eV\n")
                        f.write(f"  Mean adsorption energy: {np.mean(energies):.4f} eV\n")
                        f.write(f"  Std deviation: {np.std(energies):.4f} eV\n")
                
        print(f"Report saved: {report_path}")
        
        # Also save as JSON for programmatic access
        json_path = self.comparison_dir / "comparison_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                "config": self.config,
                "targets": self.targets,
                "results": self.results,
                "rankings": self.rankings
            }, f, indent=2)
            
        print(f"JSON results saved: {json_path}")
        
    def generate_ranking_plot(self):
        """Generate visualization of energy rankings"""
        
        # Skip if no rankings
        if not self.rankings:
            return

        def format_target_label(key: str) -> str:
            if key in (None, "__no_target__", "null"):
                return "None"
            return key

        combos = []
        for substrate, target_rankings in self.rankings.items():
            if not target_rankings:
                continue
            for target_key, metrics in target_rankings.items():
                combos.append((substrate, target_key, metrics))

        if not combos:
            return

        default_plot_saved = False

        for substrate, target_key, metrics in combos:
            plots_to_create = []
            if "interaction" in metrics and metrics["interaction"]:
                plots_to_create.append(("Interaction Energy (eV)", metrics["interaction"], 'green', 'red'))
            if "adsorption" in metrics and metrics["adsorption"]:
                plots_to_create.append(("Adsorption Energy (eV)", metrics["adsorption"], 'blue', 'orange'))

            if not plots_to_create:
                continue

            fig, axes = plt.subplots(1, len(plots_to_create), figsize=(6*len(plots_to_create), 5))
            if len(plots_to_create) == 1:
                axes = [axes]

            for ax, (title, data, neg_color, pos_color) in zip(axes, plots_to_create):
                probes = [p for p, _ in data]
                energies = [e for _, e in data]
                colors = [neg_color if e < 0 else pos_color for e in energies]
                bars = ax.bar(range(len(probes)), energies, color=colors)

                ax.set_xticks(range(len(probes)))
                ax.set_xticklabels(probes, rotation=45, ha='right')
                ax.set_ylabel(title)
                target_label = format_target_label(target_key)
                ax.set_title(f"{title.split(' (')[0]}\n{substrate} | Target: {target_label}")
                ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                ax.grid(True, alpha=0.3)

                for bar, energy in zip(bars, energies):
                    height = bar.get_height()
                    vertical_align = 'bottom' if height >= 0 else 'top'
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                            f'{energy:.3f}', ha='center', va=vertical_align)

            plt.tight_layout()

            safe_substrate = substrate.replace(' ', '_')
            safe_target = format_target_label(target_key).replace(' ', '_').replace('/', '_')
            plot_filename = f"energy_comparison_{safe_substrate}_{safe_target}.png"
            plot_path = self.comparison_dir / plot_filename
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')

            if not default_plot_saved:
                default_path = self.comparison_dir / "energy_comparison.png"
                plt.savefig(default_path, dpi=150, bbox_inches='tight')
                default_plot_saved = True

            plt.close(fig)
            print(f"Plot saved: {plot_path}")


def main():
    """Command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Batch comparison of multiple probe molecules",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example configuration for batch comparison:
{
  "probes": ["glucose", "fructose", "sucrose", "maltose"],
  "target": "protein_binding_site",
  "substrate": "Graphene",
  "comparison_name": "sugar_screening"
}

This will test each probe molecule and rank them by adsorption affinity.
        """
    )
    
    parser.add_argument("config", help="JSON configuration file")
    parser.add_argument("--plot", action="store_true", 
                       help="Generate comparison plots")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)
        
    # Run batch comparison
    batch = BatchComparison(config_file=args.config)
    batch.run_comparison()


if __name__ == "__main__":
    main()
