#!/usr/bin/env python3
"""
Intelligent Atomic Simulation Environment Builder
Automatically creates simulation environments from JSON configuration
"""

import json
import os
import sys
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ASE imports
from ase import Atoms
from ase.io import read, write
from ase.build import molecule as ase_molecule
from ase.geometry import get_distances
from ase.constraints import FixAtoms
from ase.visualize import view
from scipy.spatial.transform import Rotation

# Import our molecule downloader
from molecule_downloader import MoleculeDownloader


# Van der Waals radii in Angstrom
# Source: Mantina et al. (2009) J. Phys. Chem. A, 113, 5806-5812
# https://pubs.acs.org/doi/10.1021/jp8111556
VDW_RADII = {
    # Main group elements (Mantina 2009 - validated)
    'H': 1.10, 'He': 1.40,
    'Li': 1.81, 'Be': 1.53, 'B': 1.92, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'F': 1.47, 'Ne': 1.54,
    'Na': 2.27, 'Mg': 1.73, 'Al': 1.84, 'Si': 2.10, 'P': 1.80, 'S': 1.80, 'Cl': 1.75, 'Ar': 1.88,
    'K': 2.75, 'Ca': 2.31, 'Ga': 1.87, 'Ge': 2.11, 'As': 1.85, 'Se': 1.90, 'Br': 1.83, 'Kr': 2.02,
    'Rb': 3.03, 'Sr': 2.49, 'In': 1.93, 'Sn': 2.17, 'Sb': 2.06, 'Te': 2.06, 'I': 1.98, 'Xe': 2.16,
    'Cs': 3.43, 'Ba': 2.68, 'Tl': 1.96, 'Pb': 2.02, 'Bi': 2.07, 'Po': 1.97, 'At': 2.02, 'Rn': 2.20,
    # Transition metals (approximate, from various sources)
    'Ti': 2.00, 'V': 1.97, 'Cr': 1.97, 'Mn': 1.97, 'Fe': 1.96, 'Co': 1.95, 'Ni': 1.94,
    'Cu': 1.96, 'Zn': 2.01, 'Zr': 2.16, 'Mo': 2.09, 'Ru': 2.07, 'Rh': 2.02, 'Pd': 2.05,
    'Ag': 2.03, 'Cd': 2.18, 'W': 2.10, 'Pt': 2.13, 'Au': 2.14, 'Hg': 2.23
}


class SimulationBuilder:
    """Build atomic simulation environments from configuration"""

    # Base directory for all relative paths (directory containing this script)
    BASE_DIR = Path(__file__).parent.resolve()

    def __init__(self, config_file: str = None, config_dict: dict = None, workspace: str = None):
        """
        Initialize builder with configuration

        Args:
            config_file: Path to JSON configuration file
            config_dict: Configuration dictionary (alternative to file)
            workspace: Optional workspace directory for output. If provided,
                       simulations and molecules are saved here instead of BASE_DIR.
                       Substrates and rare_molecules are always read from BASE_DIR.
        """
        if config_file:
            with open(config_file, 'r') as f:
                self.config = json.load(f)
        elif config_dict:
            self.config = config_dict
        else:
            raise ValueError("Must provide either config_file or config_dict")

        # Determine workspace (for outputs) vs BASE_DIR (for shared resources)
        if workspace:
            workspace_path = Path(workspace).resolve()
            self.molecules_dir = workspace_path / "molecules"
            self.output_dir = workspace_path / "simulations"
        else:
            self.molecules_dir = self.BASE_DIR / "molecules"
            self.output_dir = self.BASE_DIR / self.config.get("output_dir", "simulations")

        # Shared resources always from BASE_DIR (read-only)
        self.substrate_dir = self.BASE_DIR / "substrate"
        self.rare_molecules_dir = self.BASE_DIR / "rare_molecules"

        # Ensure output directories exist
        self.molecules_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize molecule downloader with workspace molecules dir
        self.downloader = MoleculeDownloader(
            output_dir=str(self.molecules_dir),
            rare_dir=str(self.rare_molecules_dir)
        )

        # Store loaded structures
        self.substrate = None
        self.probe = None
        self.target = None

        # Charge and spin for omol task (stored in atoms.info)
        self.charge = self.config.get("charge", 0)
        self.spin = self.config.get("spin", 1)
        
    def load_substrate(self) -> Optional[Atoms]:
        """Load substrate from CONTCAR file"""
        substrate_name = self.config.get("substrate", None)
        
        if not substrate_name or substrate_name.lower() == "vacuum":
            print("No substrate specified, using vacuum box")
            return None
            
        substrate_path = self.substrate_dir / substrate_name / "CONTCAR"
        
        if not substrate_path.exists():
            print(f"Warning: Substrate file not found: {substrate_path}")
            print("Available substrates:", [d.name for d in self.substrate_dir.iterdir() if d.is_dir()])
            return None
            
        try:
            substrate = read(substrate_path)
            print(f"Loaded substrate: {substrate_name} ({len(substrate)} atoms)")
            
            # Ensure proper cell and PBC
            if substrate.cell.rank < 3:
                print("Warning: Substrate cell not properly defined")
                
            # Fix bottom layers (optional)
            if self.config.get("fix_substrate_layers", 1) > 0:
                z_positions = substrate.positions[:, 2]
                z_min = z_positions.min()
                z_threshold = z_min + self.config.get("fix_substrate_layers", 1) * 2.5  # Approximate layer spacing
                mask = z_positions < z_threshold
                substrate.set_constraint(FixAtoms(mask=mask))
                print(f"Fixed {mask.sum()} substrate atoms")
                
            return substrate
            
        except Exception as e:
            print(f"Error loading substrate: {e}")
            return None
            
    def download_and_load_molecule(self, name: str) -> Optional[Atoms]:
        """Download molecule if needed and load it"""
        # Sanitize filename
        filename = name.replace(' ', '_').replace('/', '_')

        # Check multiple locations for pre-existing molecules
        # 1. Workspace molecules directory
        # 2. Global shared molecules directory (BASE_DIR/molecules)
        # 3. Rare molecules directory (BASE_DIR/rare_molecules)
        search_paths = [
            self.molecules_dir / f"{filename}.sdf",
            self.BASE_DIR / "molecules" / f"{filename}.sdf",
            self.BASE_DIR / "rare_molecules" / f"{filename}.sdf",
        ]

        sdf_path = None
        for path in search_paths:
            if path.exists():
                sdf_path = path
                break

        # Download to workspace if not found anywhere
        if sdf_path is None:
            sdf_path = self.molecules_dir / f"{filename}.sdf"
            print(f"Downloading molecule: {name}")
            result = self.downloader.download_molecule(name, prefer_3d=True)
            if not result:
                print(f"Failed to download {name}")
                return None

        # Load molecule
        try:
            mol = read(str(sdf_path))
            print(f"Loaded molecule: {name} ({len(mol)} atoms) from {sdf_path.parent.name}/")
            return mol
        except Exception as e:
            print(f"Error loading molecule {name}: {e}")
            return None
            
    def calculate_cluster_properties(self, molecules: List[Atoms]) -> dict:
        """
        Calculate properties of a molecular cluster for solvation.

        Args:
            molecules: List of ASE Atoms objects forming the cluster

        Returns:
            dict with:
                - center: Center of mass of cluster
                - bbox: Bounding box dimensions [dx, dy, dz]
                - radius: Approximate spherical radius
                - surface_area: Estimated surface area (Å²)
                - positions: All atomic positions
        """
        # Combine all positions
        all_positions = []
        all_symbols = []
        total_mass = 0
        weighted_pos = np.zeros(3)

        for mol in molecules:
            positions = mol.get_positions()
            symbols = mol.get_chemical_symbols()
            masses = mol.get_masses()

            all_positions.extend(positions)
            all_symbols.extend(symbols)

            for pos, mass in zip(positions, masses):
                weighted_pos += pos * mass
                total_mass += mass

        all_positions = np.array(all_positions)
        center = weighted_pos / total_mass if total_mass > 0 else np.mean(all_positions, axis=0)

        # Calculate bounding box
        min_coords = all_positions.min(axis=0)
        max_coords = all_positions.max(axis=0)
        bbox = max_coords - min_coords

        # Add vdW radii padding
        padding = 3.4  # Approximate diameter of common atoms
        bbox_padded = bbox + padding

        # Approximate spherical radius (use average of bbox dimensions)
        radius = np.mean(bbox_padded) / 2

        # Estimate surface area (approximation using ellipsoid formula)
        a, b, c = bbox_padded / 2
        # Knud Thomsen's approximation for ellipsoid surface area
        p = 1.6075
        surface_area = 4 * np.pi * ((a**p * b**p + a**p * c**p + b**p * c**p) / 3) ** (1/p)

        return {
            'center': center,
            'bbox': bbox_padded,
            'radius': radius,
            'surface_area': surface_area,
            'positions': all_positions,
            'symbols': all_symbols
        }

    def estimate_solvent_count(self, cluster_props: dict, shell_thickness: float = 3.0,
                               solvent_radius: float = 1.4, density_factor: float = 1.0) -> int:
        """
        Estimate number of solvent molecules needed for solvation shell.

        Args:
            cluster_props: Output from calculate_cluster_properties
            shell_thickness: Thickness of solvation shell in Å (default: 3.0)
            solvent_radius: Approximate radius of solvent molecule (default: 1.4 for water)
            density_factor: Multiplier for density (default: 1.0)

        Returns:
            Estimated number of solvent molecules
        """
        # Calculate shell volume
        inner_radius = cluster_props['radius']
        outer_radius = inner_radius + shell_thickness

        # Volume of spherical shell
        shell_volume = (4/3) * np.pi * (outer_radius**3 - inner_radius**3)

        # Volume per solvent molecule (approximate as sphere)
        solvent_volume = (4/3) * np.pi * solvent_radius**3

        # Packing efficiency (random packing ~64%, but we use lower for safety)
        packing_efficiency = 0.5

        # Estimate count
        count = int(shell_volume * packing_efficiency * density_factor / solvent_volume)

        # Ensure reasonable bounds
        count = max(4, min(count, 100))  # At least 4, at most 100

        return count

    def place_solvent_molecules(self, cluster_props: dict, solvent: Atoms,
                                count: int, shell_thickness: float = 3.0,
                                min_separation: float = 2.5) -> List[Atoms]:
        """
        Place solvent molecules around a cluster.

        Args:
            cluster_props: Output from calculate_cluster_properties
            solvent: ASE Atoms object of solvent molecule
            count: Number of solvent molecules to place
            shell_thickness: Thickness of solvation shell
            min_separation: Minimum distance between solvent molecules

        Returns:
            List of positioned solvent molecules
        """
        center = cluster_props['center']
        inner_radius = cluster_props['radius']
        outer_radius = inner_radius + shell_thickness

        placed_solvents = []
        placed_positions = []  # Track centers of placed solvents

        # Use Fibonacci sphere for even distribution
        golden_ratio = (1 + np.sqrt(5)) / 2
        max_attempts = count * 10

        i = 0
        attempts = 0

        while len(placed_solvents) < count and attempts < max_attempts:
            attempts += 1

            # Fibonacci sphere point - use modulo to wrap around
            # and add random offset to avoid repeating same positions
            n_points = count * 2
            i_wrapped = i % n_points
            offset = (i // n_points) * 0.5  # Small offset for each wrap

            theta = 2 * np.pi * (i_wrapped + offset) / golden_ratio
            phi = np.arccos(np.clip(1 - 2 * (i_wrapped + 0.5) / n_points, -1.0, 1.0))

            # Random radius within shell
            r = np.random.uniform(inner_radius + 1.0, outer_radius)

            # Convert to Cartesian
            x = center[0] + r * np.sin(phi) * np.cos(theta)
            y = center[1] + r * np.sin(phi) * np.sin(theta)
            z = center[2] + r * np.cos(phi)

            new_pos = np.array([x, y, z])

            # Check overlap with cluster
            min_dist_cluster = float('inf')
            for cpos in cluster_props['positions']:
                dist = np.linalg.norm(new_pos - cpos)
                min_dist_cluster = min(min_dist_cluster, dist)

            if min_dist_cluster < 2.5:  # Too close to cluster
                i += 1
                continue

            # Check overlap with other placed solvents
            too_close = False
            for ppos in placed_positions:
                if np.linalg.norm(new_pos - ppos) < min_separation:
                    too_close = True
                    break

            if too_close:
                i += 1
                continue

            # Place solvent
            solvent_copy = solvent.copy()
            solvent_com = solvent_copy.get_center_of_mass()
            solvent_copy.translate(new_pos - solvent_com)

            # Random rotation for variety
            from scipy.spatial.transform import Rotation
            random_rot = Rotation.random().as_matrix()
            solvent_copy.positions = (solvent_copy.positions - new_pos) @ random_rot.T + new_pos

            placed_solvents.append(solvent_copy)
            placed_positions.append(new_pos)
            i += 1

        return placed_solvents

    def build_solvated_system(self, cluster_molecules: List[Atoms], cell: np.ndarray,
                              solvation_config: dict) -> Tuple[Atoms, dict]:
        """
        Build a solvated molecular system.

        Args:
            cluster_molecules: List of molecules forming the solute cluster
            cell: Simulation cell
            solvation_config: Configuration dict with:
                - solvent: Solvent molecule name (default: "water")
                - mode: "auto" or not specified for automatic count
                - count: Manual solvent count (overrides auto)
                - shell_thickness: Solvation shell thickness in Å (default: 3.0)
                - density: Density factor (default: 1.0)

        Returns:
            Tuple of (solvated Atoms object, info dict)
        """
        # Parse config
        solvent_name = solvation_config.get("solvent", "water")
        mode = solvation_config.get("mode", "auto")
        manual_count = solvation_config.get("count", None)
        shell_thickness = solvation_config.get("shell_thickness", 3.0)
        density_factor = solvation_config.get("density", 1.0)

        # Download solvent molecule
        solvent_mol = self.download_and_load_molecule(solvent_name)
        if solvent_mol is None:
            print(f"Warning: Could not load solvent '{solvent_name}', skipping solvation")
            return None, {}

        # Calculate cluster properties
        cluster_props = self.calculate_cluster_properties(cluster_molecules)

        print(f"\nSolvation Analysis:")
        print(f"  Cluster size: {cluster_props['bbox'][0]:.1f} x {cluster_props['bbox'][1]:.1f} x {cluster_props['bbox'][2]:.1f} Å")
        print(f"  Cluster radius: {cluster_props['radius']:.1f} Å")
        print(f"  Surface area: {cluster_props['surface_area']:.1f} Ų")

        # Determine solvent count
        if manual_count is not None:
            count = int(manual_count)
            print(f"  Solvent count: {count} (manual)")
        else:
            # Estimate solvent radius based on solvent type
            solvent_radii = {"water": 1.4, "methanol": 2.0, "ethanol": 2.2, "acetone": 2.5}
            solvent_radius = solvent_radii.get(solvent_name.lower(), 1.5)

            count = self.estimate_solvent_count(
                cluster_props,
                shell_thickness=shell_thickness,
                solvent_radius=solvent_radius,
                density_factor=density_factor
            )
            print(f"  Estimated solvent count: {count} (auto, shell={shell_thickness} Å)")

        # Place solvent molecules
        placed_solvents = self.place_solvent_molecules(
            cluster_props, solvent_mol, count,
            shell_thickness=shell_thickness
        )

        print(f"  Successfully placed: {len(placed_solvents)} {solvent_name} molecules")

        # Build combined system
        solvated_system = Atoms(cell=cell, pbc=[True, True, True])

        # Add cluster molecules
        for mol in cluster_molecules:
            solvated_system.extend(mol.copy())

        # Add solvent molecules
        for sol in placed_solvents:
            solvated_system.extend(sol)

        # Center in cell
        solvated_system.center()

        info = {
            'solvent': solvent_name,
            'solvent_count': len(placed_solvents),
            'cluster_radius': cluster_props['radius'],
            'shell_thickness': shell_thickness
        }

        return solvated_system, info

    def calculate_contact_distance(self, mol1: Atoms, mol2: Atoms) -> float:
        """
        Calculate van der Waals contact distance between two molecules.

        This finds the minimum sum of vdW radii for any atom pair between
        the two molecules, representing the closest they can approach.

        Args:
            mol1, mol2: ASE Atoms objects

        Returns:
            Contact distance in Angstrom
        """
        symbols1 = mol1.get_chemical_symbols()
        symbols2 = mol2.get_chemical_symbols()

        # Find the pair with smallest combined vdW radii
        min_contact = float('inf')
        for s1 in set(symbols1):
            for s2 in set(symbols2):
                r1 = VDW_RADII.get(s1, 1.70)  # Default to carbon
                r2 = VDW_RADII.get(s2, 1.70)
                contact = r1 + r2
                if contact < min_contact:
                    min_contact = contact

        return min_contact

    def parse_relative_position(self, position_config: dict, reference_pos: np.ndarray) -> np.ndarray:
        """
        Parse relative position configuration.

        Args:
            position_config: Dict with keys:
                - relative_to: "probe" (reference molecule)
                - lateral_offset: horizontal distance in Å (default: 0)
                - vertical_offset: vertical distance in Å (default: 0)
                - direction: "x", "y", "-x", "-y", "radial", "random" (default: "x")
            reference_pos: Position of reference molecule [x, y, z]

        Returns:
            np.array of absolute [x, y, z] coordinates
        """
        lateral = position_config.get("lateral_offset", 0.0)
        vertical = position_config.get("vertical_offset", 0.0)
        direction = position_config.get("direction", "x")

        # Calculate offset based on direction
        if direction == "x":
            offset = np.array([lateral, 0.0, vertical])
        elif direction == "-x":
            offset = np.array([-lateral, 0.0, vertical])
        elif direction == "y":
            offset = np.array([0.0, lateral, vertical])
        elif direction == "-y":
            offset = np.array([0.0, -lateral, vertical])
        elif direction in ("radial", "random"):
            # Random angle in xy-plane
            angle = np.random.uniform(0, 2 * np.pi)
            offset = np.array([
                lateral * np.cos(angle),
                lateral * np.sin(angle),
                vertical
            ])
        else:
            print(f"Warning: Unknown direction '{direction}', using 'x'")
            offset = np.array([lateral, 0.0, vertical])

        return reference_pos + offset

    def calculate_overlap(self, atoms1: Atoms, atoms2: Atoms,
                         cutoff: float = 1.5) -> bool:
        """
        Check if two structures overlap
        
        Args:
            atoms1, atoms2: Atomic structures
            cutoff: Minimum allowed distance (Angstrom)
            
        Returns:
            True if structures overlap
        """
        # Use cell and pbc from atoms2 (the existing structure)
        if atoms2.cell is not None and atoms2.cell.rank > 0:
            cell = atoms2.cell
            pbc = atoms2.pbc
        else:
            cell = None
            pbc = False
            
        dist_array = get_distances(atoms1.positions, atoms2.positions, 
                                  cell=cell, pbc=pbc)[1]
        
        min_dist = np.min(dist_array)
        return min_dist < cutoff
        
    def find_safe_position(self, mol_to_place: Atoms, existing: Atoms,
                          initial_pos: np.ndarray, 
                          min_dist: float = 1.5,  # Reduced from 2.0 to allow closer approach
                          max_attempts: int = 100) -> np.ndarray:
        """
        Find a safe position for molecule avoiding overlaps
        
        Args:
            mol_to_place: Molecule to position
            existing: Existing structure to avoid
            initial_pos: Initial desired position
            min_dist: Minimum separation distance
            max_attempts: Maximum positioning attempts
            
        Returns:
            Safe position vector
        """
        # Try initial position
        mol_test = mol_to_place.copy()
        mol_test.translate(initial_pos - mol_test.get_center_of_mass())
        
        if not self.calculate_overlap(mol_test, existing, min_dist):
            return initial_pos
            
        print(f"Initial position has overlap, searching for safe position...")
        
        # Try systematic displacement
        for attempt in range(max_attempts):
            # Try different strategies
            if attempt < 20:
                # Small random lateral displacement
                dx, dy = np.random.randn(2) * 2.0
                dz = 0
            elif attempt < 40:
                # Larger lateral displacement
                angle = 2 * np.pi * attempt / 20
                radius = 3.0 + (attempt - 20) * 0.2
                dx = radius * np.cos(angle)
                dy = radius * np.sin(angle)
                dz = 0
            elif attempt < 60:
                # Vertical displacement
                dx, dy = 0, 0
                dz = (attempt - 40) * 0.5
            else:
                # Random 3D displacement
                dx, dy, dz = np.random.randn(3) * 3.0
                
            test_pos = initial_pos + np.array([dx, dy, dz])
            mol_test = mol_to_place.copy()
            mol_test.translate(test_pos - mol_test.get_center_of_mass())
            
            if not self.calculate_overlap(mol_test, existing, min_dist):
                print(f"Found safe position after {attempt + 1} attempts")
                return test_pos
                
        print("Warning: Could not find overlap-free position, using best effort")
        return initial_pos + np.array([0, 0, 3.0])  # Move up as last resort
    
    def parse_position(self, position_config, cell, substrate_top=0.0):
        """
        Parse position configuration to absolute coordinates

        Args:
            position_config: Can be:
                - "auto": Use automatic positioning
                - [x, y, z]: Absolute coordinates in Å
                - {"x": 0.5, "y": 0.5, "z": 12.0}: Fractional x,y (0-1), absolute z
                - {"frac": [0.5, 0.5, 0.3]}: All fractional coordinates
                - {"cylindrical": {"r": 5.0, "theta": 45, "z": 12.0}}: Cylindrical coords
                - {"relative_to": "probe", "lateral_offset": 4.0, ...}: Relative positioning
            cell: Unit cell for fractional coordinates
            substrate_top: Z-position of substrate top (for relative positioning)

        Returns:
            np.array of absolute [x, y, z] coordinates, "auto", or {"_relative": config}
        """
        if position_config is None or position_config == "auto":
            return "auto"

        # List format: absolute coordinates
        if isinstance(position_config, (list, tuple)) and len(position_config) == 3:
            return np.array(position_config)

        # Dictionary format: various coordinate systems
        if isinstance(position_config, dict):
            # Relative positioning (handled later in build_simulation)
            if "relative_to" in position_config:
                return {"_relative": position_config}

            # Fractional x,y with absolute z
            if "x" in position_config and "y" in position_config and "z" in position_config:
                x = position_config["x"] * cell[0, 0] if position_config["x"] <= 1.0 else position_config["x"]
                y = position_config["y"] * cell[1, 1] if position_config["y"] <= 1.0 else position_config["y"]
                z = position_config["z"]
                # Handle relative z positioning
                if isinstance(z, dict) and "above_substrate" in z:
                    z = substrate_top + z["above_substrate"]
                return np.array([x, y, z])

            # Full fractional coordinates
            if "frac" in position_config:
                frac = np.array(position_config["frac"])
                return np.dot(frac, cell)

            # Cylindrical coordinates (useful for pores)
            if "cylindrical" in position_config:
                cyl = position_config["cylindrical"]
                r = cyl["r"]
                theta = np.radians(cyl["theta"])
                z = cyl["z"]
                center = cell.diagonal()[:2] / 2
                x = center[0] + r * np.cos(theta)
                y = center[1] + r * np.sin(theta)
                return np.array([x, y, z])

        print(f"Warning: Unrecognized position format {position_config}, using auto")
        return "auto"
    
    def parse_orientation(self, orientation_config):
        """
        Parse orientation configuration to rotation matrix
        
        Args:
            orientation_config: Can be:
                - "auto" or None: No rotation
                - [rx, ry, rz]: Euler angles in degrees
                - {"axis": [x, y, z], "angle": degrees}: Axis-angle rotation
                - {"euler": [rx, ry, rz], "convention": "xyz"}: Explicit Euler angles
                - {"quaternion": [w, x, y, z]}: Quaternion rotation
                - {"align": "flat"/"standing"/"tilted"}: Preset orientations
                
        Returns:
            3x3 rotation matrix or "auto"
        """
        if orientation_config is None or orientation_config == "auto":
            return "auto"
        
        # List format: Euler angles in degrees
        if isinstance(orientation_config, (list, tuple)) and len(orientation_config) == 3:
            rotation = Rotation.from_euler('xyz', orientation_config, degrees=True)
            return rotation.as_matrix()
        
        # Dictionary format: various rotation specifications
        if isinstance(orientation_config, dict):
            # Axis-angle rotation
            if "axis" in orientation_config and "angle" in orientation_config:
                axis = np.array(orientation_config["axis"])
                axis = axis / np.linalg.norm(axis)  # Normalize
                angle = np.radians(orientation_config["angle"])
                rotation = Rotation.from_rotvec(axis * angle)
                return rotation.as_matrix()
            
            # Explicit Euler angles with convention
            if "euler" in orientation_config:
                angles = orientation_config["euler"]
                convention = orientation_config.get("convention", "xyz")
                rotation = Rotation.from_euler(convention, angles, degrees=True)
                return rotation.as_matrix()
            
            # Quaternion
            if "quaternion" in orientation_config:
                quat = orientation_config["quaternion"]
                rotation = Rotation.from_quat(quat)
                return rotation.as_matrix()
            
            # Preset orientations
            if "align" in orientation_config:
                align = orientation_config["align"]
                if align == "flat":
                    # Rotate to lie flat (90 degrees around x-axis)
                    rotation = Rotation.from_euler('x', 90, degrees=True)
                elif align == "standing":
                    # No rotation, molecule stands as is
                    rotation = Rotation.from_euler('xyz', [0, 0, 0], degrees=True)
                elif align == "tilted":
                    # 45 degree tilt
                    rotation = Rotation.from_euler('x', 45, degrees=True)
                else:
                    print(f"Unknown alignment preset: {align}")
                    return "auto"
                return rotation.as_matrix()
        
        print(f"Warning: Unrecognized orientation format {orientation_config}, using auto")
        return "auto"
    
    def apply_transformation(self, molecule, position, orientation):
        """
        Apply position and orientation transformations to molecule
        
        Args:
            molecule: ASE Atoms object to transform
            position: Target position (np.array) or "auto"
            orientation: Rotation matrix or "auto"
            
        Returns:
            Transformed molecule (copy)
        """
        mol_copy = molecule.copy()
        
        # Apply orientation first (rotate around center of mass)
        if not isinstance(orientation, str) and orientation is not None:
            com = mol_copy.get_center_of_mass()
            # Translate to origin
            mol_copy.positions -= com
            # Apply rotation
            mol_copy.positions = mol_copy.positions @ orientation.T
            # Translate back
            mol_copy.positions += com
        
        # Apply position
        if not isinstance(position, str) and position is not None:
            current_com = mol_copy.get_center_of_mass()
            mol_copy.translate(position - current_com)
        
        return mol_copy
        
    def build_simulation(self) -> Dict[str, Atoms]:
        """
        Build all simulation structures according to configuration
        
        Returns:
            Dictionary of simulation structures
        """
        structures = {}
        
        # Load components
        print("\n" + "="*60)
        print("Loading components...")
        print("="*60)
        
        # Load substrate
        self.substrate = self.load_substrate()
        
        # Load molecules
        probe_name = self.config.get("probe")
        target_name = self.config.get("target")
        
        if probe_name:
            self.probe = self.download_and_load_molecule(probe_name)
        if target_name:
            self.target = self.download_and_load_molecule(target_name)
            
        # Get positioning parameters (backward compatible)
        probe_height = self.config.get("probe_height", 2.5)  # Height above substrate
        target_height = self.config.get("target_height", 6.0)  # Height above substrate
        probe_target_distance_config = self.config.get("probe_target_distance", 4.0)

        # Handle contact distance mode
        if probe_target_distance_config == "contact" and self.probe and self.target:
            probe_target_distance = self.calculate_contact_distance(self.probe, self.target)
            print(f"Using contact distance: {probe_target_distance:.2f} Å (van der Waals)")
        else:
            probe_target_distance = float(probe_target_distance_config) if probe_target_distance_config != "contact" else 4.0
        
        # Get custom position/orientation if specified
        probe_position_config = self.config.get("probe_position", "auto")
        probe_orientation_config = self.config.get("probe_orientation", "auto")
        target_position_config = self.config.get("target_position", "auto")
        target_orientation_config = self.config.get("target_orientation", "auto")
        
        # Determine box size
        if self.substrate:
            cell = self.substrate.cell.copy()
            pbc = [True, True, True]  # Full PBC for VASP/FAIRChem compatibility
            # Add vacuum above
            cell[2, 2] = max(cell[2, 2], 30.0)  # Ensure enough vacuum
        else:
            # Create vacuum box
            box_size = self.config.get("box_size", [30.0, 30.0, 40.0])
            cell = np.diag(box_size)
            pbc = [True, True, True]  # Full PBC even for vacuum
            
        # Get substrate top
        if self.substrate:
            substrate_top = self.substrate.positions[:, 2].max()
        else:
            substrate_top = 0.0
            
        print(f"\nSubstrate top at z = {substrate_top:.2f} Å")
        
        # Build structures
        print("\n" + "="*60)
        print("Building structures...")
        print("="*60)
        
        # 1. Substrate only (if exists)
        if self.substrate:
            structures["substrate_only"] = self.substrate.copy()
            print("Created: substrate_only")
            
        # 2. Probe + Substrate
        if self.probe and self.substrate:
            system_probe_sub = self.substrate.copy()
            
            # Parse position and orientation
            probe_position = self.parse_position(probe_position_config, cell, substrate_top)
            probe_orientation = self.parse_orientation(probe_orientation_config)
            
            # Use automatic positioning if not specified
            if isinstance(probe_position, str) and probe_position == "auto":
                center_xy = cell.diagonal()[:2] / 2
                probe_pos = np.array([center_xy[0], center_xy[1], substrate_top + probe_height])
            else:
                probe_pos = probe_position
            
            # Apply transformations
            probe_mol = self.apply_transformation(self.probe, probe_pos, probe_orientation)
            
            # Check and adjust for overlaps (only if using auto position)
            if isinstance(probe_position, str) and probe_position == "auto":
                safe_pos = self.find_safe_position(probe_mol, system_probe_sub, probe_pos)
                probe_mol.translate(safe_pos - probe_pos)
            
            # Combine
            system_probe_sub.extend(probe_mol)
            system_probe_sub.cell = cell
            system_probe_sub.pbc = pbc
            structures["probe_substrate"] = system_probe_sub
            
            # Report what was used
            if not (isinstance(probe_position, str) and probe_position == "auto") or not (isinstance(probe_orientation, str) and probe_orientation == "auto"):
                print(f"Created: probe_substrate (custom placement)")
            else:
                print("Created: probe_substrate")
            
        # 3. Probe in vacuum (use same cell as substrate system for consistency)
        if self.probe:
            probe_vac = self.probe.copy()
            # Use the same cell dimensions as the substrate system
            probe_vac.cell = cell.copy()
            probe_vac.pbc = pbc
            # Center in the cell
            probe_vac.center()
            structures["probe_vacuum"] = probe_vac
            print("Created: probe_vacuum")
            
        # 4. Target in vacuum (use same cell as substrate system for consistency)
        if self.target:
            target_vac = self.target.copy()
            # Use the same cell dimensions as the substrate system
            target_vac.cell = cell.copy()
            target_vac.pbc = pbc
            # Center in the cell
            target_vac.center()
            structures["target_vacuum"] = target_vac
            print("Created: target_vacuum")
            
        # 5. Both probe and target in vacuum (for reference) - target above probe
        if self.probe and self.target:
            # Create vacuum box
            vacuum_box = Atoms(cell=cell, pbc=[True, True, True])  # Full PBC
            center_xy = cell.diagonal()[:2] / 2
            
            # Parse positions and orientations for vacuum setup
            probe_position_vac = self.parse_position(probe_position_config, cell, 0)
            probe_orientation_vac = self.parse_orientation(probe_orientation_config)
            target_position_vac = self.parse_position(target_position_config, cell, 0)
            target_orientation_vac = self.parse_orientation(target_orientation_config)
            
            # Position probe
            if isinstance(probe_position_vac, str) and probe_position_vac == "auto":
                probe_pos = np.array([center_xy[0], center_xy[1], cell[2,2]/2 - probe_target_distance/2])
            elif isinstance(probe_position_vac, dict) and "_relative" in probe_position_vac:
                # Probe cannot be relative to itself, use auto
                probe_pos = np.array([center_xy[0], center_xy[1], cell[2,2]/2 - probe_target_distance/2])
            else:
                probe_pos = probe_position_vac

            probe_mol = self.apply_transformation(self.probe, probe_pos, probe_orientation_vac)
            vacuum_box.extend(probe_mol)
            actual_probe_pos = probe_mol.get_center_of_mass()

            # Position target
            if isinstance(target_position_vac, str) and target_position_vac == "auto":
                target_pos = np.array([center_xy[0], center_xy[1], cell[2,2]/2 + probe_target_distance/2])
            elif isinstance(target_position_vac, dict) and "_relative" in target_position_vac:
                # Handle relative positioning to probe
                rel_config = target_position_vac["_relative"]
                if rel_config.get("relative_to") == "probe":
                    target_pos = self.parse_relative_position(rel_config, actual_probe_pos)
                    print(f"Target positioned relative to probe: offset=({rel_config.get('lateral_offset', 0)}, {rel_config.get('vertical_offset', 0)}) direction={rel_config.get('direction', 'x')}")
                else:
                    print(f"Warning: Unknown relative_to value '{rel_config.get('relative_to')}', using auto")
                    target_pos = np.array([center_xy[0], center_xy[1], cell[2,2]/2 + probe_target_distance/2])
            else:
                target_pos = target_position_vac
            
            target_mol = self.apply_transformation(self.target, target_pos, target_orientation_vac)
            
            # Check and fix overlap (only if using auto positions)
            if isinstance(probe_position_vac, str) and probe_position_vac == "auto" and isinstance(target_position_vac, str) and target_position_vac == "auto":
                safe_target_pos = self.find_safe_position(target_mol, vacuum_box, target_pos, min_dist=2.5)
                target_mol.translate(safe_target_pos - target_pos)
            
            vacuum_box.extend(target_mol)
            structures["probe_target_vacuum"] = vacuum_box
            
            # Report what was used
            custom_used = (not (isinstance(probe_position_vac, str) and probe_position_vac == "auto") or not (isinstance(probe_orientation_vac, str) and probe_orientation_vac == "auto") or 
                          not (isinstance(target_position_vac, str) and target_position_vac == "auto") or not (isinstance(target_orientation_vac, str) and target_orientation_vac == "auto"))
            if custom_used:
                print("Created: probe_target_vacuum (custom placement)")
            else:
                print("Created: probe_target_vacuum (probe + target in vacuum, target above probe)")
            
        # 6. Three-component system: Probe + Target + Substrate
        # This will be created after probe_substrate is optimized
        if self.probe and self.target and self.substrate:
            # Create the three-component system
            system_three = self.substrate.copy()
            
            # Parse probe position and orientation (same as used above)
            probe_position = self.parse_position(probe_position_config, cell, substrate_top)
            probe_orientation = self.parse_orientation(probe_orientation_config)
            
            # Add probe at its position
            center_xy = cell.diagonal()[:2] / 2
            if isinstance(probe_position, str) and probe_position == "auto":
                probe_pos = np.array([center_xy[0], center_xy[1], substrate_top + probe_height])
            elif isinstance(probe_position, dict) and "_relative" in probe_position:
                # Probe cannot be relative to itself, use auto
                probe_pos = np.array([center_xy[0], center_xy[1], substrate_top + probe_height])
            else:
                probe_pos = probe_position

            probe_mol = self.apply_transformation(self.probe, probe_pos, probe_orientation)

            # Check and adjust for overlaps (only if using auto position)
            if isinstance(probe_position, str) and probe_position == "auto":
                safe_pos = self.find_safe_position(probe_mol, system_three, probe_pos)
                probe_mol.translate(safe_pos - probe_pos)

            system_three.extend(probe_mol)
            actual_probe_pos = probe_mol.get_center_of_mass()

            # Parse target position and orientation
            target_position = self.parse_position(target_position_config, cell, substrate_top)
            target_orientation = self.parse_orientation(target_orientation_config)

            # Now add target - handle various positioning modes
            if isinstance(target_position, str) and target_position == "auto":
                # Place target above probe
                target_pos = np.array([center_xy[0], center_xy[1], substrate_top + target_height])
            elif isinstance(target_position, dict) and "_relative" in target_position:
                # Handle relative positioning to probe
                rel_config = target_position["_relative"]
                if rel_config.get("relative_to") == "probe":
                    target_pos = self.parse_relative_position(rel_config, actual_probe_pos)
                    print(f"Target positioned relative to probe: offset=({rel_config.get('lateral_offset', 0)}, {rel_config.get('vertical_offset', 0)}) direction={rel_config.get('direction', 'x')}")
                else:
                    print(f"Warning: Unknown relative_to value '{rel_config.get('relative_to')}', using auto")
                    target_pos = np.array([center_xy[0], center_xy[1], substrate_top + target_height])
            else:
                target_pos = target_position

            target_mol = self.apply_transformation(self.target, target_pos, target_orientation)

            # Check for overlaps with the existing system (only if using auto position)
            if isinstance(target_position, str) and target_position == "auto":
                safe_target_pos = self.find_safe_position(target_mol, system_three, target_pos, min_dist=2.5)
                target_mol.translate(safe_target_pos - target_pos)
            
            system_three.extend(target_mol)
            system_three.cell = cell
            system_three.pbc = pbc
            structures["probe_target_substrate"] = system_three
            print("Created: probe_target_substrate (three-component system)")

        # 7. Solvated systems (if solvation config provided)
        solvation_config = self.config.get("solvation", None)
        if solvation_config and solvation_config.get("enabled", False) and self.probe:
            print("\n" + "="*60)
            print("Building solvated structures...")
            print("="*60)

            shell_thickness = solvation_config.get("shell_thickness", 3.0)

            # 7a. Always create probe_solvated
            probe_for_solv = self.probe.copy()
            probe_for_solv.cell = cell.copy()
            probe_for_solv.pbc = [True, True, True]
            probe_for_solv.center()

            probe_props = self.calculate_cluster_properties([probe_for_solv])
            probe_cell_size = 2 * (probe_props['radius'] + shell_thickness + 5.0)
            probe_solv_cell = np.diag([max(probe_cell_size, cell[0, 0]),
                                       max(probe_cell_size, cell[1, 1]),
                                       max(probe_cell_size, cell[2, 2])])

            probe_solvated, probe_solv_info = self.build_solvated_system(
                [probe_for_solv], probe_solv_cell, solvation_config
            )
            if probe_solvated is not None:
                structures["probe_solvated"] = probe_solvated
                print(f"Created: probe_solvated ({probe_solv_info['solvent_count']} {probe_solv_info['solvent']} molecules)")

            # 7b. Create target_solvated if target exists
            if self.target:
                target_for_solv = self.target.copy()
                target_for_solv.cell = cell.copy()
                target_for_solv.pbc = [True, True, True]
                target_for_solv.center()

                target_props = self.calculate_cluster_properties([target_for_solv])
                target_cell_size = 2 * (target_props['radius'] + shell_thickness + 5.0)
                target_solv_cell = np.diag([max(target_cell_size, cell[0, 0]),
                                            max(target_cell_size, cell[1, 1]),
                                            max(target_cell_size, cell[2, 2])])

                target_solvated, target_solv_info = self.build_solvated_system(
                    [target_for_solv], target_solv_cell, solvation_config
                )
                if target_solvated is not None:
                    structures["target_solvated"] = target_solvated
                    print(f"Created: target_solvated ({target_solv_info['solvent_count']} {target_solv_info['solvent']} molecules)")

                # 7c. Create probe_target_solvated (complex)
                probe_for_complex = self.probe.copy()
                probe_for_complex.cell = cell.copy()
                probe_for_complex.pbc = [True, True, True]
                probe_for_complex.center()

                target_for_complex = self.target.copy()
                target_for_complex.cell = cell.copy()
                target_for_complex.pbc = [True, True, True]
                # Position target relative to probe
                probe_com = probe_for_complex.get_center_of_mass()
                target_com = target_for_complex.get_center_of_mass()
                contact_dist = self.calculate_contact_distance(probe_for_complex, target_for_complex)
                target_for_complex.translate(probe_com - target_com + np.array([0, 0, contact_dist]))

                complex_molecules = [probe_for_complex, target_for_complex]
                complex_props = self.calculate_cluster_properties(complex_molecules)
                complex_cell_size = 2 * (complex_props['radius'] + shell_thickness + 5.0)
                complex_solv_cell = np.diag([max(complex_cell_size, cell[0, 0]),
                                             max(complex_cell_size, cell[1, 1]),
                                             max(complex_cell_size, cell[2, 2])])

                complex_solvated, complex_solv_info = self.build_solvated_system(
                    complex_molecules, complex_solv_cell, solvation_config
                )
                if complex_solvated is not None:
                    structures["probe_target_solvated"] = complex_solvated
                    print(f"Created: probe_target_solvated ({complex_solv_info['solvent_count']} {complex_solv_info['solvent']} molecules)")

        # Set charge and spin in atoms.info for all structures (required for omol task)
        if self.charge != 0 or self.spin != 1:
            print(f"\nSetting charge={self.charge}, spin={self.spin} for all structures (omol task)")
        for name, atoms in structures.items():
            atoms.info["charge"] = self.charge
            atoms.info["spin"] = self.spin

        return structures
        
    def save_structures(self, structures: Dict[str, Atoms]):
        """Save all structures to files"""
        print("\n" + "="*60)
        print("Saving structures...")
        print("="*60)
        
        # Create run directory
        run_name = self.config.get("run_name", "simulation")
        run_dir = self.output_dir / run_name
        run_dir.mkdir(exist_ok=True)
        
        # Save configuration
        config_path = run_dir / "config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2)
        print(f"Saved configuration: {config_path}")
        
        # Save structures
        for name, atoms in structures.items():
            # Save in multiple formats
            vasp_path = run_dir / f"{name}.vasp"
            xyz_path = run_dir / f"{name}.xyz"

            # Sort atoms by chemical symbol for VASP compatibility
            write(vasp_path, atoms, format='vasp', sort=True)
            write(xyz_path, atoms, format='xyz')
            
            print(f"Saved: {name} ({len(atoms)} atoms)")
            print(f"  - VASP format: {vasp_path}")
            print(f"  - XYZ format: {xyz_path}")
            
        # Generate summary
        summary_path = run_dir / "summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("Simulation Builder Summary\n")
            f.write("="*60 + "\n\n")
            f.write(f"Run name: {run_name}\n")
            f.write(f"Substrate: {self.config.get('substrate', 'vacuum')}\n")
            f.write(f"Probe: {self.config.get('probe', 'None')}\n")
            f.write(f"Target: {self.config.get('target', 'None')}\n")
            f.write(f"Probe height: {self.config.get('probe_height', 2.5)} Å\n")
            f.write(f"Target height: {self.config.get('target_height', 6.0)} Å\n")
            f.write(f"Probe-Target distance: {self.config.get('probe_target_distance', 4.0)} Å\n")
            f.write("\nGenerated structures:\n")
            for name, atoms in structures.items():
                f.write(f"  - {name}: {len(atoms)} atoms\n")
                
        print(f"\nSummary saved: {summary_path}")
        print(f"All files saved to: {run_dir}")
        
    def run(self):
        """Execute the complete workflow"""
        print("\n" + "="*60)
        print("SIMULATION BUILDER")
        print("="*60)
        
        # Build structures
        structures = self.build_simulation()
        
        if not structures:
            print("Error: No structures generated")
            return
            
        # Save structures
        self.save_structures(structures)
        
        print("\n" + "="*60)
        print("COMPLETED SUCCESSFULLY")
        print("="*60)
        

def main():
    """Command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Build atomic simulation environments from JSON configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example JSON configuration:
{
  "run_name": "glucose_caffeine_graphene",
  "substrate": "Graphene",
  "molecule_a": "glucose",
  "molecule_b": "caffeine", 
  "height_a": 3.0,
  "height_b": 6.0,
  "distance_ab": 5.0,
  "fix_substrate_layers": 1,
  "output_dir": "simulations"
}

Available substrates: Graphene, MoS2, BP, Si, ZnO (or "vacuum" for no substrate)
        """
    )
    
    parser.add_argument("config", help="JSON configuration file")
    parser.add_argument("--visualize", action="store_true", 
                       help="Visualize structures after creation")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)
        
    # Run builder
    builder = SimulationBuilder(config_file=args.config)
    builder.run()
    
    # Optional visualization
    if args.visualize:
        try:
            from ase.visualize import view
            run_name = builder.config.get("run_name", "simulation")
            full_system_path = builder.output_dir / run_name / "full_system.vasp"
            if full_system_path.exists():
                atoms = read(full_system_path)
                view(atoms)
        except:
            print("Visualization not available (requires ase-gui)")
            

if __name__ == "__main__":
    main()