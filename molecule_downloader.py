#!/usr/bin/env python3
"""
Simple Molecule Downloader - Get SDF files from chemical names
Author: Claude
"""

import os
import sys
import requests
import argparse
from typing import Optional, List
from urllib.parse import quote

# Optional imports for enhanced functionality
try:
    import pubchempy as pcp
    HAS_PUBCHEMPY = True
except ImportError:
    HAS_PUBCHEMPY = False
    
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


def is_3d_structure(filepath: str) -> bool:
    """
    Check if an SDF file contains a 3D structure.

    Detection methods:
    1. Check header line for "3D" indicator (e.g., "-OEChem-...3D")
    2. Check if z-coordinates are all zero (indicates 2D)

    Args:
        filepath: Path to SDF file

    Returns:
        True if 3D structure, False if 2D or unknown
    """
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()

        if len(lines) < 5:
            return False

        # Method 1: Check header line (line 2, index 1) for "3D" indicator
        header = lines[1] if len(lines) > 1 else ""
        if "3D" in header.upper():
            return True
        if "2D" in header.upper():
            return False

        # Method 2: Check z-coordinates in atom block
        # Line 4 (index 3) contains atom count: "  N  M  ..."
        # Atom coordinates start from line 5 (index 4)
        counts_line = lines[3].split()
        if len(counts_line) < 2:
            return False

        try:
            n_atoms = int(counts_line[0])
        except ValueError:
            return False

        # Check z-coordinates of atoms
        z_coords = []
        for i in range(4, min(4 + n_atoms, len(lines))):
            parts = lines[i].split()
            if len(parts) >= 4:
                try:
                    z = float(parts[2])
                    z_coords.append(z)
                except ValueError:
                    continue

        if not z_coords:
            return False

        # If all z-coords are essentially zero, it's 2D
        all_zero = all(abs(z) < 1e-6 for z in z_coords)
        return not all_zero

    except Exception:
        return False


class MoleculeDownloader:
    """Download molecular structures from PubChem by name, with local rare_molecules support"""

    def __init__(self, output_dir: str = "molecules", rare_dir: str = "rare_molecules"):
        """
        Initialize downloader

        Args:
            output_dir: Directory to save downloaded files
            rare_dir: Directory containing pre-optimized rare/complex molecules
        """
        self.output_dir = output_dir
        self.rare_dir = rare_dir
        os.makedirs(output_dir, exist_ok=True)
        self.pubchem_base = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
        
    def check_rare_molecules(self, name: str) -> Optional[str]:
        """
        Check if molecule exists in rare_molecules folder
        
        Args:
            name: Chemical name
            
        Returns:
            Path to file in rare_molecules if found, None otherwise
        """
        if not os.path.exists(self.rare_dir):
            return None
        
        # Normalize name for comparison (lowercase, replace spaces/dashes)
        normalized_name = name.lower().replace(' ', '_').replace('-', '_')
        
        # Check for exact match or common variations
        possible_names = [
            f"{normalized_name}.sdf",
            f"{normalized_name.replace('_', '')}.sdf",  # No underscores
            f"{normalized_name.replace('_', '-')}.sdf",  # Dashes instead
        ]
        
        # Also check for special naming conventions
        if "beta-cd" in normalized_name or "beta_cd" in normalized_name or "betacd" in normalized_name:
            possible_names.extend(["beta_cd.sdf", "beta-cd.sdf", "betacd.sdf"])
        if "cnt" in normalized_name or "nanotube" in normalized_name:
            possible_names.extend(["CNT.sdf", "cnt.sdf", "carbon_nanotube.sdf"])
        
        for filename in possible_names:
            rare_path = os.path.join(self.rare_dir, filename)
            if os.path.exists(rare_path):
                # Copy to molecules directory for consistency
                import shutil
                dest_name = f"{name.replace(' ', '_').replace('/', '_')}.sdf"
                dest_path = os.path.join(self.output_dir, dest_name)
                shutil.copy2(rare_path, dest_path)
                print(f"✓ Found '{name}' in rare_molecules collection")
                print(f"  Copied from: {rare_path}")
                print(f"  Saved to: {dest_path}")
                return dest_path
        
        return None
    
    def download_by_name_rest(self, name: str, prefer_3d: bool = True) -> Optional[str]:
        """
        Download molecule using PubChem REST API (after checking rare_molecules)
        
        Args:
            name: Chemical name (e.g., "glucose", "aspirin")
            prefer_3d: Try to get 3D conformation if available
            
        Returns:
            Path to saved file or None if failed
        """
        safe_name = quote(name)
        filename = f"{name.replace(' ', '_').replace('/', '_')}.sdf"
        filepath = os.path.join(self.output_dir, filename)
        got_2d = False
        
        # Try 3D first if preferred
        if prefer_3d:
            url_3d = f"{self.pubchem_base}/compound/name/{safe_name}/record/SDF?record_type=3d"
            print(f"Attempting to download 3D structure for '{name}'...")
            
            try:
                response = requests.get(url_3d, timeout=30)
                if response.status_code == 200 and len(response.content) > 50:
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    print(f"✓ Downloaded 3D structure to: {filepath}")
                    return filepath
                else:
                    print(f"No 3D structure available, trying 2D...")
            except Exception as e:
                print(f"3D download failed: {e}")
        
        # Fallback to 2D
        url_2d = f"{self.pubchem_base}/compound/name/{safe_name}/record/SDF"
        try:
            response = requests.get(url_2d, timeout=30)
            if response.status_code == 200:
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                print(f"✓ Downloaded 2D structure to: {filepath}")
                got_2d = True
                
                # Auto-convert 2D to 3D if RDKit is available
                if prefer_3d and got_2d and HAS_RDKIT:
                    print("Automatically converting 2D to 3D using RDKit...")
                    generated_3d = self.generate_3d_from_smiles(name)
                    if generated_3d:
                        return generated_3d
                    else:
                        print("3D generation failed, using 2D structure")
                
                return filepath
            else:
                print(f"✗ Failed to download '{name}': HTTP {response.status_code}")
                return None
        except Exception as e:
            print(f"✗ Download failed for '{name}': {e}")
            return None
            
    def download_by_name_pubchempy(self, name: str, prefer_3d: bool = True) -> Optional[str]:
        """
        Download using PubChemPy library (if installed)
        
        Args:
            name: Chemical name
            prefer_3d: Try to get 3D conformation
            
        Returns:
            Path to saved file or None
        """
        if not HAS_PUBCHEMPY:
            return self.download_by_name_rest(name, prefer_3d)
            
        filename = f"{name.replace(' ', '_').replace('/', '_')}.sdf"
        filepath = os.path.join(self.output_dir, filename)
        got_2d = False
        
        try:
            # Try 3D first
            if prefer_3d:
                print(f"Attempting to download 3D structure for '{name}' via PubChemPy...")
                try:
                    pcp.download(
                        outformat="SDF",
                        path=filepath,
                        identifier=name,
                        namespace="name",
                        record_type="3d"
                    )
                    if os.path.exists(filepath) and os.path.getsize(filepath) > 50:
                        print(f"✓ Downloaded 3D structure to: {filepath}")
                        return filepath
                except:
                    print("No 3D structure available, trying 2D...")
            
            # Fallback to 2D
            pcp.download(
                outformat="SDF",
                path=filepath,
                identifier=name,
                namespace="name"
            )
            if os.path.exists(filepath):
                print(f"✓ Downloaded 2D structure to: {filepath}")
                got_2d = True
                
                # Auto-convert 2D to 3D if RDKit is available
                if prefer_3d and got_2d and HAS_RDKIT:
                    print("Automatically converting 2D to 3D using RDKit...")
                    generated_3d = self.generate_3d_from_smiles(name)
                    if generated_3d:
                        return generated_3d
                    else:
                        print("3D generation failed, using 2D structure")
                
                return filepath
                
        except Exception as e:
            print(f"PubChemPy download failed: {e}")
            # Fallback to REST
            return self.download_by_name_rest(name, prefer_3d)
            
        return None
        
    def generate_3d_from_smiles(self, name: str) -> Optional[str]:
        """
        Generate 3D structure using RDKit from SMILES
        
        Args:
            name: Chemical name
            
        Returns:
            Path to generated 3D file or None
        """
        if not HAS_RDKIT:
            print("RDKit not installed, cannot generate 3D structures")
            return None
            
        if not HAS_PUBCHEMPY:
            print("PubChemPy needed to get SMILES for 3D generation")
            return None
            
        try:
            # Get SMILES from PubChem
            print(f"Getting SMILES for '{name}' to generate 3D structure...")
            compounds = pcp.get_compounds(name, "name")
            if not compounds:
                print(f"No compound found for '{name}'")
                return None
                
            smiles = compounds[0].isomeric_smiles or compounds[0].canonical_smiles
            if not smiles:
                print(f"No SMILES available for '{name}'")
                return None
                
            print(f"SMILES: {smiles}")
            
            # Generate 3D with RDKit
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print("Failed to parse SMILES")
                return None
                
            mol = Chem.AddHs(mol)
            
            # Generate 3D coordinates
            result = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            if result != 0:
                print("Failed to generate 3D coordinates")
                return None
                
            # Optimize geometry
            AllChem.UFFOptimizeMolecule(mol)
            
            # Save as SDF
            filename = f"{name.replace(' ', '_').replace('/', '_')}_generated3d.sdf"
            filepath = os.path.join(self.output_dir, filename)
            
            writer = Chem.SDWriter(filepath)
            writer.write(mol)
            writer.close()
            
            print(f"✓ Generated 3D structure saved to: {filepath}")
            return filepath
            
        except Exception as e:
            print(f"3D generation failed: {e}")
            return None
            
    def download_molecule(self, name: str, prefer_3d: bool = True) -> Optional[str]:
        """
        Main method to download/retrieve a molecule
        Checks rare_molecules first, then tries PubChem

        Args:
            name: Chemical name
            prefer_3d: Try to get 3D conformation if available

        Returns:
            Path to saved file or None if failed
        """
        # First check if already downloaded
        filename = f"{name.replace(' ', '_').replace('/', '_')}.sdf"
        existing_path = os.path.join(self.output_dir, filename)
        if os.path.exists(existing_path):
            # Validate existing file: check if it's 3D when 3D is preferred
            if prefer_3d and not is_3d_structure(existing_path):
                print(f"⚠ Existing file is 2D structure: {existing_path}")
                print(f"  Attempting to get 3D structure...")
                # Remove the 2D file and re-download
                os.remove(existing_path)
            else:
                print(f"✓ Using existing file: {existing_path}")
                return existing_path

        # Check rare_molecules collection
        rare_path = self.check_rare_molecules(name)
        if rare_path:
            return rare_path

        # Try downloading from PubChem
        print(f"Not found in rare_molecules, trying PubChem...")
        if HAS_PUBCHEMPY:
            return self.download_by_name_pubchempy(name, prefer_3d)
        else:
            return self.download_by_name_rest(name, prefer_3d)
    
    def download_batch(self, names: List[str], prefer_3d: bool = True) -> dict:
        """
        Download multiple molecules (auto-converts 2D to 3D if RDKit available)
        
        Args:
            names: List of chemical names
            prefer_3d: Try to get 3D conformations (auto-generates if only 2D available)
            
        Returns:
            Dictionary mapping names to file paths
        """
        results = {}
        
        for i, name in enumerate(names, 1):
            print(f"\n[{i}/{len(names)}] Processing '{name}'...")
            
            # Use unified download method that checks rare_molecules first
            filepath = self.download_molecule(name, prefer_3d)
                            
            results[name] = filepath
            
        return results


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(
        description="Download molecular structures from PubChem by name",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download single molecule (auto-converts to 3D if only 2D available)
  python molecule_downloader.py glucose
  
  # Download multiple molecules
  python molecule_downloader.py glucose aspirin caffeine benzene
  
  # Download from file (one name per line)
  python molecule_downloader.py --file molecules.txt
  
  # Force 2D only (no 3D conversion)
  python molecule_downloader.py glucose --2d
  
  # Specify output directory
  python molecule_downloader.py glucose --output ./my_molecules
        """
    )
    
    parser.add_argument("names", nargs="*", help="Chemical names to download")
    parser.add_argument("-f", "--file", help="File containing chemical names (one per line)")
    parser.add_argument("-o", "--output", default="molecules", help="Output directory")
    parser.add_argument("--3d", action="store_true", dest="prefer_3d", 
                       help="Prefer 3D structures (default: True, auto-converts 2D to 3D if RDKit available)")
    parser.add_argument("--2d", action="store_true", dest="prefer_2d",
                       help="Prefer 2D structures (disables auto-conversion to 3D)")
    
    args = parser.parse_args()
    
    # Collect names
    names = list(args.names) if args.names else []
    
    if args.file:
        try:
            with open(args.file, 'r') as f:
                names.extend([line.strip() for line in f if line.strip()])
        except FileNotFoundError:
            print(f"Error: File '{args.file}' not found")
            sys.exit(1)
            
    if not names:
        print("Error: No chemical names provided")
        print("Use -h for help")
        sys.exit(1)
        
    # Determine 3D preference
    prefer_3d = not args.prefer_2d
    
    # Show configuration
    print("=" * 60)
    print("Molecule Downloader")
    print("=" * 60)
    print(f"Output directory: {args.output}")
    print(f"Prefer 3D: {prefer_3d}")
    print(f"Auto-convert 2D to 3D: {prefer_3d and HAS_RDKIT}")
    print(f"PubChemPy available: {HAS_PUBCHEMPY}")
    print(f"RDKit available: {HAS_RDKIT}")
    print(f"Molecules to download: {len(names)}")
    print("=" * 60)
    
    # Initialize downloader
    downloader = MoleculeDownloader(args.output)
    
    # Download molecules
    results = downloader.download_batch(names, prefer_3d)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    successful = sum(1 for v in results.values() if v is not None)
    print(f"Successfully downloaded: {successful}/{len(names)}")
    
    if successful < len(names):
        print("\nFailed downloads:")
        for name, path in results.items():
            if path is None:
                print(f"  ✗ {name}")
                
    print("\nAll files saved to:", os.path.abspath(args.output))


if __name__ == "__main__":
    main()