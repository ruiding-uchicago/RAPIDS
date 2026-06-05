# FAIRChem Auto-UMA Simulation Toolkit - User Manual

**Version 1.10.0** - Relative Positioning & Contact Distance Mode!

## Overview

This is an intelligent molecular simulation toolkit based on FAIRChem UMA models, designed for calculating intermolecular interactions and adsorption energies. The system uses machine learning-accelerated quantum chemistry calculations to rapidly evaluate molecular interaction strengths in vacuum or on 2D material surfaces.

## Core Features

### 1. Automatic Molecule Download
- **Local rare_molecules collection**: Checks pre-optimized rare/complex molecules first
- **Direct structure retrieval from chemical names**: Input common names like "glucose", "caffeine"
- **Automatic 3D structure generation**: Converts 2D to 3D using RDKit if PubChem only has 2D
- **Supported formats**: SDF, XYZ, VASP, and other molecular file formats

### 2. Intelligent Structure Building
- **Automatic overlap avoidance**: Smart algorithms ensure reasonable intermolecular spacing
- **Supported substrates**:
  - Graphene
  - MoS2 (Molybdenum disulfide)
  - BP (Black phosphorus)
  - Si (Silicon)
  - ZnO (Zinc oxide)
  - vacuum (no substrate)

### 3. FAIRChem Energy Calculations
- **Uses UMA-S-1p1 model**: Universal Model for Atoms (UMA) developed by Meta AI
- **Calculation types**:
  - Molecular adsorption energy
  - Intermolecular interaction energy
  - Structure optimization

### 4. Smart Optimization Features
- **Automatic box sizing**: Calculates simulation box dimensions based on molecular size
- **Smart continuation**: Auto-continues when near convergence, stops when forces too large
- **Structure validation**: Detects abnormal bond lengths and large structural changes

## Quick Start

### Environment Setup

```bash
# Install required packages
pip install fairchem-core
pip install pubchempy

# For Web GUI (NEW in v1.2.0)
pip install flask flask-cors

# Optional: Install additional packages for full functionality
pip install rdkit scipy
```

### Minimal Configuration Run

Use an example config:

```bash
python smart_fairchem_flow.py example_configs/tutorials/01_simplest.json
```

Or create your own minimal config:

```json
{
  "probe": "glucose",
  "substrate": "Graphene"
}
```

Run simulation:

```bash
python smart_fairchem_flow.py config.json
```

Only 2 parameters needed - everything else is automatic!

## Example Configurations

The `example_configs/` folder contains ready-to-use configurations organized by complexity:

- **tutorials/** - Step-by-step learning examples (start here!)
  - Includes `05_rare_molecules.json` showing beta-CD from local collection
- **screening/** - Batch comparison for multiple molecules
- **advanced/** - Custom positioning and orientation
- **applications/** - Real research scenarios (PFAS, MOFs, CNT, etc.)
  - `beta_cd_pfas_encapsulation.json` - Beta-cyclodextrin with PFAS
  - `cnt_molecule_screening.json` - Carbon nanotube interactions

See `example_configs/README.md` for detailed descriptions.

## Configuration Parameters

### Required Parameters
- `probe`: Probe molecule name (primary study object)
- `substrate`: Substrate material or "vacuum"

### Optional Parameters

#### Basic Parameters
- `target`: Target molecule (for probe-target interaction studies)
- `task_name`: FAIRChem task type, default "omol". Options:
  - "omol": OMol25, has VV10 dispersion (recommended for molecular interactions)
  - "oc20": Open Catalyst 2020, for catalysis systems
  - "omat": OMat24, for materials/surfaces
- `box_size`: Simulation box dimensions, default "auto" for automatic calculation
- `probe_height`: Probe molecule height from substrate, default 2.5 Å (used with auto positioning)
- `target_height`: Target molecule height, default probe_height + 3.5 Å (used with auto positioning)
- `probe_target_distance`: Probe-target spacing, default 4.0 Å
  - Numeric value (e.g., `4.0`): Fixed distance in Å
  - `"contact"`: Automatically calculate van der Waals contact distance
- `fmax`: Optimization convergence criterion, default 0.05 eV/Å
- `max_steps`: Maximum optimization steps per run, default 100
- `max_continuations`: Maximum continuation attempts, default 3

#### Advanced Placement Parameters (NEW)
- `probe_position`: Custom probe placement, default "auto"
  - `"auto"`: Automatic centering above substrate
  - `[x, y, z]`: Absolute coordinates in Å
  - `{"x": 0.5, "y": 0.5, "z": 12.0}`: Fractional x,y (0-1), absolute z
  - `{"frac": [0.5, 0.5, 0.3]}`: All fractional coordinates
  - `{"cylindrical": {"r": 5.0, "theta": 45, "z": 12.0}}`: Cylindrical coordinates (useful for pores)
- `probe_orientation`: Custom probe orientation, default "auto"
  - `"auto"`: No rotation
  - `[rx, ry, rz]`: Euler angles in degrees
  - `{"axis": [x, y, z], "angle": degrees}`: Axis-angle rotation
  - `{"align": "flat"/"standing"/"tilted"}`: Preset orientations
- `target_position`: Custom target placement (same formats as probe_position, plus relative mode)
  - All formats from probe_position, plus:
  - `{"relative_to": "probe", "lateral_offset": 4.0, "vertical_offset": 1.0, "direction": "x"}`: Position relative to probe
    - `relative_to`: Reference molecule ("probe")
    - `lateral_offset`: Horizontal distance in Å (default: 0)
    - `vertical_offset`: Vertical distance in Å (default: 0)
    - `direction`: Offset direction - "x", "-x", "y", "-y", "radial", "random" (default: "x")
- `target_orientation`: Custom target orientation (same formats as probe_orientation)

### Complete Configuration Example

```json
{
  "probe": "glucose",
  "target": "caffeine",
  "substrate": "Graphene",
  "task_name": "omat",
  "box_size": "auto",
  "probe_height": 2.5,
  "target_height": 6.0,
  "probe_target_distance": 4.0,
  "fmax": 0.05,
  "max_steps": 100,
  "max_continuations": 3,
  "output_dir": "simulations",
  "device": "cpu"
}
```

## Output Files

Generated in `simulations/[run_name]/` directory:

### Structure Files
- `probe_vacuum.vasp`: Probe molecule in vacuum
- `target_vacuum.vasp`: Target molecule in vacuum (if applicable)
- `substrate_only.vasp`: Pure substrate structure
- `probe_substrate.vasp`: Probe molecule on substrate (initial)
- `probe_target_vacuum.vasp`: Probe-target complex in vacuum
- `probe_target_substrate.vasp`: Three-component system (initial)
- `*_optimized.vasp`: Optimized structures (from smart_fairchem_flow.py)
  - `probe_substrate_optimized.vasp`: Optimized probe on substrate
  - `probe_target_substrate_optimized.vasp`: Optimized three-component system

### Optimization Logs
- `*_opt_0.log`: Optimization trajectories for each structure

### Analysis Results
- `interactions.json`: Calculated interaction energies
- `smart_report.txt`: Complete calculation report

#### Energy Definitions (all in eV)
- **Adsorption energy** = E(probe + substrate) − E(probe) − E(substrate)
- **Interaction energy (vacuum)** = E(probe + target in vacuum) − E(probe) − E(target)
- **Total three-component interaction** = E(probe + target + substrate) − E(probe) − E(target) − E(substrate)

Negative values indicate the combined system is more stable than the separated molecules (favourable adsorption/interaction); positive values mean the configuration is unfavourable and likely to dissociate.

## Results Interpretation

### Adsorption Energy
```
E_ads = E(probe_substrate) - E(probe_vacuum) - E(substrate_only)
```
- **Negative values**: Stable adsorption, more negative = more stable
- **Positive values**: Unstable, possibly local minimum

### Interaction Energy (Vacuum)
```
E_int_vac = E(probe_target_vacuum) - E(probe_vacuum) - E(target_vacuum)
```
- **Negative values**: Attractive interaction
- **Positive values**: Repulsive interaction

### Target Adsorption to Adsorbed Probe (Three-Component)
```
E_ads = E(probe_target_substrate) - E(probe_substrate_optimized) - E(target_vacuum)
```
- **Negative values**: Favorable adsorption
- **Positive values**: Unfavorable adsorption
- This represents the realistic scenario where target adsorbs to an already-adsorbed probe

### Substrate Effect on Interaction
```
E_substrate_effect = E_ads - E_int_vac
```
- **Negative values**: Substrate enhances probe-target interaction
- **Positive values**: Substrate weakens probe-target interaction

## Smart Features

### 1. Automatic Box Sizing
- Analyzes molecular dimensions
- Adds 10 Å buffer zone
- Ensures minimum size 25×25×35 Å

### 2. Smart Continuation Logic
- Force < 2×target: Continue optimization
- Force > 0.5 eV/Å: Stop to avoid structural distortion
- Automatically determines if continuation worthwhile

### 3. Structure Validation
- RMSD > 3 Å: Warns of large structural changes
- Shortest bond < 0.8 Å: Warns of abnormally short bonds

## FAQ

### Python Beginner Questions

#### Q: What is "cd" and how do I use it?
A: "cd" means "change directory" - it's how you navigate folders in the command line:
- `cd Desktop` - goes to your Desktop folder
- `cd ..` - goes back one folder
- `cd` (by itself) - shows your current location
- If you get lost, close and reopen Anaconda Prompt to start fresh

#### Q: I get "python: command not found" error?
A: You need to use **Anaconda Prompt** (not regular Command Prompt):
1. Click Windows Start Menu
2. Type "Anaconda Prompt" 
3. Click to open it
4. Now try `python` command again

#### Q: How do I know if the simulation is running?
A: You'll see output like this:
```
Building structures...
SMART OPTIMIZATION
Optimizing probe_substrate...
```
The simulation can take 5-30 minutes. As long as you see new text appearing, it's working!

#### Q: Where are my results?
A: Results are saved in a `simulations` folder:
1. Look in the same folder where you ran the command
2. Open `simulations` folder
3. Find folder with your molecule names
4. The `.txt` files can be opened with Notepad

#### Q: I get "No such file or directory" error?
A: Make sure you're in the right folder:
1. After extracting RAPIDS, note where you saved it
2. In Anaconda Prompt, navigate there:
   - If on Desktop: `cd Desktop/RAPIDS-main`
   - If in Documents: `cd Documents/RAPIDS-main`

#### Q: Can I close the window while it's running?
A: No! Keep the Anaconda Prompt window open until you see "COMPLETED SUCCESSFULLY" or similar message.

#### Q: How do I stop a running simulation?
A: Press `Ctrl+C` (hold Ctrl key and press C) in the Anaconda Prompt window.

#### Q: I get "ModuleNotFoundError: No module named 'fairchem'"?
A: You need to install the packages first:
```bash
pip install fairchem-core
pip install pubchempy
```
Run these commands in Anaconda Prompt, then try your simulation again.

#### Q: How do I create my own simulation?
A: Create a simple text file with `.json` extension:
1. Open Notepad
2. Type:
   ```json
   {
     "probe": "water",
     "substrate": "Graphene"
   }
   ```
3. Save as `my_test.json` (important: select "All Files" in Save dialog)
4. Run: `python smart_fairchem_flow.py my_test.json`

#### Q: What molecules can I use?
A: You can use common chemical names like:
- Simple molecules: "water", "methane", "ethanol", "glucose"
- Drugs: "aspirin", "caffeine", "ibuprofen"
- Any molecule on PubChem - just use its common name!

#### Q: The download is stuck at "Downloading model..."?
A: The UMA model is large (>1GB). First time download can take 10-30 minutes depending on internet speed. This only happens once - future runs will be fast!

#### Q: Can I run this on my laptop?
A: Yes! RAPIDS works on regular laptops:
- Windows 10/11: ✅ Works great
- Mac: ✅ Works great  
- 8GB RAM minimum (16GB better)
- No special graphics card needed (uses CPU by default)

### Scientific Questions

#### Q: Why is adsorption energy positive?
A: Possible reasons:
1. Local minimum rather than global minimum
2. Poor initial positioning
3. Weak substrate-molecule interaction

### Q: What if optimization doesn't converge?
A: System automatically continues up to 3 times. If still not converged:
1. Check if initial structure is reasonable
2. Adjust `fmax` convergence criterion
3. Increase `max_continuations`

### Q: How to use GPU acceleration?
A: Set in configuration file:
```json
{
  "device": "cuda"
}
```

## Web GUI Interface (NEW in v1.2.0)

RAPIDS now includes a complete web-based graphical interface for easier interaction with all features!

### Starting the Web Interface

```bash
# Navigate to RAPIDS directory
cd RAPIDS

# Start the web server
python web_server.py
```

Then open your browser to: **http://localhost:5001**

### Web GUI Features

#### 1. Single Molecule Simulation
- **Intuitive Input**: Enter molecule names or SMILES directly
- **Visual Substrate Selection**: Click to choose from 9 available substrates
- **Real-time Progress**: Watch simulation output stream live
- **3D Visualization**: Automatic generation of interactive molecular structures
- **Advanced Parameters**: Easy access to all simulation settings

#### 2. Batch Screening
- **Multi-molecule Testing**: Input multiple molecules at once
- **CSV Upload**: Import molecule lists from files
- **Individual Progress Tracking**: Monitor each simulation separately
- **Interactive Results Table**: Sort by energy values
- **Comparison Charts**: Visual bar graphs for energy comparisons
- **Per-molecule 3D Views**: Access visualization for each result

#### 3. Quick Test Examples
Pre-configured examples for immediate testing:
- Water on Graphene
- Caffeine on MoS2
- Three-component systems

### Web Interface Advantages

1. **No Command Line Required**: Perfect for beginners
2. **Real-time Feedback**: See output as calculations run
3. **Visual Results**: Interactive 3D structures and charts
4. **Easy Comparison**: Side-by-side molecule screening
5. **One-click Export**: Download results and visualizations

### Browser Requirements

- Modern browser (Chrome, Firefox, Safari, Edge)
- JavaScript enabled
- No installation needed - runs locally

## Command Line Tools

### Single Simulation
```bash
python smart_fairchem_flow.py config.json

# Example
python smart_fairchem_flow.py example_configs/tutorials/01_simplest.json
```

### Batch Comparison (Screening)
```bash
python batch_comparison.py batch_config.json

# Example - single substrate
python batch_comparison.py example_configs/screening/sugar_screening.json

# Example - multiple substrates (NEW!)
python batch_comparison.py example_configs/applications/pfas_mof_comprehensive.json
```

**Multiple Substrates Support (NEW):**
- Use `"substrates": ["Sub1", "Sub2", "Sub3"]` for multiple substrates
- Use `"substrate": "Sub1"` for single substrate (backward compatible)
- Automatically tests all probe-substrate combinations
- Generates rankings for each substrate separately

### Batch Geometry Optimization
```bash
# Optimize only atomic positions (default)
python batch_opt.py /path/to/folder

# Optimize atoms and cell parameters (full relaxation)
python batch_opt.py /path/to/folder --relax-cell

# Optimize atoms and cell for 2D materials (maintain x/y ratio)
python batch_opt.py /path/to/folder --relax-cell --iso-2d
```

**Features:**
- Searches for all `.vasp` files in target folder
- Skips already optimized files (containing "_opted" in filename)
- Shows real-time optimization progress
- Saves optimized structures with descriptive suffixes:
  - `*_opted.vasp`: Atoms-only optimization
  - `*_opted_cell.vasp`: Full cell relaxation
  - `*_opted_cell_iso2d.vasp`: Isotropic 2D cell relaxation
- Generates `optimization_summary.txt` with energies and forces
- Supports cell optimization for periodic systems
- Special isotropic mode for 2D materials maintains x/y aspect ratio

**Cell Relaxation Modes:**
- **Default**: Only atomic positions optimized, cell fixed
- **--relax-cell**: Full cell optimization with all degrees of freedom
- **--relax-cell --iso-2d**: For 2D materials, maintains x/y ratio while allowing isotropic scaling

### Download Molecules
```bash
python molecule_downloader.py
# Enter molecule name to download
```

### Build Structures
```bash
python simulation_builder.py config.json
# Only builds structures, doesn't run optimization
```

## Advanced Placement Examples

### MOF Pore Center Placement
```json
{
  "probe": "PFOS",
  "substrate": "Cu_HHTP",
  "probe_position": {"frac": [0.5, 0.5, 0.4]},
  "probe_orientation": {"align": "flat"}
}
```

### Multiple Position Screening
```json
{
  "probe": "PFHxS",
  "substrate": "Co_HHTP",
  "probe_position": {"cylindrical": {"r": 0, "theta": 0, "z": 15.0}},
  "probe_orientation": [0, 0, 45]
}
```

### Precise Orientation Control
```json
{
  "probe": "PFDoDA",
  "substrate": "Ni_HHTP",
  "probe_position": [15.0, 15.0, 12.0],
  "probe_orientation": {"axis": [1, 0, 0], "angle": 90}
}
```

### Relative Positioning (NEW in v1.4.0)
Position target molecule relative to probe:
```json
{
  "probe": "benzene",
  "target": "water",
  "substrate": "Graphene",
  "target_position": {
    "relative_to": "probe",
    "lateral_offset": 4.0,
    "vertical_offset": 1.0,
    "direction": "x"
  }
}
```
This places water 4Å in +x direction and 1Å above benzene.

Available directions:
- `"x"`, `"-x"`: Along/against x-axis
- `"y"`, `"-y"`: Along/against y-axis
- `"radial"`: Random angle from probe center
- `"random"`: Same as radial

### Contact Distance Mode (NEW in v1.4.0)
Automatically place molecules at van der Waals contact distance:
```json
{
  "probe": "benzene",
  "target": "water",
  "substrate": "vacuum",
  "probe_target_distance": "contact"
}
```
The system calculates the optimal approach distance based on atomic radii (Mantina 2009).

### Automatic Solvation (NEW in v1.4.0)
Add explicit solvent molecules around the probe/target cluster:

**Auto mode** (system calculates water count based on cluster size):
```json
{
  "probe": "aspirin",
  "target": "caffeine",
  "substrate": "vacuum",
  "solvation": {
    "solvent": "water",
    "mode": "auto",
    "shell_thickness": 3.0,
    "density": 1.0
  }
}
```

**Manual mode** (specify exact count):
```json
{
  "probe": "glucose",
  "substrate": "vacuum",
  "solvation": {
    "solvent": "water",
    "count": 20
  }
}
```

Solvation parameters:
- `solvent`: Solvent molecule name (default: "water")
- `mode`: "auto" for automatic count calculation
- `count`: Manual solvent count (overrides auto)
- `shell_thickness`: Solvation shell thickness in Å (default: 3.0)
- `density`: Density factor, 1.0 = normal density (default: 1.0)

Output:
```
Solvation Analysis:
  Cluster size: 11.4 x 9.9 x 6.5 Å
  Cluster radius: 4.6 Å
  Estimated solvent count: 62 (auto, shell=3.0 Å)
  Successfully placed: 62 water molecules
```

## Rare Molecules Collection

The `rare_molecules/` folder contains pre-optimized 3D structures for complex molecules that are difficult to obtain from PubChem:

- **beta_cd.sdf** - Beta-cyclodextrin (optimized 3D structure)
- **CNT.sdf** - H-capped carbon nanotube

### How It Works

1. When you specify a molecule name, the system first checks `rare_molecules/`
2. If found, it uses the pre-optimized structure
3. If not found, it downloads from PubChem as usual

### Adding Your Own Rare Molecules

Simply place your optimized `.sdf` files in the `rare_molecules/` folder with descriptive names:
- Use lowercase with underscores: `molecule_name.sdf`
- The system will automatically find them when referenced

### Example Usage

```json
{
  "probe": "beta-CD",
  "target": "PFOS",
  "substrate": "Graphene"
}
```

In this example:
- `beta-CD` will be loaded from `rare_molecules/beta_cd.sdf`
- `PFOS` will be downloaded from PubChem
- Both work seamlessly together

## Advanced Usage

### Batch Calculations
Create multiple configuration files and run with script:

```bash
# Run all tutorials
for config in example_configs/tutorials/*.json; do
    python smart_fairchem_flow.py "$config"
done

# Or use batch_comparison.py for screening
python batch_comparison.py example_configs/screening/sugar_screening.json
```

### Custom Substrates
Add new materials in the `create_substrate()` method in `simulation_builder.py`.

### Model Selection
Modify `model_name` to use other FAIRChem models:
- uma-s-1p1 (default, most stable)
- equiformer_v2 (high accuracy)
- schnet (fast)

## Technical Details

### Dependencies
- **FAIRChem-Core**: ML-accelerated quantum chemistry (`pip install fairchem-core`)
- **PubChemPy**: Molecular database interface (`pip install pubchempy`)
- **ASE**: Atomic Simulation Environment (included with fairchem-core)
- **RDKit**: Molecular processing and 3D generation (optional, for 2D→3D conversion)
- **NumPy**: Numerical computing (included with fairchem-core)
- **SciPy**: Scientific computing for advanced rotations (optional)

### Periodic Boundary Conditions
All structures use full PBC [True, True, True] for VASP/FAIRChem compatibility.

### Energy Reference States
Vacuum and substrate calculations use identical box sizes to ensure consistent energy reference states.

## Troubleshooting

### Error: Inconsistent PBC settings
Ensure all structures have PBC set to [True, True, True]

### Error: Molecule not found
1. Check chemical name spelling
2. Try using IUPAC name
3. Manually download SDF file

### Warning: Large structural changes
1. Check initial geometry
2. Reduce optimization step size
3. Use stricter convergence criteria

## Contact & Support

If encountering issues, check:
1. Configuration file format is correct
2. Environment is properly activated
3. Review warnings in smart_report.txt

## License

This tool is based on the FAIRChem open-source project and follows its license agreement.
