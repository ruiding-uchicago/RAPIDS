# RAPIDS: Rapid Atomistic Probe-target Interaction Discovery Scaffold

<div align="center">
  <img src="icon.png" alt="RAPIDS Logo" width="200"/>
  
  **RAPIDS** - **R**apid **A**tomistic **P**robe-target **I**nteraction **D**iscovery **S**caffold
  
  *ML-accelerated molecular interaction calculations using FAIRChem's Universal Model for Atoms (UMA)*
</div>

---

## 🚀 First Time Using Python? Start Here!

**New to Python?** No worries! Here's the simplest way to get started:

1. **Download Anaconda** (easiest for beginners): 
   - Go to [anaconda.com/download](https://www.anaconda.com/download)
   - Download the installer for your system (Windows/Mac/Linux)
   - Run the installer (default settings are fine!)

2. **Open Anaconda Prompt** (Windows) or Terminal (Mac/Linux):
   - Windows: Search "Anaconda Prompt" in Start Menu
   - Mac/Linux: Open Terminal

3. **Copy-paste these commands** (one at a time):
   ```bash
   pip install fairchem-core
   pip install pubchempy
   ```

4. **Download RAPIDS**:
   - Click the green "Code" button above → "Download ZIP"
   - Extract to your Desktop or Documents folder

5. **Run your first simulation**:
   ```bash
   cd Desktop/RAPIDS-main
   python smart_fairchem_flow.py example_configs/tutorials/01_simplest.json
   ```

📺 **Need visual help?** Watch: [How to Install Anaconda on Windows](https://www.youtube.com/results?search_query=install+anaconda+windows+2024)

---

## Overview

RAPIDS is designed for researchers with minimal computational chemistry background to perform quick, qualitative dry-lab simulations of probe-target-substrate interactions. Only 2 parameters required to start!

## 🌐 NEW: Web GUI Interface

**Run RAPIDS from your browser!** No command line needed:

```bash
# Start the web server
python web_server.py

# Open your browser to http://localhost:5001
```

The web interface provides:
- 🎯 **Intuitive molecule input** - Type molecule names or SMILES
- 📊 **Real-time progress tracking** - Watch simulations as they run
- 🔬 **Interactive 3D visualization** - Rotate and zoom molecular structures
- 📈 **Batch screening** - Compare multiple molecules at once
- 💾 **One-click downloads** - Get results in JSON, VASP, or report format

## Features

- **Automatic molecule download** from chemical names (PubChem + rare molecules collection)
- **Smart optimization** with auto-continuation and structure validation
- **Batch screening** - Compare multiple molecules and rank by adsorption affinity
- **Three-component system** - Calculates target adsorption to already-adsorbed probe on substrate
- **Advanced placement** - Custom molecular positioning for MOF pores, etc.
- **Cross-platform** - Works on Mac, Linux, and Windows
- **Minimum hardware required** - User can use cuda, or default CPU for edge device users.

## Quick Start

### Prerequisites
1. **Get UMA model access**: Register at [HuggingFace](https://huggingface.co/facebook/UMA) and request access of the checkpoints
2. **Download model** (one-time): Run `python download_model.py` and enter your HuggingFace token

Or manually:
```python
from huggingface_hub import login
from fairchem.core import pretrained_mlip
login(token="hf_YOUR_TOKEN_HERE")  # Get token from HuggingFace settings
model = pretrained_mlip.get_predict_unit("uma-s-1p1")
```

### Setup Environment
```bash
# Install required packages
pip install fairchem-core
pip install pubchempy
```

### Run Simulations
```bash
# Run simplest example (just 2 parameters!)
python smart_fairchem_flow.py example_configs/tutorials/01_simplest.json
```

Or create your own minimal config:
```json
{
  "probe": "glucose",
  "substrate": "Graphene"
}
```

### DFT escalation: ORCA (default) or GPU4PySCF

The SP and GeoSP runners now use **ORCA by default**. Their historical
`*_gpu.py` filenames are retained for compatibility. On macOS they look for
ORCA 6.1.1 at `~/Library/orca_6_1_1/orca` and OpenMPI at
`~/Library/openmpi-4.1.1`; override these with `--orca-executable` and
`--orca-openmpi-root` (or `RAPIDS_ORCA_EXE` / `RAPIDS_OPENMPI_ROOT`).

```bash
# Single-point arm; add --probe and --target to obtain a binding energy
python pyscf_opt_tests/run_sp_gpu.py \
  --functional pbe-d3bj --complex complex.xyz

# Geometry optimization (def2-TZVP) followed by def2-TZVPD single point
python pyscf_opt_tests/run_geomopt_gpu.py \
  --functional pbe-d3bj --complex complex.xyz --orca-nprocs 4

# Optional CUDA backend
pip install -r requirements-gpu4pyscf.txt
python pyscf_opt_tests/run_sp_gpu.py \
  --backend gpu4pyscf --functional pbe-d3bj --complex complex.xyz
```

Available functionals are `pbe-d3bj`, `wb97x-d3bj`, and `wb97m-v`; use
`--functional all` to run all three. ORCA wB97M-V jobs use `SCNL` so the VV10
nonlocal correlation is included self-consistently, matching the GPU4PySCF arm.

### Backbone selection (UMA / MACE-omol / ORB-omol)

RAPIDS runs on the FAIRChem **UMA omol** head by default. The relaxation/scoring
backbone is pluggable via `backbone_factory.get_backbone_calculator(...)`, so the
core flow can instead use **MACE-omol** or **ORB-omol** (the alternative backbones
from the paper's cross-backbone diagnostic). Select one of three ways
(precedence: config key > env var > default `"uma"`):

```json
{
  "probe": "glucose",
  "target": "caffeine",
  "backbone": "uma"          // "uma" (default) | "mace-omol" | "orb-omol"
}
```
```bash
RAPIDS_BACKBONE=orb-omol python smart_fairchem_flow.py my_config.json
```

The default UMA path is unchanged. Alternative backbones need their own package,
imported only when that backbone is selected:

| backbone     | extra package | install                     | default checkpoint |
|--------------|---------------|-----------------------------|--------------------|
| `uma`        | fairchem-core | `pip install fairchem-core` | `uma-s-1p2`         |
| `mace-omol`  | MACE          | `pip install mace-torch`    | `omol` (`mace_off`) |
| `orb-omol`   | orb-models    | `pip install orb-models`    | `orb-omol`          |

Override the checkpoint with the `model_name` config key. Selecting a backbone
whose package is not installed raises a clear `ImportError` with the pip hint.

## Batch Screening

Compare multiple molecules to find the strongest adsorbate:

```bash
# Screen sugars interaction with caffeine
python batch_comparison.py example_configs/screening/sugar_screening.json
```

Multi-substrate support:
```json
{
  "probes": ["PFHxS", "PFOS", "PFDoDA"],
  "substrates": ["Co_HHTP", "Cu_HHTP", "Ni_HHTP"]
}
```

## File Structure

```
smart_fairchem_flow.py    # Single molecule simulation
batch_comparison.py        # Multi-probe screening  
batch_opt.py              # Batch geometry optimization
simulation_builder.py      # Structure generation
molecule_downloader.py     # Molecule retrieval

example_configs/          # Ready-to-use examples
├── tutorials/            # Beginner (use smart_fairchem_flow.py)
├── screening/            # Batch screening (use batch_comparison.py) 
├── advanced/             # Custom positioning (use smart_fairchem_flow.py)
└── applications/         # Research cases (mixed usage)

rare_molecules/           # Complex molecules (beta-CD, CNT)
substrate/                # 2D materials & MOFs
molecules/                # Downloaded molecules (auto-created)
simulations/              # Results (auto-created)
```

## Supported Substrates

**2D Materials:** Graphene, MoS2, BP, Si, ZnO  
**MOFs:** Co_HHTP, Cu_HHTP, Ni_HHTP  
**Special:** vacuum (no substrate)

## Output

Results in `simulations/[run_name]/`:
- Optimized structures (`.vasp`)
  - `probe_substrate_optimized.vasp` - Probe on substrate
  - `probe_target_substrate_optimized.vasp` - Three-component system
- Interaction energies (`interactions.json`)
  - Probe adsorption energy
  - Target adsorption to adsorbed probe
  - Substrate effect on interaction
- Analysis report (`smart_report.txt`)

### Energy Definitions (all values in eV)
- **Adsorption energy** = E(probe+substrate) − E(probe) − E(substrate)
- **Interaction energy (vacuum)** = E(probe+target in vacuum) − E(probe) − E(target)
- **Total three-component interaction** = E(probe+target+substrate) − E(probe) − E(target) − E(substrate)

Negative values indicate the process releases energy (favorable adsorption/interaction); positive values mean the configuration is energetically unfavourable compared to separated components.

## Documentation

See [USER_MANUAL.md](USER_MANUAL.md) for detailed parameters and troubleshooting.

## License

Based on [FAIRChem](https://github.com/FAIR-Chem/fairchem).
