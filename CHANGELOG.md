# Changelog

All notable changes to RAPIDS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] - 2025-09-09

## [1.3.0] - 2026-??-??

### Added
- Multitarget batch screening support across CLI and web UI
- Target selector and energy definitions displayed throughout GUI tabs
- Energy definition summaries in README and USER_MANUAL for quick reference

### Changed
- Batch comparison output JSON, reports, and plots now organized by substrate/target pairing
- Visualization streaming supports multiple target directories and tagged links

### Fixed
- Ensured quick test and single-run tabs expose consistent energy interpretation guidance

### Added
- 🌐 **Web GUI Interface** - Complete browser-based interface for RAPIDS
  - `web_gui.html` - Single-file HTML interface with all functionality
  - `web_server.py` - Flask-based backend server with streaming support
  - Real-time simulation output streaming via Server-Sent Events
  - Interactive 3D molecular visualization for each simulation
  - Batch screening with progress tracking and result visualization
  - Support for all 9 substrates (Graphene, BP, MoS2, Si, ZnO, MOFs)
  - Auto-generation of 3D visualizations for batch results
  - Quick test examples for immediate experimentation

### Features
- **Single Molecule Simulation Tab**
  - Intuitive molecule input (name/SMILES)
  - Visual substrate selection
  - Advanced parameter control
  - Real-time progress bar and terminal output
  - Automatic 3D visualization generation
  
- **Batch Screening Tab**
  - Multi-molecule input (manual or CSV upload)
  - Individual progress tracking for each molecule
  - Interactive results table with energy rankings
  - Comparison charts (bar graphs)
  - Per-molecule 3D visualization access
  
- **Results Display**
  - Adsorption energy
  - Interaction energy
  - Optimization steps
  - Runtime tracking
  - Substrate effects analysis

### Changed
- Enhanced visualization generation to support batch workflows
- Improved error handling with typo tolerance (caffine/caffeine)
- Better directory structure detection for complex naming patterns

### Fixed
- Proper handling of three-component interaction data
- Correct parsing of batch comparison results
- Visualization file generation for all batch molecules

## [1.1.0] - 2025-01-09

### Added
- Interactive 3D structure visualization with `visualize_structures.py`
- Energy display in visualization (shows interaction energies for each structure)
- Version tracking system with `version.py`
- Color-coded energy values (green for favorable, red for unfavorable)
- Multi-structure toggle in single HTML viewer
- Master index page for all visualizations
- Version display in reports and visualizations

### Changed
- Enhanced smart_report.txt to include version information
- Improved visualization with spin control, labels, and background toggle

### Fixed
- Spin toggle now properly starts and stops rotation
- Surface rendering no longer accumulates when switching styles
- Improved state management in visualizer

## [1.0.0] - 2024-09-05

### Initial Release
- Core RAPIDS functionality
- Smart FAIRChem workflow with auto box sizing
- Three-component system calculations
- Batch comparison and screening
- Automatic molecule download from PubChem
- Support for 2D materials and MOFs
- Smart optimization with continuation
- Comprehensive documentation and examples
