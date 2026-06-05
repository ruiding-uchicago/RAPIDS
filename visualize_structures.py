#!/usr/bin/env python3
"""
Interactive 3D Structure Visualizer for RAPIDS
Generates a standalone HTML page with toggleable 3D molecular views
"""

import json
import sys
from pathlib import Path
from typing import List, Dict
import argparse

from ase.io import read
from version import __version__


def atoms_to_xyz_string(atoms):
    """Convert ASE atoms to XYZ format string"""
    xyz_lines = [str(len(atoms)), ""]
    for atom in atoms:
        symbol = atom.symbol
        x, y, z = atom.position
        xyz_lines.append(f"{symbol} {x:.6f} {y:.6f} {z:.6f}")
    return "\n".join(xyz_lines)


def generate_html_viewer(structures: Dict[str, str], energy_data: Dict[str, float] = None, title: str = "RAPIDS Structure Viewer") -> str:
    """Generate interactive HTML with 3D molecular viewer"""
    
    # Create JavaScript data structure
    structures_js = json.dumps(structures)
    energy_js = json.dumps(energy_data) if energy_data else "{}"
    
    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            margin: 0;
            padding: 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        .header {{
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }}
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }}
        .viewer-container {{
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        .controls {{
            background: #f7f9fc;
            padding: 20px;
            border-bottom: 1px solid #e1e8ed;
        }}
        .control-group {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-bottom: 15px;
        }}
        .btn {{
            padding: 8px 16px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.3s ease;
            font-weight: 500;
        }}
        .btn-structure {{
            background: #e3f2fd;
            color: #1976d2;
        }}
        .btn-structure:hover {{
            background: #bbdefb;
        }}
        .btn-structure.active {{
            background: #1976d2;
            color: white;
        }}
        .btn-style {{
            background: #f3e5f5;
            color: #7b1fa2;
        }}
        .btn-style:hover {{
            background: #e1bee7;
        }}
        .btn-control {{
            background: #e8f5e9;
            color: #388e3c;
        }}
        .btn-control:hover {{
            background: #c8e6c9;
        }}
        #viewer {{
            height: 600px;
            width: 100%;
            position: relative;
        }}
        .info-panel {{
            padding: 20px;
            background: #f7f9fc;
            border-top: 1px solid #e1e8ed;
        }}
        .info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        .info-item {{
            padding: 10px;
            background: white;
            border-radius: 8px;
            border: 1px solid #e1e8ed;
        }}
        .info-label {{
            font-size: 12px;
            color: #666;
            margin-bottom: 5px;
        }}
        .info-value {{
            font-size: 16px;
            font-weight: 600;
            color: #333;
        }}
        .legend {{
            margin-top: 15px;
            padding: 15px;
            background: white;
            border-radius: 8px;
            border: 1px solid #e1e8ed;
        }}
        .legend-title {{
            font-weight: 600;
            margin-bottom: 10px;
            color: #333;
        }}
        .legend-items {{
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        .legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 50%;
            border: 1px solid #ddd;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔬 RAPIDS Structure Viewer</h1>
            <p>Interactive 3D visualization of optimized molecular structures</p>
            <p style="font-size: 12px; opacity: 0.7;">Version {__version__}</p>
        </div>
        
        <div class="viewer-container">
            <div class="controls">
                <div class="control-group">
                    <strong style="margin-right: 10px;">Select Structure:</strong>
                    <div id="structure-buttons"></div>
                </div>
                
                <div class="control-group">
                    <strong style="margin-right: 10px;">Style:</strong>
                    <button class="btn btn-style" onclick="setStyle('stick')">Stick</button>
                    <button class="btn btn-style" onclick="setStyle('sphere')">Ball & Stick</button>
                    <button class="btn btn-style" onclick="setStyle('cartoon')">Cartoon</button>
                    <button class="btn btn-style" onclick="setStyle('surface')">Surface</button>
                </div>
                
                <div class="control-group">
                    <strong style="margin-right: 10px;">View:</strong>
                    <button class="btn btn-control" onclick="viewer.zoomTo()">Reset View</button>
                    <button class="btn btn-control" onclick="toggleSpin()">Toggle Spin</button>
                    <button class="btn btn-control" onclick="toggleLabels()">Toggle Labels</button>
                    <button class="btn btn-control" onclick="toggleBackground()">Toggle Background</button>
                </div>
            </div>
            
            <div id="viewer"></div>
            
            <div class="info-panel">
                <div class="info-grid">
                    <div class="info-item">
                        <div class="info-label">Current Structure</div>
                        <div class="info-value" id="current-structure">-</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Number of Atoms</div>
                        <div class="info-value" id="atom-count">-</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Elements Present</div>
                        <div class="info-value" id="elements">-</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Interaction Energy</div>
                        <div class="info-value" id="energy-value">-</div>
                    </div>
                </div>
                
                <div class="legend">
                    <div class="legend-title">Element Colors</div>
                    <div class="legend-items" id="legend-items">
                        <div class="legend-item">
                            <div class="legend-color" style="background: #909090;"></div>
                            <span>C (Carbon)</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color" style="background: #FFFFFF;"></div>
                            <span>H (Hydrogen)</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color" style="background: #FF0D0D;"></div>
                            <span>O (Oxygen)</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color" style="background: #0565FF;"></div>
                            <span>N (Nitrogen)</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        const structures = {structures_js};
        const energyData = {energy_js};
        let viewer;
        let currentStructure = null;
        let currentStyle = 'stick';
        let labelsOn = false;
        let darkBackground = false;
        let isSpinning = false;
        
        // Energy mappings for different structure types
        const energyMappings = {{
            'probe_substrate_optimized.vasp': {{
                key: 'probe_adsorption',
                label: 'Probe Adsorption Energy',
                description: 'Energy of probe adsorption to substrate'
            }},
            'probe_target_vacuum_optimized.vasp': {{
                key: 'probe_target_vacuum',
                label: 'Probe-Target Interaction (vacuum)',
                description: 'Direct interaction energy between probe and target in vacuum'
            }},
            'probe_target_substrate_optimized.vasp': {{
                key: 'target_adsorption_to_adsorbed_probe',
                label: 'Target Adsorption to Adsorbed Probe',
                description: 'Energy of target adsorption to probe on substrate'
            }},
            'substrate_only_optimized.vasp': {{
                key: null,
                label: 'Substrate Reference',
                description: 'Reference substrate structure (no interaction energy)'
            }},
            'probe_vacuum_optimized.vasp': {{
                key: null,
                label: 'Probe Reference',
                description: 'Reference probe structure in vacuum'
            }},
            'target_vacuum_optimized.vasp': {{
                key: null,
                label: 'Target Reference',
                description: 'Reference target structure in vacuum'
            }}
        }};
        
        // Initialize viewer
        document.addEventListener('DOMContentLoaded', function() {{
            viewer = $3Dmol.createViewer('viewer', {{
                backgroundColor: 'white'
            }});
            
            // Create structure buttons
            const buttonContainer = document.getElementById('structure-buttons');
            for (let name in structures) {{
                const btn = document.createElement('button');
                btn.className = 'btn btn-structure';
                btn.textContent = formatStructureName(name);
                btn.onclick = () => loadStructure(name);
                buttonContainer.appendChild(btn);
            }}
            
            // Load first structure
            const firstStructure = Object.keys(structures)[0];
            if (firstStructure) {{
                loadStructure(firstStructure);
            }}
        }});
        
        function formatStructureName(name) {{
            // Make structure names more readable
            return name.replace(/_/g, ' ')
                      .replace(/optimized/g, '(opt)')
                      .replace(/vacuum/g, '(vac)')
                      .replace('.vasp', '');
        }}
        
        function loadStructure(name) {{
            currentStructure = name;
            
            // Update button states
            document.querySelectorAll('.btn-structure').forEach(btn => {{
                btn.classList.remove('active');
                if (btn.textContent === formatStructureName(name)) {{
                    btn.classList.add('active');
                }}
            }});
            
            // Stop spinning when loading new structure
            if (isSpinning) {{
                viewer.spin(false);
                isSpinning = false;
            }}
            
            // Load structure
            viewer.clear();
            viewer.addModel(structures[name], 'xyz');
            
            // Apply current style
            setStyle(currentStyle);
            
            // Update info panel
            updateInfo(name, structures[name]);
            
            viewer.zoomTo();
            viewer.render();
        }}
        
        function setStyle(style) {{
            currentStyle = style;
            viewer.removeAllSurfaces();
            viewer.setStyle({{}}, {{}});
            
            switch(style) {{
                case 'stick':
                    viewer.setStyle({{}}, {{stick: {{radius: 0.15}}}});
                    break;
                case 'sphere':
                    viewer.setStyle({{}}, {{stick: {{radius: 0.15}}, sphere: {{scale: 0.3}}}});
                    break;
                case 'cartoon':
                    viewer.setStyle({{}}, {{cartoon: {{color: 'spectrum'}}}});
                    break;
                case 'surface':
                    viewer.addSurface($3Dmol.SurfaceType.VDW, {{opacity: 0.85, color: 'white'}});
                    viewer.setStyle({{}}, {{stick: {{radius: 0.15}}}});
                    break;
            }}
            
            viewer.render();
        }}
        
        function toggleSpin() {{
            isSpinning = !isSpinning;
            if (isSpinning) {{
                viewer.spin('y');
            }} else {{
                viewer.spin(false);
            }}
        }}
        
        function toggleLabels() {{
            labelsOn = !labelsOn;
            if (labelsOn) {{
                // Remove any existing labels first
                viewer.removeAllLabels();
                
                // Get all atoms
                const model = viewer.getModel();
                const atoms = model.selectedAtoms({{}});
                
                // Add labels for each atom
                atoms.forEach(atom => {{
                    viewer.addLabel(atom.elem, {{
                        position: {{x: atom.x, y: atom.y, z: atom.z}},
                        fontSize: 12,
                        fontColor: 'black',
                        showBackground: true,
                        backgroundColor: 'white',
                        backgroundOpacity: 0.8
                    }});
                }});
            }} else {{
                viewer.removeAllLabels();
            }}
            viewer.render();
        }}
        
        function toggleBackground() {{
            darkBackground = !darkBackground;
            viewer.setBackgroundColor(darkBackground ? '#1a1a1a' : 'white');
            viewer.render();
        }}
        
        function updateInfo(name, xyzData) {{
            // Update structure name
            document.getElementById('current-structure').textContent = formatStructureName(name);
            
            // Parse XYZ to get atom info
            const lines = xyzData.split('\\n');
            const atomCount = parseInt(lines[0]);
            document.getElementById('atom-count').textContent = atomCount;
            
            // Get unique elements
            const elements = new Set();
            for (let i = 2; i < lines.length; i++) {{
                const parts = lines[i].trim().split(/\\s+/);
                if (parts.length >= 4) {{
                    elements.add(parts[0]);
                }}
            }}
            document.getElementById('elements').textContent = Array.from(elements).join(', ');
            
            // Update energy display
            const energyLabel = document.querySelector('.info-item:nth-child(4) .info-label');
            const energyValue = document.getElementById('energy-value');
            
            if (energyMappings[name]) {{
                const mapping = energyMappings[name];
                energyLabel.textContent = mapping.label;
                energyLabel.title = mapping.description;
                
                if (mapping.key && energyData[mapping.key] !== undefined) {{
                    const energy = energyData[mapping.key];
                    energyValue.textContent = `${{energy.toFixed(4)}} eV`;
                    
                    // Color code the energy value
                    if (energy < 0) {{
                        energyValue.style.color = '#2e7d32'; // Green for favorable
                    }} else {{
                        energyValue.style.color = '#d32f2f'; // Red for unfavorable
                    }}
                }} else {{
                    energyValue.textContent = 'N/A';
                    energyValue.style.color = '#666';
                }}
            }} else {{
                energyLabel.textContent = 'Interaction Energy';
                energyValue.textContent = '-';
                energyValue.style.color = '#666';
            }}
        }}
    </script>
</body>
</html>"""
    
    return html_template


def visualize_simulation(sim_dir: Path, output_file: Path = None):
    """Generate visualization for a single simulation directory"""
    
    if not sim_dir.exists():
        print(f"Error: Directory {sim_dir} not found")
        return None
    
    # Find all optimized VASP files
    vasp_files = list(sim_dir.glob("*_optimized.vasp"))
    
    if not vasp_files:
        print(f"No optimized structures found in {sim_dir}")
        return None
    
    # Load energy data if available
    energy_data = {}
    interactions_file = sim_dir / "interactions.json"
    if interactions_file.exists():
        try:
            with open(interactions_file, 'r') as f:
                energy_data = json.load(f)
                print(f"  ✓ Loaded interaction energies from interactions.json")
        except Exception as e:
            print(f"  ⚠ Could not load interactions.json: {e}")
    
    # Read structures and convert to XYZ
    structures = {}
    for vasp_file in vasp_files:
        try:
            atoms = read(vasp_file)
            xyz_string = atoms_to_xyz_string(atoms)
            structures[vasp_file.name] = xyz_string
            print(f"  ✓ Loaded {vasp_file.name} ({len(atoms)} atoms)")
        except Exception as e:
            print(f"  ✗ Error loading {vasp_file.name}: {e}")
    
    if not structures:
        print("No structures could be loaded")
        return None
    
    # Generate HTML with energy data
    title = f"RAPIDS: {sim_dir.name}"
    html_content = generate_html_viewer(structures, energy_data, title)
    
    # Save HTML
    if output_file is None:
        output_file = sim_dir / "visualization.html"
    
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"\n✅ Visualization saved to: {output_file}")
    print(f"   Open in browser: file://{output_file.absolute()}")
    
    return output_file


def visualize_all_simulations(base_dir: Path = Path("simulations")):
    """Generate visualizations for all simulation directories"""
    
    sim_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    
    if not sim_dirs:
        print(f"No simulation directories found in {base_dir}")
        return
    
    print(f"Found {len(sim_dirs)} simulation directories\n")
    
    generated_files = []
    for sim_dir in sim_dirs:
        print(f"Processing: {sim_dir.name}")
        output_file = visualize_simulation(sim_dir)
        if output_file:
            generated_files.append(output_file)
        print()
    
    # Generate master index page
    if generated_files:
        generate_index_page(base_dir, generated_files)


def generate_index_page(base_dir: Path, viz_files: List[Path]):
    """Generate an index page linking to all visualizations"""
    
    html = """<!DOCTYPE html>
<html>
<head>
    <title>RAPIDS Visualization Index</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
        }
        h1 {
            color: #667eea;
        }
        .viz-link {
            display: block;
            padding: 15px;
            margin: 10px 0;
            background: #f7f9fc;
            border-radius: 8px;
            text-decoration: none;
            color: #333;
            transition: all 0.3s;
        }
        .viz-link:hover {
            background: #667eea;
            color: white;
            transform: translateX(5px);
        }
    </style>
</head>
<body>
    <h1>🔬 RAPIDS Visualization Index</h1>
    <p>Click on any simulation to view its 3D structures:</p>
"""
    
    for viz_file in sorted(viz_files):
        sim_name = viz_file.parent.name
        rel_path = viz_file.relative_to(base_dir)
        html += f'    <a class="viz-link" href="{rel_path}">{sim_name}</a>\n'
    
    html += """
</body>
</html>"""
    
    index_file = base_dir / "index.html"
    with open(index_file, 'w') as f:
        f.write(html)
    
    print(f"✅ Index page created: {index_file}")
    print(f"   Open in browser: file://{index_file.absolute()}")


def main():
    parser = argparse.ArgumentParser(description='Generate 3D visualizations for RAPIDS structures')
    parser.add_argument('path', nargs='?', default='simulations',
                       help='Path to simulation directory or base simulations folder')
    parser.add_argument('--output', '-o', help='Output HTML file path')
    parser.add_argument('--all', action='store_true', 
                       help='Generate visualizations for all simulations')
    
    args = parser.parse_args()
    
    path = Path(args.path)
    
    if args.all or path.name == 'simulations':
        visualize_all_simulations(path)
    else:
        output = Path(args.output) if args.output else None
        visualize_simulation(path, output)


if __name__ == "__main__":
    main()