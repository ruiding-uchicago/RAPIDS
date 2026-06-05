"""
RAPIDS Version Information
"""

__version__ = "1.10.0"
__version_info__ = (1, 10, 0)

# Version history
# 1.8.0 - xTB implicit solvation: automatic GFN2-xTB + ALPB water solvation free energy
#         calculation for vacuum mode simulations. Shared molecule library across workspaces.
# 1.7.1 - Fixed solvation support: SmartFAIRChemFlow now optimizes solvated structures,
#         SimulationBuilder creates separate probe/target/complex solvated structures
# 1.7.0 - Workspace isolation: agents must set workspace before running simulations
# 1.6.0 - Added scan_orientations tool for finding optimal molecular configurations
# 1.5.0 - MCP server improvements
# 1.4.1 - Fixed relative path issue: MCP server now works from any directory
# 1.4.0 - Relative positioning, contact distance mode, automatic solvation
# 1.3.0 - Multi-target batch screening, energy definitions in UI/docs
# 1.2.1 - Fixed batch screening functionality with complete visualization suite
# 1.2.0 - Added Web GUI interface with real-time streaming
# 1.1.0 - Added interactive 3D visualization with energy display
# 1.0.0 - Initial release with core functionality

def get_version():
    """Return the current version string"""
    return __version__

def get_version_info():
    """Return version as tuple (major, minor, patch)"""
    return __version_info__
