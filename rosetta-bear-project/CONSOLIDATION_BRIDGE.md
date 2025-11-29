# Rosetta Bear Project - Consolidation Bridge

**Status:** LEGACY - Root Level Preferred

This subdirectory contains the original Rosetta Bear project structure. As of z=0.90, the CBS runtime components have been elevated to root level for unified access.

## Structure Mapping

| Legacy (rosetta-bear-project/) | Root Level |
|-------------------------------|------------|
| `cbs_boot_loader.py` | `cbs_boot_loader.py` |
| `cbs_memory_manager.py` | `cbs_memory_manager.py` |
| `cbs_reasoning_engine.py` | `cbs_reasoning_engine.py` |
| `cbs_update_manager.py` | `cbs_update_manager.py` |
| `cbs_interactive_demo.py` | `cbs_interactive_demo.py` |
| `ghmp.py` | `ghmp.py` |
| `generated_tools/triadic_rhz/` | `generated_tools/triadic_rhz/` |
| `tool_shed_specs/` | `tool_shed_specs/` |
| `scripts/` | `scripts/` |
| `cbs_demo/` | `cbs_demo/` |

## Usage

**Preferred:** Use root-level files for all CBS operations.

```python
# Recommended (root level)
from cbs_boot_loader import CBSBootLoader
from generated_tools.rosetta_firmware import ROSETTA_FIRMWARE_TOOLS

# Legacy (still functional but deprecated)
# import sys; sys.path.insert(0, 'rosetta-bear-project')
# from cbs_boot_loader import CBSBootLoader
```

## Why Two Directories?

The `rosetta-bear-project/` subdirectory preserves:

1. Original project structure for reference
2. Additional `docs/` specific to RHZ firmware publishing
3. Isolated `cbs_demo/` state for clean experimentation
4. `requirements.txt` for standalone installation

## Consolidation Notes

- Root-level CBS files are the **authoritative source**
- This subdirectory is maintained for backwards compatibility
- New development should use root-level structure
- z=0.90 state transfer package references root-level paths

## Bridge Registry

See `/bridge_registry.yaml` for complete cross-component mapping.
