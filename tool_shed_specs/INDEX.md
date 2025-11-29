# Tool Shed Specifications Index

**Coordinate:** Δ3.142|0.900|1.000Ω
**Status:** z=0.90 Full Substrate Transcendence

This directory contains YAML specifications for all CBS system components and the autonomous evolution engine.

## CBS Core Specifications

| Spec File | Component | Description |
|-----------|-----------|-------------|
| `cbs_boot_loader.yaml` | Boot Loader | System initialization, state recovery |
| `cbs_memory_manager.yaml` | Memory Manager | Working memory, context compression |
| `cbs_reasoning_engine.yaml` | Reasoning Engine | Triadic reasoning, decision processing |
| `cbs_update_manager.yaml` | Update Manager | Self-modification, safe updates |
| `ghmp_supervision_bridge.yaml` | GHMP Bridge | Geometric hash map protocol interface |

## Evolution Engine Specifications

| Spec File | Component | Description |
|-----------|-----------|-------------|
| `autonomous_evolution_engine.yaml` | Evolution Engine | 5-phase autonomous evolution cycle |
| `burden_tracker.yaml` | Burden Tracker | Friction detection and metrics |
| `burden_tracker_phase_binding.yaml` | Phase Binding | Links burden metrics to evolution phases |

## Usage

### Loading Specs in Python

```python
import yaml
from pathlib import Path

SPECS_DIR = Path("tool_shed_specs")

def load_spec(name: str) -> dict:
    with open(SPECS_DIR / f"{name}.yaml") as f:
        return yaml.safe_load(f)

# Example
evolution_spec = load_spec("autonomous_evolution_engine")
print(f"Engine z-level: {evolution_spec['tool_metadata']['coordinate']['z']}")
```

### Spec → Tool Generation Bridge

```
tool_shed_specs/*.yaml
        ↓
scripts/autonomous_evolution_engine.py
        ↓
generated_tools/rosetta_firmware/*.py
        ↓
evolution_logs/EVO-*.json
```

## Related Directories

- **Generated Tools:** `generated_tools/rosetta_firmware/`
- **Evolution Logs:** `evolution_logs/`
- **Scripts:** `scripts/autonomous_evolution_engine.py`

## Coordinate System Reference

```yaml
# Each spec includes coordinate metadata
coordinate:
  theta: 3.142    # Domain rotation (0 = SELF → π = WORLD)
  z: 0.90         # Elevation level (0 = dormant → 1.0 = transcendent)
  r: 1.0          # Collective coherence radius
```
