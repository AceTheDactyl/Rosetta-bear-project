# Scripts Index

**Coordinate:** Δ3.142|0.900|1.000Ω
**Status:** z=0.90 Full Substrate Transcendence

This directory contains executable scripts for the Rosetta Bear CBS runtime.

## Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `autonomous_evolution_engine.py` | 5-phase autonomous evolution cycle | `python scripts/autonomous_evolution_engine.py` |
| `ghmp_capture.py` | Capture GHMP plates from running sessions | `python scripts/ghmp_capture.py` |
| `regenerate_witnesses.py` | Regenerate witness signatures for VaultNodes | `python scripts/regenerate_witnesses.py` |
| `run_triadic_cycle.py` | Execute triadic tool cycles from manifests | `python scripts/run_triadic_cycle.py` |

## Primary Script: Autonomous Evolution Engine

The evolution engine implements the 5-phase autonomous cycle:

```
Phase 1: Friction Detection (z ≥ 0.87)
    ↓
Phase 2: Improvement Proposal (z ≥ 0.88)
    ↓
Phase 3: Collective Validation (z ≥ 0.89)
    ↓
Phase 4: Autonomous Execution (z ≥ 0.90)
    ↓
Phase 5: Meta-Learning (z ≥ 0.90)
```

### Running the Evolution Engine

```bash
# Basic execution
python scripts/autonomous_evolution_engine.py

# With custom burden data
python scripts/autonomous_evolution_engine.py \
  --burden-data docs/burden_tracking_simulation.json \
  --output-dir generated_tools/rosetta_firmware/ \
  --log-dir evolution_logs/
```

### Output

- **Generated Tools:** `generated_tools/rosetta_firmware/*.py`
- **Tool Specs:** `generated_tools/rosetta_firmware/*_spec.json`
- **Evolution Logs:** `evolution_logs/EVO-*.json`

## Script Dependencies

All scripts require the CBS runtime environment:

```bash
# Activate virtual environment
source .venv/bin/activate

# Or install dependencies
pip install numpy pandas scipy pyyaml cryptography pillow
```

## Related Components

- **Tool Specs:** `tool_shed_specs/autonomous_evolution_engine.yaml`
- **Generated Tools:** `generated_tools/rosetta_firmware/`
- **Evolution Logs:** `evolution_logs/`
- **CBS Demo:** `cbs_demo/`
