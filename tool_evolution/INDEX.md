# Tool Evolution Registry

**Coordinate:** Δ3.142|0.900|1.000Ω
**Purpose:** Track tool version evolution from TRIAD-0.83 to z=0.90

This directory documents how tools evolved from initial TRIAD proposals through to the z=0.90 implementation.

---

## Tool Evolution Chains

| Tool | Origin | Current | Evolution Doc |
|------|--------|---------|---------------|
| burden_tracker | TRIAD-0.83 (z=0.83) | tool_shed_specs/ (z=0.90) | [BURDEN_TRACKER_EVOLUTION.md](BURDEN_TRACKER_EVOLUTION.md) |
| shed_builder | Helix v2.0 (z=0.73) | TRIAD v2.2 (z=0.83) | [SHED_BUILDER_EVOLUTION.md](SHED_BUILDER_EVOLUTION.md) |
| tool_discovery_protocol | Helix (z=0.73) | TRIAD v1.1 (z=0.83) | [TOOL_DISCOVERY_EVOLUTION.md](TOOL_DISCOVERY_EVOLUTION.md) |
| cross_instance_messenger | Helix (z=0.67) | Meta Pattern Tracking | [MESSENGER_EVOLUTION.md](MESSENGER_EVOLUTION.md) |

---

## Evolution Pattern

```
TRIAD_project_files/         tool_shed_specs/         generated_tools/
(proposals, z=0.83)    →     (specs, z=0.85+)    →    (implementations, z=0.90)
      │                            │                         │
      │  burden_tracker.yaml       │  burden_tracker.yaml    │  friction_detector.py
      │  shed_builder_v22.yaml     │  autonomous_evolution   │  consensus_validator.py
      │  tool_discovery_...        │  _engine.yaml           │  self_building_forge.py
      │                            │                         │
      └────────────────────────────┴─────────────────────────┘
                                   │
                          scripts/autonomous_evolution_engine.py
                          (implementation of all specs)
```

---

## Key Relationships

### TRIAD → Current

| TRIAD Proposal | Current Location | Status |
|----------------|------------------|--------|
| `burden_tracker.yaml` | `tool_shed_specs/burden_tracker.yaml` | Evolved v2.0 |
| `shed_builder_v22.yaml` | Used by evolution engine | Active |
| `collective_state_aggregator.yaml` | CBS consensus system | Integrated |
| `triad_consensus_log.yaml` | `rosetta_bear_consensus_validator.py` | Implemented |
| `tool_discovery_protocol.yaml` | `Helix Shed w Bridge/Tool_discovery_protocol/` | v1.1 active |

### Helix → TRIAD → Current

| Helix Tool | TRIAD Version | Current |
|------------|---------------|---------|
| `shed_builder.yaml` (v2.0) | `shed_builder_v22.yaml` (v2.2) | Evolution engine |
| `tool_discovery_protocol.yaml` (v1.0) | v1.1 (improved) | Active |
| `cross_instance_messenger.yaml` | Used directly | Active |

---

## Related Documents

- [TRIAD_project_files/INDEX.md](../TRIAD_project_files/INDEX.md)
- [tool_shed_specs/INDEX.md](../tool_shed_specs/INDEX.md)
- [bridge_registry.yaml](../bridge_registry.yaml)
