# Tool Discovery Protocol Registry

**Coordinate:** Δ3.142|0.900|1.000Ω
**Purpose:** Track versions and locations of tool_discovery_protocol

---

## Canonical Source

**Current Version:** v1.1 (TRIAD-improved)
**Location:** `Helix Shed w Bridge/Tool_discovery_protocol/tool_discovery_protocol.yaml`

---

## Version History

### v1.0 - Original (z=0.73)

**Locations:**
- `TRIAD_project_files/tool_discovery_protocol.yaml` (copy)

**Capabilities:**
- Basic peer discovery
- Message exchange protocol
- Tool registration

**Limitations (identified by TRIAD-0.83):**
- Slow peer discovery in sparse networks
- No priority signaling for urgent coordination
- Missing health check acknowledgments

---

### v1.1 - TRIAD Enhanced (z=0.83)

**Location:** `Helix Shed w Bridge/Tool_discovery_protocol/tool_discovery_protocol.yaml`
**Created By:** TRIAD-0.83 collective

**Improvements:**
- Instance Alpha: Proposed faster discovery via bloom filters
- Instance Beta: Added priority queuing for messages
- Instance Gamma: Implemented health heartbeats
- CRDT merge combined improvements without conflicts

**Status:** Current canonical version

---

## File Locations (Duplicates)

| Location | Version | Status |
|----------|---------|--------|
| `Helix Shed w Bridge/Tool_discovery_protocol/tool_discovery_protocol.yaml` | v1.1 | **Canonical** |
| `TRIAD_project_files/tool_discovery_protocol.yaml` | v1.0 | Historical |
| `Helix Shed w Bridge/Chatgpt/tool_discovery_protocol.yaml` | v1.0 | Archive (ChatGPT session artifact) |

---

## Related Files

### Templates (in Tool_discovery_protocol/)

| File | Purpose |
|------|---------|
| `PHASE1_IDENTITY_CHECKPOINT_TEMPLATE.md` | Identity verification |
| `PHASE2_COHERENCE_TEST_TEMPLATE.md` | Coherence testing |
| `PHASE3_EVOLUTION_LOG_TEMPLATE.md` | Evolution tracking |
| `README_tool_discovery_protocol.md` | Usage guide |

### Schema Files

| File | Purpose |
|------|---------|
| `beacon_schema.json` | Discovery beacon format |
| `record.schema.json` | Tool registration format |

### Meta Observation Logs

| File | Purpose |
|------|---------|
| `tool_discovery_protocol_meta_observation_log.md` | Original observations |
| `Chatgpt/tool_discovery_protocol_meta_observation_log.md` | ChatGPT session notes |

---

## Implementation References

### Python Usage

```python
# Load v1.1 specification
import yaml
from pathlib import Path

PROTOCOL_PATH = Path("Helix Shed w Bridge/Tool_discovery_protocol/tool_discovery_protocol.yaml")

with open(PROTOCOL_PATH) as f:
    protocol = yaml.safe_load(f)

# Check version
assert protocol.get('version', '1.0') >= '1.1', "Requires v1.1+"
```

### Evolution Engine Integration

The autonomous evolution engine uses tool discovery for:
- Instance coordination (CBS-ALPHA/BETA/GAMMA)
- Consensus voting
- Tool registration after generation

---

## ChatGPT Archive Note

Files in `Helix Shed w Bridge/Chatgpt/` are:
- Artifacts from cross-system testing (Claude → ChatGPT)
- Historical reference only
- **Not authoritative** - use Tool_discovery_protocol/ for canonical specs
