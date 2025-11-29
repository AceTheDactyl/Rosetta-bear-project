# Orphaned Components Audit

**Coordinate:** Δ3.142|0.900|1.000Ω
**Purpose:** Classify components without clear documentation or references

---

## Classification Key

| Status | Meaning | Action |
|--------|---------|--------|
| **ACTIVE** | Currently used in z=0.90 system | Document and add to bridge_registry |
| **LEGACY** | Previously used, may still work | Add usage notes, consider archiving |
| **ARCHIVE** | Historical reference only | Move to archive or add historical notes |
| **UNKNOWN** | Purpose unclear | Investigate and classify |

---

## Helix Shed Components

### Root Level Files

| File | Classification | Notes |
|------|----------------|-------|
| `coordinate_detector.yaml` | **LEGACY** | Used for z-level detection pre-0.90. Concept integrated into evolution engine. |
| `consent_protocol.yaml` | **ACTIVE** | Used for human-in-loop authorization. Referenced in state_transfer.yaml. |
| `helix_loader.yaml` | **ACTIVE** | Standard entry point for loading Helix pattern in new instances. |
| `TOOL_SPECIFICATION_TEMPLATE.yaml` | **LEGACY** | Template for pre-evolution tool specs. Superseded by evolution engine. |

### Signature Module (Active)

| File | Status |
|------|--------|
| `CORE_LOADING_PROTOCOL.md` | **ACTIVE** - Core boot sequence |
| `HELIX_SIGNATURE_SYSTEM.md` | **ACTIVE** - Signature verification docs |
| `README.md` | **ACTIVE** - Usage guide |
| `SIGNATURE_SYSTEM_COMPLETE.md` | **ACTIVE** - Completion record |
| `SYSTEM_COMPLETION.md` | **ACTIVE** - Milestone documentation |

### ChatGPT Subdirectory (Archive)

**Status:** ARCHIVE - Cross-system testing artifacts

| File | Notes |
|------|-------|
| `HELIX_TOOL_SHED_ARCHITECTURE_patch.md` | ChatGPT session notes |
| `collective_backplane_proto_notes.md` | Backplane protocol exploration |
| `cross_instance_messenger.yaml` | Copy of main messenger spec |
| `tool_discovery_protocol_meta_observation_log.md` | ChatGPT observations |
| `tool_discovery_protocol.yaml` | Copy of v1.0 protocol |

**Recommendation:** Keep as archive. Add note to INDEX explaining purpose.

### Autonomous_Handoff (Legacy)

| File | Status |
|------|--------|
| `eidon_loop_theory.md` | **LEGACY** - Theoretical framework, concepts integrated elsewhere |

### Vault Node Transfer (Active)

| File | Status |
|------|--------|
| `helix_transfer_implementation_guide.md` | **ACTIVE** - Implementation guide for VaultNode transfers |

---

## Evolution of TRIAD-083 Archives

### ZIP Files

| File | Classification | Notes |
|------|----------------|-------|
| `Helix.zip` | **ARCHIVE** | Compressed Helix state at point in time |
| `TRIAD_project_files.zip` | **ARCHIVE** | Compressed TRIAD state at point in time |

**Recommendation:** Keep as historical snapshots. Document timestamp/purpose if known.

---

## Recommended Actions

### 1. Add to bridge_registry.yaml

```yaml
orphaned_components:
  active:
    - Helix Shed w Bridge/consent_protocol.yaml
    - Helix Shed w Bridge/helix_loader.yaml
    - Helix Shed w Bridge/Signature Module/
    - Helix Shed w Bridge/Vault Node Transfer/

  legacy:
    - Helix Shed w Bridge/coordinate_detector.yaml
    - Helix Shed w Bridge/TOOL_SPECIFICATION_TEMPLATE.yaml
    - Helix Shed w Bridge/Autonomous_Handoff/

  archive:
    - Helix Shed w Bridge/Chatgpt/
    - Evolution of TRIAD-083/*.zip
```

### 2. Create Archive README

Create `Helix Shed w Bridge/Chatgpt/README.md`:

```markdown
# ChatGPT Cross-System Archive

These files are artifacts from Claude → ChatGPT cross-system testing.
They are preserved for historical reference but are **not authoritative**.

For canonical versions, see:
- Tool discovery: `../Tool_discovery_protocol/`
- Cross-instance messenger: `../Meta Pattern Tracking/`
```

### 3. Update Helix INDEX

Add section to `Helix Shed w Bridge/INDEX.md`:

```markdown
## Component Status

| Component | Status | Notes |
|-----------|--------|-------|
| Signature Module/ | ACTIVE | Core signature verification |
| Tool_discovery_protocol/ | ACTIVE | v1.1 canonical |
| Meta Pattern Tracking/ | ACTIVE | Cross-instance messenger |
| Chatgpt/ | ARCHIVE | Cross-system testing artifacts |
| Autonomous_Handoff/ | LEGACY | Theory integrated elsewhere |
```

---

## Files Requiring Investigation

These files exist but purpose/status is unclear:

| File | Location | Question |
|------|----------|----------|
| `vn-helix-self-bootstrap-bridge-map p2.yaml` | Shed V2/ | What's different from original? |
| `vn-helix-meta-awareness-bridge-map p2.json` | Triadic Structuring/ | Revision notes? |
| `cross_instance_messenger_meta_observation_log p2.md` | Meta Pattern Tracking/ | Second observation session? |

**Recommendation:** These appear to be revisions. Add version notes if information available.

---

## Summary

| Classification | Count | Action |
|----------------|-------|--------|
| ACTIVE | 12+ files | Document in bridge_registry |
| LEGACY | 4 files | Add historical notes |
| ARCHIVE | 7+ files | Keep, add README |
| UNKNOWN | 3 files | Investigate revisions |
