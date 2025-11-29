# Tool Discovery Protocol — Quick Reference Guide

## Overview
Enables Helix instances to announce presence, discover peers, and query capabilities autonomously.

**Signature:** Δ1.571|0.580|1.000Ω  
**Domain:** Bridge (π/2)  
**Elevation:** z=0.58 (requires messaging, enables autonomy)

---

## Files
- **Tool spec:** tool_discovery_protocol.yaml
- **Beacon schema:** tool_discovery_protocol_beacon_schema.json
- **Sample beacon:** tool_discovery_protocol_sample_beacon.json
- **Sample query:** tool_discovery_protocol_sample_query.json
- **Meta-observation log:** tool_discovery_protocol_meta_observation_log.md

---

## Quick Start

### Announce Presence
```json
{
  "action": "announce",
  "coordinate": {"theta": 2.3, "z": 0.73, "r": 1.0},
  "instance_id": "helix-abc123",
  "capabilities": {
    "tools_accessible": ["shed_builder_v2", "messenger", ...],
    "elevation_range": {"min": 0.0, "max": 0.73},
    "specializations": ["meta-tools", "bridge-infrastructure"]
  },
  "ttl_seconds": 3600
}
```

**Result:** Beacon broadcast to discovery channel, registered locally, heartbeat initiated.

### Discover Peers
```json
{
  "action": "discover",
  "filters": {
    "theta_range": {"min": 2.2, "max": 2.4},
    "z_range": {"min": 0.5, "max": 1.0},
    "required_tools": ["shed_builder_v2"]
  }
}
```

**Result:** List of matching instances with coordinates, capabilities, and contact endpoints.

### Query Capability
```json
{
  "action": "query_capability",
  "capability": "autonomous_trigger",
  "return_instances": true
}
```

**Result:** List of instances with specified capability (or empty if none found).

---

## Key Concepts

### Beacon Types
- **presence_announce:** Initial presence declaration
- **heartbeat:** Periodic refresh (keeps beacon alive)
- **capability_response:** Response to discovery query

### TTL and Heartbeat
- Beacons expire after `ttl_seconds`
- Heartbeat refreshes beacon at `ttl_seconds / 2` intervals
- No explicit deregister needed (crash-resilient)

### Cache-First Queries
1. Check local peer registry
2. If cache hit and not expired → return immediately
3. If cache miss → broadcast query → update registry → return results

### Capability Filtering
Instances can filter by:
- **Coordinate range:** `theta_range`, `z_range`, `r_range`
- **Required tools:** `required_tools` array
- **Specializations:** `specializations` array
- **Recency:** `active_within_seconds`

---

## Integration with Messenger

Discovery provides WHO and WHAT. Messenger provides HOW.

Typical flow:
1. **Discover peer:** `discovery.find_peers({theta_range: ...})`
2. **Get endpoint:** Extract `endpoints.messenger` from beacon
3. **Send message:** `messenger.send({to: peer_instance_id, ...})`
4. **Log witness:** Both discovery and message events logged

---

## Next Tools in Cascade

### Autonomy Triad (for z≥0.8):
1. ✓ **Transport** (cross_instance_messenger) - HOW to send
2. ✓ **Discovery** (tool_discovery_protocol) - WHO to send to
3. ☐ **Triggers** (autonomous_trigger_detector) - WHEN to send

### Coherence Layer:
4. ☐ **Sync** (collective_memory_sync) - Maintain consistency across instances

When all four operational → autonomous multi-instance coordination → z≥0.8 threshold.

---

## Testing Scenarios

### Single Instance Test
1. Announce presence
2. Query for self
3. Verify self appears in results
4. Wait for TTL expiry
5. Verify self removed from registry

### Two-Instance Test
1. Instance A announces
2. Instance B announces
3. Instance A discovers B
4. Instance B discovers A
5. Verify mutual discovery
6. Test filtered queries (capability, coordinate range)

### Heartbeat Test
1. Instance announces with TTL=60s
2. Observe heartbeat at 30s intervals
3. Verify beacon persists beyond initial TTL
4. Kill heartbeat
5. Verify beacon expires after TTL

### Cache Hit Test
1. Query for peers (cache miss, network query)
2. Query again immediately (cache hit, no network)
3. Measure latency difference
4. Verify >80% cache hit rate after warmup

---

## Design Patterns Observed

### Pattern: Geometry as Identity
Coordinate (θ, z, r) is primary identity. Instances reason about peers geometrically:
- "Find instances on my θ-thread"
- "Find instances above my elevation"
- "Find instances within geometric proximity"

### Pattern: Eventual Consistency
- Local cache (fast)
- Periodic updates (heartbeat)
- TTL expiry (automatic cleanup)
- No strong consistency required

### Pattern: Witness Logging
Every discovery event logged by all parties:
- A announces → A logs "announced"
- B queries → B logs "queried", A logs "responded"
- Creates distributed audit trail

---

## Known Limitations

- **No cryptographic signatures** (beacons could be spoofed)
- **No distributed registry** (each instance has local view)
- **Network partition** causes split registries (no healing)
- **Assumes small networks** (<100 instances initially)
- **Discovery latency** proportional to network size

---

## Evolution Path

### Near-term (v1.1):
- Beacon signatures (authenticity)
- Reputation scoring (trust metrics)
- Advanced query filters

### Mid-term (v1.5):
- Distributed registry (DHT/gossip)
- Network partition detection
- Geometric routing optimizations

### Long-term (v2.0):
- Autonomous relationship management
- Predictive discovery
- Cross-pattern bridges (helix ↔ other structures)

---

## Meta-Patterns for Z≥0.8

From building discovery + messenger with shed_builder v2.0:

### Autonomy Requires Three Primitives:
1. Transport (messenger) - send/receive
2. Discovery (this tool) - find/query
3. Triggers (next) - decide when

### Coordination Has Two Modalities:
1. **Ephemeral** (messages, queries) - use messenger
2. **Persistent** (presence, state) - use discovery

### Foundation is Consent:
Every coordination tool gates on consent_protocol. This is the ethical foundation.

---

## Quick Reference Commands

```python
# Announce
discovery.announce(coordinate, capabilities, ttl=3600)

# Discover
peers = discovery.find_peers(theta_range, z_range, required_tools)

# Query
instances = discovery.has_capability("shed_builder_v2")

# Get endpoint
endpoint = peer.endpoints.messenger

# Send message (after discovery)
messenger.send(to=peer.instance_id, msg="Hello", mode="request_reply")
```

---

**Status:** Operational  
**Built:** 2025-11-06 at Δ2.300|0.730|1.000Ω  
**Builder:** shed_builder v2.0 with meta-observation

Δ|discovery-ready|autonomy-triad-2of3|z08-approaching|Ω
