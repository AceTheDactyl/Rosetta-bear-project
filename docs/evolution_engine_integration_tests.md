# Evolution Engine Integration Tests

## Overview

This document specifies integration tests for the Autonomous Evolution Engine's connections to:
- **shed_builder v2.2** - Tool creation and modification
- **burden_tracker v2.0** - Friction metrics feed
- **collective_memory_sync** - Cross-instance state coordination

These integrations are critical for achieving z=0.90 (Full Substrate Transcendence).

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     AUTONOMOUS EVOLUTION ENGINE                          │
│                           (z = 0.90 target)                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                │
│   │  Friction   │───▶│ Improvement │───▶│ Collective  │                │
│   │  Detection  │    │  Proposal   │    │ Validation  │                │
│   └─────────────┘    └─────────────┘    └─────────────┘                │
│          │                  │                  │                        │
│          │                  │                  │                        │
│   ┌──────┴──────┐    ┌──────┴──────┐    ┌──────┴──────┐                │
│   │  Autonomous │───▶│    Meta     │    │             │                │
│   │  Execution  │    │  Learning   │    │             │                │
│   └─────────────┘    └─────────────┘    │             │                │
│          │                  │           │             │                │
└──────────┼──────────────────┼───────────┼─────────────┼────────────────┘
           │                  │           │             │
           ▼                  ▼           ▼             ▼
┌──────────────────┐  ┌──────────────┐  ┌─────────────────────────────┐
│  shed_builder    │  │burden_tracker│  │   collective_memory_sync    │
│     v2.2         │  │    v2.0      │  │                             │
├──────────────────┤  ├──────────────┤  ├─────────────────────────────┤
│ • Tool creation  │  │ • Friction   │  │ • Cross-instance state      │
│ • Tool modify    │  │   metrics    │  │ • Consensus coordination    │
│ • Validation     │  │ • Pattern    │  │ • Learning propagation      │
│ • GHMP plates    │  │   detection  │  │ • Vote collection           │
└──────────────────┘  └──────────────┘  └─────────────────────────────┘
```

---

## Integration 1: Evolution Engine ↔ shed_builder v2.2

### Connection Specification

| Property | Value |
|----------|-------|
| Interface | Internal API |
| Protocol | Function call / Event bus |
| Auth | Capability token (z ≥ 0.88 required) |
| Data Format | JSON |

### Data Flows

#### 1.1 Tool Modification Request

**Direction:** Evolution Engine → shed_builder

```json
{
  "operation": "modify_tool",
  "proposal_id": "PROP-2025-001",
  "authorization": {
    "token": "eyJ...",
    "issued_by": "consensus_validator",
    "z_level_at_issue": 0.89
  },
  "target_tool": {
    "name": "burden_tracker",
    "version": "2.0.0",
    "component": "friction_detection"
  },
  "modification": {
    "type": "parameter_update",
    "changes": [
      {
        "parameter": "detection_sensitivity",
        "old_value": 0.5,
        "new_value": 0.6,
        "rationale": "Reduce false positives per friction report"
      }
    ]
  },
  "rollback_spec": {
    "checkpoint_id": "CP-20250129-001",
    "revert_command": "restore_parameter"
  }
}
```

#### 1.2 Modification Result

**Direction:** shed_builder → Evolution Engine

```json
{
  "operation": "modify_tool_result",
  "proposal_id": "PROP-2025-001",
  "status": "success",
  "result": {
    "tool_version_new": "2.0.1",
    "changes_applied": 1,
    "validation_passed": true,
    "ghmp_plate_generated": "burden_tracker_v2.0.1_mod.png"
  },
  "metrics": {
    "execution_time_ms": 1250,
    "validation_time_ms": 3400
  }
}
```

### Integration Tests

#### SHED-INT-001: Basic Tool Modification

**Objective:** Verify evolution engine can modify tools via shed_builder

**Setup:**
- shed_builder v2.2 running
- Evolution engine at z ≥ 0.88
- Valid capability token

**Steps:**
1. Generate improvement proposal for tool parameter change
2. Obtain consensus authorization
3. Submit modification request to shed_builder
4. Verify modification applied
5. Verify GHMP plate generated

**Assertions:**
- [ ] Modification request accepted
- [ ] Tool parameter changed
- [ ] New version number assigned
- [ ] Validation tests pass
- [ ] GHMP plate contains change record

**Rollback Test:**
1. Trigger rollback via evolution engine
2. Verify original parameter restored
3. Verify rollback recorded in GHMP

---

#### SHED-INT-002: Tool Creation Request

**Objective:** Verify evolution engine can request new tool creation

**Setup:**
- Pattern detected: "missing_capability_X"
- Proposal generated for new tool

**Steps:**
1. Submit tool creation request with spec
2. shed_builder generates tool skeleton
3. Evolution engine provides implementation hints
4. Validation cycle runs
5. Tool deployed to tool_shed

**Assertions:**
- [ ] Tool spec validated
- [ ] Skeleton generated correctly
- [ ] Implementation compiles/passes lint
- [ ] Tool added to registry
- [ ] GHMP plate sealed for new tool

---

#### SHED-INT-003: Authorization Enforcement

**Objective:** Verify shed_builder rejects unauthorized modifications

**Steps:**
1. Attempt modification without token → Rejected
2. Attempt modification with expired token → Rejected
3. Attempt modification with z < 0.88 token → Rejected
4. Attempt modification with valid token → Accepted

**Assertions:**
- [ ] All unauthorized attempts return 403
- [ ] Security event logged for each rejection
- [ ] Valid token proceeds normally

---

#### SHED-INT-004: Cascading Tool Updates

**Objective:** Verify dependent tools updated correctly

**Scenario:**
- Tool A depends on Tool B
- Proposal modifies Tool B interface

**Steps:**
1. Submit proposal affecting Tool B
2. shed_builder identifies Tool A as dependent
3. Compatibility check runs
4. If breaking: require Tool A update in same proposal
5. If compatible: proceed with Tool B update only

**Assertions:**
- [ ] Dependencies identified automatically
- [ ] Breaking changes require coordinated update
- [ ] Compatible changes proceed independently
- [ ] Dependency graph updated in registry

---

## Integration 2: Evolution Engine ↔ burden_tracker v2.0

### Connection Specification

| Property | Value |
|----------|-------|
| Interface | Event bus subscription |
| Protocol | Pub/Sub |
| Topic | `friction_metrics` |
| Data Format | Friction Report (JSON) |

### Data Flows

#### 2.1 Friction Report Feed

**Direction:** burden_tracker → Evolution Engine (continuous)

```json
{
  "report_id": "FR-20250129-1430",
  "timestamp": "2025-01-29T14:30:00Z",
  "reporting_instance": "CBS-ALPHA",
  "aggregate_burden": 0.42,
  "category_breakdown": {
    "cognitive_load": 0.38,
    "execution_latency": 0.45,
    "coordination_overhead": 0.52,
    "error_recovery": 0.33
  },
  "friction_events": [
    {
      "event_id": "FE-001",
      "event_type": "execution_latency_spike",
      "severity": "moderate",
      "source": "memory_access",
      "magnitude": 0.23,
      "correlation_id": "CORR-MEM-001"
    }
  ],
  "patterns_detected": [
    {
      "pattern_name": "repeated_memory_thrashing",
      "confidence": 0.85,
      "occurrences": 12,
      "first_seen": "2025-01-29T12:00:00Z"
    }
  ],
  "improvement_opportunities": [
    {
      "opportunity_id": "OPP-001",
      "description": "Increase memory cache to reduce thrashing",
      "predicted_reduction": 0.15,
      "affected_categories": ["execution_latency", "cognitive_load"]
    }
  ]
}
```

#### 2.2 Pattern Query

**Direction:** Evolution Engine → burden_tracker

```json
{
  "query": "get_pattern_history",
  "pattern_name": "repeated_memory_thrashing",
  "time_range": {
    "start": "2025-01-22T00:00:00Z",
    "end": "2025-01-29T14:30:00Z"
  }
}
```

### Integration Tests

#### BURDEN-INT-001: Friction Feed Subscription

**Objective:** Verify evolution engine receives friction reports

**Setup:**
- burden_tracker operational
- Evolution engine subscribed to `friction_metrics`

**Steps:**
1. burden_tracker generates friction report
2. Report published to topic
3. Evolution engine receives report
4. Report parsed and stored

**Assertions:**
- [ ] Report received within 5 seconds of generation
- [ ] All fields parsed correctly
- [ ] Report stored in evolution context
- [ ] Subscription remains active (no drops)

---

#### BURDEN-INT-002: Pattern-to-Proposal Pipeline

**Objective:** Verify detected patterns generate proposals

**Setup:**
- Pattern detection enabled
- min_predicted_improvement = 0.05

**Steps:**
1. Inject friction pattern with predicted_reduction = 0.15
2. Evolution engine receives pattern
3. Proposal generation triggered
4. Proposal references original pattern

**Assertions:**
- [ ] Pattern triggers proposal generation
- [ ] Proposal.source_pattern matches pattern_name
- [ ] Predicted burden delta matches pattern's predicted_reduction
- [ ] Latency: pattern → proposal < 60 seconds

---

#### BURDEN-INT-003: Feedback Loop Verification

**Objective:** Verify executed improvements reduce burden

**Steps:**
1. Record baseline burden (T0)
2. Execute improvement proposal
3. Wait for burden_tracker sampling window
4. Record new burden (T1)
5. Calculate delta

**Assertions:**
- [ ] Burden at T1 < Burden at T0
- [ ] Delta approximately matches predicted_reduction (±20%)
- [ ] burden_tracker shows trend "decreasing"
- [ ] Meta-learning captures success

---

#### BURDEN-INT-004: High-Volume Friction Handling

**Objective:** Verify system handles friction spikes

**Setup:**
- Generate 100 friction events per minute

**Steps:**
1. Flood burden_tracker with events
2. Monitor evolution engine queue depth
3. Verify no events dropped
4. Verify proposal rate limiting enforced

**Assertions:**
- [ ] All events received (100% delivery)
- [ ] Queue depth bounded (< 1000)
- [ ] Proposals limited to max_per_cycle
- [ ] System remains responsive

---

## Integration 3: Evolution Engine ↔ collective_memory_sync

### Connection Specification

| Property | Value |
|----------|-------|
| Interface | Sync protocol |
| Protocol | Custom (over internal network) |
| Participants | All active instances |
| Consistency | Eventual (< 5s) |

### Data Flows

#### 3.1 Consensus Vote Request

**Direction:** Evolution Engine → collective_memory_sync → All Instances

```json
{
  "message_type": "consensus_request",
  "proposal_id": "PROP-2025-001",
  "requesting_instance": "CBS-ALPHA",
  "proposal_summary": {
    "description": "Increase memory cache size",
    "predicted_burden_delta": -0.12,
    "risk_level": "low"
  },
  "vote_deadline": "2025-01-29T18:30:00Z",
  "quorum_required": 3
}
```

#### 3.2 Vote Response

**Direction:** Instance → collective_memory_sync → Evolution Engine

```json
{
  "message_type": "consensus_vote",
  "proposal_id": "PROP-2025-001",
  "voting_instance": "CBS-BETA",
  "vote": "approve",
  "reasoning": "Predicted improvement aligns with observed friction",
  "signature": "ed25519:...",
  "timestamp": "2025-01-29T15:45:00Z"
}
```

#### 3.3 Learning Propagation

**Direction:** Evolution Engine → collective_memory_sync → All Instances

```json
{
  "message_type": "learning_update",
  "learning_id": "LEARN-2025-001",
  "source_instance": "CBS-ALPHA",
  "pattern": {
    "friction_type": "memory_thrashing",
    "successful_intervention": "cache_size_increase",
    "observed_reduction": 0.14
  },
  "confidence": 0.88,
  "propagate_to": "all"
}
```

### Integration Tests

#### SYNC-INT-001: Consensus Round Trip

**Objective:** Verify consensus completes across instances

**Setup:**
- 3 instances running
- All connected via collective_memory_sync

**Steps:**
1. Instance A initiates consensus request
2. Request propagates to B and C
3. B and C submit votes
4. Votes collected at A
5. Consensus result calculated

**Assertions:**
- [ ] Request received by all instances (< 2s)
- [ ] All votes returned (< 30s)
- [ ] Consensus correctly calculated
- [ ] Result propagated to all instances

---

#### SYNC-INT-002: Learning Propagation

**Objective:** Verify learnings sync across collective

**Steps:**
1. Instance A completes evolution cycle
2. Meta-learning extracts pattern
3. Learning published via sync
4. Query Instance B and C for pattern

**Assertions:**
- [ ] Learning available on B within 5 seconds
- [ ] Learning available on C within 5 seconds
- [ ] Pattern library versions match
- [ ] No conflicts during merge

---

#### SYNC-INT-003: Partition Tolerance

**Objective:** Verify system handles network partition

**Steps:**
1. Partition: A can reach B, but not C
2. Initiate consensus (A → B succeeds, A → C fails)
3. Wait for timeout
4. Verify graceful degradation

**Assertions:**
- [ ] Partition detected within timeout
- [ ] Consensus proceeds with available quorum (if met)
- [ ] C marked as unreachable
- [ ] Alert generated
- [ ] Reconciliation on partition heal

---

#### SYNC-INT-004: Conflict Resolution

**Objective:** Verify concurrent updates don't corrupt state

**Steps:**
1. Instance A and B simultaneously update same pattern
2. Both publish via sync
3. Observe conflict handling

**Assertions:**
- [ ] Conflict detected
- [ ] Resolution strategy applied (latest-wins / merge)
- [ ] Final state consistent across all instances
- [ ] Conflict logged for review

---

#### SYNC-INT-005: Cross-Instance Execution Coordination

**Objective:** Verify execution doesn't duplicate

**Scenario:**
- Proposal approved
- Multiple instances could execute

**Steps:**
1. Consensus completes with authorization
2. Execution coordinator selects single executor
3. Verify only one instance executes
4. Result propagated to all

**Assertions:**
- [ ] Only one instance executes
- [ ] Execution lock acquired atomically
- [ ] Other instances receive result
- [ ] No duplicate side effects

---

## End-to-End Integration Scenarios

### E2E-001: Complete Autonomous Cycle

**Objective:** Verify full cycle without human intervention

**Duration:** 30 minutes

**Steps:**
1. **T+0:** Inject sustained friction (memory_thrashing pattern)
2. **T+1m:** burden_tracker detects and reports
3. **T+2m:** Evolution engine generates proposal
4. **T+3m:** Consensus request sent via sync
5. **T+5m:** All instances vote (approve)
6. **T+6m:** Authorization granted
7. **T+7m:** shed_builder receives modification request
8. **T+10m:** Modification applied, validated
9. **T+15m:** burden_tracker shows reduced friction
10. **T+20m:** Meta-learning captures success
11. **T+25m:** Learning propagated to all instances

**Success Criteria:**
- [ ] No human intervention required
- [ ] All phases complete successfully
- [ ] Burden measurably reduced
- [ ] Learning available collective-wide

---

### E2E-002: Rejection and Revision Cycle

**Objective:** Verify rejected proposals are revised

**Steps:**
1. Generate proposal with high risk
2. Consensus rejects (veto from one instance)
3. Evolution engine revises proposal (lower risk approach)
4. Re-submit for consensus
5. Approval on second attempt
6. Execute successfully

**Success Criteria:**
- [ ] Initial rejection handled gracefully
- [ ] Revision incorporates rejection feedback
- [ ] Second proposal differs from first
- [ ] Eventually succeeds

---

### E2E-003: Rollback Recovery

**Objective:** Verify system recovers from failed execution

**Steps:**
1. Execute proposal that causes burden increase
2. Automatic rollback triggered
3. System restored to checkpoint
4. Failure recorded in meta-learning
5. Similar proposals flagged higher risk

**Success Criteria:**
- [ ] Rollback completes within threshold
- [ ] No permanent damage
- [ ] Learning prevents repetition
- [ ] System returns to stable state

---

### E2E-004: Multi-Tool Coordinated Update

**Objective:** Verify coordinated updates across tools

**Scenario:**
- Improvement requires changes to both burden_tracker AND shed_builder

**Steps:**
1. Proposal includes two modifications
2. Consensus covers both changes
3. shed_builder executes atomically
4. Both tools updated
5. Integration validated

**Success Criteria:**
- [ ] Atomic update (both or neither)
- [ ] No intermediate inconsistent state
- [ ] Cross-tool validation passes
- [ ] Single GHMP plate for transaction

---

## Performance Benchmarks

### PERF-INT-001: Friction-to-Proposal Latency

| Metric | Target | Notes |
|--------|--------|-------|
| Detection latency | < 30s | Time from friction event to detection |
| Proposal generation | < 60s | Time from detection to proposal |
| Total (friction → proposal) | < 90s | End-to-end |

### PERF-INT-002: Consensus Latency

| Metric | Target | Notes |
|--------|--------|-------|
| Request propagation | < 2s | Time to reach all instances |
| Vote collection | < 60s | Time for all votes (3 instances) |
| Result calculation | < 1s | Time to compute consensus |

### PERF-INT-003: Execution Latency

| Metric | Target | Notes |
|--------|--------|-------|
| shed_builder round trip | < 30s | For simple modifications |
| Validation | < 60s | Including test runs |
| GHMP plate generation | < 10s | Encoding time |

### PERF-INT-004: Learning Propagation

| Metric | Target | Notes |
|--------|--------|-------|
| Single learning sync | < 5s | To all instances |
| Pattern library query | < 100ms | Even at 10K patterns |
| Conflict resolution | < 1s | When detected |

---

## Test Environment Configuration

```yaml
test_environment:
  instances:
    count: 3
    names: [CBS-ALPHA, CBS-BETA, CBS-GAMMA]

  components:
    evolution_engine:
      version: "1.0.0"
      z_level_simulated: 0.89

    shed_builder:
      version: "2.2.0"
      test_mode: true

    burden_tracker:
      version: "2.0.0"
      sampling_interval: 10  # faster for testing

    collective_memory_sync:
      version: "1.0.0"
      sync_interval: 1  # faster for testing

  network:
    latency_ms: 10
    partition_simulation: enabled

  ghmp:
    test_plates_dir: "/tmp/ghmp_test_plates"
    encryption: enabled

  timeouts:
    consensus: 120  # seconds (reduced for testing)
    execution: 60
    sync: 30
```

---

## Appendix: Mock Data Generators

### Friction Event Generator

```python
def generate_friction_event(severity="moderate"):
    return {
        "event_id": f"FE-{uuid4().hex[:8]}",
        "event_type": random.choice([
            "execution_latency_spike",
            "memory_thrashing",
            "consensus_delay",
            "tool_failure"
        ]),
        "severity": severity,
        "source": random.choice(["memory_access", "tool_exec", "consensus", "network"]),
        "magnitude": random.uniform(0.1, 0.5),
        "timestamp": datetime.utcnow().isoformat()
    }
```

### Proposal Generator

```python
def generate_test_proposal(risk_level="low"):
    return {
        "proposal_id": f"PROP-TEST-{uuid4().hex[:8]}",
        "description": "Test improvement proposal",
        "predicted_burden_delta": -random.uniform(0.05, 0.20),
        "risk_level": risk_level,
        "implementation_spec": {
            "target": "test_component",
            "change_type": "parameter_update"
        }
    }
```

---

*Document Version: 1.0.0*
*Last Updated: 2025-01-29*
*Integration Test Coverage: shed_builder, burden_tracker, collective_memory_sync*
