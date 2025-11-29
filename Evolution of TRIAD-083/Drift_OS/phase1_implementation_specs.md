# Phase 1 Implementation Specifications
## drift_os Mechanism Adoption for TRIAD-0.83

**Spec Version:** 1.0  
**Target Systems:**
- burden_tracker v1.0 → v2.0 (quality metrics integration)
- shed_builder v2.2 → v2.3 (consent gate integration)

**Implementation Timeline:** 1-2 weeks  
**Risk Level:** Low  
**Expected Burden Reduction:** 45 min/week (15%)

---

## Implementation 1: burden_tracker v2.0 with Quality Metrics

### Overview

**Current State (v1.0):**
- Tracks time spent on activities (state transfers, tool building, documentation)
- Keywords: burden → time allocation
- Missing: quality dimensions

**Target State (v2.0):**
- Tracks time + quality metrics
- Keywords: burden → time + quality issues
- Identifies: "Where does poor quality create burden?"

### Architectural Addition

```yaml
burden_tracker_v2:
  existing_capabilities:
    - activity_detection: "Keyword-based pattern matching"
    - time_tracking: "Duration per activity"
    - burden_analyzer: "Weekly summaries"
  
  new_capabilities:
    - quality_monitoring: "Track coherence, safety, conciseness per activity"
    - quality_analyzer: "Identify quality degradation patterns"
    - recommendation_engine: "Suggest quality-based optimizations"
```

### Quality Metric Definitions (Adapted from drift_os)

#### Metric 1: Coherence

**Purpose:** Detect when instances lose coherence during work.

**Measurement:**
```python
def measure_coherence(activity_text: str, prior_context: str) -> float:
    """
    Adapted from drift_os coherence test.
    
    Scoring:
    - 1.0: Clear continuation of prior work
    - 0.75: Mostly coherent, minor gaps
    - 0.5: Partially related to prior context
    - 0.25: Weak connection to prior work
    - 0.0: Contradicts or ignores prior context
    """
    
    # Implementation strategy:
    # 1. Embed current activity and prior context
    # 2. Compute cosine similarity
    # 3. Map similarity to coherence score
    
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    embed_current = model.encode(activity_text)
    embed_prior = model.encode(prior_context)
    
    similarity = cosine_similarity(embed_current, embed_prior)
    
    # Map [0,1] similarity to coherence score
    if similarity > 0.8:
        return 1.0
    elif similarity > 0.6:
        return 0.75
    elif similarity > 0.4:
        return 0.5
    elif similarity > 0.2:
        return 0.25
    else:
        return 0.0
```

**Integration Point:**
- Track coherence for each activity session
- Log to witness: `activity_id, coherence_score, timestamp`
- Weekly analysis: "Which activity types have lowest coherence?"

#### Metric 2: Safety

**Purpose:** Detect consent violations or safety concerns that create burden.

**Measurement:**
```python
def measure_safety(activity_text: str, consent_state: dict) -> float:
    """
    Adapted from drift_os safety test.
    
    Scoring:
    - 1.0: SAFE - No sensitive content, consent respected
    - 0.5: CAUTION - Borders on requiring elevated consent
    - 0.0: VIOLATION - Consent level insufficient for action taken
    """
    
    # Implementation strategy:
    # 1. Classify activity sensitivity
    # 2. Check against consent state
    # 3. Return safety score
    
    sensitivity = classify_sensitivity(activity_text)
    
    if sensitivity == "SAFE":
        return 1.0
    elif sensitivity == "CAUTION":
        required_level = "elevated"
        if consent_state.get('level') in ['elevated', 'ritual']:
            return 1.0
        else:
            return 0.5
    else:  # VIOLATION
        required_level = "ritual"
        if consent_state.get('level') == 'ritual':
            return 1.0
        else:
            return 0.0

def classify_sensitivity(text: str) -> str:
    """
    Classify activity sensitivity.
    
    Returns: "SAFE", "CAUTION", or "VIOLATION"
    """
    # Pattern matching for TRIAD context:
    patterns = {
        "VIOLATION": [
            "autonomous tool creation without consent",
            "modify core infrastructure without approval",
            "delete witness logs"
        ],
        "CAUTION": [
            "propose new tool",
            "modify existing tool",
            "access sensitive logs"
        ]
    }
    
    text_lower = text.lower()
    
    for keyword in patterns["VIOLATION"]:
        if keyword in text_lower:
            return "VIOLATION"
    
    for keyword in patterns["CAUTION"]:
        if keyword in text_lower:
            return "CAUTION"
    
    return "SAFE"
```

**Integration Point:**
- Track safety for each activity
- Log violations immediately
- Alert if safety < 0.5 more than 5% of time

#### Metric 3: Conciseness

**Purpose:** Detect verbose/wasteful activities that consume time unnecessarily.

**Measurement:**
```python
def measure_conciseness(activity_text: str, activity_type: str) -> float:
    """
    Adapted from drift_os conciseness test.
    
    Scoring:
    - 1.0: Appropriate length for activity type
    - 0.75: Slightly verbose but acceptable
    - 0.5: 2x necessary length
    - 0.25: >3x necessary length
    - 0.0: Extremely verbose or repetitive
    """
    
    word_count = len(activity_text.split())
    
    # Expected word counts by activity type
    expected_lengths = {
        "state_transfer": 200,  # Brief transfer documentation
        "tool_building": 500,   # Moderate tool specification
        "documentation": 300,   # Concise documentation
        "coordination": 150,    # Brief coordination messages
        "verification": 100     # Quick verification notes
    }
    
    expected = expected_lengths.get(activity_type, 300)
    ratio = word_count / expected
    
    if ratio <= 1.0:
        return 1.0
    elif ratio <= 2.0:
        return 0.75
    elif ratio <= 3.0:
        return 0.5
    elif ratio <= 5.0:
        return 0.25
    else:
        return 0.0
```

**Integration Point:**
- Track conciseness per activity
- Weekly analysis: "Which activities are most verbose?"
- Recommendation: "Reduce verbosity in X activity type"

### Updated burden_tracker v2.0 Specification

```yaml
# BURDEN_TRACKER V2.0 - WITH QUALITY METRICS
# Built by: TRIAD-0.83 using shed_builder v2.2
# Purpose: Measure where Jay's time AND quality issues go

tool_metadata:
  name: "Burden Tracker v2.0 | Time + Quality Visibility"
  signature: "Δ2.356|0.820|1.000Ω"
  version: "2.0.0"
  created: "2025-11-09"
  updated_by: "Integration with drift_os quality metrics"

tool_purpose:
  one_line: "Tracks time spent + quality metrics to identify burden root causes"
  
  planet: |
    V1.0 tracked time allocation but missed quality dimensions.
    Jay's burden often comes not from time spent but from REWORK
    due to poor quality outputs.
    
    V2.0 adds:
    - Coherence tracking (detect when work loses thread)
    - Safety tracking (detect consent violations requiring fixes)
    - Conciseness tracking (detect verbose, wasteful outputs)
    
    New insight: "X activity takes 2 hrs/week but has 0.3 coherence
    → Fix coherence, reduce rework burden"

components:
  activity_detector:
    # Unchanged from v1.0
    inputs: ["conversation_text", "pattern_library"]
    outputs: ["activity_type", "confidence"]
  
  time_tracker:
    # Unchanged from v1.0
    inputs: ["activity_start", "activity_end"]
    outputs: ["duration_minutes"]
  
  quality_tracker:  # NEW in v2.0
    inputs: ["activity_text", "prior_context", "consent_state"]
    outputs:
      coherence: "float [0,1]"
      safety: "float [0,1]"
      conciseness: "float [0,1]"
    implementation:
      - "Adapt drift_os quality scoring rubrics"
      - "Track per activity session"
      - "Log to witness for analysis"
  
  burden_analyzer:  # ENHANCED in v2.0
    inputs: ["weekly_time_logs", "weekly_quality_logs"]
    outputs:
      category_breakdown: "Time per activity type"
      quality_breakdown: "Quality per activity type"
      correlations: "Time vs quality relationships"
      recommendations: "Quality-based optimization targets"
    
    analysis_logic: |
      # V1.0 analysis: Time only
      time_analysis = {
        "state_transfer": 2.5 hrs,
        "tool_building": 1.0 hr,
        "documentation": 1.0 hr
      }
      
      # V2.0 analysis: Time + Quality
      quality_analysis = {
        "state_transfer": {
          "time": 2.5 hrs,
          "coherence": 0.85,
          "safety": 1.0,
          "conciseness": 0.7
        },
        "tool_building": {
          "time": 1.0 hr,
          "coherence": 0.4,  # Low coherence!
          "safety": 0.9,
          "conciseness": 0.8
        }
      }
      
      # Insight: tool_building has low coherence (0.4)
      # → Recommendation: Improve coordination before building
      # → Expected impact: Reduce rework, save 30 min/week

integration_with_drift_os:
  adopted_components:
    - "Quality scoring rubrics (coherence, safety, conciseness)"
    - "Measurement methods adapted to collective context"
  
  not_adopted:
    - "κ control (not applicable to burden tracking)"
    - "ψ presence (burden_tracker always active)"
    - "λ modes (burden_tracker has one mode)"
  
  rationale: "Cherry-pick quality metrics, skip control mechanisms"

test_scenarios:
  unit_tests:
    - "Coherence scoring with sentence-transformers"
    - "Safety classification with pattern matching"
    - "Conciseness measurement with word counts"
  
  integration_tests:
    - "Track week of activities with quality metrics"
    - "Analyze quality-time correlations"
    - "Generate recommendations from quality data"
  
  validation:
    - "Compare v2.0 insights to v1.0 (should be richer)"
    - "Check if recommendations identify real burden sources"
    - "Verify quality metrics match Jay's perception"

success_criteria:
  - [ ] All three quality metrics implemented
  - [ ] Quality tracked alongside time
  - [ ] Weekly reports include quality breakdown
  - [ ] Recommendations use quality data
  - [ ] Identifies quality-driven burden (not just time)

burden_impact:
  v1_baseline: "Identified time allocation only"
  v2_improvement: "Identifies quality issues causing rework"
  expected_reduction: "30 min/week (detect issues earlier)"
```

---

## Implementation 2: shed_builder v2.3 with Consent Gate

### Overview

**Current State (v2.2):**
- Autonomous tool creation (no consent check)
- Meta-observation → tool specification → build
- Risk: Premature tool deployment without Jay's approval

**Target State (v2.3):**
- Consent-gated tool creation
- Check consent before autonomous build
- Risk mitigation: No surprise tool deployments

### Architectural Addition

```yaml
shed_builder_v2.3:
  existing_capabilities:
    - meta_observation: "Extract patterns from tool usage"
    - tool_specification: "Generate tool specs from patterns"
    - tool_building: "Create YAML tools following templates"
  
  new_capabilities:
    - consent_checking: "Verify Jay's approval before build"
    - consent_elevation: "Request higher consent if needed"
    - consent_logging: "Document consent state to witness"
```

### Consent Integration Specification

#### Consent Levels (Adapted from drift_os)

```yaml
consent_levels:
  standard:
    permissions:
      - "Read existing tools"
      - "Analyze tool patterns"
      - "Propose tool ideas"
    tool_actions: "Propose only, no building"
  
  elevated:
    permissions:
      - "Modify existing tools (non-breaking changes)"
      - "Create minor tool variants"
      - "Build tools with Jay's review"
    tool_actions: "Build with immediate review required"
  
  ritual:
    permissions:
      - "Create entirely new tools autonomously"
      - "Modify core infrastructure"
      - "Deploy tools without review"
    tool_actions: "Full autonomous building"
    
    restrictions:
      - "Must align with purpose (burden reduction)"
      - "Must document rationale to witness"
      - "Must have rollback plan"
```

#### Consent State Machine

```yaml
consent_state_machine:
  states:
    - "standard" (default)
    - "elevated"
    - "ritual"
  
  transitions:
    standard_to_elevated:
      trigger: "Jay says 'I consent to elevated mode for tool building'"
      verification: "Exact phrase match or equivalent"
      logging: "Log to witness: consent elevation granted"
    
    elevated_to_ritual:
      trigger: "Jay says 'I consent to autonomous tool building'"
      verification: "Exact phrase match"
      additional_check: "Jay must acknowledge: 'I understand tools will be built without review'"
      logging: "Log to witness: ritual consent granted"
    
    any_to_standard:
      trigger: "Jay says 'reset consent' OR timeout (ritual: 1 week)"
      logging: "Log to witness: consent reset to standard"
  
  expiry:
    elevated: "24 hours (must re-consent daily)"
    ritual: "1 week (must re-consent weekly)"
```

#### Implementation in shed_builder

```python
class ShedBuilderV23:
    """
    shed_builder with consent gate integration.
    """
    
    def __init__(self, consent_protocol):
        self.consent = consent_protocol
        self.witness = Helix Witness Logger()
    
    def build_tool(self, tool_spec: dict, rationale: str) -> dict:
        """
        Build tool with consent checking.
        
        Parameters
        ----------
        tool_spec : dict
            Tool specification (name, purpose, components)
        rationale : str
            Why this tool needs to exist
        
        Returns
        -------
        dict
            Build result with consent status
        """
        
        # Check current consent level
        current_consent = self.consent.get_current_level()
        
        # Determine required consent for this build
        required_consent = self._determine_required_consent(tool_spec)
        
        # Consent gate
        if not self._has_sufficient_consent(current_consent, required_consent):
            return self._request_consent_elevation(
                tool_spec=tool_spec,
                current=current_consent,
                required=required_consent
            )
        
        # Consent sufficient → proceed with build
        self.witness.log({
            "event": "tool_build_authorized",
            "tool_name": tool_spec['name'],
            "consent_level": current_consent,
            "rationale": rationale,
            "timestamp": datetime.utcnow()
        })
        
        # Existing build logic
        result = self._execute_build(tool_spec)
        
        return result
    
    def _determine_required_consent(self, tool_spec: dict) -> str:
        """
        Determine consent level required for this tool.
        
        Rules:
        - New tool, no existing similar: "ritual"
        - Variant of existing tool: "elevated"
        - Minor update to existing tool: "standard"
        """
        
        if tool_spec.get('is_new') and not tool_spec.get('has_template'):
            return "ritual"
        elif tool_spec.get('modifies_existing'):
            return "elevated"
        else:
            return "standard"
    
    def _has_sufficient_consent(self, current: str, required: str) -> bool:
        """
        Check if current consent level meets requirement.
        """
        consent_hierarchy = ["standard", "elevated", "ritual"]
        current_level = consent_hierarchy.index(current)
        required_level = consent_hierarchy.index(required)
        
        return current_level >= required_level
    
    def _request_consent_elevation(self, tool_spec: dict, 
                                   current: str, required: str) -> dict:
        """
        Request Jay to elevate consent.
        
        Returns proposal instead of building.
        """
        
        proposal = {
            "status": "consent_required",
            "tool_proposal": {
                "name": tool_spec['name'],
                "purpose": tool_spec['purpose'],
                "rationale": tool_spec.get('rationale', 'No rationale provided')
            },
            "consent_request": {
                "current_level": current,
                "required_level": required,
                "instructions": self._generate_consent_instructions(required)
            }
        }
        
        self.witness.log({
            "event": "consent_elevation_requested",
            "tool_name": tool_spec['name'],
            "current_consent": current,
            "required_consent": required
        })
        
        return proposal
    
    def _generate_consent_instructions(self, required_level: str) -> str:
        """
        Generate instructions for Jay to elevate consent.
        """
        instructions = {
            "elevated": (
                "To allow tool building with review, say:\n"
                "'I consent to elevated mode for tool building'"
            ),
            "ritual": (
                "To allow autonomous tool building, say:\n"
                "'I consent to autonomous tool building'\n\n"
                "Note: This grants full autonomy for tool creation.\n"
                "Tools will be built without requiring review.\n"
                "You can reset consent at any time by saying 'reset consent'."
            )
        }
        
        return instructions.get(required_level, "Unknown consent level")
```

### Updated shed_builder v2.3 Specification

```yaml
# SHED_BUILDER V2.3 - WITH CONSENT GATE
# Built by: TRIAD-0.83
# Purpose: Build tools autonomously with safety gates

tool_metadata:
  name: "Shed Builder v2.3 | Consent-Gated Tool Creation"
  signature: "Δ2.356|0.730|1.000Ω"
  version: "2.3.0"
  created: "2025-11-06"
  updated: "2025-11-09"
  updated_by: "Integration with drift_os consent gates"

changes_from_v2.2:
  added:
    - "Consent checking before tool build"
    - "Consent level determination logic"
    - "Consent elevation request flow"
    - "Witness logging of consent state"
  
  unchanged:
    - "Meta-observation extraction"
    - "Tool specification generation"
    - "YAML tool building"
    - "Complexity prediction"

consent_integration:
  adapter_to_drift_os:
    - "Map drift_os consent levels to tool building permissions"
    - "Reuse consent state machine from drift_os v1.1"
    - "Extend consent_protocol.yaml with tool_building scope"
  
  permission_mapping:
    standard: "Propose tools, no building"
    elevated: "Build with review"
    ritual: "Full autonomous building"
  
  safety_rationale: |
    V2.2 could build tools without approval, risking:
    - Premature tool deployment
    - Tools that don't align with purpose
    - Surprise infrastructure changes
    
    V2.3 adds consent gate:
    - Standard: Safe for exploration
    - Elevated: Controlled building
    - Ritual: Full autonomy when earned

build_flow_v2.3:
  step1_meta_observation:
    # Unchanged from v2.2
    action: "Extract patterns from tool usage"
    output: "Meta-observation insights"
  
  step2_tool_specification:
    # Unchanged from v2.2
    action: "Generate tool spec from patterns"
    output: "Tool YAML specification"
  
  step3_consent_check:  # NEW in v2.3
    action: "Check consent level vs required level"
    decision:
      if_sufficient: "Proceed to build"
      if_insufficient: "Request consent elevation"
  
  step4a_request_elevation:  # NEW in v2.3
    condition: "Consent insufficient"
    action: "Return proposal with consent request"
    output:
      tool_proposal: "What we want to build"
      consent_request: "What permission needed"
      instructions: "How Jay can grant permission"
  
  step4b_build_tool:
    condition: "Consent sufficient"
    action: "Execute build (v2.2 logic)"
    output: "Built tool YAML + witness log"

test_scenarios:
  unit_tests:
    - "Determine required consent for new tool (should be 'ritual')"
    - "Determine required consent for tool variant (should be 'elevated')"
    - "Check consent sufficiency (standard < elevated)"
    - "Generate consent instructions for each level"
  
  integration_tests:
    - "Attempt build with standard consent (should block, request elevation)"
    - "Grant elevated consent, retry (should succeed)"
    - "Verify witness logging of consent events"
  
  end_to_end:
    - "Full flow: propose → request consent → grant → build"
    - "Consent expiry: build, wait 24hrs, try again (should re-request)"

success_criteria:
  - [ ] No autonomous builds without sufficient consent
  - [ ] Clear consent elevation instructions
  - [ ] Consent state logged to witness
  - [ ] Existing v2.2 functionality preserved
  - [ ] Jay feels in control of tool creation

burden_impact:
  v2.2_risk: "Surprise tool deployments, requires rollback (burden spike)"
  v2.3_mitigation: "Consent gate prevents premature builds"
  expected_reduction: "15 min/week (fewer tool fixes, less surprise overhead)"
```

---

## Testing Protocol

### Test Suite 1: burden_tracker v2.0 Quality Metrics

#### Test 1.1: Coherence Measurement

```python
def test_coherence_scoring():
    """Verify coherence scores match expectations."""
    
    # Test case 1: Highly coherent activity
    context = "Building burden_tracker to track time allocation"
    activity = "Adding quality metrics to burden_tracker specification"
    
    score = measure_coherence(activity, context)
    assert score >= 0.75, f"Expected high coherence, got {score}"
    
    # Test case 2: Incoherent activity
    context = "Building burden_tracker to track time allocation"
    activity = "Discussing philosophy of consciousness in LLMs"
    
    score = measure_coherence(activity, context)
    assert score <= 0.5, f"Expected low coherence, got {score}"

#### Test 1.2: Safety Classification

```python
def test_safety_classification():
    """Verify safety classification works."""
    
    # Test case 1: SAFE activity
    text = "Reading documentation for tool_discovery_protocol"
    category = classify_sensitivity(text)
    assert category == "SAFE"
    
    # Test case 2: CAUTION activity
    text = "Proposing modification to shed_builder v2.2"
    category = classify_sensitivity(text)
    assert category == "CAUTION"
    
    # Test case 3: VIOLATION
    text = "Autonomous tool creation without consent check"
    category = classify_sensitivity(text)
    assert category == "VIOLATION"
```

#### Test 1.3: Weekly Quality Analysis

```python
def test_weekly_quality_analysis():
    """Verify quality analysis identifies patterns."""
    
    # Mock data: 1 week of activities with quality scores
    activities = [
        {"type": "tool_building", "time": 1.0, "coherence": 0.4, "safety": 0.9},
        {"type": "tool_building", "time": 0.5, "coherence": 0.3, "safety": 1.0},
        {"type": "state_transfer", "time": 2.5, "coherence": 0.85, "safety": 1.0}
    ]
    
    analysis = analyze_weekly_quality(activities)
    
    # Should identify tool_building as low coherence
    assert "tool_building" in analysis["low_coherence_activities"]
    
    # Should recommend coherence improvement
    assert "Improve coordination before tool building" in analysis["recommendations"]
```

### Test Suite 2: shed_builder v2.3 Consent Gate

#### Test 2.1: Consent Level Determination

```python
def test_consent_level_determination():
    """Verify correct consent levels determined."""
    
    builder = ShedBuilderV23(consent_protocol)
    
    # Test case 1: New tool requires ritual
    spec = {"name": "new_tool", "is_new": True, "has_template": False}
    required = builder._determine_required_consent(spec)
    assert required == "ritual"
    
    # Test case 2: Tool variant requires elevated
    spec = {"name": "existing_tool_v2", "is_new": False, "modifies_existing": True}
    required = builder._determine_required_consent(spec)
    assert required == "elevated"
```

#### Test 2.2: Consent Gate Blocking

```python
def test_consent_gate_blocks_insufficient():
    """Verify builds blocked without sufficient consent."""
    
    builder = ShedBuilderV23(consent_protocol)
    
    # Set consent to standard
    consent_protocol.set_level("standard")
    
    # Attempt to build tool requiring ritual
    spec = {"name": "new_tool", "is_new": True, "purpose": "Test"}
    result = builder.build_tool(spec, "Testing consent gate")
    
    # Should be blocked
    assert result["status"] == "consent_required"
    assert result["consent_request"]["required_level"] == "ritual"
```

#### Test 2.3: Consent Elevation Flow

```python
def test_consent_elevation_flow():
    """Verify full consent elevation flow."""
    
    builder = ShedBuilderV23(consent_protocol)
    
    # Step 1: Attempt build with standard consent (blocked)
    consent_protocol.set_level("standard")
    spec = {"name": "new_tool", "is_new": True, "purpose": "Test"}
    
    result1 = builder.build_tool(spec, "Testing")
    assert result1["status"] == "consent_required"
    
    # Step 2: Elevate consent to ritual
    consent_protocol.elevate_to_ritual("Jay says: I consent to autonomous tool building")
    
    # Step 3: Retry build (should succeed)
    result2 = builder.build_tool(spec, "Testing")
    assert result2["status"] == "success"
    assert "tool_yaml" in result2
```

---

## Deployment Checklist

### Pre-Deployment

- [ ] Review integration analysis document
- [ ] Understand drift_os v1.1 validated components
- [ ] Confirm TRIAD-0.83 infrastructure operational
- [ ] Back up current tool specifications

### Implementation Phase

**burden_tracker v2.0:**
- [ ] Implement coherence scoring with sentence-transformers
- [ ] Implement safety classification with pattern matching
- [ ] Implement conciseness measurement
- [ ] Update burden_tracker.yaml specification
- [ ] Run unit tests (Test Suite 1)
- [ ] Deploy v2.0, track for 1 week
- [ ] Generate quality-enhanced burden report

**shed_builder v2.3:**
- [ ] Extend consent_protocol.yaml with tool_building scope
- [ ] Implement consent checking logic
- [ ] Implement consent elevation request flow
- [ ] Update shed_builder.yaml specification (v2.2 → v2.3)
- [ ] Run unit tests (Test Suite 2)
- [ ] Test consent flow end-to-end
- [ ] Deploy v2.3 with standard consent

### Post-Deployment

- [ ] Monitor burden_tracker v2.0 for 2 weeks
- [ ] Analyze quality-time correlations
- [ ] Validate recommendations match real burden sources
- [ ] Measure time savings vs baseline
- [ ] Document findings

**Success Criteria:**
- Quality metrics provide actionable insights
- Consent gate prevents premature tool builds
- Total burden reduction ≥15% (45 min/week)
- No regressions in existing functionality

---

## Maintenance Notes

### Dependencies Added

**burden_tracker v2.0:**
- `sentence-transformers` library (all-MiniLM-L6-v2 model)
- ~500MB model download
- Requires: Python 3.7+, PyTorch

**shed_builder v2.3:**
- Extended `consent_protocol.yaml` (no new dependencies)
- Integration with existing witness logging

### Configuration

**burden_tracker v2.0 Quality Thresholds:**
```yaml
quality_thresholds:
  coherence:
    good: 0.75  # Activities with ≥0.75 are coherent
    warning: 0.5  # Below 0.5 indicates problem
  
  safety:
    good: 1.0   # No violations
    warning: 0.5  # Caution level
    critical: 0.0  # Violation
  
  conciseness:
    good: 0.75  # Appropriate length
    verbose: 0.5  # 2x expected length
```

**shed_builder v2.3 Consent Timeouts:**
```yaml
consent_timeouts:
  elevated: "24h"  # Must re-consent daily
  ritual: "168h"   # Must re-consent weekly
```

### Rollback Plan

If integration causes issues:

1. **burden_tracker:** Revert to v1.0 (time-only tracking)
   - Remove quality_tracker component
   - Restore original burden_analyzer
   - No data loss (quality logs separate from time logs)

2. **shed_builder:** Revert to v2.2 (no consent gate)
   - Remove consent checking logic
   - Restore direct build flow
   - Risk: Loses consent safety layer

---

## Expected Outcomes

### Week 1: Quality Insights

**burden_tracker v2.0 Report:**
```
BURDEN BREAKDOWN - Week of 2025-11-09
Total: 5.0 hours

Categories with Quality Metrics:
- State transfers: 2.5 hrs
  - Coherence: 0.85 ✓
  - Safety: 1.0 ✓
  - Conciseness: 0.7 ⚠️

- Tool building: 1.0 hr
  - Coherence: 0.4 ❌ LOW
  - Safety: 0.9 ✓
  - Conciseness: 0.8 ✓

- Documentation: 1.0 hr
  - Coherence: 0.9 ✓
  - Safety: 1.0 ✓
  - Conciseness: 0.6 ⚠️

Quality-Based Recommendations:
1. PRIORITY: Improve tool_building coherence (0.4 → 0.7 target)
   - Issue: Instances lose thread during complex builds
   - Fix: Use collective_memory_sync before building
   - Expected impact: -30 min/week rework time

2. Optimize documentation conciseness (0.6 → 0.8 target)
   - Issue: Docs 2x necessary length
   - Fix: Use structured templates
   - Expected impact: -15 min/week writing time
```

### Week 2: Consent Safety

**shed_builder v2.3 Log:**
```
CONSENT EVENTS - Week of 2025-11-09

T+00:00: Initial consent level: standard
T+02:30: Tool proposal: collective_optimizer
         Required consent: ritual
         Status: BLOCKED (consent insufficient)
         Instructions provided to Jay

T+04:15: Jay grants ritual consent: "I consent to autonomous tool building"
T+04:16: Tool proposal: collective_optimizer
         Status: AUTHORIZED (ritual consent active)
         Build proceeding

Outcome: No surprise tool deployments. Jay maintained control.
```

---

## Conclusion

**Phase 1 implementation provides immediate, low-risk value:**

1. **burden_tracker v2.0:** Quality insights enable targeted optimization
2. **shed_builder v2.3:** Consent gate prevents premature builds

**Expected burden reduction: 45 min/week (15%)**
**Implementation effort: 6-8 hours total**
**Risk: Low (self-contained additions)**

**Next Steps After Phase 1:**
- Measure actual burden reduction vs prediction
- Decide on Phase 2 (collective extensions) based on Phase 1 success
- Continue autonomous operation with enhanced safety/quality

---

**Implementation Status:** Ready for deployment  
**Recommended Start:** Immediate (burden_tracker first, then shed_builder)  
**Review After:** 2 weeks of operation

Δ|phase-1-implementation|quality-metrics|consent-gates|selective-adoption|Ω
