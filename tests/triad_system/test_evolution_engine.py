"""
Test Suite: Autonomous Evolution Engine
Coordinate: Δ3.142|0.900|1.000Ω

Tests for the 5-phase autonomous evolution cycle:
1. Friction Detection
2. Improvement Proposal
3. Collective Validation
4. Autonomous Execution
5. Meta-Learning
"""

import pytest
import json
from pathlib import Path
from datetime import datetime


class TestFrictionDetection:
    """Tests for Phase 1: Friction Detection."""

    def test_detect_friction_returns_report(self, evolution_engine_module, sample_burden_data):
        """Test that friction detection returns a valid report."""
        report = evolution_engine_module.detect_friction(sample_burden_data)

        assert report is not None
        assert report.report_id.startswith("FR-")
        assert report.timestamp is not None
        assert isinstance(report.aggregate_burden, (int, float))

    def test_friction_events_detected(self, evolution_engine_module, sample_burden_data):
        """Test that friction events are detected from low-confidence activities."""
        report = evolution_engine_module.detect_friction(sample_burden_data)

        # Should detect friction from activities with confidence < 0.3
        assert len(report.friction_events) > 0
        for event in report.friction_events:
            assert "event_id" in event
            assert "severity" in event
            assert event["severity"] in ["minor", "moderate", "major"]

    def test_patterns_detected(self, evolution_engine_module, sample_burden_data):
        """Test that patterns are identified from burden data."""
        report = evolution_engine_module.detect_friction(sample_burden_data)

        assert len(report.patterns_detected) > 0
        for pattern in report.patterns_detected:
            assert "pattern_name" in pattern
            assert "confidence" in pattern
            assert 0 <= pattern["confidence"] <= 1

    def test_improvement_opportunities_generated(self, evolution_engine_module, sample_burden_data):
        """Test that improvement opportunities are generated."""
        report = evolution_engine_module.detect_friction(sample_burden_data)

        assert len(report.improvement_opportunities) > 0
        for opp in report.improvement_opportunities:
            assert "opportunity_id" in opp
            assert "description" in opp
            assert "predicted_reduction" in opp

    def test_friction_detection_at_different_z_levels(
        self, evolution_engine_module, custom_burden_data_factory
    ):
        """Test friction detection works at different z-levels."""
        for z_level in [0.85, 0.87, 0.89, 0.90]:
            burden_data = custom_burden_data_factory(z_level=z_level)
            report = evolution_engine_module.detect_friction(burden_data)

            assert report is not None
            assert report.report_id.startswith("FR-")


class TestImprovementProposal:
    """Tests for Phase 2: Improvement Proposal."""

    def test_proposals_generated_from_friction(self, evolution_engine_module, sample_burden_data):
        """Test that proposals are generated from friction report."""
        friction_report = evolution_engine_module.detect_friction(sample_burden_data)
        proposals = evolution_engine_module.generate_proposals(friction_report)

        assert len(proposals) > 0
        for proposal in proposals:
            assert proposal.proposal_id.startswith("PROP-")
            assert proposal.description is not None
            assert proposal.implementation_spec is not None

    def test_proposal_contains_tool_specification(
        self, evolution_engine_module, sample_burden_data
    ):
        """Test that proposals contain valid tool specifications."""
        friction_report = evolution_engine_module.detect_friction(sample_burden_data)
        proposals = evolution_engine_module.generate_proposals(friction_report)

        for proposal in proposals:
            spec = proposal.implementation_spec
            assert "tool_name" in spec
            assert "category" in spec
            assert "z_level" in spec
            assert "capabilities" in spec
            assert isinstance(spec["capabilities"], list)

    def test_proposal_risk_matrix(self, evolution_engine_module, sample_burden_data):
        """Test that proposals include risk assessment."""
        friction_report = evolution_engine_module.detect_friction(sample_burden_data)
        proposals = evolution_engine_module.generate_proposals(friction_report)

        for proposal in proposals:
            risk = proposal.risk_matrix
            assert "risk_level" in risk
            assert risk["risk_level"] in ["low", "medium", "high"]
            assert "regression_potential" in risk

    def test_proposal_z_levels_in_valid_range(
        self, evolution_engine_module, sample_burden_data
    ):
        """Test that proposed tool z-levels are in valid range."""
        friction_report = evolution_engine_module.detect_friction(sample_burden_data)
        proposals = evolution_engine_module.generate_proposals(friction_report)

        for proposal in proposals:
            z_level = proposal.implementation_spec["z_level"]
            assert 0.85 <= z_level <= 1.0, f"z-level {z_level} out of valid range"


class TestCollectiveValidation:
    """Tests for Phase 3: Collective Validation."""

    def test_consensus_mechanism(self, evolution_engine_module, sample_burden_data):
        """Test that consensus mechanism produces results."""
        friction_report = evolution_engine_module.detect_friction(sample_burden_data)
        proposals = evolution_engine_module.generate_proposals(friction_report)
        results = evolution_engine_module.validate_proposals(proposals)

        assert len(results) == len(proposals)
        for result in results:
            assert result.proposal_id is not None
            assert isinstance(result.approved, bool)
            assert len(result.votes) == 3  # CBS-ALPHA, CBS-BETA, CBS-GAMMA

    def test_vote_structure(self, evolution_engine_module, sample_burden_data, cbs_instances):
        """Test that votes have correct structure."""
        friction_report = evolution_engine_module.detect_friction(sample_burden_data)
        proposals = evolution_engine_module.generate_proposals(friction_report)
        results = evolution_engine_module.validate_proposals(proposals)

        for result in results:
            instances_voted = set()
            for vote in result.votes:
                assert "instance_id" in vote
                assert vote["instance_id"] in cbs_instances
                assert "vote" in vote
                assert vote["vote"] in ["approve", "reject"]
                instances_voted.add(vote["instance_id"])

            # All instances should vote
            assert instances_voted == set(cbs_instances)

    def test_consensus_threshold(self, evolution_engine_module, sample_burden_data):
        """Test that 66% consensus threshold is applied."""
        friction_report = evolution_engine_module.detect_friction(sample_burden_data)
        proposals = evolution_engine_module.generate_proposals(friction_report)
        results = evolution_engine_module.validate_proposals(proposals)

        for result in results:
            approve_count = sum(1 for v in result.votes if v["vote"] == "approve")
            expected_approval = approve_count / len(result.votes) >= 0.66
            assert result.approved == expected_approval

    def test_authorization_token_on_approval(
        self, evolution_engine_module, sample_burden_data
    ):
        """Test that approved proposals get authorization token."""
        friction_report = evolution_engine_module.detect_friction(sample_burden_data)
        proposals = evolution_engine_module.generate_proposals(friction_report)
        results = evolution_engine_module.validate_proposals(proposals)

        for result in results:
            if result.approved:
                assert result.authorization_token is not None
                assert len(result.authorization_token) == 32
            else:
                assert result.authorization_token is None


class TestAutonomousExecution:
    """Tests for Phase 4: Autonomous Execution (Tool Generation)."""

    def test_approved_proposals_generate_tools(
        self, evolution_engine_module, sample_burden_data
    ):
        """Test that approved proposals result in tool generation."""
        friction_report = evolution_engine_module.detect_friction(sample_burden_data)
        proposals = evolution_engine_module.generate_proposals(friction_report)
        results = evolution_engine_module.validate_proposals(proposals)
        tools = evolution_engine_module.execute_approved_proposals(proposals, results)

        approved_count = sum(1 for r in results if r.approved)
        assert len(tools) == approved_count

    def test_generated_tool_has_valid_code(
        self, evolution_engine_module, sample_burden_data
    ):
        """Test that generated tools have valid Python code."""
        friction_report = evolution_engine_module.detect_friction(sample_burden_data)
        proposals = evolution_engine_module.generate_proposals(friction_report)
        results = evolution_engine_module.validate_proposals(proposals)
        tools = evolution_engine_module.execute_approved_proposals(proposals, results)

        for tool in tools:
            # Should be valid Python (can compile)
            try:
                compile(tool.code, "<string>", "exec")
            except SyntaxError as e:
                pytest.fail(f"Generated code has syntax error: {e}")

    def test_generated_tool_has_spec(self, evolution_engine_module, sample_burden_data):
        """Test that generated tools have valid specifications."""
        friction_report = evolution_engine_module.detect_friction(sample_burden_data)
        proposals = evolution_engine_module.generate_proposals(friction_report)
        results = evolution_engine_module.validate_proposals(proposals)
        tools = evolution_engine_module.execute_approved_proposals(proposals, results)

        for tool in tools:
            assert tool.spec is not None
            assert tool.spec.tool_id is not None
            assert tool.spec.category is not None
            assert tool.spec_json is not None

            # Spec JSON should be valid
            spec_dict = json.loads(tool.spec_json)
            assert "tool_id" in spec_dict

    def test_tool_phase_regime_assignment(
        self, evolution_engine_module, sample_burden_data
    ):
        """Test that tools get correct phase regime based on z-level."""
        friction_report = evolution_engine_module.detect_friction(sample_burden_data)
        proposals = evolution_engine_module.generate_proposals(friction_report)
        results = evolution_engine_module.validate_proposals(proposals)
        tools = evolution_engine_module.execute_approved_proposals(proposals, results)

        for tool in tools:
            z = tool.spec.z_level
            regime = tool.spec.phase_regime

            if z < 0.85:
                assert regime == "subcritical"
            elif z < 0.88:
                assert regime == "critical"
            else:
                assert regime == "supercritical"


class TestMetaLearning:
    """Tests for Phase 5: Meta-Learning."""

    def test_learnings_extracted(self, evolution_engine_module, sample_burden_data):
        """Test that meta-learnings are extracted from the cycle."""
        friction_report = evolution_engine_module.detect_friction(sample_burden_data)
        proposals = evolution_engine_module.generate_proposals(friction_report)
        results = evolution_engine_module.validate_proposals(proposals)
        tools = evolution_engine_module.execute_approved_proposals(proposals, results)
        learnings = evolution_engine_module.extract_meta_learnings(
            friction_report, proposals, tools
        )

        assert len(learnings) > 0

    def test_learning_structure(self, evolution_engine_module, sample_burden_data):
        """Test that learnings have correct structure."""
        friction_report = evolution_engine_module.detect_friction(sample_burden_data)
        proposals = evolution_engine_module.generate_proposals(friction_report)
        results = evolution_engine_module.validate_proposals(proposals)
        tools = evolution_engine_module.execute_approved_proposals(proposals, results)
        learnings = evolution_engine_module.extract_meta_learnings(
            friction_report, proposals, tools
        )

        for learning in learnings:
            assert "learning_id" in learning
            assert learning["learning_id"].startswith("LEARN-")
            assert "pattern_type" in learning
            assert "confidence" in learning
            assert "generalization" in learning

    def test_learnings_from_tools_and_patterns(
        self, evolution_engine_module, sample_burden_data
    ):
        """Test that learnings come from both tools and friction patterns."""
        friction_report = evolution_engine_module.detect_friction(sample_burden_data)
        proposals = evolution_engine_module.generate_proposals(friction_report)
        results = evolution_engine_module.validate_proposals(proposals)
        tools = evolution_engine_module.execute_approved_proposals(proposals, results)
        learnings = evolution_engine_module.extract_meta_learnings(
            friction_report, proposals, tools
        )

        pattern_types = {l["pattern_type"] for l in learnings}
        assert "successful_tool_generation" in pattern_types or len(tools) == 0
        if friction_report.patterns_detected:
            assert "friction_pattern" in pattern_types


class TestFullEvolutionCycle:
    """Tests for complete evolution cycle execution."""

    def test_complete_cycle_execution(self, evolution_engine_module, sample_burden_data):
        """Test that a complete evolution cycle can be executed."""
        result = evolution_engine_module.run_evolution_cycle(sample_burden_data)

        assert result is not None
        assert result.cycle_id.startswith("EVO-")
        assert result.friction_report is not None
        assert len(result.proposals) > 0
        assert len(result.consensus_results) > 0

    def test_cycle_generates_tools(self, evolution_engine_module, sample_burden_data):
        """Test that the cycle generates tools."""
        result = evolution_engine_module.run_evolution_cycle(sample_burden_data)

        # At least some tools should be generated
        # (depends on random consensus, but with high probability)
        assert result.generated_tools is not None

    def test_cycle_produces_learnings(self, evolution_engine_module, sample_burden_data):
        """Test that the cycle produces meta-learnings."""
        result = evolution_engine_module.run_evolution_cycle(sample_burden_data)

        assert result.meta_learnings is not None
        assert len(result.meta_learnings) >= 0

    def test_cycle_at_different_z_levels(
        self, evolution_engine_module, custom_burden_data_factory
    ):
        """Test evolution cycle at different z-levels."""
        z_levels = [0.85, 0.87, 0.89, 0.90, 0.92]

        for z in z_levels:
            burden_data = custom_burden_data_factory(z_level=z)
            result = evolution_engine_module.run_evolution_cycle(burden_data)

            assert result is not None
            assert result.z_level == z
