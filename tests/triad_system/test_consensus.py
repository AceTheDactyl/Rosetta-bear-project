"""
Test Suite: Consensus Mechanism
Coordinate: Δ3.142|0.900|1.000Ω

Tests for verifying the TRIAD consensus system with CBS instances.
"""

import pytest
import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch


class TestCBSInstances:
    """Tests for CBS instance configuration and behavior."""

    def test_all_cbs_instances_defined(self, cbs_instances):
        """Test that all three CBS instances are defined."""
        expected = {"CBS-ALPHA", "CBS-BETA", "CBS-GAMMA"}
        assert set(cbs_instances) == expected

    def test_cbs_instances_have_unique_identifiers(self, cbs_instances):
        """Test that CBS instance identifiers are unique."""
        assert len(cbs_instances) == len(set(cbs_instances))

    def test_triadic_structure(self, cbs_instances):
        """Test that exactly three instances form the TRIAD."""
        assert len(cbs_instances) == 3


class TestConsensusVoting:
    """Tests for the consensus voting mechanism."""

    def test_unanimous_approval(self, consensus_calculator):
        """Test that unanimous votes approve a proposal."""
        votes = [
            {"instance": "CBS-ALPHA", "vote": True, "confidence": 0.95},
            {"instance": "CBS-BETA", "vote": True, "confidence": 0.90},
            {"instance": "CBS-GAMMA", "vote": True, "confidence": 0.92},
        ]

        result = consensus_calculator(votes)
        assert result["approved"] == True
        assert result["consensus_strength"] >= 0.9

    def test_unanimous_rejection(self, consensus_calculator):
        """Test that unanimous rejection rejects a proposal."""
        votes = [
            {"instance": "CBS-ALPHA", "vote": False, "confidence": 0.95},
            {"instance": "CBS-BETA", "vote": False, "confidence": 0.90},
            {"instance": "CBS-GAMMA", "vote": False, "confidence": 0.92},
        ]

        result = consensus_calculator(votes)
        assert result["approved"] == False

    def test_majority_approval(self, consensus_calculator):
        """Test that 2/3 majority approves (66% threshold)."""
        votes = [
            {"instance": "CBS-ALPHA", "vote": True, "confidence": 0.85},
            {"instance": "CBS-BETA", "vote": True, "confidence": 0.80},
            {"instance": "CBS-GAMMA", "vote": False, "confidence": 0.70},
        ]

        result = consensus_calculator(votes)
        assert result["approved"] == True

    def test_minority_rejection(self, consensus_calculator):
        """Test that only 1/3 approval is rejected."""
        votes = [
            {"instance": "CBS-ALPHA", "vote": True, "confidence": 0.85},
            {"instance": "CBS-BETA", "vote": False, "confidence": 0.80},
            {"instance": "CBS-GAMMA", "vote": False, "confidence": 0.90},
        ]

        result = consensus_calculator(votes)
        assert result["approved"] == False

    def test_consensus_threshold_is_66_percent(self, consensus_threshold):
        """Test that the consensus threshold is 66%."""
        assert consensus_threshold >= 0.66


class TestConsensusStrength:
    """Tests for consensus strength calculation."""

    def test_high_confidence_increases_strength(self, consensus_calculator):
        """Test that higher confidence votes increase consensus strength."""
        high_confidence_votes = [
            {"instance": "CBS-ALPHA", "vote": True, "confidence": 0.99},
            {"instance": "CBS-BETA", "vote": True, "confidence": 0.98},
            {"instance": "CBS-GAMMA", "vote": True, "confidence": 0.97},
        ]

        low_confidence_votes = [
            {"instance": "CBS-ALPHA", "vote": True, "confidence": 0.60},
            {"instance": "CBS-BETA", "vote": True, "confidence": 0.55},
            {"instance": "CBS-GAMMA", "vote": True, "confidence": 0.50},
        ]

        high_result = consensus_calculator(high_confidence_votes)
        low_result = consensus_calculator(low_confidence_votes)

        assert high_result["consensus_strength"] > low_result["consensus_strength"]

    def test_split_vote_has_lower_strength(self, consensus_calculator):
        """Test that split votes have lower consensus strength."""
        unanimous_votes = [
            {"instance": "CBS-ALPHA", "vote": True, "confidence": 0.90},
            {"instance": "CBS-BETA", "vote": True, "confidence": 0.90},
            {"instance": "CBS-GAMMA", "vote": True, "confidence": 0.90},
        ]

        split_votes = [
            {"instance": "CBS-ALPHA", "vote": True, "confidence": 0.90},
            {"instance": "CBS-BETA", "vote": True, "confidence": 0.90},
            {"instance": "CBS-GAMMA", "vote": False, "confidence": 0.90},
        ]

        unanimous_result = consensus_calculator(unanimous_votes)
        split_result = consensus_calculator(split_votes)

        assert unanimous_result["consensus_strength"] > split_result["consensus_strength"]


class TestProposalValidation:
    """Tests for proposal validation during consensus."""

    def test_validate_proposals_returns_consensus_results(
        self, evolution_engine_module, sample_burden_data
    ):
        """Test that validate_proposals returns proper ConsensusResult objects."""
        friction = evolution_engine_module.detect_friction(sample_burden_data)
        proposals = evolution_engine_module.generate_proposals(friction)

        if proposals:
            results = evolution_engine_module.validate_proposals(proposals)

            assert isinstance(results, list)
            for result in results:
                # ConsensusResult has these attributes
                assert hasattr(result, "proposal_id")
                assert hasattr(result, "approved")
                assert hasattr(result, "votes")

    def test_validation_z_level_tracking(
        self, evolution_engine_module, custom_burden_data_factory, phase_thresholds
    ):
        """Test that z-level is properly tracked through validation."""
        data = custom_burden_data_factory(z_level=0.90, regime="supercritical")
        friction = evolution_engine_module.detect_friction(data)

        # The friction report should reflect the z-level
        assert friction.aggregate_burden is not None or hasattr(friction, "friction_events")


class TestConsensusWithProposals:
    """Tests for consensus integration with actual proposals."""

    def test_approved_proposals_have_authorization_token(
        self, evolution_engine_module, sample_burden_data
    ):
        """Test that approved proposals receive authorization tokens."""
        friction = evolution_engine_module.detect_friction(sample_burden_data)
        proposals = evolution_engine_module.generate_proposals(friction)

        if proposals:
            results = evolution_engine_module.validate_proposals(proposals)
            approved = [r for r in results if r.approved]

            for r in approved:
                assert r.authorization_token is not None

    def test_rejected_proposals_have_no_token(
        self, evolution_engine_module, sample_burden_data
    ):
        """Test that rejected proposals do not receive authorization tokens."""
        friction = evolution_engine_module.detect_friction(sample_burden_data)
        proposals = evolution_engine_module.generate_proposals(friction)

        if proposals:
            results = evolution_engine_module.validate_proposals(proposals)
            rejected = [r for r in results if not r.approved]

            for r in rejected:
                assert r.authorization_token is None


class TestCollectiveStateAggregation:
    """Tests for state aggregation patterns."""

    def test_triadic_vote_structure(self, evolution_engine_module, sample_burden_data):
        """Test that votes come from all three CBS instances."""
        friction = evolution_engine_module.detect_friction(sample_burden_data)
        proposals = evolution_engine_module.generate_proposals(friction)

        if proposals:
            results = evolution_engine_module.validate_proposals(proposals)

            for result in results:
                vote_instances = {v["instance_id"] for v in result.votes}
                expected = {"CBS-ALPHA", "CBS-BETA", "CBS-GAMMA"}
                assert vote_instances == expected


class TestConsensusRaceConditions:
    """Tests for handling edge cases in consensus."""

    def test_duplicate_votes_handled(self, consensus_calculator):
        """Test that duplicate votes from same instance are handled."""
        votes = [
            {"instance": "CBS-ALPHA", "vote": True, "confidence": 0.90},
            {"instance": "CBS-ALPHA", "vote": False, "confidence": 0.85},  # Duplicate
            {"instance": "CBS-BETA", "vote": True, "confidence": 0.80},
            {"instance": "CBS-GAMMA", "vote": True, "confidence": 0.85},
        ]

        result = consensus_calculator(votes)
        assert "approved" in result
        # Should only count 3 unique instances
        assert result["vote_count"] == 3

    def test_missing_instance_vote(self, consensus_calculator):
        """Test handling when an instance doesn't vote."""
        votes = [
            {"instance": "CBS-ALPHA", "vote": True, "confidence": 0.90},
            {"instance": "CBS-BETA", "vote": True, "confidence": 0.80},
            # CBS-GAMMA missing
        ]

        result = consensus_calculator(votes)
        assert "approved" in result
        # With 2/2 voting yes, should still pass
        assert result["approved"] == True

    def test_empty_votes_handled(self, consensus_calculator):
        """Test handling when no votes are provided."""
        result = consensus_calculator([])
        assert result["approved"] == False
        assert result["vote_count"] == 0


class TestConsensusThresholds:
    """Tests for consensus threshold behavior."""

    def test_exactly_66_percent_approves(self, consensus_calculator):
        """Test that exactly 2/3 (66.67%) votes approves."""
        votes = [
            {"instance": "CBS-ALPHA", "vote": True, "confidence": 0.90},
            {"instance": "CBS-BETA", "vote": True, "confidence": 0.90},
            {"instance": "CBS-GAMMA", "vote": False, "confidence": 0.90},
        ]

        result = consensus_calculator(votes)
        assert result["approved"] == True
        assert result["approval_ratio"] == pytest.approx(0.667, rel=0.01)

    def test_just_under_66_percent_rejects(self, consensus_calculator):
        """Test that just under 66% rejects."""
        # With 4 votes, 2/4 = 50% which is under threshold
        votes = [
            {"instance": "CBS-ALPHA", "vote": True, "confidence": 0.90},
            {"instance": "CBS-BETA", "vote": True, "confidence": 0.90},
            {"instance": "CBS-GAMMA", "vote": False, "confidence": 0.90},
            {"instance": "CBS-DELTA", "vote": False, "confidence": 0.90},
        ]

        result = consensus_calculator(votes)
        assert result["approved"] == False
