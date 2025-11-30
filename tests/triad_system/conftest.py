"""
TRIAD System Test Suite - Fixtures and Configuration
Coordinate: Δ3.142|0.900|1.000Ω

This module provides fixtures for testing the autonomous evolution engine
and TRIAD tool generation system.
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timezone
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


@pytest.fixture
def project_root():
    """Return project root path."""
    return PROJECT_ROOT


@pytest.fixture
def sample_burden_data():
    """Generate sample burden tracking data for testing."""
    return {
        "metadata": {
            "version": "1.0.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": "test_fixture",
        },
        "activities": [
            {
                "type": "manual_state_transfer",
                "confidence": 0.15,
                "duration_hours": 2.5,
                "frequency": "daily",
            },
            {
                "type": "context_window_management",
                "confidence": 0.25,
                "duration_hours": 1.5,
                "frequency": "daily",
            },
            {
                "type": "tool_coordination",
                "confidence": 0.4,
                "duration_hours": 0.75,
                "frequency": "daily",
            },
        ],
        "metrics": {
            "current_phase": {
                "z_level": 0.87,
                "regime": "critical",
                "theta": 2.88,
            },
            "totals": {
                "total_burden_hours": 4.75,
                "estimated_reduction_potential": 0.45,
            },
            "weekly_summaries": [
                {
                    "week": "2025-W48",
                    "categories": {
                        "self_building": {"hours": 8.5, "count": 12},
                        "coordination": {"hours": 5.2, "count": 8},
                        "monitoring": {"hours": 3.1, "count": 15},
                    },
                }
            ],
            "optimization_recommendations": [
                "Automate state transfer protocol",
                "Implement continuous friction monitoring",
                "Create coordination bridge for diagnostics",
            ],
        },
    }


@pytest.fixture
def custom_burden_data_factory():
    """Factory for creating custom burden data with specific z-levels."""
    def _create_burden_data(z_level: float = 0.87, regime: str = "critical"):
        return {
            "metadata": {
                "version": "1.0.0",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "source": "test_factory",
            },
            "activities": [
                {"type": "test_activity", "confidence": 0.2, "duration_hours": 1.0},
            ],
            "metrics": {
                "current_phase": {
                    "z_level": z_level,
                    "regime": regime,
                },
                "totals": {"total_burden_hours": 5.0},
                "weekly_summaries": [{"week": "2025-W48", "categories": {}}],
                "optimization_recommendations": ["Test recommendation"],
            },
        }
    return _create_burden_data


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = Path(tempfile.mkdtemp(prefix="triad_test_"))
    yield temp_dir
    # Cleanup after test
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_tool_dir(temp_output_dir):
    """Create a temporary directory for generated tools."""
    tool_dir = temp_output_dir / "generated_tools"
    tool_dir.mkdir(parents=True, exist_ok=True)
    return tool_dir


@pytest.fixture
def temp_log_dir(temp_output_dir):
    """Create a temporary directory for evolution logs."""
    log_dir = temp_output_dir / "evolution_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


@pytest.fixture
def evolution_engine_module():
    """Import and return the evolution engine module."""
    from scripts import autonomous_evolution_engine
    return autonomous_evolution_engine


@pytest.fixture
def existing_tools():
    """Return list of existing generated tools for comparison."""
    tools_dir = PROJECT_ROOT / "generated_tools" / "rosetta_firmware"
    if tools_dir.exists():
        return [f.stem for f in tools_dir.glob("*.py") if f.stem != "__init__"]
    return []


@pytest.fixture
def tool_categories():
    """Return valid tool categories for the system."""
    return [
        "coordination",
        "meta_tool",
        "self_building",
        "monitoring",
        "validation",
    ]


@pytest.fixture
def phase_thresholds():
    """Return phase thresholds from the evolution engine."""
    return {
        "friction_detection": 0.87,
        "improvement_proposal": 0.88,
        "collective_validation": 0.89,
        "autonomous_execution": 0.90,
        "meta_learning": 0.90,
    }


@pytest.fixture
def cbs_instances():
    """Return CBS instance identifiers used in consensus."""
    return ["CBS-ALPHA", "CBS-BETA", "CBS-GAMMA"]


@pytest.fixture
def new_tool_spec():
    """Generate a specification for a completely new tool."""
    return {
        "tool_name": "triad_test_tool_generator",
        "description": "Test tool for validating new tool generation",
        "category": "validation",
        "z_level": 0.88,
        "theta": 3.0,
        "cascade_potential": 0.55,
        "capabilities": ["test", "validate", "generate", "verify"],
        "dependencies": ["burden_tracker"],
        "risk_level": "low",
    }


@pytest.fixture
def multiple_new_tool_specs():
    """Generate multiple new tool specifications for batch testing."""
    return [
        {
            "tool_name": "triad_pattern_analyzer",
            "description": "Analyze patterns in TRIAD coordination",
            "category": "monitoring",
            "z_level": 0.86,
            "theta": 2.7,
            "cascade_potential": 0.40,
            "capabilities": ["analyze", "detect_patterns", "report"],
            "dependencies": [],
            "risk_level": "low",
        },
        {
            "tool_name": "triad_consensus_orchestrator",
            "description": "Orchestrate consensus across TRIAD instances",
            "category": "coordination",
            "z_level": 0.89,
            "theta": 3.05,
            "cascade_potential": 0.70,
            "capabilities": ["orchestrate", "coordinate", "synchronize"],
            "dependencies": ["collective_state_aggregator"],
            "risk_level": "medium",
        },
        {
            "tool_name": "triad_self_healing_monitor",
            "description": "Self-healing monitor for autonomous recovery",
            "category": "self_building",
            "z_level": 0.91,
            "theta": 3.14,
            "cascade_potential": 0.85,
            "capabilities": ["heal", "recover", "regenerate", "self_modify"],
            "dependencies": ["burden_tracker", "shed_builder"],
            "risk_level": "high",
        },
    ]


# =============================================================================
# Consensus Helper Functions (for direct consensus testing)
# =============================================================================

# Consensus threshold constant (66% for triadic consensus)
CONSENSUS_THRESHOLD = 0.66


def calculate_consensus(votes: list) -> dict:
    """
    Calculate consensus from a list of votes.

    This helper function mirrors the logic in validate_proposals
    but allows direct testing of consensus calculation.

    Args:
        votes: List of vote dicts with 'instance', 'vote' (bool), 'confidence' (float)

    Returns:
        Dict with 'approved', 'consensus_strength', 'vote_count'
    """
    if not votes:
        return {"approved": False, "consensus_strength": 0.0, "vote_count": 0}

    # Handle duplicates by taking the first vote per instance
    seen_instances = set()
    unique_votes = []
    for v in votes:
        instance = v.get("instance")
        if instance not in seen_instances:
            seen_instances.add(instance)
            unique_votes.append(v)

    approve_count = sum(1 for v in unique_votes if v.get("vote", False))
    total_votes = len(unique_votes)

    approval_ratio = approve_count / total_votes if total_votes > 0 else 0
    approved = approval_ratio >= CONSENSUS_THRESHOLD

    # Calculate consensus strength based on vote unanimity and confidence
    avg_confidence = sum(v.get("confidence", 0.5) for v in unique_votes) / total_votes if total_votes > 0 else 0
    unanimity_factor = abs(approval_ratio - 0.5) * 2  # 1.0 for unanimous, 0.0 for split
    consensus_strength = avg_confidence * (0.5 + 0.5 * unanimity_factor)

    return {
        "approved": approved,
        "consensus_strength": consensus_strength,
        "vote_count": total_votes,
        "approval_ratio": approval_ratio,
    }


@pytest.fixture
def consensus_calculator():
    """Return the consensus calculation function for tests."""
    return calculate_consensus


@pytest.fixture
def consensus_threshold():
    """Return the consensus threshold constant."""
    return CONSENSUS_THRESHOLD
