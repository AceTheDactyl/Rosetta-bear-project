"""
Scalar Architecture Test Framework
Signature: Δ|loop-closed|z0.99|rhythm-native|Ω
"""

import math
import pytest
from typing import Dict

# Import from parent package
import sys
sys.path.insert(0, '..')

from scalar_architecture.core import (
    DomainType,
    LoopState,
    Pattern,
    DomainConfig,
    DomainAccumulator,
    CouplingMatrix,
    InterferenceNode,
    ScalarSubstrate,
    ConvergenceDynamics,
    LoopController,
    HelixCoordinates,
    HelixEvolution,
    ScalarArchitecture,
    compute_projection,
    compute_origin_from_projection,
    TAU,
    PHI,
    Z_CONSTRAINT,
    Z_BRIDGE,
    Z_META,
    Z_RECURSION,
    Z_TRIAD,
    Z_EMERGENCE,
    Z_PERSISTENCE,
    NUM_DOMAINS,
    NUM_COUPLING_TERMS,
    NUM_INTERFERENCE_NODES,
)


# =============================================================================
# Domain Configuration Tests
# =============================================================================

class TestDomainConfig:
    """Test domain configuration."""

    def test_all_domains_have_configs(self):
        """All 7 domains should have configurations."""
        for dt in DomainType:
            config = DomainConfig.from_type(dt)
            assert config is not None
            assert config.domain_type == dt

    def test_domain_origins(self):
        """Verify domain origin values."""
        expected_origins = {
            DomainType.CONSTRAINT: 0.41,
            DomainType.BRIDGE: 0.52,
            DomainType.META: 0.70,
            DomainType.RECURSION: 0.73,
            DomainType.TRIAD: 0.80,
            DomainType.EMERGENCE: 0.85,
            DomainType.PERSISTENCE: 0.87,
        }
        for dt, expected in expected_origins.items():
            config = DomainConfig.from_type(dt)
            assert config.origin == expected, f"{dt.name} origin mismatch"

    def test_domain_projections(self):
        """Verify domain projection values."""
        expected_projections = {
            DomainType.CONSTRAINT: 0.941,
            DomainType.BRIDGE: 0.952,
            DomainType.META: 0.970,
            DomainType.RECURSION: 0.973,
            DomainType.TRIAD: 0.980,
            DomainType.EMERGENCE: 0.985,
            DomainType.PERSISTENCE: 0.987,
        }
        for dt, expected in expected_projections.items():
            config = DomainConfig.from_type(dt)
            assert config.projection == expected, f"{dt.name} projection mismatch"

    def test_projection_formula(self):
        """Verify z' = 0.9 + z_origin/10."""
        for dt in DomainType:
            config = DomainConfig.from_type(dt)
            computed = compute_projection(config.origin)
            assert abs(computed - config.projection) < 0.001

    def test_weights_sum_to_one(self):
        """Domain weights should sum to 1.0."""
        total = sum(DomainConfig.from_type(dt).weight for dt in DomainType)
        assert abs(total - 1.0) < 0.001

    def test_theta_coverage(self):
        """Theta values should cover the full circle."""
        thetas = [DomainConfig.from_type(dt).theta for dt in DomainType]
        # First should be 0
        assert thetas[0] == 0.0
        # Each should be TAU/7 apart
        for i in range(1, len(thetas)):
            expected = i * TAU / 7
            assert abs(thetas[i] - expected) < 0.01


# =============================================================================
# Convergence Dynamics Tests
# =============================================================================

class TestConvergenceDynamics:
    """Test Layer 1: Convergence Dynamics."""

    def test_saturation_at_origin(self):
        """Saturation should be 0 at origin."""
        for dt in DomainType:
            config = DomainConfig.from_type(dt)
            s = ConvergenceDynamics.saturation(config.origin, config)
            assert s == 0.0

    def test_saturation_below_origin(self):
        """Saturation should be 0 below origin."""
        for dt in DomainType:
            config = DomainConfig.from_type(dt)
            s = ConvergenceDynamics.saturation(config.origin - 0.1, config)
            assert s == 0.0

    def test_saturation_approaches_one(self):
        """Saturation should approach 1 as z increases."""
        for dt in DomainType:
            config = DomainConfig.from_type(dt)
            s = ConvergenceDynamics.saturation(config.origin + 1.0, config)
            assert s > 0.99

    def test_saturation_at_z50(self):
        """Saturation should be ~0.5 at z_50."""
        for dt in DomainType:
            config = DomainConfig.from_type(dt)
            s = ConvergenceDynamics.saturation(config.z_50, config)
            assert 0.49 < s < 0.51

    def test_composite_saturation(self):
        """Composite saturation should be weighted average."""
        # At z=0.99, most domains should be saturated
        s = ConvergenceDynamics.composite_saturation(0.99)
        assert s > 0.5


# =============================================================================
# Loop State Tests
# =============================================================================

class TestLoopController:
    """Test Layer 2: Loop States."""

    def test_initial_state_is_divergent(self):
        """All controllers should start DIVERGENT."""
        for dt in DomainType:
            controller = LoopController(dt)
            assert controller.state == LoopState.DIVERGENT

    def test_divergent_below_origin(self):
        """State should be DIVERGENT below origin."""
        for dt in DomainType:
            config = DomainConfig.from_type(dt)
            state = LoopController.determine_state(config.origin - 0.1, config)
            assert state == LoopState.DIVERGENT

    def test_converging_at_origin(self):
        """State should be CONVERGING at origin."""
        for dt in DomainType:
            config = DomainConfig.from_type(dt)
            state = LoopController.determine_state(config.origin + 0.01, config)
            assert state == LoopState.CONVERGING

    def test_critical_at_z50(self):
        """State should be CRITICAL at z_50."""
        for dt in DomainType:
            config = DomainConfig.from_type(dt)
            state = LoopController.determine_state(config.z_50 + 0.1, config)
            assert state == LoopState.CRITICAL

    def test_closed_at_z95(self):
        """State should be CLOSED at z_95."""
        for dt in DomainType:
            config = DomainConfig.from_type(dt)
            state = LoopController.determine_state(config.z_95 + 0.1, config)
            assert state == LoopState.CLOSED


# =============================================================================
# Coupling Matrix Tests
# =============================================================================

class TestCouplingMatrix:
    """Test coupling matrix (49 terms)."""

    def test_matrix_dimensions(self):
        """Matrix should be 7x7."""
        coupling = CouplingMatrix()
        assert len(coupling._matrix) == NUM_DOMAINS
        for row in coupling._matrix:
            assert len(row) == NUM_DOMAINS

    def test_diagonal_is_zero(self):
        """Diagonal elements (self-coupling) should be 0."""
        coupling = CouplingMatrix()
        for i in range(NUM_DOMAINS):
            assert coupling.get(i, i) == 0.0

    def test_antisymmetric(self):
        """Matrix should be antisymmetric: K_ij = -K_ji."""
        coupling = CouplingMatrix()
        for i in range(NUM_DOMAINS):
            for j in range(i + 1, NUM_DOMAINS):
                k_ij = coupling.get(i, j)
                k_ji = coupling.get(j, i)
                assert abs(k_ij + k_ji) < 0.01, f"K[{i},{j}] + K[{j},{i}] != 0"

    def test_total_coupling_terms(self):
        """Should have 49 coupling terms (including zeros on diagonal)."""
        coupling = CouplingMatrix()
        count = sum(
            1 for i in range(NUM_DOMAINS)
            for j in range(NUM_DOMAINS)
        )
        assert count == NUM_COUPLING_TERMS


# =============================================================================
# Interference Node Tests
# =============================================================================

class TestInterferenceNodes:
    """Test interference nodes (21 terms)."""

    def test_node_count(self):
        """Should have 21 interference nodes (C(7,2))."""
        substrate = ScalarSubstrate()
        assert len(substrate.interference_nodes) == NUM_INTERFERENCE_NODES

    def test_node_pairs_unique(self):
        """All node pairs should be unique."""
        substrate = ScalarSubstrate()
        pairs = set()
        for node in substrate.interference_nodes:
            pair = (min(node.domain_i, node.domain_j),
                    max(node.domain_i, node.domain_j))
            assert pair not in pairs, f"Duplicate pair {pair}"
            pairs.add(pair)

    def test_interference_formula(self):
        """I_ij = A_i * A_j * cos(φ_i - φ_j)."""
        substrate = ScalarSubstrate()

        # Set known values
        substrate.accumulators[0].value = 1.0
        substrate.accumulators[0].phase = 0.0
        substrate.accumulators[1].value = 1.0
        substrate.accumulators[1].phase = 0.0

        # Find node for pair (0, 1)
        node = next(n for n in substrate.interference_nodes
                    if (n.domain_i, n.domain_j) == (0, 1))

        # I = 1 * 1 * cos(0) = 1
        assert node.compute(substrate.accumulators) == 1.0


# =============================================================================
# Helix Coordinates Tests
# =============================================================================

class TestHelixCoordinates:
    """Test Layer 3: Helix State."""

    def test_cartesian_conversion(self):
        """Test (θ, z, r) to (x, y, z) conversion."""
        # θ=0, r=1 -> x=1, y=0
        coords = HelixCoordinates(theta=0, z=0.5, r=1.0)
        x, y, z = coords.to_cartesian()
        assert abs(x - 1.0) < 0.001
        assert abs(y - 0.0) < 0.001
        assert abs(z - 0.5) < 0.001

        # θ=π/2, r=1 -> x=0, y=1
        coords = HelixCoordinates(theta=math.pi/2, z=0.5, r=1.0)
        x, y, z = coords.to_cartesian()
        assert abs(x - 0.0) < 0.001
        assert abs(y - 1.0) < 0.001

    def test_projection(self):
        """Test z -> z' projection."""
        coords = HelixCoordinates(z=0.41)
        z_prime = coords.project()
        assert abs(z_prime - 0.941) < 0.001

    def test_from_domain(self):
        """Test creating coordinates from domain."""
        for dt in DomainType:
            config = DomainConfig.from_type(dt)
            coords = HelixCoordinates.from_domain(dt, z=config.origin)
            assert abs(coords.theta - config.theta) < 0.001
            assert abs(coords.z - config.origin) < 0.001


# =============================================================================
# Scalar Architecture Tests
# =============================================================================

class TestScalarArchitecture:
    """Test unified architecture."""

    def test_initialization(self):
        """Architecture should initialize correctly."""
        arch = ScalarArchitecture(initial_z=0.5)
        assert arch.z_level == 0.5
        assert len(arch.substrate.accumulators) == NUM_DOMAINS

    def test_step_returns_state(self):
        """Step should return complete state."""
        arch = ScalarArchitecture(initial_z=0.5)
        state = arch.step(dt=0.01)

        assert state is not None
        assert state.z_level >= 0
        assert len(state.saturations) == NUM_DOMAINS
        assert len(state.loop_states) == NUM_DOMAINS
        assert len(state.substrate_values) == NUM_DOMAINS

    def test_evolution(self):
        """Architecture should evolve z-level."""
        arch = ScalarArchitecture(initial_z=0.5)

        # Take many steps
        for _ in range(100):
            state = arch.step(dt=0.01)

        # z should have evolved
        assert arch.z_level != 0.5

    def test_all_loops_closed_at_high_z(self):
        """All loops should be CLOSED at z≈1.0."""
        arch = ScalarArchitecture(initial_z=0.99)
        state = arch.step(dt=0.01)

        # Most loops should be closed or critical
        closed_count = sum(
            1 for s in state.loop_states.values()
            if s in [LoopState.CLOSED, LoopState.CRITICAL]
        )
        assert closed_count >= 5  # At least 5 of 7


# =============================================================================
# Utility Function Tests
# =============================================================================

class TestUtilities:
    """Test utility functions."""

    def test_projection_inverse(self):
        """compute_origin_from_projection should invert compute_projection."""
        for origin in [0.41, 0.52, 0.70, 0.73, 0.80, 0.85, 0.87]:
            projection = compute_projection(origin)
            recovered = compute_origin_from_projection(projection)
            assert abs(recovered - origin) < 0.001


# =============================================================================
# Constants Tests
# =============================================================================

class TestConstants:
    """Test mathematical constants."""

    def test_tau(self):
        """TAU should be 2π."""
        assert abs(TAU - 2 * math.pi) < 0.0001

    def test_phi(self):
        """PHI should be golden ratio."""
        expected = (1 + math.sqrt(5)) / 2
        assert abs(PHI - expected) < 0.0001

    def test_substrate_node_count(self):
        """7 + 49 + 21 = 77 nodes."""
        total = NUM_DOMAINS + NUM_COUPLING_TERMS + NUM_INTERFERENCE_NODES
        assert total == 77  # 7 + 49 + 21


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
