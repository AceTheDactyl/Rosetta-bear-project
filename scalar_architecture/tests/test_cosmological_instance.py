"""
Comprehensive Test Suite for Cosmological Instance
Validates the unified synthesis from all observation points

Signature: Δ|loop-closed|z0.99|rhythm-native|Ω

Test Categories:
1. Unit Tests - Individual component validation
2. Integration Tests - Cross-layer interactions
3. Convergence Tests - Stability and fixed points
4. Memory Tests - Holographic encoding/retrieval
5. Self-Reference Tests - Recursive observation
6. Cosmological Tests - Vortex stage progression

Run with: pytest test_cosmological_instance.py -v
"""

import math
import time
import numpy as np
import pytest
from typing import List, Tuple

# Import the cosmological instance module
import sys
sys.path.insert(0, '..')

from scalar_architecture.cosmological_instance import (
    # Core classes
    CosmologicalInstance,
    InstanceState,
    Observation,
    ObservationPoint,
    VortexStage,
    VortexTracker,

    # Observers
    SubstrateObserver,
    ConvergenceObserver,
    LoopStateObserver,
    HelixObserver,
    MemoryObserver,
    MetaObserver,

    # Factory functions
    create_instance,
    create_evolved_instance,
    create_fixed_point_instance,

    # Validation
    validate_instance,
    validate_all,

    # Constants
    VORTEX_STAGES,
    BETA_CRITICAL,
    FIXED_POINT_EPSILON,
    SIGNATURE_DELTA,
    SIGNATURE_OMEGA,
)

from scalar_architecture.core import (
    DomainType,
    LoopState,
    Pattern,
    TAU,
    NUM_DOMAINS,
)

from scalar_architecture.holographic_memory import (
    K_CRITICAL,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def fresh_instance():
    """Create a fresh instance at origin."""
    return create_instance(birth_z=0.41)


@pytest.fixture
def evolved_instance():
    """Create an instance evolved to z=0.90."""
    instance = create_instance(birth_z=0.41)
    instance.evolve_to_z(0.90, max_steps=500)
    return instance


@pytest.fixture
def high_z_instance():
    """Create an instance at near-transcendence."""
    instance = create_instance(birth_z=0.41)
    instance.z_level = 0.99
    return instance


# =============================================================================
# Unit Tests: Instance Creation
# =============================================================================

class TestInstanceCreation:
    """Test instance creation and initialization."""

    def test_create_instance_default(self):
        """Instance creates with default parameters."""
        instance = create_instance()
        assert instance is not None
        assert instance.instance_id is not None
        assert len(instance.instance_id) == 16

    def test_create_instance_custom_z(self):
        """Instance creates at specified z-level."""
        instance = create_instance(birth_z=0.50)
        assert instance.z_level == 0.50

    def test_create_instance_custom_id(self):
        """Instance creates with custom ID."""
        instance = create_instance(instance_id="test_instance")
        assert instance.instance_id == "test_instance"

    def test_instance_has_architecture(self, fresh_instance):
        """Instance has scalar architecture."""
        assert fresh_instance.architecture is not None
        assert fresh_instance.architecture.substrate is not None

    def test_instance_has_memory(self, fresh_instance):
        """Instance has holographic memory."""
        assert fresh_instance.memory is not None
        assert fresh_instance.memory.n > 0

    def test_instance_has_all_observers(self, fresh_instance):
        """Instance has all 6 observers."""
        assert len(fresh_instance.observers) == 6
        for point in ObservationPoint:
            assert point.value in fresh_instance.observers

    def test_instance_has_vortex_tracker(self, fresh_instance):
        """Instance has vortex tracker with 7 stages."""
        assert fresh_instance.vortex_tracker is not None
        assert len(fresh_instance.vortex_tracker.stages) == 7

    def test_instance_birth_time(self, fresh_instance):
        """Instance records birth time."""
        assert fresh_instance.birth_time > 0
        assert fresh_instance.birth_time <= time.time()


# =============================================================================
# Unit Tests: Observation Points
# =============================================================================

class TestObservationPoints:
    """Test individual observation perspectives."""

    def test_substrate_observer(self, fresh_instance):
        """Substrate observer returns valid observation."""
        observer = SubstrateObserver()
        obs = observer.observe(fresh_instance)

        assert obs.point == ObservationPoint.SUBSTRATE
        assert 'accumulator_values' in obs.data
        assert 'interference' in obs.data
        assert 0 <= obs.coherence <= 1

    def test_convergence_observer(self, fresh_instance):
        """Convergence observer returns valid observation."""
        observer = ConvergenceObserver()
        obs = observer.observe(fresh_instance)

        assert obs.point == ObservationPoint.CONVERGENCE
        assert 'z_level' in obs.data
        assert 'composite_saturation' in obs.data
        assert obs.data['z_level'] == fresh_instance.z_level

    def test_loop_state_observer(self, fresh_instance):
        """Loop state observer returns valid observation."""
        observer = LoopStateObserver()
        obs = observer.observe(fresh_instance)

        assert obs.point == ObservationPoint.LOOP_STATE
        assert 'domain_states' in obs.data
        assert len(obs.data['domain_states']) == NUM_DOMAINS

    def test_helix_observer(self, fresh_instance):
        """Helix observer returns valid observation."""
        observer = HelixObserver()
        obs = observer.observe(fresh_instance)

        assert obs.point == ObservationPoint.HELIX
        assert 'theta' in obs.data
        assert 'z' in obs.data
        assert 'r' in obs.data
        assert 0 <= obs.data['theta'] <= TAU

    def test_memory_observer(self, fresh_instance):
        """Memory observer returns valid observation."""
        observer = MemoryObserver()
        obs = observer.observe(fresh_instance)

        assert obs.point == ObservationPoint.MEMORY
        assert 'order_parameter_r' in obs.data
        assert 'capacity' in obs.data

    def test_meta_observer(self, fresh_instance):
        """Meta observer returns valid observation."""
        observer = MetaObserver()
        obs = observer.observe(fresh_instance)

        assert obs.point == ObservationPoint.META
        assert 'observation_signatures' in obs.data
        assert 'meta_coherence' in obs.data

    def test_observe_all(self, fresh_instance):
        """observe_all returns observations from all points."""
        observations = fresh_instance.observe_all()

        assert len(observations) == 6
        for point in ObservationPoint:
            assert point.value in observations

    def test_observation_signature_format(self, fresh_instance):
        """Observation signatures have correct format."""
        obs = fresh_instance.observe(ObservationPoint.SUBSTRATE)
        sig = obs.signature

        parts = sig.split('|')
        assert len(parts) == 3
        assert parts[0] == 'substrate'


# =============================================================================
# Unit Tests: Vortex Tracker
# =============================================================================

class TestVortexTracker:
    """Test vortex stage progression."""

    def test_tracker_initialization(self):
        """Tracker initializes with 7 stages."""
        tracker = VortexTracker()
        assert len(tracker.stages) == 7

    def test_stage_order(self):
        """Stages are in correct z-level order."""
        tracker = VortexTracker()
        z_levels = [s.z_threshold for s in tracker.stages]
        assert z_levels == sorted(z_levels)

    def test_stage_activation(self):
        """Stages activate at correct z-levels."""
        tracker = VortexTracker()

        # Initially no stages active
        assert tracker.completion_fraction() == 0.0

        # Activate first stage
        tracker.update(0.42)
        assert tracker.stages[0].activated
        assert not tracker.stages[1].activated

    def test_all_stages_activate(self):
        """All stages activate at z=0.99."""
        tracker = VortexTracker()
        tracker.update(0.99)

        for stage in tracker.stages:
            assert stage.activated
        assert tracker.completion_fraction() == 1.0

    def test_current_stage(self):
        """current_stage returns highest activated."""
        tracker = VortexTracker()
        tracker.update(0.75)

        current = tracker.current_stage(0.75)
        assert current is not None
        assert current.name == "AUTOCATALYSIS"  # z=0.73

    def test_stage_domain_mapping(self):
        """Each stage maps to correct domain."""
        tracker = VortexTracker()

        for i, stage in enumerate(tracker.stages):
            assert stage.domain == DomainType(i)


# =============================================================================
# Integration Tests: Evolution
# =============================================================================

class TestInstanceEvolution:
    """Test instance evolution dynamics."""

    def test_step_advances_recursion(self, fresh_instance):
        """Each step increments recursion depth."""
        initial_depth = fresh_instance.recursion_depth
        fresh_instance.step()
        assert fresh_instance.recursion_depth == initial_depth + 1

    def test_step_records_trajectory(self, fresh_instance):
        """Steps record trajectory history."""
        fresh_instance.step()
        assert len(fresh_instance.trajectory_history) >= 1

    def test_step_records_meta_history(self, fresh_instance):
        """Steps record meta-coherence history."""
        fresh_instance.step()
        assert len(fresh_instance.meta_history) >= 1

    def test_step_returns_state(self, fresh_instance):
        """Step returns valid InstanceState."""
        state = fresh_instance.step()

        assert isinstance(state, InstanceState)
        assert state.z_level == fresh_instance.z_level
        assert state.signature == fresh_instance.signature

    def test_evolve_to_z(self, fresh_instance):
        """evolve_to_z reaches target z-level."""
        target = 0.70
        states = fresh_instance.evolve_to_z(target, max_steps=500)

        assert fresh_instance.z_level >= target - 0.01
        assert len(states) > 0

    def test_evolve_activates_vortex_stages(self, fresh_instance):
        """Evolution activates vortex stages."""
        initial_completion = fresh_instance.vortex_tracker.completion_fraction()
        fresh_instance.evolve_to_z(0.99, max_steps=1000)
        final_completion = fresh_instance.vortex_tracker.completion_fraction()

        assert final_completion > initial_completion

    def test_state_history_grows(self, fresh_instance):
        """State history grows with evolution."""
        for _ in range(10):
            fresh_instance.step()

        assert len(fresh_instance.state_history) >= 10


# =============================================================================
# Integration Tests: Cross-Layer Dynamics
# =============================================================================

class TestCrossLayerDynamics:
    """Test interactions between architecture layers."""

    def test_z_affects_saturation(self, fresh_instance):
        """Z-level affects convergence saturation."""
        obs_low = fresh_instance.observe(ObservationPoint.CONVERGENCE)
        low_saturation = obs_low.data['composite_saturation']

        fresh_instance.z_level = 0.99
        obs_high = fresh_instance.observe(ObservationPoint.CONVERGENCE)
        high_saturation = obs_high.data['composite_saturation']

        assert high_saturation > low_saturation

    def test_z_affects_loop_states(self, fresh_instance):
        """Z-level affects loop state distribution."""
        fresh_instance.z_level = 0.41
        obs_low = fresh_instance.observe(ObservationPoint.LOOP_STATE)

        fresh_instance.z_level = 0.99
        obs_high = fresh_instance.observe(ObservationPoint.LOOP_STATE)

        # More advanced states at higher z
        low_advanced = obs_low.data['closed_count'] + obs_low.data['critical_count']
        high_advanced = obs_high.data['closed_count'] + obs_high.data['critical_count']

        assert high_advanced >= low_advanced

    def test_helix_r_tracks_coherence(self, evolved_instance):
        """Helix r parameter tracks overall coherence."""
        obs = evolved_instance.observe(ObservationPoint.HELIX)
        assert 0 <= obs.data['r'] <= 1

    def test_memory_order_parameter_valid(self, evolved_instance):
        """Memory order parameter is valid."""
        obs = evolved_instance.observe(ObservationPoint.MEMORY)
        r = obs.data['order_parameter_r']
        assert 0 <= r <= 1


# =============================================================================
# Convergence Tests: Stability and Fixed Points
# =============================================================================

class TestConvergence:
    """Test stability and fixed point behavior."""

    def test_meta_coherence_bounded(self, evolved_instance):
        """Meta-coherence stays in [0, 1]."""
        for _ in range(100):
            evolved_instance.step()
            obs = evolved_instance.observe(ObservationPoint.META)
            assert 0 <= obs.coherence <= 1

    def test_helix_r_stable(self, evolved_instance):
        """Helix r parameter remains stable."""
        r_values = []
        for _ in range(50):
            evolved_instance.step()
            r_values.append(evolved_instance.architecture.helix.r)

        # Check variance is bounded
        variance = np.var(r_values)
        assert variance < 0.5  # Not diverging

    def test_z_level_bounded(self, fresh_instance):
        """Z-level stays in [0, 1] during evolution."""
        for _ in range(100):
            fresh_instance.step()
            if fresh_instance.z_level < 0.99:
                fresh_instance.z_level += 0.001

            assert 0 <= fresh_instance.z_level <= 1.0

    def test_fixed_point_detection(self):
        """Fixed point is detected when coherence stabilizes."""
        instance = create_instance(birth_z=0.95)
        instance.z_level = 0.99

        # Run until potential fixed point
        for _ in range(500):
            instance.step()

        # Should have meta history
        assert len(instance.meta_history) >= 10

    def test_high_coherence_at_high_z(self, high_z_instance):
        """High z-level produces high coherence."""
        for _ in range(50):
            high_z_instance.step()

        obs = high_z_instance.observe(ObservationPoint.META)
        # At z=0.99, expect reasonable coherence
        assert obs.coherence > 0.3


# =============================================================================
# Memory Tests: Holographic Encoding/Retrieval
# =============================================================================

class TestHolographicMemory:
    """Test holographic memory operations."""

    def test_encode_experience(self, fresh_instance):
        """Experience encoding works."""
        content = np.random.randn(64)
        pattern = fresh_instance.encode_experience(
            "test_experience",
            content,
            valence=0.5,
            arousal=0.3
        )

        assert pattern is not None
        assert pattern.id == "test_experience"
        assert "test_experience" in fresh_instance.memory.memories

    def test_recall_with_noise(self, fresh_instance):
        """Recall works with noisy query."""
        # Encode
        original = np.random.randn(64)
        fresh_instance.encode_experience("recall_test", original)

        # Query with noise
        query = original + np.random.randn(64) * 0.1
        memory_id, retrieved = fresh_instance.recall(query)

        assert retrieved is not None
        assert len(retrieved) == 64

    def test_vortex_stage_encoding(self, fresh_instance):
        """Vortex stage transitions are encoded."""
        # Evolve through stages
        fresh_instance.evolve_to_z(0.90, max_steps=500)

        # Check vortex memories exist
        vortex_memories = [
            mid for mid in fresh_instance.memory.memories
            if mid.startswith("vortex_")
        ]
        assert len(vortex_memories) > 0

    def test_memory_capacity(self, fresh_instance):
        """Memory reports valid capacity."""
        capacity = fresh_instance.memory.capacity
        assert capacity > 0
        assert capacity > fresh_instance.memory.n  # Higher-order should exceed linear


# =============================================================================
# Self-Reference Tests: Recursive Observation
# =============================================================================

class TestSelfReference:
    """Test recursive self-observation."""

    def test_meta_observes_all_others(self, fresh_instance):
        """Meta observer includes all other observations."""
        obs = fresh_instance.observe(ObservationPoint.META)
        signatures = obs.data['observation_signatures']

        # Should have 5 other observers (not itself)
        assert len(signatures) == 5

    def test_meta_coherence_is_average(self, evolved_instance):
        """Meta-coherence is average of component coherences."""
        obs = evolved_instance.observe(ObservationPoint.META)

        coherences = obs.data['coherence_vector']
        expected_meta = np.mean(coherences)

        assert abs(obs.data['meta_coherence'] - expected_meta) < 0.01

    def test_recursion_depth_tracks(self, fresh_instance):
        """Recursion depth increments with steps."""
        initial = fresh_instance.recursion_depth

        for i in range(10):
            fresh_instance.step()
            assert fresh_instance.recursion_depth == initial + i + 1

    def test_instance_can_observe_itself(self, fresh_instance):
        """Instance observing itself doesn't crash."""
        for _ in range(10):
            observations = fresh_instance.observe_all()
            meta = observations['meta']
            assert meta.data['observation_count'] == 5


# =============================================================================
# Cosmological Tests: Vortex Progression
# =============================================================================

class TestCosmologicalProgression:
    """Test cosmological vortex stage progression."""

    def test_seven_stages_exist(self):
        """There are exactly 7 vortex stages."""
        assert len(VORTEX_STAGES) == 7

    def test_stages_map_to_domains(self, fresh_instance):
        """Each stage maps to a domain."""
        for i, stage in enumerate(fresh_instance.vortex_tracker.stages):
            assert stage.domain == DomainType(i)

    def test_stage_names_match(self):
        """Stage names match expected sequence."""
        expected_names = [
            "QUANTUM_FOAM",
            "NUCLEOSYNTHESIS",
            "CARBON_RESONANCE",
            "AUTOCATALYSIS",
            "PHASE_LOCK",
            "NEURAL_EMERGENCE",
            "RECURSIVE_WITNESS"
        ]

        for i, (name, z, desc) in enumerate(VORTEX_STAGES):
            assert name == expected_names[i]

    def test_evolution_through_all_stages(self, fresh_instance):
        """Instance can evolve through all 7 stages."""
        fresh_instance.evolve_to_z(0.99, max_steps=1000)

        completion = fresh_instance.vortex_tracker.completion_fraction()
        assert completion == 1.0

    def test_current_vortex_stage_updates(self, fresh_instance):
        """current_vortex_stage reflects evolution."""
        stage1 = fresh_instance.current_vortex_stage()

        fresh_instance.evolve_to_z(0.85, max_steps=500)
        stage2 = fresh_instance.current_vortex_stage()

        assert stage2 != stage1


# =============================================================================
# Validation Tests
# =============================================================================

class TestValidation:
    """Test instance validation functions."""

    def test_validate_fresh_instance(self, fresh_instance):
        """Fresh instance passes all validations."""
        passed, details = validate_all(fresh_instance)
        assert passed, f"Validation failed: {details}"

    def test_validate_evolved_instance(self, evolved_instance):
        """Evolved instance passes all validations."""
        passed, details = validate_all(evolved_instance)
        assert passed, f"Validation failed: {details}"

    def test_validate_architecture_valid(self, fresh_instance):
        """Architecture validation passes."""
        validations = validate_instance(fresh_instance)
        assert validations['architecture_valid']

    def test_validate_memory_valid(self, fresh_instance):
        """Memory validation passes."""
        validations = validate_instance(fresh_instance)
        assert validations['memory_valid']

    def test_validate_observers_complete(self, fresh_instance):
        """Observer validation passes."""
        validations = validate_instance(fresh_instance)
        assert validations['observers_complete']

    def test_validate_z_bounds(self, fresh_instance):
        """Z-level bounds validation passes."""
        validations = validate_instance(fresh_instance)
        assert validations['z_in_bounds']

    def test_validate_signature_format(self, fresh_instance):
        """Signature format validation passes."""
        validations = validate_instance(fresh_instance)
        assert validations['signature_valid']


# =============================================================================
# Signature Tests
# =============================================================================

class TestSignature:
    """Test instance signature generation."""

    def test_signature_starts_with_delta(self, fresh_instance):
        """Signature starts with Δ."""
        assert fresh_instance.signature.startswith(SIGNATURE_DELTA)

    def test_signature_ends_with_omega(self, fresh_instance):
        """Signature ends with Ω."""
        assert fresh_instance.signature.endswith(SIGNATURE_OMEGA)

    def test_signature_contains_z_level(self, fresh_instance):
        """Signature contains z-level."""
        sig = fresh_instance.signature
        assert f"z{fresh_instance.z_level:.2f}" in sig

    def test_signature_contains_closed_count(self, evolved_instance):
        """Signature contains closed loop count."""
        sig = evolved_instance.signature
        assert "-closed" in sig

    def test_signature_format(self, fresh_instance):
        """Signature has correct format with pipe separators."""
        sig = fresh_instance.signature
        parts = sig.split('|')
        assert len(parts) == 5


# =============================================================================
# Factory Function Tests
# =============================================================================

class TestFactoryFunctions:
    """Test factory function behavior."""

    def test_create_instance_defaults(self):
        """create_instance works with defaults."""
        instance = create_instance()
        assert instance is not None
        assert instance.z_level == 0.41

    def test_create_evolved_instance(self):
        """create_evolved_instance reaches target z."""
        instance = create_evolved_instance(target_z=0.80)
        assert instance.z_level >= 0.79

    def test_create_fixed_point_instance(self):
        """create_fixed_point_instance runs to high z."""
        instance = create_fixed_point_instance()
        assert instance.z_level >= 0.90


# =============================================================================
# Performance Tests
# =============================================================================

class TestPerformance:
    """Test performance characteristics."""

    def test_step_time_bounded(self, fresh_instance):
        """Single step completes in reasonable time."""
        start = time.time()
        fresh_instance.step()
        elapsed = time.time() - start

        assert elapsed < 1.0  # Less than 1 second

    def test_observe_all_time_bounded(self, fresh_instance):
        """observe_all completes in reasonable time."""
        start = time.time()
        fresh_instance.observe_all()
        elapsed = time.time() - start

        assert elapsed < 1.0

    def test_evolution_time_linear(self, fresh_instance):
        """Evolution time scales reasonably."""
        start = time.time()
        fresh_instance.evolve_to_z(0.70, max_steps=100)
        elapsed = time.time() - start

        # Should complete 100 steps in reasonable time
        assert elapsed < 10.0


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_z_at_zero(self):
        """Instance works at z=0."""
        instance = create_instance(birth_z=0.0)
        instance.step()
        assert instance.z_level >= 0

    def test_z_at_one(self):
        """Instance works at z=1.0."""
        instance = create_instance(birth_z=0.99)
        instance.z_level = 1.0
        instance.step()
        assert instance.z_level <= 1.0

    def test_many_steps(self, fresh_instance):
        """Instance handles many steps."""
        for _ in range(1000):
            fresh_instance.step()

        assert fresh_instance.recursion_depth >= 1000

    def test_empty_trajectory_history(self):
        """Instance handles empty trajectory history."""
        instance = create_instance()
        obs = instance.observe(ObservationPoint.HELIX)
        # Should not crash
        assert 'angular_velocity' in obs.data


# =============================================================================
# Run All Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
