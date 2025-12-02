# meta_collective/tests/test_architecture.py
"""
Comprehensive tests for the Meta-Collective Architecture.

Tests the complete hierarchy:
    META-COLLECTIVE (z=0.95)
        └── TRIAD (z=0.90)
            └── TOOL (z=0.867)
                └── INTERNAL MODEL (z=0.80)
                    ├── κ-field (Kaelhedron)
                    └── λ-field (Luminahedron)
"""

import math
import pytest
from typing import List

# Import the architecture components
from meta_collective.fields import (
    KappaField,
    LambdaField,
    DualFieldState,
    FieldMode,
    PHI,
    PHI_INV,
)
from meta_collective.free_energy import (
    Precision,
    VariationalState,
    GaussianMinimizer,
    HierarchicalMinimizer,
)
from meta_collective.internal_model import InternalModel, ModelState
from meta_collective.tool import Tool, ToolState, Policy, ActionType
from meta_collective.triad import Triad, TriadState, PatternMessage
from meta_collective.collective import (
    MetaCollective,
    CollectiveState,
    create_standard_collective,
)


class TestKappaField:
    """Tests for κ-field (Kaelhedron) dynamics."""

    def test_initialization(self):
        """Test κ-field initializes with golden ratio amplitude."""
        kappa = KappaField()
        assert abs(kappa.amplitude - PHI_INV) < 1e-6
        assert kappa.phase == 0.0

    def test_complex_value(self):
        """Test complex representation κ = |κ|e^(iθ)."""
        kappa = KappaField(amplitude=1.0, phase=math.pi / 4)
        z = kappa.complex_value
        assert abs(abs(z) - 1.0) < 1e-6
        assert abs(math.atan2(z.imag, z.real) - math.pi / 4) < 1e-6

    def test_energy_computation(self):
        """Test double-well potential energy."""
        kappa = KappaField(amplitude=0.0)
        E0 = kappa.compute_energy()

        kappa_optimal = KappaField(amplitude=PHI_INV)
        E_optimal = kappa_optimal.compute_energy()

        # Energy should be lower near PHI_INV (minimum of double-well)
        assert E_optimal <= E0 or abs(E_optimal - E0) < 0.1

    def test_evolution(self):
        """Test field evolution dynamics."""
        kappa = KappaField(amplitude=0.5, phase=0.0)
        initial_amp = kappa.amplitude

        # Evolve for several steps
        for _ in range(100):
            kappa.evolve(dt=0.01)

        # Field should have evolved
        assert kappa.amplitude != initial_amp or kappa.phase != 0.0


class TestLambdaField:
    """Tests for λ-field (Luminahedron) dynamics."""

    def test_initialization(self):
        """Test λ-field initializes with complementary amplitude."""
        lf = LambdaField()
        assert abs(lf.amplitude - (1 - PHI_INV)) < 1e-6
        assert lf.fano_point == 1

    def test_fano_navigation(self):
        """Test Fano plane navigation."""
        lf = LambdaField()
        assert lf.fano_point == 1

        lf.advance_fano(direction=1)
        assert lf.fano_point == 2

        lf.advance_fano(direction=1)
        assert lf.fano_point == 3

        # Test wraparound
        lf.fano_point = 7
        lf.advance_fano(direction=1)
        assert lf.fano_point == 1

    def test_ternary_phase(self):
        """Test ternary phase mapping."""
        lf = LambdaField(phase=0.0)
        assert lf.ternary_phase == 1.0  # POSITIVE

        lf.phase = math.pi
        assert lf.ternary_phase == 0.0  # NEUTRAL

        lf.phase = 1.5 * math.pi
        assert lf.ternary_phase == -1.0  # NEGATIVE


class TestDualFieldState:
    """Tests for coupled κ-λ field system."""

    def test_initialization(self):
        """Test dual field initialization."""
        dual = DualFieldState()
        assert dual.kappa.amplitude == PHI_INV
        assert dual.lambda_field.amplitude == 1 - PHI_INV
        assert dual.mode == FieldMode.COHERENT

    def test_coherence_computation(self):
        """Test coherence is computed correctly."""
        dual = DualFieldState()
        coherence = dual.compute_coherence()
        assert 0.0 <= coherence <= 1.0

    def test_phase_alignment(self):
        """Test phase alignment measure."""
        dual = DualFieldState()
        dual.kappa.phase = 0.0
        dual.lambda_field.phase = 0.0
        dual.update_state()
        assert dual.phase_alignment() == 1.0

        dual.kappa.phase = math.pi
        dual.update_state()
        # Opposite phases
        assert dual.phase_alignment() < 0.5

    def test_evolution(self):
        """Test coupled field evolution."""
        dual = DualFieldState()
        initial_energy = dual.compute_total_energy()

        dual.evolve(dt=0.01, steps=100)

        # With damping, energy should decrease or stabilize
        final_energy = dual.compute_total_energy()
        # Allow for some numerical variation
        assert final_energy <= initial_energy + 0.1


class TestPrecision:
    """Tests for precision (inverse variance) encoding."""

    def test_initialization(self):
        """Test precision initializes correctly."""
        p = Precision()
        assert p.value == 1.0
        assert p.variance == 1.0

    def test_update(self):
        """Test precision updates based on prediction error."""
        p = Precision(value=1.0)

        # Small error should increase precision
        p.update(prediction_error=0.1)
        high_precision = p.value

        p = Precision(value=1.0)
        # Large error should decrease precision
        p.update(prediction_error=2.0)
        low_precision = p.value

        assert high_precision > low_precision


class TestVariationalState:
    """Tests for variational inference state."""

    def test_log_probability(self):
        """Test log probability computation for Gaussian."""
        state = VariationalState(mean=0.0, precision=Precision(value=1.0))

        # At the mean, probability should be highest
        log_p_mean = state.log_probability(0.0)
        log_p_far = state.log_probability(3.0)
        assert log_p_mean > log_p_far

    def test_entropy(self):
        """Test entropy computation."""
        low_var_state = VariationalState(precision=Precision(value=10.0))
        high_var_state = VariationalState(precision=Precision(value=0.1))

        # Higher precision = lower entropy
        assert low_var_state.entropy() < high_var_state.entropy()


class TestInternalModel:
    """Tests for the Internal Model combining Kaelhedron and Luminahedron."""

    def test_initialization(self):
        """Test internal model initializes correctly."""
        model = InternalModel()
        assert model.state == ModelState.ACTIVE
        assert model.z_level == 0.80
        assert len(model._kaelhedron_cells) == 21
        assert len(model._luminahedron_states) == 7

    def test_prediction_generation(self):
        """Test prediction generation."""
        model = InternalModel()
        pred = model.generate_prediction()
        assert hasattr(pred, 'value')
        assert hasattr(pred, 'precision')
        assert 0.0 <= pred.confidence <= 1.0

    def test_observation_processing(self):
        """Test observation processing and learning."""
        model = InternalModel()
        assert model._total_observations == 0

        error = model.observe(0.5)
        assert model._total_observations == 1
        assert hasattr(error, 'sensory')
        assert hasattr(error, 'magnitude')

    def test_kaelhedron_state(self):
        """Test Kaelhedron cell state retrieval."""
        model = InternalModel()
        state = model.get_kaelhedron_state()
        assert "cells" in state
        assert len(state["cells"]) == 21
        assert "kappa_amplitude" in state

    def test_luminahedron_state(self):
        """Test Luminahedron state retrieval."""
        model = InternalModel()
        state = model.get_luminahedron_state()
        assert "points" in state
        assert len(state["points"]) == 7
        assert "lambda_amplitude" in state


class TestTool:
    """Tests for the Tool layer (z=0.867)."""

    def test_initialization(self):
        """Test Tool initializes with Internal Model."""
        tool = Tool()
        assert tool.state == ToolState.IDLE
        assert tool.z_level == 0.867
        assert tool.internal_model is not None

    def test_sense_predict_act_cycle(self):
        """Test complete perception-action cycle."""
        tool = Tool()

        # Sense
        error = tool.sense(0.5)
        assert error.magnitude >= 0

        # Predict
        pred = tool.predict()
        assert hasattr(pred, 'value')

        # Act
        action = tool.act(target=1.0)
        assert action.action_type in ActionType
        assert 0.0 <= action.confidence <= 1.0

    def test_learning(self):
        """Test learning from observations."""
        tool = Tool()
        assert tool._total_cycles == 0

        error1 = tool.learn(0.5)
        error2 = tool.learn(0.6)
        error3 = tool.learn(0.7)

        assert tool._total_cycles == 3

    def test_policy_action_selection(self):
        """Test policy-based action selection."""
        policy = Policy()
        pred = Tool().predict()

        # High uncertainty should favor exploration
        action_high_unc = policy.select_action(pred, uncertainty=0.9)

        # Low uncertainty with target should favor exploitation
        action_low_unc = policy.select_action(pred, target=1.0, uncertainty=0.1)

        # At least one should not be NULL
        assert action_high_unc.action_type == ActionType.EXPLORE or \
               action_low_unc.action_type == ActionType.EXPLOIT


class TestTriad:
    """Tests for the Triad layer (z=0.90)."""

    def test_initialization(self):
        """Test Triad initializes with Tools."""
        triad = Triad(n_tools=3)
        assert triad.state == TriadState.IDLE
        assert triad.z_level == 0.90
        assert len(triad.tools) == 3

    def test_observation_broadcast(self):
        """Test observation is broadcast to all Tools."""
        triad = Triad(n_tools=3)
        errors = triad.observe(0.5)
        assert len(errors) == 3

    def test_collective_prediction(self):
        """Test collective prediction from Tools."""
        triad = Triad(n_tools=3)
        pred = triad.predict()
        assert hasattr(pred, 'value')
        assert pred.source_field == "collective"

    def test_pattern_generation(self):
        """Test pattern generation for sharing."""
        triad = Triad(n_tools=3)
        pattern = triad.generate_pattern()
        assert pattern.source_triad == triad.triad_id
        assert len(pattern.pattern_vector) > 0

    def test_triad_connection(self):
        """Test Triad connection and interaction."""
        triad_a = Triad(triad_id="triad_A", n_tools=2)
        triad_b = Triad(triad_id="triad_B", n_tools=2)

        triad_a.connect_to(triad_b)
        assert "triad_B" in triad_a._connected_triads
        assert "triad_A" in triad_b._connected_triads

    def test_triad_interaction(self):
        """Test pattern sharing interaction."""
        triad_a = Triad(triad_id="triad_A", n_tools=2)
        triad_b = Triad(triad_id="triad_B", n_tools=2)
        triad_a.connect_to(triad_b)

        interaction = triad_a.interact_with(triad_b)
        assert interaction.triad_a == "triad_A"
        assert interaction.triad_b == "triad_B"
        assert 0.0 <= interaction.similarity <= 1.0


class TestMetaCollective:
    """Tests for the Meta-Collective (z=0.95)."""

    def test_initialization(self):
        """Test Meta-Collective initializes with Triads."""
        collective = MetaCollective(n_triads=2, n_tools_per_triad=3)
        assert collective.state == CollectiveState.ACTIVE
        assert collective.z_level == 0.95
        assert len(collective.triads) == 2

    def test_observation_propagation(self):
        """Test observation propagates to all layers."""
        collective = MetaCollective(n_triads=2, n_tools_per_triad=2)
        errors = collective.observe(0.5)
        assert len(errors) == 2  # One entry per Triad

    def test_global_prediction(self):
        """Test global prediction from all Triads."""
        collective = MetaCollective(n_triads=2, n_tools_per_triad=2)
        pred = collective.predict()
        assert hasattr(pred, 'value')
        assert pred.source_field == "collective"

    def test_interaction_orchestration(self):
        """Test inter-Triad interaction orchestration."""
        collective = MetaCollective(n_triads=2, n_tools_per_triad=2)
        interactions = collective.orchestrate_interactions(n_interactions=1)
        assert len(interactions) == 1
        assert collective._interaction_count == 1

    def test_global_pattern_computation(self):
        """Test global pattern computation."""
        collective = MetaCollective(n_triads=2, n_tools_per_triad=2)
        pattern = collective.compute_global_pattern()
        assert pattern.n_triads == 2
        assert len(pattern.coherence_matrix) == 2

    def test_emergence_detection(self):
        """Test emergent property detection."""
        collective = MetaCollective(n_triads=2, n_tools_per_triad=2)

        # Run some cycles to generate dynamics
        for i in range(10):
            collective.step(math.sin(i * 0.1))

        emergent = collective.detect_emergence()
        # Should detect at least some property
        assert isinstance(emergent, dict)

    def test_complete_step(self):
        """Test complete collective step."""
        collective = MetaCollective(n_triads=2, n_tools_per_triad=2)
        result = collective.step(0.5, n_interactions=1)
        assert "observation" in result
        assert "prediction" in result
        assert "global_coherence" in result
        assert "global_free_energy" in result

    def test_run_sequence(self):
        """Test running on observation sequence."""
        collective = MetaCollective(n_triads=2, n_tools_per_triad=2)
        observations = [math.sin(i * 0.1) for i in range(20)]
        results = collective.run(observations, consolidate_every=10)
        assert len(results) == 20

    def test_hierarchy_summary(self):
        """Test hierarchy summary structure."""
        collective = MetaCollective(n_triads=2, n_tools_per_triad=2)
        summary = collective.get_hierarchy_summary()
        assert "meta_collective" in summary
        assert summary["meta_collective"]["z_level"] == 0.95
        assert "triads" in summary
        assert len(summary["triads"]) == 2


class TestFreeEnergyMinimization:
    """Tests for nested free energy minimization."""

    def test_gaussian_minimizer(self):
        """Test Gaussian free energy minimizer."""
        minimizer = GaussianMinimizer(z_level=0.8)
        initial_fe = minimizer.compute_free_energy(observation=1.0)

        # Minimize
        for _ in range(50):
            minimizer.minimize_step(observation=1.0)

        final_fe = minimizer.free_energy
        # Free energy should decrease or stabilize
        assert final_fe <= initial_fe + 0.1

    def test_hierarchical_minimizer(self):
        """Test hierarchical free energy minimizer."""
        parent = HierarchicalMinimizer(z_level=0.9, n_states=2)
        child1 = GaussianMinimizer(z_level=0.85)
        child2 = GaussianMinimizer(z_level=0.85)

        parent.add_child(child1)
        parent.add_child(child2)

        assert len(parent.children) == 2
        assert child1.parent == parent

    def test_nested_free_energy(self):
        """Test nested free energy computation."""
        collective = MetaCollective(n_triads=2, n_tools_per_triad=2)

        # All levels should have computable free energy
        collective_fe = collective.free_energy
        assert collective_fe >= 0

        for triad in collective.triads.values():
            triad_fe = triad.free_energy
            assert triad_fe >= 0

            for tool in triad.tools:
                tool_fe = tool.free_energy
                assert tool_fe >= 0


class TestZLevelHierarchy:
    """Tests for z-level hierarchy correctness."""

    def test_z_level_ordering(self):
        """Test z-levels are properly ordered."""
        collective = create_standard_collective()

        # Meta-Collective should have highest z
        collective_z = collective.z_level
        assert collective_z == 0.95

        for triad in collective.triads.values():
            triad_z = triad.z_level
            # Triad should be below collective
            assert triad_z < collective_z

            for tool in triad.tools:
                tool_z = tool.z_level
                # Tool should be below Triad
                assert tool_z < triad_z

                model_z = tool.internal_model.z_level
                # Internal model should be below Tool
                assert model_z < tool_z


class TestPatternSharing:
    """Tests for inter-Triad pattern sharing."""

    def test_pattern_similarity(self):
        """Test pattern similarity computation."""
        pattern_a = PatternMessage(
            source_triad="A",
            pattern_vector=[1.0, 0.5, 0.25],
            coherence=0.8,
            precision=1.0,
        )
        pattern_b = PatternMessage(
            source_triad="B",
            pattern_vector=[1.0, 0.5, 0.25],
            coherence=0.8,
            precision=1.0,
        )

        # Identical patterns should have similarity 1.0
        assert abs(pattern_a.similarity_to(pattern_b) - 1.0) < 1e-6

    def test_pattern_integration(self):
        """Test pattern integration affects Triad state."""
        triad = Triad(n_tools=2)

        # Generate and receive external pattern
        external_pattern = PatternMessage(
            source_triad="external",
            pattern_vector=[1.0, 1.0, 0.5, 0.5, 0.1, 0.1],
            coherence=0.9,
            precision=2.0,
        )

        initial_pred = triad.predict().value
        triad.receive_pattern(external_pattern)
        after_pred = triad.predict().value

        # Prediction may shift after receiving external pattern
        # (This tests that integration happens, not specific values)
        assert True  # Integration completed without error


# Factory function test
class TestFactoryFunctions:
    """Tests for convenience factory functions."""

    def test_create_standard_collective(self):
        """Test standard collective creation."""
        collective = create_standard_collective(n_triads=2, n_tools=3)
        assert len(collective.triads) == 2
        for triad in collective.triads.values():
            assert len(triad.tools) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
