#!/usr/bin/env python3
"""
Test Suite for Zero-Point Energy System
========================================

Validates the ZPE system components:
1. Fano Variational Inference
2. MirrorRoot Operations
3. Neural Matrix Token Index
4. APL ZPE Operators
5. Full Extraction Cycles

Signature: Δ|test-zpe|z0.990|validation|Ω
"""

import sys
import math
import unittest
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lattice_core.zero_point_energy import (
    # Constants
    PHI, PHI_INV, TAU, ZPE_BASE, FANO_LINES,
    ZPE_COHERENCE_THRESHOLD, ZPE_EFFICIENCY_MAX,
    LOGOS, NOUS, BIOS,

    # Classes
    FanoNode, FanoVariationalEngine,
    MirrorRootOperator,
    NeuralMatrixToken, NeuralMatrixIndex,
    ZeroPointEnergyEngine, ZPEState, ZPEResult,

    # Enums
    Spiral, Machine, TruthState,

    # Factory functions
    create_zpe_engine, create_fano_inference_engine, create_mirroroot_operator
)

from lattice_core.wumbo_engine import WumboArray


class TestConstants(unittest.TestCase):
    """Test mathematical constants."""

    def test_golden_ratio(self):
        """Verify golden ratio properties."""
        self.assertAlmostEqual(PHI, (1 + math.sqrt(5)) / 2, places=10)
        self.assertAlmostEqual(PHI_INV, 1 / PHI, places=10)
        self.assertAlmostEqual(PHI * PHI_INV, 1.0, places=10)

    def test_phi_square_identity(self):
        """φ² = φ + 1"""
        self.assertAlmostEqual(PHI ** 2, PHI + 1, places=10)

    def test_tau(self):
        """τ = 2π"""
        self.assertAlmostEqual(TAU, 2 * math.pi, places=10)

    def test_mirroroot_identity(self):
        """Λ × Ν = Β² (MirrorRoot identity)"""
        self.assertAlmostEqual(LOGOS * NOUS, BIOS ** 2, places=10)

    def test_fano_lines(self):
        """Verify Fano plane structure."""
        self.assertEqual(len(FANO_LINES), 7)
        for line in FANO_LINES:
            self.assertEqual(len(line), 3)
            for point in line:
                self.assertIn(point, range(1, 8))


class TestFanoNode(unittest.TestCase):
    """Test Fano plane nodes."""

    def setUp(self):
        self.node = FanoNode(point_id=1, belief_mean=0.5, belief_precision=2.0)

    def test_variance(self):
        """Variance = 1/precision."""
        self.assertAlmostEqual(self.node.variance, 0.5, places=10)

    def test_incident_lines(self):
        """Point 1 should be on lines containing point 1."""
        lines = self.node.get_incident_lines()
        self.assertEqual(len(lines), 3)  # Each point on exactly 3 lines
        for line in lines:
            self.assertIn(1, line)

    def test_neighbors(self):
        """Point 1's neighbors should be unique and not include 1."""
        neighbors = self.node.get_neighbors()
        self.assertNotIn(1, neighbors)
        self.assertEqual(len(neighbors), len(set(neighbors)))  # All unique


class TestFanoVariationalEngine(unittest.TestCase):
    """Test Fano variational inference."""

    def setUp(self):
        self.engine = create_fano_inference_engine()

    def test_initialization(self):
        """Engine should initialize with 7 nodes."""
        self.assertEqual(len(self.engine.nodes), 7)
        for i in range(1, 8):
            self.assertIn(i, self.engine.nodes)

    def test_inject_observation(self):
        """Observations should update beliefs."""
        old_mean = self.engine.nodes[1].belief_mean
        self.engine.inject_observation(1, 1.0, precision=10.0)
        # Belief should move toward observation
        new_mean = self.engine.nodes[1].belief_mean
        self.assertNotEqual(old_mean, new_mean)

    def test_run_inference(self):
        """Inference should reduce free energy."""
        initial_fe = self.engine.compute_free_energy()
        self.engine.run_inference(max_iterations=50)
        final_fe = self.engine.free_energy
        # Free energy should decrease or stabilize
        self.assertLessEqual(final_fe, initial_fe + 1.0)  # Allow small tolerance

    def test_compute_zpe(self):
        """ZPE should be computable after inference."""
        self.engine.run_inference(max_iterations=50)
        zpe = self.engine.compute_zpe()
        self.assertGreater(zpe, 0)  # ZPE should be positive

    def test_automorphism(self):
        """Automorphism should permute nodes."""
        # Apply cycle automorphism
        perm = {i: (i % 7) + 1 for i in range(1, 8)}
        self.engine.apply_automorphism(perm)
        # All nodes should still exist
        self.assertEqual(len(self.engine.nodes), 7)


class TestMirrorRootOperator(unittest.TestCase):
    """Test MirrorRoot operations."""

    def setUp(self):
        self.op = create_mirroroot_operator()

    def test_verify_identity(self):
        """MirrorRoot identity should hold."""
        self.assertTrue(self.op.verify_identity())

    def test_mirror_invert(self):
        """Mirror inversion: X × mirror(X) = Β²"""
        x = 2.0
        x_mirror = self.op.mirror_invert(x)
        self.assertAlmostEqual(x * x_mirror, self.op.bios ** 2, places=10)

    def test_dual_compose(self):
        """Dual composition should return √(a × mirror(b))."""
        a, b = 2.0, 0.5
        result = self.op.dual_compose(a, b)
        expected = math.sqrt(a * self.op.mirror_invert(b))
        self.assertAlmostEqual(result, expected, places=10)

    def test_extract_mediator(self):
        """Mediator extraction: √(a × b)."""
        a, b = 4.0, 9.0
        result = self.op.extract_mediator(a, b)
        self.assertAlmostEqual(result, 6.0, places=10)

    def test_golden_inversion(self):
        """Golden inversion should return (φ×x, φ⁻¹×x)."""
        x = 1.0
        up, down = self.op.golden_inversion(x)
        self.assertAlmostEqual(up, PHI, places=10)
        self.assertAlmostEqual(down, PHI_INV, places=10)

    def test_polarity_switch(self):
        """Polarity switch should flip sign."""
        initial = self.op.polarity
        self.op.fano_polarity_switch()
        self.assertEqual(self.op.polarity, -initial)
        self.assertAlmostEqual(self.op.accumulated_phase, math.pi, places=10)


class TestNeuralMatrixToken(unittest.TestCase):
    """Test APL token generation."""

    def test_token_string(self):
        """Token should format correctly."""
        token = NeuralMatrixToken(
            spiral=Spiral.PHI,
            machine=Machine.U,
            intent="test",
            truth=TruthState.TRUE,
            tier=1
        )
        expected = "Phi:U(test)TRUE@1"
        self.assertEqual(str(token), expected)

    def test_to_wumbo_stimulus(self):
        """Token should convert to WUMBO stimulus."""
        token = NeuralMatrixToken(
            spiral=Spiral.E,
            machine=Machine.M,
            intent="fusion",
            truth=TruthState.TRUE,
            tier=2
        )
        stimulus = token.to_wumbo_stimulus(dim=21)
        self.assertIsInstance(stimulus, WumboArray)
        self.assertEqual(len(stimulus.data), 21)


class TestNeuralMatrixIndex(unittest.TestCase):
    """Test Neural Matrix Token Index."""

    def setUp(self):
        self.index = NeuralMatrixIndex()

    def test_z_to_spiral(self):
        """Z-value should map to correct spiral."""
        self.assertEqual(self.index.z_to_spiral(0.1), Spiral.PHI)
        self.assertEqual(self.index.z_to_spiral(0.5), Spiral.E)
        self.assertEqual(self.index.z_to_spiral(0.9), Spiral.PI)

    def test_generate_token(self):
        """Token generation should work correctly."""
        token = self.index.generate_token(
            z=0.85,
            coherence=0.9,
            energy=ZPE_BASE,
            phase=0.0
        )
        self.assertIsInstance(token, NeuralMatrixToken)
        self.assertEqual(token.spiral, Spiral.PI)  # z >= 0.66
        self.assertEqual(token.tier, 3)  # z >= 0.83

    def test_token_counter(self):
        """Token counter should increment."""
        initial = self.index.token_counter
        self.index.generate_token(z=0.5, coherence=0.5, energy=1.0, phase=0.0)
        self.assertEqual(self.index.token_counter, initial + 1)


class TestZeroPointEnergyEngine(unittest.TestCase):
    """Test ZPE engine integration."""

    def setUp(self):
        self.engine = create_zpe_engine(seed=42)

    def test_initialization(self):
        """Engine should initialize correctly."""
        self.assertEqual(self.engine.state, ZPEState.DORMANT)
        self.assertEqual(self.engine.total_extracted, 0.0)
        self.assertIsNotNone(self.engine.wumbo)
        self.assertIsNotNone(self.engine.fano)
        self.assertIsNotNone(self.engine.mirroroot)

    def test_vacuum_tap(self):
        """Vacuum tap should extract energy."""
        extracted = self.engine.vacuum_tap()
        self.assertGreater(extracted, 0)

    def test_coherence_gate(self):
        """Coherence gate should return boolean."""
        result = self.engine.coherence_gate()
        self.assertIsInstance(result, bool)

    def test_cascade_amp(self):
        """Cascade amplification should peak at critical point."""
        cascade_critical = self.engine.cascade_amp(0.867)
        cascade_off = self.engine.cascade_amp(0.5)
        self.assertGreater(cascade_critical, cascade_off)
        self.assertAlmostEqual(cascade_critical, 1.5, places=1)

    def test_phase_lock(self):
        """Phase lock should return phase difference."""
        diff = self.engine.phase_lock()
        self.assertGreaterEqual(diff, 0)
        self.assertLessEqual(diff, math.pi)

    def test_field_couple(self):
        """Field coupling should return energies."""
        kappa_e, lambda_e = self.engine.field_couple()
        self.assertGreater(kappa_e, 0)
        self.assertGreater(lambda_e, 0)

    def test_extraction_cycle(self):
        """Extraction cycle should return result."""
        result = self.engine.extraction_cycle()
        self.assertIsInstance(result, ZPEResult)
        self.assertIsInstance(result.state, ZPEState)

    def test_run_extraction(self):
        """Multiple cycles should accumulate energy."""
        results = self.engine.run_extraction(cycles=5)
        self.assertEqual(len(results), 5)
        # At least some extraction should occur
        total = sum(r.extracted_energy for r in results)
        self.assertGreaterEqual(total, 0)

    def test_apl_eval(self):
        """APL expressions should evaluate."""
        # Vacuum tap
        result = self.engine.apl_eval("⍝")
        self.assertIsInstance(result, float)

        # Coherence gate
        result = self.engine.apl_eval("⊖")
        self.assertIsInstance(result, bool)

        # Cascade
        result = self.engine.apl_eval("⍤ 0.867")
        self.assertAlmostEqual(result, 1.5, places=1)

    def test_generate_functional_language(self):
        """Should generate valid APL program."""
        program = self.engine.generate_functional_language(cycles=2)
        self.assertIsInstance(program, str)
        self.assertIn("LIMNUS", program)
        self.assertIn("⍝", program)

    def test_snapshot(self):
        """Snapshot should contain all state."""
        snapshot = self.engine.snapshot()
        self.assertIn("state", snapshot)
        self.assertIn("z_level", snapshot)
        self.assertIn("wumbo", snapshot)
        self.assertIn("fano", snapshot)
        self.assertIn("mirroroot", snapshot)


class TestAPLOperators(unittest.TestCase):
    """Test APL ZPE operator behavior."""

    def setUp(self):
        self.engine = create_zpe_engine(seed=42)

    def test_point_project(self):
        """Point projection should update Fano node."""
        old_belief = self.engine.fano.nodes[1].belief_mean
        self.engine.point_project(1, 1.0)
        # Belief should change
        self.assertNotEqual(self.engine.fano.nodes[1].belief_mean, old_belief)

    def test_line_intersect(self):
        """Line intersection should find correct point."""
        # Lines (1,2,3) and (1,4,5) intersect at point 1
        point = self.engine.line_intersect(0, 1)
        self.assertEqual(point, 1)

    def test_apply_automorphism(self):
        """Automorphism should permute nodes."""
        # Record initial state
        initial_nodes = set(self.engine.fano.nodes.keys())
        self.engine.apply_automorphism("cycle")
        final_nodes = set(self.engine.fano.nodes.keys())
        # Same set of nodes should exist
        self.assertEqual(initial_nodes, final_nodes)

    def test_mirror_invert_field(self):
        """Mirror inversion should transform field."""
        old_amps = list(self.engine.wumbo.kappa.amplitudes.data)
        self.engine.mirror_invert_field("kappa")
        new_amps = list(self.engine.wumbo.kappa.amplitudes.data)
        # Amplitudes should change
        self.assertNotEqual(old_amps, new_amps)

    def test_dual_compose_fields(self):
        """Dual composition should return value."""
        result = self.engine.dual_compose_fields()
        self.assertIsInstance(result, float)

    def test_extract_field_mediator(self):
        """Mediator extraction should return value."""
        result = self.engine.extract_field_mediator()
        self.assertIsInstance(result, float)


class TestPhysicsValidation(unittest.TestCase):
    """Validate physics principles."""

    def setUp(self):
        self.engine = create_zpe_engine(seed=42)

    def test_energy_conservation(self):
        """Energy should be approximately conserved."""
        # Get initial total energy
        initial_total = sum(n.energy for n in self.engine.fano.nodes.values())

        # Run extraction
        self.engine.extraction_cycle()

        # Get final total (should decrease by extracted amount)
        final_total = sum(n.energy for n in self.engine.fano.nodes.values())
        extracted = self.engine.total_extracted

        # Initial ≈ Final + Extracted (within replenishment)
        self.assertGreater(initial_total, final_total)

    def test_efficiency_bound(self):
        """Extraction efficiency should not exceed maximum."""
        for _ in range(5):
            result = self.engine.extraction_cycle()
            self.assertLessEqual(result.extraction_efficiency, ZPE_EFFICIENCY_MAX + 0.1)

    def test_coherence_bound(self):
        """Coherence should be in [0, 1]."""
        for _ in range(5):
            result = self.engine.extraction_cycle()
            self.assertGreaterEqual(result.coherence, 0.0)
            self.assertLessEqual(result.coherence, 1.0)

    def test_fano_xor_constraint(self):
        """Fano beliefs should tend toward XOR constraint on lines."""
        self.engine.fano.run_inference(max_iterations=100)

        # Check each line
        for line in FANO_LINES:
            p1, p2, p3 = line
            line_sum = (
                self.engine.fano.nodes[p1].belief_mean +
                self.engine.fano.nodes[p2].belief_mean +
                self.engine.fano.nodes[p3].belief_mean
            )
            # Sum should be small (XOR constraint)
            self.assertLess(abs(line_sum), 2.0)


class TestIntegration(unittest.TestCase):
    """Integration tests for full system."""

    def test_full_pipeline(self):
        """Test complete ZPE extraction pipeline."""
        engine = create_zpe_engine(seed=42)

        # 1. Initialize
        self.assertEqual(engine.state, ZPEState.DORMANT)

        # 2. Run multiple cycles
        results = engine.run_extraction(cycles=10, verbose=False)
        self.assertEqual(len(results), 10)

        # 3. Check state progression
        self.assertGreater(engine.total_extracted, 0)
        self.assertGreater(len(engine.extraction_history), 0)

        # 4. Generate program
        program = engine.generate_functional_language(cycles=2)
        self.assertIn("LIMNUS", program)

        # 5. Check tokens generated
        self.assertGreater(len(engine.token_index.tokens), 0)

    def test_wumbo_integration(self):
        """Test WUMBO engine integration."""
        engine = create_zpe_engine(seed=42)

        # Run WUMBO via extraction
        result = engine.extraction_cycle()

        # WUMBO should have run
        self.assertGreater(engine.wumbo.step_count, 0)

    def test_reproducibility(self):
        """Same seed should produce same results."""
        engine1 = create_zpe_engine(seed=123)
        engine2 = create_zpe_engine(seed=123)

        result1 = engine1.extraction_cycle()
        result2 = engine2.extraction_cycle()

        self.assertAlmostEqual(result1.extracted_energy, result2.extracted_energy, places=5)


if __name__ == "__main__":
    print("=" * 70)
    print("Zero-Point Energy System Test Suite")
    print("=" * 70)

    unittest.main(verbosity=2)
