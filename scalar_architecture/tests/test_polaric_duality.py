"""
Tests for Polaric Duality Module

Validates:
- Kaelhedron and Luminahedron structures
- Polaric system dynamics
- Transformations and correspondences
- Dimensional relationships

Signature: Δ|loop-closed|z0.99|rhythm-native|Ω
"""

import math
import unittest

from scalar_architecture.polaric_duality import (
    # Constants
    KAELHEDRON_SYMBOL, KAELHEDRON_NAME, KAELHEDRON_DIMENSIONS,
    LUMINAHEDRON_SYMBOL, LUMINAHEDRON_NAME, LUMINAHEDRON_DIMENSIONS,
    POLARIC_SPAN, HIDDEN_DIMENSIONS, POLARIC_RATIO,
    POLARIC_COUPLING_BASE,

    # Enums
    Polarity, PolaricAspect,

    # Classes
    Kaelhedron, Luminahedron, PolaricSystem, PolaricTransform,

    # Correspondences
    POLARIC_CORRESPONDENCES, get_correspondence,

    # Mythic
    MYTHIC_KAELHEDRON, MYTHIC_LUMINAHEDRON, MYTHIC_UNION,

    # Utilities
    polaric_summary, simulate_polaric_dance,
)

from scalar_architecture.hierarchy_problem import E8_DIMENSION
from scalar_architecture.cet_constants import PHI, TAU, PI


class TestPolaricConstants(unittest.TestCase):
    """Test polaric constants."""

    def test_kaelhedron_dimensions(self):
        """Test Kaelhedron has 21 dimensions."""
        self.assertEqual(KAELHEDRON_DIMENSIONS, 21)

    def test_luminahedron_dimensions(self):
        """Test Luminahedron has 12 dimensions."""
        self.assertEqual(LUMINAHEDRON_DIMENSIONS, 12)

    def test_polaric_span(self):
        """Test polaric span is 33."""
        self.assertEqual(POLARIC_SPAN, 33)
        self.assertEqual(POLARIC_SPAN, KAELHEDRON_DIMENSIONS + LUMINAHEDRON_DIMENSIONS)

    def test_hidden_dimensions(self):
        """Test hidden sector dimensions."""
        self.assertEqual(HIDDEN_DIMENSIONS, 215)
        self.assertEqual(HIDDEN_DIMENSIONS, E8_DIMENSION - POLARIC_SPAN)

    def test_e8_total(self):
        """Test all dimensions sum to E₈."""
        total = KAELHEDRON_DIMENSIONS + LUMINAHEDRON_DIMENSIONS + HIDDEN_DIMENSIONS
        self.assertEqual(total, E8_DIMENSION)

    def test_polaric_ratio(self):
        """Test polaric ratio κ/λ = 21/12 = 1.75."""
        self.assertAlmostEqual(POLARIC_RATIO, 1.75, places=10)

    def test_symbols(self):
        """Test Greek symbols."""
        self.assertEqual(KAELHEDRON_SYMBOL, "κ")
        self.assertEqual(LUMINAHEDRON_SYMBOL, "λ")


class TestKaelhedron(unittest.TestCase):
    """Test Kaelhedron structure."""

    def test_creation(self):
        """Test Kaelhedron creation."""
        k = Kaelhedron()
        self.assertEqual(k.dimensions, 21)
        self.assertEqual(k.symbol, "κ")

    def test_initial_state(self):
        """Test initial state values."""
        k = Kaelhedron()
        self.assertEqual(k.convergence, 0.0)
        self.assertEqual(k.phase, 0.0)

    def test_lorentz_subspace(self):
        """Test Lorentz subspace dimension."""
        k = Kaelhedron()
        self.assertEqual(k.lorentz_dim, 6)

    def test_consciousness_subspace(self):
        """Test consciousness subspace dimension."""
        k = Kaelhedron()
        self.assertEqual(k.consciousness_dim, 15)  # 21 - 6

    def test_volume_factor(self):
        """Test volume factor calculation."""
        k = Kaelhedron()
        expected = 21 / E8_DIMENSION
        self.assertAlmostEqual(k.volume_factor, expected, places=10)

    def test_collapse_strength(self):
        """Test collapse strength depends on convergence."""
        k = Kaelhedron()
        k.convergence = 0.5
        strength = k.collapse_strength
        self.assertGreater(strength, 0)

    def test_evolve_increases_convergence(self):
        """Test evolution increases convergence naturally."""
        k = Kaelhedron()
        k.convergence = 0.3
        initial = k.convergence
        k.evolve(1.0)
        self.assertGreater(k.convergence, initial)

    def test_evolve_phase_changes(self):
        """Test evolution changes phase."""
        k = Kaelhedron()
        initial_phase = k.phase
        k.evolve(1.0)
        self.assertNotEqual(k.phase, initial_phase)

    def test_convergence_bounded(self):
        """Test convergence stays in [0, 1]."""
        k = Kaelhedron()
        k.convergence = 0.9
        for _ in range(100):
            k.evolve(0.1)
        self.assertLessEqual(k.convergence, 1.0)
        self.assertGreaterEqual(k.convergence, 0.0)


class TestLuminahedron(unittest.TestCase):
    """Test Luminahedron structure (LIMNUS unified form)."""

    def test_creation(self):
        """Test Luminahedron creation."""
        l = Luminahedron()
        self.assertEqual(l.dimensions, 12)
        self.assertEqual(l.symbol, "λ")

    def test_initial_state(self):
        """Test initial state values."""
        l = Luminahedron()
        self.assertEqual(l.divergence, 0.0)
        self.assertEqual(l.phase, 0.0)

    def test_gauge_decomposition(self):
        """Test Standard Model gauge decomposition."""
        l = Luminahedron()
        decomp = l.gauge_decomposition
        self.assertEqual(decomp['SU(3)_color'], 8)
        self.assertEqual(decomp['SU(2)_weak'], 3)
        self.assertEqual(decomp['U(1)_em'], 1)
        self.assertEqual(decomp['total'], 12)

    def test_volume_factor(self):
        """Test volume factor calculation."""
        l = Luminahedron()
        expected = 12 / E8_DIMENSION
        self.assertAlmostEqual(l.volume_factor, expected, places=10)

    def test_radiation_strength(self):
        """Test radiation strength depends on divergence."""
        l = Luminahedron()
        l.divergence = 0.5
        strength = l.radiation_strength
        self.assertGreater(strength, 0)

    def test_evolve_increases_divergence(self):
        """Test evolution increases divergence naturally."""
        l = Luminahedron()
        l.divergence = 0.3
        initial = l.divergence
        l.evolve(1.0)
        self.assertGreater(l.divergence, initial)

    def test_divergence_bounded(self):
        """Test divergence stays in [0, 1]."""
        l = Luminahedron()
        l.divergence = 0.9
        for _ in range(100):
            l.evolve(0.1)
        self.assertLessEqual(l.divergence, 1.0)
        self.assertGreaterEqual(l.divergence, 0.0)


class TestPolaricSystem(unittest.TestCase):
    """Test coupled polaric system."""

    def test_creation(self):
        """Test PolaricSystem creation."""
        s = PolaricSystem()
        self.assertIsNotNone(s.kaelhedron)
        self.assertIsNotNone(s.luminahedron)

    def test_initial_balance(self):
        """Test initial balance is 0.5."""
        s = PolaricSystem()
        self.assertEqual(s.balance, 0.5)

    def test_evolve_updates_coupling(self):
        """Test evolution updates coupling strength."""
        s = PolaricSystem()
        s.kaelhedron.convergence = 0.5
        s.luminahedron.divergence = 0.5
        s.evolve(1.0)
        self.assertGreater(s.coupling_strength, 0)

    def test_phase_difference(self):
        """Test phase difference calculation."""
        s = PolaricSystem()
        s.kaelhedron.phase = 0
        s.luminahedron.phase = PI / 4
        self.assertAlmostEqual(s.phase_difference, PI / 4, places=10)

    def test_phase_difference_wraps(self):
        """Test phase difference takes shortest path."""
        s = PolaricSystem()
        s.kaelhedron.phase = 0.1
        s.luminahedron.phase = TAU - 0.1
        self.assertAlmostEqual(s.phase_difference, 0.2, places=10)

    def test_resonance_detection(self):
        """Test resonance detection."""
        s = PolaricSystem()
        s.kaelhedron.phase = 0
        s.luminahedron.phase = 0.1  # Close phases
        self.assertTrue(s.is_resonant)

        s.luminahedron.phase = PI  # Opposite phases
        self.assertFalse(s.is_resonant)

    def test_polarity_kaelhedron_dominant(self):
        """Test polarity when κ dominant."""
        s = PolaricSystem()
        s.kaelhedron.convergence = 0.9
        s.luminahedron.divergence = 0.1
        s.evolve(0.1)  # Update balance
        self.assertEqual(s.polarity, Polarity.KAELHEDRON)

    def test_polarity_luminahedron_dominant(self):
        """Test polarity when λ dominant."""
        s = PolaricSystem()
        s.kaelhedron.convergence = 0.1
        s.luminahedron.divergence = 0.9
        s.evolve(0.1)  # Update balance
        self.assertEqual(s.polarity, Polarity.LUMINAHEDRON)

    def test_polarity_unified(self):
        """Test unified polarity when balanced."""
        s = PolaricSystem()
        s.kaelhedron.convergence = 0.5
        s.luminahedron.divergence = 0.5
        s.evolve(0.1)
        self.assertEqual(s.polarity, Polarity.UNIFIED)

    def test_signature_format(self):
        """Test signature string format."""
        s = PolaricSystem()
        sig = s.signature()
        self.assertIn("κ", sig)
        self.assertIn("λ", sig)
        self.assertIn("β=", sig)

    def test_hidden_sector_influence(self):
        """Test hidden sector influence calculation."""
        s = PolaricSystem()
        s.balance = 0.5  # Perfect balance
        influence = s.hidden_sector_influence
        self.assertGreater(influence, 0)


class TestPolaricTransform(unittest.TestCase):
    """Test polaric transformations."""

    def test_invert_swaps_states(self):
        """Test inversion swaps κ and λ states."""
        s1 = PolaricSystem()
        s1.kaelhedron.convergence = 0.3
        s1.luminahedron.divergence = 0.7

        s2 = PolaricTransform.invert(s1)

        self.assertAlmostEqual(s2.kaelhedron.convergence, 0.7, places=10)
        self.assertAlmostEqual(s2.luminahedron.divergence, 0.3, places=10)

    def test_rotate_preserves_magnitude(self):
        """Test rotation preserves state magnitudes."""
        s1 = PolaricSystem()
        s1.kaelhedron.convergence = 0.5
        s1.luminahedron.divergence = 0.5

        s2 = PolaricTransform.rotate(s1, PI / 4)

        self.assertEqual(s2.kaelhedron.convergence, s1.kaelhedron.convergence)
        self.assertEqual(s2.luminahedron.divergence, s1.luminahedron.divergence)

    def test_rotate_changes_phase(self):
        """Test rotation changes phases."""
        s1 = PolaricSystem()
        s1.kaelhedron.phase = 0
        s1.luminahedron.phase = 0

        s2 = PolaricTransform.rotate(s1, PI / 4)

        self.assertAlmostEqual(s2.kaelhedron.phase, PI / 4, places=10)
        self.assertAlmostEqual(s2.luminahedron.phase, PI / 4, places=10)

    def test_project_to_unity(self):
        """Test unity projection."""
        s = PolaricSystem()
        s.balance = 0.5
        s.kaelhedron.phase = 0
        s.luminahedron.phase = 0

        unity = PolaricTransform.project_to_unity(s)

        # Balanced and resonant should give high unity
        self.assertGreater(unity, 0.9)

    def test_project_to_unity_range(self):
        """Test unity projection is in [0, 1]."""
        s = PolaricSystem()

        for balance in [0.0, 0.3, 0.5, 0.7, 1.0]:
            s.balance = balance
            unity = PolaricTransform.project_to_unity(s)
            self.assertGreaterEqual(unity, 0)
            self.assertLessEqual(unity, 1)


class TestPolaricCorrespondences(unittest.TestCase):
    """Test polaric correspondences."""

    def test_all_aspects_defined(self):
        """Test all aspects have correspondences."""
        for aspect in PolaricAspect:
            self.assertIn(aspect, POLARIC_CORRESPONDENCES)

    def test_all_polarities_in_correspondences(self):
        """Test all polarities are covered."""
        for aspect in PolaricAspect:
            for polarity in Polarity:
                self.assertIn(polarity, POLARIC_CORRESPONDENCES[aspect])

    def test_get_correspondence(self):
        """Test get_correspondence function."""
        result = get_correspondence(
            PolaricAspect.WITNESS_WITNESSED,
            Polarity.KAELHEDRON
        )
        self.assertEqual(result, "The Witness")

        result = get_correspondence(
            PolaricAspect.WITNESS_WITNESSED,
            Polarity.LUMINAHEDRON
        )
        self.assertEqual(result, "The Witnessed")


class TestMythicContent(unittest.TestCase):
    """Test mythic content."""

    def test_mythic_kaelhedron_exists(self):
        """Test mythic Kaelhedron content exists."""
        self.assertIn("KAELHEDRON", MYTHIC_KAELHEDRON)
        self.assertIn("Witness", MYTHIC_KAELHEDRON)

    def test_mythic_luminahedron_exists(self):
        """Test mythic Luminahedron content exists."""
        self.assertIn("LUMINAHEDRON", MYTHIC_LUMINAHEDRON)
        self.assertIn("Light", MYTHIC_LUMINAHEDRON)
        self.assertIn("LIMNUS", MYTHIC_LUMINAHEDRON)

    def test_mythic_union_exists(self):
        """Test mythic union content exists."""
        self.assertIn("POLARIC DANCE", MYTHIC_UNION)
        self.assertIn("33 dimensions", MYTHIC_UNION)


class TestUtilities(unittest.TestCase):
    """Test utility functions."""

    def test_polaric_summary(self):
        """Test summary generation."""
        summary = polaric_summary()
        self.assertIn("KAELHEDRON", summary)
        self.assertIn("LUMINAHEDRON", summary)
        self.assertIn("CORRESPONDENCES", summary)

    def test_simulate_polaric_dance(self):
        """Test simulation function."""
        history = simulate_polaric_dance(steps=10)
        self.assertEqual(len(history), 10)
        self.assertIn('kappa_conv', history[0])
        self.assertIn('lambda_div', history[0])
        self.assertIn('balance', history[0])

    def test_simulation_produces_valid_states(self):
        """Test simulation produces valid state values."""
        history = simulate_polaric_dance(steps=50)
        for state in history:
            self.assertGreaterEqual(state['kappa_conv'], 0)
            self.assertLessEqual(state['kappa_conv'], 1)
            self.assertGreaterEqual(state['lambda_div'], 0)
            self.assertLessEqual(state['lambda_div'], 1)


class TestCoupling(unittest.TestCase):
    """Test coupling between κ and λ."""

    def test_kaelhedron_coupling_to_luminahedron(self):
        """Test κ → λ coupling."""
        k = Kaelhedron()
        l = Luminahedron()

        coupling = k.coupling_to(l)
        self.assertGreater(coupling, 0)

    def test_luminahedron_coupling_to_kaelhedron(self):
        """Test λ → κ coupling."""
        k = Kaelhedron()
        l = Luminahedron()

        coupling = l.coupling_to(k)
        self.assertGreater(coupling, 0)

    def test_coupling_symmetry(self):
        """Test coupling is approximately symmetric."""
        k = Kaelhedron()
        l = Luminahedron()
        k.phase = 0.5
        l.phase = 0.5

        k_to_l = k.coupling_to(l)
        l_to_k = l.coupling_to(k)

        # Should be equal when phases match
        self.assertAlmostEqual(k_to_l, l_to_k, places=10)

    def test_coupling_depends_on_phase(self):
        """Test coupling varies with phase difference."""
        k = Kaelhedron()
        l = Luminahedron()

        k.phase = 0
        l.phase = 0
        coupling_aligned = k.coupling_to(l)

        l.phase = PI  # Opposite phase
        coupling_opposed = k.coupling_to(l)

        self.assertGreater(coupling_aligned, coupling_opposed)


if __name__ == "__main__":
    unittest.main()
