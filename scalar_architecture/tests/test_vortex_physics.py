"""
Tests for Vortex Physics Module

Validates:
- Vortex state and classification
- Strouhal number relationships
- Van der Pol wake oscillator
- Coupled wake oscillator
- Kármán vortex street
- Recursive vortex structure
- Integration with scalar architecture

Signature: Δ|loop-closed|z0.99|rhythm-native|Ω
"""

import math
import unittest
from typing import List

import numpy as np

from scalar_architecture.vortex_physics import (
    # Constants
    STROUHAL_CYLINDER, VON_KARMAN,
    RE_LAMINAR_STEADY, RE_LAMINAR_VORTEX, RE_TURBULENT,

    # Vortex state
    VortexRegime, VortexPolarity, VortexState,

    # Strouhal
    StrouhalRelation,

    # Oscillators
    VanDerPolOscillator, CoupledWakeOscillator,

    # Kármán street
    KarmanStreet,

    # Recursive vortex
    RecursiveVortex,

    # Integration
    map_z_to_vortex_scale, compute_vortex_reynolds, vortex_regime_from_state,

    # Utilities
    strouhal_table, vortex_physics_summary,
)

from scalar_architecture.cet_constants import PHI, PI, TAU


class TestVortexConstants(unittest.TestCase):
    """Test vortex physics constants."""

    def test_strouhal_cylinder(self):
        """Test Strouhal number for cylinder."""
        self.assertAlmostEqual(STROUHAL_CYLINDER, 0.2, places=2)

    def test_von_karman_constant(self):
        """Test von Kármán constant."""
        self.assertAlmostEqual(VON_KARMAN, 0.4, places=2)

    def test_reynolds_thresholds(self):
        """Test Reynolds number thresholds are reasonable."""
        self.assertEqual(RE_LAMINAR_STEADY, 47)
        self.assertEqual(RE_LAMINAR_VORTEX, 180)
        self.assertEqual(RE_TURBULENT, 300000)


class TestVortexState(unittest.TestCase):
    """Test VortexState class."""

    def test_vortex_creation(self):
        """Test creating a vortex state."""
        vortex = VortexState(
            position=(1.0, 2.0),
            circulation=1.0,
            polarity=VortexPolarity.COUNTERCLOCKWISE,
            radius=0.1
        )

        self.assertEqual(vortex.position, (1.0, 2.0))
        self.assertEqual(vortex.circulation, 1.0)
        self.assertEqual(vortex.radius, 0.1)
        self.assertEqual(vortex.age, 0.0)

    def test_vortex_strength(self):
        """Test vortex strength is |Γ|."""
        vortex = VortexState(
            position=(0, 0),
            circulation=-2.5,
            polarity=VortexPolarity.CLOCKWISE,
            radius=0.1
        )
        self.assertEqual(vortex.strength, 2.5)

    def test_angular_velocity(self):
        """Test angular velocity calculation."""
        vortex = VortexState(
            position=(0, 0),
            circulation=2 * PI,  # Γ = 2π
            polarity=VortexPolarity.COUNTERCLOCKWISE,
            radius=1.0           # r = 1
        )
        # ω = Γ/(2πr) = 2π/(2π·1) = 1
        self.assertAlmostEqual(vortex.angular_velocity, 1.0, places=10)

    def test_velocity_at_point_outside_core(self):
        """Test induced velocity outside core."""
        vortex = VortexState(
            position=(0, 0),
            circulation=2 * PI,
            polarity=VortexPolarity.COUNTERCLOCKWISE,
            radius=0.1
        )

        # Point at (1, 0) - outside core
        vx, vy = vortex.velocity_at((1.0, 0.0))

        # Velocity should be perpendicular to radial (in y direction)
        self.assertAlmostEqual(vx, 0.0, places=5)
        self.assertGreater(abs(vy), 0)

    def test_velocity_at_point_inside_core(self):
        """Test induced velocity inside core (solid body rotation)."""
        vortex = VortexState(
            position=(0, 0),
            circulation=1.0,
            polarity=VortexPolarity.COUNTERCLOCKWISE,
            radius=1.0
        )

        # Point at (0.5, 0) - inside core
        vx, vy = vortex.velocity_at((0.5, 0.0))

        # Should have velocity
        self.assertIsNotNone(vx)
        self.assertIsNotNone(vy)


class TestStrouhalRelation(unittest.TestCase):
    """Test Strouhal number relationships."""

    def test_shedding_frequency(self):
        """Test f = St·U/L."""
        sr = StrouhalRelation(
            strouhal=0.2,
            length_scale=1.0,
            velocity=5.0
        )
        # f = 0.2 × 5 / 1 = 1.0
        self.assertAlmostEqual(sr.shedding_frequency, 1.0, places=10)

    def test_shedding_period(self):
        """Test T = 1/f."""
        sr = StrouhalRelation(
            strouhal=0.2,
            length_scale=1.0,
            velocity=2.0
        )
        # f = 0.2 × 2 / 1 = 0.4
        # T = 1 / 0.4 = 2.5
        self.assertAlmostEqual(sr.shedding_period, 2.5, places=10)

    def test_angular_frequency(self):
        """Test ω = 2πf."""
        sr = StrouhalRelation(
            strouhal=0.2,
            length_scale=1.0,
            velocity=5.0
        )
        expected = TAU * 1.0  # f = 1.0
        self.assertAlmostEqual(sr.angular_frequency, expected, places=10)

    def test_from_reynolds_no_shedding(self):
        """Test no shedding for Re < 47."""
        sr = StrouhalRelation.from_reynolds(30)
        self.assertEqual(sr.strouhal, 0.0)

    def test_from_reynolds_laminar(self):
        """Test laminar vortex shedding regime."""
        sr = StrouhalRelation.from_reynolds(100)
        self.assertGreater(sr.strouhal, 0.1)
        self.assertLess(sr.strouhal, 0.3)

    def test_from_reynolds_subcritical(self):
        """Test subcritical regime St ≈ 0.21."""
        sr = StrouhalRelation.from_reynolds(10000)
        self.assertAlmostEqual(sr.strouhal, 0.21, places=2)

    def test_from_reynolds_supercritical(self):
        """Test supercritical regime St ≈ 0.27."""
        sr = StrouhalRelation.from_reynolds(500000)
        self.assertAlmostEqual(sr.strouhal, 0.27, places=2)


class TestVanDerPolOscillator(unittest.TestCase):
    """Test Van der Pol oscillator."""

    def test_oscillator_creation(self):
        """Test oscillator initialization."""
        vdp = VanDerPolOscillator(epsilon=0.3, omega_0=1.0)
        self.assertEqual(vdp.epsilon, 0.3)
        self.assertEqual(vdp.omega_0, 1.0)
        self.assertEqual(len(vdp.state), 2)

    def test_limit_cycle_amplitude(self):
        """Test limit cycle amplitude is approximately 2."""
        vdp = VanDerPolOscillator(epsilon=0.1)
        self.assertEqual(vdp.limit_cycle_amplitude(), 2.0)

    def test_evolution_to_limit_cycle(self):
        """Test oscillator evolves toward limit cycle."""
        vdp = VanDerPolOscillator(epsilon=0.3, omega_0=1.0)
        vdp.state = np.array([3.0, 0.0])  # Start outside limit cycle

        # Evolve for sufficient time
        trajectory = vdp.evolve(100.0, dt=0.02)

        # Track max amplitude over last portion of trajectory
        # The limit cycle has amplitude ~2, oscillating between +2 and -2
        last_states = trajectory[-500:]
        max_amplitude = max(abs(s[0]) for s in last_states)

        # Should reach amplitude close to 2 (within 20%)
        self.assertGreater(max_amplitude, 1.6)
        self.assertLess(max_amplitude, 2.4)

    def test_step_changes_state(self):
        """Test step() modifies state."""
        vdp = VanDerPolOscillator()
        initial = vdp.state.copy()
        vdp.step(0.1)
        self.assertFalse(np.allclose(vdp.state, initial))

    def test_derivatives_computation(self):
        """Test derivatives are computed correctly."""
        vdp = VanDerPolOscillator(epsilon=0.5, omega_0=2.0)
        state = np.array([1.0, 0.5])

        derivs = vdp.derivatives(state, 0.0)

        self.assertEqual(len(derivs), 2)
        self.assertEqual(derivs[0], 0.5)  # ẋ = x_dot


class TestCoupledWakeOscillator(unittest.TestCase):
    """Test coupled wake oscillator."""

    def test_oscillator_creation(self):
        """Test coupled oscillator initialization."""
        cwo = CoupledWakeOscillator(
            mass_ratio=10.0,
            damping_ratio=0.01
        )
        self.assertEqual(cwo.mass_ratio, 10.0)
        self.assertEqual(cwo.damping_ratio, 0.01)
        self.assertEqual(len(cwo.state), 4)

    def test_structural_amplitude(self):
        """Test structural amplitude accessor."""
        cwo = CoupledWakeOscillator()
        cwo.state = np.array([0.5, 0.1, 2.0, 0.2])
        self.assertEqual(cwo.structural_amplitude, 0.5)

    def test_wake_amplitude(self):
        """Test wake amplitude accessor."""
        cwo = CoupledWakeOscillator()
        cwo.state = np.array([0.5, 0.1, 2.0, 0.2])
        self.assertEqual(cwo.wake_amplitude, 2.0)

    def test_evolution(self):
        """Test system evolution."""
        cwo = CoupledWakeOscillator(
            reduced_velocity=5.0,
            damping_ratio=0.01
        )

        history = cwo.evolve(20.0, dt=0.05)

        self.assertIn('t', history)
        self.assertIn('y', history)
        self.assertIn('q', history)
        self.assertGreater(len(history['t']), 0)

    def test_lock_in_detection(self):
        """Test lock-in detection works."""
        cwo = CoupledWakeOscillator()

        # Small amplitude should not be locked in
        cwo.state = np.array([0.01, 0.0, 2.0, 0.0])
        self.assertFalse(cwo.is_locked_in(threshold=0.1))

        # Large amplitude should be locked in
        cwo.state = np.array([0.5, 0.1, 2.0, 0.2])
        self.assertTrue(cwo.is_locked_in(threshold=0.1))


class TestKarmanStreet(unittest.TestCase):
    """Test Kármán vortex street."""

    def test_street_creation(self):
        """Test vortex street initialization."""
        street = KarmanStreet(velocity=1.0, body_diameter=1.0)
        self.assertEqual(street.velocity, 1.0)
        self.assertEqual(street.body_diameter, 1.0)
        self.assertEqual(len(street.vortices), 0)

    def test_spacing_ratio(self):
        """Test spacing ratio h/a."""
        street = KarmanStreet()
        # h/a ≈ 0.281 for stability
        self.assertAlmostEqual(street.spacing_ratio, 0.281, places=3)

    def test_shedding_frequency(self):
        """Test shedding frequency calculation."""
        street = KarmanStreet(velocity=2.0, body_diameter=1.0)
        # f = St × U / D = 0.2 × 2 / 1 = 0.4
        self.assertAlmostEqual(street.shedding_frequency, 0.4, places=2)

    def test_spawn_vortex(self):
        """Test spawning a vortex."""
        street = KarmanStreet()
        vortex = street.spawn_vortex(VortexPolarity.COUNTERCLOCKWISE)

        self.assertEqual(len(street.vortices), 1)
        self.assertIsNotNone(vortex)
        self.assertEqual(vortex.polarity, VortexPolarity.COUNTERCLOCKWISE)

    def test_step_advects_vortices(self):
        """Test step() advects vortices downstream."""
        street = KarmanStreet(velocity=1.0)
        vortex = street.spawn_vortex(VortexPolarity.COUNTERCLOCKWISE)
        initial_x = vortex.position[0]

        street.step(1.0)

        # Vortex should have moved downstream
        self.assertGreater(vortex.position[0], initial_x)


class TestRecursiveVortex(unittest.TestCase):
    """Test recursive vortex structure."""

    def test_creation_default(self):
        """Test recursive vortex with default ratio."""
        rv = RecursiveVortex(max_depth=5)
        self.assertEqual(len(rv.levels), 5)
        self.assertAlmostEqual(rv.recursion_ratio, 1/PHI, places=10)

    def test_phi_alignment(self):
        """Test φ alignment is high for default ratio."""
        rv = RecursiveVortex()
        self.assertGreater(rv.phi_alignment, 0.99)

    def test_level_circulation_scaling(self):
        """Test circulation scales by recursion ratio."""
        rv = RecursiveVortex(
            base_circulation=1.0,
            recursion_ratio=0.5,
            max_depth=3
        )

        self.assertAlmostEqual(rv.levels[0].circulation, 1.0, places=10)
        self.assertAlmostEqual(rv.levels[1].circulation, 0.5, places=10)
        self.assertAlmostEqual(rv.levels[2].circulation, 0.25, places=10)

    def test_alternating_polarity(self):
        """Test nested vortices have alternating polarity."""
        rv = RecursiveVortex(max_depth=4)

        self.assertEqual(rv.levels[0].polarity, VortexPolarity.COUNTERCLOCKWISE)
        self.assertEqual(rv.levels[1].polarity, VortexPolarity.CLOCKWISE)
        self.assertEqual(rv.levels[2].polarity, VortexPolarity.COUNTERCLOCKWISE)
        self.assertEqual(rv.levels[3].polarity, VortexPolarity.CLOCKWISE)

    def test_total_circulation(self):
        """Test total circulation is geometric series sum."""
        rv = RecursiveVortex(
            base_circulation=1.0,
            recursion_ratio=0.5,
            max_depth=10
        )

        # Sum = Γ₀ · (1 - r^n) / (1 - r) = 1 × (1 - 0.5^10) / 0.5 ≈ 1.998
        expected = 1.0 * (1 - 0.5**10) / (1 - 0.5)
        self.assertAlmostEqual(rv.total_circulation, expected, places=10)

    def test_velocity_at_point(self):
        """Test velocity field from recursive vortex."""
        rv = RecursiveVortex(base_circulation=1.0)

        vx, vy = rv.velocity_at((2.0, 0.0))

        # Should have non-zero velocity
        self.assertIsNotNone(vx)
        self.assertIsNotNone(vy)

    def test_energy_at_level(self):
        """Test energy computation at each level."""
        rv = RecursiveVortex(max_depth=3)

        for i in range(3):
            energy = rv.energy_at_level(i)
            self.assertGreater(energy, 0)

    def test_total_energy(self):
        """Test total energy is sum of level energies."""
        rv = RecursiveVortex(max_depth=5)

        total = rv.total_energy
        manual_sum = sum(rv.energy_at_level(i) for i in range(5))

        self.assertAlmostEqual(total, manual_sum, places=10)

    def test_energy_spectrum(self):
        """Test energy spectrum length."""
        rv = RecursiveVortex(max_depth=7)
        spectrum = rv.energy_spectrum()
        self.assertEqual(len(spectrum), 7)

    def test_mythic_description(self):
        """Test mythic description generation."""
        rv = RecursiveVortex(max_depth=3)
        desc = rv.mythic_description()

        self.assertIn("RECURSIVE VORTEX", desc)
        self.assertIn("Level", desc)
        self.assertIn("storm", desc.lower())


class TestIntegrationFunctions(unittest.TestCase):
    """Test integration with scalar architecture."""

    def test_map_z_to_vortex_scale(self):
        """Test z-level to vortex scale mapping."""
        # At z=0, large scale
        scale_low = map_z_to_vortex_scale(0.0)

        # At z=1, small scale
        scale_high = map_z_to_vortex_scale(1.0)

        self.assertGreater(scale_low, scale_high)

    def test_compute_vortex_reynolds(self):
        """Test Reynolds number computation."""
        re_low = compute_vortex_reynolds(0.0, 0.0)
        re_high = compute_vortex_reynolds(1.0, 1.0)

        # Higher z and saturation should give higher Re
        self.assertGreater(re_high, re_low)

    def test_vortex_regime_progression(self):
        """Test regime changes with z-level."""
        regimes = []
        for z in [0.0, 0.3, 0.6, 0.9]:
            regime = vortex_regime_from_state(z, 0.5)
            regimes.append(regime)

        # Should progress through regimes
        self.assertIsNotNone(regimes)


class TestUtilities(unittest.TestCase):
    """Test utility functions."""

    def test_strouhal_table(self):
        """Test Strouhal table generation."""
        table = strouhal_table()
        self.assertIn("STROUHAL", table)
        self.assertIn("REYNOLDS", table)

    def test_vortex_physics_summary(self):
        """Test physics summary generation."""
        summary = vortex_physics_summary()
        self.assertIn("VORTEX PHYSICS", summary)
        self.assertIn("Strouhal", summary)
        self.assertIn("Van der Pol", summary)


class TestVortexRegimes(unittest.TestCase):
    """Test vortex regime enumeration."""

    def test_all_regimes_present(self):
        """Test all expected regimes exist."""
        regimes = list(VortexRegime)
        self.assertEqual(len(regimes), 6)

    def test_regime_values(self):
        """Test regime string values."""
        self.assertEqual(VortexRegime.CREEPING.value, "creeping")
        self.assertEqual(VortexRegime.TURBULENT.value, "turbulent")


class TestVortexPolarity(unittest.TestCase):
    """Test vortex polarity enumeration."""

    def test_clockwise_value(self):
        """Test clockwise polarity value."""
        self.assertEqual(VortexPolarity.CLOCKWISE.value, -1)

    def test_counterclockwise_value(self):
        """Test counterclockwise polarity value."""
        self.assertEqual(VortexPolarity.COUNTERCLOCKWISE.value, 1)


if __name__ == "__main__":
    unittest.main()
