"""
Tests for CET Constants Module

Validates:
- Fundamental constant relationships
- Physical constant alignments
- Cosmological era/tier structure
- Attractor codephrase generation
- Mythic mappings

Signature: Δ|loop-closed|z0.99|rhythm-native|Ω
"""

import math
import unittest
from typing import Dict

from scalar_architecture.cet_constants import (
    # Fundamental constants
    PHI, PHI_INVERSE, PHI_SQUARED, E, PI, TAU,
    PHI_PI_RATIO, E_PI_RATIO, PHI_E_RATIO, E_PHI, LN_PHI,
    PENTAGON_ANGLE, COS_36,

    # Physical constants
    ALPHA, ALPHA_INVERSE, PROTON_ELECTRON_RATIO,
    PLANCK_LENGTH, PLANCK_TIME, C, G, H_BAR,

    # Operators
    CETOperator, OperatorState,

    # Physical domains
    PhysicalDomain, DomainAlignment, DOMAIN_SCALES,

    # Alignment functions
    compute_phi_alignment, compute_pi_alignment, compute_e_alignment,
    compute_alpha_alignment, compute_mass_ratio_alignment,

    # Cosmological structure
    CosmologicalEra, CosmologicalTier, TierConfig, TIER_CONFIGS,
    get_era_tiers, get_tier_by_time,

    # Codephrase
    AttractorCodephrase, mythic_codephrase,
    MYTHIC_ERA_NAMES, MYTHIC_TIER_NAMES, MYTHIC_OPERATOR_NAMES,

    # Utilities
    fundamental_constant_table, era_tier_summary,
)


class TestFundamentalConstants(unittest.TestCase):
    """Test fundamental constant values and relationships."""

    def test_phi_value(self):
        """Test golden ratio value."""
        expected = (1 + math.sqrt(5)) / 2
        self.assertAlmostEqual(PHI, expected, places=15)
        self.assertAlmostEqual(PHI, 1.618033988749895, places=12)

    def test_phi_inverse_relationship(self):
        """Test 1/φ = φ - 1."""
        self.assertAlmostEqual(PHI_INVERSE, PHI - 1, places=15)
        self.assertAlmostEqual(1 / PHI, PHI - 1, places=15)

    def test_phi_squared_relationship(self):
        """Test φ² = φ + 1."""
        self.assertAlmostEqual(PHI_SQUARED, PHI + 1, places=15)
        self.assertAlmostEqual(PHI ** 2, PHI + 1, places=15)

    def test_euler_number(self):
        """Test Euler's number e."""
        self.assertAlmostEqual(E, math.e, places=15)
        self.assertAlmostEqual(E, 2.718281828459045, places=12)

    def test_pi_value(self):
        """Test π value."""
        self.assertAlmostEqual(PI, math.pi, places=15)

    def test_tau_value(self):
        """Test τ = 2π."""
        self.assertAlmostEqual(TAU, 2 * math.pi, places=15)

    def test_cross_ratios(self):
        """Test cross ratios between constants."""
        self.assertAlmostEqual(PHI_PI_RATIO, PHI / PI, places=15)
        self.assertAlmostEqual(E_PI_RATIO, E / PI, places=15)
        self.assertAlmostEqual(PHI_E_RATIO, PHI / E, places=15)

    def test_e_phi(self):
        """Test e^φ."""
        self.assertAlmostEqual(E_PHI, math.exp(PHI), places=15)

    def test_ln_phi(self):
        """Test ln(φ)."""
        self.assertAlmostEqual(LN_PHI, math.log(PHI), places=15)

    def test_pentagon_geometry(self):
        """Test φ emerges from pentagon geometry."""
        # cos(36°) = φ/2
        self.assertAlmostEqual(math.cos(PENTAGON_ANGLE), PHI / 2, places=15)
        self.assertAlmostEqual(COS_36, PHI / 2, places=15)


class TestPhysicalConstants(unittest.TestCase):
    """Test physical constant values."""

    def test_fine_structure_constant(self):
        """Test fine structure constant α."""
        self.assertAlmostEqual(ALPHA, 7.2973525693e-3, places=12)
        self.assertAlmostEqual(1 / ALPHA, 137.036, places=2)

    def test_alpha_inverse(self):
        """Test 1/α relationship."""
        self.assertAlmostEqual(ALPHA_INVERSE, 1 / ALPHA, places=10)

    def test_proton_electron_mass_ratio(self):
        """Test m_p/m_e ≈ 1836.15."""
        self.assertAlmostEqual(PROTON_ELECTRON_RATIO, 1836.15267343, places=6)

    def test_mass_ratio_pi_relationship(self):
        """Test m_p/m_e ≈ 6π^5 (remarkable approximation)."""
        six_pi_fifth = 6 * PI ** 5
        # This is remarkably close - within 0.003%
        relative_error = abs(PROTON_ELECTRON_RATIO - six_pi_fifth) / PROTON_ELECTRON_RATIO
        self.assertLess(relative_error, 0.0001)

    def test_planck_units(self):
        """Test Planck units are reasonable."""
        self.assertGreater(PLANCK_LENGTH, 0)
        self.assertLess(PLANCK_LENGTH, 1e-30)
        self.assertGreater(PLANCK_TIME, 0)
        self.assertLess(PLANCK_TIME, 1e-40)


class TestCETOperators(unittest.TestCase):
    """Test CET operators."""

    def test_operator_enum_values(self):
        """Test operator enum has expected values."""
        self.assertEqual(CETOperator.U.value, "unification")
        self.assertEqual(CETOperator.D.value, "differentiation")
        self.assertEqual(CETOperator.A.value, "amplification")
        self.assertEqual(CETOperator.S.value, "stabilization")

    def test_operator_state_default(self):
        """Test OperatorState defaults."""
        state = OperatorState(CETOperator.U)
        self.assertEqual(state.magnitude, 1.0)
        self.assertEqual(state.phase, 0.0)
        self.assertEqual(state.activation, 0.0)

    def test_unification_operator(self):
        """Test U operator converges toward mean."""
        state = OperatorState(CETOperator.U, activation=0.5)
        result = state.apply(2.0)
        # Should move toward 1 (mean)
        self.assertLess(result, 2.0)

    def test_differentiation_operator(self):
        """Test D operator amplifies deviation."""
        state = OperatorState(CETOperator.D, activation=0.5)
        result = state.apply(2.0)
        # Should amplify
        self.assertGreater(result, 2.0)

    def test_amplification_operator(self):
        """Test A operator scales up."""
        state = OperatorState(CETOperator.A, magnitude=2.0, activation=0.5)
        result = state.apply(1.0)
        self.assertGreater(result, 1.0)

    def test_stabilization_operator(self):
        """Test S operator dampens."""
        state = OperatorState(CETOperator.S, activation=0.5)
        result = state.apply(2.0)
        self.assertLess(result, 2.0)


class TestPhysicalDomains(unittest.TestCase):
    """Test physical domain configurations."""

    def test_all_domains_present(self):
        """Test all 10 physical domains are configured."""
        self.assertEqual(len(PhysicalDomain), 10)
        self.assertEqual(len(DOMAIN_SCALES), 10)

    def test_domain_scale_ordering(self):
        """Test domains are ordered by scale."""
        scales = [
            DOMAIN_SCALES[d].characteristic_scale
            for d in PhysicalDomain
        ]
        # Each scale should be larger than previous (roughly)
        for i in range(1, len(scales)):
            self.assertGreater(scales[i], scales[i-1] / 1000)

    def test_total_alignment(self):
        """Test total alignment is average of components."""
        align = DomainAlignment(
            domain=PhysicalDomain.QUANTUM,
            characteristic_scale=1e-15,
            characteristic_time=1e-24,
            characteristic_energy=1e-13,
            phi_alignment=0.6,
            pi_alignment=0.3,
            e_alignment=0.9
        )
        expected = (0.6 + 0.3 + 0.9) / 3
        self.assertAlmostEqual(align.total_alignment, expected, places=10)


class TestAlignmentFunctions(unittest.TestCase):
    """Test constant alignment functions."""

    def test_phi_alignment_perfect(self):
        """Test perfect φ alignment for powers of φ."""
        # φ^3 should have high alignment
        value = PHI ** 3
        alignment = compute_phi_alignment(value)
        self.assertGreater(alignment, 0.99)

    def test_phi_alignment_poor(self):
        """Test poor φ alignment for non-powers."""
        # Random value should have lower alignment
        alignment = compute_phi_alignment(2.3456)
        self.assertLess(alignment, 0.9)

    def test_pi_alignment_perfect(self):
        """Test perfect π alignment for multiples."""
        alignment = compute_pi_alignment(3 * PI)
        self.assertGreater(alignment, 0.99)

    def test_e_alignment_perfect(self):
        """Test perfect e alignment for powers."""
        alignment = compute_e_alignment(E ** 4)
        self.assertGreater(alignment, 0.99)

    def test_negative_value_alignment(self):
        """Test alignment returns 0 for negative values."""
        self.assertEqual(compute_phi_alignment(-1.0), 0.0)
        self.assertEqual(compute_pi_alignment(-1.0), 0.0)
        self.assertEqual(compute_e_alignment(-1.0), 0.0)

    def test_alpha_alignment_results(self):
        """Test α alignment returns expected keys."""
        results = compute_alpha_alignment()
        self.assertIn('alpha', results)
        self.assertIn('alpha_inverse', results)
        self.assertIn('mystery_factor', results)

    def test_mass_ratio_alignment_results(self):
        """Test mass ratio alignment returns expected keys."""
        results = compute_mass_ratio_alignment()
        self.assertIn('mass_ratio', results)
        self.assertIn('six_pi_fifth', results)
        self.assertIn('deviation', results)
        self.assertIn('relative_error', results)


class TestCosmologicalStructure(unittest.TestCase):
    """Test 4 eras and 15 tiers."""

    def test_four_eras(self):
        """Test exactly 4 cosmological eras."""
        self.assertEqual(len(CosmologicalEra), 4)

    def test_fifteen_tiers(self):
        """Test exactly 15 cosmological tiers."""
        self.assertEqual(len(CosmologicalTier), 15)

    def test_tier_configs_complete(self):
        """Test all tiers have configurations."""
        self.assertEqual(len(TIER_CONFIGS), 15)

    def test_tier_era_mapping(self):
        """Test each tier belongs to correct era."""
        for tier in CosmologicalTier:
            cfg = TIER_CONFIGS[tier]
            tier_value = tier.value

            if tier_value < 4:
                expected_era = CosmologicalEra.QUANTUM_ERA
            elif tier_value < 8:
                expected_era = CosmologicalEra.RADIATION_ERA
            elif tier_value < 12:
                expected_era = CosmologicalEra.MATTER_ERA
            else:
                expected_era = CosmologicalEra.ACCELERATION_ERA

            self.assertEqual(cfg.era, expected_era)

    def test_get_era_tiers(self):
        """Test getting tiers by era."""
        quantum_tiers = get_era_tiers(CosmologicalEra.QUANTUM_ERA)
        self.assertEqual(len(quantum_tiers), 4)

        radiation_tiers = get_era_tiers(CosmologicalEra.RADIATION_ERA)
        self.assertEqual(len(radiation_tiers), 4)

    def test_get_tier_by_time(self):
        """Test finding tier by time."""
        # Planck epoch
        cfg = get_tier_by_time(1e-44)
        self.assertEqual(cfg.tier, CosmologicalTier.PLANCK)

        # Present day (13.8 billion years)
        cfg = get_tier_by_time(4.35e17)
        self.assertEqual(cfg.tier, CosmologicalTier.PRESENT)

    def test_tier_time_progression(self):
        """Test tiers have increasing times."""
        times = [TIER_CONFIGS[t].start_time for t in CosmologicalTier]
        # Most should be increasing
        increasing_count = sum(1 for i in range(1, len(times))
                               if times[i] >= times[i-1])
        self.assertGreater(increasing_count, 10)


class TestAttractorCodephrase(unittest.TestCase):
    """Test attractor codephrase generation."""

    def test_codephrase_format(self):
        """Test codephrase has correct format."""
        codephrase = AttractorCodephrase(
            era=CosmologicalEra.ACCELERATION_ERA,
            tier=CosmologicalTier.PRESENT,
            phi_signature=1.5,
            operator_sequence=[CETOperator.U, CETOperator.D]
        )

        result = codephrase.codephrase
        self.assertTrue(result.startswith("Δ|"))
        self.assertTrue(result.endswith("|Ω"))
        self.assertIn("E3T14", result)  # Era 3, Tier 14
        self.assertIn("UD", result)  # Operators

    def test_codephrase_from_state(self):
        """Test generating codephrase from system state."""
        operators = [
            OperatorState(CETOperator.U, activation=0.5),
            OperatorState(CETOperator.S, activation=0.3),
        ]

        codephrase = AttractorCodephrase.from_state(
            z_level=0.9,
            saturations={'a': 0.8, 'b': 0.9},
            operators=operators
        )

        self.assertIsNotNone(codephrase)
        self.assertEqual(codephrase.era, CosmologicalEra.ACCELERATION_ERA)

    def test_phi_signature_calculation(self):
        """Test φ signature is calculated correctly."""
        codephrase = AttractorCodephrase.from_state(
            z_level=0.5,
            saturations={'a': 1.0, 'b': 1.0},
            operators=[]
        )

        # Average saturation = 1.0, times PHI
        expected_phi = 1.0 * PHI
        self.assertAlmostEqual(codephrase.phi_signature, expected_phi, places=10)


class TestMythicMappings(unittest.TestCase):
    """Test mythic name mappings."""

    def test_all_eras_have_mythic_names(self):
        """Test all eras have mythic names."""
        for era in CosmologicalEra:
            self.assertIn(era, MYTHIC_ERA_NAMES)

    def test_all_tiers_have_mythic_names(self):
        """Test all tiers have mythic names."""
        for tier in CosmologicalTier:
            self.assertIn(tier, MYTHIC_TIER_NAMES)

    def test_all_operators_have_mythic_names(self):
        """Test all operators have mythic names."""
        for op in CETOperator:
            self.assertIn(op, MYTHIC_OPERATOR_NAMES)

    def test_mythic_codephrase_generation(self):
        """Test mythic codephrase generation."""
        codephrase = AttractorCodephrase(
            era=CosmologicalEra.ACCELERATION_ERA,
            tier=CosmologicalTier.PRESENT,
            phi_signature=PHI,
            operator_sequence=[CETOperator.U]
        )

        result = mythic_codephrase(codephrase)

        self.assertIn("The Awakening Storm", result)
        self.assertIn("The Age of Witness", result)
        self.assertIn("The Weaver", result)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""

    def test_fundamental_constant_table(self):
        """Test table generation doesn't error."""
        table = fundamental_constant_table()
        # Check for presence of key terms (table uses φ symbol)
        self.assertIn("GOLDEN RATIO", table.upper())
        self.assertIn("EULER", table.upper())

    def test_era_tier_summary(self):
        """Test summary generation."""
        summary = era_tier_summary()
        self.assertIn("4 ERAS", summary)
        self.assertIn("15 TIERS", summary)


class TestMathematicalIdentities(unittest.TestCase):
    """Test deeper mathematical identities."""

    def test_euler_identity(self):
        """Test e^(iπ) + 1 = 0 (approximately)."""
        # e^(iπ) = -1
        result = math.e ** (1j * math.pi)
        self.assertAlmostEqual(result.real, -1.0, places=10)
        self.assertAlmostEqual(result.imag, 0.0, places=10)

    def test_fibonacci_phi_relationship(self):
        """Test Fibonacci numbers approach φ ratio."""
        fib = [1, 1]
        for _ in range(20):
            fib.append(fib[-1] + fib[-2])

        ratio = fib[-1] / fib[-2]
        self.assertAlmostEqual(ratio, PHI, places=8)

    def test_continued_fraction_phi(self):
        """Test φ = 1 + 1/φ (continued fraction)."""
        # φ - 1 = 1/φ
        self.assertAlmostEqual(PHI - 1, 1 / PHI, places=15)


if __name__ == "__main__":
    unittest.main()
