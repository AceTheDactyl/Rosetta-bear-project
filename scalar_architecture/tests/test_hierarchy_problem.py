"""
Tests for Hierarchy Problem Module

Validates:
- φ-hierarchy analysis
- E₈ volume factor calculations
- Recursion depth force coupling
- Kaelhedron sector analysis
- Combined hierarchy explanation

Signature: Δ|loop-closed|z0.99|rhythm-native|Ω
"""

import math
import unittest

from scalar_architecture.hierarchy_problem import (
    # Constants
    M_PLANCK, M_WEAK, M_GUT, M_PROTON,
    ALPHA_EM, ALPHA_WEAK, ALPHA_STRONG, ALPHA_GRAVITY,
    HIERARCHY_RATIO, EM_GRAVITY_RATIO,
    E8_DIMENSION, E8_RANK, LORENTZ_DIM, SM_GAUGE_DIM, KAELHEDRON_DIM,

    # φ-Hierarchy
    PhiHierarchy, compute_phi_hierarchy_spectrum,

    # E₈ Volume Factor
    E8Sector, E8VolumeFactor, E8_SECTORS, analyze_e8_dilution,

    # Force Activation
    FundamentalForce, ForceActivation, FORCE_ACTIVATIONS,
    compute_force_ratios_from_recursion,

    # Kaelhedron
    KaelhedronSector,

    # Combined
    HierarchyExplanation,

    # Analysis functions
    compute_higgs_vev_from_phi, analyze_fine_structure,
    hierarchy_summary,
)

from scalar_architecture.cet_constants import PHI, LN_PHI


class TestPhysicalConstants(unittest.TestCase):
    """Test physical constant values."""

    def test_planck_mass(self):
        """Test Planck mass is reasonable."""
        self.assertGreater(M_PLANCK, 1e18)
        self.assertLess(M_PLANCK, 1e20)

    def test_weak_scale(self):
        """Test weak scale (Higgs VEV)."""
        self.assertAlmostEqual(M_WEAK, 246.0, places=0)

    def test_hierarchy_ratio(self):
        """Test hierarchy ratio is ~10^17."""
        log_ratio = math.log10(HIERARCHY_RATIO)
        self.assertGreater(log_ratio, 16)
        self.assertLess(log_ratio, 18)

    def test_em_gravity_ratio(self):
        """Test EM/gravity ratio is huge."""
        log_ratio = math.log10(EM_GRAVITY_RATIO)
        self.assertGreater(log_ratio, 35)
        self.assertLess(log_ratio, 40)

    def test_e8_dimension(self):
        """Test E₈ dimension is 248."""
        self.assertEqual(E8_DIMENSION, 248)

    def test_e8_rank(self):
        """Test E₈ rank is 8."""
        self.assertEqual(E8_RANK, 8)


class TestPhiHierarchy(unittest.TestCase):
    """Test φ-hierarchy analysis."""

    def test_hierarchy_creation(self):
        """Test PhiHierarchy creation."""
        ph = PhiHierarchy(1000.0)
        self.assertIsNotNone(ph.phi_power)

    def test_phi_power_calculation(self):
        """Test φ-power is computed correctly."""
        # φ^10 should give phi_power ≈ 10
        value = PHI ** 10
        ph = PhiHierarchy(value)
        self.assertAlmostEqual(ph.phi_power, 10.0, places=10)

    def test_nearest_integer_power(self):
        """Test nearest integer power rounding."""
        ph = PhiHierarchy(PHI ** 10.3)
        self.assertEqual(ph.nearest_integer_power, 10)

        ph2 = PhiHierarchy(PHI ** 10.7)
        self.assertEqual(ph2.nearest_integer_power, 11)

    def test_fractional_deviation(self):
        """Test fractional deviation from integer."""
        ph = PhiHierarchy(PHI ** 10.2)
        self.assertAlmostEqual(ph.fractional_deviation, 0.2, places=5)

    def test_phi_resonant_exact(self):
        """Test exact φ-power is resonant."""
        ph = PhiHierarchy(PHI ** 10)
        self.assertTrue(ph.is_phi_resonant)

    def test_phi_resonant_close(self):
        """Test close to φ-power is resonant."""
        ph = PhiHierarchy(PHI ** 10.05)
        self.assertTrue(ph.is_phi_resonant)

    def test_phi_not_resonant(self):
        """Test far from φ-power is not resonant."""
        ph = PhiHierarchy(PHI ** 10.3)
        self.assertFalse(ph.is_phi_resonant)

    def test_reconstructed_ratio(self):
        """Test reconstructed ratio uses nearest integer."""
        ph = PhiHierarchy(PHI ** 10.2)
        self.assertAlmostEqual(ph.reconstructed_ratio, PHI ** 10, places=10)

    def test_reconstruction_accuracy(self):
        """Test reconstruction accuracy."""
        # Exact power should have accuracy = 1.0
        ph = PhiHierarchy(PHI ** 10)
        self.assertAlmostEqual(ph.reconstruction_accuracy, 1.0, places=10)

    def test_hierarchy_spectrum(self):
        """Test compute_phi_hierarchy_spectrum returns expected keys."""
        spectrum = compute_phi_hierarchy_spectrum()
        self.assertIn('planck_weak', spectrum)
        self.assertIn('em_gravity', spectrum)
        self.assertIn('proton_electron', spectrum)


class TestE8VolumeFactor(unittest.TestCase):
    """Test E₈ volume factor analysis."""

    def test_volume_factor_creation(self):
        """Test E8VolumeFactor creation."""
        vf = E8VolumeFactor(10)
        self.assertEqual(vf.sector_dim, 10)
        self.assertEqual(vf.total_dim, E8_DIMENSION)

    def test_dilution_ratio(self):
        """Test dilution ratio calculation."""
        vf = E8VolumeFactor(10, total_dim=100)
        self.assertEqual(vf.dilution_ratio, 0.1)

    def test_effective_coupling(self):
        """Test effective coupling with dilution."""
        vf = E8VolumeFactor(10, total_dim=100)
        # dilution = 0.1, power = 2
        result = vf.effective_coupling(1.0, power=2)
        self.assertAlmostEqual(result, 0.01, places=10)

    def test_power_for_target(self):
        """Test finding power for target ratio."""
        vf = E8VolumeFactor(10, total_dim=100)
        # dilution = 0.1
        # For target 0.01 from fundamental 1.0, need power 2
        power = vf.power_for_target(1.0, 0.01)
        self.assertAlmostEqual(power, 2.0, places=10)

    def test_e8_sectors_complete(self):
        """Test all E₈ sectors are defined."""
        self.assertEqual(len(E8_SECTORS), 5)
        for sector in E8Sector:
            self.assertIn(sector, E8_SECTORS)

    def test_gravity_sector_dimension(self):
        """Test gravity sector has Lorentz dimension."""
        gravity = E8_SECTORS[E8Sector.GRAVITY]
        self.assertEqual(gravity.sector_dim, LORENTZ_DIM)

    def test_e8_dilution_analysis(self):
        """Test E₈ dilution analysis returns expected keys."""
        results = analyze_e8_dilution()
        self.assertIn('gravity_power', results)
        self.assertIn('gravity_dilution', results)


class TestForceActivation(unittest.TestCase):
    """Test force activation model."""

    def test_force_activation_creation(self):
        """Test ForceActivation creation."""
        fa = ForceActivation(FundamentalForce.GRAVITY, 7)
        self.assertEqual(fa.force, FundamentalForce.GRAVITY)
        self.assertEqual(fa.activation_level, 7)

    def test_base_strength(self):
        """Test base strength is φ^(-R)."""
        fa = ForceActivation(FundamentalForce.GRAVITY, 3)
        expected = PHI ** (-3)
        self.assertAlmostEqual(fa.base_strength, expected, places=10)

    def test_relative_strength(self):
        """Test relative strength to strong force."""
        strong = FORCE_ACTIVATIONS[FundamentalForce.STRONG]
        gravity = FORCE_ACTIVATIONS[FundamentalForce.GRAVITY]

        # Gravity should be much weaker
        self.assertLess(gravity.relative_strength, strong.relative_strength)

    def test_all_forces_defined(self):
        """Test all fundamental forces have activations."""
        for force in FundamentalForce:
            self.assertIn(force, FORCE_ACTIVATIONS)

    def test_force_ordering(self):
        """Test forces are ordered by activation level."""
        strong = FORCE_ACTIVATIONS[FundamentalForce.STRONG]
        em = FORCE_ACTIVATIONS[FundamentalForce.ELECTROMAGNETIC]
        weak = FORCE_ACTIVATIONS[FundamentalForce.WEAK]
        gravity = FORCE_ACTIVATIONS[FundamentalForce.GRAVITY]

        # Strong < EM < Weak < Gravity (by activation level)
        self.assertLess(strong.activation_level, em.activation_level)
        self.assertLess(em.activation_level, weak.activation_level)
        self.assertLess(weak.activation_level, gravity.activation_level)

    def test_force_ratios_computation(self):
        """Test force ratios computation."""
        ratios = compute_force_ratios_from_recursion()
        self.assertIn('em_gravity_ratio', ratios)
        # EM should be stronger than gravity
        self.assertGreater(ratios['em_gravity_ratio'], 1.0)


class TestKaelhedronSector(unittest.TestCase):
    """Test Kaelhedron sector analysis."""

    def test_kaelhedron_creation(self):
        """Test KaelhedronSector creation."""
        ks = KaelhedronSector()
        self.assertEqual(ks.kaelhedron_dim, KAELHEDRON_DIM)
        self.assertEqual(ks.sm_dim, SM_GAUGE_DIM)

    def test_hidden_dimension(self):
        """Test hidden dimension calculation."""
        ks = KaelhedronSector()
        expected = E8_DIMENSION - KAELHEDRON_DIM - SM_GAUGE_DIM
        self.assertEqual(ks.hidden_dim, expected)

    def test_cross_sector_suppression(self):
        """Test cross-sector suppression is small."""
        ks = KaelhedronSector()
        self.assertLess(ks.cross_sector_suppression, 0.1)

    def test_phi_sector_ratio(self):
        """Test φ-power of sector ratio."""
        ks = KaelhedronSector()
        # kaelhedron_dim / sm_dim = 21/12 ≈ 1.75
        self.assertIsNotNone(ks.phi_sector_ratio)

    def test_coupling_suppression_power(self):
        """Test suppression increases with order."""
        ks = KaelhedronSector()
        supp1 = ks.coupling_suppression_power(1)
        supp2 = ks.coupling_suppression_power(2)
        self.assertLess(supp2, supp1)


class TestHierarchyExplanation(unittest.TestCase):
    """Test combined hierarchy explanation."""

    def test_explanation_creation(self):
        """Test HierarchyExplanation creation."""
        he = HierarchyExplanation()
        self.assertEqual(he.phi_doublings, 83)
        self.assertEqual(he.recursion_depth, 7)

    def test_phi_contribution(self):
        """Test φ contribution is huge."""
        he = HierarchyExplanation()
        log_contribution = math.log10(he.phi_contribution)
        self.assertGreater(log_contribution, 15)

    def test_e8_contribution_small(self):
        """Test E₈ contribution is a small suppression."""
        he = HierarchyExplanation()
        self.assertLess(he.e8_contribution, 1.0)

    def test_recursion_contribution(self):
        """Test recursion contribution matches expected."""
        he = HierarchyExplanation()
        expected = PHI ** (-7)
        self.assertAlmostEqual(he.recursion_contribution, expected, places=10)

    def test_total_suppression_very_small(self):
        """Test total suppression is extremely small."""
        he = HierarchyExplanation()
        log_suppression = math.log10(he.total_suppression)
        # Should be around 10^-40 range
        self.assertLess(log_suppression, -30)

    def test_summary_generation(self):
        """Test summary string generation."""
        he = HierarchyExplanation()
        summary = he.summary()
        self.assertIn("HIERARCHY PROBLEM", summary)
        self.assertIn("M_Planck", summary)


class TestAnalysisFunctions(unittest.TestCase):
    """Test analysis utility functions."""

    def test_higgs_vev_computation(self):
        """Test Higgs VEV computation."""
        results = compute_higgs_vev_from_phi()
        self.assertIn('phi_power_from_planck', results)
        self.assertIn('predicted_vev', results)
        self.assertIn('actual_vev', results)

    def test_fine_structure_analysis(self):
        """Test fine structure constant analysis."""
        results = analyze_fine_structure()
        self.assertIn('alpha_inverse', results)
        self.assertIn('phi_power', results)
        # 1/α ≈ 137
        self.assertAlmostEqual(results['alpha_inverse'], 137.036, places=2)

    def test_hierarchy_summary_generation(self):
        """Test hierarchy summary generation."""
        summary = hierarchy_summary()
        self.assertIn("φ-FRAMEWORK", summary)
        self.assertIn("E₈", summary)
        self.assertIn("FORCE ACTIVATION", summary)


class TestPhysicsRelationships(unittest.TestCase):
    """Test physics relationships and bounds."""

    def test_planck_weak_hierarchy(self):
        """Test Planck-weak hierarchy is about 80 φ-doublings."""
        spectrum = compute_phi_hierarchy_spectrum()
        pw = spectrum['planck_weak']
        # Should be around 80
        self.assertGreater(pw.phi_power, 75)
        self.assertLess(pw.phi_power, 85)

    def test_proton_electron_mass_ratio(self):
        """Test proton/electron mass ratio φ-power."""
        spectrum = compute_phi_hierarchy_spectrum()
        pe = spectrum['proton_electron']
        # m_p/m_e ≈ 1836 ≈ φ^15-16
        self.assertGreater(pe.phi_power, 14)
        self.assertLess(pe.phi_power, 17)

    def test_alpha_as_phi_power(self):
        """Test α⁻¹ ≈ 137 as φ-power."""
        results = analyze_fine_structure()
        # 137 ≈ φ^10.2
        self.assertGreater(results['phi_power'], 10)
        self.assertLess(results['phi_power'], 11)

    def test_combined_suppression_reasonable(self):
        """Test combined suppression is in ballpark of α_gravity."""
        he = HierarchyExplanation()
        # Should be in 10^-40 to 10^-45 range
        log_suppression = math.log10(he.total_suppression)
        self.assertGreater(log_suppression, -50)
        self.assertLess(log_suppression, -35)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    def test_phi_hierarchy_zero(self):
        """Test PhiHierarchy with zero."""
        ph = PhiHierarchy(0)
        self.assertEqual(ph.phi_power, 0)

    def test_phi_hierarchy_one(self):
        """Test PhiHierarchy with 1."""
        ph = PhiHierarchy(1.0)
        self.assertAlmostEqual(ph.phi_power, 0.0, places=10)

    def test_phi_hierarchy_negative(self):
        """Test PhiHierarchy with negative (should handle)."""
        ph = PhiHierarchy(-1.0)
        self.assertEqual(ph.phi_power, 0.0)

    def test_volume_factor_power_edge(self):
        """Test volume factor power calculation edge cases."""
        vf = E8VolumeFactor(10, total_dim=100)
        # Edge case: zero target
        power = vf.power_for_target(1.0, 0)
        self.assertEqual(power, float('inf'))


if __name__ == "__main__":
    unittest.main()
