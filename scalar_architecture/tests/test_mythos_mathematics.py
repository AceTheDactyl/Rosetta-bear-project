"""
Tests for Mythos Mathematics Module

Validates:
- MythosEquation structure
- Catalog completeness
- Verification functions
- Rosetta Stone lookups
- Mathematical-mythic equivalence

Signature: Δ|loop-closed|z0.99|rhythm-native|Ω
"""

import math
import unittest

from scalar_architecture.mythos_mathematics import (
    # Enums
    MythosCategory,

    # Core class
    MythosEquation,

    # Equation catalogs
    ERA_MATHEMATICS,
    OPERATOR_MATHEMATICS,
    POLARIC_MATHEMATICS,
    VORTEX_MATHEMATICS,
    RECURSION_MATHEMATICS,
    GEOMETRY_MATHEMATICS,
    COMPLETE_CATALOG,

    # Rosetta Stone
    MythosRosettaStone,
    ROSETTA_STONE,

    # Verification
    verify_mythos_equation,
    verify_all,

    # Utilities
    mythos_mathematics_summary,
    lookup_mythos,
    lookup_number,
)

from scalar_architecture.cet_constants import (
    PHI, PHI_INVERSE, PHI_SQUARED, PI,
    CosmologicalEra, CETOperator,
)
from scalar_architecture.hierarchy_problem import E8_DIMENSION
from scalar_architecture.polaric_duality import (
    KAELHEDRON_DIMENSIONS, LUMINAHEDRON_DIMENSIONS,
    POLARIC_SPAN, HIDDEN_DIMENSIONS,
)


class TestMythosCategory(unittest.TestCase):
    """Test MythosCategory enum."""

    def test_all_categories_exist(self):
        """Test all expected categories exist."""
        expected = ['ERA', 'TIER', 'OPERATOR', 'POLARIC', 'VORTEX', 'RECURSION', 'GEOMETRY', 'DYNAMICS']
        for cat in expected:
            self.assertIn(cat, [c.name for c in MythosCategory])

    def test_category_values(self):
        """Test category values are strings."""
        for cat in MythosCategory:
            self.assertIsInstance(cat.value, str)


class TestMythosEquation(unittest.TestCase):
    """Test MythosEquation dataclass."""

    def test_equation_creation(self):
        """Test creating a MythosEquation."""
        eq = MythosEquation(
            category=MythosCategory.RECURSION,
            name="test_equation",
            narrative_form="Test narrative",
            mathematical_form="x = 1",
            latex=r"x = 1",
        )
        self.assertEqual(eq.name, "test_equation")
        self.assertEqual(eq.category, MythosCategory.RECURSION)

    def test_equation_with_numerical_value(self):
        """Test equation with numerical value."""
        eq = MythosEquation(
            category=MythosCategory.GEOMETRY,
            name="phi_test",
            narrative_form="Golden ratio",
            mathematical_form="phi = (1+sqrt(5))/2",
            latex=r"\varphi = \frac{1+\sqrt{5}}{2}",
            numerical_value=PHI,
        )
        self.assertAlmostEqual(eq.numerical_value, 1.618033988749895, places=10)

    def test_equation_signature(self):
        """Test signature generation."""
        eq = MythosEquation(
            category=MythosCategory.ERA,
            name="test",
            narrative_form="Test",
            mathematical_form="x=1",
            latex="x=1",
            numerical_value=42.0,
        )
        sig = eq.signature
        self.assertIn("cosmological_era", sig)
        self.assertIn("test", sig)
        self.assertIn("42", sig)

    def test_equation_components(self):
        """Test equation components dictionary."""
        eq = MythosEquation(
            category=MythosCategory.POLARIC,
            name="test",
            narrative_form="Test",
            mathematical_form="x=1",
            latex="x=1",
            components={'dimension': 21, 'symbol': 'κ'}
        )
        self.assertEqual(eq.components['dimension'], 21)
        self.assertEqual(eq.components['symbol'], 'κ')


class TestEraMathematics(unittest.TestCase):
    """Test era mathematics catalog."""

    def test_all_eras_covered(self):
        """Test all cosmological eras have equations."""
        for era in CosmologicalEra:
            self.assertIn(era, ERA_MATHEMATICS)

    def test_quantum_era_equation(self):
        """Test quantum era (Dreaming Void) equation."""
        eq = ERA_MATHEMATICS[CosmologicalEra.QUANTUM_ERA]
        self.assertEqual(eq.name, "dreaming_void")
        self.assertIn("Void", eq.narrative_form)
        self.assertIn("Planck", eq.verification)

    def test_radiation_era_equation(self):
        """Test radiation era (Burning Light) equation."""
        eq = ERA_MATHEMATICS[CosmologicalEra.RADIATION_ERA]
        self.assertEqual(eq.name, "burning_light")
        self.assertIn("Light", eq.narrative_form)
        self.assertAlmostEqual(eq.numerical_value, 0.5, places=10)

    def test_matter_era_equation(self):
        """Test matter era (Gathering Darkness) equation."""
        eq = ERA_MATHEMATICS[CosmologicalEra.MATTER_ERA]
        self.assertEqual(eq.name, "gathering_darkness")
        self.assertIn("Darkness", eq.narrative_form)
        self.assertAlmostEqual(eq.numerical_value, 2/3, places=10)

    def test_acceleration_era_equation(self):
        """Test acceleration era (Awakening Storm) equation."""
        eq = ERA_MATHEMATICS[CosmologicalEra.ACCELERATION_ERA]
        self.assertEqual(eq.name, "awakening_storm")
        self.assertIn("Storm", eq.narrative_form)
        self.assertEqual(eq.numerical_value, -1)  # w = -1


class TestOperatorMathematics(unittest.TestCase):
    """Test operator mathematics catalog."""

    def test_all_operators_covered(self):
        """Test all CET operators have equations."""
        for op in CETOperator:
            self.assertIn(op, OPERATOR_MATHEMATICS)

    def test_weaver_equation(self):
        """Test Weaver (U) operator equation."""
        eq = OPERATOR_MATHEMATICS[CETOperator.U]
        self.assertEqual(eq.name, "the_weaver")
        self.assertIn("Weaver", eq.narrative_form)

    def test_separator_equation(self):
        """Test Separator (D) operator equation."""
        eq = OPERATOR_MATHEMATICS[CETOperator.D]
        self.assertEqual(eq.name, "the_separator")
        self.assertIn("Separator", eq.narrative_form)

    def test_amplifier_equation(self):
        """Test Amplifier (A) operator equation."""
        eq = OPERATOR_MATHEMATICS[CETOperator.A]
        self.assertEqual(eq.name, "the_amplifier")
        self.assertIn("Amplifier", eq.narrative_form)

    def test_anchor_equation(self):
        """Test Anchor (S) operator equation."""
        eq = OPERATOR_MATHEMATICS[CETOperator.S]
        self.assertEqual(eq.name, "the_anchor")
        self.assertIn("Anchor", eq.narrative_form)


class TestPolaricMathematics(unittest.TestCase):
    """Test polaric duality mathematics."""

    def test_kaelhedron_dimensions(self):
        """Test 21 faces of consciousness."""
        eq = POLARIC_MATHEMATICS["kaelhedron_consciousness"]
        self.assertEqual(eq.numerical_value, 21)
        self.assertEqual(eq.numerical_value, KAELHEDRON_DIMENSIONS)
        self.assertIn("21", eq.narrative_form)
        self.assertIn("consciousness", eq.narrative_form)

    def test_luminahedron_dimensions(self):
        """Test 12 faces of manifestation."""
        eq = POLARIC_MATHEMATICS["luminahedron_manifestation"]
        self.assertEqual(eq.numerical_value, 12)
        self.assertEqual(eq.numerical_value, LUMINAHEDRON_DIMENSIONS)
        self.assertIn("12", eq.narrative_form)

    def test_polaric_span(self):
        """Test 33 dimensions of becoming."""
        eq = POLARIC_MATHEMATICS["polaric_dance_33"]
        self.assertEqual(eq.numerical_value, 33)
        self.assertEqual(eq.numerical_value, POLARIC_SPAN)
        self.assertIn("33", eq.narrative_form)

    def test_hidden_sector(self):
        """Test hidden sector dimensions."""
        eq = POLARIC_MATHEMATICS["hidden_dimensions"]
        self.assertEqual(eq.numerical_value, 215)
        self.assertEqual(eq.numerical_value, HIDDEN_DIMENSIONS)

    def test_e8_decomposition(self):
        """Test E₈ = κ + λ + hidden."""
        k = POLARIC_MATHEMATICS["kaelhedron_consciousness"].numerical_value
        l = POLARIC_MATHEMATICS["luminahedron_manifestation"].numerical_value
        h = POLARIC_MATHEMATICS["hidden_dimensions"].numerical_value
        self.assertEqual(k + l + h, E8_DIMENSION)

    def test_witness_equation(self):
        """Test The Witness equation."""
        eq = POLARIC_MATHEMATICS["witness_observer"]
        self.assertIn("Witness", eq.narrative_form)
        # Mathematical form describes observation operator
        self.assertIn("O", eq.mathematical_form)

    def test_gravity_breath(self):
        """Test gravity is its breath."""
        eq = POLARIC_MATHEMATICS["gravity_breath"]
        self.assertIn("Gravity", eq.narrative_form)
        self.assertIn("breath", eq.narrative_form)
        # Should reference Einstein equations
        self.assertIn("R_μν", eq.mathematical_form)


class TestVortexMathematics(unittest.TestCase):
    """Test vortex stage mathematics."""

    def test_vortex_count(self):
        """Test all 7 vortex stages have equations."""
        self.assertEqual(len(VORTEX_MATHEMATICS), 7)

    def test_quantum_foam(self):
        """Test quantum foam stage."""
        eq = VORTEX_MATHEMATICS["quantum_foam"]
        self.assertAlmostEqual(eq.numerical_value, 0.41, places=10)
        self.assertIn("Planck", eq.narrative_form)

    def test_nucleosynthesis(self):
        """Test nucleosynthesis stage."""
        eq = VORTEX_MATHEMATICS["nucleosynthesis"]
        self.assertAlmostEqual(eq.numerical_value, 0.52, places=10)

    def test_carbon_resonance(self):
        """Test carbon resonance stage."""
        eq = VORTEX_MATHEMATICS["carbon_resonance"]
        self.assertAlmostEqual(eq.numerical_value, 0.70, places=10)
        self.assertIn("Triple-alpha", eq.narrative_form)

    def test_autocatalysis(self):
        """Test autocatalysis stage."""
        eq = VORTEX_MATHEMATICS["autocatalysis"]
        self.assertAlmostEqual(eq.numerical_value, 0.73, places=10)

    def test_phase_lock(self):
        """Test phase lock stage."""
        eq = VORTEX_MATHEMATICS["phase_lock"]
        self.assertAlmostEqual(eq.numerical_value, 0.80, places=10)
        self.assertIn("Kuramoto", eq.mathematical_form)

    def test_neural_emergence(self):
        """Test neural emergence stage."""
        eq = VORTEX_MATHEMATICS["neural_emergence"]
        self.assertAlmostEqual(eq.numerical_value, 0.85, places=10)
        self.assertIn("Consciousness", eq.narrative_form)

    def test_recursive_witness(self):
        """Test recursive witness stage."""
        eq = VORTEX_MATHEMATICS["recursive_witness"]
        self.assertAlmostEqual(eq.numerical_value, 0.87, places=10)
        self.assertIn("f(f(x))", eq.mathematical_form)


class TestRecursionMathematics(unittest.TestCase):
    """Test recursion mathematics."""

    def test_storm_remembers(self):
        """Test 'The storm that remembers the first storm'."""
        eq = RECURSION_MATHEMATICS["storm_remembers"]
        self.assertIn("storm", eq.narrative_form.lower())
        self.assertIn("remembers", eq.narrative_form.lower())
        # Should be idempotent: f(f(x)) = f(x)
        self.assertIn("f(f(x)) = f(x)", eq.mathematical_form)

    def test_recursive_spiral(self):
        """Test recursive spiral equation."""
        eq = RECURSION_MATHEMATICS["recursive_spiral"]
        self.assertIn("spiral", eq.narrative_form.lower())
        # Should have θ, z, r coordinates
        self.assertIn("theta", eq.latex.lower())

    def test_self_reference_loop(self):
        """Test self-reference loop."""
        eq = RECURSION_MATHEMATICS["self_reference_loop"]
        self.assertIn("observer", eq.narrative_form.lower())
        self.assertIn("observing", eq.narrative_form.lower())

    def test_fixed_point(self):
        """Test fixed point recognition."""
        eq = RECURSION_MATHEMATICS["fixed_point_recognition"]
        self.assertIsNotNone(eq.numerical_value)
        self.assertLess(eq.numerical_value, 1e-5)  # Should be small epsilon

    def test_watcher_watched(self):
        """Test watcher and watched equation."""
        eq = RECURSION_MATHEMATICS["watcher_watched"]
        self.assertIn("watcher", eq.narrative_form.lower())
        self.assertIn("watched", eq.narrative_form.lower())
        # Balance should be 0.5
        self.assertEqual(eq.numerical_value, 0.5)


class TestGeometryMathematics(unittest.TestCase):
    """Test geometry mathematics."""

    def test_golden_spiral(self):
        """Test golden spiral/φ equation."""
        eq = GEOMETRY_MATHEMATICS["golden_spiral"]
        self.assertAlmostEqual(eq.numerical_value, PHI, places=10)
        self.assertIn("varphi", eq.latex)  # LaTeX uses \varphi

    def test_phi_properties(self):
        """Test φ properties in equation."""
        eq = GEOMETRY_MATHEMATICS["golden_spiral"]
        comps = eq.components
        self.assertAlmostEqual(comps['value'], PHI, places=10)
        self.assertAlmostEqual(comps['squared'], PHI_SQUARED, places=10)
        self.assertAlmostEqual(comps['inverse'], PHI_INVERSE, places=10)

    def test_phi_hierarchy(self):
        """Test φ hierarchy (80 doublings)."""
        eq = GEOMETRY_MATHEMATICS["phi_hierarchy"]
        self.assertEqual(eq.numerical_value, 80)
        self.assertIn("Planck", eq.narrative_form)
        self.assertIn("Higgs", eq.narrative_form)

    def test_e8_totality(self):
        """Test E₈ = 248 equation."""
        eq = GEOMETRY_MATHEMATICS["e8_totality"]
        self.assertEqual(eq.numerical_value, 248)
        self.assertEqual(eq.numerical_value, E8_DIMENSION)

    def test_seven_domains(self):
        """Test 7 domains equation."""
        eq = GEOMETRY_MATHEMATICS["seven_domains"]
        self.assertEqual(eq.numerical_value, 7)
        origins = eq.components['origins']
        self.assertEqual(len(origins), 7)


class TestCompleteCatalog(unittest.TestCase):
    """Test complete catalog."""

    def test_catalog_not_empty(self):
        """Test catalog has entries."""
        self.assertGreater(len(COMPLETE_CATALOG), 0)

    def test_catalog_has_all_categories(self):
        """Test catalog covers all categories."""
        categories_present = set()
        for eq in COMPLETE_CATALOG.values():
            categories_present.add(eq.category)

        # Should have at least ERA, OPERATOR, POLARIC, VORTEX, RECURSION, GEOMETRY
        expected_cats = {
            MythosCategory.ERA,
            MythosCategory.OPERATOR,
            MythosCategory.POLARIC,
            MythosCategory.VORTEX,
            MythosCategory.RECURSION,
            MythosCategory.GEOMETRY,
        }
        for cat in expected_cats:
            self.assertIn(cat, categories_present)

    def test_all_equations_valid(self):
        """Test all equations have required fields."""
        for name, eq in COMPLETE_CATALOG.items():
            self.assertIsNotNone(eq.category, f"{name} missing category")
            self.assertIsNotNone(eq.name, f"{name} missing name")
            self.assertGreater(len(eq.narrative_form), 0, f"{name} empty narrative")
            self.assertGreater(len(eq.mathematical_form), 0, f"{name} empty math")
            self.assertGreater(len(eq.latex), 0, f"{name} empty latex")


class TestVerification(unittest.TestCase):
    """Test verification functions."""

    def test_verify_single_equation(self):
        """Test verifying a single equation."""
        eq = POLARIC_MATHEMATICS["kaelhedron_consciousness"]
        result = verify_mythos_equation(eq)
        self.assertTrue(result['passed'])
        self.assertTrue(result['checks']['numerical'])

    def test_verify_all(self):
        """Test verifying all equations."""
        passed, results = verify_all()
        self.assertTrue(passed, "Some equations failed verification")

    def test_verify_returns_details(self):
        """Test verification returns details."""
        eq = RECURSION_MATHEMATICS["storm_remembers"]
        result = verify_mythos_equation(eq)
        self.assertIn('name', result)
        self.assertIn('category', result)
        self.assertIn('checks', result)


class TestRosettaStone(unittest.TestCase):
    """Test Rosetta Stone translation."""

    def test_rosetta_stone_exists(self):
        """Test global Rosetta Stone instance exists."""
        self.assertIsNotNone(ROSETTA_STONE)
        self.assertIsInstance(ROSETTA_STONE, MythosRosettaStone)

    def test_myth_to_math_storm(self):
        """Test translating 'remembers' (storm that remembers)."""
        eq = ROSETTA_STONE.myth_to_math("remembers")
        self.assertIsNotNone(eq)
        self.assertIn("f(f(x))", eq.mathematical_form)

    def test_myth_to_math_witness(self):
        """Test translating 'witness'."""
        eq = ROSETTA_STONE.myth_to_math("witness")
        self.assertIsNotNone(eq)

    def test_myth_to_math_21(self):
        """Test translating '21 faces'."""
        eq = ROSETTA_STONE.myth_to_math("21 faces")
        self.assertIsNotNone(eq)
        self.assertEqual(eq.numerical_value, 21)

    def test_math_to_myth_21(self):
        """Test finding myth for number 21."""
        equations = ROSETTA_STONE.math_to_myth(21)
        self.assertGreater(len(equations), 0)
        narratives = [eq.narrative_form for eq in equations]
        self.assertTrue(any("21" in n for n in narratives))

    def test_math_to_myth_phi(self):
        """Test finding myth for φ."""
        equations = ROSETTA_STONE.math_to_myth(PHI, tolerance=0.01)
        self.assertGreater(len(equations), 0)

    def test_get_by_category(self):
        """Test getting equations by category."""
        era_eqs = ROSETTA_STONE.get_by_category(MythosCategory.ERA)
        self.assertEqual(len(era_eqs), 4)  # 4 eras

    def test_rosetta_summary(self):
        """Test Rosetta Stone summary."""
        summary = ROSETTA_STONE.summary()
        self.assertIn("MYTHOS", summary)
        self.assertIn("Total equations", summary)


class TestLookupFunctions(unittest.TestCase):
    """Test lookup utility functions."""

    def test_lookup_mythos_storm(self):
        """Test lookup_mythos for 'remembers' (storm that remembers)."""
        result = lookup_mythos("remembers")
        self.assertIsNotNone(result)
        self.assertIn("f(f(x))", result)

    def test_lookup_mythos_light(self):
        """Test lookup_mythos for burning light."""
        result = lookup_mythos("burning light")
        self.assertIsNotNone(result)

    def test_lookup_mythos_not_found(self):
        """Test lookup_mythos returns None for unknown."""
        result = lookup_mythos("xyzzy not a real phrase")
        self.assertIsNone(result)

    def test_lookup_number_21(self):
        """Test lookup_number for 21."""
        results = lookup_number(21)
        self.assertGreater(len(results), 0)
        self.assertTrue(any("21" in r for r in results))

    def test_lookup_number_33(self):
        """Test lookup_number for 33."""
        results = lookup_number(33)
        self.assertGreater(len(results), 0)
        self.assertTrue(any("33" in r for r in results))


class TestMythicMathematicalEquivalence(unittest.TestCase):
    """Test that mythic and mathematical forms are truly equivalent."""

    def test_dreaming_void_is_planck(self):
        """Test 'Dreaming Void' = Planck epoch physics."""
        eq = ERA_MATHEMATICS[CosmologicalEra.QUANTUM_ERA]
        # Numerical value should be Planck time
        self.assertLess(eq.numerical_value, 1e-43)

    def test_21_faces_is_kaelhedron_dim(self):
        """Test '21 faces' = dim(κ) = 21."""
        eq = POLARIC_MATHEMATICS["kaelhedron_consciousness"]
        self.assertEqual(eq.numerical_value, KAELHEDRON_DIMENSIONS)
        # Verify E₈ subset relationship in math form
        self.assertIn("E₈", eq.mathematical_form)

    def test_storm_is_idempotent(self):
        """Test 'storm remembers first storm' = f∘f = f."""
        eq = RECURSION_MATHEMATICS["storm_remembers"]
        # Should express idempotency
        self.assertIn("idempotent", eq.mathematical_form.lower())
        self.assertIn("f(f(x)) = f(x)", eq.mathematical_form)

    def test_33_is_polaric_span(self):
        """Test '33 dimensions' = κ + λ."""
        eq = POLARIC_MATHEMATICS["polaric_dance_33"]
        # Components should show decomposition
        comps = eq.components
        self.assertEqual(comps['kaelhedron'] + comps['luminahedron'], 33)

    def test_golden_spiral_is_phi(self):
        """Test golden spiral = φ equation."""
        eq = GEOMETRY_MATHEMATICS["golden_spiral"]
        # Should express φ² = φ + 1 relationship (using Unicode φ)
        self.assertIn("φ", eq.mathematical_form)
        self.assertIn("φ+1", eq.mathematical_form)


class TestSummaryGeneration(unittest.TestCase):
    """Test summary generation."""

    def test_mythos_mathematics_summary(self):
        """Test summary generation."""
        summary = mythos_mathematics_summary()
        self.assertIn("MYTHOS", summary)
        self.assertIn("MATHEMATICS", summary)
        # Should include the signature equation
        self.assertIn("f(f(x)) = f(x)", summary)


if __name__ == "__main__":
    unittest.main()
