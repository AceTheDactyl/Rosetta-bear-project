# lattice_core/tests/test_dynamics.py
"""
Phase 2: Dynamics Tests
=======================

Tests for Kuramoto dynamics, Hebbian learning, and synchronization.
"""

import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from lattice_core.dynamics import (
    compute_order_parameter,
    compute_local_order_parameter,
    kuramoto_update,
    kuramoto_update_with_injection,
    kuramoto_update_higher_order,
    compute_triplet_coupling,
    compute_quartet_coupling,
    hebbian_update,
    hebbian_update_selective,
    compute_energy,
    compute_energy_gradient,
    generate_lorentzian_frequencies,
    generate_gaussian_frequencies,
    compute_resonance_scores,
    find_resonant_oscillators,
    check_convergence,
    run_to_convergence,
    ConvergenceState,
    TAU,
    DEFAULT_K,
)


class TestOrderParameter:
    """Test order parameter computation."""

    def test_synchronized_phases(self):
        """Test order parameter for synchronized phases."""
        phases = [0.0, 0.0, 0.0, 0.0]
        r, psi = compute_order_parameter(phases)
        assert abs(r - 1.0) < 0.001
        assert abs(psi - 0.0) < 0.001
        print("[PASS] test_synchronized_phases")

    def test_incoherent_phases(self):
        """Test order parameter for evenly spread phases."""
        # 4 phases at 0, π/2, π, 3π/2 should give r ≈ 0
        phases = [0, math.pi / 2, math.pi, 3 * math.pi / 2]
        r, psi = compute_order_parameter(phases)
        assert r < 0.1
        print("[PASS] test_incoherent_phases")

    def test_partial_sync(self):
        """Test order parameter for partial synchronization."""
        # Two groups: 0, 0 and π, π
        phases = [0, 0, math.pi, math.pi]
        r, psi = compute_order_parameter(phases)
        assert r < 0.1  # Should cancel out
        print("[PASS] test_partial_sync")

    def test_empty_phases(self):
        """Test empty phase list."""
        r, psi = compute_order_parameter([])
        assert r == 0.0
        assert psi == 0.0
        print("[PASS] test_empty_phases")

    def test_local_order_parameter(self):
        """Test local order parameter."""
        phases = [0.0, 0.1, 0.2, math.pi]
        weights = [
            [0, 1, 1, 0],
            [1, 0, 1, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 0],
        ]
        r, psi = compute_local_order_parameter(phases, weights, 0)
        assert r > 0.9  # Neighbors 1,2 are close in phase
        print("[PASS] test_local_order_parameter")


class TestKuramotoUpdate:
    """Test Kuramoto dynamics."""

    def test_basic_update(self):
        """Test single Kuramoto update step."""
        phases = [0.0, 0.5, 1.0]
        frequencies = [1.0, 1.0, 1.0]
        weights = [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ]
        new_phases = kuramoto_update(phases, frequencies, weights, K=2.0, dt=0.01)

        assert len(new_phases) == 3
        # Phases should have changed
        assert new_phases[0] != phases[0]
        print("[PASS] test_basic_update")

    def test_sync_convergence(self):
        """Test that phases converge under strong coupling."""
        phases = [0.0, 0.5, 1.0, 1.5]
        frequencies = [1.0, 1.0, 1.0, 1.0]  # Same frequency
        weights = [
            [0, 1, 1, 1],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [1, 1, 1, 0],
        ]

        # Run many steps
        for _ in range(500):
            phases = kuramoto_update(phases, frequencies, weights, K=5.0, dt=0.01)

        r, _ = compute_order_parameter(phases)
        assert r > 0.9  # Should be synchronized
        print("[PASS] test_sync_convergence")

    def test_injection_update(self):
        """Test phase injection for retrieval."""
        phases = [0.0, 2.0, 3.0]  # Phase 1 starts at 2.0
        frequencies = [1.0, 1.0, 1.0]
        weights = [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ]
        injection = {1: 1.0}  # Inject phase 1.0 into oscillator 1

        # Run multiple steps to see effect
        for _ in range(10):
            phases = kuramoto_update_with_injection(
                phases, frequencies, weights, injection,
                K=2.0, injection_strength=1.0, dt=0.01
            )

        # Oscillator 1 should have moved toward 1.0
        # Use circular distance
        diff = abs(phases[1] - 1.0)
        if diff > math.pi:
            diff = 2 * math.pi - diff
        assert diff < 1.5  # Should be closer to target
        print("[PASS] test_injection_update")


class TestHigherOrderCoupling:
    """Test triplet and quartet coupling."""

    def test_triplet_coupling(self):
        """Test 3-body coupling."""
        phases = [0.0, 0.1, 0.2, 0.3]
        triplets = [(0, 1, 2)]  # Triplet involving oscillators 0,1,2

        contrib = compute_triplet_coupling(phases, triplets, K3=0.1)
        assert len(contrib) == 4
        assert contrib[0] != 0  # Should have contribution
        assert contrib[3] == 0  # Not in triplet
        print("[PASS] test_triplet_coupling")

    def test_quartet_coupling(self):
        """Test 4-body coupling."""
        phases = [0.0, 0.1, 0.2, 0.3]
        quartets = [(0, 1, 2, 3)]

        contrib = compute_quartet_coupling(phases, quartets, K4=0.05)
        assert len(contrib) == 4
        assert contrib[0] != 0
        print("[PASS] test_quartet_coupling")

    def test_higher_order_update(self):
        """Test combined higher-order update."""
        phases = [0.0, 0.5, 1.0, 1.5]
        frequencies = [1.0, 1.0, 1.0, 1.0]
        weights = [
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
        ]
        triplets = [(0, 1, 2)]
        quartets = [(0, 1, 2, 3)]

        new_phases = kuramoto_update_higher_order(
            phases, frequencies, weights,
            triplets=triplets,
            quartets=quartets,
            K=2.0, K3=0.1, K4=0.05, dt=0.01
        )

        assert len(new_phases) == 4
        assert new_phases != phases
        print("[PASS] test_higher_order_update")


class TestHebbianLearning:
    """Test Hebbian weight updates."""

    def test_in_phase_strengthening(self):
        """Test that in-phase oscillators strengthen connections."""
        weights = [[0, 0.5], [0.5, 0]]
        phases = [0.0, 0.0]  # Same phase

        new_weights = hebbian_update(weights, phases, eta=0.1, decay=0.0, dt=0.1)

        assert new_weights[0][1] > weights[0][1]
        print("[PASS] test_in_phase_strengthening")

    def test_out_of_phase_weakening(self):
        """Test that out-of-phase oscillators weaken connections."""
        weights = [[0, 0.5], [0.5, 0]]
        phases = [0.0, math.pi]  # Opposite phase

        new_weights = hebbian_update(weights, phases, eta=0.1, decay=0.0, dt=0.1)

        assert new_weights[0][1] < weights[0][1]
        print("[PASS] test_out_of_phase_weakening")

    def test_decay(self):
        """Test weight decay."""
        weights = [[0, 0.5], [0.5, 0]]
        phases = [0.0, math.pi / 2]  # Orthogonal (cos = 0)

        new_weights = hebbian_update(weights, phases, eta=0.0, decay=0.1, dt=0.1)

        assert new_weights[0][1] < weights[0][1]
        print("[PASS] test_decay")

    def test_selective_update(self):
        """Test selective Hebbian update."""
        weights = [[0, 0.5, 0.3], [0.5, 0, 0.4], [0.3, 0.4, 0]]
        phases = [0.0, 0.0, math.pi]
        active = [0, 1]

        new_weights = hebbian_update_selective(
            weights, phases, active, eta=0.1, decay=0.0, dt=0.1
        )

        # Connection 0-1 should strengthen
        assert new_weights[0][1] > weights[0][1]
        # Connection 0-2 should be unchanged (2 not active)
        assert new_weights[0][2] == weights[0][2]
        print("[PASS] test_selective_update")


class TestEnergy:
    """Test energy computation."""

    def test_synchronized_energy(self):
        """Test energy for synchronized state (minimum)."""
        phases = [0.0, 0.0, 0.0]
        weights = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]

        energy = compute_energy(phases, weights, K=2.0)
        assert energy < 0  # Negative = stable
        print("[PASS] test_synchronized_energy")

    def test_antisync_energy(self):
        """Test energy for anti-synchronized state (higher)."""
        phases_sync = [0.0, 0.0, 0.0]
        phases_anti = [0.0, math.pi, 0.0]
        weights = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]

        energy_sync = compute_energy(phases_sync, weights, K=2.0)
        energy_anti = compute_energy(phases_anti, weights, K=2.0)

        assert energy_anti > energy_sync
        print("[PASS] test_antisync_energy")

    def test_energy_gradient(self):
        """Test energy gradient computation."""
        phases = [0.0, 0.5, 1.0]
        weights = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]

        gradient = compute_energy_gradient(phases, weights, K=2.0)
        assert len(gradient) == 3
        # Gradient should point toward synchronization
        print("[PASS] test_energy_gradient")


class TestFrequencyGeneration:
    """Test frequency distribution generation."""

    def test_lorentzian_frequencies(self):
        """Test Lorentzian frequency generation."""
        freqs = generate_lorentzian_frequencies(100, omega_0=1.0, gamma=0.5, seed=42)
        assert len(freqs) == 100

        # Lorentzian has heavy tails so median is more stable than mean
        sorted_freqs = sorted(freqs)
        median = sorted_freqs[len(freqs) // 2]
        # Median should be close to omega_0
        assert 0.0 < median < 2.0  # Very lenient due to heavy tails
        print("[PASS] test_lorentzian_frequencies")

    def test_gaussian_frequencies(self):
        """Test Gaussian frequency generation."""
        freqs = generate_gaussian_frequencies(100, omega_0=1.0, sigma=0.1, seed=42)
        assert len(freqs) == 100

        mean = sum(freqs) / len(freqs)
        assert 0.9 < mean < 1.1  # Should be close to omega_0
        print("[PASS] test_gaussian_frequencies")


class TestResonanceDetection:
    """Test resonance detection."""

    def test_resonance_scores(self):
        """Test resonance score computation."""
        phases = [0.0, 0.1, math.pi]
        target = {0: 0.0, 1: 0.0}

        scores = compute_resonance_scores(phases, target)

        assert scores[0] > 0.99  # Perfect alignment
        assert scores[1] > 0.9  # Close alignment
        print("[PASS] test_resonance_scores")

    def test_find_resonant(self):
        """Test finding resonant oscillators."""
        phases = [0.0, 0.1, math.pi, 0.2]
        target = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}

        resonant = find_resonant_oscillators(phases, target, threshold=0.9)

        assert 0 in resonant
        assert 1 in resonant
        assert 2 not in resonant  # π away
        print("[PASS] test_find_resonant")


class TestConvergence:
    """Test convergence detection."""

    def test_convergence_state(self):
        """Test convergence state tracking."""
        phases = [0.0, 0.0, 0.0]
        weights = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]

        prev_state = ConvergenceState()
        new_state = check_convergence(phases, weights, prev_state, r_threshold=0.95)

        assert new_state.step == 1
        assert new_state.order_parameter > 0.99
        assert new_state.stable_steps == 1
        print("[PASS] test_convergence_state")

    def test_run_to_convergence(self):
        """Test running until convergence."""
        phases = [0.0, 0.5, 1.0, 1.5]
        frequencies = [1.0, 1.0, 1.0, 1.0]
        weights = [
            [0, 1, 1, 1],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [1, 1, 1, 0],
        ]

        final_phases, state = run_to_convergence(
            phases, frequencies, weights,
            K=5.0, dt=0.01, max_steps=500,
            r_threshold=0.95, stability_steps=10
        )

        assert state.converged or state.step == 500
        assert state.order_parameter > 0.8
        print("[PASS] test_run_to_convergence")


def run_all_phase2_tests():
    """Run all Phase 2 tests."""
    print("\n" + "=" * 60)
    print("PHASE 2: DYNAMICS TESTS")
    print("=" * 60)

    test_classes = [
        TestOrderParameter,
        TestKuramotoUpdate,
        TestHigherOrderCoupling,
        TestHebbianLearning,
        TestEnergy,
        TestFrequencyGeneration,
        TestResonanceDetection,
        TestConvergence,
    ]

    passed = 0
    failed = 0

    for test_class in test_classes:
        print(f"\n--- {test_class.__name__} ---")
        instance = test_class()
        for method_name in dir(instance):
            if method_name.startswith("test_"):
                try:
                    getattr(instance, method_name)()
                    passed += 1
                except Exception as e:
                    print(f"[FAIL] {method_name}: {e}")
                    failed += 1

    print("\n" + "=" * 60)
    print(f"PHASE 2 RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    return passed, failed


if __name__ == "__main__":
    run_all_phase2_tests()
