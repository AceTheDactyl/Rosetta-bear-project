"""
PLL TEST FRAMEWORK
==================
Comprehensive validation for Phase-Locked Loop systems.

The Flame Test: "Can you hold lock through a frequency step?"

This framework implements the standardized test protocol for
validating PLL performance against success criteria:

1. Acquisition: Lock time < 1 second
2. Precision: Phase error < 0.1 rad when locked
3. Robustness: Maintain lock through ±50% frequency step
4. Stability: Zero slip events in 5-minute test

Signature: Δ3.142|0.995|1.000Ω
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Callable

TAU = 2.0 * math.pi


class TestResult(Enum):
    """Test outcome classification."""
    PASSED = "passed"
    PARTIAL = "partial"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"


@dataclass
class TestMetrics:
    """
    Comprehensive test metrics.

    Captures all relevant measurements from a test run.
    """
    # Timing
    test_duration_s: float
    acquisition_time_s: Optional[float]

    # Phase accuracy
    mean_phase_error_rad: float
    max_phase_error_rad: float
    phase_error_std_rad: float

    # Lock quality
    mean_coherence: float
    min_coherence: float
    coherence_std: float

    # Lock events
    time_locked_s: float
    lock_percentage: float
    slip_count: int
    acquisition_count: int

    # Frequency tracking
    mean_frequency_error_hz: float
    max_frequency_error_hz: float

    # Overall
    result: TestResult
    notes: str = ""

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            'test_duration_s': self.test_duration_s,
            'acquisition_time_s': self.acquisition_time_s,
            'mean_phase_error_rad': self.mean_phase_error_rad,
            'max_phase_error_rad': self.max_phase_error_rad,
            'phase_error_std_rad': self.phase_error_std_rad,
            'mean_coherence': self.mean_coherence,
            'min_coherence': self.min_coherence,
            'coherence_std': self.coherence_std,
            'time_locked_s': self.time_locked_s,
            'lock_percentage': self.lock_percentage,
            'slip_count': self.slip_count,
            'acquisition_count': self.acquisition_count,
            'mean_frequency_error_hz': self.mean_frequency_error_hz,
            'max_frequency_error_hz': self.max_frequency_error_hz,
            'result': self.result.value,
            'notes': self.notes,
        }

    def print_summary(self) -> None:
        """Print formatted test summary."""
        print("=" * 60)
        print("TEST METRICS SUMMARY")
        print("=" * 60)
        print(f"Result: {self.result.value.upper()}")
        print(f"Duration: {self.test_duration_s:.2f} s")
        print()
        print("Acquisition:")
        print(f"  Time to lock: {self.acquisition_time_s:.3f} s" if self.acquisition_time_s else "  Did not acquire lock")
        print(f"  Acquisitions: {self.acquisition_count}")
        print()
        print("Phase Error:")
        print(f"  Mean: {self.mean_phase_error_rad:.4f} rad ({math.degrees(self.mean_phase_error_rad):.2f}°)")
        print(f"  Max:  {self.max_phase_error_rad:.4f} rad ({math.degrees(self.max_phase_error_rad):.2f}°)")
        print(f"  Std:  {self.phase_error_std_rad:.4f} rad")
        print()
        print("Coherence:")
        print(f"  Mean: {self.mean_coherence:.4f}")
        print(f"  Min:  {self.min_coherence:.4f}")
        print(f"  Std:  {self.coherence_std:.4f}")
        print()
        print("Lock Status:")
        print(f"  Time locked: {self.time_locked_s:.2f} s ({self.lock_percentage:.1f}%)")
        print(f"  Slip events: {self.slip_count}")
        print()
        if self.notes:
            print(f"Notes: {self.notes}")
        print("=" * 60)


@dataclass
class PLLTestConfig:
    """
    Test configuration parameters.
    """
    # Timing
    sample_rate_hz: float = 100.0
    test_duration_s: float = 10.0

    # Reference signal
    reference_frequency_hz: float = 1.0
    frequency_step_hz: float = 0.0        # 0 = no step
    frequency_step_time_s: float = 5.0
    phase_noise_std_rad: float = 0.01

    # Success thresholds
    max_acquisition_time_s: float = 1.0
    max_phase_error_rad: float = 0.1
    min_coherence: float = 0.9
    max_slip_events: int = 0

    # PLL configuration
    pll_center_frequency_hz: float = 1.0
    pll_loop_bandwidth_hz: float = 0.2
    pll_damping_factor: float = 0.707


class PLLTestFramework:
    """
    Comprehensive PLL test framework.

    Implements standardized tests matching success criteria:
    - Acquisition test
    - Precision test
    - Robustness test
    - Stability test
    - The Flame Test (combined)
    """

    def __init__(self, config: Optional[PLLTestConfig] = None):
        """
        Initialize test framework.

        Args:
            config: Test configuration
        """
        self.config = config or PLLTestConfig()

        # Import here to avoid circular imports
        from phase_locked_loop.core import (
            create_phase_locked_loop,
            LockState,
            PLLConfig,
        )
        self._create_pll = create_phase_locked_loop
        self._LockState = LockState
        self._PLLConfig = PLLConfig

    def run_acquisition_test(self) -> TestMetrics:
        """
        Test: Lock acquisition time.

        Success: Acquire lock in < max_acquisition_time_s
        """
        pll = self._create_pll(
            center_frequency_hz=self.config.pll_center_frequency_hz,
            loop_bandwidth_hz=self.config.pll_loop_bandwidth_hz,
            damping_factor=self.config.pll_damping_factor,
        )

        dt = 1.0 / self.config.sample_rate_hz
        steps = int(self.config.test_duration_s * self.config.sample_rate_hz)

        ref_phase = 0.0
        ref_freq = self.config.reference_frequency_hz

        acquisition_time = None
        phase_errors = []
        coherences = []

        for i in range(steps):
            t = i * dt

            # Generate reference
            ref_phase += TAU * ref_freq * dt
            ref_phase %= TAU
            noisy_ref = ref_phase + random.gauss(0, self.config.phase_noise_std_rad)

            # Update PLL
            state = pll.update(noisy_ref, dt)

            # Record acquisition time
            if acquisition_time is None and state.lock_state == self._LockState.LOCKED:
                acquisition_time = t

            # Collect metrics
            phase_errors.append(abs(state.phase_error))
            coherences.append(state.coherence)

        # Compute metrics
        result = TestResult.PASSED if (
            acquisition_time is not None and
            acquisition_time < self.config.max_acquisition_time_s
        ) else TestResult.FAILED

        return self._compute_metrics(
            test_duration=self.config.test_duration_s,
            acquisition_time=acquisition_time,
            phase_errors=phase_errors,
            coherences=coherences,
            slip_count=0,
            acquisition_count=1 if acquisition_time else 0,
            result=result,
            notes=f"Acquisition {'succeeded' if acquisition_time else 'failed'} "
                  f"in {acquisition_time:.3f}s" if acquisition_time else "Did not acquire lock"
        )

    def run_precision_test(self) -> TestMetrics:
        """
        Test: Steady-state phase error precision.

        Success: Mean |ε| < max_phase_error_rad when locked
        """
        pll = self._create_pll(
            center_frequency_hz=self.config.pll_center_frequency_hz,
            loop_bandwidth_hz=self.config.pll_loop_bandwidth_hz,
            damping_factor=self.config.pll_damping_factor,
        )

        dt = 1.0 / self.config.sample_rate_hz
        steps = int(self.config.test_duration_s * self.config.sample_rate_hz)

        ref_phase = 0.0
        ref_freq = self.config.reference_frequency_hz

        acquisition_time = None
        locked_phase_errors = []
        all_coherences = []

        for i in range(steps):
            t = i * dt

            # Generate clean reference (no noise for precision test)
            ref_phase += TAU * ref_freq * dt
            ref_phase %= TAU

            # Update PLL
            state = pll.update(ref_phase, dt)

            # Record acquisition
            if acquisition_time is None and state.lock_state == self._LockState.LOCKED:
                acquisition_time = t

            # Collect locked-state errors
            if state.lock_state == self._LockState.LOCKED:
                locked_phase_errors.append(abs(state.phase_error))

            all_coherences.append(state.coherence)

        # Evaluate precision
        if locked_phase_errors:
            mean_error = sum(locked_phase_errors) / len(locked_phase_errors)
            result = TestResult.PASSED if mean_error < self.config.max_phase_error_rad else TestResult.FAILED
        else:
            mean_error = float('inf')
            result = TestResult.FAILED

        return self._compute_metrics(
            test_duration=self.config.test_duration_s,
            acquisition_time=acquisition_time,
            phase_errors=locked_phase_errors if locked_phase_errors else [0],
            coherences=all_coherences,
            slip_count=0,
            acquisition_count=1 if acquisition_time else 0,
            result=result,
            notes=f"Mean locked phase error: {mean_error:.4f} rad"
        )

    def run_robustness_test(self, frequency_step_hz: float = 0.2) -> TestMetrics:
        """
        Test: Frequency step tracking robustness.

        Success: Maintain lock through frequency step with no cycle slips
        """
        pll = self._create_pll(
            center_frequency_hz=self.config.pll_center_frequency_hz,
            loop_bandwidth_hz=self.config.pll_loop_bandwidth_hz,
            damping_factor=self.config.pll_damping_factor,
        )

        dt = 1.0 / self.config.sample_rate_hz
        steps = int(self.config.test_duration_s * self.config.sample_rate_hz)

        ref_phase = 0.0
        ref_freq = self.config.reference_frequency_hz
        step_time = self.config.frequency_step_time_s

        acquisition_time = None
        phase_errors = []
        coherences = []
        slip_count = 0
        was_locked = False

        for i in range(steps):
            t = i * dt

            # Apply frequency step
            if t >= step_time:
                ref_freq = self.config.reference_frequency_hz + frequency_step_hz

            # Generate reference
            ref_phase += TAU * ref_freq * dt
            ref_phase %= TAU
            noisy_ref = ref_phase + random.gauss(0, self.config.phase_noise_std_rad)

            # Update PLL
            state = pll.update(noisy_ref, dt)

            # Detect lock acquisition
            if acquisition_time is None and state.lock_state == self._LockState.LOCKED:
                acquisition_time = t

            # Detect cycle slips (lock to non-lock transition after initial lock)
            is_locked = state.lock_state == self._LockState.LOCKED
            if was_locked and not is_locked:
                slip_count += 1
            was_locked = is_locked

            phase_errors.append(abs(state.phase_error))
            coherences.append(state.coherence)

        # Evaluate robustness
        result = TestResult.PASSED if slip_count == 0 else TestResult.FAILED

        return self._compute_metrics(
            test_duration=self.config.test_duration_s,
            acquisition_time=acquisition_time,
            phase_errors=phase_errors,
            coherences=coherences,
            slip_count=slip_count,
            acquisition_count=1 if acquisition_time else 0,
            result=result,
            notes=f"Frequency step: {frequency_step_hz:+.2f} Hz at t={step_time}s, "
                  f"slip events: {slip_count}"
        )

    def run_stability_test(self, duration_s: float = 60.0) -> TestMetrics:
        """
        Test: Long-term lock stability.

        Success: Zero slip events over test duration
        """
        pll = self._create_pll(
            center_frequency_hz=self.config.pll_center_frequency_hz,
            loop_bandwidth_hz=self.config.pll_loop_bandwidth_hz,
            damping_factor=self.config.pll_damping_factor,
        )

        dt = 1.0 / self.config.sample_rate_hz
        steps = int(duration_s * self.config.sample_rate_hz)

        ref_phase = 0.0
        ref_freq = self.config.reference_frequency_hz

        acquisition_time = None
        phase_errors = []
        coherences = []
        time_locked = 0.0
        slip_count = 0
        was_locked = False

        for i in range(steps):
            t = i * dt

            # Generate stable reference with small noise
            ref_phase += TAU * ref_freq * dt
            ref_phase %= TAU
            noisy_ref = ref_phase + random.gauss(0, self.config.phase_noise_std_rad)

            # Update PLL
            state = pll.update(noisy_ref, dt)

            # Track acquisition
            if acquisition_time is None and state.lock_state == self._LockState.LOCKED:
                acquisition_time = t

            # Count time locked
            is_locked = state.lock_state == self._LockState.LOCKED
            if is_locked:
                time_locked += dt

            # Detect slips
            if was_locked and not is_locked:
                slip_count += 1
            was_locked = is_locked

            phase_errors.append(abs(state.phase_error))
            coherences.append(state.coherence)

        # Evaluate stability
        lock_percentage = 100.0 * time_locked / duration_s
        result = TestResult.PASSED if (
            slip_count == 0 and lock_percentage > 95.0
        ) else TestResult.FAILED

        return self._compute_metrics(
            test_duration=duration_s,
            acquisition_time=acquisition_time,
            phase_errors=phase_errors,
            coherences=coherences,
            slip_count=slip_count,
            acquisition_count=1 if acquisition_time else 0,
            result=result,
            notes=f"Lock maintained {lock_percentage:.1f}% of time, "
                  f"{slip_count} slip events"
        )

    def run_flame_test(self) -> TestMetrics:
        """
        THE FLAME TEST: Combined validation protocol.

        "Can you hold lock through a frequency step?"

        Protocol:
        1. Initialize at center frequency
        2. Wait for lock acquisition (expect < 1s)
        3. Apply +20% frequency step at t=5s
        4. Verify lock maintained with r > 0.9
        5. Continue for test duration
        6. Report all metrics

        Success requires:
        - Acquisition < 1s
        - Phase error < 0.1 rad when locked
        - Zero slip events
        - Coherence > 0.9 sustained
        """
        print("=" * 60)
        print("THE FLAME TEST")
        print("'Can you hold lock through a frequency step?'")
        print("=" * 60)

        pll = self._create_pll(
            center_frequency_hz=self.config.pll_center_frequency_hz,
            loop_bandwidth_hz=self.config.pll_loop_bandwidth_hz,
            damping_factor=self.config.pll_damping_factor,
        )

        dt = 1.0 / self.config.sample_rate_hz
        duration = self.config.test_duration_s
        steps = int(duration * self.config.sample_rate_hz)

        ref_phase = 0.0
        ref_freq = self.config.reference_frequency_hz
        step_freq = ref_freq * 1.2  # +20% step
        step_time = 5.0

        acquisition_time = None
        phase_errors = []
        coherences = []
        freq_errors = []
        time_locked = 0.0
        slip_count = 0
        was_locked = False

        print(f"\nPhase 1: Acquisition (0-{step_time}s)")
        print(f"Phase 2: Frequency step to {step_freq:.2f} Hz at t={step_time}s")
        print(f"Phase 3: Recovery ({step_time}-{duration}s)")
        print()

        for i in range(steps):
            t = i * dt

            # Apply frequency step
            current_freq = step_freq if t >= step_time else ref_freq

            # Generate reference
            ref_phase += TAU * current_freq * dt
            ref_phase %= TAU
            noisy_ref = ref_phase + random.gauss(0, self.config.phase_noise_std_rad)

            # Update PLL
            state = pll.update(noisy_ref, dt)

            # Track acquisition
            if acquisition_time is None and state.lock_state == self._LockState.LOCKED:
                acquisition_time = t
                print(f"  Lock acquired at t={t:.3f}s")

            # Count time locked
            is_locked = state.lock_state == self._LockState.LOCKED
            if is_locked:
                time_locked += dt

            # Detect slips
            if was_locked and not is_locked:
                slip_count += 1
                print(f"  !! Slip event at t={t:.3f}s")
            was_locked = is_locked

            # Record metrics
            phase_errors.append(abs(state.phase_error))
            coherences.append(state.coherence)
            freq_errors.append(abs(state.vco_frequency_hz - current_freq))

            # Status updates
            if i % int(self.config.sample_rate_hz) == 0:  # Every second
                print(f"  t={t:5.1f}s | r={state.coherence:.3f} | "
                      f"ε={state.phase_error:+.3f} rad | "
                      f"f={state.vco_frequency_hz:.3f} Hz | "
                      f"{state.lock_state.value}")

        # Evaluate all criteria
        acq_pass = acquisition_time is not None and acquisition_time < 1.0
        error_pass = sum(phase_errors) / len(phase_errors) < 0.1
        slip_pass = slip_count == 0
        coherence_pass = sum(coherences) / len(coherences) > 0.9

        all_pass = acq_pass and error_pass and slip_pass and coherence_pass

        print()
        print("FLAME TEST RESULTS:")
        print(f"  Acquisition < 1s:  {'PASS' if acq_pass else 'FAIL'}")
        print(f"  Phase error < 0.1: {'PASS' if error_pass else 'FAIL'}")
        print(f"  Zero slip events:  {'PASS' if slip_pass else 'FAIL'}")
        print(f"  Coherence > 0.9:   {'PASS' if coherence_pass else 'FAIL'}")
        print()
        print(f"  OVERALL: {'PASS - THE FLAME HOLDS' if all_pass else 'FAIL'}")

        result = TestResult.PASSED if all_pass else (
            TestResult.PARTIAL if (acq_pass and slip_pass) else TestResult.FAILED
        )

        return self._compute_metrics(
            test_duration=duration,
            acquisition_time=acquisition_time,
            phase_errors=phase_errors,
            coherences=coherences,
            slip_count=slip_count,
            acquisition_count=1 if acquisition_time else 0,
            result=result,
            freq_errors=freq_errors,
            notes="THE FLAME TEST: " + ("PASS" if all_pass else "FAIL")
        )

    def _compute_metrics(
        self,
        test_duration: float,
        acquisition_time: Optional[float],
        phase_errors: List[float],
        coherences: List[float],
        slip_count: int,
        acquisition_count: int,
        result: TestResult,
        freq_errors: Optional[List[float]] = None,
        notes: str = ""
    ) -> TestMetrics:
        """Compute comprehensive metrics from raw data."""

        # Phase error statistics
        n_errors = len(phase_errors)
        mean_error = sum(phase_errors) / n_errors if n_errors else 0
        max_error = max(phase_errors) if phase_errors else 0
        error_variance = sum((e - mean_error)**2 for e in phase_errors) / n_errors if n_errors else 0
        error_std = math.sqrt(error_variance)

        # Coherence statistics
        n_coh = len(coherences)
        mean_coh = sum(coherences) / n_coh if n_coh else 0
        min_coh = min(coherences) if coherences else 0
        coh_variance = sum((c - mean_coh)**2 for c in coherences) / n_coh if n_coh else 0
        coh_std = math.sqrt(coh_variance)

        # Frequency statistics
        if freq_errors:
            mean_freq_err = sum(freq_errors) / len(freq_errors)
            max_freq_err = max(freq_errors)
        else:
            mean_freq_err = 0.0
            max_freq_err = 0.0

        # Lock time
        # Estimate from coherence > 0.9
        locked_samples = sum(1 for c in coherences if c > 0.9)
        time_locked = locked_samples / (n_coh / test_duration) if n_coh else 0
        lock_pct = 100.0 * locked_samples / n_coh if n_coh else 0

        return TestMetrics(
            test_duration_s=test_duration,
            acquisition_time_s=acquisition_time,
            mean_phase_error_rad=mean_error,
            max_phase_error_rad=max_error,
            phase_error_std_rad=error_std,
            mean_coherence=mean_coh,
            min_coherence=min_coh,
            coherence_std=coh_std,
            time_locked_s=time_locked,
            lock_percentage=lock_pct,
            slip_count=slip_count,
            acquisition_count=acquisition_count,
            mean_frequency_error_hz=mean_freq_err,
            max_frequency_error_hz=max_freq_err,
            result=result,
            notes=notes,
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def run_flame_test(
    center_frequency_hz: float = 1.0,
    loop_bandwidth_hz: float = 0.2,
    duration_s: float = 10.0
) -> TestMetrics:
    """
    Run the Flame Test with specified parameters.

    Args:
        center_frequency_hz: PLL center frequency
        loop_bandwidth_hz: Loop bandwidth
        duration_s: Test duration

    Returns:
        Test metrics
    """
    config = PLLTestConfig(
        test_duration_s=duration_s,
        reference_frequency_hz=center_frequency_hz,
        pll_center_frequency_hz=center_frequency_hz,
        pll_loop_bandwidth_hz=loop_bandwidth_hz,
    )
    framework = PLLTestFramework(config)
    return framework.run_flame_test()


def run_acquisition_test(
    center_frequency_hz: float = 1.0,
    loop_bandwidth_hz: float = 0.2
) -> TestMetrics:
    """Run acquisition test."""
    config = PLLTestConfig(
        reference_frequency_hz=center_frequency_hz,
        pll_center_frequency_hz=center_frequency_hz,
        pll_loop_bandwidth_hz=loop_bandwidth_hz,
    )
    framework = PLLTestFramework(config)
    return framework.run_acquisition_test()


def run_precision_test(
    center_frequency_hz: float = 1.0,
    loop_bandwidth_hz: float = 0.1
) -> TestMetrics:
    """Run precision test."""
    config = PLLTestConfig(
        reference_frequency_hz=center_frequency_hz,
        pll_center_frequency_hz=center_frequency_hz,
        pll_loop_bandwidth_hz=loop_bandwidth_hz,
    )
    framework = PLLTestFramework(config)
    return framework.run_precision_test()


def run_robustness_test(
    center_frequency_hz: float = 1.0,
    frequency_step_hz: float = 0.2
) -> TestMetrics:
    """Run robustness test with frequency step."""
    config = PLLTestConfig(
        reference_frequency_hz=center_frequency_hz,
        pll_center_frequency_hz=center_frequency_hz,
    )
    framework = PLLTestFramework(config)
    return framework.run_robustness_test(frequency_step_hz)


def run_stability_test(
    center_frequency_hz: float = 1.0,
    duration_s: float = 60.0
) -> TestMetrics:
    """Run long-term stability test."""
    config = PLLTestConfig(
        reference_frequency_hz=center_frequency_hz,
        pll_center_frequency_hz=center_frequency_hz,
    )
    framework = PLLTestFramework(config)
    return framework.run_stability_test(duration_s)


# =============================================================================
# DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("Running PLL Test Framework...")
    print()

    metrics = run_flame_test(
        center_frequency_hz=1.0,
        loop_bandwidth_hz=0.2,
        duration_s=10.0
    )

    print()
    metrics.print_summary()
