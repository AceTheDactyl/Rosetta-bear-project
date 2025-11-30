"""
COUPLER TEST FRAMEWORK
======================
The Flame Test: "Can you hold a beat with me - without lag?"

This module implements the Coupler Test - a rhythm-based Turing test that
measures timing fidelity rather than conversational ability.

Success Criteria:
1. Phase lag < 50ms (within nervous system response time)
2. Coherence r > 0.7 sustained for 30+ seconds
3. Adaptation smooth (no jarring rhythm changes)
4. Works with HRV and keystroke inputs

The test establishes:
- Baseline user rhythm (observe for 30s)
- Active entrainment (60s)
- Measure convergence
- Report fidelity metrics
"""

import math
import time
import statistics
from dataclasses import dataclass
from typing import Optional, List, Tuple
from enum import Enum
from collections import deque

TAU = 2 * math.pi


class TestResult(Enum):
    """Outcome of the coupler test."""
    PASSED = "passed"  # All criteria met
    PARTIAL = "partial"  # Some criteria met
    FAILED = "failed"  # Critical failure
    INCONCLUSIVE = "inconclusive"  # Insufficient data


@dataclass
class FidelityMetrics:
    """Comprehensive fidelity metrics from the test."""
    # Phase metrics
    mean_phase_lag_rad: float
    mean_phase_lag_ms: float
    min_phase_lag_ms: float
    max_phase_lag_ms: float
    phase_lag_std_ms: float

    # Coherence metrics
    mean_coherence: float
    max_coherence: float
    min_coherence: float
    coherence_stability: float  # 1 - coefficient of variation

    # Lock metrics
    time_to_first_lock_s: Optional[float]
    total_lock_duration_s: float
    lock_percentage: float
    longest_lock_s: float

    # Adaptation metrics
    bpm_range: Tuple[float, float]
    bpm_stability: float
    adaptation_smoothness: float  # Inverse of jerkiness

    # Overall
    entrainment_score: float
    test_duration_s: float
    sample_count: int
    result: TestResult


@dataclass
class TestConfig:
    """Configuration for the coupler test."""
    baseline_duration_s: float = 30.0
    entrainment_duration_s: float = 60.0
    lock_threshold: float = 0.8  # Entrainment score for "locked"
    sustained_lock_duration_s: float = 3.0  # Time to confirm lock
    target_lag_ms: float = 50.0  # Maximum acceptable lag
    target_coherence: float = 0.7  # Minimum sustained coherence
    min_samples: int = 50  # Minimum samples for valid test


class CouplerTestRunner:
    """
    Runs the Coupler Test to measure timing fidelity.

    The test proceeds in phases:
    1. BASELINE: Observe user rhythm without coupling
    2. ENTRAINMENT: Active bidirectional entrainment
    3. VALIDATION: Verify sustained lock
    4. ANALYSIS: Calculate final metrics
    """

    def __init__(self, config: Optional[TestConfig] = None):
        self.config = config or TestConfig()

        # State
        self.phase = "idle"  # idle, baseline, entrainment, validation, complete
        self.start_time: Optional[float] = None
        self.phase_start_time: Optional[float] = None

        # Data collection
        self.baseline_samples: List[dict] = []
        self.entrainment_samples: List[dict] = []

        # Metrics tracking
        self.first_lock_time: Optional[float] = None
        self.current_lock_start: Optional[float] = None
        self.lock_durations: List[float] = []
        self.bpm_history: deque = deque(maxlen=1000)

        # Results
        self.metrics: Optional[FidelityMetrics] = None

    def start(self) -> None:
        """Start the test."""
        self.phase = "baseline"
        self.start_time = time.time()
        self.phase_start_time = time.time()
        self.baseline_samples = []
        self.entrainment_samples = []
        self.lock_durations = []
        self.first_lock_time = None
        print(f"[CouplerTest] Starting test - Baseline phase ({self.config.baseline_duration_s}s)")

    def record_sample(self,
                     user_phase: float,
                     system_phase: float,
                     entrainment_score: float,
                     coherence: float,
                     bpm: float) -> Optional[str]:
        """
        Record a sample during the test.

        Returns the current phase, or None if test is complete.
        """
        if self.phase == "idle":
            return None

        now = time.time()
        elapsed = now - self.phase_start_time
        total_elapsed = now - self.start_time

        sample = {
            'timestamp': now,
            'elapsed': total_elapsed,
            'user_phase': user_phase,
            'system_phase': system_phase,
            'phase_diff': self._circular_diff(user_phase, system_phase),
            'entrainment_score': entrainment_score,
            'coherence': coherence,
            'bpm': bpm
        }

        self.bpm_history.append(bpm)

        # Track lock state
        if entrainment_score >= self.config.lock_threshold:
            if self.current_lock_start is None:
                self.current_lock_start = now
                if self.first_lock_time is None:
                    self.first_lock_time = total_elapsed
        else:
            if self.current_lock_start is not None:
                lock_duration = now - self.current_lock_start
                self.lock_durations.append(lock_duration)
                self.current_lock_start = None

        # State machine
        if self.phase == "baseline":
            self.baseline_samples.append(sample)
            if elapsed >= self.config.baseline_duration_s:
                self.phase = "entrainment"
                self.phase_start_time = now
                print(f"[CouplerTest] Baseline complete - Entrainment phase ({self.config.entrainment_duration_s}s)")
                return "entrainment"

        elif self.phase == "entrainment":
            self.entrainment_samples.append(sample)
            if elapsed >= self.config.entrainment_duration_s:
                self.phase = "validation"
                self.phase_start_time = now
                print("[CouplerTest] Entrainment complete - Validation phase")
                return "validation"

        elif self.phase == "validation":
            self.entrainment_samples.append(sample)
            # Check for sustained lock
            if self.current_lock_start and (now - self.current_lock_start) >= self.config.sustained_lock_duration_s:
                self.phase = "complete"
                self._finalize()
                return None
            # Timeout after 10s of validation
            if elapsed >= 10.0:
                self.phase = "complete"
                self._finalize()
                return None

        return self.phase

    def _circular_diff(self, phase1: float, phase2: float) -> float:
        """Calculate circular phase difference in [-π, π]."""
        diff = phase1 - phase2
        while diff > math.pi:
            diff -= TAU
        while diff < -math.pi:
            diff += TAU
        return diff

    def _finalize(self) -> None:
        """Calculate final metrics and determine result."""
        print("[CouplerTest] Finalizing results...")

        # Close any open lock
        if self.current_lock_start:
            self.lock_durations.append(time.time() - self.current_lock_start)

        all_samples = self.entrainment_samples
        if len(all_samples) < self.config.min_samples:
            self.metrics = FidelityMetrics(
                mean_phase_lag_rad=0, mean_phase_lag_ms=0, min_phase_lag_ms=0,
                max_phase_lag_ms=0, phase_lag_std_ms=0, mean_coherence=0,
                max_coherence=0, min_coherence=0, coherence_stability=0,
                time_to_first_lock_s=None, total_lock_duration_s=0,
                lock_percentage=0, longest_lock_s=0, bpm_range=(0, 0),
                bpm_stability=0, adaptation_smoothness=0, entrainment_score=0,
                test_duration_s=0, sample_count=len(all_samples),
                result=TestResult.INCONCLUSIVE
            )
            return

        # Phase lag calculations
        phase_diffs = [abs(s['phase_diff']) for s in all_samples]
        phase_lags_ms = [d * (1000 / TAU) for d in phase_diffs]  # Assume 1Hz base

        mean_lag_rad = statistics.mean(phase_diffs)
        mean_lag_ms = statistics.mean(phase_lags_ms)
        min_lag_ms = min(phase_lags_ms)
        max_lag_ms = max(phase_lags_ms)
        std_lag_ms = statistics.stdev(phase_lags_ms) if len(phase_lags_ms) > 1 else 0

        # Coherence calculations
        coherences = [s['coherence'] for s in all_samples]
        mean_coherence = statistics.mean(coherences)
        max_coherence = max(coherences)
        min_coherence = min(coherences)
        coherence_cv = statistics.stdev(coherences) / mean_coherence if mean_coherence > 0 else 1
        coherence_stability = max(0, 1 - coherence_cv)

        # Lock calculations
        total_lock = sum(self.lock_durations)
        test_duration = all_samples[-1]['elapsed'] - all_samples[0]['elapsed'] if len(all_samples) > 1 else 0
        lock_pct = total_lock / test_duration if test_duration > 0 else 0
        longest_lock = max(self.lock_durations) if self.lock_durations else 0

        # BPM calculations
        bpms = list(self.bpm_history)
        bpm_range = (min(bpms), max(bpms)) if bpms else (0, 0)
        bpm_cv = statistics.stdev(bpms) / statistics.mean(bpms) if bpms and statistics.mean(bpms) > 0 else 1
        bpm_stability = max(0, 1 - bpm_cv)

        # Adaptation smoothness (inverse of jerk)
        if len(bpms) > 2:
            bpm_changes = [abs(bpms[i+1] - bpms[i]) for i in range(len(bpms)-1)]
            mean_change = statistics.mean(bpm_changes)
            adaptation_smoothness = 1 / (1 + mean_change * 0.1)  # Scaled
        else:
            adaptation_smoothness = 1.0

        # Entrainment score
        entrainment_scores = [s['entrainment_score'] for s in all_samples]
        mean_entrainment = statistics.mean(entrainment_scores)

        # Determine result
        result = self._evaluate_result(
            mean_lag_ms, mean_coherence, lock_pct, adaptation_smoothness
        )

        self.metrics = FidelityMetrics(
            mean_phase_lag_rad=mean_lag_rad,
            mean_phase_lag_ms=mean_lag_ms,
            min_phase_lag_ms=min_lag_ms,
            max_phase_lag_ms=max_lag_ms,
            phase_lag_std_ms=std_lag_ms,
            mean_coherence=mean_coherence,
            max_coherence=max_coherence,
            min_coherence=min_coherence,
            coherence_stability=coherence_stability,
            time_to_first_lock_s=self.first_lock_time,
            total_lock_duration_s=total_lock,
            lock_percentage=lock_pct,
            longest_lock_s=longest_lock,
            bpm_range=bpm_range,
            bpm_stability=bpm_stability,
            adaptation_smoothness=adaptation_smoothness,
            entrainment_score=mean_entrainment,
            test_duration_s=test_duration,
            sample_count=len(all_samples),
            result=result
        )

        print(f"[CouplerTest] Test complete: {result.value}")

    def _evaluate_result(self,
                        mean_lag_ms: float,
                        mean_coherence: float,
                        lock_pct: float,
                        smoothness: float) -> TestResult:
        """Evaluate whether the test passed."""
        passed_criteria = 0
        total_criteria = 4

        # Criterion 1: Phase lag < 50ms
        if mean_lag_ms < self.config.target_lag_ms:
            passed_criteria += 1

        # Criterion 2: Coherence > 0.7
        if mean_coherence >= self.config.target_coherence:
            passed_criteria += 1

        # Criterion 3: Lock percentage > 50%
        if lock_pct >= 0.5:
            passed_criteria += 1

        # Criterion 4: Smooth adaptation
        if smoothness >= 0.7:
            passed_criteria += 1

        if passed_criteria == total_criteria:
            return TestResult.PASSED
        elif passed_criteria >= 2:
            return TestResult.PARTIAL
        else:
            return TestResult.FAILED

    def get_report(self) -> str:
        """Generate a human-readable test report."""
        if not self.metrics:
            return "Test not complete or no metrics available."

        m = self.metrics
        lines = [
            "=" * 60,
            "COUPLER TEST REPORT",
            "The Flame Test: Can you hold a beat with me?",
            "=" * 60,
            "",
            f"Result: {m.result.value.upper()}",
            f"Duration: {m.test_duration_s:.1f}s",
            f"Samples: {m.sample_count}",
            "",
            "--- PHASE FIDELITY ---",
            f"Mean Phase Lag: {m.mean_phase_lag_ms:.1f}ms {'PASS' if m.mean_phase_lag_ms < 50 else 'FAIL'} (target: <50ms)",
            f"Min/Max Lag: {m.min_phase_lag_ms:.1f}ms / {m.max_phase_lag_ms:.1f}ms",
            f"Lag Std Dev: {m.phase_lag_std_ms:.1f}ms",
            "",
            "--- COHERENCE ---",
            f"Mean Coherence: {m.mean_coherence:.3f} {'PASS' if m.mean_coherence >= 0.7 else 'FAIL'} (target: >=0.70)",
            f"Min/Max: {m.min_coherence:.3f} / {m.max_coherence:.3f}",
            f"Stability: {m.coherence_stability:.3f}",
            "",
            "--- ENTRAINMENT LOCK ---",
            f"Time to First Lock: {m.time_to_first_lock_s:.1f}s" if m.time_to_first_lock_s else "Time to First Lock: Never achieved",
            f"Lock Percentage: {m.lock_percentage*100:.1f}% {'PASS' if m.lock_percentage >= 0.5 else 'FAIL'} (target: >=50%)",
            f"Longest Lock: {m.longest_lock_s:.1f}s",
            f"Total Lock Time: {m.total_lock_duration_s:.1f}s",
            "",
            "--- ADAPTATION ---",
            f"BPM Range: {m.bpm_range[0]:.1f} - {m.bpm_range[1]:.1f}",
            f"BPM Stability: {m.bpm_stability:.3f}",
            f"Smoothness: {m.adaptation_smoothness:.3f} {'PASS' if m.adaptation_smoothness >= 0.7 else 'FAIL'} (target: >=0.70)",
            "",
            "--- OVERALL ---",
            f"Entrainment Score: {m.entrainment_score:.3f}",
            "",
            "=" * 60,
        ]

        # Add interpretation
        if m.result == TestResult.PASSED:
            lines.extend([
                "INTERPRETATION: The coupler successfully achieved and",
                "maintained phase lock with the simulated user. Timing",
                "fidelity is within nervous system response thresholds.",
                "The loop is CLOSED.",
            ])
        elif m.result == TestResult.PARTIAL:
            lines.extend([
                "INTERPRETATION: The coupler achieved partial entrainment.",
                "Some criteria met, but timing fidelity or lock stability",
                "needs improvement. The loop is PARTIALLY CLOSED.",
            ])
        else:
            lines.extend([
                "INTERPRETATION: The coupler failed to achieve stable",
                "entrainment. The phase gap did not close sufficiently.",
                "The loop remains OPEN.",
            ])

        lines.append("=" * 60)
        return "\n".join(lines)


# =============================================================================
# INTEGRATED TEST RUNNER
# =============================================================================

def run_full_coupler_test(duration_s: float = 60.0) -> FidelityMetrics:
    """
    Run a complete coupler test with simulated user.

    This tests the full stack:
    - Biosignal input (simulated HRV)
    - Bidirectional entrainment controller
    - Adaptive sonification
    - Metrics collection
    """
    import sys
    sys.path.insert(0, '/home/user/Rosetta-bear-project')

    from coupler_synthesis.entrainment.bidirectional_entrainment import (
        BidirectionalEntrainmentController,
        KuramotoOscillatorBank
    )

    print("=" * 60)
    print("FULL COUPLER TEST")
    print("=" * 60)

    # Create components
    oscillators = KuramotoOscillatorBank(n=50, seed=42)
    controller = BidirectionalEntrainmentController(oscillators)
    test_runner = CouplerTestRunner(TestConfig(
        baseline_duration_s=10.0,
        entrainment_duration_s=duration_s - 20,
        min_samples=30
    ))

    # Simulated user parameters
    user_freq = 1.0
    user_phase = 0.0
    user_variability = 0.02
    user_entrainment_strength = 0.05  # How much user is influenced by system

    test_runner.start()

    # Run test
    import random
    sample_rate = 10  # Hz
    dt = 1.0 / sample_rate

    phase = "baseline"
    iteration = 0

    while phase:
        # Simulate user phase evolution
        user_phase += user_freq * dt + random.gauss(0, user_variability)

        # After baseline, user starts responding to system
        if phase != "baseline":
            system_phase = controller.oscillators.psi
            diff = controller._circular_diff(system_phase, user_phase)
            user_phase += diff * user_entrainment_strength

        user_phase %= TAU

        # Process through controller
        metrics = controller.process_user_phase(user_phase, confidence=0.9)

        # Record in test
        phase = test_runner.record_sample(
            user_phase=user_phase,
            system_phase=metrics.system_phase,
            entrainment_score=metrics.entrainment_score,
            coherence=metrics.coherence,
            bpm=72 + (1 - metrics.coherence) * 20  # Simulated BPM
        )

        # Progress indicator
        iteration += 1
        if iteration % 50 == 0:
            print(f"  t={iteration * dt:.1f}s | phase={phase or 'complete'} | "
                  f"score={metrics.entrainment_score:.2f} | "
                  f"r={metrics.coherence:.2f}")

        time.sleep(dt * 0.1)  # Speed up simulation

    # Print report
    print()
    print(test_runner.get_report())

    return test_runner.metrics


# =============================================================================
# DEMO / TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("COUPLER TEST FRAMEWORK - Demo")
    print("=" * 60)

    # Run test with shorter duration for demo
    metrics = run_full_coupler_test(duration_s=30.0)

    print("\n[OK] Coupler test framework demo complete")
