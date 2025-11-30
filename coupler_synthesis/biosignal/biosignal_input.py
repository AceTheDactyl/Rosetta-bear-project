"""
BIOSIGNAL INPUT MODULE
======================
Extracts phase information from biological signals for bidirectional entrainment.

Sources:
- HRV (Heart Rate Variability) via serial/bluetooth
- Keystroke dynamics (typing cadence as nervous system proxy)
- rPPG simulation (camera-based pulse detection placeholder)

Phase extraction follows the principle:
    phase = (signal_interval / typical_interval) * 2π % 2π
"""

import time
import math
import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional, Callable, List
from abc import ABC, abstractmethod

TAU = 2 * math.pi
TYPICAL_RR_INTERVAL = 800  # ms, ~75 BPM
TYPICAL_KEYSTROKE_INTERVAL = 200  # ms average typing


@dataclass
class BiosignalSample:
    """A single biosignal sample with extracted phase."""
    timestamp: float  # Unix timestamp in ms
    raw_value: float  # Original measurement (RR interval, keystroke gap, etc.)
    phase: float  # Extracted phase [0, 2π)
    source: str  # 'hrv', 'keystroke', 'rppg', 'simulated'
    confidence: float  # Quality metric [0, 1]


class BiosignalSource(ABC):
    """Abstract base class for biosignal sources."""

    @abstractmethod
    def start(self) -> None:
        """Start signal acquisition."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop signal acquisition."""
        pass

    @abstractmethod
    def get_latest_phase(self) -> Optional[float]:
        """Get the most recent phase value."""
        pass

    @abstractmethod
    def get_history(self, n: int = 10) -> List[BiosignalSample]:
        """Get the last n samples."""
        pass


class KeystrokeDynamicsAnalyzer(BiosignalSource):
    """
    Extracts phase from keystroke timing patterns.

    The inter-keystroke interval reflects nervous system state:
    - Faster typing → higher stress/arousal
    - Rhythmic patterns → stable cognitive state
    - Irregular patterns → cognitive load/distraction
    """

    def __init__(self,
                 typical_interval: float = TYPICAL_KEYSTROKE_INTERVAL,
                 buffer_size: int = 100):
        self.typical_interval = typical_interval
        self.buffer_size = buffer_size
        self.samples: deque = deque(maxlen=buffer_size)
        self.last_keystroke_time: Optional[float] = None
        self.running = False
        self._callback: Optional[Callable[[BiosignalSample], None]] = None

    def start(self) -> None:
        """Start keystroke monitoring."""
        self.running = True
        self.last_keystroke_time = None
        print("[KeystrokeDynamics] Started monitoring")

    def stop(self) -> None:
        """Stop keystroke monitoring."""
        self.running = False
        print("[KeystrokeDynamics] Stopped monitoring")

    def on_keystroke(self, timestamp_ms: Optional[float] = None) -> Optional[BiosignalSample]:
        """
        Record a keystroke event and extract phase.

        Args:
            timestamp_ms: Timestamp in milliseconds. If None, uses current time.

        Returns:
            BiosignalSample if interval could be calculated, None otherwise.
        """
        if not self.running:
            return None

        now = timestamp_ms if timestamp_ms else time.time() * 1000

        if self.last_keystroke_time is None:
            self.last_keystroke_time = now
            return None

        interval = now - self.last_keystroke_time
        self.last_keystroke_time = now

        # Skip unrealistic intervals
        if interval < 10 or interval > 5000:
            return None

        # Extract phase
        phase = (interval / self.typical_interval) * TAU % TAU

        # Calculate confidence based on interval regularity
        # Lower confidence for very fast or very slow typing
        regularity = 1.0 - abs(interval - self.typical_interval) / self.typical_interval
        confidence = max(0.0, min(1.0, regularity))

        sample = BiosignalSample(
            timestamp=now,
            raw_value=interval,
            phase=phase,
            source='keystroke',
            confidence=confidence
        )

        self.samples.append(sample)

        if self._callback:
            self._callback(sample)

        return sample

    def set_callback(self, callback: Callable[[BiosignalSample], None]) -> None:
        """Set callback for new samples."""
        self._callback = callback

    def get_latest_phase(self) -> Optional[float]:
        """Get the most recent phase value."""
        if not self.samples:
            return None
        return self.samples[-1].phase

    def get_history(self, n: int = 10) -> List[BiosignalSample]:
        """Get the last n samples."""
        return list(self.samples)[-n:]

    def get_typing_rhythm_stats(self) -> dict:
        """Calculate statistics about typing rhythm."""
        if len(self.samples) < 3:
            return {'mean_interval': 0, 'std_interval': 0, 'rhythm_score': 0}

        intervals = [s.raw_value for s in self.samples]
        mean = sum(intervals) / len(intervals)
        variance = sum((x - mean) ** 2 for x in intervals) / len(intervals)
        std = math.sqrt(variance)

        # Rhythm score: inverse of coefficient of variation
        cv = std / mean if mean > 0 else 1.0
        rhythm_score = max(0, 1 - cv)

        return {
            'mean_interval': mean,
            'std_interval': std,
            'rhythm_score': rhythm_score
        }


class HRVSimulator(BiosignalSource):
    """
    Simulates HRV data for testing without real hardware.

    Generates realistic RR intervals with:
    - Baseline heart rate
    - Respiratory sinus arrhythmia (RSA)
    - Random variation
    """

    def __init__(self,
                 base_bpm: float = 72.0,
                 rsa_amplitude: float = 50.0,  # ms
                 rsa_frequency: float = 0.25,  # Hz (typical breathing rate)
                 noise_std: float = 20.0,  # ms
                 buffer_size: int = 100):
        self.base_rr = 60000.0 / base_bpm  # Convert BPM to RR interval in ms
        self.rsa_amplitude = rsa_amplitude
        self.rsa_frequency = rsa_frequency
        self.noise_std = noise_std
        self.buffer_size = buffer_size
        self.samples: deque = deque(maxlen=buffer_size)
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._callback: Optional[Callable[[BiosignalSample], None]] = None
        self.start_time = 0

    def start(self) -> None:
        """Start HRV simulation."""
        self.running = True
        self.start_time = time.time()
        self._thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self._thread.start()
        print("[HRVSimulator] Started simulation")

    def stop(self) -> None:
        """Stop HRV simulation."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        print("[HRVSimulator] Stopped simulation")

    def _simulation_loop(self) -> None:
        """Main simulation loop."""
        import random

        last_beat = time.time() * 1000

        while self.running:
            now = time.time()
            elapsed = now - self.start_time

            # Calculate current RR interval with RSA modulation
            rsa_component = self.rsa_amplitude * math.sin(TAU * self.rsa_frequency * elapsed)
            noise = random.gauss(0, self.noise_std)
            current_rr = self.base_rr + rsa_component + noise
            current_rr = max(400, min(1500, current_rr))  # Clamp to realistic range

            # Wait for next beat
            time.sleep(current_rr / 1000.0)

            # Extract phase
            phase = (current_rr / TYPICAL_RR_INTERVAL) * TAU % TAU

            # Confidence based on how normal the interval is
            deviation = abs(current_rr - self.base_rr) / self.base_rr
            confidence = max(0.0, 1.0 - deviation)

            sample = BiosignalSample(
                timestamp=time.time() * 1000,
                raw_value=current_rr,
                phase=phase,
                source='simulated',
                confidence=confidence
            )

            self.samples.append(sample)

            if self._callback:
                self._callback(sample)

    def set_callback(self, callback: Callable[[BiosignalSample], None]) -> None:
        """Set callback for new samples."""
        self._callback = callback

    def get_latest_phase(self) -> Optional[float]:
        """Get the most recent phase value."""
        if not self.samples:
            return None
        return self.samples[-1].phase

    def get_history(self, n: int = 10) -> List[BiosignalSample]:
        """Get the last n samples."""
        return list(self.samples)[-n:]

    def set_arousal(self, arousal: float) -> None:
        """
        Adjust simulation parameters based on arousal level [0, 1].

        Higher arousal → faster heart rate, less variability.
        """
        arousal = max(0, min(1, arousal))

        # BPM ranges from 60 (relaxed) to 120 (stressed)
        target_bpm = 60 + arousal * 60
        self.base_rr = 60000.0 / target_bpm

        # Variability decreases with stress
        self.rsa_amplitude = 50 * (1 - arousal * 0.7)
        self.noise_std = 20 * (1 - arousal * 0.5)


class MultisourceBiosignalManager:
    """
    Manages multiple biosignal sources and provides unified phase output.

    Combines signals using weighted averaging based on:
    - Source reliability
    - Signal confidence
    - Recency
    """

    def __init__(self):
        self.sources: dict[str, BiosignalSource] = {}
        self.weights: dict[str, float] = {}
        self.latest_samples: dict[str, BiosignalSample] = {}
        self._unified_callback: Optional[Callable[[float, float], None]] = None

    def add_source(self, name: str, source: BiosignalSource, weight: float = 1.0) -> None:
        """Add a biosignal source with a weight."""
        self.sources[name] = source
        self.weights[name] = weight
        source.set_callback(lambda s, n=name: self._on_sample(n, s))

    def remove_source(self, name: str) -> None:
        """Remove a biosignal source."""
        if name in self.sources:
            self.sources[name].stop()
            del self.sources[name]
            del self.weights[name]
            if name in self.latest_samples:
                del self.latest_samples[name]

    def start_all(self) -> None:
        """Start all sources."""
        for source in self.sources.values():
            source.start()

    def stop_all(self) -> None:
        """Stop all sources."""
        for source in self.sources.values():
            source.stop()

    def _on_sample(self, source_name: str, sample: BiosignalSample) -> None:
        """Handle new sample from a source."""
        self.latest_samples[source_name] = sample

        # Calculate unified phase
        unified_phase, unified_confidence = self.get_unified_phase()

        if self._unified_callback and unified_phase is not None:
            self._unified_callback(unified_phase, unified_confidence)

    def set_unified_callback(self, callback: Callable[[float, float], None]) -> None:
        """Set callback for unified phase updates. Args: (phase, confidence)."""
        self._unified_callback = callback

    def get_unified_phase(self) -> tuple[Optional[float], float]:
        """
        Get weighted average phase from all sources.

        Returns:
            Tuple of (phase, confidence). Phase is None if no samples.
        """
        if not self.latest_samples:
            return None, 0.0

        now = time.time() * 1000
        max_age = 5000  # 5 second max age

        # Collect valid samples with their effective weights
        valid_phases = []
        total_weight = 0.0

        for name, sample in self.latest_samples.items():
            age = now - sample.timestamp
            if age > max_age:
                continue

            # Decay weight with age
            age_factor = 1.0 - (age / max_age)

            # Effective weight = base weight × confidence × age factor
            effective_weight = self.weights[name] * sample.confidence * age_factor

            valid_phases.append((sample.phase, effective_weight))
            total_weight += effective_weight

        if not valid_phases or total_weight == 0:
            return None, 0.0

        # Circular mean for phase averaging
        sum_sin = sum(w * math.sin(p) for p, w in valid_phases)
        sum_cos = sum(w * math.cos(p) for p, w in valid_phases)

        unified_phase = math.atan2(sum_sin / total_weight, sum_cos / total_weight)
        if unified_phase < 0:
            unified_phase += TAU

        # Confidence is normalized total weight
        max_possible_weight = sum(self.weights.values())
        unified_confidence = total_weight / max_possible_weight if max_possible_weight > 0 else 0

        return unified_phase, unified_confidence


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def hrv_to_phase(rr_interval_ms: float, typical_rr: float = TYPICAL_RR_INTERVAL) -> float:
    """Convert RR interval to phase [0, 2π)."""
    return (rr_interval_ms / typical_rr) * TAU % TAU


def keystroke_to_phase(interval_ms: float, typical: float = TYPICAL_KEYSTROKE_INTERVAL) -> float:
    """Convert inter-keystroke interval to phase [0, 2π)."""
    return (interval_ms / typical) * TAU % TAU


def phase_difference(phase1: float, phase2: float) -> float:
    """
    Calculate circular phase difference.
    Result is in range [-π, π].
    """
    diff = phase1 - phase2
    while diff > math.pi:
        diff -= TAU
    while diff < -math.pi:
        diff += TAU
    return diff


def entrainment_score(user_phase: float, system_phase: float) -> float:
    """
    Calculate entrainment score [0, 1].
    1.0 = perfectly in phase
    0.0 = completely out of phase
    """
    diff = phase_difference(user_phase, system_phase)
    return (math.cos(diff) + 1) / 2


# =============================================================================
# DEMO / TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("BIOSIGNAL INPUT MODULE - Test")
    print("=" * 60)

    # Test HRV simulator
    print("\n--- HRV Simulator Test ---")
    hrv = HRVSimulator(base_bpm=72)

    def on_hrv_sample(sample: BiosignalSample):
        print(f"  HRV: RR={sample.raw_value:.0f}ms, phase={sample.phase:.3f}, conf={sample.confidence:.2f}")

    hrv.set_callback(on_hrv_sample)
    hrv.start()
    time.sleep(5)
    hrv.stop()

    # Test keystroke dynamics
    print("\n--- Keystroke Dynamics Test ---")
    ks = KeystrokeDynamicsAnalyzer()
    ks.start()

    # Simulate keystrokes at varying intervals
    import random
    base_time = time.time() * 1000
    for i in range(10):
        interval = 150 + random.gauss(0, 50)
        base_time += interval
        sample = ks.on_keystroke(base_time)
        if sample:
            print(f"  Keystroke: interval={sample.raw_value:.0f}ms, phase={sample.phase:.3f}")

    stats = ks.get_typing_rhythm_stats()
    print(f"  Rhythm stats: mean={stats['mean_interval']:.0f}ms, rhythm_score={stats['rhythm_score']:.2f}")
    ks.stop()

    # Test multisource manager
    print("\n--- Multisource Manager Test ---")
    manager = MultisourceBiosignalManager()
    manager.add_source('hrv', HRVSimulator(base_bpm=70), weight=1.0)
    manager.add_source('keystroke', KeystrokeDynamicsAnalyzer(), weight=0.5)

    def on_unified_phase(phase: float, confidence: float):
        print(f"  Unified: phase={phase:.3f}, confidence={confidence:.2f}")

    manager.set_unified_callback(on_unified_phase)
    manager.start_all()

    # Simulate some keystrokes while HRV runs
    ks_source = manager.sources['keystroke']
    base_time = time.time() * 1000
    for i in range(5):
        time.sleep(0.3)
        base_time = time.time() * 1000
        ks_source.on_keystroke(base_time)

    time.sleep(2)
    manager.stop_all()

    print("\n[OK] Biosignal input module test complete")
