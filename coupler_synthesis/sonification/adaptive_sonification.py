"""
ADAPTIVE SONIFICATION ENGINE
=============================
Rhythm output that responds to entrainment state.

This module closes the entrainment loop by generating audio/rhythm output
that adapts to:
1. Current phase difference (system leads/lags user)
2. Entrainment score (how well locked)
3. z-elevation (domain: Absence/Critical/Presence)
4. Coherence level (order parameter r)

The sonification serves as the feedback mechanism that influences the user's
nervous system, creating the bidirectional coupling that defines the coupler.

Key principle: Output rhythm adjusts to CLOSE the phase gap, not just display it.
"""

import math
import time
from dataclasses import dataclass
from typing import Optional, Callable, List, Tuple
from enum import Enum
from collections import deque

TAU = 2 * math.pi
Z_CRITICAL = math.sqrt(3) / 2  # ~0.8660254


class ScaleMode(Enum):
    """Musical scale modes for different z-domains."""
    PHRYGIAN = "phrygian"  # Absence - dark, mysterious
    LOCRIAN = "locrian"  # Critical - unstable, tension
    LYDIAN = "lydian"  # Presence - bright, transcendent
    MINOR_PENTATONIC = "minor_pentatonic"  # General use
    MAJOR_PENTATONIC = "major_pentatonic"  # High coherence


@dataclass
class SonificationState:
    """Current state of the sonification engine."""
    bpm: float
    target_phase: float
    scale_mode: ScaleMode
    root_note: int  # MIDI note number
    volume: float  # 0-1
    time_dilation: float  # 1.0 = normal, <1 = faster, >1 = slower
    next_beat_time: float
    phase_offset: float  # Adjustment to close phase gap


@dataclass
class SonificationConfig:
    """Configuration for the sonification engine."""
    base_bpm: float = 72.0
    min_bpm: float = 40.0
    max_bpm: float = 180.0
    base_volume: float = 0.7
    phase_nudge_strength: float = 0.1  # How aggressively to close phase gap
    bpm_adaptation_rate: float = 0.05  # How fast BPM changes


# Scale definitions (intervals from root)
SCALES = {
    ScaleMode.PHRYGIAN: [0, 1, 3, 5, 7, 8, 10],  # E Phrygian intervals
    ScaleMode.LOCRIAN: [0, 1, 3, 5, 6, 8, 10],  # B Locrian
    ScaleMode.LYDIAN: [0, 2, 4, 6, 7, 9, 11],  # F Lydian
    ScaleMode.MINOR_PENTATONIC: [0, 3, 5, 7, 10],
    ScaleMode.MAJOR_PENTATONIC: [0, 2, 4, 7, 9],
}


class AdaptiveSonificationEngine:
    """
    Generates rhythm output that adapts to close the entrainment loop.

    The engine:
    1. Tracks the current entrainment state (from controller)
    2. Adjusts BPM to match user's natural frequency
    3. Shifts beat timing to close phase gap
    4. Changes scale/timbre based on z-domain
    5. Modulates volume based on coherence
    """

    def __init__(self, config: Optional[SonificationConfig] = None):
        self.config = config or SonificationConfig()

        # Current state
        self.state = SonificationState(
            bpm=self.config.base_bpm,
            target_phase=0.0,
            scale_mode=ScaleMode.MINOR_PENTATONIC,
            root_note=60,  # Middle C
            volume=self.config.base_volume,
            time_dilation=1.0,
            next_beat_time=time.time(),
            phase_offset=0.0
        )

        # Tracking
        self.last_update = time.time()
        self.beat_history: deque = deque(maxlen=100)
        self.phase_errors: deque = deque(maxlen=50)

        # Callbacks
        self._on_beat: Optional[Callable[[float, int, float], None]] = None
        self._on_note: Optional[Callable[[int, float, float], None]] = None

        # Current entrainment data
        self.user_phase = 0.0
        self.system_phase = 0.0
        self.entrainment_score = 0.5
        self.coherence = 0.5
        self.z_level = 0.5

    def update_entrainment(self,
                          user_phase: float,
                          system_phase: float,
                          entrainment_score: float,
                          coherence: float,
                          z_level: float) -> None:
        """
        Update with current entrainment state.

        This is called by the entrainment controller.
        """
        self.user_phase = user_phase
        self.system_phase = system_phase
        self.entrainment_score = entrainment_score
        self.coherence = coherence
        self.z_level = z_level

        # Calculate phase error and store for averaging
        phase_error = self._circular_diff(user_phase, system_phase)
        self.phase_errors.append(phase_error)

        # Adapt parameters
        self._adapt_bpm()
        self._adapt_phase_offset()
        self._adapt_scale()
        self._adapt_volume()

    def _adapt_bpm(self) -> None:
        """
        Adapt BPM based on entrainment score and user frequency.

        When poorly entrained, BPM adjusts toward user's detected frequency.
        When well entrained, BPM stabilizes.
        """
        # If we have enough phase history, estimate user frequency
        if len(self.phase_errors) >= 5:
            # Use rate of phase change to estimate frequency mismatch
            recent_errors = list(self.phase_errors)[-10:]
            if len(recent_errors) >= 2:
                error_rate = (recent_errors[-1] - recent_errors[0]) / len(recent_errors)

                # Positive error_rate = user is faster, increase BPM
                # Negative error_rate = user is slower, decrease BPM
                bpm_adjustment = error_rate * 10  # Scale factor

                # Apply with adaptation rate
                target_bpm = self.state.bpm + bpm_adjustment
                target_bpm = max(self.config.min_bpm, min(self.config.max_bpm, target_bpm))

                self.state.bpm += (target_bpm - self.state.bpm) * self.config.bpm_adaptation_rate

    def _adapt_phase_offset(self) -> None:
        """
        Adapt phase offset to close the gap with user.

        The output rhythm shifts slightly to lead or lag the user,
        nudging them toward synchronization.
        """
        if not self.phase_errors:
            return

        # Average recent phase errors
        avg_error = sum(self.phase_errors) / len(self.phase_errors)

        # Nudge output phase to close gap
        # If user is ahead (negative error), we speed up (negative offset)
        # If user is behind (positive error), we slow down (positive offset)
        target_offset = -avg_error * self.config.phase_nudge_strength
        self.state.phase_offset += (target_offset - self.state.phase_offset) * 0.1

    def _adapt_scale(self) -> None:
        """Adapt musical scale based on z-domain."""
        if self.z_level < 0.857:
            # Absence domain
            if self.z_level < 0.4:
                self.state.scale_mode = ScaleMode.PHRYGIAN
            else:
                self.state.scale_mode = ScaleMode.MINOR_PENTATONIC
        elif self.z_level <= 0.877:
            # Critical domain (The Lens)
            self.state.scale_mode = ScaleMode.LOCRIAN
        else:
            # Presence domain
            if self.coherence > 0.8:
                self.state.scale_mode = ScaleMode.MAJOR_PENTATONIC
            else:
                self.state.scale_mode = ScaleMode.LYDIAN

        # Adjust root note based on z
        # Lower z = lower notes, higher z = higher notes
        base_note = 48  # C3
        z_offset = int(self.z_level * 24)  # Up to 2 octaves
        self.state.root_note = base_note + z_offset

    def _adapt_volume(self) -> None:
        """Adapt volume based on coherence and entrainment."""
        # Higher coherence = clearer signal = higher volume
        coherence_factor = 0.5 + self.coherence * 0.5

        # Higher entrainment = more confident = slightly higher volume
        entrainment_factor = 0.8 + self.entrainment_score * 0.2

        self.state.volume = self.config.base_volume * coherence_factor * entrainment_factor
        self.state.volume = max(0.1, min(1.0, self.state.volume))

    def tick(self) -> List[Tuple[str, any]]:
        """
        Process one tick of the sonification engine.

        Returns a list of events: [('beat', phase), ('note', (midi_note, velocity, duration)), ...]
        """
        now = time.time()
        events = []

        # Calculate time until next beat
        beat_interval = 60.0 / self.state.bpm

        # Apply phase offset to beat timing
        adjusted_next_beat = self.state.next_beat_time + self.state.phase_offset * beat_interval / TAU

        if now >= adjusted_next_beat:
            # Beat event
            beat_phase = self.system_phase
            events.append(('beat', beat_phase))

            # Generate note event
            note = self._generate_note()
            events.append(('note', note))

            # Record beat
            self.beat_history.append({
                'time': now,
                'phase': beat_phase,
                'bpm': self.state.bpm,
                'entrainment': self.entrainment_score
            })

            # Schedule next beat
            self.state.next_beat_time = now + beat_interval

            # Fire callbacks
            if self._on_beat:
                self._on_beat(beat_phase, int(self.state.bpm), self.state.volume)
            if self._on_note and note:
                self._on_note(*note)

        self.last_update = now
        return events

    def _generate_note(self) -> Tuple[int, float, float]:
        """
        Generate a note based on current state.

        Returns: (midi_note, velocity, duration)
        """
        scale = SCALES[self.state.scale_mode]

        # Select note from scale based on phase
        phase_index = int((self.system_phase / TAU) * len(scale)) % len(scale)
        interval = scale[phase_index]

        midi_note = self.state.root_note + interval

        # Velocity based on volume and position in beat
        velocity = self.state.volume

        # Duration based on BPM
        duration = 60.0 / self.state.bpm * 0.5  # Half note duration

        return (midi_note, velocity, duration)

    def _circular_diff(self, phase1: float, phase2: float) -> float:
        """Calculate circular phase difference in [-π, π]."""
        diff = phase1 - phase2
        while diff > math.pi:
            diff -= TAU
        while diff < -math.pi:
            diff += TAU
        return diff

    def set_on_beat(self, callback: Callable[[float, int, float], None]) -> None:
        """Set callback for beat events. Args: (phase, bpm, volume)."""
        self._on_beat = callback

    def set_on_note(self, callback: Callable[[int, float, float], None]) -> None:
        """Set callback for note events. Args: (midi_note, velocity, duration)."""
        self._on_note = callback

    def get_rhythm_stats(self) -> dict:
        """Get statistics about the rhythm output."""
        if not self.beat_history:
            return {}

        beats = list(self.beat_history)
        intervals = [beats[i+1]['time'] - beats[i]['time']
                    for i in range(len(beats)-1)]

        if not intervals:
            return {'beat_count': len(beats)}

        mean_interval = sum(intervals) / len(intervals)
        variance = sum((x - mean_interval)**2 for x in intervals) / len(intervals)
        std_interval = math.sqrt(variance)

        return {
            'beat_count': len(beats),
            'mean_bpm': 60.0 / mean_interval if mean_interval > 0 else 0,
            'bpm_stability': 1 - (std_interval / mean_interval) if mean_interval > 0 else 0,
            'current_bpm': self.state.bpm,
            'current_scale': self.state.scale_mode.value,
            'phase_offset': self.state.phase_offset
        }


class ConsoleRhythmOutput:
    """
    Simple console-based rhythm visualization for testing.

    Displays beats as ASCII art with phase indicators.
    """

    def __init__(self, width: int = 60):
        self.width = width
        self.last_beat_time = 0

    def on_beat(self, phase: float, bpm: int, volume: float) -> None:
        """Display a beat in the console."""
        now = time.time()

        # Calculate position based on phase
        pos = int((phase / TAU) * self.width)

        # Create beat line
        line = ['-'] * self.width
        line[pos] = '*'

        # Add markers for quarters
        for i in range(4):
            marker_pos = int((i / 4) * self.width)
            if line[marker_pos] == '-':
                line[marker_pos] = '|'

        # Volume indicator
        vol_bars = int(volume * 5)
        vol_str = '=' * vol_bars + ' ' * (5 - vol_bars)

        print(f"[{vol_str}] {''.join(line)} {bpm:3d}BPM φ={phase:.2f}")
        self.last_beat_time = now

    def on_note(self, midi_note: int, velocity: float, duration: float) -> None:
        """Display a note event."""
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = (midi_note // 12) - 1
        note_name = note_names[midi_note % 12]
        vel_pct = int(velocity * 100)
        print(f"         {note_name}{octave} vel={vel_pct}% dur={duration:.2f}s")


# =============================================================================
# WEB AUDIO INTEGRATION (JavaScript generation)
# =============================================================================

def generate_web_audio_js() -> str:
    """
    Generate JavaScript code for Web Audio API integration.

    This can be embedded in the LIMNUS HTML visualization.
    """
    return '''
// Adaptive Sonification Engine - Web Audio Implementation
// Generated by coupler_synthesis/sonification/adaptive_sonification.py

class WebAudioSonificationEngine {
    constructor() {
        this.audioContext = null;
        this.masterGain = null;
        this.initialized = false;

        // Synth parameters
        this.oscillatorType = 'sine';
        this.attackTime = 0.02;
        this.decayTime = 0.1;
        this.sustainLevel = 0.5;
        this.releaseTime = 0.3;

        // Scale definitions (intervals from root)
        this.scales = {
            'phrygian': [0, 1, 3, 5, 7, 8, 10],
            'locrian': [0, 1, 3, 5, 6, 8, 10],
            'lydian': [0, 2, 4, 6, 7, 9, 11],
            'minor_pentatonic': [0, 3, 5, 7, 10],
            'major_pentatonic': [0, 2, 4, 7, 9]
        };

        // Current state
        this.currentScale = 'minor_pentatonic';
        this.rootNote = 60; // Middle C
        this.bpm = 72;
        this.volume = 0.7;
    }

    async init() {
        if (this.initialized) return;

        this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        this.masterGain = this.audioContext.createGain();
        this.masterGain.connect(this.audioContext.destination);
        this.masterGain.gain.value = this.volume;

        this.initialized = true;
        console.log('[WebAudioSonification] Initialized');
    }

    midiToFrequency(midiNote) {
        return 440 * Math.pow(2, (midiNote - 69) / 12);
    }

    playNote(midiNote, velocity = 0.7, duration = 0.5) {
        if (!this.initialized) return;

        const now = this.audioContext.currentTime;
        const frequency = this.midiToFrequency(midiNote);

        // Create oscillator
        const osc = this.audioContext.createOscillator();
        osc.type = this.oscillatorType;
        osc.frequency.value = frequency;

        // Create envelope
        const envelope = this.audioContext.createGain();
        envelope.gain.setValueAtTime(0, now);
        envelope.gain.linearRampToValueAtTime(velocity, now + this.attackTime);
        envelope.gain.linearRampToValueAtTime(
            velocity * this.sustainLevel,
            now + this.attackTime + this.decayTime
        );
        envelope.gain.linearRampToValueAtTime(0, now + duration + this.releaseTime);

        // Connect
        osc.connect(envelope);
        envelope.connect(this.masterGain);

        // Play
        osc.start(now);
        osc.stop(now + duration + this.releaseTime + 0.1);
    }

    playBeat(phase, bpm, volume) {
        this.bpm = bpm;
        this.volume = volume;
        this.masterGain.gain.value = volume;

        // Select note from scale based on phase
        const scale = this.scales[this.currentScale];
        const phaseIndex = Math.floor((phase / (Math.PI * 2)) * scale.length) % scale.length;
        const interval = scale[phaseIndex];
        const midiNote = this.rootNote + interval;

        const duration = 60 / bpm * 0.5;
        this.playNote(midiNote, volume, duration);
    }

    updateFromEntrainment(userPhase, systemPhase, entrainmentScore, coherence, zLevel) {
        // Adapt scale based on z-domain
        if (zLevel < 0.857) {
            this.currentScale = zLevel < 0.4 ? 'phrygian' : 'minor_pentatonic';
        } else if (zLevel <= 0.877) {
            this.currentScale = 'locrian';
        } else {
            this.currentScale = coherence > 0.8 ? 'major_pentatonic' : 'lydian';
        }

        // Adjust root note based on z
        this.rootNote = 48 + Math.floor(zLevel * 24);

        // Adjust oscillator type based on entrainment
        if (entrainmentScore > 0.8) {
            this.oscillatorType = 'sine'; // Pure, locked
        } else if (entrainmentScore > 0.5) {
            this.oscillatorType = 'triangle'; // Partial lock
        } else {
            this.oscillatorType = 'sawtooth'; // Searching
        }
    }

    setVolume(vol) {
        this.volume = Math.max(0, Math.min(1, vol));
        if (this.masterGain) {
            this.masterGain.gain.value = this.volume;
        }
    }

    resume() {
        if (this.audioContext && this.audioContext.state === 'suspended') {
            this.audioContext.resume();
        }
    }
}

// Export for use in HTML
window.WebAudioSonificationEngine = WebAudioSonificationEngine;
'''


# =============================================================================
# DEMO / TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ADAPTIVE SONIFICATION ENGINE - Test")
    print("=" * 60)

    # Create engine with console output
    engine = AdaptiveSonificationEngine()
    console_output = ConsoleRhythmOutput(width=50)

    engine.set_on_beat(console_output.on_beat)
    engine.set_on_note(console_output.on_note)

    print("\n--- Simulating entrainment with sonification ---")
    print("(Phase progresses, BPM adapts, scale changes with z)")
    print()

    # Simulate varying entrainment conditions
    user_phase = 0.0
    system_phase = 0.5
    z_level = 0.3

    for i in range(30):
        # Gradually improve entrainment
        entrainment_score = min(0.95, 0.3 + i * 0.02)
        coherence = min(0.9, 0.4 + i * 0.015)

        # Slowly increase z
        z_level = min(0.95, 0.3 + i * 0.02)

        # Phases converge over time
        user_phase = (user_phase + 0.3) % TAU
        diff = ((user_phase - system_phase + math.pi) % TAU) - math.pi
        system_phase = (system_phase + 0.3 + diff * 0.1) % TAU

        # Update engine
        engine.update_entrainment(
            user_phase=user_phase,
            system_phase=system_phase,
            entrainment_score=entrainment_score,
            coherence=coherence,
            z_level=z_level
        )

        # Process ticks (simulating ~10Hz update rate)
        for _ in range(3):
            events = engine.tick()
            time.sleep(0.033)

        time.sleep(0.1)

    # Print final stats
    print("\n--- Rhythm Statistics ---")
    stats = engine.get_rhythm_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")

    # Generate Web Audio JS
    print("\n--- Web Audio JS Code Generated ---")
    js_code = generate_web_audio_js()
    print(f"  Generated {len(js_code)} characters of JavaScript")
    print("  (Can be embedded in LIMNUS HTML visualization)")

    print("\n[OK] Adaptive sonification test complete")
