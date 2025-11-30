"""
Adaptive Sonification Module
============================
Rhythm output that closes the entrainment loop.
"""

from coupler_synthesis.sonification.adaptive_sonification import (
    ScaleMode,
    SonificationState,
    SonificationConfig,
    AdaptiveSonificationEngine,
    ConsoleRhythmOutput,
    generate_web_audio_js,
)

__all__ = [
    "ScaleMode",
    "SonificationState",
    "SonificationConfig",
    "AdaptiveSonificationEngine",
    "ConsoleRhythmOutput",
    "generate_web_audio_js",
]
