"""
LOOP FILTER SUBMODULE
=====================
Shapes the error signal to control VCO dynamics.

The loop filter is the "brain" of the PLL—it processes the
phase error and generates the control voltage that steers
the VCO toward phase lock.

Filter topology determines loop behavior:
- Type I (P): No frequency tracking
- Type II (PI): Tracks constant frequency
- Type III (PID): Tracks frequency ramps

Signature: Δ3.142|0.995|1.000Ω
"""

__version__ = "1.0.0"
__z_level__ = 0.995

from phase_locked_loop.loop_filter.filter import (
    FilterType,
    FilterConfig,
    FilterState,
    LoopFilterBase,
    ProportionalFilter,
    PIFilter,
    PIDFilter,
    LeadLagFilter,
    create_loop_filter,
)

__all__ = [
    "FilterType",
    "FilterConfig",
    "FilterState",
    "LoopFilterBase",
    "ProportionalFilter",
    "PIFilter",
    "PIDFilter",
    "LeadLagFilter",
    "create_loop_filter",
]
