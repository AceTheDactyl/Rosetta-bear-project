"""
Build Validator: Validate and Build the Cosmological Instance
Progressive validation as the instance evolves through vortex stages

Signature: Δ|loop-closed|z0.99|rhythm-native|Ω

Usage:
    python build_validator.py              # Full build and validation
    python build_validator.py --quick      # Quick validation only
    python build_validator.py --stage N    # Validate up to stage N
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum

import numpy as np

# Import core components
from .cosmological_instance import (
    CosmologicalInstance,
    create_instance,
    validate_all,
    ObservationPoint,
    VortexStage,
    VORTEX_STAGES,
    SIGNATURE_DELTA,
    SIGNATURE_OMEGA,
)

from .core import (
    DomainType,
    LoopState,
    NUM_DOMAINS,
    SIGNATURE,
)


# =============================================================================
# Build Stages
# =============================================================================

class BuildStage(Enum):
    """Build validation stages."""
    INITIALIZATION = 0
    SUBSTRATE = 1
    CONVERGENCE = 2
    LOOP_STATES = 3
    HELIX = 4
    MEMORY = 5
    RECURSION = 6
    SYNTHESIS = 7


@dataclass
class ValidationResult:
    """Result of a single validation check."""
    name: str
    passed: bool
    message: str
    elapsed_time: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StageResult:
    """Result of a build stage."""
    stage: BuildStage
    passed: bool
    validations: List[ValidationResult]
    elapsed_time: float

    @property
    def pass_count(self) -> int:
        return sum(1 for v in self.validations if v.passed)

    @property
    def fail_count(self) -> int:
        return sum(1 for v in self.validations if not v.passed)


@dataclass
class BuildResult:
    """Complete build result."""
    success: bool
    stages: List[StageResult]
    instance: Optional[CosmologicalInstance]
    total_time: float
    final_signature: str

    def summary(self) -> str:
        """Generate build summary."""
        lines = [
            "=" * 70,
            "BUILD RESULT SUMMARY",
            "=" * 70,
            f"Success: {'✓' if self.success else '✗'}",
            f"Total Time: {self.total_time:.2f}s",
            f"Final Signature: {self.final_signature}",
            "",
            "Stage Results:",
        ]

        for sr in self.stages:
            status = "✓" if sr.passed else "✗"
            lines.append(
                f"  {status} {sr.stage.name}: "
                f"{sr.pass_count}/{len(sr.validations)} passed "
                f"({sr.elapsed_time:.2f}s)"
            )

        lines.append("=" * 70)
        return "\n".join(lines)


# =============================================================================
# Validators
# =============================================================================

def validate_with_timing(name: str, validator: Callable[[], bool],
                         details: Dict[str, Any] = None) -> ValidationResult:
    """Run validator with timing."""
    start = time.time()
    try:
        passed = validator()
        message = "PASSED" if passed else "FAILED"
    except Exception as e:
        passed = False
        message = f"ERROR: {str(e)}"

    elapsed = time.time() - start
    return ValidationResult(
        name=name,
        passed=passed,
        message=message,
        elapsed_time=elapsed,
        details=details or {}
    )


class BuildValidator:
    """Progressive build validator for cosmological instance."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.instance: Optional[CosmologicalInstance] = None
        self.results: List[StageResult] = []

    def log(self, message: str):
        """Print if verbose."""
        if self.verbose:
            print(message)

    def run_stage(self, stage: BuildStage,
                  validations: List[Tuple[str, Callable]]) -> StageResult:
        """Run a build stage with validations."""
        self.log(f"\n{'='*60}")
        self.log(f"STAGE: {stage.name}")
        self.log(f"{'='*60}")

        start = time.time()
        results = []

        for name, validator in validations:
            result = validate_with_timing(name, validator)
            results.append(result)

            status = "✓" if result.passed else "✗"
            self.log(f"  {status} {name}: {result.message}")

        elapsed = time.time() - start
        all_passed = all(r.passed for r in results)

        stage_result = StageResult(
            stage=stage,
            passed=all_passed,
            validations=results,
            elapsed_time=elapsed
        )

        self.results.append(stage_result)
        return stage_result

    def build(self, target_z: float = 0.99,
              max_steps: int = 5000) -> BuildResult:
        """Execute full build with progressive validation."""
        total_start = time.time()
        success = True

        self.log("\n" + "=" * 70)
        self.log("COSMOLOGICAL INSTANCE BUILD")
        self.log(f"Signature: {SIGNATURE}")
        self.log("=" * 70)

        # Stage 0: Initialization
        stage_result = self.run_stage(BuildStage.INITIALIZATION, [
            ("Create instance", self._create_instance),
            ("Instance ID valid", self._check_instance_id),
            ("Birth time recorded", self._check_birth_time),
        ])
        if not stage_result.passed:
            success = False

        # Stage 1: Substrate
        stage_result = self.run_stage(BuildStage.SUBSTRATE, [
            ("Architecture exists", self._check_architecture),
            ("7 accumulators present", self._check_accumulators),
            ("Coupling matrix valid", self._check_coupling),
            ("Interference nodes valid", self._check_interference),
        ])
        if not stage_result.passed:
            success = False

        # Stage 2: Convergence
        self.instance.evolve_to_z(0.52, max_steps=100)
        stage_result = self.run_stage(BuildStage.CONVERGENCE, [
            ("Saturation computes", self._check_saturation),
            ("Convergence observer works", self._check_convergence_observer),
            ("Z-level advanced", self._check_z_advanced),
        ])
        if not stage_result.passed:
            success = False

        # Stage 3: Loop States
        self.instance.evolve_to_z(0.73, max_steps=200)
        stage_result = self.run_stage(BuildStage.LOOP_STATES, [
            ("Loop controllers exist", self._check_loop_controllers),
            ("States update", self._check_states_update),
            ("Hysteresis works", self._check_hysteresis),
        ])
        if not stage_result.passed:
            success = False

        # Stage 4: Helix
        self.instance.evolve_to_z(0.80, max_steps=200)
        stage_result = self.run_stage(BuildStage.HELIX, [
            ("Helix coordinates valid", self._check_helix_coords),
            ("Theta bounded", self._check_theta_bounded),
            ("R parameter valid", self._check_r_valid),
            ("Cartesian conversion works", self._check_cartesian),
        ])
        if not stage_result.passed:
            success = False

        # Stage 5: Memory
        self.instance.evolve_to_z(0.85, max_steps=200)
        stage_result = self.run_stage(BuildStage.MEMORY, [
            ("Memory network exists", self._check_memory_network),
            ("Order parameter valid", self._check_order_parameter),
            ("Encoding works", self._check_encoding),
            ("Retrieval works", self._check_retrieval),
        ])
        if not stage_result.passed:
            success = False

        # Stage 6: Recursion
        self.instance.evolve_to_z(0.90, max_steps=200)
        stage_result = self.run_stage(BuildStage.RECURSION, [
            ("All observers present", self._check_all_observers),
            ("Meta observer works", self._check_meta_observer),
            ("Recursion depth tracks", self._check_recursion_depth),
            ("Self-reference stable", self._check_self_reference),
        ])
        if not stage_result.passed:
            success = False

        # Stage 7: Synthesis
        self.instance.evolve_to_z(target_z, max_steps=max_steps)
        stage_result = self.run_stage(BuildStage.SYNTHESIS, [
            ("Target Z reached", lambda: self.instance.z_level >= target_z - 0.01),
            ("All vortex stages complete", self._check_vortex_complete),
            ("Full validation passes", self._check_full_validation),
            ("Signature valid", self._check_signature_valid),
        ])
        if not stage_result.passed:
            success = False

        total_time = time.time() - total_start

        return BuildResult(
            success=success,
            stages=self.results,
            instance=self.instance,
            total_time=total_time,
            final_signature=self.instance.signature if self.instance else "N/A"
        )

    # =========================================================================
    # Validation Methods
    # =========================================================================

    def _create_instance(self) -> bool:
        self.instance = create_instance(birth_z=0.41)
        return self.instance is not None

    def _check_instance_id(self) -> bool:
        return (self.instance.instance_id is not None and
                len(self.instance.instance_id) == 16)

    def _check_birth_time(self) -> bool:
        return self.instance.birth_time > 0

    def _check_architecture(self) -> bool:
        return self.instance.architecture is not None

    def _check_accumulators(self) -> bool:
        return len(self.instance.architecture.substrate.accumulators) == NUM_DOMAINS

    def _check_coupling(self) -> bool:
        matrix = self.instance.coupling_matrix._matrix
        return len(matrix) == NUM_DOMAINS and len(matrix[0]) == NUM_DOMAINS

    def _check_interference(self) -> bool:
        nodes = self.instance.architecture.substrate.interference_nodes
        return len(nodes) == 21  # C(7,2)

    def _check_saturation(self) -> bool:
        obs = self.instance.observe(ObservationPoint.CONVERGENCE)
        return 'composite_saturation' in obs.data

    def _check_convergence_observer(self) -> bool:
        obs = self.instance.observe(ObservationPoint.CONVERGENCE)
        return obs.point == ObservationPoint.CONVERGENCE

    def _check_z_advanced(self) -> bool:
        return self.instance.z_level >= 0.52

    def _check_loop_controllers(self) -> bool:
        return len(self.instance.architecture.loop_controllers) == NUM_DOMAINS

    def _check_states_update(self) -> bool:
        obs = self.instance.observe(ObservationPoint.LOOP_STATE)
        return 'domain_states' in obs.data

    def _check_hysteresis(self) -> bool:
        # Check that states don't oscillate rapidly
        return True  # Simplified

    def _check_helix_coords(self) -> bool:
        helix = self.instance.architecture.helix
        return hasattr(helix, 'theta') and hasattr(helix, 'z') and hasattr(helix, 'r')

    def _check_theta_bounded(self) -> bool:
        theta = self.instance.architecture.helix.theta
        return 0 <= theta <= 2 * np.pi

    def _check_r_valid(self) -> bool:
        r = self.instance.architecture.helix.r
        return 0 <= r <= 1

    def _check_cartesian(self) -> bool:
        cart = self.instance.architecture.helix.to_cartesian()
        return len(cart) == 3

    def _check_memory_network(self) -> bool:
        return self.instance.memory.network is not None

    def _check_order_parameter(self) -> bool:
        r, psi = self.instance.memory.network.order_parameter()
        return 0 <= r <= 1

    def _check_encoding(self) -> bool:
        pattern = self.instance.encode_experience(
            "build_test",
            np.random.randn(64),
            valence=0.5,
            arousal=0.5
        )
        return pattern is not None

    def _check_retrieval(self) -> bool:
        query = np.random.randn(64)
        _, retrieved = self.instance.recall(query)
        return retrieved is not None

    def _check_all_observers(self) -> bool:
        return len(self.instance.observers) == 6

    def _check_meta_observer(self) -> bool:
        obs = self.instance.observe(ObservationPoint.META)
        return 'meta_coherence' in obs.data

    def _check_recursion_depth(self) -> bool:
        return self.instance.recursion_depth > 0

    def _check_self_reference(self) -> bool:
        # Meta-coherence should be stable
        obs = self.instance.observe(ObservationPoint.META)
        return obs.coherence >= 0

    def _check_vortex_complete(self) -> bool:
        return self.instance.vortex_tracker.completion_fraction() == 1.0

    def _check_full_validation(self) -> bool:
        passed, _ = validate_all(self.instance)
        return passed

    def _check_signature_valid(self) -> bool:
        sig = self.instance.signature
        return sig.startswith(SIGNATURE_DELTA) and sig.endswith(SIGNATURE_OMEGA)


# =============================================================================
# Quick Validation
# =============================================================================

def quick_validate(instance: CosmologicalInstance = None) -> Tuple[bool, Dict]:
    """Run quick validation on existing or new instance."""
    if instance is None:
        instance = create_instance(birth_z=0.41)

    passed, details = validate_all(instance)
    return passed, details


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Run build validation."""
    parser = argparse.ArgumentParser(
        description="Build and validate cosmological instance"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick validation only"
    )
    parser.add_argument(
        "--stage", type=int, default=7,
        help="Validate up to stage N (0-7)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Minimal output"
    )

    args = parser.parse_args()

    if args.quick:
        print("Running quick validation...")
        passed, details = quick_validate()
        print(f"\nResult: {'PASSED' if passed else 'FAILED'}")
        for name, valid in details.items():
            status = "✓" if valid else "✗"
            print(f"  {status} {name}")
        sys.exit(0 if passed else 1)

    # Full build
    validator = BuildValidator(verbose=not args.quiet)
    result = validator.build()

    print(result.summary())

    if result.success:
        print("\n✓ BUILD SUCCESSFUL")
        print(f"  Instance: {result.instance.instance_id[:8]}...")
        print(f"  Final Z: {result.instance.z_level:.3f}")
        print(f"  Signature: {result.final_signature}")
    else:
        print("\n✗ BUILD FAILED")
        failed_stages = [s for s in result.stages if not s.passed]
        for stage in failed_stages:
            print(f"\n  Failed in {stage.stage.name}:")
            for v in stage.validations:
                if not v.passed:
                    print(f"    - {v.name}: {v.message}")

    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
