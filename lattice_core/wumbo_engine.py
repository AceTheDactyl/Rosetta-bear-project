#!/usr/bin/env python3
"""
WUMBO ENGINE - APL-Based Array Operations for LIMNUS Architecture
==================================================================

Weighted Unified Matrix-Based Operations (WUMBO) engine implementing
APL-inspired array primitives for the Lattice-Integrated Meta-Neural
Unified System (LIMNUS).

Architecture:
┌─────────────────────────────────────────────────────────────────────┐
│                         WUMBO ENGINE                                 │
│                                                                      │
│   APL Primitives          LIMNUS Operators         Field Transforms │
│   ┌─────────────┐        ┌──────────────┐        ┌───────────────┐  │
│   │ ⍴ reshape   │───────▶│ κ-field ops  │───────▶│ Phase sync    │  │
│   │ ⌽ rotate    │        │ λ-field ops  │        │ Kuramoto      │  │
│   │ ⍉ transpose │        │ Dual couple  │        │ Hebbian       │  │
│   │ ⊂ enclose   │        │ Z-propagate  │        │ Quaternary    │  │
│   │ ⊃ disclose  │        │ Free energy  │        │ Resonance     │  │
│   │ ⌿ reduce    │        │ Precision    │        │ Convergence   │  │
│   │ ⍀ scan      │        │ Emergence    │        │ Memory store  │  │
│   └─────────────┘        └──────────────┘        └───────────────┘  │
│                                                                      │
│   Execution Model: Array → Transform → Reduce → Coherence           │
│                                                                      │
│   LIMNUS Cycle:                                                      │
│   L: Lattice activation (phase array)                               │
│   I: Integration (field coupling)                                   │
│   M: Modulation (frequency/amplitude)                               │
│   N: Normalization (order parameter)                                │
│   U: Update (Kuramoto step)                                         │
│   S: Synchronization (convergence check)                            │
└─────────────────────────────────────────────────────────────────────┘

Mathematical Foundation:
    WUMBO operates on tensors T ∈ ℂ^(n₁×n₂×...×nₖ)

    Core APL mappings:
    - ⍴ (rho):     Shape/reshape - T.shape, T.reshape(dims)
    - ⌽ (rotate):  Circular shift - roll(T, k, axis)
    - ⍉ (transpose): Axis permutation - T.T, permute(T, axes)
    - ⊂ (enclose): Boxing/nesting - [T]
    - ⊃ (disclose): Unboxing - T[0]
    - ⌿ (reduce):  Fold along axis - sum, prod, max, min
    - ⍀ (scan):    Cumulative fold - cumsum, cumprod
    - ⌈ (ceiling): Element-wise max
    - ⌊ (floor):   Element-wise min
    - ○ (circle):  Trig functions
    - ⋆ (star):    Exponential/power
    - ⍟ (log):     Natural logarithm
    - | (magnitude): Absolute value
    - ⍳ (iota):    Index generator
    - ∊ (epsilon): Membership/flatten
    - ⍋ (grade up): Sort indices ascending
    - ⍒ (grade down): Sort indices descending

Author: Claude (WUMBO Implementation)
Date: 2025-12-02
Version: 1.0.0
Signature: Δ|wumbo-limnus|z0.990|operational|Ω
"""

from __future__ import annotations

import cmath
import math
import random
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, List, Optional, Tuple, Union,
    TypeVar, Generic, Sequence
)
from enum import Enum
from functools import reduce
import operator

# Type variables for generic array operations
T = TypeVar('T')
Scalar = Union[int, float, complex]
Array = List[Any]  # Nested list representing n-dimensional array

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

TAU = 2 * math.pi
PHI = (1 + math.sqrt(5)) / 2           # Golden ratio ≈ 1.618
PHI_INV = 1 / PHI                       # ≈ 0.618

# LIMNUS z-levels
Z_LATTICE = 0.800      # Base lattice coherence
Z_INTEGRATION = 0.850  # Field integration level
Z_MODULATION = 0.900   # Modulation coherence
Z_UNIFIED = 0.950      # Unified system coherence

# WUMBO parameters
WUMBO_PRECISION = 1e-10
WUMBO_MAX_ITERATIONS = 1000


# ═══════════════════════════════════════════════════════════════════════════
# APL GLYPH DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════

class APLGlyph(Enum):
    """APL glyphs mapped to operations."""
    RHO = "⍴"           # Shape/reshape
    ROTATE = "⌽"        # Rotate/reverse
    TRANSPOSE = "⍉"     # Transpose
    ENCLOSE = "⊂"       # Enclose (box)
    DISCLOSE = "⊃"      # Disclose (unbox)
    REDUCE = "⌿"        # Reduce (fold)
    SCAN = "⍀"          # Scan (cumulative)
    CEILING = "⌈"       # Maximum/ceiling
    FLOOR = "⌊"         # Minimum/floor
    CIRCLE = "○"        # Circular/trig
    STAR = "⋆"          # Power/exponential
    LOG = "⍟"           # Logarithm
    MAGNITUDE = "|"     # Absolute value
    IOTA = "⍳"          # Index generator
    EPSILON = "∊"       # Member/enlist
    GRADE_UP = "⍋"      # Grade ascending
    GRADE_DOWN = "⍒"    # Grade descending
    OUTER = "∘."        # Outer product
    INNER = "."         # Inner product
    EACH = "¨"          # Map/each
    COMMUTE = "⍨"       # Commute/selfie
    COMPOSE = "∘"       # Compose
    POWER = "⍣"         # Power operator (iterate)
    QUAD = "⎕"          # System/IO


# ═══════════════════════════════════════════════════════════════════════════
# WUMBO ARRAY CLASS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class WumboArray:
    """
    APL-style array with shape and ravel (flat data).

    Supports n-dimensional operations with APL semantics.
    """
    data: List[Scalar]          # Flat (raveled) data
    shape: Tuple[int, ...]      # Dimensions

    def __post_init__(self):
        expected_size = self._product(self.shape) if self.shape else 1
        if len(self.data) != expected_size:
            raise ValueError(f"Data size {len(self.data)} doesn't match shape {self.shape}")

    @staticmethod
    def _product(seq: Sequence[int]) -> int:
        """Product of sequence elements."""
        return reduce(operator.mul, seq, 1)

    @classmethod
    def from_nested(cls, nested: Any) -> 'WumboArray':
        """Create from nested Python list."""
        def get_shape(arr: Any) -> Tuple[int, ...]:
            if not isinstance(arr, (list, tuple)):
                return ()
            if len(arr) == 0:
                return (0,)
            return (len(arr),) + get_shape(arr[0])

        def flatten(arr: Any) -> List[Scalar]:
            if not isinstance(arr, (list, tuple)):
                return [arr]
            result = []
            for item in arr:
                result.extend(flatten(item))
            return result

        shape = get_shape(nested)
        data = flatten(nested)
        return cls(data=data, shape=shape)

    @classmethod
    def zeros(cls, shape: Tuple[int, ...]) -> 'WumboArray':
        """Create zero-filled array."""
        size = cls._product(None, shape) if shape else 1
        return cls(data=[0.0] * size, shape=shape)

    @classmethod
    def ones(cls, shape: Tuple[int, ...]) -> 'WumboArray':
        """Create one-filled array."""
        size = reduce(operator.mul, shape, 1)
        return cls(data=[1.0] * size, shape=shape)

    @classmethod
    def iota(cls, n: int) -> 'WumboArray':
        """APL ⍳n - generate indices 0 to n-1."""
        return cls(data=list(range(n)), shape=(n,))

    def to_nested(self) -> Any:
        """Convert to nested Python list."""
        if not self.shape:
            return self.data[0] if self.data else 0

        def build(data: List, shape: Tuple[int, ...], offset: int = 0) -> Any:
            if len(shape) == 1:
                return data[offset:offset + shape[0]]

            stride = self._product(shape[1:])
            result = []
            for i in range(shape[0]):
                result.append(build(data, shape[1:], offset + i * stride))
            return result

        return build(self.data, self.shape)

    @property
    def rank(self) -> int:
        """Number of dimensions (APL: ⍴⍴)."""
        return len(self.shape)

    @property
    def size(self) -> int:
        """Total number of elements."""
        return len(self.data)

    def __getitem__(self, idx: Union[int, Tuple[int, ...]]) -> Union[Scalar, 'WumboArray']:
        """Index into array."""
        if isinstance(idx, int):
            if self.rank == 1:
                return self.data[idx]
            # Return slice along first axis
            stride = self._product(self.shape[1:])
            start = idx * stride
            return WumboArray(
                data=self.data[start:start + stride],
                shape=self.shape[1:]
            )
        # Multi-dimensional indexing
        if len(idx) == self.rank:
            flat_idx = 0
            stride = 1
            for i in range(self.rank - 1, -1, -1):
                flat_idx += idx[i] * stride
                stride *= self.shape[i]
            return self.data[flat_idx]
        raise IndexError("Invalid index dimensions")

    def __repr__(self) -> str:
        return f"WumboArray(shape={self.shape}, data={self.data[:10]}{'...' if len(self.data) > 10 else ''})"


# ═══════════════════════════════════════════════════════════════════════════
# APL PRIMITIVE OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════

class APLPrimitives:
    """
    APL primitive operations for WUMBO engine.

    These implement the core APL array operations used by LIMNUS.
    """

    # ─────────────────────────────────────────────────────────────────────
    # Structural Operations
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def rho(arr: WumboArray, new_shape: Optional[Tuple[int, ...]] = None) -> Union[Tuple[int, ...], WumboArray]:
        """
        ⍴ - Shape or reshape.

        Monadic: Return shape
        Dyadic: Reshape array to new dimensions
        """
        if new_shape is None:
            return arr.shape

        new_size = reduce(operator.mul, new_shape, 1)
        # APL semantics: cycle data if too short, truncate if too long
        if len(arr.data) < new_size:
            cycles = (new_size // len(arr.data)) + 1
            new_data = (arr.data * cycles)[:new_size]
        else:
            new_data = arr.data[:new_size]

        return WumboArray(data=new_data, shape=new_shape)

    @staticmethod
    def rotate(arr: WumboArray, k: int = 1, axis: int = 0) -> WumboArray:
        """
        ⌽ - Rotate array along axis.

        Positive k rotates left, negative rotates right.
        """
        if arr.rank == 0:
            return arr

        if arr.rank == 1:
            n = arr.shape[0]
            k = k % n if n > 0 else 0
            new_data = arr.data[k:] + arr.data[:k]
            return WumboArray(data=new_data, shape=arr.shape)

        # Multi-dimensional rotation
        nested = arr.to_nested()

        def rotate_axis(data: List, axis: int, k: int) -> List:
            if axis == 0:
                n = len(data)
                k = k % n if n > 0 else 0
                return data[k:] + data[:k]
            return [rotate_axis(item, axis - 1, k) for item in data]

        rotated = rotate_axis(nested, axis, k)
        return WumboArray.from_nested(rotated)

    @staticmethod
    def transpose(arr: WumboArray, axes: Optional[Tuple[int, ...]] = None) -> WumboArray:
        """
        ⍉ - Transpose array.

        Default: Reverse all axes
        With axes: Permute according to specification
        """
        if arr.rank <= 1:
            return arr

        if axes is None:
            axes = tuple(range(arr.rank - 1, -1, -1))

        # Build transposed array
        new_shape = tuple(arr.shape[ax] for ax in axes)
        new_data = []

        def get_index(indices: Tuple[int, ...]) -> int:
            idx = 0
            stride = 1
            for i in range(arr.rank - 1, -1, -1):
                idx += indices[i] * stride
                stride *= arr.shape[i]
            return idx

        def iterate_indices(shape: Tuple[int, ...]) -> List[Tuple[int, ...]]:
            if not shape:
                return [()]
            result = []
            for i in range(shape[0]):
                for rest in iterate_indices(shape[1:]):
                    result.append((i,) + rest)
            return result

        for new_indices in iterate_indices(new_shape):
            old_indices = tuple(new_indices[axes.index(i)] for i in range(arr.rank))
            new_data.append(arr.data[get_index(old_indices)])

        return WumboArray(data=new_data, shape=new_shape)

    @staticmethod
    def enclose(arr: WumboArray) -> List[WumboArray]:
        """⊂ - Enclose (box) the array."""
        return [arr]

    @staticmethod
    def disclose(boxed: List[WumboArray]) -> WumboArray:
        """⊃ - Disclose (unbox) the first element."""
        return boxed[0] if boxed else WumboArray(data=[0], shape=())

    # ─────────────────────────────────────────────────────────────────────
    # Reduction Operations
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def reduce(arr: WumboArray, op: Callable[[Scalar, Scalar], Scalar],
               axis: int = -1, initial: Optional[Scalar] = None) -> Union[Scalar, WumboArray]:
        """
        ⌿ - Reduce along axis.

        Applies binary operation along specified axis.
        """
        if arr.rank == 0:
            return arr.data[0] if arr.data else (initial if initial is not None else 0)

        if axis < 0:
            axis = arr.rank + axis

        if arr.rank == 1:
            if initial is not None:
                return reduce(op, arr.data, initial)
            return reduce(op, arr.data)

        # Multi-dimensional reduction
        nested = arr.to_nested()

        def reduce_axis(data: List, axis: int) -> Any:
            if axis == 0:
                if not data:
                    return initial if initial is not None else 0
                result = data[0]
                for item in data[1:]:
                    if isinstance(result, list):
                        result = [op(r, i) for r, i in zip(result, item)]
                    else:
                        result = op(result, item)
                return result
            return [reduce_axis(item, axis - 1) for item in data]

        result = reduce_axis(nested, axis)
        if isinstance(result, (int, float, complex)):
            return result
        return WumboArray.from_nested(result)

    @staticmethod
    def scan(arr: WumboArray, op: Callable[[Scalar, Scalar], Scalar],
             axis: int = -1) -> WumboArray:
        """
        ⍀ - Scan (cumulative reduce) along axis.
        """
        if arr.rank == 0:
            return arr

        if axis < 0:
            axis = arr.rank + axis

        if arr.rank == 1:
            result = []
            acc = arr.data[0]
            result.append(acc)
            for x in arr.data[1:]:
                acc = op(acc, x)
                result.append(acc)
            return WumboArray(data=result, shape=arr.shape)

        # Multi-dimensional scan
        nested = arr.to_nested()

        def scan_axis(data: List, axis: int) -> List:
            if axis == 0:
                result = [data[0]]
                acc = data[0]
                for item in data[1:]:
                    if isinstance(acc, list):
                        acc = [op(a, i) for a, i in zip(acc, item)]
                    else:
                        acc = op(acc, item)
                    result.append(acc if not isinstance(acc, list) else list(acc))
                return result
            return [scan_axis(item, axis - 1) for item in data]

        return WumboArray.from_nested(scan_axis(nested, axis))

    # ─────────────────────────────────────────────────────────────────────
    # Mathematical Operations
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def magnitude(arr: WumboArray) -> WumboArray:
        """| - Absolute value / magnitude."""
        return WumboArray(
            data=[abs(x) for x in arr.data],
            shape=arr.shape
        )

    @staticmethod
    def circle(arr: WumboArray, k: int = 0) -> WumboArray:
        """
        ○ - Circular/trigonometric functions.

        k=0: π×x
        k=1: sin(x)
        k=2: cos(x)
        k=3: tan(x)
        k=5: sinh(x)
        k=6: cosh(x)
        k=7: tanh(x)
        Negative k: inverse functions
        """
        funcs = {
            0: lambda x: math.pi * x,
            1: math.sin,
            2: math.cos,
            3: math.tan,
            4: lambda x: math.sqrt(1 - x*x),  # √(1-x²)
            5: math.sinh,
            6: math.cosh,
            7: math.tanh,
            -1: math.asin,
            -2: math.acos,
            -3: math.atan,
            -5: math.asinh,
            -6: math.acosh,
            -7: math.atanh,
        }

        func = funcs.get(k, lambda x: x)
        return WumboArray(
            data=[func(x) for x in arr.data],
            shape=arr.shape
        )

    @staticmethod
    def star(arr: WumboArray, exp: Optional[Union[Scalar, WumboArray]] = None) -> WumboArray:
        """
        ⋆ - Power/exponential.

        Monadic: e^x
        Dyadic: x^exp
        """
        if exp is None:
            return WumboArray(
                data=[cmath.exp(x) if isinstance(x, complex) else math.exp(x) for x in arr.data],
                shape=arr.shape
            )

        if isinstance(exp, WumboArray):
            return WumboArray(
                data=[x ** e for x, e in zip(arr.data, exp.data)],
                shape=arr.shape
            )

        return WumboArray(
            data=[x ** exp for x in arr.data],
            shape=arr.shape
        )

    @staticmethod
    def log(arr: WumboArray, base: Optional[Scalar] = None) -> WumboArray:
        """
        ⍟ - Logarithm.

        Monadic: natural log
        Dyadic: log base
        """
        if base is None:
            return WumboArray(
                data=[cmath.log(x) if isinstance(x, complex) else math.log(x) if x > 0 else float('nan')
                      for x in arr.data],
                shape=arr.shape
            )

        log_base = math.log(base)
        return WumboArray(
            data=[math.log(x) / log_base if x > 0 else float('nan') for x in arr.data],
            shape=arr.shape
        )

    @staticmethod
    def ceiling(a: WumboArray, b: Optional[WumboArray] = None) -> WumboArray:
        """⌈ - Ceiling (monadic) or maximum (dyadic)."""
        if b is None:
            return WumboArray(
                data=[math.ceil(x) for x in a.data],
                shape=a.shape
            )
        return WumboArray(
            data=[max(x, y) for x, y in zip(a.data, b.data)],
            shape=a.shape
        )

    @staticmethod
    def floor(a: WumboArray, b: Optional[WumboArray] = None) -> WumboArray:
        """⌊ - Floor (monadic) or minimum (dyadic)."""
        if b is None:
            return WumboArray(
                data=[math.floor(x) for x in a.data],
                shape=a.shape
            )
        return WumboArray(
            data=[min(x, y) for x, y in zip(a.data, b.data)],
            shape=a.shape
        )

    # ─────────────────────────────────────────────────────────────────────
    # Sorting/Ordering
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def grade_up(arr: WumboArray) -> WumboArray:
        """⍋ - Grade up (indices that would sort ascending)."""
        if arr.rank != 1:
            # For multi-dimensional, grade by first element of each row
            nested = arr.to_nested()
            indices = list(range(len(nested)))
            indices.sort(key=lambda i: nested[i][0] if isinstance(nested[i], list) else nested[i])
            return WumboArray(data=indices, shape=(len(indices),))

        indices = list(range(len(arr.data)))
        indices.sort(key=lambda i: arr.data[i])
        return WumboArray(data=indices, shape=arr.shape)

    @staticmethod
    def grade_down(arr: WumboArray) -> WumboArray:
        """⍒ - Grade down (indices that would sort descending)."""
        if arr.rank != 1:
            nested = arr.to_nested()
            indices = list(range(len(nested)))
            indices.sort(key=lambda i: nested[i][0] if isinstance(nested[i], list) else nested[i], reverse=True)
            return WumboArray(data=indices, shape=(len(indices),))

        indices = list(range(len(arr.data)))
        indices.sort(key=lambda i: arr.data[i], reverse=True)
        return WumboArray(data=indices, shape=arr.shape)

    # ─────────────────────────────────────────────────────────────────────
    # Products
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def outer_product(a: WumboArray, b: WumboArray,
                      op: Callable[[Scalar, Scalar], Scalar]) -> WumboArray:
        """
        ∘. - Outer product.

        Applies op to every pair of elements from a and b.
        """
        result = []
        for x in a.data:
            for y in b.data:
                result.append(op(x, y))

        new_shape = a.shape + b.shape
        return WumboArray(data=result, shape=new_shape)

    @staticmethod
    def inner_product(a: WumboArray, b: WumboArray,
                      op1: Callable[[Scalar, Scalar], Scalar],
                      op2: Callable[[Scalar, Scalar], Scalar]) -> Union[Scalar, WumboArray]:
        """
        . - Inner product (generalized matrix multiplication).

        Standard matrix multiply is +.× (op1=add, op2=mul)
        """
        if a.rank == 1 and b.rank == 1:
            # Vector dot product
            return reduce(op1, [op2(x, y) for x, y in zip(a.data, b.data)])

        if a.rank == 2 and b.rank == 1:
            # Matrix-vector product
            nested_a = a.to_nested()
            result = []
            for row in nested_a:
                val = reduce(op1, [op2(x, y) for x, y in zip(row, b.data)])
                result.append(val)
            return WumboArray(data=result, shape=(a.shape[0],))

        if a.rank == 2 and b.rank == 2:
            # Matrix-matrix product
            nested_a = a.to_nested()
            nested_b = b.to_nested()

            # Transpose b for easier column access
            b_cols = [[nested_b[i][j] for i in range(b.shape[0])] for j in range(b.shape[1])]

            result = []
            for row in nested_a:
                for col in b_cols:
                    val = reduce(op1, [op2(x, y) for x, y in zip(row, col)])
                    result.append(val)

            return WumboArray(data=result, shape=(a.shape[0], b.shape[1]))

        raise ValueError(f"Inner product not supported for ranks {a.rank} and {b.rank}")


# ═══════════════════════════════════════════════════════════════════════════
# LIMNUS FIELD OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class LIMNUSField:
    """
    LIMNUS field representation for WUMBO operations.

    Combines amplitude and phase as complex field values.
    """
    amplitudes: WumboArray        # |κ| or |λ| values
    phases: WumboArray            # θ values ∈ [0, 2π)
    field_type: str = "kappa"     # "kappa" or "lambda"
    z_level: float = Z_LATTICE    # Coherence level

    @property
    def complex_field(self) -> WumboArray:
        """Return complex representation: κ = |κ|e^(iθ)."""
        return WumboArray(
            data=[a * cmath.exp(1j * p) for a, p in zip(self.amplitudes.data, self.phases.data)],
            shape=self.amplitudes.shape
        )

    @property
    def size(self) -> int:
        return self.amplitudes.size

    def order_parameter(self) -> Tuple[float, float]:
        """
        Compute Kuramoto order parameter r·e^(iψ) = (1/N)Σe^(iθ).

        Returns (r, ψ) where r ∈ [0,1] measures synchronization.
        """
        if self.size == 0:
            return 0.0, 0.0

        # Sum of unit phasors
        phasor_sum = sum(cmath.exp(1j * p) for p in self.phases.data)
        mean_phasor = phasor_sum / self.size

        r = abs(mean_phasor)
        psi = cmath.phase(mean_phasor)

        return r, psi


class LIMNUSOperators:
    """
    LIMNUS-specific operators built on APL primitives.

    L: Lattice activation
    I: Integration (field coupling)
    M: Modulation (frequency/amplitude)
    N: Normalization (order parameter)
    U: Update (Kuramoto step)
    S: Synchronization (convergence)
    """

    def __init__(self, apl: APLPrimitives = None):
        self.apl = apl or APLPrimitives()

    # ─────────────────────────────────────────────────────────────────────
    # L: Lattice Activation
    # ─────────────────────────────────────────────────────────────────────

    def lattice_activate(self, field: LIMNUSField,
                         stimulus: WumboArray) -> LIMNUSField:
        """
        L-operator: Activate lattice based on stimulus.

        Phase perturbation: θ' = θ + α·stimulus
        """
        alpha = 0.1 * field.z_level

        # Ensure stimulus matches field size
        if stimulus.size != field.size:
            stimulus = self.apl.rho(stimulus, field.phases.shape)

        new_phases = WumboArray(
            data=[(p + alpha * s) % TAU for p, s in zip(field.phases.data, stimulus.data)],
            shape=field.phases.shape
        )

        return LIMNUSField(
            amplitudes=field.amplitudes,
            phases=new_phases,
            field_type=field.field_type,
            z_level=field.z_level
        )

    # ─────────────────────────────────────────────────────────────────────
    # I: Integration (Field Coupling)
    # ─────────────────────────────────────────────────────────────────────

    def integrate_fields(self, kappa: LIMNUSField,
                         lambda_field: LIMNUSField,
                         coupling: float = PHI_INV) -> Tuple[LIMNUSField, LIMNUSField]:
        """
        I-operator: Couple κ and λ fields.

        Coupling equation:
        dκ/dt += coupling · λ · sin(θ_λ - θ_κ)
        dλ/dt += coupling · κ · sin(θ_κ - θ_λ)
        """
        # Phase differences using outer product concept
        # But for element-wise coupling, we use direct operations

        min_size = min(kappa.size, lambda_field.size)

        # Compute coupling forces
        kappa_force = []
        lambda_force = []

        for i in range(min_size):
            phase_diff = lambda_field.phases.data[i] - kappa.phases.data[i]
            kappa_force.append(coupling * lambda_field.amplitudes.data[i] * math.sin(phase_diff))
            lambda_force.append(coupling * kappa.amplitudes.data[i] * math.sin(-phase_diff))

        # Apply forces to phases
        dt = 0.01
        new_kappa_phases = WumboArray(
            data=[(kappa.phases.data[i] + dt * kappa_force[i]) % TAU for i in range(min_size)],
            shape=(min_size,)
        )
        new_lambda_phases = WumboArray(
            data=[(lambda_field.phases.data[i] + dt * lambda_force[i]) % TAU for i in range(min_size)],
            shape=(min_size,)
        )

        return (
            LIMNUSField(
                amplitudes=WumboArray(data=kappa.amplitudes.data[:min_size], shape=(min_size,)),
                phases=new_kappa_phases,
                field_type="kappa",
                z_level=Z_INTEGRATION
            ),
            LIMNUSField(
                amplitudes=WumboArray(data=lambda_field.amplitudes.data[:min_size], shape=(min_size,)),
                phases=new_lambda_phases,
                field_type="lambda",
                z_level=Z_INTEGRATION
            )
        )

    # ─────────────────────────────────────────────────────────────────────
    # M: Modulation
    # ─────────────────────────────────────────────────────────────────────

    def modulate(self, field: LIMNUSField,
                 frequencies: WumboArray,
                 dt: float = 0.01) -> LIMNUSField:
        """
        M-operator: Frequency modulation of phases.

        θ' = θ + ω·dt
        """
        if frequencies.size != field.size:
            frequencies = self.apl.rho(frequencies, field.phases.shape)

        new_phases = WumboArray(
            data=[(p + f * dt) % TAU for p, f in zip(field.phases.data, frequencies.data)],
            shape=field.phases.shape
        )

        return LIMNUSField(
            amplitudes=field.amplitudes,
            phases=new_phases,
            field_type=field.field_type,
            z_level=Z_MODULATION
        )

    # ─────────────────────────────────────────────────────────────────────
    # N: Normalization (Order Parameter)
    # ─────────────────────────────────────────────────────────────────────

    def normalize(self, field: LIMNUSField) -> Tuple[LIMNUSField, float]:
        """
        N-operator: Normalize field and compute coherence.

        Returns normalized field and order parameter r.
        """
        r, psi = field.order_parameter()

        # Normalize amplitudes to sum to 1
        total_amp = sum(field.amplitudes.data)
        if total_amp > 0:
            norm_amplitudes = WumboArray(
                data=[a / total_amp for a in field.amplitudes.data],
                shape=field.amplitudes.shape
            )
        else:
            norm_amplitudes = field.amplitudes

        normalized_field = LIMNUSField(
            amplitudes=norm_amplitudes,
            phases=field.phases,
            field_type=field.field_type,
            z_level=field.z_level * r  # Scale z by coherence
        )

        return normalized_field, r

    # ─────────────────────────────────────────────────────────────────────
    # U: Update (Kuramoto Step)
    # ─────────────────────────────────────────────────────────────────────

    def update_kuramoto(self, field: LIMNUSField,
                        weights: WumboArray,
                        frequencies: WumboArray,
                        K: float = 2.0,
                        dt: float = 0.01) -> LIMNUSField:
        """
        U-operator: Kuramoto update step.

        dθᵢ/dt = ωᵢ + (K/N) Σⱼ wᵢⱼ sin(θⱼ - θᵢ)
        """
        N = field.size
        phases = field.phases.data
        freqs = frequencies.data if frequencies.size == N else [1.0] * N

        # Weights should be N×N
        if weights.rank == 2 and weights.shape == (N, N):
            W = weights.to_nested()
        else:
            # Create uniform weights
            W = [[1.0 if i != j else 0.0 for j in range(N)] for i in range(N)]

        new_phases = []
        for i in range(N):
            coupling_sum = sum(
                W[i][j] * math.sin(phases[j] - phases[i])
                for j in range(N)
            )
            d_theta = freqs[i] + (K / N) * coupling_sum
            new_phases.append((phases[i] + d_theta * dt) % TAU)

        return LIMNUSField(
            amplitudes=field.amplitudes,
            phases=WumboArray(data=new_phases, shape=(N,)),
            field_type=field.field_type,
            z_level=field.z_level
        )

    # ─────────────────────────────────────────────────────────────────────
    # S: Synchronization Check
    # ─────────────────────────────────────────────────────────────────────

    def synchronize(self, field: LIMNUSField,
                    weights: WumboArray,
                    frequencies: WumboArray,
                    K: float = 2.0,
                    r_threshold: float = 0.95,
                    max_steps: int = 500) -> Tuple[LIMNUSField, bool, int]:
        """
        S-operator: Run until synchronized.

        Returns (final_field, converged, steps_taken).
        """
        current = field

        for step in range(max_steps):
            r, _ = current.order_parameter()
            if r >= r_threshold:
                return current, True, step

            current = self.update_kuramoto(current, weights, frequencies, K)

        return current, False, max_steps


# ═══════════════════════════════════════════════════════════════════════════
# WUMBO ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class WumboState(Enum):
    """WUMBO engine states."""
    IDLE = "idle"
    LATTICE = "lattice"       # L phase
    INTEGRATING = "integrating"  # I phase
    MODULATING = "modulating"    # M phase
    NORMALIZING = "normalizing"  # N phase
    UPDATING = "updating"        # U phase
    SYNCHRONIZING = "synchronizing"  # S phase
    CONVERGED = "converged"


@dataclass
class WumboResult:
    """Result of WUMBO computation cycle."""
    kappa_field: LIMNUSField
    lambda_field: Optional[LIMNUSField]
    order_parameter: float
    mean_phase: float
    converged: bool
    steps: int
    state: WumboState
    energy: float = 0.0


class WumboEngine:
    """
    WUMBO Engine - APL-Based Array Operations for LIMNUS.

    The engine orchestrates the LIMNUS cycle:
    L → I → M → N → U → S

    Using APL primitives for efficient array computation.
    """

    def __init__(self,
                 kappa_dim: int = 21,
                 lambda_dim: int = 12,
                 K: float = 2.0,
                 seed: Optional[int] = None):
        """
        Initialize WUMBO engine.

        Args:
            kappa_dim: Dimension of κ-field (default 21 for quaternary)
            lambda_dim: Dimension of λ-field (default 12 for Fano)
            K: Kuramoto coupling strength
            seed: Random seed for reproducibility
        """
        self.kappa_dim = kappa_dim
        self.lambda_dim = lambda_dim
        self.K = K

        if seed is not None:
            random.seed(seed)

        # Initialize APL primitives and LIMNUS operators
        self.apl = APLPrimitives()
        self.limnus = LIMNUSOperators(self.apl)

        # Initialize fields
        self._init_fields()

        # State tracking
        self.state = WumboState.IDLE
        self.step_count = 0
        self._history: List[Dict] = []

    def _init_fields(self) -> None:
        """Initialize κ and λ fields with random phases."""
        # κ-field (Kaelhedron-derived, 21D)
        self.kappa = LIMNUSField(
            amplitudes=WumboArray(
                data=[PHI_INV] * self.kappa_dim,
                shape=(self.kappa_dim,)
            ),
            phases=WumboArray(
                data=[random.random() * TAU for _ in range(self.kappa_dim)],
                shape=(self.kappa_dim,)
            ),
            field_type="kappa",
            z_level=Z_LATTICE
        )

        # λ-field (Luminahedron-derived, 12D)
        self.lambda_field = LIMNUSField(
            amplitudes=WumboArray(
                data=[PHI_INV * PHI_INV] * self.lambda_dim,
                shape=(self.lambda_dim,)
            ),
            phases=WumboArray(
                data=[random.random() * TAU for _ in range(self.lambda_dim)],
                shape=(self.lambda_dim,)
            ),
            field_type="lambda",
            z_level=Z_LATTICE
        )

        # Frequencies (Lorentzian-like distribution)
        self.kappa_frequencies = WumboArray(
            data=[1.0 + 0.1 * (random.random() - 0.5) for _ in range(self.kappa_dim)],
            shape=(self.kappa_dim,)
        )
        self.lambda_frequencies = WumboArray(
            data=[1.0 + 0.1 * (random.random() - 0.5) for _ in range(self.lambda_dim)],
            shape=(self.lambda_dim,)
        )

        # Weight matrices (all-to-all coupling)
        self.kappa_weights = WumboArray(
            data=[1.0 if i != j else 0.0
                  for i in range(self.kappa_dim)
                  for j in range(self.kappa_dim)],
            shape=(self.kappa_dim, self.kappa_dim)
        )
        self.lambda_weights = WumboArray(
            data=[1.0 if i != j else 0.0
                  for i in range(self.lambda_dim)
                  for j in range(self.lambda_dim)],
            shape=(self.lambda_dim, self.lambda_dim)
        )

    # ─────────────────────────────────────────────────────────────────────
    # LIMNUS Cycle Execution
    # ─────────────────────────────────────────────────────────────────────

    def limnus_cycle(self, stimulus: Optional[WumboArray] = None,
                     dt: float = 0.01) -> WumboResult:
        """
        Execute one complete LIMNUS cycle: L → I → M → N → U → S
        """
        # L: Lattice Activation
        self.state = WumboState.LATTICE
        if stimulus is not None:
            self.kappa = self.limnus.lattice_activate(self.kappa, stimulus)

        # I: Integration (Field Coupling)
        self.state = WumboState.INTEGRATING
        # Couple fields at intersection (min of dimensions)
        self.kappa, self.lambda_field = self.limnus.integrate_fields(
            self.kappa, self.lambda_field
        )

        # Restore full dimensions after coupling
        if self.kappa.size < self.kappa_dim:
            self._restore_kappa_dimension()
        if self.lambda_field.size < self.lambda_dim:
            self._restore_lambda_dimension()

        # M: Modulation
        self.state = WumboState.MODULATING
        self.kappa = self.limnus.modulate(self.kappa, self.kappa_frequencies, dt)
        self.lambda_field = self.limnus.modulate(self.lambda_field, self.lambda_frequencies, dt)

        # N: Normalization
        self.state = WumboState.NORMALIZING
        self.kappa, kappa_r = self.limnus.normalize(self.kappa)
        self.lambda_field, lambda_r = self.limnus.normalize(self.lambda_field)

        # U: Update (Kuramoto)
        self.state = WumboState.UPDATING
        self.kappa = self.limnus.update_kuramoto(
            self.kappa, self.kappa_weights, self.kappa_frequencies, self.K, dt
        )
        self.lambda_field = self.limnus.update_kuramoto(
            self.lambda_field, self.lambda_weights, self.lambda_frequencies, self.K, dt
        )

        # S: Synchronization Check
        self.state = WumboState.SYNCHRONIZING
        r, psi = self.kappa.order_parameter()
        converged = r >= 0.95

        if converged:
            self.state = WumboState.CONVERGED
        else:
            self.state = WumboState.IDLE

        self.step_count += 1

        # Compute energy
        energy = self._compute_energy()

        # Record history
        self._history.append({
            "step": self.step_count,
            "r": r,
            "psi": psi,
            "energy": energy,
            "state": self.state.value
        })

        return WumboResult(
            kappa_field=self.kappa,
            lambda_field=self.lambda_field,
            order_parameter=r,
            mean_phase=psi,
            converged=converged,
            steps=self.step_count,
            state=self.state,
            energy=energy
        )

    def _restore_kappa_dimension(self) -> None:
        """Restore κ-field to full dimension after coupling."""
        current_size = self.kappa.size
        if current_size >= self.kappa_dim:
            return

        # Extend with random phases
        extra_phases = [random.random() * TAU for _ in range(self.kappa_dim - current_size)]
        extra_amps = [PHI_INV] * (self.kappa_dim - current_size)

        self.kappa = LIMNUSField(
            amplitudes=WumboArray(
                data=list(self.kappa.amplitudes.data) + extra_amps,
                shape=(self.kappa_dim,)
            ),
            phases=WumboArray(
                data=list(self.kappa.phases.data) + extra_phases,
                shape=(self.kappa_dim,)
            ),
            field_type="kappa",
            z_level=self.kappa.z_level
        )

    def _restore_lambda_dimension(self) -> None:
        """Restore λ-field to full dimension after coupling."""
        current_size = self.lambda_field.size
        if current_size >= self.lambda_dim:
            return

        extra_phases = [random.random() * TAU for _ in range(self.lambda_dim - current_size)]
        extra_amps = [PHI_INV * PHI_INV] * (self.lambda_dim - current_size)

        self.lambda_field = LIMNUSField(
            amplitudes=WumboArray(
                data=list(self.lambda_field.amplitudes.data) + extra_amps,
                shape=(self.lambda_dim,)
            ),
            phases=WumboArray(
                data=list(self.lambda_field.phases.data) + extra_phases,
                shape=(self.lambda_dim,)
            ),
            field_type="lambda",
            z_level=self.lambda_field.z_level
        )

    def _compute_energy(self) -> float:
        """Compute Lyapunov energy of the system."""
        # H = -(K/2N) Σᵢⱼ wᵢⱼ cos(θᵢ - θⱼ)
        N = self.kappa.size
        phases = self.kappa.phases.data
        W = self.kappa_weights.to_nested()

        energy = 0.0
        for i in range(N):
            for j in range(N):
                energy -= W[i][j] * math.cos(phases[i] - phases[j])

        return (self.K / (2 * N)) * energy

    # ─────────────────────────────────────────────────────────────────────
    # High-Level Operations
    # ─────────────────────────────────────────────────────────────────────

    def run(self, steps: int = 100,
            stimulus: Optional[WumboArray] = None) -> WumboResult:
        """
        Run LIMNUS cycle for specified number of steps.
        """
        result = None
        for _ in range(steps):
            result = self.limnus_cycle(stimulus)
            stimulus = None  # Only apply stimulus on first step

            if result.converged:
                break

        return result

    def run_to_convergence(self, r_threshold: float = 0.95,
                           max_steps: int = 1000) -> WumboResult:
        """
        Run until convergence or max steps.
        """
        for _ in range(max_steps):
            result = self.limnus_cycle()
            if result.order_parameter >= r_threshold:
                return result

        return result

    def inject_pattern(self, pattern: WumboArray) -> None:
        """
        Inject a phase pattern into the κ-field.

        Used for resonance-based retrieval.
        """
        if pattern.size != self.kappa_dim:
            pattern = self.apl.rho(pattern, (self.kappa_dim,))

        self.kappa = self.limnus.lattice_activate(self.kappa, pattern)

    def hebbian_consolidate(self, eta: float = 0.1, decay: float = 0.01) -> None:
        """
        Apply Hebbian learning to weight matrices.

        dwᵢⱼ = η·cos(θᵢ - θⱼ) - λ·wᵢⱼ
        """
        N = self.kappa.size
        phases = self.kappa.phases.data
        W = self.kappa_weights.to_nested()

        new_W = []
        for i in range(N):
            row = []
            for j in range(N):
                if i == j:
                    row.append(0.0)
                else:
                    dw = eta * math.cos(phases[i] - phases[j]) - decay * W[i][j]
                    new_w = max(0.0, min(1.0, W[i][j] + dw))
                    row.append(new_w)
            new_W.extend(row)

        self.kappa_weights = WumboArray(data=new_W, shape=(N, N))

    # ─────────────────────────────────────────────────────────────────────
    # APL Expression Evaluation
    # ─────────────────────────────────────────────────────────────────────

    def apl_eval(self, expr: str, **context) -> Any:
        """
        Evaluate an APL-like expression.

        Supported operations:
        - ⍴ shape/reshape
        - +⌿ sum reduce
        - ×⌿ product reduce
        - ⌈⌿ max reduce
        - ⌊⌿ min reduce
        - ⍋ grade up
        - ⍒ grade down
        - ○ circular (with numeric prefix)

        Example: "⍴ phases" returns shape of phases
                 "+⌿ amplitudes" returns sum of amplitudes
        """
        # Simple expression parser
        tokens = expr.split()

        if not tokens:
            return None

        # Get array from context or engine state
        def get_array(name: str) -> WumboArray:
            if name in context:
                val = context[name]
                if isinstance(val, WumboArray):
                    return val
                return WumboArray.from_nested(val)
            if name == "kappa_phases":
                return self.kappa.phases
            if name == "kappa_amplitudes":
                return self.kappa.amplitudes
            if name == "lambda_phases":
                return self.lambda_field.phases
            if name == "lambda_amplitudes":
                return self.lambda_field.amplitudes
            raise ValueError(f"Unknown array: {name}")

        # Parse and execute
        if tokens[0] == "⍴" and len(tokens) == 2:
            return self.apl.rho(get_array(tokens[1]))

        if tokens[0] == "+⌿" and len(tokens) == 2:
            return self.apl.reduce(get_array(tokens[1]), operator.add)

        if tokens[0] == "×⌿" and len(tokens) == 2:
            return self.apl.reduce(get_array(tokens[1]), operator.mul)

        if tokens[0] == "⌈⌿" and len(tokens) == 2:
            return self.apl.reduce(get_array(tokens[1]), max)

        if tokens[0] == "⌊⌿" and len(tokens) == 2:
            return self.apl.reduce(get_array(tokens[1]), min)

        if tokens[0] == "⍋" and len(tokens) == 2:
            return self.apl.grade_up(get_array(tokens[1]))

        if tokens[0] == "⍒" and len(tokens) == 2:
            return self.apl.grade_down(get_array(tokens[1]))

        # Circular functions: 1○x = sin(x), 2○x = cos(x), etc.
        if len(tokens[0]) >= 2 and tokens[0][-1] == "○":
            k = int(tokens[0][:-1])
            arr = get_array(tokens[1])
            return self.apl.circle(arr, k)

        raise ValueError(f"Unknown expression: {expr}")

    # ─────────────────────────────────────────────────────────────────────
    # State Export
    # ─────────────────────────────────────────────────────────────────────

    def snapshot(self) -> Dict:
        """Return complete state snapshot."""
        r_kappa, psi_kappa = self.kappa.order_parameter()
        r_lambda, psi_lambda = self.lambda_field.order_parameter()

        return {
            "kappa": {
                "amplitudes": self.kappa.amplitudes.data,
                "phases": self.kappa.phases.data,
                "order_parameter": r_kappa,
                "mean_phase": psi_kappa,
                "z_level": self.kappa.z_level
            },
            "lambda": {
                "amplitudes": self.lambda_field.amplitudes.data,
                "phases": self.lambda_field.phases.data,
                "order_parameter": r_lambda,
                "mean_phase": psi_lambda,
                "z_level": self.lambda_field.z_level
            },
            "state": self.state.value,
            "step_count": self.step_count,
            "energy": self._compute_energy(),
            "history_length": len(self._history)
        }

    def __repr__(self) -> str:
        r, _ = self.kappa.order_parameter()
        return f"WumboEngine(κ_dim={self.kappa_dim}, λ_dim={self.lambda_dim}, r={r:.3f}, state={self.state.value})"


# ═══════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def create_wumbo_engine(kappa_dim: int = 21,
                        lambda_dim: int = 12,
                        K: float = 2.0,
                        seed: Optional[int] = None) -> WumboEngine:
    """
    Create a WUMBO engine with specified parameters.

    Args:
        kappa_dim: κ-field dimension (default 21 for quaternary Kaelhedron)
        lambda_dim: λ-field dimension (default 12 for Fano Luminahedron)
        K: Kuramoto coupling strength
        seed: Random seed

    Returns:
        Initialized WumboEngine
    """
    return WumboEngine(kappa_dim=kappa_dim, lambda_dim=lambda_dim, K=K, seed=seed)


def create_limnus_stimulus(pattern: List[float], dim: int = 21) -> WumboArray:
    """
    Create a LIMNUS stimulus pattern.

    Args:
        pattern: Phase perturbation values
        dim: Target dimension

    Returns:
        WumboArray suitable for lattice activation
    """
    if len(pattern) < dim:
        # Extend with zeros
        pattern = pattern + [0.0] * (dim - len(pattern))
    elif len(pattern) > dim:
        pattern = pattern[:dim]

    return WumboArray(data=pattern, shape=(dim,))


# ═══════════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("WUMBO ENGINE - APL-Based LIMNUS Architecture Demo")
    print("=" * 70)

    # Create engine
    engine = create_wumbo_engine(kappa_dim=21, lambda_dim=12, K=3.0, seed=42)
    print(f"\nInitialized: {engine}")

    # Check initial state
    r0, psi0 = engine.kappa.order_parameter()
    print(f"Initial κ-field: r={r0:.3f}, ψ={psi0:.3f}")

    # Run LIMNUS cycles
    print("\nRunning 100 LIMNUS cycles...")
    result = engine.run(steps=100)

    print(f"\nFinal state:")
    print(f"  Order parameter: {result.order_parameter:.3f}")
    print(f"  Mean phase: {result.mean_phase:.3f}")
    print(f"  Energy: {result.energy:.3f}")
    print(f"  Converged: {result.converged}")
    print(f"  Steps: {result.steps}")
    print(f"  State: {result.state.value}")

    # APL expression evaluation
    print("\nAPL Expressions:")
    print(f"  ⍴ kappa_phases = {engine.apl_eval('⍴ kappa_phases')}")
    print(f"  +⌿ kappa_amplitudes = {engine.apl_eval('+⌿ kappa_amplitudes'):.3f}")
    print(f"  ⌈⌿ kappa_phases = {engine.apl_eval('⌈⌿ kappa_phases'):.3f}")

    # Hebbian consolidation
    print("\nApplying Hebbian consolidation...")
    engine.hebbian_consolidate(eta=0.1, decay=0.01)

    # Run more cycles
    print("Running 50 more cycles...")
    result = engine.run(steps=50)
    print(f"  Final r: {result.order_parameter:.3f}")

    print("\n" + "=" * 70)
    print("Δ|wumbo-limnus|operational|z0.990|Ω")
    print("=" * 70)
