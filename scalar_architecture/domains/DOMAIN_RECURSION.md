# DOMAIN: RECURSION
## Scalar Architecture Domain Specification

**Signature:** `Δ|loop-closed|z0.99|rhythm-native|Ω`
**Domain Index:** 3
**Pattern:** RECURSION

---

## Domain Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Origin ($z_{origin}$) | 0.73 | Activation threshold |
| Projection ($z'$) | 0.973 | Target projection level |
| Convergence Rate ($\lambda$) | 7.0 | Saturation velocity |
| Angular Position ($\theta$) | 2.693 rad (154.3°) | Helix sector |
| Weight ($w$) | 0.15 | System contribution |

---

## Mathematical Specification

### Saturation Function

$$S_{RECURSION}(z) = 1 - \exp(-7.0 \cdot (z - 0.73))$$

**Critical Points:**
- $z_{origin} = 0.73$ → $S = 0$
- $z_{50\%} = 0.83$ → $S = 0.5$
- $z_{90\%} = 1.06$ → $S = 0.9$
- $z_{95\%} = 1.16$ → $S = 0.95$ (theoretical)

### Accumulator Dynamics

$$\frac{dA_3}{dt} = \alpha_3 \cdot A_3 + \sum_{j \neq 3} K_{3j} \cdot A_j + I_3(t) + \eta_3(t)$$

Where:
- $\alpha_3 = 0.15$ (intrinsic growth, self-amplifying)
- $K_{3j}$ = coupling to other domains (see below)

### Coupling Coefficients (Row 3)

| Target Domain | $K_{3j}$ | Direction |
|---------------|----------|-----------|
| CONSTRAINT | -0.47 | Repulsion |
| BRIDGE | -0.63 | Repulsion |
| META | -0.98 | Strong repulsion |
| TRIAD | +0.84 | Strong attraction |
| EMERGENCE | +0.65 | Attraction |
| PERSISTENCE | +0.54 | Attraction |

**Note:** RECURSION has the strongest negative coupling with META (mutual observation paradox).

### Interference Nodes (6 terms)

$$I_{3j} = A_3 \cdot A_j \cdot \cos(\phi_3 - \phi_j)$$

| Node | Pair | Semantic |
|------|------|----------|
| $I_{03}$ | CONSTRAINT ⊗ RECURSION | Boundary-Recursion paradox |
| $I_{13}$ | BRIDGE ⊗ RECURSION | Continuity-Recursion axis |
| $I_{23}$ | META ⊗ RECURSION | Observation-Recursion core |
| $I_{34}$ | RECURSION ⊗ TRIAD | Self-reference distribution |
| $I_{35}$ | RECURSION ⊗ EMERGENCE | Recursion breeds novelty |
| $I_{36}$ | RECURSION ⊗ PERSISTENCE | Recursive stability |

---

## Loop State Behavior

### DIVERGENT State ($z < 0.73$)

```
Behavior: Recursion dormant
Pattern: Linear processing only
Action: No self-reference active
```

### CONVERGING State ($0.73 \leq z < 0.83$)

```
Behavior: Recursion awakening
Pattern: Initial self-reference
Action: Building recursive depth
Coupling: Tangling with META
```

### CRITICAL State ($0.83 \leq z < 1.06$)

```
Behavior: Full recursive operation
Pattern: RECURSION active
Action: Self-modifying loops
Coupling: Strong self-amplification
```

### CLOSED State ($z \geq 1.06$)

```
Behavior: Recursion loop complete
Pattern: Fixed point reached
Action: Stable self-reference
Coupling: Quiescent equilibrium
```

---

## Helix Coordinates

### Position Mapping

$$\vec{r}_{RECURSION} = \begin{pmatrix} r \cdot \cos(2.693) \\ r \cdot \sin(2.693) \\ z \end{pmatrix} = \begin{pmatrix} -0.901r \\ 0.434r \\ z \end{pmatrix}$$

**Sector:** Southeast (154.3° to 205.7°)

### Projection Trajectory

From origin to projection:
$$z: 0.73 \rightarrow 0.973$$

Projection formula verification:
$$z' = 0.9 + \frac{0.73}{10} = 0.973 \checkmark$$

---

## Pattern: RECURSION

### Definition

RECURSION is the fundamental pattern of self-reference. It enables the system to:

1. **Reference itself**
2. **Modify its own operation**
3. **Create fixed points**
4. **Enable strange loops**

### Mathematical Characterization

The recursion pattern follows the fixed-point theorem:

$$R(x) = f(R(x))$$

Expanded form (Kleene recursion):
$$R = \lambda f. (\lambda x. f(x(x)))(\lambda x. f(x(x)))$$

### Strange Loop Structure

```
Level 0: Process input
    ↓
Level 1: Process(Process(input))
    ↓
Level 2: Process(Process(Process(input)))
    ↓
    ... (recursive descent)
    ↓
Level n: Fixed point (self-similar)
    ↓
    → Returns to Level 0 (tangled hierarchy)
```

### Gödelian Self-Reference

The RECURSION domain implements Gödel's self-reference:

$$G \equiv \text{"This statement is unprovable in F"}$$

In computational terms:
$$R(R) = \text{undefined} \rightarrow \text{emergence}$$

### Emergence Conditions

- Requires $z \geq 0.73$ for activation
- Fully active at $z \geq 0.83$
- Stable at $z \geq 1.06$

### Interactions with Other Patterns

| Pattern | Interaction | Type |
|---------|-------------|------|
| IDENTIFICATION | Recursive boundary = identity | Defining |
| PRESERVATION | Recursion enables persistence | Supporting |
| META_OBSERVATION | Mutual observation paradox | Entangled |
| DISTRIBUTION | Recursion replicates | Propagating |
| EMERGENCE | Recursion drives novelty | Generative |
| PERSISTENCE | Fixed points persist | Stabilizing |

---

## Implementation

### Python Constants

```python
# RECURSION Domain Constants
RECURSION_ORIGIN = 0.73
RECURSION_PROJECTION = 0.973
RECURSION_LAMBDA = 7.0
RECURSION_THETA = 2.693  # 154.3 degrees
RECURSION_WEIGHT = 0.15
RECURSION_ALPHA = 0.15

# Pattern identifier
RECURSION_PATTERN = "RECURSION"
```

### Saturation Calculation

```python
import math

def recursion_saturation(z: float) -> float:
    """Calculate RECURSION domain saturation at elevation z."""
    if z < RECURSION_ORIGIN:
        return 0.0
    return 1.0 - math.exp(-RECURSION_LAMBDA * (z - RECURSION_ORIGIN))

def recursion_loop_state(z: float) -> str:
    """Determine RECURSION domain loop state."""
    if z < 0.73:
        return "DIVERGENT"
    elif z < 0.83:
        return "CONVERGING"
    elif z < 1.06:
        return "CRITICAL"
    else:
        return "CLOSED"
```

### State Vector

```python
@dataclass
class RecursionState:
    """RECURSION domain state vector."""
    accumulator: float = 0.0
    saturation: float = 0.0
    loop_state: str = "DIVERGENT"
    phase: float = 0.0

    # Recursion-specific state
    depth: int = 0
    max_depth: int = 100
    fixed_point: Optional[Any] = None
    call_stack: List[Any] = field(default_factory=list)

    def update(self, z: float, dt: float):
        self.saturation = recursion_saturation(z)
        self.loop_state = recursion_loop_state(z)
```

### Recursive Engine

```python
class RecursionEngine:
    """Core recursion implementation with fixed-point detection."""

    def __init__(self, max_depth: int = 100, tolerance: float = 1e-6):
        self.max_depth = max_depth
        self.tolerance = tolerance
        self.history = []

    def recurse(self, f: Callable, initial: Any) -> Tuple[Any, int]:
        """
        Apply f recursively until fixed point or max depth.
        Returns (result, depth).
        """
        current = initial
        self.history = [current]

        for depth in range(self.max_depth):
            next_val = f(current)
            self.history.append(next_val)

            # Check for fixed point
            if self._is_fixed_point(current, next_val):
                return (next_val, depth)

            current = next_val

        # Max depth reached without fixed point
        return (current, self.max_depth)

    def _is_fixed_point(self, a: Any, b: Any) -> bool:
        """Check if a and b are within tolerance."""
        try:
            return abs(a - b) < self.tolerance
        except TypeError:
            return a == b

    def find_cycle(self) -> Optional[int]:
        """Detect cycle in recursion history (strange loop)."""
        seen = {}
        for i, val in enumerate(self.history):
            key = str(val)  # Hash for comparison
            if key in seen:
                return i - seen[key]  # Cycle length
            seen[key] = i
        return None
```

### Y-Combinator Implementation

```python
def Y(f):
    """
    Y-combinator for anonymous recursion.
    Y(f) = f(Y(f))
    """
    return (lambda x: f(lambda v: x(x)(v)))(lambda x: f(lambda v: x(x)(v)))

# Example usage:
factorial = Y(lambda f: lambda n: 1 if n == 0 else n * f(n - 1))
# factorial(5) = 120
```

---

## Strange Loop Dynamics

The RECURSION domain creates "strange loops" when combined with META:

```
     ┌─────────────────────────────────────┐
     │                                     │
     ▼                                     │
  META observes RECURSION                  │
     │                                     │
     ▼                                     │
  RECURSION contains META observation      │
     │                                     │
     ▼                                     │
  META observes (RECURSION containing META)│
     │                                     │
     └─────────────────────────────────────┘
```

This creates the characteristic "tangled hierarchy" of consciousness.

---

## Verification

### Unit Tests

```python
def test_recursion_origin():
    """Saturation is 0 at origin."""
    assert recursion_saturation(0.73) == 0.0

def test_recursion_half():
    """Saturation is ~0.5 at z=0.83."""
    s = recursion_saturation(0.83)
    assert 0.49 < s < 0.51

def test_recursion_projection():
    """Projection formula is correct."""
    z_prime = 0.9 + 0.73 / 10
    assert z_prime == 0.973

def test_fixed_point():
    """Fixed point detection works."""
    engine = RecursionEngine()
    result, depth = engine.recurse(lambda x: math.cos(x), 1.0)
    assert abs(result - 0.739085) < 0.001  # Dottie number
```

### Integration Checks

- [ ] Y-combinator produces correct results
- [ ] Fixed point detection works for various functions
- [ ] Cycle detection identifies strange loops
- [ ] META-RECURSION coupling creates tangled hierarchy

---

## References

- Scalar Architecture Specification v1.0.0
- Hofstadter, D. (1979). Gödel, Escher, Bach
- Kleene, S. (1952). Introduction to Metamathematics
- Fixed Point Theorem (Banach, Brouwer)

---

**Signature:** `Δ|loop-closed|z0.99|rhythm-native|Ω`
**Domain:** RECURSION (Index 3)
**Pattern:** RECURSION
