# 🧠 Tesseract Lattice Node - Core Engine

## Overview

The **Tesseract Lattice Engine** implements a holographic memory system based on Kuramoto oscillator synchronization. Memory is encoded in the phase relationships of coupled oscillators organized in 4D tesseract geometry.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ TESSERACT LATTICE ENGINE                                         │
│ ┌────────────────────────────────────────────────────┐          │
│ │ Kuramoto Oscillator Network                        │          │
│ │                                                    │          │
│ │ Plate₁ ←→ Plate₂ ←→ Plate₃ ←→ ... ←→ PlateN       │          │
│ │ ↕         ↕         ↕              ↕               │          │
│ │ Plate₄ ←→ Plate₅ ←→ Plate₆ ←→ ... ←→ PlateM       │          │
│ │                                                    │          │
│ │ Each plate: (position, phase, frequency)           │          │
│ └────────────────────────────────────────────────────┘          │
│                                                                  │
│ • update() - Kuramoto dynamics integration                       │
│ • resonance_retrieval() - Phase perturbation + evolution        │
│ • Hebbian learning - Connection strengthening                   │
│ • Order parameter tracking                                       │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### plate.py - Memory Plate Structure

```python
from lattice_core import MemoryPlate, EmotionalState

# Create from valence-arousal
plate = MemoryPlate.from_valence_arousal(
    valence=0.7,   # Positive emotion
    arousal=0.3,   # Moderate excitement
    temporal=0.0,  # Current time
    abstract=0.5   # Mid-abstraction
)

# Set content
plate.set_content_from_text("Had a great meeting today")

# Check properties
print(f"Position: {plate.position}")  # (0.7, 0.3, 0.0, 0.5)
print(f"Phase: {plate.phase}")        # θ ∈ [0, 2π)
```

### dynamics.py - Kuramoto Mathematics

```python
from lattice_core import (
    kuramoto_update,
    compute_order_parameter,
    hebbian_update,
    compute_energy
)

# Update phases
new_phases = kuramoto_update(phases, frequencies, weights, K=2.0, dt=0.01)

# Compute synchronization
r, psi = compute_order_parameter(phases)
print(f"Order parameter: r={r:.3f}")  # 0 = incoherent, 1 = synchronized

# Apply Hebbian learning
new_weights = hebbian_update(weights, phases, eta=0.1, decay=0.01)

# Compute energy
H = compute_energy(phases, weights, K=2.0)
```

### tesseract_lattice_engine.py - Main Engine

```python
from lattice_core import TesseractLatticeEngine, LatticeConfig

# Configure
config = LatticeConfig(
    K=2.5,                    # Coupling strength
    enable_quartet=True,       # P ~ N³ capacity
    hebbian_rate=0.1,         # Learning rate
    r_threshold=0.95          # Convergence threshold
)

# Create engine
engine = TesseractLatticeEngine(config=config)

# Add plates
for plate in plates:
    engine.add_plate(plate)

# Run dynamics
r, energy = engine.update(steps=100)

# Retrieve by resonance
results = engine.resonance_retrieval(
    content=query_embedding,
    emotional_position=(0.5, 0.3, 0.0, 0.5),
    top_k=5
)
```

## Key Equations

### Kuramoto Dynamics
```
dθᵢ/dt = ωᵢ + (K/N) Σⱼ wᵢⱼ sin(θⱼ - θᵢ)
```

### Order Parameter
```
r·e^(iψ) = (1/N) Σⱼ e^(iθⱼ)
```

### Hebbian Learning
```
dwᵢⱼ/dt = η·cos(θᵢ - θⱼ) - λ·wᵢⱼ
```

### Energy (Lyapunov)
```
H = -(K/2N) Σᵢⱼ wᵢⱼ cos(θᵢ - θⱼ)
```

### Higher-Order Coupling
- Pairwise: `P ~ 0.14·N` (linear capacity)
- Triplet: `P ~ N²` (quadratic)
- Quartet: `P ~ N³` (cubic - enables billions of patterns)

## 4D Tesseract Space

```
w (abstraction)
↑
│ ◆────────◆
│ ╱│      ╱│
│◆─┼─────◆ │
│ │◆────┼─◆
│ │╱    │╱
│ ◆─────◆
│
└──────────┼────→ x (valence)
          ╱
         ╱
        ↙
       z (temporal)    y (arousal)

Dimensions:
- x (valence): Emotional positivity (-1 to +1)
- y (arousal): Activation level (-1 to +1)
- z (temporal): Time position
- w (abstraction): Concrete (0) to abstract (1)
```

## Performance

| N (plates) | Pairwise Capacity | Quartet Capacity |
|------------|-------------------|------------------|
| 10         | 1.4               | 1,000            |
| 100        | 14                | 1,000,000        |
| 1,000      | 140               | 1,000,000,000    |
| 10,000     | 1,400             | 10^12            |

## Usage Example

```python
from lattice_core import TesseractLatticeEngine, MemoryPlate
import random

# Create engine
engine = TesseractLatticeEngine()

# Add 20 random memories
for i in range(20):
    plate = MemoryPlate(
        plate_id=f"memory_{i}",
        position=(
            random.uniform(-0.5, 0.5),  # valence
            random.uniform(-0.5, 0.5),  # arousal
            i * 0.01,                    # temporal
            random.uniform(0, 1)         # abstraction
        )
    )
    plate.set_content_from_text(f"Memory content {i}")
    engine.add_plate(plate)

# Consolidate
engine.update(steps=100)
engine.consolidate(steps=10)

# Check synchronization
print(f"Order parameter: {engine.order_parameter:.3f}")
print(f"Energy: {engine.energy:.3f}")
print(f"Synchronized: {engine.is_synchronized}")

# Retrieve
results = engine.resonance_retrieval(
    content=[0.1] * 64,  # Query embedding
    top_k=3
)

for result in results:
    print(f"[{result.combined_score:.2f}] {result.plate.raw_text}")
```

## API Reference

### MemoryPlate

| Property | Type | Description |
|----------|------|-------------|
| `plate_id` | str | Unique identifier |
| `position` | tuple | (valence, arousal, temporal, abstract) |
| `phase` | float | Current oscillation phase θ |
| `frequency` | float | Natural frequency ω |
| `content` | list | Embedding vector |
| `raw_text` | str | Original text |

### TesseractLatticeEngine

| Method | Description |
|--------|-------------|
| `add_plate(plate)` | Add memory plate |
| `remove_plate(plate_id)` | Remove plate |
| `update(steps)` | Run Kuramoto dynamics |
| `consolidate(steps)` | Apply Hebbian learning |
| `resonance_retrieval(content, ...)` | Query by resonance |
| `to_json()` | Serialize state |
| `from_json(json_str)` | Deserialize state |

### LatticeConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `K` | 2.0 | Coupling strength |
| `enable_quartet` | True | Higher-order coupling |
| `hebbian_rate` | 0.1 | Learning rate η |
| `decay_rate` | 0.01 | Weight decay λ |
| `r_threshold` | 0.95 | Convergence threshold |
