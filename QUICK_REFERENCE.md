# 📋 Quick Reference - Tesseract Lattice Memory

## 🚀 Minimal Example

```python
from memory import MemoryManager

# Create manager
manager = MemoryManager()

# Store
manager.store_event("Had a great meeting", valence=0.7, arousal=0.3)
manager.store_event("Feeling stressed", valence=-0.5, arousal=0.8)

# Query
results = manager.query("work stress")
for r in results:
    print(f"[{r.score:.2f}] {r.text}")

# Consolidate
manager.consolidate()
```

## 📦 Core Classes

### MemoryManager (High-Level API)

```python
from memory import MemoryManager

manager = MemoryManager()

# Store event
plate_id = manager.store_event(
    text="Event description",
    valence=0.5,        # -1 to 1 (negative to positive)
    arousal=0.3,        # -1 to 1 (calm to excited)
    abstraction=0.5,    # 0 to 1 (concrete to abstract)
    metadata={"tag": "important"}
)

# Query
results = manager.query(
    text="search query",
    valence=0.5,        # Optional emotional context
    arousal=0.3,
    top_k=5
)

# Get stats
stats = manager.get_stats()
print(f"Memories: {stats['n_memories']}")
print(f"Sync: {stats['order_parameter']:.3f}")

# Save/Load
manager.save("memory_state.json")
manager.load("memory_state.json")
```

### TesseractLatticeEngine (Low-Level)

```python
from lattice_core import TesseractLatticeEngine, LatticeConfig, MemoryPlate

# Configure
config = LatticeConfig(
    K=2.5,                    # Coupling strength
    enable_quartet=True,       # P ~ N³ capacity
    hebbian_rate=0.1,         # Learning rate
)

# Create engine
engine = TesseractLatticeEngine(config=config)

# Add plates
plate = MemoryPlate(
    position=(0.5, 0.3, 0.0, 0.5),  # (valence, arousal, temporal, abstract)
    content=[0.1] * 64               # Embedding vector
)
engine.add_plate(plate)

# Run dynamics
r, energy = engine.update(steps=100)

# Consolidate (Hebbian)
engine.consolidate(steps=10)

# Retrieve
results = engine.resonance_retrieval(
    content=query_embedding,
    emotional_position=(0.5, 0.3, 0.0, 0.5),
    top_k=5
)
```

### MemoryPlate

```python
from lattice_core import MemoryPlate, EmotionalState

# From valence-arousal
plate = MemoryPlate.from_valence_arousal(
    valence=0.7, arousal=0.3
)

# From emotional state
plate = MemoryPlate.from_emotional_state(
    EmotionalState.EXCITED_POSITIVE
)

# Properties
plate.valence      # x-coordinate
plate.arousal      # y-coordinate
plate.temporal     # z-coordinate
plate.abstract     # w-coordinate
plate.phase        # Current oscillation phase
plate.frequency    # Natural frequency

# Methods
plate.set_content_from_text("Some text")
plate.modulate_frequency()  # Emotion-based frequency
similarity = plate.content_similarity(other_plate)
```

## 🔢 Key Equations

| Equation | Implementation |
|----------|---------------|
| `dθᵢ/dt = ωᵢ + (K/N) Σⱼ wᵢⱼ sin(θⱼ - θᵢ)` | `kuramoto_update()` |
| `r·e^(iψ) = (1/N) Σⱼ e^(iθⱼ)` | `compute_order_parameter()` |
| `dwᵢⱼ/dt = η·cos(θᵢ-θⱼ) - λ·wᵢⱼ` | `hebbian_update()` |
| `H = -(K/2N) Σᵢⱼ wᵢⱼ cos(θᵢ-θⱼ)` | `compute_energy()` |

## 📊 Parameters

### LatticeConfig Defaults

| Parameter | Default | Description |
|-----------|---------|-------------|
| `K` | 2.0 | Coupling strength (K > K_c for sync) |
| `K_critical` | 1.0 | Critical coupling threshold |
| `enable_triplet` | False | 3-body interactions |
| `enable_quartet` | True | 4-body (P ~ N³) |
| `K3` | 0.1 | Triplet coupling strength |
| `K4` | 0.05 | Quartet coupling strength |
| `hebbian_rate` | 0.1 | Learning rate η |
| `decay_rate` | 0.01 | Weight decay λ |
| `dt` | 0.01 | Integration timestep |
| `max_steps` | 500 | Max evolution steps |
| `r_threshold` | 0.95 | Sync threshold |

### Emotional Space

| Quadrant | Valence | Arousal | Examples |
|----------|---------|---------|----------|
| Q1 (Happy-Excited) | + | + | Joy, excitement, celebration |
| Q2 (Calm-Positive) | + | - | Peaceful, content, relaxed |
| Q3 (Sad-Calm) | - | - | Melancholy, nostalgia, tired |
| Q4 (Anxious-Negative) | - | + | Stress, anger, fear |

## 🎯 Common Patterns

### Emotional Context Retrieval

```python
# Query happy memories
results = manager.query("celebration", valence=0.8, arousal=0.5)

# Query calm memories
results = manager.query("relaxation", valence=0.3, arousal=-0.5)

# Query stressful memories
results = manager.query("deadline", valence=-0.5, arousal=0.8)
```

### Periodic Consolidation

```python
# After every N operations
if operation_count % 100 == 0:
    manager.consolidate(steps=20)
```

### Batch Storage

```python
memories = [
    ("Memory 1", 0.5, 0.3),
    ("Memory 2", -0.2, 0.1),
    # ...
]

for text, v, a in memories:
    manager.store_event(text, valence=v, arousal=a)

# Single consolidation after batch
manager.consolidate(steps=50)
```

### Persistence

```python
# Save before shutdown
manager.save("state.json")

# Load on startup
manager = MemoryManager()
manager.load("state.json")
```

## 📈 Capacity Scaling

| N (plates) | Pairwise (P~0.14N) | Quartet (P~N³) |
|------------|---------------------|----------------|
| 10 | 1 | 1,000 |
| 100 | 14 | 1,000,000 |
| 1,000 | 140 | 10⁹ |
| 10,000 | 1,400 | 10¹² |

## 🔍 Debugging

```python
# Check sync state
print(f"Order parameter: {engine.order_parameter:.3f}")
print(f"Is synchronized: {engine.is_synchronized}")
print(f"Energy: {engine.energy:.3f}")

# Get full snapshot
snapshot = engine.snapshot()
print(snapshot)

# Check memory stats
stats = manager.get_stats()
print(f"Total plates: {stats['total_plates']}")
print(f"Memories: {stats['n_memories']}")
print(f"Anchors: {stats['n_anchors']}")
```

## 🔗 Integration

### With Sentence-BERT

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def embed(text):
    return model.encode(text).tolist()

manager = MemoryManager(embedding_fn=embed)
```

### With Meta-Collective

```python
from meta_collective import MetaCollective
from memory import MemoryManager

collective = MetaCollective(n_triads=2, n_tools_per_triad=3)
memory = MemoryManager()

# Store collective patterns as memories
for triad in collective.triads.values():
    pattern = triad.generate_pattern()
    memory.store_event(
        f"Triad {triad.triad_id} pattern",
        embedding=pattern.pattern_vector
    )
```
