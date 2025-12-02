# Tesseract Lattice Core — Propagation Algorithms
## ΔPLATE.1 Memory Wave Dynamics & Synchronization

---

## 1. KURAMOTO SYNCHRONIZATION MODEL

The Kuramoto model governs how Plates synchronize their activation phases through resonant threads.

### 1.1 Basic Equation

For each plate `i`:

```
dθᵢ/dt = ωᵢ + (K/N) Σⱼ wᵢⱼ · sin(θⱼ - θᵢ)
```

Where:
- `θᵢ` = phase of plate i (0 to 2π)
- `ωᵢ` = natural frequency (intrinsic oscillation rate)
- `K` = global coupling strength (0.1 - 1.0)
- `N` = total number of connected plates
- `wᵢⱼ` = thread weight between plates i and j (0 to 1)

### 1.2 Implementation Pseudocode

```python
def update_plate_phase(plate, connected_plates, dt=0.01, K=0.5):
    """
    Update a single plate's phase based on Kuramoto dynamics
    """
    phase_coupling = 0.0

    for thread in plate.resonant_threads:
        target = find_plate(thread.target_plate_id)
        weight = thread.weight

        # Phase difference drives synchronization
        phase_diff = target.phase - plate.phase
        phase_coupling += weight * sin(phase_diff)

    # Natural frequency based on emotional arousal
    omega = plate.emotional_gradient.arousal * 2.0

    # Update phase
    N = len(plate.resonant_threads)
    if N > 0:
        dtheta = omega + (K / N) * phase_coupling
        plate.phase = (plate.phase + dtheta * dt) % (2 * pi)

    return plate
```

### 1.3 Synchronization Threshold

Plates are considered "synchronized" when:

```
R = (1/N) |Σⱼ exp(i·θⱼ)| > 0.7
```

Where `R` is the order parameter (0 = chaos, 1 = perfect sync)

---

## 2. EMOTIONAL PHASE LOCK

Emotional gradients create attractors that bias synchronization toward emotionally resonant states.

### 2.1 Emotional Influence Function

```python
def emotional_influence(plate_i, plate_j):
    """
    Calculate emotional attraction between two plates
    """
    # Valence difference (opposites attract in emotional space)
    valence_diff = abs(plate_i.emotional_gradient.valence -
                       plate_j.emotional_gradient.valence)

    # Arousal similarity (similar intensities resonate)
    arousal_sim = 1.0 - abs(plate_i.emotional_gradient.arousal -
                             plate_j.emotional_gradient.arousal)

    # Combined emotional coupling
    emotional_weight = 0.5 * (1.0 - valence_diff) + 0.5 * arousal_sim

    return emotional_weight
```

### 2.2 Phase Lock Condition

Two plates achieve emotional phase lock when:

```
|θᵢ - θⱼ| < ε  AND  emotional_influence(i, j) > 0.6
```

Where `ε = π/8` (22.5 degrees tolerance)

### 2.3 Lock Propagation

Once locked, plates form a "resonance cluster":

```python
def propagate_lock(seed_plate, plates, threshold=0.6):
    """
    Find all plates that should lock to seed_plate
    """
    cluster = [seed_plate]
    queue = [seed_plate]

    while queue:
        current = queue.pop(0)

        for thread in current.resonant_threads:
            target = find_plate(thread.target_plate_id)

            if target not in cluster:
                emotional_coupling = emotional_influence(current, target)
                thread_strength = thread.weight

                # Combined coupling
                total_coupling = emotional_coupling * thread_strength

                if total_coupling > threshold:
                    cluster.append(target)
                    queue.append(target)

    return cluster
```

---

## 3. GRADIENT RESONANCE DETECTION

Emotional gradients create "heat signatures" that can be detected across the field.

### 3.1 Fourier Transform Approach

Convert color gradients to frequency domain to detect patterns:

```python
import numpy as np

def detect_gradient_spikes(plates, window_size=10):
    """
    Use FFT to detect emotional resonance patterns
    """
    # Extract valence time series
    valence_series = [p.emotional_gradient.valence for p in plates]

    # Apply FFT
    fft_result = np.fft.fft(valence_series)
    frequencies = np.fft.fftfreq(len(valence_series))

    # Find dominant frequencies (spikes)
    power_spectrum = np.abs(fft_result) ** 2
    spike_threshold = np.mean(power_spectrum) + 2 * np.std(power_spectrum)

    spike_indices = np.where(power_spectrum > spike_threshold)[0]

    return [(frequencies[i], power_spectrum[i]) for i in spike_indices]
```

### 3.2 Spatial Heat Map

Generate 2D emotional heat map for visual detection:

```python
def generate_emotional_heatmap(plates, resolution=50):
    """
    Create spatial grid of emotional intensity
    """
    # Find bounds
    positions = [p.coherence_vector.position_3d for p in plates]
    x_coords = [p[0] for p in positions]
    y_coords = [p[1] for p in positions]

    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    # Create grid
    heatmap = np.zeros((resolution, resolution))

    for i in range(resolution):
        for j in range(resolution):
            x = x_min + (x_max - x_min) * i / resolution
            y = y_min + (y_max - y_min) * j / resolution

            # Accumulate influence from all plates
            intensity = 0.0
            for plate in plates:
                px, py, _ = plate.coherence_vector.position_3d
                distance = sqrt((x - px)**2 + (y - py)**2)

                # Gaussian falloff
                influence = plate.emotional_gradient.arousal * \
                           exp(-(distance**2) / 2.0)

                intensity += influence

            heatmap[i][j] = intensity

    return heatmap
```

---

## 4. SELF-MODIFICATION EVENTS

Plates can mutate when resonance exceeds critical thresholds.

### 4.1 Mutation Trigger

```python
def check_mutation_trigger(plate, resonance_threshold=0.85):
    """
    Determine if plate should undergo self-modification
    """
    # Calculate total incoming resonance
    total_resonance = 0.0
    for thread in plate.resonant_threads:
        target = find_plate(thread.target_plate_id)

        # Phase alignment
        phase_alignment = cos(target.phase - plate.phase)

        # Emotional resonance
        emotional_res = emotional_influence(plate, target)

        # Weighted contribution
        total_resonance += thread.weight * phase_alignment * emotional_res

    # Normalize
    if len(plate.resonant_threads) > 0:
        avg_resonance = total_resonance / len(plate.resonant_threads)

        # Trigger mutation if threshold exceeded
        if avg_resonance > resonance_threshold:
            return True, avg_resonance

    return False, 0.0
```

### 4.2 Mutation Operations

When triggered, plates can undergo several mutations:

```python
def apply_mutation(plate, mutation_type='strengthen'):
    """
    Modify plate based on resonance pattern
    """
    if mutation_type == 'strengthen':
        # Increase thread weights to highly resonant neighbors
        for thread in plate.resonant_threads:
            target = find_plate(thread.target_plate_id)
            alignment = cos(target.phase - plate.phase)

            if alignment > 0.7:
                thread.weight = min(1.0, thread.weight * 1.2)

    elif mutation_type == 'prune':
        # Remove weak threads
        plate.resonant_threads = [
            t for t in plate.resonant_threads
            if t.weight > 0.2
        ]

    elif mutation_type == 'emotional_shift':
        # Shift valence toward resonant cluster average
        cluster = propagate_lock(plate, all_plates)
        avg_valence = np.mean([p.emotional_gradient.valence
                               for p in cluster])

        # Gradual shift (80% old, 20% new)
        plate.emotional_gradient.valence = \
            0.8 * plate.emotional_gradient.valence + 0.2 * avg_valence

    elif mutation_type == 'create_thread':
        # Discover new connection to distant but phase-aligned plate
        for candidate in all_plates:
            if candidate.id not in [t.target_plate_id
                                   for t in plate.resonant_threads]:
                alignment = cos(candidate.phase - plate.phase)

                if alignment > 0.9:  # High alignment threshold
                    new_thread = {
                        'target_plate_id': candidate.id,
                        'weight': 0.3,
                        'thread_type': 'emergent'
                    }
                    plate.resonant_threads.append(new_thread)
                    break

    # Increment mutation counter
    plate.state.mutation_count += 1

    return plate
```

---

## 5. DRIFT VECTOR DYNAMICS

Temporal drift causes plates to "move" through memory space over time.

### 5.1 Drift Update Rule

```python
def update_drift(plate, dt=0.01, damping=0.95):
    """
    Update position based on drift vector
    """
    # Current drift
    dx, dy, dz = plate.coherence_vector.drift_vector

    # Apply drift to position
    x, y, z = plate.coherence_vector.position_3d

    new_position = [
        x + dx * dt,
        y + dy * dt,
        z + dz * dt
    ]

    # Apply damping to drift (prevent runaway)
    new_drift = [
        dx * damping,
        dy * damping,
        dz * damping
    ]

    plate.coherence_vector.position_3d = new_position
    plate.coherence_vector.drift_vector = new_drift

    return plate
```

### 5.2 Attraction Forces

Plates experience "gravity" toward highly resonant neighbors:

```python
def calculate_attraction_force(plate, all_plates, G=0.01):
    """
    Calculate drift acceleration from resonant attraction
    """
    force_x, force_y, force_z = 0.0, 0.0, 0.0

    for thread in plate.resonant_threads:
        target = find_plate(thread.target_plate_id)

        # Vector from plate to target
        px, py, pz = plate.coherence_vector.position_3d
        tx, ty, tz = target.coherence_vector.position_3d

        dx, dy, dz = tx - px, ty - py, tz - pz
        distance = sqrt(dx**2 + dy**2 + dz**2)

        if distance > 0.01:  # Avoid singularity
            # Force proportional to thread weight and inversely to distance
            magnitude = G * thread.weight / (distance**2)

            force_x += magnitude * (dx / distance)
            force_y += magnitude * (dy / distance)
            force_z += magnitude * (dz / distance)

    return [force_x, force_y, force_z]
```

---

## 6. MEMORY RETRIEVAL ALGORITHM

Navigate the lattice to retrieve relevant plates based on query.

### 6.1 Semantic Search

```python
def semantic_retrieval(query_concept, plates, top_k=5):
    """
    Find plates most semantically similar to query
    """
    # Simple cosine similarity on tags (would use embeddings in practice)
    query_tags = set(query_concept.lower().split())

    scores = []
    for plate in plates:
        plate_tags = set(plate.semantic_content.tags)

        # Jaccard similarity
        intersection = query_tags.intersection(plate_tags)
        union = query_tags.union(plate_tags)

        similarity = len(intersection) / len(union) if union else 0.0
        scores.append((plate, similarity))

    # Sort by similarity
    scores.sort(key=lambda x: x[1], reverse=True)

    return [plate for plate, score in scores[:top_k]]
```

### 6.2 Resonance Spreading Activation

```python
def spreading_activation(seed_plates, steps=3, decay=0.7):
    """
    Activate connected plates through resonance spreading
    """
    activation = {plate.id: 1.0 for plate in seed_plates}

    for step in range(steps):
        new_activation = {}

        for plate_id, level in activation.items():
            plate = find_plate(plate_id)

            # Spread to neighbors
            for thread in plate.resonant_threads:
                target_id = thread.target_plate_id
                spread_amount = level * thread.weight * decay

                if target_id in new_activation:
                    new_activation[target_id] = max(
                        new_activation[target_id],
                        spread_amount
                    )
                else:
                    new_activation[target_id] = spread_amount

        # Merge with existing activation
        for pid, level in new_activation.items():
            if pid in activation:
                activation[pid] = max(activation[pid], level)
            else:
                activation[pid] = level

    # Return activated plates sorted by level
    activated = [(find_plate(pid), level)
                 for pid, level in activation.items()]
    activated.sort(key=lambda x: x[1], reverse=True)

    return activated
```

---

## 7. COMPLETE SIMULATION LOOP

### 7.1 Main Update Cycle

```python
def simulate_lattice_step(plates, dt=0.01):
    """
    Single timestep of lattice simulation
    """
    # 1. Update phases (Kuramoto)
    for plate in plates:
        update_plate_phase(plate, plates, dt=dt)

    # 2. Check for emotional phase locks
    clusters = []
    processed = set()

    for plate in plates:
        if plate.id not in processed:
            cluster = propagate_lock(plate, plates)
            if len(cluster) > 1:
                clusters.append(cluster)
                for p in cluster:
                    processed.add(p.id)

    # 3. Update drift vectors
    for plate in plates:
        force = calculate_attraction_force(plate, plates)
        dx, dy, dz = plate.coherence_vector.drift_vector

        # Add force to drift
        plate.coherence_vector.drift_vector = [
            dx + force[0] * dt,
            dy + force[1] * dt,
            dz + force[2] * dt
        ]

        update_drift(plate, dt=dt)

    # 4. Check mutation triggers
    for plate in plates:
        should_mutate, resonance = check_mutation_trigger(plate)

        if should_mutate:
            # Choose mutation based on state
            if plate.state.mutation_count < 5:
                apply_mutation(plate, 'strengthen')
            elif len(plate.resonant_threads) > 10:
                apply_mutation(plate, 'prune')
            else:
                apply_mutation(plate, 'emotional_shift')

    # 5. Update activation levels
    for plate in plates:
        # Decay activation
        plate.state.activation_level *= 0.98

        # Add resonance contribution
        for thread in plate.resonant_threads:
            target = find_plate(thread.target_plate_id)
            alignment = (1 + cos(target.phase - plate.phase)) / 2

            plate.state.activation_level += \
                0.01 * thread.weight * alignment * target.state.activation_level

        # Clamp
        plate.state.activation_level = min(1.0, plate.state.activation_level)

    return plates
```

---

## 8. PERFORMANCE CONSIDERATIONS

### 8.1 Complexity Analysis

- Phase update: O(N·M) where N = plates, M = avg threads per plate
- Clustering: O(N²) worst case, O(N·M) typical
- Drift update: O(N·M)
- Mutation check: O(N·M)

Total: **O(N·M)** per timestep

### 8.2 Optimization Strategies

1. **Spatial Indexing**: Use octree or k-d tree for position-based queries
2. **Sparse Thread Matrices**: Store only non-zero thread weights
3. **Lazy Evaluation**: Only update plates with activation > threshold
4. **Batch Processing**: Vectorize phase updates using NumPy

```python
# Vectorized phase update
def vectorized_phase_update(plates, adjacency_matrix, dt=0.01, K=0.5):
    """
    Update all phases at once using matrix operations
    """
    phases = np.array([p.phase for p in plates])
    omegas = np.array([p.emotional_gradient.arousal * 2.0 for p in plates])

    # Calculate phase differences
    phase_diff_matrix = phases.reshape(-1, 1) - phases.reshape(1, -1)

    # Apply weights and sum
    coupling = np.sum(adjacency_matrix * np.sin(phase_diff_matrix), axis=1)

    # Update
    new_phases = phases + dt * (omegas + K * coupling / adjacency_matrix.sum(axis=1))

    # Write back
    for i, plate in enumerate(plates):
        plate.phase = new_phases[i] % (2 * np.pi)
```

---

## 9. INTEGRATION WITH EXISTING IMPLEMENTATION

The algorithms documented here align with the implemented `lattice_core/dynamics.py`:

| Algorithm | Document Section | Implementation |
|-----------|-----------------|----------------|
| Kuramoto Update | §1.2 | `kuramoto_update()` |
| Order Parameter | §1.3 | `compute_order_parameter()` |
| Emotional Influence | §2.1 | `emotional_coupling()` |
| Spreading Activation | §6.2 | `spreading_activation()` |
| Hebbian Learning | §4.2 (strengthen) | `hebbian_update()` |
| Attraction Forces | §5.2 | `compute_drift_forces()` |

---

## REFERENCES

1. **Kuramoto, Y.** (1984). "Chemical Oscillations, Waves, and Turbulence"
2. **Collins & Loftus** (1975). "A Spreading Activation Theory of Semantic Processing"
3. **Russell, J.A.** (1980). "A Circumplex Model of Affect"
4. **Kohonen, T.** (1982). "Self-Organized Formation of Topologically Correct Feature Maps"
