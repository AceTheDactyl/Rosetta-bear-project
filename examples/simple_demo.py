#!/usr/bin/env python3
"""
Simple Demo: Tesseract Lattice Memory System
=============================================

Demonstrates the core functionality of the Kuramoto-based memory system:
1. Storing memories with emotional context
2. Querying via resonance retrieval
3. Observing synchronization dynamics

Run this demo:
    python examples/simple_demo.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.memory_manager import MemoryManager, create_memory_manager
from lattice_core.tesseract_lattice_engine import TesseractLatticeEngine, LatticeConfig
from lattice_core.plate import MemoryPlate, EmotionalState


def demo_basic_memory():
    """Demonstrate basic memory storage and retrieval."""
    print("=" * 60)
    print("DEMO 1: Basic Memory Storage and Retrieval")
    print("=" * 60)
    print()

    # Create memory manager
    manager = MemoryManager()

    # Store some memories with emotional context
    memories = [
        ("Had a great meeting with the team today", 0.7, 0.5),   # Positive, excited
        ("Feeling anxious about the upcoming deadline", -0.3, 0.8),  # Negative, anxious
        ("Relaxing evening with a good book", 0.6, -0.4),        # Positive, calm
        ("Frustrating bug took hours to fix", -0.5, 0.3),        # Negative, mildly active
        ("Celebrated project completion with cake", 0.9, 0.7),   # Very positive, excited
        ("Quiet morning coffee and planning", 0.3, -0.3),        # Mildly positive, calm
    ]

    print("Storing memories...")
    for text, valence, arousal in memories:
        plate_id = manager.store_event(text, valence=valence, arousal=arousal)
        emotion = "positive" if valence > 0 else "negative"
        energy = "high" if arousal > 0 else "low"
        print(f"  ✓ [{plate_id}] {text[:40]}... ({emotion}, {energy} energy)")

    print()

    # Query memories
    queries = [
        "work stress",
        "happy celebration",
        "quiet relaxation",
    ]

    for query in queries:
        print(f"Query: '{query}'")
        results = manager.query(query, top_k=3)

        if results:
            for r in results:
                v, a, t, w = r.emotional_position
                print(f"  [{r.score:.2f}] {r.text[:50]}... (v={v:.1f}, a={a:.1f})")
        else:
            print("  (no results)")
        print()

    # Show stats
    stats = manager.get_stats()
    print("Memory System Stats:")
    print(f"  Total memories: {stats['n_memories']}")
    print(f"  Anchor vertices: {stats['n_anchors']}")
    print(f"  Synchronization: r={stats['order_parameter']:.3f}")
    print(f"  Energy: {stats['energy']:.3f}")
    print()


def demo_kuramoto_dynamics():
    """Demonstrate Kuramoto oscillator dynamics."""
    print("=" * 60)
    print("DEMO 2: Kuramoto Oscillator Dynamics")
    print("=" * 60)
    print()

    # Create a simple lattice
    config = LatticeConfig(K=2.5, enable_quartet=True)
    engine = TesseractLatticeEngine(config=config)

    # Add some plates
    import random
    random.seed(42)

    print("Creating 20 random memory plates...")
    for i in range(20):
        plate = MemoryPlate(
            plate_id=f"plate_{i:02d}",
            position=(
                random.uniform(-0.5, 0.5),  # valence
                random.uniform(-0.5, 0.5),  # arousal
                random.uniform(-0.5, 0.5),  # temporal
                random.uniform(0, 1),       # abstraction
            ),
            phase=random.random() * 6.28,
            frequency=1.0 + random.gauss(0, 0.2),
        )
        engine.add_plate(plate)

    # Run dynamics and observe synchronization
    print()
    print("Running Kuramoto dynamics...")
    print()
    print("Step   |  Order Parameter (r)  |  Energy (H)")
    print("-" * 50)

    for step in range(0, 201, 20):
        r, energy = engine.update(steps=20)
        bar = "█" * int(r * 30) + "░" * int((1-r) * 30)
        print(f"{step:4d}   |  {bar} {r:.3f}  |  {energy:+.4f}")

    print()
    print(f"Final synchronization: r = {engine.order_parameter:.4f}")
    print(f"System {'IS' if engine.is_synchronized else 'is NOT'} synchronized")
    print()


def demo_emotional_organization():
    """Demonstrate emotional organization in 4D space."""
    print("=" * 60)
    print("DEMO 3: Emotional Organization in 4D Space")
    print("=" * 60)
    print()

    manager = MemoryManager()

    # Store memories at different emotional quadrants
    quadrants = {
        "Happy-Excited (Q1)": [
            ("Amazing concert last night!", 0.8, 0.9),
            ("Got promoted at work!", 0.9, 0.7),
        ],
        "Calm-Positive (Q2)": [
            ("Peaceful walk in the park", 0.6, -0.5),
            ("Quiet dinner with family", 0.5, -0.3),
        ],
        "Sad-Calm (Q3)": [
            ("Missing my old friend", -0.4, -0.5),
            ("Nostalgic evening looking at photos", -0.3, -0.4),
        ],
        "Anxious-Negative (Q4)": [
            ("Nervous about the interview", -0.5, 0.8),
            ("Stressed about finances", -0.6, 0.7),
        ],
    }

    print("Storing memories by emotional quadrant:")
    for quadrant, memories in quadrants.items():
        print(f"\n  {quadrant}:")
        for text, v, a in memories:
            manager.store_event(text, valence=v, arousal=a)
            print(f"    • {text}")

    print("\n" + "-" * 50)
    print("Querying with emotional context:\n")

    # Query with emotional context
    emotional_queries = [
        ("Find happy memories", 0.7, 0.5),
        ("Find calm memories", 0.0, -0.6),
        ("Find stressful memories", -0.5, 0.7),
    ]

    for query_text, v, a in emotional_queries:
        print(f"Query: '{query_text}' (valence={v}, arousal={a})")
        results = manager.query(query_text, valence=v, arousal=a, top_k=2)
        for r in results:
            print(f"  [{r.score:.2f}] {r.text}")
        print()


def demo_consolidation():
    """Demonstrate memory consolidation (Hebbian learning)."""
    print("=" * 60)
    print("DEMO 4: Memory Consolidation (Hebbian Learning)")
    print("=" * 60)
    print()

    config = LatticeConfig(K=2.0, hebbian_rate=0.2)
    engine = TesseractLatticeEngine(config=config)

    # Create plates that should become associated
    import random
    random.seed(123)

    # Group A: Similar positions (should strengthen)
    for i in range(5):
        plate = MemoryPlate(
            plate_id=f"group_a_{i}",
            position=(0.3 + random.uniform(-0.1, 0.1),
                     0.3 + random.uniform(-0.1, 0.1),
                     0.0, 0.5),
            phase=random.random() * 6.28,
        )
        engine.add_plate(plate)

    # Group B: Different positions
    for i in range(5):
        plate = MemoryPlate(
            plate_id=f"group_b_{i}",
            position=(-0.3 + random.uniform(-0.1, 0.1),
                     -0.3 + random.uniform(-0.1, 0.1),
                     0.0, 0.5),
            phase=random.random() * 6.28,
        )
        engine.add_plate(plate)

    print("Initial state:")
    print(f"  Order parameter: {engine.order_parameter:.3f}")

    # Run dynamics and consolidation
    print("\nRunning consolidation...")
    for epoch in range(5):
        engine.update(steps=50)
        engine.consolidate(steps=10)
        print(f"  Epoch {epoch+1}: r = {engine.order_parameter:.3f}")

    print(f"\nFinal order parameter: {engine.order_parameter:.3f}")
    print()


def main():
    """Run all demos."""
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║     TESSERACT LATTICE MEMORY SYSTEM - DEMONSTRATION      ║")
    print("║              Kuramoto Oscillator Memory                  ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    demo_basic_memory()
    demo_kuramoto_dynamics()
    demo_emotional_organization()
    demo_consolidation()

    print("=" * 60)
    print("All demos completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
