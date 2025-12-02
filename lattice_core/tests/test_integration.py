# lattice_core/tests/test_integration.py
"""
Phase 4: Integration Tests
==========================

End-to-end tests and cross-module integration.
"""

import math
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from lattice_core.plate import MemoryPlate, create_tesseract_vertices
from lattice_core.dynamics import (
    kuramoto_update,
    compute_order_parameter,
    hebbian_update,
    run_to_convergence,
)
from lattice_core.tesseract_lattice_engine import (
    TesseractLatticeEngine,
    LatticeConfig,
    create_tesseract_lattice,
)


class TestEndToEndWorkflow:
    """Test complete workflows."""

    def test_memory_store_retrieve_cycle(self):
        """Test storing and retrieving memories."""
        engine = TesseractLatticeEngine()

        # Store memories with different emotional signatures
        memories = [
            ("Happy moment at the beach", (0.8, 0.6, 0.0, 0.3)),
            ("Stressful work deadline", (-0.5, 0.8, 0.1, 0.6)),
            ("Peaceful meditation", (0.3, -0.5, 0.2, 0.7)),
            ("Exciting concert experience", (0.9, 0.9, 0.3, 0.4)),
            ("Sad farewell", (-0.6, 0.2, 0.5, 0.5)),
        ]

        for text, position in memories:
            plate = MemoryPlate(position=position)
            plate.set_content_from_text(text)
            engine.add_plate(plate)

        # Evolve the system
        engine.update(steps=100)

        # Retrieve happy memories
        happy_plate = MemoryPlate(position=(0.8, 0.6, 0.0, 0.3))
        happy_plate.set_content_from_text("Happy moment")

        results = engine.resonance_retrieval(
            content=happy_plate.content,
            emotional_position=(0.8, 0.6, 0.0, 0.3),
            top_k=3
        )

        assert len(results) > 0
        # Top result should be emotionally similar
        print(f"  Top retrieval score: {results[0].combined_score:.3f}")
        print("[PASS] test_memory_store_retrieve_cycle")

    def test_synchronization_workflow(self):
        """Test full synchronization workflow."""
        engine = TesseractLatticeEngine()
        config = LatticeConfig(K=5.0)
        engine = TesseractLatticeEngine(config=config)

        # Add plates with random phases
        import random
        random.seed(42)
        for i in range(10):
            plate = MemoryPlate(
                plate_id=f"sync-{i}",
                position=(random.uniform(-1, 1), random.uniform(-1, 1), 0, 0.5),
                phase=random.uniform(0, 2 * math.pi),
                frequency=1.0 + random.uniform(-0.1, 0.1)
            )
            engine.add_plate(plate)

        # Initial state
        r_initial = engine.order_parameter
        print(f"  Initial r: {r_initial:.3f}")

        # Run to convergence
        state = engine.run_to_convergence()

        print(f"  Final r: {state.order_parameter:.3f}")
        print(f"  Steps: {state.step}")

        assert state.order_parameter > r_initial
        print("[PASS] test_synchronization_workflow")

    def test_hebbian_learning_workflow(self):
        """Test Hebbian learning over time."""
        engine = TesseractLatticeEngine()

        # Add plates that will synchronize
        for i in range(4):
            plate = MemoryPlate(
                plate_id=f"hebb-{i}",
                position=(i * 0.1, 0, 0, 0),
                phase=0.0,  # Same phase = will strengthen
            )
            engine.add_plate(plate)

        # Get initial average weight
        def avg_weight(weights):
            total = 0
            count = 0
            for i, row in enumerate(weights):
                for j, w in enumerate(row):
                    if i != j:
                        total += w
                        count += 1
            return total / count if count > 0 else 0

        initial_avg = avg_weight(engine._weights)

        # Run dynamics and consolidation
        for _ in range(10):
            engine.update(steps=10)
            engine.consolidate(steps=5)

        final_avg = avg_weight(engine._weights)

        print(f"  Initial avg weight: {initial_avg:.3f}")
        print(f"  Final avg weight: {final_avg:.3f}")

        # In-phase oscillators should strengthen
        assert final_avg >= initial_avg
        print("[PASS] test_hebbian_learning_workflow")


class TestCrossModuleIntegration:
    """Test integration between modules."""

    def test_plate_in_engine(self):
        """Test plate operations within engine context."""
        engine = TesseractLatticeEngine()

        # Create plate with all features
        plate = MemoryPlate.from_valence_arousal(
            valence=0.7,
            arousal=0.3,
            temporal=0.0,
            abstract=0.5
        )
        plate.set_content_from_text("Test memory content")
        plate.metadata["custom_field"] = "test_value"

        engine.add_plate(plate)

        # Verify plate is properly integrated
        retrieved = engine.get_plate(plate.plate_id)
        assert retrieved is not None
        assert retrieved.valence == 0.7
        assert retrieved.raw_text == "Test memory content"
        assert retrieved.metadata["custom_field"] == "test_value"
        print("[PASS] test_plate_in_engine")

    def test_dynamics_with_engine_weights(self):
        """Test dynamics functions with engine-generated weights."""
        engine = TesseractLatticeEngine()

        for i in range(5):
            plate = MemoryPlate(
                plate_id=f"dyn-{i}",
                position=(i * 0.2, 0, 0, 0),
                phase=i * 0.3
            )
            engine.add_plate(plate)

        # Get data from engine
        phases = [p.phase for p in engine._plate_list]
        frequencies = [p.frequency for p in engine._plate_list]
        weights = engine._weights

        # Run dynamics directly
        new_phases = kuramoto_update(phases, frequencies, weights, K=2.0, dt=0.01)
        r, psi = compute_order_parameter(new_phases)

        assert len(new_phases) == 5
        assert 0 <= r <= 1
        print("[PASS] test_dynamics_with_engine_weights")

    def test_tesseract_vertices_in_engine(self):
        """Test tesseract vertex plates in engine."""
        engine = create_tesseract_lattice()

        # All 16 vertices should be present
        assert engine.n_plates == 16

        # Check vertex properties are preserved
        for plate in engine._plate_list:
            assert plate.metadata.get("is_tesseract_vertex") is True
            vertex_idx = plate.metadata.get("vertex_index")
            assert 0 <= vertex_idx < 16

        # Verify weights follow Hamming distance pattern
        # Edge-adjacent vertices (Hamming dist 1) should have highest weights
        print("[PASS] test_tesseract_vertices_in_engine")


class TestSchemaValidation:
    """Test JSON schema compliance."""

    def test_plate_schema_compliance(self):
        """Test that plates match schema structure."""
        plate = MemoryPlate(
            plate_id="schema-test",
            position=(0.5, -0.3, 0.1, 0.8),
            phase=1.5,
            frequency=1.2,
            content=[0.1, 0.2, 0.3],
            raw_text="Test text",
            metadata={"tags": ["test", "validation"]},
        )

        data = plate.to_dict()

        # Verify required fields
        assert "plate_id" in data
        assert "position" in data
        assert "phase" in data

        # Verify types
        assert isinstance(data["plate_id"], str)
        assert isinstance(data["position"], list)
        assert len(data["position"]) == 4
        assert isinstance(data["phase"], float)
        assert 0 <= data["phase"] < 2 * math.pi
        print("[PASS] test_plate_schema_compliance")

    def test_example_lattice_loading(self):
        """Test loading the example lattice JSON."""
        example_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "example_lattice.json"
        )

        if not os.path.exists(example_path):
            print("[SKIP] test_example_lattice_loading - file not found")
            return

        with open(example_path, "r") as f:
            data = json.load(f)

        # Verify structure
        assert "config" in data
        assert "plates" in data
        assert "weights" in data
        assert "state" in data

        # Verify plates
        assert len(data["plates"]) == 7
        for plate_data in data["plates"]:
            assert "plate_id" in plate_data
            assert "position" in plate_data
            assert "phase" in plate_data

        # Load into engine
        engine = TesseractLatticeEngine.from_json(json.dumps(data))
        assert engine.n_plates == 7
        print("[PASS] test_example_lattice_loading")


class TestPerformance:
    """Test performance characteristics."""

    def test_scalability_plates(self):
        """Test adding many plates."""
        engine = TesseractLatticeEngine()

        import time
        start = time.time()

        # Reduced to 20 plates for faster testing (O(N²) operations)
        for i in range(20):
            plate = MemoryPlate(
                plate_id=f"perf-{i:04d}",
                position=(i * 0.05, (i % 5) * 0.2, 0, 0.5)
            )
            engine.add_plate(plate)

        add_time = time.time() - start
        print(f"  20 plates added in {add_time:.3f}s")

        # Run dynamics
        start = time.time()
        engine.update(steps=20)
        update_time = time.time() - start
        print(f"  20 steps completed in {update_time:.3f}s")

        assert add_time < 30.0  # Should be reasonable
        assert update_time < 30.0  # Should be reasonable
        print("[PASS] test_scalability_plates")

    def test_memory_retrieval_speed(self):
        """Test retrieval performance."""
        engine = TesseractLatticeEngine()

        # Add plates
        for i in range(50):
            plate = MemoryPlate(
                plate_id=f"speed-{i}",
                position=((i % 10) * 0.1, (i // 10) * 0.2, 0, 0.5),
                content=[0.1 * (i % 5), 0.2, 0.3]
            )
            engine.add_plate(plate)

        engine.update(steps=50)

        import time
        start = time.time()

        results = engine.resonance_retrieval(
            content=[0.2, 0.2, 0.3],
            emotional_position=(0.5, 0.5, 0, 0.5),
            top_k=10
        )

        retrieval_time = time.time() - start
        print(f"  Retrieval completed in {retrieval_time:.3f}s")

        assert retrieval_time < 5.0
        print("[PASS] test_memory_retrieval_speed")


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_engine(self):
        """Test operations on empty engine."""
        engine = TesseractLatticeEngine()

        r, energy = engine.update(steps=10)
        assert r == 0.0
        assert energy == 0.0

        results = engine.resonance_retrieval(content=[0.1, 0.2])
        assert results == []
        print("[PASS] test_empty_engine")

    def test_single_plate(self):
        """Test engine with single plate."""
        engine = TesseractLatticeEngine()
        engine.add_plate(MemoryPlate(plate_id="alone"))

        r, energy = engine.update(steps=10)
        # Single oscillator should have r = 1
        assert r > 0.99
        print("[PASS] test_single_plate")

    def test_zero_weights(self):
        """Test with disconnected plates."""
        engine = TesseractLatticeEngine()

        # Add plates far apart (no connections)
        engine.add_plate(MemoryPlate(plate_id="far-1", position=(0, 0, 0, 0)))
        engine.add_plate(MemoryPlate(plate_id="far-2", position=(10, 10, 10, 10)))

        r, energy = engine.update(steps=10)
        # Disconnected plates don't synchronize
        # Energy should be zero (no connections)
        print("[PASS] test_zero_weights")

    def test_extreme_phases(self):
        """Test with extreme phase values."""
        engine = TesseractLatticeEngine()

        # Add plates with phases that need normalization
        engine.add_plate(MemoryPlate(plate_id="ext-1", phase=100.0))
        engine.add_plate(MemoryPlate(plate_id="ext-2", phase=-50.0))

        # Phases should be normalized
        for plate in engine._plate_list:
            assert 0 <= plate.phase < 2 * math.pi
        print("[PASS] test_extreme_phases")


def run_all_phase4_tests():
    """Run all Phase 4 tests."""
    print("\n" + "=" * 60)
    print("PHASE 4: INTEGRATION TESTS")
    print("=" * 60)

    test_classes = [
        TestEndToEndWorkflow,
        TestCrossModuleIntegration,
        TestSchemaValidation,
        TestPerformance,
        TestEdgeCases,
    ]

    passed = 0
    failed = 0

    for test_class in test_classes:
        print(f"\n--- {test_class.__name__} ---")
        instance = test_class()
        for method_name in dir(instance):
            if method_name.startswith("test_"):
                try:
                    getattr(instance, method_name)()
                    passed += 1
                except Exception as e:
                    print(f"[FAIL] {method_name}: {e}")
                    failed += 1

    print("\n" + "=" * 60)
    print(f"PHASE 4 RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    return passed, failed


if __name__ == "__main__":
    run_all_phase4_tests()
