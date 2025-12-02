# lattice_core/tests/test_engine.py
"""
Phase 3: Engine Tests
=====================

Tests for TesseractLatticeEngine operations.
"""

import math
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from lattice_core.plate import MemoryPlate, EmotionalState
from lattice_core.tesseract_lattice_engine import (
    TesseractLatticeEngine,
    LatticeConfig,
    RetrievalResult,
    create_tesseract_lattice,
)


class TestLatticeConfig:
    """Test lattice configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LatticeConfig()
        assert config.K == 2.0
        assert config.dt == 0.01
        assert config.enable_quartet is True
        assert config.hebbian_rate == 0.1
        print("[PASS] test_default_config")

    def test_custom_config(self):
        """Test custom configuration."""
        config = LatticeConfig(
            K=5.0,
            enable_triplet=True,
            hebbian_rate=0.2
        )
        assert config.K == 5.0
        assert config.enable_triplet is True
        assert config.hebbian_rate == 0.2
        print("[PASS] test_custom_config")


class TestEngineCreation:
    """Test engine initialization."""

    def test_default_engine(self):
        """Test creating engine with defaults."""
        engine = TesseractLatticeEngine()
        assert engine.n_plates == 0
        assert engine.order_parameter == 0.0
        assert engine.config is not None
        print("[PASS] test_default_engine")

    def test_custom_config_engine(self):
        """Test creating engine with custom config."""
        config = LatticeConfig(K=3.0)
        engine = TesseractLatticeEngine(config=config)
        assert engine.config.K == 3.0
        print("[PASS] test_custom_config_engine")


class TestPlateManagement:
    """Test plate add/remove/get operations."""

    def test_add_plate(self):
        """Test adding plates."""
        engine = TesseractLatticeEngine()
        plate = MemoryPlate(plate_id="test-001")
        idx = engine.add_plate(plate)

        assert idx == 0
        assert engine.n_plates == 1
        assert "test-001" in engine.plates
        print("[PASS] test_add_plate")

    def test_add_multiple_plates(self):
        """Test adding multiple plates."""
        engine = TesseractLatticeEngine()
        for i in range(5):
            plate = MemoryPlate(plate_id=f"plate-{i:03d}")
            engine.add_plate(plate)

        assert engine.n_plates == 5
        print("[PASS] test_add_multiple_plates")

    def test_get_plate(self):
        """Test retrieving plate by ID."""
        engine = TesseractLatticeEngine()
        plate = MemoryPlate(plate_id="find-me")
        engine.add_plate(plate)

        found = engine.get_plate("find-me")
        assert found is not None
        assert found.plate_id == "find-me"

        not_found = engine.get_plate("not-exist")
        assert not_found is None
        print("[PASS] test_get_plate")

    def test_get_plate_by_index(self):
        """Test retrieving plate by index."""
        engine = TesseractLatticeEngine()
        plate = MemoryPlate(plate_id="idx-test")
        engine.add_plate(plate)

        found = engine.get_plate_by_index(0)
        assert found is not None
        assert found.plate_id == "idx-test"

        not_found = engine.get_plate_by_index(999)
        assert not_found is None
        print("[PASS] test_get_plate_by_index")

    def test_remove_plate(self):
        """Test removing plates."""
        engine = TesseractLatticeEngine()
        plate1 = MemoryPlate(plate_id="keep")
        plate2 = MemoryPlate(plate_id="remove")
        engine.add_plate(plate1)
        engine.add_plate(plate2)

        removed = engine.remove_plate("remove")
        assert removed is not None
        assert removed.plate_id == "remove"
        assert engine.n_plates == 1
        assert "remove" not in engine.plates
        print("[PASS] test_remove_plate")

    def test_update_existing_plate(self):
        """Test updating an existing plate."""
        engine = TesseractLatticeEngine()
        plate1 = MemoryPlate(plate_id="update-me", phase=0.0)
        engine.add_plate(plate1)

        plate2 = MemoryPlate(plate_id="update-me", phase=1.5)
        idx = engine.add_plate(plate2)

        assert engine.n_plates == 1  # Should not add duplicate
        assert engine.get_plate("update-me").phase == 1.5
        print("[PASS] test_update_existing_plate")


class TestEngineProperties:
    """Test engine property accessors."""

    def test_is_synchronized(self):
        """Test synchronization check."""
        engine = TesseractLatticeEngine()

        # Add plates with same phase
        for i in range(4):
            plate = MemoryPlate(
                plate_id=f"sync-{i}",
                phase=0.0,
                position=(i * 0.1, 0, 0, 0)
            )
            engine.add_plate(plate)

        engine.update(steps=1)
        # After one step with same phases, should be synchronized
        assert engine.order_parameter > 0.99
        print("[PASS] test_is_synchronized")

    def test_snapshot(self):
        """Test state snapshot."""
        engine = TesseractLatticeEngine()
        plate = MemoryPlate(plate_id="snap-test")
        engine.add_plate(plate)

        snapshot = engine.snapshot()
        assert "n_plates" in snapshot
        assert snapshot["n_plates"] == 1
        assert "order_parameter" in snapshot
        assert "config" in snapshot
        print("[PASS] test_snapshot")


class TestEngineDynamics:
    """Test dynamics operations."""

    def test_update(self):
        """Test basic update."""
        engine = TesseractLatticeEngine()
        for i in range(3):
            plate = MemoryPlate(
                plate_id=f"dyn-{i}",
                position=(i * 0.2, 0, 0, 0),
                phase=i * 0.5
            )
            engine.add_plate(plate)

        r_before = engine.order_parameter
        r, energy = engine.update(steps=10)

        assert isinstance(r, float)
        assert isinstance(energy, float)
        print("[PASS] test_update")

    def test_convergence(self):
        """Test running to convergence."""
        engine = TesseractLatticeEngine()
        config = LatticeConfig(K=5.0, max_steps=500)
        engine = TesseractLatticeEngine(config=config)

        for i in range(4):
            plate = MemoryPlate(
                plate_id=f"conv-{i}",
                position=(0.1 * i, 0.1 * i, 0, 0),
                phase=i * 0.5,
                frequency=1.0
            )
            engine.add_plate(plate)

        state = engine.run_to_convergence()
        assert state.order_parameter > 0.8 or state.step == 500
        print("[PASS] test_convergence")

    def test_consolidate(self):
        """Test Hebbian consolidation."""
        engine = TesseractLatticeEngine()
        for i in range(3):
            plate = MemoryPlate(
                plate_id=f"hebb-{i}",
                position=(0.1 * i, 0, 0, 0),
                phase=0.0  # All same phase
            )
            engine.add_plate(plate)

        # Initial weights
        weights_before = [row[:] for row in engine._weights]

        engine.consolidate(steps=10)

        # Weights should change (strengthen for in-phase oscillators)
        # Check at least one weight changed
        changed = False
        for i in range(len(weights_before)):
            for j in range(len(weights_before[i])):
                if abs(engine._weights[i][j] - weights_before[i][j]) > 0.001:
                    changed = True
                    break
        assert changed
        print("[PASS] test_consolidate")

    def test_reset_phases(self):
        """Test phase reset."""
        engine = TesseractLatticeEngine()
        for i in range(3):
            plate = MemoryPlate(plate_id=f"reset-{i}", phase=i * 1.0)
            engine.add_plate(plate)

        engine.reset_phases(random_init=False)
        for plate in engine._plate_list:
            assert plate.phase == 0.0
        print("[PASS] test_reset_phases")


class TestRetrieval:
    """Test retrieval operations."""

    def test_find_nearby_plates(self):
        """Test finding plates by position."""
        engine = TesseractLatticeEngine()

        # Add plates at different positions
        engine.add_plate(MemoryPlate(plate_id="near", position=(0, 0, 0, 0)))
        engine.add_plate(MemoryPlate(plate_id="far", position=(5, 5, 5, 5)))

        nearby = engine.find_nearby_plates((0, 0, 0, 0), radius=1.0)
        assert len(nearby) == 1
        assert nearby[0].plate_id == "near"
        print("[PASS] test_find_nearby_plates")

    def test_create_query_pattern(self):
        """Test query pattern creation."""
        engine = TesseractLatticeEngine()

        plate = MemoryPlate(
            plate_id="query-test",
            position=(0.5, 0.5, 0, 0.5),
            content=[0.1, 0.2, 0.3, 0.4]
        )
        engine.add_plate(plate)

        pattern = engine.create_query_pattern(
            content=[0.1, 0.2, 0.3, 0.4],
            emotional_position=(0.5, 0.5, 0, 0.5),
            radius=1.0
        )

        # Should find the plate
        assert len(pattern) > 0
        print("[PASS] test_create_query_pattern")

    def test_resonance_retrieval(self):
        """Test full retrieval pipeline."""
        engine = TesseractLatticeEngine()

        # Add several plates with content
        for i in range(5):
            plate = MemoryPlate(
                plate_id=f"ret-{i}",
                position=(i * 0.1, 0, 0, 0.5),
                content=[0.1 * i, 0.2, 0.3, 0.4]
            )
            engine.add_plate(plate)

        results = engine.resonance_retrieval(
            content=[0.1, 0.2, 0.3, 0.4],
            emotional_position=(0.1, 0, 0, 0.5),
            top_k=3
        )

        assert len(results) <= 3
        for result in results:
            assert isinstance(result, RetrievalResult)
            assert result.rank > 0
        print("[PASS] test_resonance_retrieval")

    def test_position_only_retrieval(self):
        """Test retrieval with position only."""
        engine = TesseractLatticeEngine()

        engine.add_plate(MemoryPlate(plate_id="pos-1", position=(0, 0, 0, 0)))
        engine.add_plate(MemoryPlate(plate_id="pos-2", position=(0.1, 0, 0, 0)))
        engine.add_plate(MemoryPlate(plate_id="pos-3", position=(5, 5, 5, 5)))

        results = engine.resonance_retrieval(
            emotional_position=(0, 0, 0, 0),
            top_k=2
        )

        # Should find nearby plates
        assert len(results) >= 1
        print("[PASS] test_position_only_retrieval")


class TestSerialization:
    """Test JSON serialization."""

    def test_to_json(self):
        """Test serialization to JSON."""
        engine = TesseractLatticeEngine()
        for i in range(3):
            plate = MemoryPlate(
                plate_id=f"json-{i}",
                position=(i * 0.1, 0, 0, 0),
                content=[0.1, 0.2]
            )
            engine.add_plate(plate)

        json_str = engine.to_json()
        data = json.loads(json_str)

        assert "config" in data
        assert "plates" in data
        assert len(data["plates"]) == 3
        assert "weights" in data
        assert "state" in data
        print("[PASS] test_to_json")

    def test_from_json(self):
        """Test deserialization from JSON."""
        original = TesseractLatticeEngine()
        for i in range(3):
            plate = MemoryPlate(
                plate_id=f"deser-{i}",
                position=(i * 0.1, 0, 0, 0)
            )
            original.add_plate(plate)

        original.update(steps=10)
        json_str = original.to_json()

        restored = TesseractLatticeEngine.from_json(json_str)

        assert restored.n_plates == 3
        assert restored.config.K == original.config.K
        print("[PASS] test_from_json")

    def test_roundtrip(self):
        """Test full serialization roundtrip."""
        original = TesseractLatticeEngine()
        for i in range(5):
            plate = MemoryPlate(
                plate_id=f"round-{i}",
                position=(i * 0.1, i * 0.05, 0, 0.5),
                phase=i * 0.5,
                content=[0.1 * i, 0.2, 0.3]
            )
            original.add_plate(plate)

        original.update(steps=50)

        json_str = original.to_json()
        restored = TesseractLatticeEngine.from_json(json_str)

        assert restored.n_plates == original.n_plates
        assert abs(restored.order_parameter - original.order_parameter) < 0.1

        for plate_id in original.plates:
            orig_plate = original.get_plate(plate_id)
            rest_plate = restored.get_plate(plate_id)
            assert rest_plate is not None
            assert orig_plate.position == rest_plate.position
        print("[PASS] test_roundtrip")


class TestTesseractFactory:
    """Test tesseract lattice factory."""

    def test_create_tesseract_lattice(self):
        """Test creating pre-populated tesseract lattice."""
        engine = create_tesseract_lattice()

        assert engine.n_plates == 16  # 4D hypercube vertices
        assert engine.order_parameter > 0  # Some convergence from initial update

        # Check that all plates are vertices
        for plate in engine._plate_list:
            assert plate.metadata.get("is_tesseract_vertex") is True
        print("[PASS] test_create_tesseract_lattice")


class TestRetrievalResult:
    """Test RetrievalResult class."""

    def test_from_plate(self):
        """Test creating result from plate."""
        plate = MemoryPlate(
            plate_id="result-test",
            phase=0.5,
            content=[0.1, 0.2, 0.3]
        )

        result = RetrievalResult.from_plate(
            plate,
            query_phase=0.5,
            query_content=[0.1, 0.2, 0.3],
            content_weight=0.5
        )

        assert result.plate == plate
        assert result.resonance_score > 0.9  # Same phase
        assert result.content_similarity > 0.9  # Same content
        assert result.combined_score > 0.9
        print("[PASS] test_from_plate")


def run_all_phase3_tests():
    """Run all Phase 3 tests."""
    print("\n" + "=" * 60)
    print("PHASE 3: ENGINE TESTS")
    print("=" * 60)

    test_classes = [
        TestLatticeConfig,
        TestEngineCreation,
        TestPlateManagement,
        TestEngineProperties,
        TestEngineDynamics,
        TestRetrieval,
        TestSerialization,
        TestTesseractFactory,
        TestRetrievalResult,
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
    print(f"PHASE 3 RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    return passed, failed


if __name__ == "__main__":
    run_all_phase3_tests()
