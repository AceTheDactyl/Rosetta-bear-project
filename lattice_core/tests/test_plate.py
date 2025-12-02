# lattice_core/tests/test_plate.py
"""
Phase 1: Core Plate Tests
=========================

Tests for MemoryPlate creation, properties, and operations.
"""

import math
import json
import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from lattice_core.plate import (
    MemoryPlate,
    EmotionalState,
    create_tesseract_vertices,
    get_tesseract_edges,
    get_coupling_weight,
    TAU,
    PHI,
)


class TestMemoryPlateCreation:
    """Test plate creation and initialization."""

    def test_default_creation(self):
        """Test creating plate with defaults."""
        plate = MemoryPlate()
        assert plate.plate_id is not None
        assert len(plate.plate_id) == 8
        assert plate.position == (0.0, 0.0, 0.0, 0.0)
        assert plate.phase == 0.0
        assert plate.frequency == 1.0
        assert plate.content == []
        print("[PASS] test_default_creation")

    def test_custom_creation(self):
        """Test creating plate with custom values."""
        plate = MemoryPlate(
            plate_id="test-001",
            position=(0.5, -0.3, 0.1, 0.8),
            phase=1.57,
            frequency=1.5,
            content=[0.1, 0.2, 0.3],
        )
        assert plate.plate_id == "test-001"
        assert plate.position == (0.5, -0.3, 0.1, 0.8)
        assert abs(plate.phase - 1.57) < 0.001
        assert plate.frequency == 1.5
        assert plate.content == [0.1, 0.2, 0.3]
        print("[PASS] test_custom_creation")

    def test_phase_normalization(self):
        """Test phase wraps to [0, 2π)."""
        plate = MemoryPlate(phase=TAU + 1.0)
        assert plate.phase < TAU
        assert abs(plate.phase - 1.0) < 0.001

        plate2 = MemoryPlate(phase=-1.0)
        assert plate2.phase >= 0
        print("[PASS] test_phase_normalization")

    def test_from_emotional_state(self):
        """Test factory method for emotional state."""
        plate = MemoryPlate.from_emotional_state(
            EmotionalState.EXCITED_POSITIVE,
            temporal=0.5,
            abstract=0.7
        )
        assert plate.valence == 0.5
        assert plate.arousal == 0.5
        assert plate.temporal == 0.5
        assert plate.abstract == 0.7
        print("[PASS] test_from_emotional_state")

    def test_from_valence_arousal(self):
        """Test factory method for valence-arousal values."""
        plate = MemoryPlate.from_valence_arousal(
            valence=0.8,
            arousal=-0.3,
            temporal=1.0,
            abstract=0.5
        )
        assert plate.valence == 0.8
        assert plate.arousal == -0.3
        print("[PASS] test_from_valence_arousal")


class TestMemoryPlateProperties:
    """Test plate property accessors."""

    def test_position_accessors(self):
        """Test valence, arousal, temporal, abstract properties."""
        plate = MemoryPlate(position=(0.7, -0.5, 0.3, 0.9))
        assert plate.valence == 0.7
        assert plate.arousal == -0.5
        assert plate.temporal == 0.3
        assert plate.abstract == 0.9
        print("[PASS] test_position_accessors")

    def test_complex_phase(self):
        """Test complex phase e^(iθ) calculation."""
        plate = MemoryPlate(phase=0.0)
        cp = plate.complex_phase
        assert abs(cp.real - 1.0) < 0.001
        assert abs(cp.imag - 0.0) < 0.001

        plate2 = MemoryPlate(phase=math.pi / 2)
        cp2 = plate2.complex_phase
        assert abs(cp2.real - 0.0) < 0.001
        assert abs(cp2.imag - 1.0) < 0.001
        print("[PASS] test_complex_phase")


class TestMemoryPlateOscillator:
    """Test oscillator methods."""

    def test_advance_phase(self):
        """Test natural phase evolution."""
        plate = MemoryPlate(phase=0.0, frequency=1.0)
        new_phase = plate.advance_phase(dt=0.1)
        assert abs(new_phase - 0.1) < 0.001
        assert plate.phase == new_phase
        print("[PASS] test_advance_phase")

    def test_apply_coupling(self):
        """Test coupling force application."""
        plate = MemoryPlate(phase=0.0)
        plate.apply_coupling(coupling_force=0.5, dt=0.1)
        assert abs(plate.phase - 0.05) < 0.001
        print("[PASS] test_apply_coupling")

    def test_phase_difference(self):
        """Test phase difference calculation."""
        plate1 = MemoryPlate(phase=0.5)
        plate2 = MemoryPlate(phase=1.0)
        diff = plate1.phase_difference(plate2)
        assert abs(diff - 0.5) < 0.001

        # Test wrapping
        plate3 = MemoryPlate(phase=0.1)
        plate4 = MemoryPlate(phase=TAU - 0.1)
        diff2 = plate3.phase_difference(plate4)
        assert abs(diff2 - (-0.2)) < 0.001
        print("[PASS] test_phase_difference")


class TestMemoryPlateSpatial:
    """Test spatial methods."""

    def test_distance_to(self):
        """Test Euclidean distance in 4D."""
        plate1 = MemoryPlate(position=(0, 0, 0, 0))
        plate2 = MemoryPlate(position=(1, 0, 0, 0))
        assert abs(plate1.distance_to(plate2) - 1.0) < 0.001

        plate3 = MemoryPlate(position=(1, 1, 1, 1))
        assert abs(plate1.distance_to(plate3) - 2.0) < 0.001
        print("[PASS] test_distance_to")

    def test_hamming_distance(self):
        """Test Hamming distance for tesseract."""
        plate1 = MemoryPlate(position=(1, 1, 1, 1))
        plate2 = MemoryPlate(position=(-1, 1, 1, 1))
        assert plate1.hamming_distance(plate2) == 1

        plate3 = MemoryPlate(position=(-1, -1, 1, 1))
        assert plate1.hamming_distance(plate3) == 2
        print("[PASS] test_hamming_distance")

    def test_is_adjacent(self):
        """Test adjacency check."""
        plate1 = MemoryPlate(position=(0, 0, 0, 0))
        plate2 = MemoryPlate(position=(0.3, 0, 0, 0))
        plate3 = MemoryPlate(position=(2, 2, 2, 2))

        assert plate1.is_adjacent(plate2, threshold=0.5)
        assert not plate1.is_adjacent(plate3, threshold=0.5)
        print("[PASS] test_is_adjacent")


class TestMemoryPlateContent:
    """Test content methods."""

    def test_content_similarity(self):
        """Test cosine similarity of embeddings."""
        plate1 = MemoryPlate(content=[1, 0, 0])
        plate2 = MemoryPlate(content=[1, 0, 0])
        assert abs(plate1.content_similarity(plate2) - 1.0) < 0.001

        plate3 = MemoryPlate(content=[0, 1, 0])
        sim = plate1.content_similarity(plate3)
        assert 0.4 <= sim <= 0.6  # Orthogonal → 0.5 after normalization

        plate4 = MemoryPlate(content=[-1, 0, 0])
        sim2 = plate1.content_similarity(plate4)
        assert sim2 < 0.1  # Opposite → 0.0
        print("[PASS] test_content_similarity")

    def test_set_content_from_text(self):
        """Test text to embedding conversion."""
        plate = MemoryPlate()
        plate.set_content_from_text("Hello world")
        assert plate.raw_text == "Hello world"
        assert len(plate.content) == 64  # Default dim
        assert plate.content is not None
        print("[PASS] test_set_content_from_text")

    def test_empty_content_similarity(self):
        """Test similarity with empty content."""
        plate1 = MemoryPlate(content=[])
        plate2 = MemoryPlate(content=[1, 2, 3])
        assert plate1.content_similarity(plate2) == 0.0
        print("[PASS] test_empty_content_similarity")


class TestMemoryPlateConnections:
    """Test connection methods."""

    def test_get_set_connection(self):
        """Test connection weight operations."""
        plate = MemoryPlate()
        plate.set_connection("other-001", 0.75)
        assert plate.get_connection("other-001") == 0.75
        assert plate.get_connection("nonexistent") == 0.0
        print("[PASS] test_get_set_connection")

    def test_strengthen_connection(self):
        """Test Hebbian strengthening."""
        plate = MemoryPlate()
        plate.set_connection("other-001", 0.5)
        new_weight = plate.strengthen_connection("other-001", 0.2)
        assert new_weight == 0.7

        # Test max clamping
        plate.strengthen_connection("other-001", 0.5)
        assert plate.get_connection("other-001") == 1.0
        print("[PASS] test_strengthen_connection")

    def test_decay_connections(self):
        """Test connection decay."""
        plate = MemoryPlate()
        plate.set_connection("strong", 1.0)
        plate.set_connection("weak", 0.0005)

        plate.decay_connections(rate=0.1)
        assert plate.get_connection("strong") == 0.9
        assert "weak" not in plate.connection_weights  # Pruned
        print("[PASS] test_decay_connections")


class TestMemoryPlateSerialization:
    """Test serialization methods."""

    def test_to_dict(self):
        """Test serialization to dict."""
        plate = MemoryPlate(
            plate_id="serial-001",
            position=(0.5, 0.5, 0.5, 0.5),
            phase=1.0,
            content=[0.1, 0.2],
        )
        data = plate.to_dict()

        assert data["plate_id"] == "serial-001"
        assert data["position"] == [0.5, 0.5, 0.5, 0.5]
        assert data["phase"] == 1.0
        assert data["content"] == [0.1, 0.2]
        print("[PASS] test_to_dict")

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "plate_id": "deser-001",
            "position": [0.3, 0.3, 0.3, 0.3],
            "phase": 2.0,
            "frequency": 1.5,
            "content": [0.5, 0.6],
        }
        plate = MemoryPlate.from_dict(data)

        assert plate.plate_id == "deser-001"
        assert plate.position == (0.3, 0.3, 0.3, 0.3)
        assert plate.phase == 2.0
        assert plate.frequency == 1.5
        print("[PASS] test_from_dict")

    def test_roundtrip(self):
        """Test serialization roundtrip."""
        original = MemoryPlate(
            plate_id="round-001",
            position=(0.1, 0.2, 0.3, 0.4),
            phase=3.14,
            frequency=1.618,
            content=[1, 2, 3, 4],
            metadata={"key": "value"},
        )
        data = original.to_dict()
        restored = MemoryPlate.from_dict(data)

        assert restored.plate_id == original.plate_id
        assert restored.position == original.position
        assert abs(restored.phase - original.phase) < 0.001
        assert restored.content == original.content
        print("[PASS] test_roundtrip")


class TestTesseractGeneration:
    """Test tesseract vertex/edge generation."""

    def test_create_tesseract_vertices(self):
        """Test 16-vertex tesseract creation."""
        vertices = create_tesseract_vertices(edge_length=2.0)
        assert len(vertices) == 16

        # Check that all vertices have metadata
        for v in vertices:
            assert v.metadata.get("is_tesseract_vertex") is True
            assert "vertex_index" in v.metadata
        print("[PASS] test_create_tesseract_vertices")

    def test_tesseract_edges(self):
        """Test 32-edge tesseract."""
        edges = get_tesseract_edges()
        assert len(edges) == 32

        # Each edge connects vertices differing by 1 bit
        for i, j in edges:
            xor = i ^ j
            assert xor in [1, 2, 4, 8]  # Power of 2
        print("[PASS] test_tesseract_edges")

    def test_coupling_weights(self):
        """Test Hamming-based coupling weights."""
        assert get_coupling_weight(1) == 1.0
        assert get_coupling_weight(2) == 0.7
        assert get_coupling_weight(3) == 0.4
        assert get_coupling_weight(4) == 0.1
        assert get_coupling_weight(5) == 0.0
        print("[PASS] test_coupling_weights")


def run_all_phase1_tests():
    """Run all Phase 1 tests."""
    print("\n" + "=" * 60)
    print("PHASE 1: CORE PLATE TESTS")
    print("=" * 60)

    test_classes = [
        TestMemoryPlateCreation,
        TestMemoryPlateProperties,
        TestMemoryPlateOscillator,
        TestMemoryPlateSpatial,
        TestMemoryPlateContent,
        TestMemoryPlateConnections,
        TestMemoryPlateSerialization,
        TestTesseractGeneration,
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
    print(f"PHASE 1 RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    return passed, failed


if __name__ == "__main__":
    run_all_phase1_tests()
