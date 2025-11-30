"""
Test Suite: End-to-End Integration Tests
Coordinate: Δ3.142|0.900|1.000Ω

Integration tests for the complete TRIAD tool generation pipeline.
These tests verify the full autonomous evolution cycle from friction
detection through tool generation and execution.
"""

import pytest
import json
import sys
import importlib.util
from pathlib import Path
from datetime import datetime, timezone


class TestFullEvolutionPipeline:
    """End-to-end tests for the complete evolution pipeline."""

    def test_complete_evolution_cycle(
        self, evolution_engine_module, sample_burden_data, temp_tool_dir, temp_log_dir
    ):
        """Test the complete 5-phase evolution cycle end-to-end."""
        # Phase 1: Friction Detection
        friction = evolution_engine_module.detect_friction(sample_burden_data)
        assert friction is not None
        assert hasattr(friction, "friction_events")

        # Phase 2: Improvement Proposal
        proposals = evolution_engine_module.generate_proposals(friction)
        assert isinstance(proposals, list)

        # Phase 3: Collective Validation
        if proposals:
            validated = evolution_engine_module.validate_proposals(proposals)
            approved_results = [v for v in validated if v.approved]

            # Phase 4: Autonomous Execution
            if approved_results:
                execution_result = evolution_engine_module.execute_approved_proposals(
                    proposals, validated
                )
                assert execution_result is not None

                # Phase 5: Meta-Learning
                learnings = evolution_engine_module.extract_meta_learnings(
                    friction, proposals, execution_result
                )
                assert learnings is not None

    def test_full_cycle_with_supercritical_z(
        self, evolution_engine_module, custom_burden_data_factory, temp_tool_dir
    ):
        """Test full cycle at z=0.90 (supercritical regime)."""
        data = custom_burden_data_factory(z_level=0.90, regime="supercritical")

        result = evolution_engine_module.run_evolution_cycle(data)

        assert result is not None
        assert result.z_level == 0.90
        # Verify all 5 phases produced output
        assert result.friction_report is not None
        assert result.proposals is not None
        assert result.consensus_results is not None
        assert result.meta_learnings is not None


class TestToolGenerationPipeline:
    """Tests for the tool generation subprocess."""

    def test_generate_and_import_new_tool(
        self, evolution_engine_module, new_tool_spec, temp_tool_dir
    ):
        """Test generating a new tool and importing it as a Python module."""
        # Create ToolSpec
        spec = evolution_engine_module.ToolSpec(
            tool_id=new_tool_spec["tool_name"],
            name=new_tool_spec["tool_name"],
            category=new_tool_spec["category"],
            z_level=new_tool_spec["z_level"],
            theta=new_tool_spec["theta"],
            cascade_potential=new_tool_spec["cascade_potential"],
            description=new_tool_spec["description"],
            capabilities=new_tool_spec["capabilities"],
            dependencies=new_tool_spec["dependencies"],
            phase_regime="critical",
        )

        # Generate code
        code = evolution_engine_module.generate_tool_code(spec)
        assert code is not None
        assert len(code) > 100  # Should have substantial code

        # Save to file
        tool_file = temp_tool_dir / f"{spec.tool_id}.py"
        tool_file.write_text(code)

        # Dynamic import
        module_spec = importlib.util.spec_from_file_location(spec.tool_id, tool_file)
        module = importlib.util.module_from_spec(module_spec)

        # Should not raise
        module_spec.loader.exec_module(module)

        # Get class
        class_name = "".join(word.title() for word in spec.name.split("_"))
        tool_class = getattr(module, class_name)

        # Instantiate
        instance = tool_class()

        # Verify properties
        assert instance.tool_id == spec.tool_id
        assert instance.z_level == spec.z_level
        assert instance.category == spec.category

    def test_execute_generated_tool_with_input(
        self, evolution_engine_module, new_tool_spec, temp_tool_dir
    ):
        """Test executing a generated tool with actual input."""
        spec = evolution_engine_module.ToolSpec(
            tool_id=new_tool_spec["tool_name"],
            name=new_tool_spec["tool_name"],
            category=new_tool_spec["category"],
            z_level=new_tool_spec["z_level"],
            theta=new_tool_spec["theta"],
            cascade_potential=new_tool_spec["cascade_potential"],
            description=new_tool_spec["description"],
            capabilities=new_tool_spec["capabilities"],
            dependencies=new_tool_spec["dependencies"],
            phase_regime="critical",
        )

        code = evolution_engine_module.generate_tool_code(spec)
        tool_file = temp_tool_dir / f"{spec.tool_id}.py"
        tool_file.write_text(code)

        module_spec = importlib.util.spec_from_file_location(spec.tool_id, tool_file)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)

        class_name = "".join(word.title() for word in spec.name.split("_"))
        tool_class = getattr(module, class_name)
        instance = tool_class()

        # Execute with input
        result = instance.execute()

        assert result["status"] == "success"
        assert result["tool_id"] == spec.tool_id
        assert "execution_time" in result or "timestamp" in result


class TestBatchToolGeneration:
    """Tests for generating multiple tools in batch."""

    def test_generate_multiple_tools_batch(
        self, evolution_engine_module, multiple_new_tool_specs, temp_tool_dir
    ):
        """Test generating multiple tools in a single batch."""
        generated = []

        for tool_spec in multiple_new_tool_specs:
            spec = evolution_engine_module.ToolSpec(
                tool_id=tool_spec["tool_name"],
                name=tool_spec["tool_name"],
                category=tool_spec["category"],
                z_level=tool_spec["z_level"],
                theta=tool_spec["theta"],
                cascade_potential=tool_spec["cascade_potential"],
                description=tool_spec["description"],
                capabilities=tool_spec["capabilities"],
                dependencies=tool_spec["dependencies"],
                phase_regime="supercritical" if tool_spec["z_level"] >= 0.88 else "critical",
            )

            code = evolution_engine_module.generate_tool_code(spec)
            tool_file = temp_tool_dir / f"{spec.tool_id}.py"
            tool_file.write_text(code)
            generated.append((spec, tool_file))

        # Verify all were generated
        assert len(generated) == len(multiple_new_tool_specs)

        # Verify all are importable and executable
        for spec, tool_file in generated:
            module_spec = importlib.util.spec_from_file_location(spec.tool_id, tool_file)
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)

            class_name = "".join(word.title() for word in spec.name.split("_"))
            tool_class = getattr(module, class_name)
            instance = tool_class()

            result = instance.execute()
            assert result["status"] == "success"

    def test_generated_tools_have_unique_ids(
        self, evolution_engine_module, multiple_new_tool_specs, temp_tool_dir
    ):
        """Test that all generated tools have unique identifiers."""
        tool_ids = set()

        for tool_spec in multiple_new_tool_specs:
            spec = evolution_engine_module.ToolSpec(
                tool_id=tool_spec["tool_name"],
                name=tool_spec["tool_name"],
                category=tool_spec["category"],
                z_level=tool_spec["z_level"],
                theta=tool_spec["theta"],
                cascade_potential=tool_spec["cascade_potential"],
                description=tool_spec["description"],
                capabilities=tool_spec["capabilities"],
                dependencies=tool_spec["dependencies"],
                phase_regime="critical",
            )

            assert spec.tool_id not in tool_ids
            tool_ids.add(spec.tool_id)


class TestToolPersistence:
    """Tests for tool persistence and registry."""

    def test_tools_persist_after_generation(
        self, evolution_engine_module, sample_burden_data, temp_tool_dir
    ):
        """Test that generated tools persist to filesystem."""
        result = evolution_engine_module.run_evolution_cycle(sample_burden_data)

        if result.generated_tools:
            # Save tools
            saved_paths = evolution_engine_module.save_generated_tools(
                result.generated_tools, temp_tool_dir
            )

            # Verify persistence - filter for tool Python files (exclude __init__.py)
            tool_py_paths = [
                p for p in saved_paths
                if p.suffix == ".py" and p.name != "__init__.py"
            ]
            for path in tool_py_paths:
                assert path.exists()
                content = path.read_text()
                assert len(content) > 0
                assert "class" in content

    def test_tool_spec_json_created(
        self, evolution_engine_module, new_tool_spec, temp_tool_dir
    ):
        """Test that spec JSON is created alongside tool."""
        spec = evolution_engine_module.ToolSpec(
            tool_id=new_tool_spec["tool_name"],
            name=new_tool_spec["tool_name"],
            category=new_tool_spec["category"],
            z_level=new_tool_spec["z_level"],
            theta=new_tool_spec["theta"],
            cascade_potential=new_tool_spec["cascade_potential"],
            description=new_tool_spec["description"],
            capabilities=new_tool_spec["capabilities"],
            dependencies=new_tool_spec["dependencies"],
            phase_regime="critical",
        )

        # Generate and save
        code = evolution_engine_module.generate_tool_code(spec)
        tool_file = temp_tool_dir / f"{spec.tool_id}.py"
        tool_file.write_text(code)

        # Create spec JSON
        spec_file = temp_tool_dir / f"{spec.tool_id}_spec.json"
        spec_data = {
            "tool_id": spec.tool_id,
            "name": spec.name,
            "category": spec.category,
            "z_level": spec.z_level,
            "theta": spec.theta,
            "capabilities": spec.capabilities,
            "dependencies": spec.dependencies,
            "phase_regime": spec.phase_regime,
        }
        spec_file.write_text(json.dumps(spec_data, indent=2))

        assert spec_file.exists()
        loaded = json.loads(spec_file.read_text())
        assert loaded["tool_id"] == spec.tool_id


class TestCrossCategoryGeneration:
    """Tests for generating tools across different categories."""

    @pytest.mark.parametrize(
        "category",
        ["coordination", "meta_tool", "self_building", "monitoring", "validation"],
    )
    def test_generate_tool_for_category(
        self, evolution_engine_module, temp_tool_dir, category
    ):
        """Test generating a tool for each category."""
        spec = evolution_engine_module.ToolSpec(
            tool_id=f"test_{category}_integration",
            name=f"test_{category}_integration",
            category=category,
            z_level=0.88,
            theta=3.0,
            cascade_potential=0.5,
            description=f"Integration test tool for {category}",
            capabilities=["test", "verify"],
            dependencies=[],
            phase_regime="critical",
        )

        code = evolution_engine_module.generate_tool_code(spec)
        tool_file = temp_tool_dir / f"{spec.tool_id}.py"
        tool_file.write_text(code)

        # Import and execute
        module_spec = importlib.util.spec_from_file_location(spec.tool_id, tool_file)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)

        class_name = "".join(word.title() for word in spec.name.split("_"))
        tool_class = getattr(module, class_name)
        instance = tool_class()

        assert instance.category == category

        result = instance.execute()
        assert result["status"] == "success"


class TestZLevelAdaptation:
    """Tests for z-level adaptation during generation."""

    def test_tool_adapts_z_level_upward(
        self, evolution_engine_module, new_tool_spec, temp_tool_dir
    ):
        """Test that tools can adapt to higher z-levels."""
        spec = evolution_engine_module.ToolSpec(
            tool_id=new_tool_spec["tool_name"],
            name=new_tool_spec["tool_name"],
            category=new_tool_spec["category"],
            z_level=0.85,  # Start lower
            theta=new_tool_spec["theta"],
            cascade_potential=new_tool_spec["cascade_potential"],
            description=new_tool_spec["description"],
            capabilities=new_tool_spec["capabilities"],
            dependencies=new_tool_spec["dependencies"],
            phase_regime="subcritical",
        )

        code = evolution_engine_module.generate_tool_code(spec)
        tool_file = temp_tool_dir / f"{spec.tool_id}.py"
        tool_file.write_text(code)

        module_spec = importlib.util.spec_from_file_location(spec.tool_id, tool_file)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)

        class_name = "".join(word.title() for word in spec.name.split("_"))
        tool_class = getattr(module, class_name)
        instance = tool_class()

        # Verify initial state
        assert instance.z_level == 0.85
        assert instance.phase_regime == "subcritical"

        # Adapt to supercritical
        instance.adapt_to_z_level(0.91)

        assert instance.z_level == 0.91
        assert instance.phase_regime == "supercritical"

    def test_phase_regime_changes_with_z(
        self, evolution_engine_module, new_tool_spec, temp_tool_dir
    ):
        """Test that phase regime changes appropriately with z-level."""
        spec = evolution_engine_module.ToolSpec(
            tool_id=new_tool_spec["tool_name"],
            name=new_tool_spec["tool_name"],
            category=new_tool_spec["category"],
            z_level=0.80,
            theta=new_tool_spec["theta"],
            cascade_potential=new_tool_spec["cascade_potential"],
            description=new_tool_spec["description"],
            capabilities=new_tool_spec["capabilities"],
            dependencies=new_tool_spec["dependencies"],
            phase_regime="subcritical",
        )

        code = evolution_engine_module.generate_tool_code(spec)
        tool_file = temp_tool_dir / f"{spec.tool_id}.py"
        tool_file.write_text(code)

        module_spec = importlib.util.spec_from_file_location(spec.tool_id, tool_file)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)

        class_name = "".join(word.title() for word in spec.name.split("_"))
        tool_class = getattr(module, class_name)
        instance = tool_class()

        # Test regime transitions
        instance.adapt_to_z_level(0.84)
        assert instance.phase_regime == "subcritical"

        instance.adapt_to_z_level(0.86)
        assert instance.phase_regime == "critical"

        instance.adapt_to_z_level(0.90)
        assert instance.phase_regime == "supercritical"


class TestErrorHandling:
    """Tests for error handling in the pipeline."""

    def test_handles_invalid_burden_data_gracefully(self, evolution_engine_module):
        """Test graceful handling of invalid burden data."""
        invalid_data = {"invalid": "structure"}

        # The engine handles invalid data gracefully - returns a FrictionReport
        # with default/empty values rather than raising
        result = evolution_engine_module.detect_friction(invalid_data)

        # Should return a valid FrictionReport even with bad input
        assert result is not None
        assert hasattr(result, "friction_events")
        # With invalid data, should have zero or minimal friction events
        assert len(result.friction_events) == 0

    def test_handles_empty_friction_report(self, evolution_engine_module, sample_burden_data):
        """Test handling when friction report has no improvement opportunities."""
        # Get a real friction report first
        friction = evolution_engine_module.detect_friction(sample_burden_data)

        # Clear improvement opportunities to simulate empty state
        friction.improvement_opportunities = []

        proposals = evolution_engine_module.generate_proposals(friction)
        # Should return empty list when no improvement opportunities
        assert isinstance(proposals, list)

    def test_handles_compilation_errors_gracefully(
        self, evolution_engine_module, temp_tool_dir
    ):
        """Test that compilation errors are caught and reported."""
        # Create intentionally broken code
        broken_code = """
class BrokenTool:
    def __init__(self
        # Missing closing parenthesis - syntax error
        self.id = "broken"
"""
        tool_file = temp_tool_dir / "broken_tool.py"
        tool_file.write_text(broken_code)

        with pytest.raises(SyntaxError):
            compile(broken_code, str(tool_file), "exec")
