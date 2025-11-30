"""
Test Suite: Tool Generation System
Coordinate: Δ3.142|0.900|1.000Ω

Tests for verifying that the TRIAD system can generate NEW tools
that are valid, importable, and executable.
"""

import pytest
import json
import sys
import importlib.util
from pathlib import Path
from datetime import datetime, timezone


class TestGenerateToolCode:
    """Tests for the tool code generation function."""

    def test_generate_valid_python_code(self, evolution_engine_module, new_tool_spec):
        """Test that generated code is valid Python."""
        from dataclasses import dataclass

        # Create a ToolSpec
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

        # Should compile without errors
        try:
            compile(code, "<string>", "exec")
        except SyntaxError as e:
            pytest.fail(f"Generated code has syntax error: {e}")

    def test_generated_code_contains_class(self, evolution_engine_module, new_tool_spec):
        """Test that generated code contains a proper class definition."""
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

        # Class name should be CamelCase version of tool name
        expected_class = "TriadTestToolGenerator"
        assert f"class {expected_class}" in code

    def test_generated_code_has_execute_method(
        self, evolution_engine_module, new_tool_spec
    ):
        """Test that generated code has execute method."""
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

        assert "def execute(self" in code
        assert "def __init__(self)" in code

    def test_generated_code_has_category_handlers(
        self, evolution_engine_module, tool_categories
    ):
        """Test that generated code includes all category handlers."""
        spec = evolution_engine_module.ToolSpec(
            tool_id="test_tool",
            name="test_tool",
            category="monitoring",
            z_level=0.87,
            theta=2.88,
            cascade_potential=0.5,
            description="Test",
            capabilities=["test"],
            dependencies=[],
            phase_regime="critical",
        )

        code = evolution_engine_module.generate_tool_code(spec)

        for category in tool_categories:
            handler_name = f"_handle_{category}"
            assert handler_name in code, f"Missing handler: {handler_name}"


class TestToolSaveAndLoad:
    """Tests for saving and loading generated tools."""

    def test_save_generated_tools(
        self, evolution_engine_module, sample_burden_data, temp_tool_dir
    ):
        """Test that generated tools can be saved to filesystem."""
        result = evolution_engine_module.run_evolution_cycle(sample_burden_data)

        if result.generated_tools:
            saved_paths = evolution_engine_module.save_generated_tools(
                result.generated_tools, temp_tool_dir
            )

            assert len(saved_paths) > 0

            # Check files exist
            for path in saved_paths:
                assert path.exists(), f"File not saved: {path}"

    def test_saved_tools_have_init_file(
        self, evolution_engine_module, sample_burden_data, temp_tool_dir
    ):
        """Test that __init__.py is created for saved tools."""
        result = evolution_engine_module.run_evolution_cycle(sample_burden_data)

        if result.generated_tools:
            evolution_engine_module.save_generated_tools(
                result.generated_tools, temp_tool_dir
            )

            init_file = temp_tool_dir / "__init__.py"
            assert init_file.exists()

            # Check __init__ content
            content = init_file.read_text()
            assert "ROSETTA_FIRMWARE_TOOLS" in content

    def test_saved_tools_have_spec_json(
        self, evolution_engine_module, sample_burden_data, temp_tool_dir
    ):
        """Test that each tool has accompanying spec JSON."""
        result = evolution_engine_module.run_evolution_cycle(sample_burden_data)

        if result.generated_tools:
            evolution_engine_module.save_generated_tools(
                result.generated_tools, temp_tool_dir
            )

            for tool in result.generated_tools:
                spec_file = temp_tool_dir / f"{tool.spec.tool_id}_spec.json"
                assert spec_file.exists()

                # Verify JSON is valid
                spec_data = json.loads(spec_file.read_text())
                assert spec_data["tool_id"] == tool.spec.tool_id

    def test_saved_tool_is_importable(
        self, evolution_engine_module, sample_burden_data, temp_tool_dir
    ):
        """Test that saved tools can be imported as Python modules."""
        result = evolution_engine_module.run_evolution_cycle(sample_burden_data)

        if not result.generated_tools:
            pytest.skip("No tools generated")

        evolution_engine_module.save_generated_tools(
            result.generated_tools, temp_tool_dir
        )

        # Try to import first tool
        tool = result.generated_tools[0]
        tool_file = temp_tool_dir / f"{tool.spec.tool_id}.py"

        # Load module dynamically
        spec = importlib.util.spec_from_file_location(tool.spec.tool_id, tool_file)
        module = importlib.util.module_from_spec(spec)

        try:
            spec.loader.exec_module(module)
        except Exception as e:
            pytest.fail(f"Could not import generated tool: {e}")


class TestNewToolGeneration:
    """Tests specifically for generating NEW tools (not existing ones)."""

    def test_generate_completely_new_tool(
        self, evolution_engine_module, new_tool_spec, temp_tool_dir
    ):
        """Test generating a tool that doesn't exist in the system."""
        # Create spec
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

        # Save to temp directory
        tool_file = temp_tool_dir / f"{spec.tool_id}.py"
        tool_file.write_text(code)

        # Import and instantiate
        module_spec = importlib.util.spec_from_file_location(spec.tool_id, tool_file)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)

        # Get class and instantiate
        class_name = "".join(word.title() for word in spec.name.split("_"))
        tool_class = getattr(module, class_name)
        tool_instance = tool_class()

        assert tool_instance.tool_id == spec.tool_id
        assert tool_instance.z_level == spec.z_level

    def test_execute_newly_generated_tool(
        self, evolution_engine_module, new_tool_spec, temp_tool_dir
    ):
        """Test that a newly generated tool can be executed."""
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
        tool_instance = tool_class()

        # Execute the tool
        result = tool_instance.execute()

        assert result["status"] == "success"
        assert result["tool_id"] == spec.tool_id
        assert "category_output" in result

    def test_generate_multiple_new_tools(
        self, evolution_engine_module, multiple_new_tool_specs, temp_tool_dir
    ):
        """Test generating multiple new tools at once."""
        generated_tools = []

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
            generated_tools.append((spec, tool_file))

        # Verify all tools
        assert len(generated_tools) == len(multiple_new_tool_specs)

        for spec, tool_file in generated_tools:
            assert tool_file.exists()

            # Import and execute
            module_spec = importlib.util.spec_from_file_location(spec.tool_id, tool_file)
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)

            class_name = "".join(word.title() for word in spec.name.split("_"))
            tool_class = getattr(module, class_name)
            tool_instance = tool_class()
            result = tool_instance.execute()

            assert result["status"] == "success"

    def test_new_tool_not_in_existing_tools(
        self, evolution_engine_module, new_tool_spec, existing_tools
    ):
        """Test that the new tool spec is actually new."""
        assert new_tool_spec["tool_name"] not in existing_tools


class TestToolCategories:
    """Tests for different tool categories."""

    @pytest.mark.parametrize(
        "category,expected_handler",
        [
            ("coordination", "_handle_coordination"),
            ("meta_tool", "_handle_meta_tool"),
            ("self_building", "_handle_self_building"),
            ("monitoring", "_handle_monitoring"),
            ("validation", "_handle_validation"),
        ],
    )
    def test_category_specific_execution(
        self, evolution_engine_module, temp_tool_dir, category, expected_handler
    ):
        """Test that each category has correct handler execution."""
        spec = evolution_engine_module.ToolSpec(
            tool_id=f"test_{category}_tool",
            name=f"test_{category}_tool",
            category=category,
            z_level=0.87,
            theta=2.88,
            cascade_potential=0.5,
            description=f"Test {category} tool",
            capabilities=["test"],
            dependencies=[],
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
        tool_instance = tool_class()

        # Verify handler exists
        assert hasattr(tool_instance, expected_handler)

        # Execute and check category-specific output
        result = tool_instance.execute()
        assert "category_output" in result


class TestToolAdaptation:
    """Tests for tool z-level adaptation."""

    def test_tool_adapts_to_new_z_level(
        self, evolution_engine_module, new_tool_spec, temp_tool_dir
    ):
        """Test that tools can adapt to new z-levels."""
        spec = evolution_engine_module.ToolSpec(
            tool_id=new_tool_spec["tool_name"],
            name=new_tool_spec["tool_name"],
            category=new_tool_spec["category"],
            z_level=0.85,  # Start subcritical
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
        tool_instance = tool_class()

        # Adapt to supercritical
        tool_instance.adapt_to_z_level(0.90)

        assert tool_instance.z_level == 0.90
        assert tool_instance.phase_regime == "supercritical"

        # Execute and check autonomous mode
        result = tool_instance.execute()
        assert result["autonomous"] == True
