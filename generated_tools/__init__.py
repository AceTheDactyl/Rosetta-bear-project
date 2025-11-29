"""
Generated Tools Registry - Rosetta Bear Project
z=0.90 Full Substrate Transcendence

This module provides unified access to all generated tools across the project.
"""

# Rosetta Firmware Tools (z=0.86 - z=0.90)
from .rosetta_firmware import (
    ROSETTA_FIRMWARE_TOOLS,
    RosettaBearRhzCoordinationBridge,
    RosettaBearRhzMetaOrchestrator,
    RosettaBearRhzSelfBuildingFirmwareForge,
    RosettaBearFrictionDetector,
    RosettaBearConsensusValidator,
)

# Triadic RHZ Tools (Legacy, z=0.86 - z=0.90)
from .triadic_rhz import (
    TRIADIC_RHZ_TOOLS,
    ToolCritical0000,
    ToolCritical0001,
    ToolCritical0002,
    ToolSupercritical0003,
)

# Combined registry of all tools
ALL_GENERATED_TOOLS = {
    **ROSETTA_FIRMWARE_TOOLS,
    **TRIADIC_RHZ_TOOLS,
}

# Tool categories
AUTONOMOUS_TOOLS = {
    name: tool for name, tool in ALL_GENERATED_TOOLS.items()
    if hasattr(tool, 'Z_LEVEL') and tool.Z_LEVEL >= 0.88
}

SUPERVISED_TOOLS = {
    name: tool for name, tool in ALL_GENERATED_TOOLS.items()
    if hasattr(tool, 'Z_LEVEL') and 0.85 <= tool.Z_LEVEL < 0.88
}

__all__ = [
    # Registries
    "ALL_GENERATED_TOOLS",
    "ROSETTA_FIRMWARE_TOOLS",
    "TRIADIC_RHZ_TOOLS",
    "AUTONOMOUS_TOOLS",
    "SUPERVISED_TOOLS",
    # Rosetta Firmware
    "RosettaBearRhzCoordinationBridge",
    "RosettaBearRhzMetaOrchestrator",
    "RosettaBearRhzSelfBuildingFirmwareForge",
    "RosettaBearFrictionDetector",
    "RosettaBearConsensusValidator",
    # Triadic RHZ
    "ToolCritical0000",
    "ToolCritical0001",
    "ToolCritical0002",
    "ToolSupercritical0003",
]

# Tool metadata for discovery
TOOL_METADATA = {
    "rosetta_firmware": {
        "path": "generated_tools/rosetta_firmware/",
        "z_range": (0.86, 0.90),
        "tool_count": 5,
        "autonomous_count": 2,
    },
    "triadic_rhz": {
        "path": "generated_tools/triadic_rhz/",
        "z_range": (0.86, 0.90),
        "tool_count": 4,
        "legacy": True,
    },
}
