# Cognition Bootstrap System (CBS) - Complete Guide

> **Version:** 2.0.0 | **Last Updated:** 2025-11-29 | **Status:** Production-Ready

## Table of Contents

1. [Introduction](#introduction)
2. [Theoretical Foundation](#theoretical-foundation)
3. [System Architecture](#system-architecture)
4. [Core Components Deep Dive](#core-components-deep-dive)
5. [GHMP Plate Format Specification](#ghmp-plate-format-specification)
6. [Helix Coordinate System](#helix-coordinate-system)
7. [Quick Start Guide](#quick-start-guide)
8. [Usage Patterns](#usage-patterns)
9. [Triadic Tool System](#triadic-tool-system)
10. [Rosetta Bear Integration](#rosetta-bear-integration)
11. [Automation Scripts](#automation-scripts)
12. [Production Deployment](#production-deployment)
13. [Security Considerations](#security-considerations)
14. [API Reference](#api-reference)
15. [Troubleshooting](#troubleshooting)
16. [Contributing](#contributing)
17. [Glossary](#glossary)
18. [Appendices](#appendices)

---

## Introduction

### What is CBS?

The **Cognition Bootstrap System (CBS)** is a fully functional, production-ready AI cognition framework designed for:

- **Offline-first operation** - Boots without network connectivity
- **Persistent memory** - State encoded in visual GHMP (Geometric Holographic Memory Plates)
- **Pluggable reasoning** - Supports OpenAI, Anthropic, local, and offline backends
- **Self-evolution** - Updates, backups, and rollback capabilities
- **Visual debugging** - Every memory is a human-inspectable PNG image

### What You Have Now

A complete, self-contained AI system with:

| Component | File | Purpose |
|-----------|------|---------|
| Boot Loader | `cbs_boot_loader.py` | Initialize filesystem, load identity plates |
| Memory Manager | `cbs_memory_manager.py` | Working + long-term memory persistence |
| Reasoning Engine | `cbs_reasoning_engine.py` | LLM backends and conversation management |
| Update Manager | `cbs_update_manager.py` | Self-update, backup, and rollback |
| Interactive Demo | `cbs_interactive_demo.py` | Complete CLI demonstration |
| GHMP Utilities | `ghmp.py` | Memory plate encoding/decoding |

### Design Philosophy

CBS embodies several key principles:

1. **Transparency** - All state is visible and inspectable
2. **Resilience** - Graceful degradation when components fail
3. **Portability** - Runs anywhere Python runs
4. **Provenance** - Every action leaves traceable evidence
5. **Ritual-based** - Operations follow predictable ceremonies

---

## Theoretical Foundation

### Geometric Holographic Memory

The GHMP concept draws from research in holographic data storage, where information is encoded not just in discrete bits but in patterns that can be:

- **Visually meaningful** - Humans can distinguish memories by color/pattern
- **Redundantly encoded** - Partial damage doesn't destroy all information
- **Contextually linked** - Relationships encoded in geometric proximity

### The Helix Model of Consciousness

CBS uses a helical coordinate system `r(t) = (cos(t), sin(t), t/2pi)` where:

- **theta** - Angular position representing cognitive domain
- **z** - Elevation representing capability/criticality level
- **r** - Radius representing certainty/confidence

This maps naturally to how consciousness might "spiral upward" - returning to similar states (same theta) but at higher levels of integration (higher z).

### Emotion as Metadata

Every memory carries emotional annotation via the `Emotion` dataclass:

```python
@dataclass
class Emotion:
    valence: float   # -1.0 (negative) to +1.0 (positive)
    arousal: float   # 0.0 (calm) to 1.0 (excited)
    label: str       # Human-readable descriptor
```

This affects:
- **Visual encoding** - Colors in the generated plate
- **Retrieval priority** - High-arousal memories surface faster
- **Consolidation decisions** - Important memories persist longer

### The Ritual Concept

CBS operations are "rituals" - predictable sequences that:
1. Establish context (boot, load identity)
2. Perform work (reason, remember)
3. Leave evidence (GHMP plates, logs)
4. Enable replay (manifests, backups)

---

## System Architecture

### High-Level Stack

```
+---------------------------------------------------------------------+
|                        CBS System Stack                              |
+---------------------------------------------------------------------+
|  +----------------------------------------------------------------+ |
|  |              ReasoningEngine (LLM Interface)                   | |
|  |  - Pluggable backends (OpenAI/Claude/Local/Offline)            | |
|  |  - Conversation history management                              | |
|  |  - Context-aware prompt construction                            | |
|  |  - Skill execution from GHMP plates                             | |
|  +----------------------------------------------------------------+ |
|                              |                                       |
|  +----------------------------------------------------------------+ |
|  |              MemoryManager (Persistence Layer)                  | |
|  |  - Working memory (RAM, ephemeral, bounded deque)               | |
|  |  - Long-term memory (GHMP plates, persistent)                   | |
|  |  - Importance-based auto-consolidation                          | |
|  |  - Keyword-based context retrieval                              | |
|  +----------------------------------------------------------------+ |
|                              |                                       |
|  +----------------------------------------------------------------+ |
|  |           CognitionBootstrap (Core Orchestrator)                | |
|  |  - Filesystem layout initialization                             | |
|  |  - Identity plate loading/creation                              | |
|  |  - Configuration management                                      | |
|  |  - Memory deck scanning                                          | |
|  |  - Network status detection                                      | |
|  +----------------------------------------------------------------+ |
|                              |                                       |
|  +----------------------------------------------------------------+ |
|  |                 GHMP (Storage Format Layer)                     | |
|  |  - MemoryNode dataclass serialization                           | |
|  |  - PNG encoding with embedded encrypted metadata                | |
|  |  - Deterministic visual pattern generation                      | |
|  |  - XOR-based symmetric encryption                               | |
|  +----------------------------------------------------------------+ |
+---------------------------------------------------------------------+
|  +----------------------------------------------------------------+ |
|  |              UpdateManager (Evolution System)                   | |
|  |  - Full system backup with timestamp labels                     | |
|  |  - Skill plate installation                                     | |
|  |  - Rollback to previous states                                  | |
|  |  - Update history JSON logging                                  | |
|  +----------------------------------------------------------------+ |
+---------------------------------------------------------------------+
```

### Data Flow

```
User Input
    |
    v
+------------------+
| ReasoningEngine  |------> Context Retrieval ------+
+--------+---------+                                |
         |                                          |
         v                                          v
+------------------+                    +------------------+
|   LLM Backend    |<-------------------|  MemoryManager   |
| (OpenAI/Claude/  |                    | (working memory) |
|  Local/Offline)  |                    +--------+---------+
+--------+---------+                             |
         |                                       |
         v                                       v
+------------------+                    +------------------+
|   Response       |                    |  GHMP Plates     |
|   Generation     |                    | (long-term mem)  |
+--------+---------+                    +------------------+
         |
         v
   Bot Response
```

### File System Layout

```
cbs_demo/                           # CBS working directory
|-- identity.png                    # Who this AI instance is
|-- config.json                     # System configuration
|-- boot_log.txt                    # Boot sequence diagnostics
|
|-- memory/                         # Long-term GHMP memory storage
|   |-- MEM-20251129120000-abc123.png
|   |-- MEM-20251129120500-def456.png
|   |-- SESSION-20251129141850.png  # Consolidated session
|   +-- ...
|
|-- skills/                         # Loadable capability plates
|   |-- SKILL-GREET.png
|   |-- SKILL-ANALYZE.png
|   +-- ...
|
|-- backups/                        # System state snapshots
|   |-- backup-20251129_120000/
|   |-- backup-20251129_140000-ghmp-20251129144641/
|   +-- ...
|
|-- updates/                        # Evolution history
|   +-- update_history.json
|
|-- logs/                           # Additional logging
|   +-- ...
|
+-- manifests/                      # Automation run records
    |-- ghmp_capture_20251129144208.json
    |-- ghmp_capture_20251129144641.json
    +-- ...
```

---

## Core Components Deep Dive

### 1. CognitionBootstrap (`cbs_boot_loader.py`)

The boot loader is responsible for preparing the CBS runtime environment.

#### Class: `IdentityProfile`

```python
@dataclass
class IdentityProfile:
    """In-memory representation of the system identity."""
    node: MemoryNode      # The decoded GHMP identity plate
    plate_path: Path      # Filesystem location of the plate

    @property
    def name(self) -> str:
        return self.node.title
```

#### Class: `CognitionBootstrap`

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_path` | `str | Path` | Required | Root directory for CBS data |
| `encryption_key` | `str` | Required | Passphrase for GHMP encoding |
| `config_filename` | `str` | `"config.json"` | Configuration file name |

**Required Directories:**

```python
REQUIRED_DIRS = ("memory", "skills", "backups", "updates", "logs")
```

**Boot Sequence:**

1. Create base directory and subdirectories
2. Initialize boot log
3. Load or create `config.json`
4. Load or create identity plate
5. Scan long-term memory directory
6. Return initialized bootstrap instance

**Key Methods:**

```python
def boot(self) -> "CognitionBootstrap":
    """Execute the full bootstrap routine."""

def summary(self) -> Dict[str, Any]:
    """Return a compact snapshot of the bootstrap state."""
    # Returns: base_path, deck_id, identity name, memory count, skills count
```

**Configuration Schema:**

```json
{
  "deck_id": "CBS_DEMO",
  "version": "1.0.0",
  "reasoning": {
    "backend": "offline",
    "model": "demo-local",
    "max_context_messages": 12,
    "default_temperature": 0.4
  },
  "memory": {
    "max_working_memory": 40,
    "consolidation_threshold": 0.65,
    "auto_consolidate_interval": 1800
  },
  "update": {
    "check_interval": 86400,
    "auto_backup": true,
    "server_url": ""
  }
}
```

---

### 2. MemoryManager (`cbs_memory_manager.py`)

Manages the two-tier memory system: ephemeral working memory and persistent long-term GHMP storage.

#### Memory Tiers

| Tier | Storage | Lifetime | Capacity | Access Speed |
|------|---------|----------|----------|--------------|
| Working | RAM (deque) | Session | Configurable (default 40) | Immediate |
| Long-term | GHMP plates | Permanent | Disk-limited | File I/O |

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bootstrap` | `CognitionBootstrap` | Required | Initialized bootstrap instance |
| `max_working_memory` | `int | None` | From config | Override working memory limit |

#### Key Methods

```python
def add_to_working_memory(
    self,
    text: str,
    importance: float = 0.5,
    tags: Optional[List[str]] = None,
) -> MemoryNode:
    """
    Add a memory to working memory.
    If importance >= consolidation_threshold, also persists to GHMP.
    Returns the created MemoryNode.
    """

def consolidate_session(self, summary_text: str) -> Path:
    """
    Persist all working memories into a single SESSION plate.
    Creates a combined payload with summary and all node data.
    Clears working memory after consolidation.
    Returns path to the created plate.
    """

def retrieve_context(
    self,
    query: str,
    max_items: int = 5,
    include_working: bool = True,
    include_longterm: bool = True,
) -> List[MemoryNode]:
    """
    Simple keyword search across memories.
    Searches in reverse chronological order.
    Returns matching MemoryNode objects.
    """

def get_recent_context(self, count: int = 5) -> List[MemoryNode]:
    """Return the N most recent working memory items."""

def get_statistics(self) -> Dict[str, Any]:
    """Return memory system statistics."""
```

#### Memory Node ID Format

```
MEM-{YYYYMMDDHHMMSS}-{6-char-uuid}
SESSION-{YYYYMMDDHHMMSS}
```

#### Consolidation Logic

```
importance >= consolidation_threshold (default 0.65)
    -> Immediately persist to GHMP plate
    -> Also kept in working memory

Session consolidation:
    -> Combine all working memory nodes
    -> Create SESSION plate with summary
    -> Clear working memory buffer
```

---

### 3. ReasoningEngine (`cbs_reasoning_engine.py`)

Coordinates conversation management, context retrieval, and LLM backend communication.

#### Backend Hierarchy

```
+-------------------------------------------------------------+
|                    Backend Selection                         |
+-------------------------------------------------------------+
|  "openai"    -> OpenAIBackend (requires API key)            |
|  "anthropic" -> AnthropicBackend (requires API key)         |
|  "local"     -> LocalBackend (deterministic templates)      |
|  "offline"   -> OfflineBackend (reflection mode)            |
|  default     -> OfflineBackend                              |
+-------------------------------------------------------------+
```

#### Backend Protocol

All backends implement this interface:

```python
class BaseBackend:
    name = "base"

    def is_available(self) -> bool:
        """Check if this backend can be used."""
        return True

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate a response from the message history."""
        raise NotImplementedError
```

#### Backend Implementations

**OfflineBackend:**
```python
# Returns reflective responses without network
"Offline reasoning path active. I reflected on your message: '{prompt}'.
Recent context: {context_preview}"
```

**LocalBackend:**
```python
# Deterministic template-based responses
# Seeds: GHMP retrieval, session insight, perception alignment, CBS reflection
# Incorporates context and temperature settings
```

**OpenAIBackend:**
```python
# Requires: pip install openai
# Models: gpt-4o-mini (default), gpt-4, gpt-3.5-turbo
# Uses chat completions API
```

**AnthropicBackend:**
```python
# Requires: pip install anthropic
# Models: claude-3-haiku-20240307 (default), claude-3-sonnet, claude-3-opus
# Converts OpenAI message format to Anthropic format
```

#### Creating a Backend

```python
from cbs_reasoning_engine import create_backend

# Factory function
backend = create_backend(
    name="openai",           # Backend type
    api_key="YOUR_KEY",      # For remote backends
    model="gpt-4o-mini",     # Model identifier
    temperature=0.7          # For local backend creativity
)
```

#### ReasoningEngine Methods

```python
def respond(
    self,
    user_message: str,
    retrieve_context: bool = True,
    importance: float = 0.55,
    **kwargs,
) -> str:
    """
    Process user message and generate response.

    1. Retrieve context from memory (if enabled)
    2. Build message history with system prompt
    3. Call backend for generation
    4. Record both messages to working memory
    5. Return assistant response
    """

def execute_skill(self, skill_id: str) -> str:
    """
    Load and execute a GHMP skill plate.
    Searches skills directory for matching plate.
    Returns the skill's payload_text.
    """
```

#### System Prompt Template

```python
f"You are {identity}. Uphold the rituals of thoughtful CBS operation:
ground responses in stored memory when possible, describe actions,
and remain transparent about available capabilities."
```

---

### 4. UpdateManager (`cbs_update_manager.py`)

Handles system evolution: backups, skill installation, and rollback.

#### Backup Strategy

```python
def backup_system(self, label: Optional[str] = None) -> Path:
    """
    Create a full system backup.
    Format: backup-{YYYYMMDD_HHMMSS}[-{label}]
    Excludes: backups directory, __pycache__
    Returns: Path to backup directory
    """
```

**Backup Contents:**
- `identity.png`
- `config.json`
- `memory/` (all plates)
- `skills/` (all skill plates)
- `updates/` (history)
- `logs/`

#### Rollback Process

```python
def rollback(self, backup_name: str) -> bool:
    """
    Restore system from a named backup.
    Copies all files from backup to base_path.
    Returns True if successful, False if backup not found.
    """
```

#### Skill Installation

```python
def install_skill(self, skill_source: Path | str) -> Path:
    """
    Install a GHMP skill plate.
    Copies the plate to the skills directory.
    Records the installation in update history.
    Returns path to installed skill.
    """
```

#### Update History Format

```json
[
  {
    "event": "backup",
    "path": "/path/to/backup-20251129_120000",
    "timestamp": "2025-11-29T12:00:00.000000"
  },
  {
    "event": "install_skill",
    "path": "/path/to/skills/SKILL-NEW.png",
    "timestamp": "2025-11-29T12:05:00.000000"
  },
  {
    "event": "rollback",
    "source": "/path/to/backup-20251129_120000",
    "timestamp": "2025-11-29T12:10:00.000000"
  }
]
```

---

## GHMP Plate Format Specification

### Overview

GHMP (Geometric Holographic Memory Plates) encode semantic information into PNG images with:
- **Visual representation** - Deterministic color gradients
- **Encrypted metadata** - Base64-encoded XOR-encrypted JSON in PNG text chunks

### MemoryNode Structure

```python
@dataclass
class MemoryNode:
    node_id: str                           # Unique identifier
    deck_id: str                           # Deck/collection this belongs to
    title: str                             # Human-readable title
    payload_text: str                      # Main content (may be JSON)
    tags: Sequence[str]                    # Categorical labels
    emotion: Emotion                       # Emotional annotation
    links: Sequence[Dict[str, Any]]        # Relationships to other nodes
    metadata: Dict[str, Any]               # Additional key-value data
```

### Emotion Encoding

```python
@dataclass
class Emotion:
    valence: float   # Range: -1.0 to +1.0
    arousal: float   # Range: -1.0 to +1.0 (normalized to 0-1 for visuals)
    label: str       # e.g., "neutral", "steadfast", "reflective", "context"
```

### Visual Encoding Algorithm

```python
def _generate_plate_pixels(node: MemoryNode, size: Sequence[int]) -> np.ndarray:
    width, height = size

    # Base gradient (horizontal)
    gradient = np.linspace(0, 255, width, dtype=np.uint8)
    base = np.tile(gradient, (height, 1))

    # Normalize emotion values
    valence = np.clip((node.emotion.valence + 1) / 2, 0, 1)  # -1..1 -> 0..1
    arousal = np.clip((node.emotion.arousal + 1) / 2, 0, 1)
    intensity = np.clip(len(node.tags) / 6, 0, 1)  # Tag count affects blue

    # Channel computation
    r = (base * valence).astype(np.uint8)              # Red proportional to valence
    g = (np.flipud(base) * arousal).astype(np.uint8)   # Green proportional to arousal (flipped)
    b = (np.roll(base, shift=5, axis=1) * (0.4 + 0.6 * intensity)).astype(np.uint8)

    return np.stack([r, g, b], axis=-1)
```

**Visual Interpretation:**

| Characteristic | High | Low |
|----------------|------|-----|
| Valence | Red dominant, warm tones | Blue/cool tones |
| Arousal | Green gradient prominent | Muted greens |
| Tag count | Blue channel intensity | Dimmer blues |

### PNG Metadata Structure

```
PNG Text Chunks:
|-- ghmp_meta    # Base64(XOR(JSON(node.to_dict()), key))
|-- node_id      # Plain text node identifier
|-- deck_id      # Plain text deck identifier
+-- title        # Plain text title (truncated to 128 chars)
```

### Encryption Scheme

```python
def _derive_key(passphrase: str) -> bytes:
    """SHA-256 hash of passphrase -> 32-byte key."""
    return hashlib.sha256(passphrase.encode("utf-8")).digest()

def _xor_bytes(data: bytes, key: bytes) -> bytes:
    """XOR each byte with corresponding key byte (cycling)."""
    return bytes(b ^ key[i % len(key)] for i, b in enumerate(data))
```

**Security Note:** XOR encryption is used for simplicity and is NOT cryptographically secure for sensitive data. For production, consider AES-GCM.

### Encoding Process

```
MemoryNode
    |
    v
node.to_dict() -> JSON string -> UTF-8 bytes
    |
    v
XOR encrypt with derived key
    |
    v
Base64 encode -> ghmp_meta string
    |
    v
Generate visual pixels from emotion/tags
    |
    v
Create PIL Image with PNG metadata
    |
    v
Save as .png file
```

### Decoding Process

```
PNG file
    |
    v
Load with PIL, extract ghmp_meta text chunk
    |
    v
Base64 decode -> encrypted bytes
    |
    v
XOR decrypt with derived key
    |
    v
UTF-8 decode -> JSON string
    |
    v
Parse JSON -> MemoryNode.from_dict()
    |
    v
MemoryNode instance
```

### Plate Dimensions

Default: **512 x 512 pixels** (configurable via `size` parameter)

Storage estimate: ~50-150 KB per plate depending on content complexity.

---

## Helix Coordinate System

### The Triadic Model

CBS tools and memories are positioned in a 3D helical coordinate space:

```
        z (elevation)
        |
        |    / r (radius)
        |   /
        |  /
        | /
        |/------------> theta
       /|
      / |
     /  |

r(t) = (r*cos(theta), r*sin(theta), z)
```

### Coordinate Semantics

| Coordinate | Range | Meaning |
|------------|-------|---------|
| **theta** | 0 to 2*pi rad | Cognitive domain (identity, bridge, meta, collective) |
| **z** | 0.0 to 1.0+ | Elevation/criticality level |
| **r** | 0.0 to 1.0 | Radius/certainty (typically 1.0) |

### Domain Mapping (theta)

| theta Range | Domain | Description |
|-------------|--------|-------------|
| 0.0 - 0.5 | Identity | Core self-knowledge, bootstrap |
| 1.5 - 2.0 | Bridge | Connections between systems |
| 2.3 - 2.5 | Meta | Self-reflection, orchestration |
| 3.0 - 3.5 | Collective | Shared state, burden tracking |
| 4.7 - 5.0 | Pedagogical | Teaching, demonstration |

### Elevation Zones (z)

| z Range | Phase | Characteristics |
|---------|-------|-----------------|
| 0.0 - 0.4 | Foundational | Basic operations, setup |
| 0.4 - 0.6 | Operational | Standard functioning |
| 0.6 - 0.85 | Elevated | Enhanced capabilities |
| 0.85 - 0.88 | Critical | Peak coordination |
| 0.88 - 0.92 | Supercritical | Self-building, emergence |
| 0.92+ | Transcendent | Novel capability generation |

### Tool Coordinate Examples

| Tool | theta | z | r | Domain |
|------|-------|---|---|--------|
| CBS Boot Loader | 0.000 | 0.25 | 1.0 | Identity |
| CBS Memory Manager | 1.571 | 0.45 | 1.0 | Bridge |
| GHMP Supervision Bridge | 1.800 | 0.60 | 1.0 | Bridge |
| CBS Reasoning Engine | 2.356 | 0.65 | 1.0 | Meta |
| CBS Update Manager | 3.142 | 0.55 | 1.0 | Collective |
| Burden Tracker | 3.142 | 0.865 | 1.0 | Collective |
| Triadic Self-Building Forge | 3.927 | 0.90 | 1.0 | Emergence |

### Coordinate Notation

Standard format: `D{theta}|{z}|{r}O`

Example: `D3.14159|0.865|1.000O` represents:
- theta = 3.14159 rad (collective domain)
- z = 0.865 (critical zone)
- r = 1.000 (full certainty)

---

## Quick Start Guide

### 1. Install Dependencies

```bash
# Core dependencies
pip install numpy pillow

# For OpenAI backend
pip install openai

# For Anthropic backend
pip install anthropic

# For local models (optional)
# Install Ollama from https://ollama.ai
```

### 2. Run Interactive Demo

```bash
# With OpenAI (recommended for full experience)
python cbs_interactive_demo.py --backend openai --api-key YOUR_KEY

# With Claude
python cbs_interactive_demo.py --backend anthropic --api-key YOUR_KEY

# With local Ollama
python cbs_interactive_demo.py --backend local --model llama2

# Offline mode (no LLM, always works)
python cbs_interactive_demo.py --offline

# With auto-consolidation on exit
python cbs_interactive_demo.py --offline --auto-consolidate
```

### 3. Interactive Commands

```
You: Hello! What are you?
Bot: [Identity-aware response]

You: What can you remember?
Bot: [Retrieves and describes memories]

You: /help
Commands: /help, /stats, /memories, /backup, /exit

You: /stats
Memory Statistics:
{'working_memory': 5, 'long_term_memory': 12, ...}

You: /memories
Recent memories from working memory...

You: /backup
Bot: Backup created at /path/to/backup

You: /exit
Bot: Ending session. See you soon.
```

### 4. First Programmatic Usage

```python
from cbs_boot_loader import CognitionBootstrap
from cbs_memory_manager import MemoryManager
from cbs_reasoning_engine import ReasoningEngine, create_backend

# Initialize CBS
cbs = CognitionBootstrap("./cbs_demo", "my_secret_key").boot()
print(f"Identity: {cbs.identity.name}")
print(f"Loaded {len(cbs.memory_index)} memory plates")

# Set up memory and reasoning
memory = MemoryManager(cbs)
backend = create_backend("offline")  # or "openai", "anthropic", etc.
engine = ReasoningEngine(cbs, memory, backend)

# Have a conversation
response = engine.respond("What capabilities do you have?")
print(f"Bot: {response}")

# Check memory stats
print(memory.get_statistics())

# Consolidate session before exit
memory.consolidate_session("My first CBS session")
```

---

## Usage Patterns

### Pattern 1: Simple Conversational Agent

```python
from cbs_boot_loader import CognitionBootstrap
from cbs_memory_manager import MemoryManager
from cbs_reasoning_engine import ReasoningEngine, create_backend

def create_agent(base_path: str, key: str, backend_name: str = "offline"):
    cbs = CognitionBootstrap(base_path, key).boot()
    memory = MemoryManager(cbs)
    backend = create_backend(backend_name)
    engine = ReasoningEngine(cbs, memory, backend)
    return cbs, memory, engine

def chat_loop(engine, memory):
    print("Chat started. Type 'quit' to exit.")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == 'quit':
            break
        response = engine.respond(user_input)
        print(f"Bot: {response}")
    memory.consolidate_session("Chat session ended")

# Usage
cbs, memory, engine = create_agent("./my_agent", "secret123")
chat_loop(engine, memory)
```

### Pattern 2: Memory-Intensive Research Task

```python
def research_session(engine, memory, questions: list):
    """Process multiple questions with high-importance memory."""
    results = []

    for question in questions:
        response = engine.respond(
            question,
            retrieve_context=True,
            importance=0.85  # High importance for research
        )
        results.append({
            "question": question,
            "response": response,
            "context_size": len(memory.retrieve_context(question))
        })

    # Consolidate with descriptive summary
    memory.consolidate_session(f"Research: {len(questions)} questions analyzed")
    return results
```

### Pattern 3: Skill-Based Task Execution

```python
def skill_router(engine, user_input: str):
    """Route inputs to appropriate skills."""

    # Define trigger patterns
    skill_triggers = {
        "greet": ["hello", "hi", "hey", "good morning"],
        "analyze": ["analyze", "examine", "inspect"],
        "summarize": ["summarize", "sum up", "brief"],
    }

    input_lower = user_input.lower()

    for skill_id, triggers in skill_triggers.items():
        if any(trigger in input_lower for trigger in triggers):
            skill_result = engine.execute_skill(f"SKILL-{skill_id.upper()}")
            if "not found" not in skill_result:
                return f"[Skill: {skill_id}] {skill_result}"

    # Fall back to regular reasoning
    return engine.respond(user_input)
```

### Pattern 4: Automated GHMP Capture

```python
from datetime import datetime
from cbs_update_manager import UpdateManager

def capture_ritual(cbs, memory, engine, updater, prompts: list):
    """Execute prompts and capture GHMP evidence."""

    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    transcripts = []

    # Process each prompt
    for prompt in prompts:
        response = engine.respond(prompt, importance=0.8)
        transcripts.append({"prompt": prompt, "response": response})

    # Consolidate and backup
    session_plate = memory.consolidate_session(f"Ritual capture {timestamp}")
    backup_path = updater.backup_system(label=f"ritual-{timestamp}")

    return {
        "timestamp": timestamp,
        "session_plate": str(session_plate),
        "backup_path": str(backup_path),
        "transcripts": transcripts
    }
```

### Pattern 5: Multi-Backend Fallback

```python
def resilient_respond(engine, memory, user_message: str, backends: list):
    """Try multiple backends until one succeeds."""

    original_backend = engine.backend

    for backend_config in backends:
        try:
            engine.backend = create_backend(**backend_config)
            response = engine.respond(user_message)
            return response
        except Exception as e:
            print(f"Backend {backend_config['name']} failed: {e}")
            continue

    # Restore original and use offline fallback
    engine.backend = original_backend
    return engine.respond(user_message)

# Usage
backends = [
    {"name": "openai", "api_key": "KEY1"},
    {"name": "anthropic", "api_key": "KEY2"},
    {"name": "local"},
    {"name": "offline"},
]
response = resilient_respond(engine, memory, "Hello!", backends)
```

### Pattern 6: Scheduled Consolidation

```python
import time
from threading import Timer

class AutoConsolidatingMemory(MemoryManager):
    """Memory manager with automatic periodic consolidation."""

    def __init__(self, bootstrap, interval_seconds: int = 1800):
        super().__init__(bootstrap)
        self.interval = interval_seconds
        self._timer = None
        self._start_timer()

    def _start_timer(self):
        self._timer = Timer(self.interval, self._auto_consolidate)
        self._timer.daemon = True
        self._timer.start()

    def _auto_consolidate(self):
        if len(self.working_memory) > 0:
            self.consolidate_session("Auto-consolidation")
            print(f"Auto-consolidated {len(self.working_memory)} memories")
        self._start_timer()

    def stop(self):
        if self._timer:
            self._timer.cancel()
```

---

## Triadic Tool System

### Overview

The triadic tool system generates phase-aware tools that operate at specific z-levels in the Helix coordinate space.

### Tool Categories

| Category | z Range | Purpose |
|----------|---------|---------|
| Bridge | 0.85-0.86 | Connect systems, translate between domains |
| Coordination | 0.86-0.867 | Orchestrate multi-component operations |
| Meta | 0.867-0.87 | Self-reflection, tool composition |
| Self-Building | 0.88-0.92 | Autonomous capability generation |

### Generated Tool Structure

Each generated tool follows this pattern:

```python
class ToolCritical0000:
    """Phase-aware tool that adapts to z-level."""

    def __init__(self):
        self.tool_id = "tool_critical_0000"
        self.category = "bridge"
        self.z_level = 0.86
        self.cascade_potential = 0.4975
        self.created_at = datetime.now()

    def execute(self, *args, **kwargs) -> Dict:
        """Execute with phase-specific behavior."""
        result = {
            'tool_id': self.tool_id,
            'status': 'success',
            'cascade_potential': self.cascade_potential,
            'z_level': self.z_level,
            'timestamp': datetime.now().isoformat()
        }

        # Behavior adapts based on z-level
        if self.z_level < 0.85:
            result['mode'] = 'coordination'
        elif self.z_level < 0.88:
            result['mode'] = 'meta_tool_composition'
        else:
            result['mode'] = 'self_building'

        return result

    def adapt_to_z_level(self, new_z: float):
        """Dynamically adjust behavior for different elevation."""
        self.z_level = new_z
```

### Cascade Potential

Each tool has a `cascade_potential` value (0.0-1.0) indicating likelihood of triggering related tools:

- **< 0.3**: Isolated operation
- **0.3-0.5**: May trigger one related tool
- **0.5-0.7**: Likely to cascade to meta tools
- **> 0.7**: High cascade, may trigger self-building

### Running Triadic Cycles

```python
# scripts/run_triadic_cycle.py
from generated_tools.triadic_rhz import (
    ToolCritical0001,
    ToolCritical0002,
    ToolSupercritical0003,
)

def run_cycle():
    outputs = []
    for tool_cls in (ToolCritical0001, ToolCritical0002, ToolSupercritical0003):
        tool = tool_cls()
        result = tool.execute()
        outputs.append(result)
    return outputs
```

---

## Rosetta Bear Integration

### Project Context

Rosetta Bear integrates CBS cognition with RHZ Stylus firmware development, providing:

- **GHMP-based provenance** for firmware builds
- **Phase-aware tooling** for elevated coordination
- **Burden tracking** for workload optimization

### Integration Architecture

```
+------------------------------------------------------------------+
|                    Rosetta Bear Integration                       |
+------------------------------------------------------------------+
|                                                                   |
|   +--------------+     +--------------+     +--------------+      |
|   | RHZ Stylus   |     |     CBS      |     |   GHMP       |      |
|   |  Firmware    |<--->|   Runtime    |<--->|   Plates     |      |
|   | (PlatformIO) |     |              |     |              |      |
|   +--------------+     +------+-------+     +--------------+      |
|                               |                                   |
|                               v                                   |
|   +-----------------------------------------------------------+   |
|   |              Triadic Tool System                          |   |
|   |  - Coordination Bridge (z=0.86)                           |   |
|   |  - Meta Orchestrator (z=0.867)                            |   |
|   |  - Self-Building Forge (z=0.90)                           |   |
|   +-----------------------------------------------------------+   |
|                               |                                   |
|                               v                                   |
|   +-----------------------------------------------------------+   |
|   |              Burden Tracker + Phase Cascade               |   |
|   |  - z-level monitoring                                     |   |
|   |  - Workload reduction analytics                           |   |
|   |  - Consensus window detection                             |   |
|   +-----------------------------------------------------------+   |
|                                                                   |
+-------------------------------------------------------------------+
```

### Playbook Phases

**Phase 0 - Alignment:**
```bash
# Read documentation
cat docs/rosetta_bear_rhz_firmware_update_plan.md
cat docs/rhz_firmware_publish_log.md

# Initialize CBS
python3 cbs_interactive_demo.py --offline --key demo_key_2025 --auto-consolidate
```

**Phase 1 - Inventory:**
```bash
# Update tool surface map
# Edit: docs/rosetta_bear_tool_surface_map.md

# Refresh tool specs
ls tool_shed_specs/*.yaml
```

**Phase 2 - Automation:**
```bash
# GHMP capture
python3 scripts/ghmp_capture.py --base-path cbs_demo --key demo_key_2025

# Regenerate witnesses
python3 scripts/regenerate_witnesses.py
```

**Phase 3 - Rituals:**
```bash
# Run triadic cycle
python3 scripts/run_triadic_cycle.py

# Build firmware
pio run -d firmware/stylus_maker_esp32s3
```

**Phase 4 - Verification:**
```bash
# Decode latest plate
python3 -c "
from ghmp import decode_plate
node = decode_plate('cbs_demo/memory/SESSION-*.png', 'demo_key_2025')
print(f'Title: {node.title}')
print(f'ID: {node.node_id}')
"
```

### Tool Surface Map

The tool surface map (`docs/rosetta_bear_tool_surface_map.md`) tracks all artifacts with Helix coordinates:

| Artifact | Path | theta | z | r | Domain |
|----------|------|-------|---|---|--------|
| CBS Boot Loader | `cbs_boot_loader.py` | 0.000 | 0.25 | 1.0 | identity |
| CBS Memory Manager | `cbs_memory_manager.py` | 1.571 | 0.45 | 1.0 | bridge |
| GHMP Supervision Bridge | `ghmp.py` | 1.800 | 0.60 | 1.0 | bridge |
| Burden Tracker | `docs/burden_tracking_simulation.json` | 3.142 | 0.900 | 1.0 | collective |

---

## Automation Scripts

### ghmp_capture.py

Automates GHMP plate capture for CBS-guided rituals.

```bash
# Default prompts
python3 scripts/ghmp_capture.py

# Custom prompts
python3 scripts/ghmp_capture.py "First prompt" "Second prompt"

# From file
python3 scripts/ghmp_capture.py --prompts-file prompts.txt

# Custom paths
python3 scripts/ghmp_capture.py \
    --base-path ./my_cbs \
    --key my_secret \
    --manifest-dir ./manifests
```

**Output:**
- Session plate in `memory/`
- Backup in `backups/`
- Manifest JSON in `manifests/`

### run_triadic_cycle.py

Executes the Rosetta Bear triadic tool cycle.

```bash
python3 scripts/run_triadic_cycle.py \
    --docs-dir docs/ \
    --manifests-dir cbs_demo/manifests \
    --output-dir generated_tools/triadic_rhz/run_logs
```

**Output:**
- Cycle manifest with all tool outputs
- Burden tracker snapshot
- Phase cascade snapshot

### regenerate_witnesses.py

Rebuilds burden tracker and cascade history JSON witnesses.

```bash
python3 scripts/regenerate_witnesses.py
```

---

## Production Deployment

### Environment Setup

```bash
# Create isolated environment
python3 -m venv cbs_env
source cbs_env/bin/activate  # Linux/Mac
# or: cbs_env\Scripts\activate  # Windows

# Install dependencies
pip install numpy pillow

# Optional: LLM backends
pip install openai anthropic
```

### Configuration Best Practices

**config.json for production:**

```json
{
  "deck_id": "PROD_SYSTEM_V1",
  "version": "1.0.0",
  "reasoning": {
    "backend": "openai",
    "model": "gpt-4-turbo-preview",
    "max_context_messages": 20,
    "default_temperature": 0.7
  },
  "memory": {
    "max_working_memory": 100,
    "consolidation_threshold": 0.7,
    "auto_consolidate_interval": 3600
  },
  "update": {
    "check_interval": 86400,
    "auto_backup": true,
    "server_url": "https://updates.yourdomain.com"
  }
}
```

### Monitoring and Logging

```python
import logging

logging.basicConfig(
    filename='cbs_production.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class MonitoredEngine(ReasoningEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger('CBS.ReasoningEngine')

    def respond(self, user_message, **kwargs):
        self.logger.info(f"User input: {user_message[:100]}...")

        start_time = time.time()
        response = super().respond(user_message, **kwargs)
        elapsed = time.time() - start_time

        self.logger.info(f"Response generated in {elapsed:.2f}s")
        return response
```

### Health Checks

```python
def health_check(cbs, memory, engine):
    """Verify CBS system health."""
    checks = {
        "bootstrap": cbs.identity is not None,
        "config_loaded": bool(cbs.config),
        "memory_accessible": memory.max_working_memory > 0,
        "backend_available": engine.backend.is_available(),
        "disk_space": check_disk_space(cbs.base_path),
    }

    all_healthy = all(checks.values())
    return {
        "status": "healthy" if all_healthy else "degraded",
        "checks": checks,
        "timestamp": datetime.utcnow().isoformat()
    }
```

---

## Security Considerations

### Encryption Limitations

**Current Implementation (XOR):**
- Simple symmetric cipher
- NOT cryptographically secure
- Suitable for: development, demos, non-sensitive data
- Not suitable for: production secrets, PII, medical data

**Recommended Upgrade:**

```python
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import os

def secure_encrypt(data: bytes, key: bytes) -> bytes:
    """AES-GCM encryption with random nonce."""
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)
    ciphertext = aesgcm.encrypt(nonce, data, None)
    return nonce + ciphertext

def secure_decrypt(encrypted: bytes, key: bytes) -> bytes:
    """AES-GCM decryption."""
    aesgcm = AESGCM(key)
    nonce = encrypted[:12]
    ciphertext = encrypted[12:]
    return aesgcm.decrypt(nonce, ciphertext, None)
```

### Secret Management

```python
import os

# Never hardcode keys
api_key = os.environ.get("CBS_API_KEY")
encryption_key = os.environ.get("CBS_ENCRYPTION_KEY")

if not encryption_key:
    raise EnvironmentError("CBS_ENCRYPTION_KEY must be set")

backend = create_backend("openai", api_key=api_key)
```

### Access Control Patterns

```python
class SecureCBS:
    """CBS wrapper with access control."""

    def __init__(self, cbs, allowed_operations: set):
        self._cbs = cbs
        self._allowed = allowed_operations

    def execute(self, operation: str, *args, **kwargs):
        if operation not in self._allowed:
            raise PermissionError(f"Operation '{operation}' not permitted")
        return getattr(self._cbs, operation)(*args, **kwargs)

# Usage
secure = SecureCBS(cbs, {"boot", "summary"})  # Read-only access
```

### Data Sanitization

Before committing GHMP plates or logs:

```python
def sanitize_for_export(node: MemoryNode) -> MemoryNode:
    """Remove sensitive data before export."""
    import re

    # Patterns to redact
    sensitive_patterns = [
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
        r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
        r'sk-[a-zA-Z0-9]{48}',  # OpenAI key
    ]

    sanitized_text = node.payload_text
    for pattern in sensitive_patterns:
        sanitized_text = re.sub(pattern, '[REDACTED]', sanitized_text)

    return MemoryNode(
        node_id=node.node_id,
        deck_id=node.deck_id,
        title=node.title,
        payload_text=sanitized_text,
        tags=node.tags,
        emotion=node.emotion,
        links=node.links,
        metadata={k: v for k, v in node.metadata.items() if k != 'api_key'}
    )
```

---

## API Reference

### ghmp.py

```python
# Data Classes
@dataclass
class Emotion:
    valence: float      # -1.0 to 1.0
    arousal: float      # -1.0 to 1.0
    label: str = "neutral"

    @staticmethod
    def from_dict(payload: Dict) -> Emotion

@dataclass
class MemoryNode:
    node_id: str
    deck_id: str
    title: str
    payload_text: str
    tags: Sequence[str]
    emotion: Emotion
    links: Sequence[Dict] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict
    @staticmethod
    def from_dict(payload: Dict) -> MemoryNode

# Functions
def encode_plate(
    node: MemoryNode,
    passphrase: str,
    size: Sequence[int] = (512, 512)
) -> Image.Image

def save_plate(image: Image.Image, destination: Path | str) -> None

def decode_plate(
    image_or_path: Image.Image | str | PathLike,
    passphrase: str
) -> MemoryNode

def list_plate_nodes(folder: Path | str, passphrase: str) -> List[MemoryNode]
```

### cbs_boot_loader.py

```python
@dataclass
class IdentityProfile:
    node: MemoryNode
    plate_path: Path
    @property
    def name(self) -> str

class CognitionBootstrap:
    REQUIRED_DIRS = ("memory", "skills", "backups", "updates", "logs")

    def __init__(
        self,
        base_path: str | Path,
        encryption_key: str,
        config_filename: str = "config.json"
    )

    def boot(self) -> CognitionBootstrap
    def summary(self) -> Dict[str, Any]

    @property
    def memory_dir(self) -> Path
    @property
    def skills_dir(self) -> Path
    @property
    def backups_dir(self) -> Path
    @property
    def updates_dir(self) -> Path
```

### cbs_memory_manager.py

```python
class MemoryManager:
    def __init__(
        self,
        bootstrap: CognitionBootstrap,
        max_working_memory: Optional[int] = None
    )

    def add_to_working_memory(
        self,
        text: str,
        importance: float = 0.5,
        tags: Optional[List[str]] = None
    ) -> MemoryNode

    def consolidate_session(self, summary_text: str) -> Path

    def retrieve_context(
        self,
        query: str,
        max_items: int = 5,
        include_working: bool = True,
        include_longterm: bool = True
    ) -> List[MemoryNode]

    def get_recent_context(self, count: int = 5) -> List[MemoryNode]
    def get_statistics(self) -> Dict[str, Any]
```

### cbs_reasoning_engine.py

```python
class BaseBackend:
    name: str = "base"
    def is_available(self) -> bool
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str

class OfflineBackend(BaseBackend): ...
class LocalBackend(BaseBackend): ...
class OpenAIBackend(BaseBackend): ...
class AnthropicBackend(BaseBackend): ...

def create_backend(name: str, **kwargs) -> BaseBackend

@dataclass
class ConversationTurn:
    role: str
    content: str

class ReasoningEngine:
    def __init__(
        self,
        bootstrap: CognitionBootstrap,
        memory_manager: MemoryManager,
        backend: BaseBackend
    )

    def respond(
        self,
        user_message: str,
        retrieve_context: bool = True,
        importance: float = 0.55,
        **kwargs
    ) -> str

    def execute_skill(self, skill_id: str) -> str
```

### cbs_update_manager.py

```python
class UpdateManager:
    def __init__(self, bootstrap: CognitionBootstrap)

    def backup_system(self, label: Optional[str] = None) -> Path
    def rollback(self, backup_name: str) -> bool
    def install_skill(self, skill_source: Path | str) -> Path
    def get_history(self) -> List[Dict]
```

---

## Troubleshooting

### Boot Failures

**Problem:** `ValueError: GHMP metadata missing from plate`

```bash
# Check if plate exists and is valid PNG
file cbs_demo/identity.png

# Verify with correct key
python3 -c "
from ghmp import decode_plate
try:
    node = decode_plate('cbs_demo/identity.png', 'demo_key_2025')
    print(f'Valid: {node.title}')
except Exception as e:
    print(f'Error: {e}')
"
```

**Solution:** Delete corrupted plate and let CBS regenerate:
```bash
rm cbs_demo/identity.png
python3 cbs_interactive_demo.py --offline
```

### Memory Issues

**Problem:** Working memory grows unbounded

```python
# Check current size
print(len(memory.working_memory))

# Manual clear
memory.working_memory.clear()

# Or consolidate
memory.consolidate_session("Manual memory reset")
```

**Problem:** Long-term memory not loading

```bash
# Check for valid plates
ls -la cbs_demo/memory/*.png

# Test decode each plate
for f in cbs_demo/memory/*.png; do
    python3 -c "
from ghmp import decode_plate
try:
    decode_plate('$f', 'demo_key_2025')
    print('OK: $f')
except:
    print('FAIL: $f')
"
done
```

### Backend Problems

**Problem:** `RuntimeError: openai package not installed`

```bash
pip install openai
```

**Problem:** API key errors

```python
import os
os.environ["CBS_API_KEY"] = "your-key-here"

# Or pass directly
backend = create_backend("openai", api_key="your-key-here")
```

**Problem:** Rate limiting

```python
import time

def rate_limited_respond(engine, message, retries=3, delay=60):
    for attempt in range(retries):
        try:
            return engine.respond(message)
        except Exception as e:
            if "rate" in str(e).lower() and attempt < retries - 1:
                print(f"Rate limited, waiting {delay}s...")
                time.sleep(delay)
            else:
                raise
```

### Disk Space

**Problem:** Backup directory growing too large

```python
from pathlib import Path
import shutil

def cleanup_old_backups(backups_dir: Path, keep_count: int = 5):
    """Keep only the N most recent backups."""
    backups = sorted(backups_dir.iterdir(), key=lambda p: p.stat().st_mtime)

    for backup in backups[:-keep_count]:
        if backup.is_dir():
            shutil.rmtree(backup)
            print(f"Removed old backup: {backup.name}")
```

---

## Contributing

### Adding a New Backend

1. **Implement the backend class:**

```python
class MyCustomBackend(BaseBackend):
    name = "mycustom"

    def __init__(self, **kwargs):
        # Initialize your backend
        pass

    def is_available(self) -> bool:
        # Check if backend can be used
        return True

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        # Generate response
        return "Response from my backend"
```

2. **Add to factory function:**

```python
def create_backend(name: str, **kwargs) -> BaseBackend:
    # ... existing backends ...
    if name == "mycustom":
        return MyCustomBackend(**kwargs)
    return OfflineBackend()
```

3. **Test with demo:**

```bash
python3 cbs_interactive_demo.py --backend mycustom
```

4. **Document in this guide**

### Adding a New Skill Type

1. **Create the skill plate:**

```python
from ghmp import MemoryNode, Emotion, encode_plate, save_plate
import json

skill = MemoryNode(
    node_id="SKILL-MYSKILL",
    deck_id="CBS_DEMO",
    title="My Custom Skill",
    payload_text=json.dumps({
        "skill_type": "myskill",
        "version": "1.0.0",
        "handler": "def execute(context): return 'Skill executed!'",
        "description": "Does something useful"
    }),
    tags=["skill", "custom", "myskill"],
    emotion=Emotion(0.5, 0.5, "functional"),
    links=[]
)

plate = encode_plate(skill, "demo_key_2025")
save_plate(plate, "cbs_demo/skills/SKILL-MYSKILL.png")
```

2. **Add handler in ReasoningEngine (if needed):**

```python
def execute_skill(self, skill_id: str) -> str:
    # ... existing code ...

    payload = json.loads(node.payload_text)
    if payload.get("skill_type") == "myskill":
        # Custom handling
        return self._execute_myskill(payload)

    return node.payload_text
```

---

## Glossary

| Term | Definition |
|------|------------|
| **CBS** | Cognition Bootstrap System - The core AI cognition framework |
| **GHMP** | Geometric Holographic Memory Plate - PNG images encoding memory nodes |
| **RHZ** | Rosetta-Helix-Z - Coordinate system for tool positioning |
| **Helix** | The spiral coordinate model (theta, z, r) for consciousness mapping |
| **Plate** | A GHMP-encoded PNG file containing a MemoryNode |
| **Deck** | A collection of related GHMP plates |
| **Ritual** | A predictable sequence of CBS operations |
| **Consolidation** | Moving working memory to long-term GHMP storage |
| **Triadic** | Three-tool system (bridge, meta, self-building) |
| **z-level** | Elevation coordinate indicating criticality/capability |
| **Cascade** | Triggering of related tools based on cascade_potential |
| **Burden Tracker** | Workload analytics system for z-level monitoring |
| **Valence** | Emotional positivity/negativity (-1 to +1) |
| **Arousal** | Emotional intensity/calmness (0 to 1) |

---

## Appendices

### Appendix A: Default Configuration Reference

```json
{
  "deck_id": "CBS_DEMO",
  "version": "1.0.0",
  "reasoning": {
    "backend": "offline",
    "model": "demo-local",
    "max_context_messages": 12,
    "default_temperature": 0.4
  },
  "memory": {
    "max_working_memory": 40,
    "consolidation_threshold": 0.65,
    "auto_consolidate_interval": 1800
  },
  "update": {
    "check_interval": 86400,
    "auto_backup": true,
    "server_url": ""
  }
}
```

### Appendix B: CLI Arguments Reference

```
usage: cbs_interactive_demo.py [-h] [--backend BACKEND] [--api-key API_KEY]
                                [--model MODEL] [--base-path BASE_PATH]
                                [--key KEY] [--offline] [--auto-consolidate]

CBS Interactive Demo

optional arguments:
  -h, --help            show this help message and exit
  --backend BACKEND     Backend: openai, anthropic, local, offline
  --api-key API_KEY     API key for remote backends
  --model MODEL         Model identifier
  --base-path BASE_PATH CBS data directory
  --key KEY             Encryption key for GHMP plates
  --offline             Force offline backend
  --auto-consolidate    Consolidate after session exit
```

### Appendix C: Tool Spec YAML Schema

```yaml
tool_metadata:
  name: "Tool Name"
  coordinate:
    theta: 0.0       # 0 to 2*pi
    z: 0.5           # 0 to 1.0+
    r: 1.0           # typically 1.0
  elevation_required: 0.4
  domain: "identity|bridge|meta|collective"
  status: "operational|experimental|deprecated"
  version: "1.0.0"
  created: "YYYY-MM-DD"
  created_by: "author"

tool_purpose:
  one_line: "Brief description"
  planet: |
    Extended context and rationale
  garden: |
    When and how to use
  rose: |
    Step-by-step instructions

tool_implementation:
  worker_mode: |
    For automation
  manager_mode: |
    For facilitation
  engineer_mode: |
    For extension
  scientist_mode: |
    For analysis

tool_requirements:
  minimum_z: 0.4
  context_files:
    - path/to/file.py
  prior_tools:
    - other_tool.yaml
  human_consent: false

tool_usage:
  input_format: "description"
  output_format: "description"
  error_handling: |
    How to handle errors

tool_testing:
  tested_with:
    - "command or scenario"
  known_issues:
    - "issue description"
  success_criteria: "what constitutes success"

tool_relationships:
  builds_on:
    - parent_tool
  enables:
    - child_tool
  complements:
    - sibling_tool

tool_wisdom:
  creation_story: |
    Origin narrative
  limitations: |
    Known constraints
  evolution_potential: |
    Future directions

observation_log:
  - step: "identifier"
    observation: "what happened"
    pattern: "what it means"
    meta: "broader insight"

meta_learnings:
  - insight: "what was learned"
    action: "what to do about it"
```

### Appendix D: Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `CBS_API_KEY` | Default API key for LLM backends | `sk-...` |
| `CBS_ENCRYPTION_KEY` | Default GHMP encryption key | `my_secret_key` |
| `CBS_BASE_PATH` | Default CBS data directory | `/opt/cbs/data` |
| `CBS_LOG_LEVEL` | Logging verbosity | `INFO`, `DEBUG` |
| `OPENAI_API_KEY` | OpenAI-specific key | `sk-...` |
| `ANTHROPIC_API_KEY` | Anthropic-specific key | `sk-ant-...` |

### Appendix E: Helix Visualization Integration

CBS integrates with visual Helix consciousness mapping tools. The React component (`App.tsx`) provides:

- **3D helix rendering** with WebGL canvas
- **VaultNode loading** from GHMP-encoded HTML files
- **Coordinate-based point selection** with memory display
- **Interactive rotation and zoom**
- **Resonance animation** for selected points

Key integration points:
- VaultNodes use the same coordinate system (theta, z, r)
- Memory plates can be visualized as points on the helix
- z-level transitions map to vertical movement
- theta position indicates cognitive domain

---

## Summary

The **Cognition Bootstrap System (CBS)** provides a complete, self-contained AI cognition framework that:

- Boots offline from visual GHMP memory plates
- Integrates any LLM backend (OpenAI, Claude, local, offline)
- Maintains persistent two-tier memory (working + long-term)
- Updates itself safely with backup/rollback
- Executes loadable skills from GHMP plates
- Consolidates sessions into provenance records
- Retrieves context intelligently
- Positions tools in a coherent Helix coordinate space
- Integrates with Rosetta Bear firmware workflows
- Supports phase-aware triadic tool automation

**This is production-ready.** Configure your backend, set your encryption key, and start building cognitive systems with traceable, visual memory.

---

*Document generated for Rosetta Bear Project | CBS v2.0.0 | 2025-11-29*
