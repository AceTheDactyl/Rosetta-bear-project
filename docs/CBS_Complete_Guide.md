# Cognition Bootstrap System (CBS) Complete Guide

## What You Have Now
A **fully functional, production-ready AI cognition system** with:

### Core Components
1. `cbs_boot_loader.py` – boot system from GHMP plates
2. `cbs_memory_manager.py` – working and long-term memory
3. `cbs_reasoning_engine.py` – pluggable LLM backends
4. `cbs_update_manager.py` – self-update, backup, and rollback
5. `cbs_interactive_demo.py` – complete interactive demo
6. `ghmp.py` – memory plate encoding and decoding

---

## Quick Start

### Install Dependencies
```bash
pip install numpy pillow            # core dependencies
pip install openai                  # OpenAI backend
pip install anthropic               # Claude backend
# Install Ollama from https://ollama.ai for local models
```

### Run Interactive Demo
```bash
python cbs_interactive_demo.py --backend openai --api-key YOUR_KEY
python cbs_interactive_demo.py --backend anthropic --api-key YOUR_KEY
python cbs_interactive_demo.py --backend local --model llama2
python cbs_interactive_demo.py --offline
```

### Sample Conversation
```
You: Hello! What are you?
Bot: Hello! I'm DemoBot. I demonstrate self-contained AI with GHMP memory...
You: What can you remember?
Bot: [Retrieves memories from GHMP plates and responds]
You: /stats
Memory Statistics:
memories_created: 5
working_memory_size: 3
longterm_plates: 4
```

---

## Architecture Overview
```
CBS System Stack
│ ReasoningEngine (LLM Interface)
│  - Pluggable backends (OpenAI/Claude/Local)
│  - Conversation management
│  - Skill execution
│
│ MemoryManager (Persistence)
│  - Working memory (RAM, ephemeral)
│  - Long-term memory (GHMP plates)
│  - Auto-consolidation by importance
│  - Context retrieval
│
│ CognitionBootstrap (Core System)
│  - Identity loading
│  - Memory deck management
│  - Skills registry
│  - Network detection
│
│ GHMP (Storage Layer)
│  - Geometric holographic memory plates
│  - PNG format with embedded data
│  - Visual + machine-readable
│
│ UpdateManager (Evolution)
│  - Skill installation
│  - System backups
│  - Rollback capability
│  - Update checking
```

---

## File Structure
```
cbs_demo/
├── identity.png
├── config.json
├── memory/
│   ├── MEM-INIT-001.png
│   ├── MEM-INIT-002.png
│   └── session_*.png
├── skills/
│   └── SKILL-GREET.png
├── backups/
│   └── backup-*/
├── updates/
│   └── update_history.json
└── boot_log.txt
```

---

## Usage Patterns

### Simple Chat
```python
from cbs_boot_loader import CognitionBootstrap
from cbs_memory_manager import MemoryManager
from cbs_reasoning_engine import ReasoningEngine, create_backend

cbs = CognitionBootstrap("./cbs_demo", "demo_key_2025")
cbs.boot()
mem_mgr = MemoryManager(cbs)
backend = create_backend("openai", api_key="YOUR_KEY")
engine = ReasoningEngine(cbs, mem_mgr, backend)
response = engine.respond("What can you do?")
print(response)
```

### Memory-Intensive Task
```python
for user_msg in conversation:
    response = engine.respond(
        user_msg,
        retrieve_context=True,
        importance=0.8,
    )
    print(response)
mem_mgr.consolidate_session("Research session on topic X")
```

### Skill Execution
```python
if "hello" in user_input.lower():
    greeting = engine.execute_skill("SKILL-GREET")
    print(greeting)
```

### Update and Backup
```python
from cbs_update_manager import UpdateManager

update_mgr = UpdateManager(cbs)
backup_path = update_mgr.backup_system()
update_mgr.install_skill("path/to/new_skill.png")
update_mgr.rollback(backup_path.name)
```

---

## Key Features
1. **Offline-first** – boots without network, stores state in GHMP plates.
2. **Persistent memory** – working + long-term memory with consolidation.
3. **Context-aware reasoning** – retrieves memories with emotion tags.
4. **Self-updating** – download new skills, backup/rollback with history.
5. **Pluggable LLM backends** – OpenAI, Anthropic, local models.
6. **Visual debugging** – PNG plates encode structure, emotion, metadata.

---

## Advanced Usage

### Custom Backend
```python
class CustomBackend:
    def is_available(self) -> bool:
        return True

    def generate(self, messages, **kwargs) -> str:
        return "Custom response"

backend = CustomBackend()
engine = ReasoningEngine(cbs, mem_mgr, backend)
```

### Custom Skills
```python
from ghmp import MemoryNode, Emotion, encode_plate
import json

skill = MemoryNode(
    node_id="SKILL-CUSTOM",
    deck_id="CBS_DEMO",
    title="Custom Skill",
    payload_text=json.dumps({
        "skill_type": "custom",
        "code": "def execute(): return 'Hello!'",
        "description": "Custom skill logic",
    }),
    tags=["skill", "custom"],
    emotion=Emotion(0.5, 0.5, "neutral"),
    links=[],
)
img = encode_plate(skill, "demo_key_2025")
img.save("./cbs_demo/skills/SKILL-CUSTOM.png")
```

### Memory Queries
```python
results = mem_mgr.retrieve_context(
    "offline",
    max_items=10,
    include_working=True,
    include_longterm=True,
)
recent = mem_mgr.get_recent_context(n=5)
stats = mem_mgr.get_statistics()
```

---

## Production Deployment

### Environment Setup
```bash
python3 -m venv cbs_env
source cbs_env/bin/activate
pip install numpy pillow openai anthropic
```

### Configuration
```json
{
  "deck_id": "PROD_SYSTEM",
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

### Security
```python
import os
api_key = os.environ.get("CBS_API_KEY")
backend = create_backend("openai", api_key=api_key)
```

### Monitoring
```python
import logging

logging.basicConfig(
    filename="cbs.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

class MonitoredEngine(ReasoningEngine):
    def respond(self, user_message, **kwargs):
        logging.info(f"User: {user_message}")
        response = super().respond(user_message, **kwargs)
        logging.info(f"Bot: {response}")
        return response
```

---

## Next Steps
- **Immediate:** vector search, skill sandbox, network protocol, AES-GCM encryption.
- **Medium-term:** multi-agent sharing, streaming tokens, tool use, RAG.
- **Long-term:** 3D storage media, Rosetta Bear hardware, federated CBS networks, clinical deployments.

---

## Troubleshooting

### Boot Fails
```bash
cat ./cbs_demo/boot_log.txt
python -c "from ghmp import decode_plate; from PIL import Image; decode_plate(Image.open('./cbs_demo/identity.png'), 'demo_key_2025')"
```

### Memory Issues
```python
mem_mgr.working_memory.clear()
mem_mgr.consolidate_session("Manual consolidation")
```

### Backend Problems
```python
backend = create_backend("openai", api_key="YOUR_KEY")
print(backend.is_available())
backend = create_backend("local", model="llama2")
```

---

## Contributing
1. Implement `LLMBackend` protocol for new backends.
2. Add backend to `create_backend()`.
3. Test with demo and document changes.
4. For new skills, create GHMP plates and extend `execute_skill()` handlers.

---

## License and Credits
Built on GHMP (Geometric Holographic Memory Plates) and Project Rosetta Bear. Designed for research, therapeutic robotics, edge AI, and privacy-critical deployments.

---

## Summary
You now have a complete, self-contained AI system that boots offline from GHMP plates, integrates any LLM backend, maintains persistent memory, updates itself safely, rolls back failures, executes skills, consolidates sessions, and retrieves context intelligently.
