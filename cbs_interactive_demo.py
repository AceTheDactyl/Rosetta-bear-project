#!/usr/bin/env python3
"""
Interactive CBS demo CLI.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from cbs_boot_loader import CognitionBootstrap
from cbs_memory_manager import MemoryManager
from cbs_reasoning_engine import ReasoningEngine, create_backend
from cbs_update_manager import UpdateManager


def parse_args() -> argparse.Namespace:
    default_base = Path(__file__).resolve().parent / "cbs_demo"
    parser = argparse.ArgumentParser(description="CBS Interactive Demo")
    parser.add_argument("--backend", default="offline", help="Backend: openai, anthropic, local, offline")
    parser.add_argument("--api-key", default=os.environ.get("CBS_API_KEY"), help="API key for remote backends")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model identifier")
    parser.add_argument("--base-path", default=str(default_base), help="CBS data directory")
    parser.add_argument("--key", default="demo_key_2025", help="Encryption key for GHMP plates")
    parser.add_argument("--offline", action="store_true", help="Force offline backend")
    parser.add_argument("--auto-consolidate", action="store_true", help="Consolidate after session exit")
    return parser.parse_args()


def main():
    args = parse_args()
    backend_name = "offline" if args.offline else args.backend

    bootstrap = CognitionBootstrap(base_path=args.base_path, encryption_key=args.key).boot()
    memory = MemoryManager(bootstrap)
    backend = create_backend(backend_name, api_key=args.api_key, model=args.model)
    engine = ReasoningEngine(bootstrap, memory, backend)
    update_mgr = UpdateManager(bootstrap)

    print("CBS Interactive Demo")
    print("--------------------")
    summary = bootstrap.summary()
    print(f"Identity: {summary['identity']}")
    print(f"Base path: {summary['base_path']}")
    print("Type '/help' for commands.\n")

    try:
        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue

            if user_input.lower() in ("/quit", "/exit"):
                print("Bot: Ending session. See you soon.")
                break
            if user_input.lower() == "/help":
                print("Commands: /help, /stats, /memories, /backup, /exit")
                continue
            if user_input.lower() == "/stats":
                print(memory.get_statistics())
                continue
            if user_input.lower() == "/memories":
                recent = memory.get_recent_context(5)
                for node in recent:
                    print(f"- {node.node_id}: {node.payload_text[:80]}")
                continue
            if user_input.lower() == "/backup":
                path = update_mgr.backup_system()
                print(f"Bot: Backup created at {path}")
                continue

            response = engine.respond(user_input, retrieve_context=True)
            print(f"Bot: {response}")

    except KeyboardInterrupt:
        print("\nBot: Session interrupted. Goodbye.")
    finally:
        if args.auto_consolidate:
            memory.consolidate_session("Interactive demo termination")


if __name__ == "__main__":
    main()
