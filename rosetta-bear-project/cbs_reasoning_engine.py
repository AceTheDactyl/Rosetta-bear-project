"""
Reasoning engine and backend adapters for the CBS runtime.

Supports OpenAI, Anthropic, local templated, and offline backends. Falls back
to a deterministic local backend if remote providers are unavailable so that
the interactive demo always works.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ghmp import decode_plate

from cbs_memory_manager import MemoryManager


# ------------------------------------------------------------------- backends
class BaseBackend:
    """Minimal interface shared by all LLM backends."""

    name = "base"

    def is_available(self) -> bool:
        return True

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        raise NotImplementedError


class OfflineBackend(BaseBackend):
    name = "offline"

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        prompt = messages[-1]["content"]
        context = kwargs.get("memory_context") or []
        context_preview = " | ".join(context[:2])
        return (
            "Offline reasoning path active. "
            f"I reflected on your message: '{prompt}'. "
            f"Recent context: {context_preview or 'no stored memories yet.'}"
        )


class LocalBackend(BaseBackend):
    name = "local"

    def __init__(self, creativity: float = 0.35):
        self.creativity = creativity

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        user_text = messages[-1]["content"]
        context = kwargs.get("memory_context") or []
        temperature = kwargs.get("temperature", self.creativity)

        seeds = [
            "I retrieved a GHMP plate that mirrors this.",
            "Linking to prior session insight.",
            "Synthesizing perception + memory alignment.",
            "Carrying forward the CBS ritual of reflection.",
        ]
        seed = random.choice(seeds)
        context_text = " ".join(context[:1]) if context else "Fresh slate."

        return (
            f"{seed} You asked: {user_text}. "
            f"Context window: {context_text}. "
            f"Creative temperature: {temperature:.2f}. "
            "Let's progress deliberately."
        )


class OpenAIBackend(BaseBackend):
    name = "openai"

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional dep
            raise RuntimeError("openai package not installed") from exc

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        response = self.client.chat.completions.create(model=self.model, messages=messages)
        return response.choices[0].message.content.strip()


class AnthropicBackend(BaseBackend):
    name = "anthropic"

    def __init__(self, api_key: str, model: str = "claude-3-haiku-20240307"):
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - optional dep
            raise RuntimeError("anthropic package not installed") from exc

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        # Convert OpenAI-style messages to Anthropic format
        system = ""
        history = []
        for msg in messages:
            role = msg["role"]
            if role == "system":
                system = msg["content"]
            else:
                history.append({"role": role, "content": msg["content"]})

        response = self.client.messages.create(
            model=self.model,
            system=system,
            max_tokens=500,
            messages=history,
        )
        return response.content[0].text.strip()


def create_backend(name: str, **kwargs) -> BaseBackend:
    """Factory returning a backend instance by name."""
    name = (name or "offline").lower()
    if name == "openai":
        return OpenAIBackend(api_key=kwargs.get("api_key"), model=kwargs.get("model", "gpt-4o-mini"))
    if name == "anthropic":
        return AnthropicBackend(api_key=kwargs.get("api_key"), model=kwargs.get("model", "claude-3-haiku-20240307"))
    if name == "local":
        return LocalBackend(creativity=kwargs.get("temperature", 0.35))
    return OfflineBackend()


# ------------------------------------------------------------------- reasoning
@dataclass
class ConversationTurn:
    role: str
    content: str


class ReasoningEngine:
    """Coordinates message history, memory lookups, and backend calls."""

    def __init__(self, bootstrap, memory_manager: MemoryManager, backend: BaseBackend):
        self.bootstrap = bootstrap
        self.memory_manager = memory_manager
        self.backend = backend
        self.history: List[ConversationTurn] = []
        identity = bootstrap.identity.node.title if bootstrap.identity else "CBS Agent"
        self.system_prompt = (
            f"You are {identity}. Uphold the rituals of thoughtful CBS operation: "
            "ground responses in stored memory when possible, describe actions, "
            "and remain transparent about available capabilities."
        )

    # ----------------------------------------------------------------- exposed
    def respond(
        self,
        user_message: str,
        retrieve_context: bool = True,
        importance: float = 0.55,
        **kwargs,
    ) -> str:
        context_nodes = []
        if retrieve_context:
            context_nodes = self.memory_manager.retrieve_context(user_message, max_items=4)
        context_text = [node.payload_text for node in context_nodes]

        messages = [{"role": "system", "content": self.system_prompt}]
        for turn in self.history[-6:]:
            messages.append({"role": turn.role, "content": turn.content})
        messages.append({"role": "user", "content": user_message})

        try:
            reply = self.backend.generate(messages, memory_context=context_text, **kwargs)
        except Exception as exc:  # pragma: no cover - fallback path
            fallback = OfflineBackend()
            reply = fallback.generate(messages, memory_context=context_text)
            reply += f"\n\n[Fallback active: {exc}]"

        self.history.append(ConversationTurn("user", user_message))
        self.history.append(ConversationTurn("assistant", reply))
        self.memory_manager.add_to_working_memory(f"User: {user_message}", importance=importance)
        self.memory_manager.add_to_working_memory(f"Assistant: {reply}", importance=importance * 0.9)
        return reply

    def execute_skill(self, skill_id: str) -> str:
        """Load a GHMP skill plate and return its payload."""
        plate_path = next((p for p in self.bootstrap.skills_dir.glob("*.png") if skill_id in p.stem), None)
        if not plate_path:
            return f"Skill {skill_id} not found."
        node = decode_plate(plate_path, self.bootstrap.encryption_key)
        return node.payload_text


__all__ = ["ReasoningEngine", "create_backend"]
