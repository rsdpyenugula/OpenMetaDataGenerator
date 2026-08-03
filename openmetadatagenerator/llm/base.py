"""LLM backend abstraction.

A backend maps a (system, user) prompt to a text completion. Concrete backends wrap
a single provider SDK and are import-lazy so users install only the extras they need.
Determinism-friendly defaults (temperature 0, fixed seed where supported) keep the
generation loop reproducible for evaluation.
"""
from __future__ import annotations

import abc
from concurrent.futures import ThreadPoolExecutor


class LLMBackend(abc.ABC):
    name: str = "base"

    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 1024):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abc.abstractmethod
    def generate(self, prompt: str, system: str = "", temperature: float | None = None) -> str:
        raise NotImplementedError

    def batch(self, prompts: list[str], system: str = "",
              temperature: float | None = None, workers: int = 8) -> list[str]:
        """Concurrent completion of many prompts, order preserved."""
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(lambda p: self.generate(p, system, temperature), prompts))
