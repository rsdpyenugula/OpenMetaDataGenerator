"""LLM backend abstraction.

A backend maps a (system, user) prompt to a text completion. Concrete backends wrap
a single provider SDK and are import-lazy so users install only the extras they need.
Determinism-friendly defaults (temperature 0, fixed seed where supported) keep the
generation loop reproducible for evaluation.
"""
from __future__ import annotations

import abc
import time
from concurrent.futures import ThreadPoolExecutor


class LLMBackend(abc.ABC):
    name: str = "base"

    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 1024,
                 retries: int = 4):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retries = retries

    @abc.abstractmethod
    def generate(self, prompt: str, system: str = "", temperature: float | None = None) -> str:
        raise NotImplementedError

    def _generate_resilient(self, prompt: str, system: str, temperature: float | None) -> str:
        """Call ``generate`` with exponential backoff on transient provider errors
        (rate limits, 5xx, timeouts). After the final attempt the object is left
        undescribed (returns "") rather than crashing a long run -- the coverage loop
        will pick it up on a later pass."""
        for attempt in range(self.retries):
            try:
                return self.generate(prompt, system, temperature)
            except Exception:
                if attempt == self.retries - 1:
                    return ""
                time.sleep(min(30.0, 1.5 * (2 ** attempt)))
        return ""

    def batch(self, prompts: list[str], system: str = "",
              temperature: float | None = None, workers: int = 8) -> list[str]:
        """Concurrent, order-preserving completion of many prompts. Individual failures
        degrade to "" (after retries) so one transient error cannot abort the batch."""
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(
                lambda p: self._generate_resilient(p, system, temperature), prompts))
