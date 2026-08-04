"""LLM backends and a provider factory."""
from __future__ import annotations

from .base import LLMBackend


def get_backend(provider: str, model: str = "", **kwargs) -> LLMBackend:
    """Instantiate an LLM backend by provider name."""
    provider = provider.lower()
    from . import backends as b
    table = {
        "openai": b.OpenAIBackend,
        "anthropic": b.AnthropicBackend,
        "bedrock": b.BedrockBackend,
        "vertex": b.VertexBackend,
        "azure": b.AzureOpenAIBackend,
        "gemini": b.GeminiBackend,
    }
    if provider not in table:
        raise ValueError(f"unknown LLM provider: {provider!r} "
                         f"(expected one of {sorted(table)})")
    return table[provider](model=model, **kwargs)


__all__ = ["LLMBackend", "get_backend"]
