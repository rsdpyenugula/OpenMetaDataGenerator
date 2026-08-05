"""Concrete LLM backends: OpenAI, Anthropic, AWS Bedrock, Google Vertex, Azure OpenAI,
and the Gemini Developer API.

Each backend reads its credentials from the standard provider environment variables
(e.g. ``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``, ``GEMINI_API_KEY``, AWS/GCP default
credential chains). The generator only depends on
:class:`~openmetadatagenerator.llm.base.LLMBackend`, so adding a provider is a matter of
implementing ``generate``.
"""
from __future__ import annotations

import os

from .base import LLMBackend

_DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-sonnet-latest",
    "bedrock": "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "vertex": "gemini-1.5-pro",
    "azure": "gpt-4o-mini",
    "gemini": "gemini-2.5-flash",
}


class OpenAIBackend(LLMBackend):
    name = "openai"

    def __init__(self, model: str = "", **kw):
        super().__init__(model or _DEFAULT_MODELS["openai"], **kw)
        from openai import OpenAI
        self._client = OpenAI()

    def generate(self, prompt: str, system: str = "", temperature: float | None = None) -> str:
        msgs = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
        r = self._client.chat.completions.create(
            model=self.model, messages=msgs, max_tokens=self.max_tokens,
            temperature=self.temperature if temperature is None else temperature)
        return (r.choices[0].message.content or "").strip()


class AnthropicBackend(LLMBackend):
    name = "anthropic"

    def __init__(self, model: str = "", **kw):
        super().__init__(model or _DEFAULT_MODELS["anthropic"], **kw)
        import anthropic
        self._client = anthropic.Anthropic()

    def generate(self, prompt: str, system: str = "", temperature: float | None = None) -> str:
        r = self._client.messages.create(
            model=self.model, max_tokens=self.max_tokens,
            temperature=self.temperature if temperature is None else temperature,
            system=system or None, messages=[{"role": "user", "content": prompt}])
        return "".join(b.text for b in r.content if getattr(b, "type", "") == "text").strip()


class BedrockBackend(LLMBackend):
    name = "bedrock"

    def __init__(self, model: str = "", region: str | None = None, **kw):
        super().__init__(model or _DEFAULT_MODELS["bedrock"], **kw)
        import boto3
        self._client = boto3.client("bedrock-runtime",
                                    region_name=region or os.environ.get("AWS_REGION", "us-east-1"))

    def generate(self, prompt: str, system: str = "", temperature: float | None = None) -> str:
        # Uses the provider-agnostic Bedrock Converse API.
        r = self._client.converse(
            modelId=self.model,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            system=[{"text": system}] if system else [],
            inferenceConfig={"maxTokens": self.max_tokens,
                             "temperature": self.temperature if temperature is None else temperature})
        return "".join(b.get("text", "") for b in r["output"]["message"]["content"]).strip()


class VertexBackend(LLMBackend):
    name = "vertex"

    def __init__(self, model: str = "", **kw):
        super().__init__(model or _DEFAULT_MODELS["vertex"], **kw)
        import vertexai
        from vertexai.generative_models import GenerativeModel
        vertexai.init(project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
                      location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"))
        self._GenerativeModel = GenerativeModel

    def generate(self, prompt: str, system: str = "", temperature: float | None = None) -> str:
        model = self._GenerativeModel(self.model, system_instruction=system or None)
        r = model.generate_content(
            prompt, generation_config={"max_output_tokens": self.max_tokens,
                                       "temperature": self.temperature if temperature is None else temperature})
        return (r.text or "").strip()


class GeminiBackend(LLMBackend):
    """Google Gemini via the Gemini Developer API (``google-genai`` SDK).

    Reads the API key from ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``) in the environment;
    the client never sees the key from source. Use ``vertex`` instead for Gemini served
    through Google Cloud Vertex AI with project-based auth.
    """
    name = "gemini"

    def __init__(self, model: str = "", **kw):
        # Gemini "flash/pro" are thinking models: hidden reasoning tokens count against
        # the output budget, so a small max_tokens truncates the JSON. Give generous
        # headroom unless the caller overrides.
        kw.setdefault("max_tokens", 8192)
        super().__init__(model or _DEFAULT_MODELS["gemini"], **kw)
        from google import genai
        self._genai = genai
        # api_key defaults to $GEMINI_API_KEY / $GOOGLE_API_KEY.
        self._client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY")
                                    or os.environ.get("GOOGLE_API_KEY"))

    def generate(self, prompt: str, system: str = "", temperature: float | None = None) -> str:
        from google.genai import types
        cfg = types.GenerateContentConfig(
            system_instruction=system or None,
            temperature=self.temperature if temperature is None else temperature,
            max_output_tokens=self.max_tokens)
        r = self._client.models.generate_content(model=self.model, contents=prompt, config=cfg)
        return (r.text or "").strip()


class AzureOpenAIBackend(LLMBackend):
    name = "azure"

    def __init__(self, model: str = "", **kw):
        super().__init__(model or os.environ.get("AZURE_OPENAI_DEPLOYMENT", _DEFAULT_MODELS["azure"]), **kw)
        from openai import AzureOpenAI
        self._client = AzureOpenAI(
            api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-06-01"))

    def generate(self, prompt: str, system: str = "", temperature: float | None = None) -> str:
        msgs = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
        r = self._client.chat.completions.create(
            model=self.model, messages=msgs, max_tokens=self.max_tokens,
            temperature=self.temperature if temperature is None else temperature)
        return (r.choices[0].message.content or "").strip()
