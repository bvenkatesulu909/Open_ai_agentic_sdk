"""
config.py — Model provider wiring for the OpenAI Agents SDK demo.

We run the official OpenAI Agents SDK but point it at OpenRouter as the
LLM backend, so you can use any OpenRouter model (gpt-4o, claude, llama,
deepseek, qwen, ...) with one API key.

Key points:
  * The SDK calls OpenAI-compatible Chat Completions endpoints.
  * OpenRouter exposes an OpenAI-compatible API at https://openrouter.ai/api/v1
  * We build an AsyncOpenAI client pointed at that base_url and wrap it in
    OpenAIChatCompletionsModel so the Agent can use ANY model string.
"""
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import (
    OpenAIChatCompletionsModel,
    ModelProvider,
    Model,
)

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
)
# Default model used across agents. Override per-agent via model=...
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "openai/gpt-4o-mini")

# Per-model fallback chain (rate-limit resilient). The runner tries
# DEFAULT_MODEL first, then these in order on failure.
FALLBACK_MODELS = [
    m.strip()
    for m in os.getenv(
        "FALLBACK_MODELS",
        "anthropic/claude-3.5-haiku,google/gemini-flash-1.5,deepseek/deepseek-chat",
    ).split(",")
    if m.strip()
]


class OpenRouterModelProvider(ModelProvider):
    """A ModelProvider that resolves any model name to an OpenRouter-backed
    OpenAIChatCompletionsModel. The SDK calls provider.get_model(name)."""

    def __init__(self, api_key: str, base_url: str):
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    def get_model(self, model_name: str | None) -> Model:
        name = model_name or DEFAULT_MODEL
        return OpenAIChatCompletionsModel(
            model=name, openai_client=self._client
        )


def get_provider() -> OpenRouterModelProvider:
    return OpenRouterModelProvider(OPENROUTER_API_KEY, OPENROUTER_BASE_URL)


def has_key() -> bool:
    return bool(OPENROUTER_API_KEY)
