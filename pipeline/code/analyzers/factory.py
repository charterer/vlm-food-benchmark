from typing import Optional

from .base import FoodAnalyzer


_GEMINI_MODELS = {
    "gemini": "gemini-2.0-flash",
    "gemini-2.0": "gemini-2.0-flash",
    "gemini-2.5": "gemini-2.5-flash",
    "gemini25": "gemini-2.5-flash",
    "gemini-3.0": "gemini-3.0-flash",
    "gemini-3.1": "gemini-3.1-flash-lite",
}

_OPENAI_MODELS = {
    "openai": "gpt-4o",
    "gpt4": "gpt-4o",
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt5": "gpt-5-mini",
    "gpt-5": "gpt-5-mini",
    "gpt-5-mini": "gpt-5-mini",
}

_ANTHROPIC_MODELS = {
    "anthropic": "claude-haiku-4-5-20251001",
    "claude": "claude-haiku-4-5-20251001",
    "haiku": "claude-haiku-4-5-20251001",
    "claude-haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-20250514",
}

_QWEN_ALIASES = {"qwen", "qwen2vl", "qwen2-vl"}


def get_analyzer(
    provider: str,
    api_key: str = "",
    model_name: Optional[str] = None,
    prompt_variant: Optional[str] = None,
    **kwargs,
) -> FoodAnalyzer:
    p = provider.lower()

    if p in _GEMINI_MODELS:
        from .gemini import GeminiAnalyzer
        return GeminiAnalyzer(api_key, model_name=model_name or _GEMINI_MODELS[p], prompt_variant=prompt_variant)

    if p in _OPENAI_MODELS:
        from .openai import OpenAIAnalyzer
        return OpenAIAnalyzer(api_key, model_name=model_name or _OPENAI_MODELS[p], prompt_variant=prompt_variant)

    if p in _ANTHROPIC_MODELS:
        from .anthropic import AnthropicAnalyzer
        return AnthropicAnalyzer(api_key, model_name=model_name or _ANTHROPIC_MODELS[p], prompt_variant=prompt_variant)

    if p in _QWEN_ALIASES:
        from .qwen2vl import Qwen2VLAnalyzer
        return Qwen2VLAnalyzer(
            model_name=model_name or "Qwen/Qwen2-VL-7B-Instruct",
            device=kwargs.get("device", "auto"),
            prompt_variant=prompt_variant,
        )

    if p == "fatsecret":
        from .fatsecret import FatSecretAnalyzer
        if ":" not in api_key:
            raise ValueError("FatSecret requires client_id:client_secret")
        cid, csec = api_key.split(":", 1)
        return FatSecretAnalyzer(client_id=cid, client_secret=csec)

    raise ValueError(f"Unknown provider: {provider!r}")
