"""
LLM Orchestrator — routes requests to the appropriate model.

Every external call retries on transient errors (503, 429, timeout).
Every tier has a fallback chain: Qwen3 → DeepSeek, Gemini → DeepSeek.
DeepSeek (final fallback) retries but returns a friendly error on exhaustion.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import httpx

from app.config import settings
from app.llm.retry import retry

logger = logging.getLogger(__name__)

OPENROUTER_BASE = "https://openrouter.ai/api/v1/chat/completions"
QWEN3_MODEL = "qwen/qwen3-235b-a22b"


class ModelTier(str, Enum):
    QWEN3 = "qwen/qwen3-235b-a22b"
    DEEPSEEK_V3 = "deepseek-chat"


@dataclass
class LLMResponse:
    content: str
    model_used: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


def detect_language(text: str) -> str:
    """Simple heuristic: if >30% CJK characters, treat as Chinese."""
    cjk_count = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    if len(text) > 0 and cjk_count / len(text) > 0.3:
        return "zh"
    return "en"


def select_model(query: str, is_analysis: bool = True, is_degraded: bool = False) -> ModelTier:
    """Route to the appropriate model based on context."""
    if is_degraded:
        return ModelTier.DEEPSEEK_V3
    return ModelTier.QWEN3


THINKING_BUDGET = 10000  # tokens reserved for Qwen3 chain-of-thought


async def _call_openrouter_raw(
    messages: list[dict],
    model: str = QWEN3_MODEL,
    max_tokens: int = 600,
) -> LLMResponse:
    """Single attempt at calling OpenRouter."""
    api_key = settings.openrouter_api_key
    if not api_key:
        return LLMResponse(content="[OpenRouter API key not configured]", model_used=model)

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens + THINKING_BUDGET,
        "temperature": 0.7,
        "thinking": {"type": "enabled", "budget_tokens": THINKING_BUDGET},
    }

    async with httpx.AsyncClient(timeout=90.0, trust_env=True) as client:
        resp = await client.post(
            OPENROUTER_BASE,
            json=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://github.com/stock-advisor",
                "X-Title": "Sid Sloth",
            },
        )
        resp.raise_for_status()
        data = resp.json()

    choice = data.get("choices", [{}])[0]
    usage = data.get("usage", {})

    return LLMResponse(
        content=choice.get("message", {}).get("content", ""),
        model_used=model,
        input_tokens=usage.get("prompt_tokens", 0),
        output_tokens=usage.get("completion_tokens", 0),
        total_tokens=usage.get("total_tokens", 0),
    )


async def call_openrouter(
    messages: list[dict],
    model: str = QWEN3_MODEL,
    max_tokens: int = 600,
) -> LLMResponse:
    """Call OpenRouter with retry."""
    return await retry(
        _call_openrouter_raw, messages, model=model, max_tokens=max_tokens,
        label="OpenRouter/Qwen3",
    )



async def _call_deepseek_raw(
    messages: list[dict],
    max_tokens: int = 600,
) -> LLMResponse:
    """Single attempt at calling DeepSeek."""
    api_key = settings.deepseek_api_key
    if not api_key:
        return LLMResponse(content="[DeepSeek API key not configured]", model_used="deepseek-chat")

    async with httpx.AsyncClient(timeout=60.0, trust_env=True) as client:
        resp = await client.post(
            "https://api.deepseek.com/chat/completions",
            json={
                "model": "deepseek-chat",
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.7,
            },
            headers={"Authorization": f"Bearer {api_key}"},
        )
        resp.raise_for_status()
        data = resp.json()

    choice = data.get("choices", [{}])[0]
    usage = data.get("usage", {})

    return LLMResponse(
        content=choice.get("message", {}).get("content", ""),
        model_used="deepseek-chat",
        input_tokens=usage.get("prompt_tokens", 0),
        output_tokens=usage.get("completion_tokens", 0),
        total_tokens=usage.get("total_tokens", 0),
    )


async def call_deepseek(
    messages: list[dict],
    max_tokens: int = 600,
) -> LLMResponse:
    """Call DeepSeek with retry. As final fallback, returns error message on exhaustion."""
    try:
        return await retry(
            _call_deepseek_raw, messages, max_tokens=max_tokens,
            label="DeepSeek",
        )
    except Exception as e:
        logger.error("DeepSeek exhausted all retries: %s", e)
        return LLMResponse(
            content="I'm sorry, all AI services are temporarily unavailable. Please try again in a few minutes.",
            model_used="none",
        )


async def chat(
    messages: list[dict],
    model_tier: ModelTier | None = None,
    query: str = "",
    is_analysis: bool = True,
    is_degraded: bool = False,
    max_tokens: int = 600,
) -> LLMResponse:
    """
    Route a chat request with retry + fallback:
      Qwen3 (retry) → DeepSeek (retry) → friendly error
      Gemini (retry) → DeepSeek (retry) → friendly error
    """
    if model_tier is None:
        model_tier = select_model(query, is_analysis=is_analysis, is_degraded=is_degraded)

    logger.info("Routing to %s (analysis=%s, degraded=%s)", model_tier.value, is_analysis, is_degraded)

    if model_tier == ModelTier.QWEN3:
        try:
            result = await call_openrouter(messages, model=QWEN3_MODEL, max_tokens=max_tokens)
            if result.content:
                return result
            logger.warning("Qwen3 returned empty response, falling back to DeepSeek")
        except Exception as e:
            logger.warning("Qwen3 failed after retries (%s: %s), falling back to DeepSeek", type(e).__name__, e)

        return await call_deepseek(messages, max_tokens=max_tokens)

    else:
        return await call_deepseek(messages, max_tokens=max_tokens)
