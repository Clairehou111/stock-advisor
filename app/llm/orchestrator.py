"""
LLM Orchestrator — routes requests to the appropriate model.

Chat (all queries):
  Primary:  Qwen3 235B A22B via OpenRouter
  Fallback: DeepSeek V3

Extraction tasks (metadata, chart analysis):
  Gemini 2.5 Flash (unchanged)

Rephrasing / signal extraction:
  DeepSeek V3 (unchanged)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

OPENROUTER_BASE = "https://openrouter.ai/api/v1/chat/completions"
QWEN3_MODEL = "qwen/qwen3-235b-a22b"


class ModelTier(str, Enum):
    QWEN3 = "qwen/qwen3-235b-a22b"
    GEMINI_FLASH = "gemini-2.5-flash"
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


async def call_openrouter(
    messages: list[dict],
    model: str = QWEN3_MODEL,
    max_tokens: int = 600,
) -> LLMResponse:
    """Call OpenRouter API (OpenAI-compatible). Works for Claude and Qwen3.

    For Qwen3: thinking tokens are separate from output tokens.
    max_tokens covers thinking + output, so we add THINKING_BUDGET on top
    of the requested output budget.
    """
    api_key = settings.openrouter_api_key
    if not api_key:
        return LLMResponse(content="[OpenRouter API key not configured]", model_used=model)

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens + THINKING_BUDGET,
        "temperature": 0.7,
        # Enable Qwen3 extended reasoning with a capped budget so the model
        # reasons deeply without burning unbounded tokens.
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


async def call_gemini_flash(
    messages: list[dict],
    max_tokens: int = 600,
) -> LLMResponse:
    """Call Gemini 2.5 Flash — used for extraction tasks only."""
    api_key = settings.gemini_api_key
    model = "gemini-2.5-flash"
    if not api_key:
        return LLMResponse(content="[Gemini API key not configured]", model_used=model)

    system_instruction = None
    contents = []
    for msg in messages:
        if msg["role"] == "system":
            system_instruction = msg["content"]
        else:
            contents.append({
                "role": "user" if msg["role"] == "user" else "model",
                "parts": [{"text": msg["content"]}],
            })

    payload = {
        "contents": contents,
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": 0.1,
        },
    }
    if system_instruction:
        payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()

    candidates = data.get("candidates", [])
    if not candidates:
        return LLMResponse(content="", model_used=model)

    parts = candidates[0].get("content", {}).get("parts", [])
    text = next((p.get("text", "") for p in parts if not p.get("thought")), "")
    usage = data.get("usageMetadata", {})

    return LLMResponse(
        content=text,
        model_used=model,
        input_tokens=usage.get("promptTokenCount", 0),
        output_tokens=usage.get("candidatesTokenCount", 0),
        total_tokens=usage.get("totalTokenCount", 0),
    )


async def call_deepseek(
    messages: list[dict],
    max_tokens: int = 600,
) -> LLMResponse:
    """Call DeepSeek API — economy tier and rephrase/extraction tasks."""
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


async def chat(
    messages: list[dict],
    model_tier: ModelTier | None = None,
    query: str = "",
    is_analysis: bool = True,
    is_degraded: bool = False,
    max_tokens: int = 600,
) -> LLMResponse:
    """
    Route a chat request:
      Primary:  Qwen3 235B A22B (OpenRouter, with extended thinking)
      Fallback: DeepSeek V3 (economy mode or on Qwen3 failure)
    """
    if model_tier is None:
        model_tier = select_model(query, is_analysis=is_analysis, is_degraded=is_degraded)

    logger.info("Routing to %s (analysis=%s, degraded=%s)", model_tier.value, is_analysis, is_degraded)

    if model_tier == ModelTier.QWEN3:
        # Primary: Qwen3 235B A22B
        try:
            result = await call_openrouter(messages, model=QWEN3_MODEL, max_tokens=max_tokens)
            if result.content:
                return result
            logger.warning("Qwen3 returned empty response, falling back to DeepSeek")
        except Exception as e:
            logger.warning("Qwen3 failed (%s: %s), falling back to DeepSeek", type(e).__name__, e)

        return await call_deepseek(messages, max_tokens=max_tokens)

    elif model_tier == ModelTier.GEMINI_FLASH:
        try:
            result = await call_gemini_flash(messages, max_tokens=max_tokens)
            if result.content:
                return result
        except Exception as e:
            logger.warning("Gemini Flash failed (%s: %s), falling back to DeepSeek", type(e).__name__, e)
        return await call_deepseek(messages, max_tokens=max_tokens)

    else:
        return await call_deepseek(messages, max_tokens=max_tokens)
