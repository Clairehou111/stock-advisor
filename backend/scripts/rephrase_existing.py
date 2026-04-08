"""
One-time script: rephrase strategy_text, trend_status, and principle_text
for all existing records in the DB to remove distinctive authorial voice.

Usage:
    python -m scripts.rephrase_existing
"""

from __future__ import annotations

import asyncio
import logging

import httpx
from sqlalchemy import select

from app.config import settings
from app.db.session import async_session
from app.models.tables import PrincipleCorpus, StockPrediction

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_REPHRASE_PROMPT = """\
Rephrase the following stock analysis note in neutral, third-person language \
as if written by a generic quantitative analyst. \
Keep all numbers, price levels, and signals intact. \
Remove any distinctive personal voice, slang, or first-person expressions. \
Keep the output roughly the same length as the input — do NOT expand short phrases into long paragraphs. \
Return only the rephrased text, no extra commentary.

Original:
{text}"""


async def _rephrase(text: str) -> str:
    if not text or not text.strip():
        return text
    api_key = settings.deepseek_api_key
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY required for rephrasing")

    async with httpx.AsyncClient(timeout=20.0, trust_env=False) as client:
        resp = await client.post(
            "https://api.deepseek.com/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": _REPHRASE_PROMPT.format(text=text)}],
                "max_tokens": 200,
                "temperature": 0.3,
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


async def main() -> None:
    async with async_session() as db:
        # Rephrase stock predictions
        result = await db.execute(
            select(StockPrediction).where(StockPrediction.is_current == True)  # noqa: E712
        )
        stocks = result.scalars().all()
        logger.info("Rephrasing %d stock predictions...", len(stocks))

        for i, stock in enumerate(stocks):
            changed = False
            if stock.strategy_text:
                stock.strategy_text = await _rephrase(stock.strategy_text)
                changed = True
                await asyncio.sleep(0.5)
            if stock.trend_status:
                stock.trend_status = await _rephrase(stock.trend_status)
                changed = True
                await asyncio.sleep(0.5)
            if changed:
                logger.info("  [%d/%d] %s done", i + 1, len(stocks), stock.ticker)

        # Rephrase principles
        result = await db.execute(select(PrincipleCorpus))
        principles = result.scalars().all()
        logger.info("Rephrasing %d principles...", len(principles))

        for i, p in enumerate(principles):
            if p.principle_text:
                p.principle_text = await _rephrase(p.principle_text)
                await asyncio.sleep(0.5)
                logger.info("  [%d/%d] done", i + 1, len(principles))

        await db.commit()
        logger.info("All records rephrased and committed.")


if __name__ == "__main__":
    asyncio.run(main())
