from __future__ import annotations

import json
import re
import logging

from openai import OpenAI

from search.clients.base import SearchResult
from search.llm.base import Prediction
from search.eval.prompts import (
    FORECASTER_SYSTEM,
    PREDICT_WITH_SEARCH,
    PREDICT_NO_SEARCH,
    GENERATE_QUERIES_SYSTEM,
    GENERATE_QUERIES_USER,
)

logger = logging.getLogger(__name__)


class OpenAILLM:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", temperature: float = 0.2):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature

    def predict(self, question: str, search_results: list[SearchResult] | None = None, cutoff_date: str | None = None, stream: bool = False) -> Prediction:
        cutoff = cutoff_date or "2024-01-01"
        system = FORECASTER_SYSTEM.format(cutoff_date=cutoff)

        if search_results:
            context = "\n\n---\n\n".join(r.to_context_string() for r in search_results)
            user = PREDICT_WITH_SEARCH.format(
                question=question, search_context=context, cutoff_date=cutoff,
            )
        else:
            user = PREDICT_NO_SEARCH.format(question=question, cutoff_date=cutoff)

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        if stream:
            raw = self._stream_response(messages)
        else:
            resp = self.client.chat.completions.create(
                model=self.model, temperature=self.temperature, messages=messages,
            )
            raw = resp.choices[0].message.content or ""

        parsed = self._parse_json(raw)

        if isinstance(parsed, dict):
            prob = float(parsed.get("probability", 0.5))
            reasoning = parsed.get("reasoning", raw)
        else:
            prob = 0.5
            reasoning = raw
        prob = max(0.01, min(0.99, prob))

        return Prediction(
            probability=prob,
            reasoning=reasoning,
            num_results_used=len(search_results) if search_results else 0,
        )

    def _stream_response(self, messages: list[dict]) -> str:
        import sys
        stream = self.client.chat.completions.create(
            model=self.model, temperature=self.temperature, messages=messages, stream=True,
        )
        chunks = []
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            chunks.append(delta)
            sys.stdout.write(delta)
            sys.stdout.flush()
        print()  # newline at end
        return "".join(chunks)

    def generate_search_queries(self, question: str, num_queries: int = 3) -> list[str]:
        system = GENERATE_QUERIES_SYSTEM.format(num_queries=num_queries)
        user = GENERATE_QUERIES_USER.format(question=question, num_queries=num_queries)

        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0.4,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )

        raw = resp.choices[0].message.content or ""
        parsed = self._parse_json(raw)

        if isinstance(parsed, list):
            return [str(q) for q in parsed[:num_queries]]
        # fallback: use the question itself
        return [question]

    def _parse_json(self, text: str) -> dict | list:
        # strip markdown code fences if present
        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```\s*$", "", text)
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON: {text[:200]}")
            return {}
