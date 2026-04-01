from __future__ import annotations
import os
import logging
from datetime import datetime, timedelta
from exa_py import Exa
from search.clients.base import SearchResult

logger = logging.getLogger(__name__)

class ExaClient:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("EXA_API_KEY", "")
        if not self.api_key:
            raise ValueError("EXA_API_KEY is required")
        self.client = Exa(api_key=self.api_key)

    @property
    def name(self) -> str:
        return "exa"

    def search(self, query: str, num_results: int = 10, before_date: str | None = None) -> list[SearchResult]:
        try:
            kwargs = dict(
                type="auto",
                num_results=num_results,
                highlights={"max_characters": 4000},
            )
            # Offset by 7 days to avoid articles that leak the resolution
            if before_date:
                dt = datetime.strptime(before_date, "%Y-%m-%d") - timedelta(days=7)
                kwargs["end_published_date"] = dt.strftime("%Y-%m-%dT00:00:00.000Z")
            response = self.client.search_and_contents(query, **kwargs)
        except Exception as e:
            logger.warning(f"Exa search failed: {e}")
            return []

        results = []
        for r in response.results:
            # highlights is a list of strings, join them
            snippet = "\n".join(r.highlights) if r.highlights else ""
            results.append(
                SearchResult(
                    url=r.url,
                    title=r.title or "",
                    snippet=snippet,
                    published_date=r.published_date,
                    score=r.score,
                )
            )
        return results
