from __future__ import annotations

import os
import logging
from datetime import datetime, timedelta

import requests

from search.clients.base import SearchResult

logger = logging.getLogger(__name__)


class TavilyClient:
    API_URL = "https://api.tavily.com/search"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("TAVILY_API_KEY", "")
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY is required")

    @property
    def name(self) -> str:
        return "tavily"

    def search(self, query: str, num_results: int = 10, before_date: str | None = None) -> list[SearchResult]:
        try:
            resp = requests.post(
                self.API_URL,
                json={
                    "api_key": self.api_key,
                    "query": query,
                    "max_results": num_results,
                    "include_raw_content": False,
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning(f"Tavily search failed: {e}")
            return []

        # tavily lacks native date filtering, so we filter post-hoc
        cutoff = None
        if before_date:
            cutoff = datetime.strptime(before_date, "%Y-%m-%d") - timedelta(days=14)

        results = []
        for r in data.get("results", []):
            if cutoff:
                pub_str = r.get("published_date")
                if not pub_str:
                    continue
                try:
                    pub = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
                    if pub.replace(tzinfo=None) > cutoff:
                        continue
                except (ValueError, TypeError):
                    continue

            results.append(
                SearchResult(
                    url=r.get("url", ""),
                    title=r.get("title", ""),
                    snippet=r.get("content", ""),
                    score=r.get("score"),
                )
            )
        return results
