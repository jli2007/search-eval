from __future__ import annotations
from search.clients.base import SearchResult


# Baseline client that returns no search results.
class NoneClient:
    @property
    def name(self) -> str:
        return "none"

    def search(self, query: str, num_results: int = 10, before_date: str | None = None) -> list[SearchResult]:
        return []
