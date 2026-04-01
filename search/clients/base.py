from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class SearchResult:
    url: str
    title: str = ""
    snippet: str = ""
    published_date: str | None = None
    score: float | None = None

    def to_context_string(self, max_snippet_len: int = 1000) -> str:
        snippet = self.snippet[:max_snippet_len]
        parts = [f"Title: {self.title}", f"URL: {self.url}"]
        if self.published_date:
            parts.append(f"Date: {self.published_date}")
        parts.append(f"Content: {snippet}")
        return "\n".join(parts)

class SearchClient(Protocol):
    @property
    def name(self) -> str: ...

    def search(self, query: str, num_results: int = 10, before_date: str | None = None) -> list[SearchResult]: ...
