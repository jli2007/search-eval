from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Prediction:
    probability: float
    reasoning: str
    search_queries_used: list[str] = field(default_factory=list)
    num_results_used: int = 0


class LLM(Protocol):
    def predict(self, question: str, search_results: list | None = None, cutoff_date: str | None = None) -> Prediction: ...

    def generate_search_queries(self, question: str, num_queries: int = 3) -> list[str]: ...
