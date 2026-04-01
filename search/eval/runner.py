from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from search.clients.base import SearchClient
from search.llm.base import LLM
from search.eval.scorer import QuestionResult, ProviderResult

logger = logging.getLogger(__name__)


@dataclass
class Question:
    id: str
    question: str
    resolution: bool
    resolution_date: str
    category: str
    source: str


def load_questions(path: str) -> list[Question]:
    questions = []
    with open(path) as f:
        # Support both .json (array) and .jsonl (one object per line)
        if path.endswith(".jsonl"):
            for line in f:
                line = line.strip()
                if line:
                    questions.append(Question(**json.loads(line)))
        else:
            for q in json.load(f):
                questions.append(Question(**q))
    return questions


# Run one question: search (depending on mode) → LLM prediction → scored result.
def run_single_question(
    question: Question,
    client: SearchClient,
    llm: LLM,
    mode: str,
    num_queries: int = 3,
    results_per_query: int = 5,
) -> QuestionResult:
    search_results = []
    queries_used = []

    if mode == "agentic":
        # LLM generates its own search strategy
        queries = llm.generate_search_queries(question.question, num_queries)
        queries_used = queries
        for q in queries:
            hits = client.search(q, num_results=results_per_query, before_date=question.resolution_date)
            search_results.extend(hits)

    elif mode == "single":
        # Use question text as the query directly
        queries_used = [question.question]
        search_results = client.search(
            question.question, num_results=results_per_query, before_date=question.resolution_date,
        )

    # mode == "no_search" → empty results, LLM uses only prior knowledge

    prediction = llm.predict(
        question=question.question,
        search_results=search_results if search_results else None,
        cutoff_date=question.resolution_date,
    )

    return QuestionResult(
        question_id=question.id,
        question_text=question.question,
        category=question.category,
        resolution=question.resolution,
        provider=client.name,
        mode=mode,
        predicted_probability=prediction.probability,
        reasoning=prediction.reasoning,
        search_queries_used=queries_used,
        num_results_used=prediction.num_results_used,
    )


# ANSI colors for live progress
_BOLD = "\033[1m"
_DIM = "\033[2m"
_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_CYAN = "\033[36m"
_RESET = "\033[0m"

# Run all questions for one provider/mode combination.
def run_eval(
    questions: list[Question],
    client: SearchClient,
    llm: LLM,
    mode: str,
    num_queries: int = 3,
    results_per_query: int = 5,
) -> ProviderResult:
    results = []
    for i, q in enumerate(questions, 1):
        header = f"{_DIM}[{client.name}/{mode}]{_RESET}"
        progress = f"{_CYAN}({i}/{len(questions)}){_RESET}"
        print(f"  {header} {progress} {q.question[:55]}...", flush=True)

        result = run_single_question(q, client, llm, mode, num_queries, results_per_query)

        # Color the probability based on how close to ground truth
        brier = result.brier_score
        b_color = _GREEN if brier < 0.15 else _YELLOW if brier < 0.35 else _RED
        actual = f"{_GREEN}YES{_RESET}" if result.resolution else f"{_RED}NO{_RESET}"
        print(f"    → P={_BOLD}{result.predicted_probability:.2f}{_RESET}  actual={actual}  brier={b_color}{brier:.4f}{_RESET}")

        results.append(result)

    pr = ProviderResult(provider=client.name, mode=mode, question_results=results)
    return pr
