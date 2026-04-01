from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from search.config import config
from search.clients.exa_client import ExaClient
from search.clients.none_client import NoneClient
from search.clients.tavily_client import TavilyClient
from search.llm.openai_llm import OpenAILLM
from search.eval.runner import load_questions, run_eval
from search.eval.scorer import (
    ProviderResult, compute_uplift, confidence_weighted_analysis,
    bootstrap_ci, paired_permutation_test, cost_summary,
)
from search.eval.visualize import plot_all


class C:
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[32m"
    RED = "\033[31m"
    YELLOW = "\033[33m"
    CYAN = "\033[36m"
    MAGENTA = "\033[35m"
    RESET = "\033[0m"

def colored(text: str, color: str) -> str:
    return f"{color}{text}{C.RESET}"


def make_client(provider: str):
    if provider == "exa":
        return ExaClient(api_key=config.exa_api_key)
    elif provider == "tavily":
        return TavilyClient(api_key=config.tavily_api_key)
    elif provider == "none":
        return NoneClient()
    else:
        raise ValueError(f"Unknown provider: {provider}")


def print_results(all_results: list[ProviderResult]):
    baseline = next((r for r in all_results if r.provider == "none"), None)
    baseline_brier = baseline.mean_brier if baseline else None

    print(f"\n{colored('=' * 60, C.BOLD)}")
    print(colored("  SEARCH BENCHMARK RESULTS", C.BOLD))
    print(colored('=' * 60, C.BOLD))

    print(f"\n{colored('FINDINGS', C.BOLD + C.CYAN)}")
    header = f"  {'Provider':<10}| {'Mode':<10}| {'Brier ↓':>20} | {'Log Loss':>9} | {'Uplift':>20}"
    print(colored(header, C.BOLD))
    print(colored(f"  {'─' * 10}|{'─' * 11}|{'─' * 22}|{'─' * 11}|{'─' * 21}", C.DIM))

    for r in all_results:
        scores = [q.brier_score for q in r.question_results]
        ci_lo, ci_hi = bootstrap_ci(scores)

        uplift_str = "—"
        uplift_color = C.DIM
        if baseline_brier and r.provider != "none":
            uplift = compute_uplift(r.mean_brier, baseline_brier)
            uplift_color = C.GREEN if uplift > 0 else C.RED
            uplift_str = f"{uplift:+.1f}%"

        brier_color = C.GREEN if r.mean_brier < 0.25 else C.YELLOW if r.mean_brier < 0.4 else C.RED
        brier_text = colored(f"{r.mean_brier:.3f} [{ci_lo:.3f}–{ci_hi:.3f}]", brier_color)
        loss_text = f"{r.mean_log_loss:.3f}"
        uplift_text = colored(uplift_str, uplift_color)
        print(f"  {r.provider:<10}| {r.mode:<10}| {brier_text:>32} | {loss_text:>9} | {uplift_text:>32}")

    print(f"\n{colored('PER-QUESTION DETAIL', C.BOLD + C.CYAN)}")
    if baseline and len(all_results) > 1:
        search_result = next((r for r in all_results if r.provider != "none"), None)
        if search_result:
            baseline_map = {r.question_id: r for r in baseline.question_results}
            for sr in search_result.question_results:
                br = baseline_map.get(sr.question_id)
                if not br:
                    continue
                actual = colored("YES", C.GREEN) if sr.resolution else colored("NO", C.RED)
                if sr.brier_score < br.brier_score:
                    delta = colored(f"▲ {br.brier_score - sr.brier_score:.3f}", C.GREEN)
                else:
                    delta = colored(f"▼ {sr.brier_score - br.brier_score:.3f}", C.RED)

                print(f"\n  {colored(sr.question_id, C.DIM)} {sr.question_text[:55]}")
                print(f"    Actual={actual}  Baseline={br.predicted_probability:.2f}  {colored(search_result.provider, C.BOLD)}={sr.predicted_probability:.2f}  {delta}")

    print(f"\n{colored('BY CATEGORY', C.BOLD + C.CYAN)}")
    for r in all_results:
        print(f"  {colored(f'[{r.provider}/{r.mode}]', C.BOLD)}")
        for cat, score in sorted(r.brier_by_category().items()):
            bar_len = int(score * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            color = C.GREEN if score < 0.2 else C.YELLOW if score < 0.4 else C.RED
            print(f"    {cat:<12} {colored(f'{score:.4f}', color)} {colored(bar, C.DIM)}")

    print(f"\n{colored('CALIBRATION', C.BOLD + C.CYAN)}")
    for r in all_results:
        ece = r.expected_calibration_error()
        ece_color = C.GREEN if ece < 0.1 else C.YELLOW if ece < 0.2 else C.RED
        print(f"  {colored(f'[{r.provider}/{r.mode}]', C.BOLD)} ECE={colored(f'{ece:.4f}', ece_color)}")
        for b in r.calibration_data():
            print(f"    {b['bin']:>7}  predicted={b['avg_predicted']:.3f}  actual={b['avg_actual']:.3f}  n={b['count']}")

    if baseline:
        print(f"\n{colored('STATISTICAL SIGNIFICANCE', C.BOLD + C.CYAN)}")
        print(colored("  (Paired permutation test, n=10000)", C.DIM))
        for r in all_results:
            if r.provider == "none":
                continue
            p_val = paired_permutation_test(r, baseline)
            if p_val < 0.01:
                sig_label = colored("**", C.GREEN)
                sig_text = colored("significant", C.GREEN)
            elif p_val < 0.05:
                sig_label = colored("*", C.GREEN)
                sig_text = colored("significant", C.GREEN)
            else:
                sig_label = " "
                sig_text = colored("not significant", C.DIM)
            p_text = colored(f"p={p_val:.3f}", C.BOLD)
            print(f"  {r.provider}/{r.mode:<10} vs baseline: {p_text} {sig_label}  ({sig_text})")

    if baseline:
        print(f"\n{colored('CONFIDENCE-WEIGHTED UPLIFT', C.BOLD + C.CYAN)}")
        print(colored("  (Does search help more when the LLM is uncertain?)", C.DIM))
        for r in all_results:
            if r.provider == "none":
                continue
            cwa = confidence_weighted_analysis(r, baseline)
            for bucket_name, bucket in cwa.items():
                if bucket["n"] == 0:
                    continue
                uplift_val = bucket["uplift_pct"]
                color = C.GREEN if uplift_val > 0 else C.RED
                label = colored(bucket_name, C.MAGENTA)
                uplift_text = colored(f"{uplift_val:+.1f}%", color)
                print(f"  [{r.provider}] {label:<22} uplift={uplift_text}  (n={bucket['n']})")

    print(f"\n{colored('COST ANALYSIS', C.BOLD + C.CYAN)}")
    print(colored("  (Search API usage per provider/mode)", C.DIM))
    cost_header = f"  {'Provider':<10}| {'Mode':<10}| {'Avg Queries':>12} | {'Avg Results':>12} | {'Total Queries':>14}"
    print(colored(cost_header, C.BOLD))
    print(colored(f"  {'─' * 10}|{'─' * 11}|{'─' * 14}|{'─' * 14}|{'─' * 15}", C.DIM))
    for r in all_results:
        cs = cost_summary(r)
        print(f"  {r.provider:<10}| {r.mode:<10}| {cs['avg_queries_per_question']:>12} | {cs['avg_results_per_question']:>12} | {cs['total_queries']:>14}")


def save_results(all_results: list[ProviderResult], output_dir: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    baseline = next((r for r in all_results if r.provider == "none"), None)

    questions_out = []
    for pr in all_results:
        for qr in pr.question_results:
            questions_out.append({
                "question_id": qr.question_id,
                "question": qr.question_text,
                "category": qr.category,
                "resolution": qr.resolution,
                "provider": qr.provider,
                "mode": qr.mode,
                "predicted_probability": qr.predicted_probability,
                "brier_score": qr.brier_score,
                "log_loss": qr.log_loss,
                "reasoning": qr.reasoning,
                "search_queries_used": qr.search_queries_used,
                "num_results_used": qr.num_results_used,
            })

    summary = []
    for r in all_results:
        scores = [q.brier_score for q in r.question_results]
        ci_lo, ci_hi = bootstrap_ci(scores)
        entry = {
            "provider": r.provider,
            "mode": r.mode,
            "n": r.n,
            "mean_brier": round(r.mean_brier, 4),
            "brier_ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
            "mean_log_loss": round(r.mean_log_loss, 4),
            "ece": round(r.expected_calibration_error(), 4),
            "brier_by_category": {k: round(v, 4) for k, v in r.brier_by_category().items()},
            "cost": cost_summary(r),
        }
        if baseline and r.provider != "none":
            entry["uplift_vs_baseline_pct"] = round(compute_uplift(r.mean_brier, baseline.mean_brier), 1)
            entry["p_value"] = round(paired_permutation_test(r, baseline), 4)
            entry["confidence_weighted"] = confidence_weighted_analysis(r, baseline)
        summary.append(entry)

    out = {"summary": summary, "questions": questions_out}
    path = Path(output_dir) / f"results_{timestamp}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n{colored('Results saved to', C.DIM)} {path}")


def main():
    parser = argparse.ArgumentParser(description="Search API Eval — Prediction Market Benchmark")
    parser.add_argument("--providers", nargs="+", default=["none", "exa", "tavily"])
    parser.add_argument("--modes", nargs="+", default=["no_search", "single", "agentic"])
    parser.add_argument("--questions", default="search/datasets/questions.jsonl")
    parser.add_argument("--output", default="search/results")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--num-queries", type=int, default=3, help="Queries per question in agentic mode")
    parser.add_argument("--results-per-query", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None, help="Only run first N questions")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )

    questions = load_questions(args.questions)
    if args.limit:
        questions = questions[:args.limit]
    print(colored(f"Loaded {len(questions)} questions", C.DIM))

    llm = OpenAILLM(api_key=config.openai_api_key, model=args.model)

    all_results: list[ProviderResult] = []

    for provider_name in args.providers:
        client = make_client(provider_name)
        modes = ["no_search"] if provider_name == "none" else [m for m in args.modes if m != "no_search"]

        for mode in modes:
            print(colored(f"\nRunning {provider_name}/{mode}...", C.BOLD + C.CYAN))
            result = run_eval(questions, client, llm, mode, args.num_queries, args.results_per_query)
            all_results.append(result)
            brier_color = C.GREEN if result.mean_brier < 0.25 else C.YELLOW
            print(f"  Mean Brier: {colored(f'{result.mean_brier:.4f}', brier_color)}")

    print_results(all_results)
    save_results(all_results, args.output)

    print(colored("\nGenerating charts...", C.DIM))
    plot_all(all_results, args.output)


if __name__ == "__main__":
    main()
