from __future__ import annotations
import math
import random
from dataclasses import dataclass, field


@dataclass
class QuestionResult:
    question_id: str
    question_text: str
    category: str
    resolution: bool
    provider: str
    mode: str
    predicted_probability: float
    reasoning: str
    search_queries_used: list[str] = field(default_factory=list)
    num_results_used: int = 0

    @property
    def brier_score(self) -> float:
        outcome = 1.0 if self.resolution else 0.0
        return (self.predicted_probability - outcome) ** 2

    # penalizes confident wrong predictions harder than brier
    @property
    def log_loss(self) -> float:
        outcome = 1.0 if self.resolution else 0.0
        p = max(0.01, min(0.99, self.predicted_probability))
        return -(outcome * math.log(p) + (1 - outcome) * math.log(1 - p))


@dataclass
class ProviderResult:
    provider: str
    mode: str
    question_results: list[QuestionResult]

    @property
    def mean_brier(self) -> float:
        if not self.question_results:
            return 0.0
        return sum(r.brier_score for r in self.question_results) / len(self.question_results)

    @property
    def mean_log_loss(self) -> float:
        if not self.question_results:
            return 0.0
        return sum(r.log_loss for r in self.question_results) / len(self.question_results)

    @property
    def n(self) -> int:
        return len(self.question_results)

    def brier_by_category(self) -> dict[str, float]:
        buckets: dict[str, list[float]] = {}
        for r in self.question_results:
            buckets.setdefault(r.category, []).append(r.brier_score)
        return {cat: sum(s) / len(s) for cat, s in buckets.items()}

    # bin predictions, compare predicted vs actual resolution rate
    def calibration_data(self, num_bins: int = 5) -> list[dict]:
        bins: list[list[QuestionResult]] = [[] for _ in range(num_bins)]
        for r in self.question_results:
            idx = min(int(r.predicted_probability * num_bins), num_bins - 1)
            bins[idx].append(r)

        data = []
        for i, bucket in enumerate(bins):
            if not bucket:
                continue
            lo = i / num_bins
            hi = (i + 1) / num_bins
            avg_pred = sum(r.predicted_probability for r in bucket) / len(bucket)
            avg_actual = sum(1.0 if r.resolution else 0.0 for r in bucket) / len(bucket)
            data.append({
                "bin": f"{lo:.1f}-{hi:.1f}",
                "avg_predicted": round(avg_pred, 3),
                "avg_actual": round(avg_actual, 3),
                "count": len(bucket),
            })
        return data

    def expected_calibration_error(self, num_bins: int = 5) -> float:
        cal = self.calibration_data(num_bins)
        total = sum(b["count"] for b in cal)
        if total == 0:
            return 0.0
        return sum(
            b["count"] / total * abs(b["avg_predicted"] - b["avg_actual"])
            for b in cal
        )


def compute_uplift(provider_brier: float, baseline_brier: float) -> float:
    if baseline_brier == 0:
        return 0.0
    return (baseline_brier - provider_brier) / baseline_brier * 100


# splits questions by baseline confidence, compares uplift in each bucket
def confidence_weighted_analysis(
    provider: ProviderResult,
    baseline: ProviderResult,
) -> dict:
    baseline_map = {r.question_id: r for r in baseline.question_results}

    uncertain, confident = [], []
    for r in provider.question_results:
        base = baseline_map.get(r.question_id)
        if not base:
            continue
        if 0.3 <= base.predicted_probability <= 0.7:
            uncertain.append((r.brier_score, base.brier_score))
        else:
            confident.append((r.brier_score, base.brier_score))

    def summarize(pairs: list[tuple[float, float]]) -> dict:
        if not pairs:
            return {"provider_brier": 0, "baseline_brier": 0, "uplift_pct": 0, "n": 0}
        p_avg = sum(p for p, _ in pairs) / len(pairs)
        b_avg = sum(b for _, b in pairs) / len(pairs)
        return {
            "provider_brier": round(p_avg, 4),
            "baseline_brier": round(b_avg, 4),
            "uplift_pct": round(compute_uplift(p_avg, b_avg), 1),
            "n": len(pairs),
        }

    return {
        "uncertain": summarize(uncertain),
        "confident": summarize(confident),
    }


# resample with replacement to estimate distribution of the mean
def bootstrap_ci(scores: list[float], n_bootstrap: int = 10000, ci: float = 0.95) -> tuple[float, float]:
    if len(scores) < 2:
        mean = scores[0] if scores else 0.0
        return (mean, mean)
    rng = random.Random(42)
    means = sorted(
        sum(rng.choices(scores, k=len(scores))) / len(scores)
        for _ in range(n_bootstrap)
    )
    lo_idx = int((1 - ci) / 2 * n_bootstrap)
    hi_idx = int((1 + ci) / 2 * n_bootstrap) - 1
    return (means[lo_idx], means[hi_idx])


# shuffle labels between provider/baseline, check if observed diff is real
def paired_permutation_test(
    provider: ProviderResult,
    baseline: ProviderResult,
    n_permutations: int = 10000,
) -> float:
    baseline_map = {r.question_id: r for r in baseline.question_results}
    pairs = []
    for r in provider.question_results:
        base = baseline_map.get(r.question_id)
        if base:
            pairs.append((r.brier_score, base.brier_score))
    if not pairs:
        return 1.0

    observed = sum(b - p for p, b in pairs) / len(pairs)
    rng = random.Random(42)
    count = 0
    for _ in range(n_permutations):
        perm_diff = 0.0
        for p_score, b_score in pairs:
            if rng.random() < 0.5:
                perm_diff += b_score - p_score
            else:
                perm_diff += p_score - b_score
        if perm_diff / len(pairs) >= observed:
            count += 1
    return count / n_permutations


def cost_summary(result: ProviderResult) -> dict:
    total_queries = sum(len(r.search_queries_used) for r in result.question_results)
    total_results = sum(r.num_results_used for r in result.question_results)
    n = result.n or 1
    return {
        "total_queries": total_queries,
        "total_results": total_results,
        "avg_queries_per_question": round(total_queries / n, 1),
        "avg_results_per_question": round(total_results / n, 1),
        "n": n,
    }
