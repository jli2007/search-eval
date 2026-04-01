from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # non-interactive backend

from search.eval.scorer import ProviderResult


def plot_all(results: list[ProviderResult], output_dir: str = "search/results"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plot_brier_comparison(results, output_dir)
    plot_calibration(results, output_dir)
    plot_category_heatmap(results, output_dir)


# Bar chart comparing mean Brier score across providers/modes.
def plot_brier_comparison(results: list[ProviderResult], output_dir: str):
    labels = [f"{r.provider}\n{r.mode}" for r in results]
    briers = [r.mean_brier for r in results]
    colors = ["#4CAF50" if b < 0.25 else "#FF9800" if b < 0.4 else "#F44336" for b in briers]

    fig, ax = plt.subplots(figsize=(max(6, len(results) * 1.5), 5))
    bars = ax.bar(labels, briers, color=colors, edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, briers):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Mean Brier Score (lower = better)")
    ax.set_title("Provider Comparison — Brier Score")
    ax.set_ylim(0, max(briers) * 1.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    path = Path(output_dir) / "brier_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# Calibration curve: predicted probability vs actual resolution rate.
# Perfect calibration = diagonal line.
def plot_calibration(results: list[ProviderResult], output_dir: str):
    fig, ax = plt.subplots(figsize=(6, 6))

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect calibration")

    colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0"]
    for i, r in enumerate(results):
        cal = r.calibration_data(num_bins=5)
        if not cal:
            continue
        predicted = [b["avg_predicted"] for b in cal]
        actual = [b["avg_actual"] for b in cal]
        color = colors[i % len(colors)]
        label = f"{r.provider}/{r.mode} (ECE={r.expected_calibration_error():.3f})"
        ax.plot(predicted, actual, "o-", color=color, label=label, markersize=8, linewidth=2)

    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Actual Resolution Rate")
    ax.set_title("Calibration Curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_aspect("equal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    path = Path(output_dir) / "calibration_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# Grouped bar chart showing Brier score per category per provider.
def plot_category_heatmap(results: list[ProviderResult], output_dir: str):
    all_cats = sorted(set(cat for r in results for cat in r.brier_by_category()))
    if not all_cats:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(all_cats) * 1.5), 5))
    bar_width = 0.8 / len(results)
    colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0"]

    for i, r in enumerate(results):
        cat_scores = r.brier_by_category()
        vals = [cat_scores.get(cat, 0) for cat in all_cats]
        positions = [j + i * bar_width for j in range(len(all_cats))]
        ax.bar(positions, vals, bar_width, label=f"{r.provider}/{r.mode}",
               color=colors[i % len(colors)], alpha=0.85)

    ax.set_xticks([j + bar_width * (len(results) - 1) / 2 for j in range(len(all_cats))])
    ax.set_xticklabels(all_cats, rotation=30, ha="right")
    ax.set_ylabel("Mean Brier Score")
    ax.set_title("Brier Score by Category")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    path = Path(output_dir) / "category_breakdown.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")
