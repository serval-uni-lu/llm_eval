import json
import yaml
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from .utils import normalize_text

# Style constants
DARK = "#2c3e50"
WHITE = "#ffffff"
PALETTE = [
    "#4c869d",
    "#a3745b",
    "#7a9b76",
    "#d0c48a",
    "#9a8caf",
    "#6a99ab",
    "#c97c5d",
    "#8fbc8f",
]


def save_results_plot(output_dir: str | Path, save_path: str | Path) -> Path:
    """Load evaluation results and save grouped bar chart."""
    output_dir = Path(output_dir)
    save_path = Path(save_path)

    with open(output_dir / "config.yaml") as f:
        config = yaml.safe_load(f)

    # Build model name mapping (folder name -> display name)
    models = {
        normalize_text(m["name"]): m["name"].split("/")[-1] for m in config["models"]
    }

    # Build categories as (dataset, metric) pairs
    categories = []  # [(dataset_name, metric_name), ...]
    for dataset_cfg in config["datasets"]:
        for metric in dataset_cfg["metrics"]:
            categories.append((dataset_cfg["name"], metric))

    # Collect scores per model per category
    data = {name: [] for name in models.values()}
    for dataset_name, metric in categories:
        dataset_folder = normalize_text(dataset_name)
        for model_folder, display_name in models.items():
            metric_file = (
                output_dir
                / dataset_folder
                / model_folder
                / "metrics"
                / f"{metric}.jsonl"
            )
            scores = []
            if metric_file.exists():
                with open(metric_file) as f:
                    for line in f:
                        row = json.loads(line)
                        if row.get("score") is not None:
                            scores.append(row["score"])
            avg = (sum(scores) / len(scores) * 100) if scores else 0
            data[display_name].append(avg)

    # Create and save plot
    fig = _grouped_bar_chart(data, categories, title=config.get("name", ""))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    return save_path


def _grouped_bar_chart(
    data: dict[str, list[float]],
    categories: list[tuple[str, str]],
    title: str = "",
) -> plt.Figure:
    """Create grouped bar chart with dataset/metric labels."""
    plt.rcParams.update(
        {
            "axes.edgecolor": DARK,
            "axes.labelcolor": DARK,
            "xtick.color": DARK,
            "ytick.color": DARK,
            "text.color": DARK,
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
        }
    )

    series = list(data.keys())
    n_series, n_cats = len(series), len(categories)

    if n_series == 0 or n_cats == 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return fig

    bar_width = max(0.15, 0.6 / n_series)
    fig, ax = plt.subplots(figsize=(max(14, n_cats * (n_series * 0.5 + 1.5)), 7))
    x = np.arange(n_cats)
    colors = [PALETTE[i % len(PALETTE)] for i in range(n_series)]

    # Max per category for highlighting
    max_per_cat = [round(max(data[s][i] for s in series), 1) for i in range(n_cats)]
    font_size = max(8, 12 - n_series)

    for idx, name in enumerate(series):
        vals = data[name]
        offset = (idx - n_series / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset,
            vals,
            bar_width * 0.9,
            label=name,
            color=colors[idx],
            alpha=0.85,
            edgecolor=WHITE,
            linewidth=2.5,
        )

        for j, (bar, val) in enumerate(zip(bars, vals)):
            is_max = val > 0 and round(val, 1) == max_per_cat[j]
            y_pos = max(bar.get_height(), 1) + 1  # Show 0.0 just above baseline
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y_pos,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=font_size,
                fontweight="bold" if is_max else "normal",
                color=DARK,
            )

    ax.set_ylabel("Score (%)", fontsize=15, color=DARK)
    ax.set_xticks(x)
    ax.set_xticklabels([""] * n_cats)  # Clear default labels

    # Custom two-line labels: dataset (bold) + metric (smaller)
    for i, (dataset, metric) in enumerate(categories):
        ax.text(
            i,
            -3,
            dataset,
            ha="center",
            va="top",
            fontsize=12,
            fontweight="bold",
            color=DARK,
        )
        ax.text(
            i,
            -7,
            metric.replace("_", " "),
            ha="center",
            va="top",
            fontsize=9,
            color=DARK,
        )

    ax.set_ylim(0, min(105, max(v for vals in data.values() for v in vals) + 10))
    ax.yaxis.grid(True, linestyle="-", alpha=0.1, color=DARK)
    ax.set_axisbelow(True)

    if title:
        fig.suptitle(title, fontsize=18, y=0.97, color=DARK, fontweight="bold")

    ncol = int(np.ceil(n_series / 2)) if n_series > 3 else n_series
    fig.legend(
        *ax.get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=ncol,
        frameon=True,
        fontsize=11,
    )
    plt.subplots_adjust(top=0.78, bottom=0.12)
    return fig
