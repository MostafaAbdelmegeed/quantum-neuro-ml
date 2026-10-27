"""Draw the nested acquisition-group evaluation used by the benchmark."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.hashsalt": "qneuro-evaluation"})
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.patch.set_facecolor("#f7f9fb")
    ax.set(xlim=(0, 10), ylim=(0, 7))
    ax.set_axis_off()
    ink, train, test = "#334155", "#167a9b", "#a85112"

    def box(x, y, width, height, label, color=ink, fill="white"):
        ax.add_patch(FancyBboxPatch(
            (x - width / 2, y - height / 2), width, height,
            boxstyle="round,pad=0.04,rounding_size=0.07",
            fc=fill, ec=color, lw=1.4,
        ))
        ax.text(x, y, label, ha="center", va="center", color=ink, fontsize=12, linespacing=1.5)

    def arrow(start, end, color=ink):
        ax.add_patch(FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=14, lw=1.4, color=color,
        ))

    ax.text(0.5, 6.65, "Test on an unseen acquisition group", fontsize=19, weight="bold", color=ink)
    box(5, 5.90, 8, 0.55, "Screened ABIDE I cohort: 432 participants, 7 acquisition groups")
    box(5, 4.95, 6.7, 0.55, "Outer split: hold out one acquisition group")
    arrow((5, 5.58), (5, 5.28))
    box(2.75, 3.95, 3.8, 0.55, "Remaining groups: training data", train, "#eaf4f8")
    box(7.35, 3.95, 3.6, 0.55, "Held-out group: test data", test, "#fcf1e7")
    arrow((3.3, 4.63), (2.75, 4.28), train)
    arrow((6.7, 4.63), (7.35, 4.28), test)
    box(2.75, 2.80, 4.1, 0.85, "Three grouped inner folds\nFit transforms and select hyperparameters", train)
    arrow((2.75, 3.62), (2.75, 3.28), train)
    box(2.75, 1.65, 4.1, 0.65, "Refit transforms and model\non the outer training groups", train)
    arrow((2.75, 2.32), (2.75, 2.03), train)
    box(7.35, 1.65, 3.6, 0.65, "Evaluate once\non the held-out group", test)
    arrow((4.86, 1.65), (5.49, 1.65), train)
    arrow((7.35, 3.62), (7.35, 2.03), test)
    box(5, 0.47, 7.8, 0.60, "Repeat all seven outer folds; report site means and paired differences")
    arrow((7.35, 1.27), (7.35, 0.82), test)
    output = Path(__file__).resolve().parents[1] / "docs/figures"
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "evaluation-flow.svg", metadata={"Date": None})
    fig.savefig(output / "evaluation-flow.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
