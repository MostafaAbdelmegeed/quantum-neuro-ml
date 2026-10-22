"""Export aggregate results and research figures without participant records."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SERIES = {
    "reference": "classical-v1-verified", "controls": "controls-v2", "pca_full": "bottleneck-v2-full",
    "pca_half": "bottleneck-v2-half", "pca_quarter": "bottleneck-v2-quarter",
    "variational": "variational-v3", "graph": "graph-v4-bounded", "permutations": "permutation-v2",
    "graph_variational": "graph-variational-v6", "covariation": "covariation-v1",
}
for _seed in (137, 2026):
    for _fraction in ("quarter", "half", "full"):
        SERIES[f"pca{_seed}_{_fraction}"] = f"sensitivity-pca{_seed}-{_fraction}"
    SERIES[f"graph{_seed}"] = f"sensitivity-graph{_seed}"
COLORS = {"logistic": "#475569", "rbf": "#167a9b", "polynomial": "#c56c21", "quantum": "#8255b5", "mlp": "#167a9b"}
LABELS = {"logistic": "Logistic", "rbf": "RBF SVM", "polynomial": "Polynomial SVM", "quantum": "Quantum", "mlp": "MLP"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_reports(root, allow_partial):
    reports, public = {}, {}
    for key, directory in SERIES.items():
        folder = root / directory
        if not (folder / "report.json").is_file():
            if allow_partial:
                continue
            raise ValueError(f"Result is incomplete: {directory}")
        receipt = json.loads((folder / "run.json").read_text())
        if receipt["state"] != "complete" or sha(folder / "report.json") != receipt["report_sha256"]:
            raise ValueError(f"Invalid completion receipt: {directory}")
        report = json.loads((folder / "report.json").read_text())
        if key != "permutations" and sha(folder / "predictions.csv") != report["predictions_sha256"]:
            raise ValueError(f"Prediction receipt differs: {directory}")
        reports[key] = report
        if key == "permutations":
            public[key] = {name: report[name] for name in (
                "permutations", "observed_balanced_accuracy", "null_mean", "null_percentiles", "exceedances", "empirical_one_sided_p")}
            public[key]["null_scores"] = [t["balanced_accuracy"] for t in report["trials"]]
        else:
            public[key] = {
                "summaries": report["summaries"], "seconds": report["seconds"],
                "training_fraction": report.get("training_fraction", 1),
                "folds": [
                    {
                        **{name: fold[name] for name in ("model", "held_out_group", "train_subjects", "test_subjects",
                                                         "metrics", "selected_parameters", "seconds")},
                        "diagnostics": {k: v for k, v in fold.get("diagnostics", {}).items() if not k.endswith("_sha256")},
                    }
                    for fold in report["folds"]
                ],
                "paired_comparisons": report.get("paired_comparisons", {}),
            }
            for name in ("parameter_grids", "protocol_configuration", "representation_configuration", "kernel_configuration",
                         "graph_configuration", "readout_configuration",
                         "python_version", "numpy_version", "sklearn_version", "seed"):
                if name in report:
                    public[key][name] = report[name]
        public[key].update(report_sha256=sha(folder / "report.json"),
                           source_snapshot_sha256=report["source_snapshot_sha256"],
                           dataset_sha256=report["dataset_sha256"],
                           run_started_utc=report["run_started_utc"], run_finished_utc=report["run_finished_utc"])
    if len({r["dataset_sha256"] for r in reports.values()}) != 1:
        raise ValueError("Result series refer to different datasets")
    return reports, public


def seed_statistics(reports):
    groups = sorted({f["held_out_group"] for f in reports["reference"]["folds"]})
    result = {}
    comparisons = []
    for fraction in ("quarter", "half", "full"):
        keys = [f"pca_{fraction}", f"pca137_{fraction}", f"pca2026_{fraction}"]
        if set(keys) <= reports.keys():
            for kind in ("raw", "residual"):
                comparisons.append((f"pca_{fraction}_{kind}", keys,
                                    {family: [f"{kind}_pca4_{family}"] * 3 for family in ("logistic", "rbf", "polynomial", "quantum")}))
    if {"graph", "graph137", "graph2026"} <= reports.keys():
        for kind in ("raw", "residual"):
            comparisons.append((f"graph_{kind}", ["graph", "graph137", "graph2026"],
                                {family: [f"{kind}_graph4_{family}"] * 3 for family in ("logistic", "rbf", "polynomial", "quantum")}))
    for key, prefix in (("variational", "pca"), ("graph_variational", "graph")):
        if key in reports:
            for kind in ("raw", "residual"):
                comparisons.append((f"{key}_{kind}", [key] * 3,
                                    {family: [f"{kind}_{prefix}4_seed{s}_{family}" for s in (42, 137, 2026)] for family in ("mlp", "quantum")}))
    def summarize(matrix):
        site = matrix.mean(axis=0)
        seed = matrix.mean(axis=1)
        rng = np.random.default_rng(42)
        bootstrap = site[rng.integers(0, len(site), size=(2000, len(site)))].mean(axis=1)
        return {"mean": float(site.mean()), "seed_means": dict(zip(["42", "137", "2026"], seed.tolist())),
                "seed_range": [float(seed.min()), float(seed.max())], "site_means": dict(zip(groups, site.tolist())),
                "site_bootstrap_95_percent": np.quantile(bootstrap, [0.025, 0.975]).tolist()}
    for name, keys, models in comparisons:
        matrices, summaries = {}, {}
        for family, names in models.items():
            summaries[family], matrices[family] = {}, {}
            for metric in ("balanced_accuracy", "auroc"):
                matrix = []
                for key, model in zip(keys, names):
                    folds = {f["held_out_group"]: f for f in reports[key]["folds"] if f["model"] == model}
                    if set(folds) != set(groups):
                        raise ValueError("Seed comparison group inventory differs")
                    matrix.append([folds[g]["metrics"][metric] for g in groups])
                matrix = np.array(matrix)
                matrices[family][metric] = matrix
                summaries[family][metric] = summarize(matrix)
        paired_results = {}
        for family in models.keys() - {"quantum"}:
            paired_results[f"quantum_minus_{family}"] = {
                metric: summarize(matrices["quantum"][metric] - matrices[family][metric])
                for metric in ("balanced_accuracy", "auroc")}
        result[name] = {"seeds": [42, 137, 2026], "models": summaries, "paired_differences": paired_results}
    return result


def point(report, name, metric="balanced_accuracy"):
    summary = report["summaries"][name]
    mean = summary["site_mean"][metric]
    lo, hi = summary["site_bootstrap_95_percent"][metric]
    return mean, mean - lo, hi - mean


def base_axis(ax, title, ylabel=True):
    ax.set_title(title, loc="left", fontweight="bold", pad=12)
    ax.axhline(0.5, color="#94a3b8", linewidth=1, linestyle=":", label="Chance")
    ax.set_ylim(0.35, 0.85)
    ax.grid(axis="y", color="#e2e8f0", linewidth=0.7)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel("Mean site balanced accuracy")
    ax.spines[["top", "right"]].set_visible(False)


def save(fig, output, name, footer="Intervals: descriptive 95% site bootstrap; seven acquisition groups; outer training sets overlap."):
    fig.text(0.01, 0.01, footer, fontsize=8, color="#64748b")
    fig.tight_layout(rect=(0, 0.045, 1, 0.87 if fig.legends else 1))
    fig.savefig(output / f"{name}.png", dpi=170, facecolor="white")
    fig.savefig(output / f"{name}.svg", facecolor="white", metadata={"Date": None, "Creator": "Quantum Neuro ML"})
    plt.close(fig)


def controls(reports, output):
    report = reports["controls"]
    names = [f"{kind}_{family}" for kind in ("covariates", "raw", "residual") for family in ("logistic", "rbf_expanded")]
    labels = ["Covariates / logistic", "Covariates / RBF", "Raw edges / logistic", "Raw edges / RBF",
              "Residual edges / logistic", "Residual edges / RBF"]
    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    for index, name in enumerate(names):
        mean, lo, hi = point(report, name)
        ax.errorbar(mean, index, xerr=[[lo], [hi]], fmt="o", capsize=3, color=COLORS["rbf" if "rbf" in name else "logistic"])
        ax.text(hi + mean + 0.006, index, f"{mean:.3f}", va="center", fontsize=9)
    ax.set_yticks(range(len(names)), labels)
    ax.invert_yaxis()
    ax.axvline(0.5, color="#94a3b8", linestyle=":")
    ax.set_xlim(0.4, 0.85)
    ax.set_xlabel("Mean site balanced accuracy")
    ax.set_title("Connectivity and nuisance controls", loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, output, "controls")


def compressed(reports, output):
    report = reports["pca_full"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    for ax, kind in zip(axes, ("raw", "residual")):
        for offset, family in enumerate(("logistic", "rbf", "polynomial", "quantum")):
            values = [point(report, f"{kind}_pca{d}_{family}") for d in (4, 6, 8)]
            means, lower, upper = np.array(values).T
            ax.errorbar(np.array([4, 6, 8]) + (offset - 1.5) * 0.11, means, yerr=[lower, upper],
                        color=COLORS[family], label=LABELS[family], marker="o", capsize=2, linewidth=1.3)
        base_axis(ax, "Raw connectivity" if kind == "raw" else "Nuisance residuals", ax is axes[0])
        ax.set_xticks([4, 6, 8])
        ax.set_xlabel("Retained PCA coordinates / qubits")
    handles, labels = axes[0].get_legend_handles_labels()
    order = [labels.index(LABELS[f]) for f in ("logistic", "rbf", "polynomial", "quantum")]
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper center", ncol=4, frameon=False)
    fig.subplots_adjust(top=0.82)
    save(fig, output, "pca-comparison")


def low_sample(reports, output):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    for ax, kind in zip(axes, ("raw", "residual")):
        for family in ("logistic", "rbf", "polynomial", "quantum"):
            values = [point(reports[key], f"{kind}_pca4_{family}")[0] for key in ("pca_quarter", "pca_half", "pca_full")]
            ax.plot([0.25, 0.5, 1], values, marker="o", color=COLORS[family], label=LABELS[family])
        base_axis(ax, "Raw connectivity / four coordinates" if kind == "raw" else "Nuisance residuals / four coordinates", ax is axes[0])
        ax.set_xticks([0.25, 0.5, 1], ["25%", "50%", "100%"])
        ax.set_xlabel("Fraction of each outer training stratum")
    axes[1].legend(frameon=False, fontsize=9)
    save(fig, output, "training-fractions", "Means across seven sites. Subsets are nested; all settings share the same untouched outer test participants.")


def paired(reports, output):
    report = reports["pca_full"]
    settings = [(kind, d) for kind in ("raw", "residual") for d in (4, 6, 8)]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.4), sharey=True)
    for ax, family in zip(axes, ("logistic", "rbf", "polynomial")):
        for index, (kind, d) in enumerate(settings):
            prefix = f"{kind}_pca{d}"
            comparison = report["paired_comparisons"][f"{prefix}_quantum_minus_{prefix}_{family}"]["balanced_accuracy"]
            mean = comparison["mean_quantum_minus_classical"]
            lo, hi = comparison["paired_site_bootstrap_95_percent"]
            ax.errorbar(mean, index, xerr=[[mean - lo], [hi - mean]], fmt="o", capsize=3, color=COLORS["quantum"])
        ax.axvline(0, color="#64748b", linestyle=":")
        ax.set_title("Quantum minus " + LABELS[family], fontsize=11, loc="left", fontweight="bold")
        ax.set_xlabel("Paired difference in balanced accuracy")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(range(len(settings)), [f"{k.title()}, {d} coordinates" for k, d in settings])
    axes[0].invert_yaxis()
    save(fig, output, "paired-kernel-differences")


def variational(reports, output, key="variational", representation="PCA"):
    report = reports[key]
    prefix = representation.lower()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    for ax, kind in zip(axes, ("raw", "residual")):
        for offset, family in enumerate(("mlp", "quantum")):
            values = [point(report, f"{kind}_{prefix}4_seed{s}_{family}") for s in (42, 137, 2026)]
            mean, lower, upper = np.array(values).T
            ax.errorbar(np.arange(3) + (offset - 0.5) * 0.12, mean, yerr=[lower, upper], fmt="o", capsize=3,
                        color=COLORS[family], label=f"{LABELS[family]} ({19 if family == 'mlp' else 21} parameters)")
        base_axis(ax, f"Raw connectivity / {representation}4" if kind == "raw" else f"Nuisance residuals / {representation}4", ax is axes[0])
        ax.set_xticks(range(3), ["42", "137", "2026"])
        ax.set_xlabel("Training seed")
    axes[1].legend(frameon=False, fontsize=9)
    save(fig, output, "variational-readouts" if key == "variational" else "graph-variational-readouts")


def graph(reports, output):
    report = reports["graph"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    for ax, kind in zip(axes, ("raw", "residual")):
        for index, family in enumerate(("logistic", "rbf", "polynomial", "quantum")):
            mean, lo, hi = point(report, f"{kind}_graph4_{family}")
            ax.errorbar(index, mean, yerr=[[lo], [hi]], fmt="o", capsize=4, color=COLORS[family])
            ax.text(index, mean + hi + 0.012, f"{mean:.3f}", ha="center", fontsize=9)
        base_axis(ax, "Graph from raw connectivity" if kind == "raw" else "Graph from nuisance residuals", ax is axes[0])
        ax.set_xticks(range(4), ["Logistic", "RBF", "Polynomial", "Quantum"], rotation=15)
        ax.set_xlabel("Readout of the same four-dimensional graph embedding")
    save(fig, output, "graph-readouts")


def covariation(reports, output):
    report = reports["covariation"]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4), sharey=True)
    families = (("Logistic", "logistic", "-"), ("RBF SVM", "rbf", "-"),
                ("Quantum depth 1", "quantum_depth1", "--"),
                ("Quantum depth 2", "quantum_depth2", ":"))
    for ax, kind in zip(axes, ("raw", "residual")):
        for label, family, linestyle in families:
            dimensions = (4,) if label.startswith("Quantum") else (4, 8, 16)
            points = [point(report, f"{kind}_covar{dimension}_{family}")
                      for dimension in dimensions]
            color = COLORS["quantum"] if label.startswith("Quantum") else COLORS[family]
            ax.errorbar(dimensions, [item[0] for item in points],
                        yerr=[[item[1] for item in points], [item[2] for item in points]],
                        marker="o", capsize=3, linestyle=linestyle, color=color, label=label)
        base_axis(ax, "Raw connectivity" if kind == "raw" else "Nuisance residuals", ax is axes[0])
        ax.set_xticks([4, 8, 16])
        ax.set_xlabel("Learned edge co-variation modules")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Classification from training-fold edge co-variation groups", x=0.02,
                 ha="left", fontweight="bold")
    save(fig, output, "edge-covariation-modules",
         "Modules learned without labels from training participants only; intervals are descriptive 95% site bootstrap.")


def permutations(reports, output):
    report = reports["permutations"]
    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    values = [t["balanced_accuracy"] for t in report["trials"]]
    ax.hist(values, bins=14, color="#167a9b", edgecolor="white", label="99 within-site label permutations")
    observed = report["observed_balanced_accuracy"]
    ax.axvline(observed, color="#b24b3d", linewidth=2, label=f"Observed logistic reference: {observed:.3f}")
    ax.set_xlim(min(0.35, min(values) - 0.02), max(0.74, observed + 0.05))
    ax.set_xlabel("Mean site balanced accuracy after full nested refitting")
    ax.set_ylabel("Permutation count")
    ax.set_title(f"Reference signal check: empirical p = {report['empirical_one_sided_p']:.2f}", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, output, "label-permutations", "One-sided +1 correction. Resolution is 0.01 with 99 permutations. This test does not remove nuisance confounding.")


def seed_figures(statistics, output):
    panels = ["pca_full_raw", "pca_full_residual", "graph_raw", "graph_residual"]
    if set(panels) <= statistics.keys():
        fig, axes = plt.subplots(2, 2, figsize=(10, 7.4), sharey=True)
        for ax, panel, title in zip(axes.flat, panels, ("PCA4 / raw", "PCA4 / residual", "Graph4 / raw", "Graph4 / residual")):
            for index, family in enumerate(("logistic", "rbf", "polynomial", "quantum")):
                values = statistics[panel]["models"][family]["balanced_accuracy"]
                mean = values["mean"]
                lo, hi = values["site_bootstrap_95_percent"]
                ax.scatter(np.array([-0.1, 0, 0.1]) + index, list(values["seed_means"].values()),
                           color=COLORS[family], alpha=0.3, s=18)
                ax.errorbar(index, mean, yerr=[[mean - lo], [hi - mean]], fmt="o", capsize=4, color=COLORS[family])
            base_axis(ax, title)
            ax.set_xticks(range(4), ["Logistic", "RBF", "Polynomial", "Quantum"])
        save(fig, output, "seed-robustness", "Large points: seeds averaged within site, then sites averaged. Intervals: seven-site bootstrap. Small points: individual seed means.")
    needed = [f"pca_{fraction}_{kind}" for fraction in ("quarter", "half", "full") for kind in ("raw", "residual")]
    if set(needed) <= statistics.keys():
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
        for ax, kind in zip(axes, ("raw", "residual")):
            for family in ("logistic", "rbf", "polynomial", "quantum"):
                values = [statistics[f"pca_{fraction}_{kind}"]["models"][family]["balanced_accuracy"]
                          for fraction in ("quarter", "half", "full")]
                means = [v["mean"] for v in values]
                lower, upper = np.array([v["seed_range"] for v in values]).T
                ax.plot([0.25, 0.5, 1], means, marker="o", color=COLORS[family], label=LABELS[family])
                ax.fill_between([0.25, 0.5, 1], lower, upper, color=COLORS[family], alpha=0.1)
            base_axis(ax, "Raw / PCA4" if kind == "raw" else "Residual / PCA4", ax is axes[0])
            ax.set_xticks([0.25, 0.5, 1], ["25%", "50%", "100%"])
            ax.set_xlabel("Fraction of each outer training stratum")
        axes[1].legend(frameon=False, fontsize=9)
        save(fig, output, "training-fractions-three-seeds", "Lines: mean over seeds 42, 137, 2026 and seven sites. Bands: minimum to maximum seed means; these are not confidence intervals.")
    panels = ["variational_raw", "variational_residual", "graph_variational_raw", "graph_variational_residual"]
    if set(panels) <= statistics.keys():
        fig, ax = plt.subplots(figsize=(8.6, 4.3))
        for index, panel in enumerate(panels):
            value = statistics[panel]["paired_differences"]["quantum_minus_mlp"]["balanced_accuracy"]
            mean = value["mean"]
            lo, hi = value["site_bootstrap_95_percent"]
            ax.errorbar(mean, index, xerr=[[mean - lo], [hi - mean]], fmt="o", capsize=4, color=COLORS["quantum"])
        ax.set_yticks(range(4), ["PCA4 / raw", "PCA4 / residual", "Frozen graph4 / raw", "Frozen graph4 / residual"])
        ax.invert_yaxis()
        ax.axvline(0, linestyle=":", color="#64748b")
        ax.set_xlabel("Paired balanced accuracy: quantum readout minus MLP")
        ax.set_title("Trainable readouts on shared four-dimensional inputs", loc="left", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        save(fig, output, "paired-trainable-readouts", "Seeds averaged within each site before paired site bootstrap. 21 quantum parameters versus 19 MLP parameters; exploratory comparisons.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/figures"))
    parser.add_argument("--aggregate-output", type=Path, default=Path("docs/results/aggregate.json"))
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    reports, public = load_reports(args.results_root, args.allow_partial)
    sensitivity = seed_statistics(reports)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlepad": 12,
                         "svg.hashsalt": "quantum-neuro-ml"})
    for needed, function in (({"controls"}, controls), ({"pca_full"}, compressed),
                             ({"pca_full", "pca_half", "pca_quarter"}, low_sample), ({"pca_full"}, paired),
                             ({"variational"}, variational), ({"graph"}, graph), ({"permutations"}, permutations)):
        if needed <= reports.keys():
            function(reports, args.output_dir)
    if "graph_variational" in reports:
        variational(reports, args.output_dir, key="graph_variational", representation="Graph")
    if "covariation" in reports:
        covariation(reports, args.output_dir)
    seed_figures(sensitivity, args.output_dir)
    with (args.results_root / SERIES["reference"] / "predictions.csv").open(newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if r["model"] == "logistic"]
    cohort = {"subjects": len(rows), "ASD": sum(int(r["label"]) for r in rows),
              "controls": sum(1 - int(r["label"]) for r in rows),
              "acquisition_groups": len({r["held_out_group"] for r in rows})}
    payload = {"cohort": cohort,
               "evaluation": "nested leave-one-acquisition-group-out; three inner grouped folds",
               "intervals": "descriptive site bootstrap; overlapping training sets; exploratory settings",
               "plot_script_sha256": sha(Path(__file__)), "matplotlib_version": matplotlib.__version__,
               "experiments": public, "seed_sensitivity": sensitivity}
    if args.audit:
        audit = json.loads(args.audit.read_text())
        if audit["audit"] != "passed":
            raise ValueError("Artifact audit did not pass")
        audited = {r["report_sha256"] for r in audit["experiments"]}
        if "permutation_test" in audit:
            audited.add(audit["permutation_test"]["report_sha256"])
        if not {r["report_sha256"] for r in public.values()} <= audited:
            raise ValueError("Some figures refer to unaudited reports")
        payload["audit_report_sha256"] = sha(args.audit)
    args.aggregate_output.parent.mkdir(parents=True, exist_ok=True)
    args.aggregate_output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"complete_series": sorted(reports), "aggregate": str(args.aggregate_output), "figures": str(args.output_dir)}))


if __name__ == "__main__":
    main()
