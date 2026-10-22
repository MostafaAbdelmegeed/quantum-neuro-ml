# Reproducing the benchmark

Run from the repository root with Python 3.11. The recorded Windows environment uses Python 3.11.15, NumPy 2.4.6, scikit-learn 1.9.1, PyTorch 2.14.1, and PennyLane 0.45.1. `uv.lock` fixes the package resolution; keep the Python version fixed because the lock also supports newer Python versions with different compatible packages.

```bash
uv sync --locked --python 3.11 --extra dev --extra graph --extra quantum
uv run --locked --extra dev --extra graph --extra quantum python -m pytest
```

## Data and reference run

Follow the [data-use requirements](data-use.md) and [metadata preparation](classical-baselines.md). Then build a new verified dataset and reference run:

```bash
uv run --locked --extra dev --extra graph --extra quantum qneuro-abide-build --workers 4 --output-dir data/cohort/v1-verified
uv run --locked --extra dev --extra graph --extra quantum qneuro-abide-baseline --dataset data/cohort/v1-verified/connectivity.npz --output-dir results/classical-v1-verified
```

Add `--offline` to the dataset build when its downloaded scan cache is complete. The cohort policy and metadata fingerprint must match; do not replace them to bypass a failed inclusion check.

## Main comparisons

The commands below create separate result series. A completed or partially populated result directory is not overwritten; choose a new name for a new attempt. The defaults use the verified dataset above.

```bash
uv run --locked --extra dev --extra graph --extra quantum qneuro-followup --phase controls --output-dir results/controls-v2
uv run --locked --extra dev --extra graph --extra quantum qneuro-followup --phase bottleneck --output-dir results/bottleneck-v2-full
uv run --locked --extra dev --extra graph --extra quantum qneuro-followup --phase bottleneck --fraction 0.25 --output-dir results/bottleneck-v2-quarter
uv run --locked --extra dev --extra graph --extra quantum qneuro-followup --phase bottleneck --fraction 0.5 --output-dir results/bottleneck-v2-half
uv run --locked --extra dev --extra graph --extra quantum python -m quantum_neuro_ml.variational_experiment --output-dir results/variational-v3
uv run --locked --extra dev --extra graph --extra quantum python -m quantum_neuro_ml.graph_experiment --config config/graph-v4.json --reuse-cache-from results/graph-v4/cache --output-dir results/graph-v4-bounded
uv run --locked --extra dev --extra graph --extra quantum python scripts/graph_variational.py --output-dir results/graph-variational-v6
uv run --locked --extra dev --extra graph --extra quantum python -m quantum_neuro_ml.permutation --output-dir results/permutation-v2 --workers 2
```

Each run writes `run.json` with its current state, a source archive, and progress records. Only `state: complete` together with a matching final report receipt indicates completion. The code captures source, lockfile, configuration, and input hashes before fitting and refuses completion if a captured file changes during execution. Keep source and environment files fixed until active runs finish. Separate experiment processes may run concurrently when resources permit.

The original pre-hardening run is preserved locally. The verified rerun matched its predictions byte for byte. The first graph run was interrupted after an uncapped polynomial SVM solver remained in one inner fit for more than nine hours; its partial receipt is retained as `results/graph-v4/`. The revised graph run records a 100,000-iteration cap and 0.01 tolerance for that readout. To reconstruct historical source bytes, use each run's `source.zip`; the current checkout may include later models and tests. Run archives and prediction files contain participant information and remain in ignored local directories.

The revised graph command above reuses the local cache from the interrupted run. Its byte hashes are captured in `cache-manifest.json` and the run archive. On a clean checkout, omit `--reuse-cache-from results/graph-v4/cache` and use a new output directory; the encoder will rebuild the same fold representations, with longer runtime.

## Seed repetitions

The [seed protocol](seed-sensitivity-protocol.md) specifies two further PCA and graph repetitions. This PowerShell loop runs all eight additional series sequentially:

```powershell
foreach ($experimentSeed in @(137, 2026)) {
    foreach ($setting in @(@("quarter", "0.25"), @("half", "0.5"), @("full", "1.0"))) {
        uv run --locked --extra dev --extra graph --extra quantum qneuro-followup --phase bottleneck --fraction $($setting[1]) --config "config/sensitivity/pca-seed${experimentSeed}.json" --output-dir "results/sensitivity-pca${experimentSeed}-$($setting[0])"
        if ($LASTEXITCODE -ne 0) { throw "PCA repetition failed" }
    }
    uv run --locked --extra dev --extra graph --extra quantum python -m quantum_neuro_ml.graph_experiment --config "config/sensitivity/graph-seed$experimentSeed.json" --output-dir "results/sensitivity-graph$experimentSeed"
    if ($LASTEXITCODE -ne 0) { throw "Graph repetition failed" }
}
```

## Audit and figures

Audit all completed predictions, grouped partitions, primary metrics, bootstrap summaries, source receipts, and null-statistic arithmetic. The auditor recomputes the observed permutation predictions against the original reference. It reconstructs every shuffled target and checks the null summaries; it does not fit the 99 null classifiers a second time.

```powershell
$resultNames = @("classical-v1-verified", "controls-v2", "bottleneck-v2-full", "bottleneck-v2-quarter", "bottleneck-v2-half", "variational-v3", "graph-v4-bounded", "graph-variational-v6")
foreach ($experimentSeed in @(137, 2026)) {
    foreach ($setting in @("quarter", "half", "full")) {
        $resultNames += "sensitivity-pca$experimentSeed-$setting"
    }
    $resultNames += "sensitivity-graph$experimentSeed"
}
$auditArguments = @("python", "scripts/audit_results.py", "--permutation-dir", "results/permutation-v2", "--output-dir", "results/audit-final-v1")
foreach ($name in $resultNames) { $auditArguments += @("--result-dir", "results/$name") }
uv run --locked --extra dev --extra graph --extra quantum @auditArguments
if ($LASTEXITCODE -ne 0) { throw "Artifact audit failed" }
```

Export the public aggregate JSON and PNG/SVG figures in a separate plotting environment:

```bash
uv run --no-project --python 3.11 --with matplotlib==3.11.2 --with numpy==2.4.6 python scripts/plot_results.py --audit results/audit-final-v1/report.json
```

The plotting command checks that every report it exports appears in the passed audit. It writes aggregate scores and model settings to `docs/results/aggregate.json` and figures to `docs/figures/`. It exports no participant rows. Three-seed intervals average seeds within site before site resampling; seed ranges are labelled separately from confidence intervals.
