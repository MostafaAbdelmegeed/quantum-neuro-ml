"""One nested acquisition-group evaluation path for all model families."""

from __future__ import annotations

import hashlib
import platform
import time
import warnings
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import sklearn
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold, LeaveOneGroupOut
from threadpoolctl import threadpool_limits


@dataclass(frozen=True)
class EvaluationData:
    features: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    subject_ids: np.ndarray


@dataclass(frozen=True)
class ModelPlan:
    name: str
    estimator: object
    grid: dict[str, list]


def validate_arrays(features: np.ndarray, labels: np.ndarray, groups: np.ndarray,
                    subject_ids: np.ndarray) -> None:
    if features.ndim != 2 or min(features.shape) < 1 or not np.isfinite(features).all():
        raise ValueError("Features must be a nonempty finite matrix")
    n = len(features)
    if any(array.ndim != 1 or len(array) != n for array in (labels, groups, subject_ids)):
        raise ValueError("Dataset arrays must be aligned one-dimensional vectors")
    if not np.issubdtype(subject_ids.dtype, np.integer) or len(np.unique(subject_ids)) != n:
        raise ValueError("Duplicate participant IDs or noninteger participant identifiers")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Expected binary labels 0 and 1")
    if len(np.unique(groups)) < 4:
        raise ValueError("Nested evaluation requires at least four acquisition groups")
    for group in np.unique(groups):
        if not str(group).strip() or set(labels[groups == group]) != {0, 1}:
            raise ValueError("Every acquisition group must contain both diagnoses")


def classification_metrics(y: np.ndarray, predicted: np.ndarray, score: np.ndarray,
                           probability: np.ndarray | None) -> dict[str, float | None]:
    result = {
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "auroc": float(roc_auc_score(y, score)),
        "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y, predicted)), "brier": None, "ece_10_bins": None,
    }
    if probability is not None:
        result["brier"] = float(brier_score_loss(y, probability))
        bins = np.minimum((probability * 10).astype(int), 9)
        result["ece_10_bins"] = float(sum(
            np.mean(bins == index) * abs(np.mean(y[bins == index]) - np.mean(probability[bins == index]))
            for index in range(10) if np.any(bins == index)
        ))
    return result


def sample_training_indices(data: EvaluationData, train: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    if not 0 < fraction <= 1:
        raise ValueError("Training fraction must be in (0, 1]")
    if fraction == 1:
        return train
    selected = []
    for group in sorted(set(data.groups[train])):
        for label in (0, 1):
            candidates = train[(data.groups[train] == group) & (data.labels[train] == label)]
            candidates = candidates[np.argsort(data.subject_ids[candidates])]
            count = min(len(candidates), max(2, int(np.ceil(len(candidates) * fraction))))
            token = hashlib.sha256(f"{seed}:{group}:{label}".encode()).digest()
            rng = np.random.default_rng(int.from_bytes(token[:8], "little"))
            selected.extend(rng.permutation(candidates)[:count].tolist())
    return np.array(sorted(selected), dtype=int)


def _array_digest(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256(str((array.shape, array.dtype.str)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _fit_plan(data: EvaluationData, plan: ModelPlan, train: np.ndarray, test: np.ndarray,
              inner: list[tuple[np.ndarray, np.ndarray]]) -> tuple[dict, list[dict]]:
    started = time.perf_counter()
    estimator = clone(plan.estimator)
    parameters, inner_score = {}, None
    convergence_warnings = 0
    if plan.grid:
        search = GridSearchCV(estimator, plan.grid, scoring="balanced_accuracy", cv=inner,
                              refit=True, n_jobs=1, error_score="raise")
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always", ConvergenceWarning)
            search.fit(data.features[train], data.labels[train])
        convergence_warnings = sum(issubclass(item.category, ConvergenceWarning) for item in captured)
        fitted, parameters = search.best_estimator_, search.best_params_
        inner_score = float(search.best_score_)
    else:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always", ConvergenceWarning)
            fitted = estimator.fit(data.features[train], data.labels[train])
        convergence_warnings = sum(issubclass(item.category, ConvergenceWarning) for item in captured)
    predicted = fitted.predict(data.features[test])
    probability = fitted.predict_proba(data.features[test])[:, 1] if hasattr(fitted, "predict_proba") else None
    score = fitted.decision_function(data.features[test]) if hasattr(fitted, "decision_function") else probability
    if score is None or not np.isfinite(score).all():
        raise ValueError("Classifier did not return finite binary scores")
    metrics = classification_metrics(data.labels[test], predicted, score, probability)
    diagnostics = {}
    diagnostics["convergence_warning_count"] = int(convergence_warnings)
    if hasattr(fitted, "steps"):
        fit_status = getattr(fitted.steps[-1][1], "fit_status_", None)
        if fit_status is not None:
            diagnostics["readout_fit_status"] = int(fit_status)
    if hasattr(fitted, "steps"):
        representation = fitted[:-1]
        diagnostics["training_representation_sha256"] = _array_digest(representation.transform(data.features[train]))
        diagnostics["test_representation_sha256"] = _array_digest(representation.transform(data.features[test]))
        for name, step in fitted.steps:
            if hasattr(step, "explained_variance_ratio_"):
                diagnostics[f"{name}_explained_variance_ratio"] = step.explained_variance_ratio_.tolist()
            if hasattr(step, "diagnostics_"):
                diagnostics[name] = step.diagnostics_
    held_out = str(data.groups[test[0]])
    fold = {"model": plan.name, "held_out_group": held_out,
            "train_subjects": len(train), "test_subjects": len(test),
            "selected_parameters": parameters, "inner_balanced_accuracy": inner_score,
            "metrics": metrics, "diagnostics": diagnostics, "seconds": time.perf_counter() - started}
    predictions = [{
        "model": plan.name, "subject_id": int(data.subject_ids[index]), "held_out_group": held_out,
        "label": int(data.labels[index]), "prediction": int(predicted[offset]),
        "score": float(score[offset]), "probability": float(probability[offset]) if probability is not None else None,
    } for offset, index in enumerate(test)]
    return fold, predictions


def summarize_predictions(folds: list[dict], predictions: list[dict], seed: int) -> dict[str, object]:
    summaries = {}
    for name in sorted({fold["model"] for fold in folds}):
        results = [fold for fold in folds if fold["model"] == name]
        means = {key: float(np.mean([fold["metrics"][key] for fold in results]))
                 if results[0]["metrics"][key] is not None else None for key in results[0]["metrics"]}
        intervals = {}
        for key in ("balanced_accuracy", "auroc"):
            values = np.array([fold["metrics"][key] for fold in results])
            rng = np.random.default_rng(seed)
            bootstrap = values[rng.integers(0, len(values), size=(2000, len(values)))].mean(axis=1)
            intervals[key] = np.quantile(bootstrap, [0.025, 0.975]).tolist()
        rows = [row for row in predictions if row["model"] == name]
        probabilities = None if rows[0]["probability"] is None else np.array([r["probability"] for r in rows])
        pooled = classification_metrics(np.array([r["label"] for r in rows]), np.array([r["prediction"] for r in rows]),
                                        np.array([r["score"] for r in rows]), probabilities)
        summaries[name] = {"site_mean": means, "site_bootstrap_95_percent": intervals,
                           "pooled_out_of_fold": pooled}
    return summaries


def evaluate_models(data: EvaluationData, plans: Sequence[ModelPlan], *, seed: int = 42,
                    training_fraction: float = 1.0,
                    on_fold: Callable[[dict, list[dict]], None] | None = None,
                    verbose: bool = True) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Fit each full model plan in inner folds and predict outer sites once."""

    validate_arrays(data.features, data.labels, data.groups, data.subject_ids)
    if not plans or len({plan.name for plan in plans}) != len(plans):
        raise ValueError("Model plans must have distinct names")
    folds, predictions = [], []
    started = time.perf_counter()
    outer = list(LeaveOneGroupOut().split(data.features, data.labels, data.groups))
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        for train, test in outer:
            train = sample_training_indices(data, train, training_fraction, seed)
            held_out = str(data.groups[test[0]])
            training_groups = data.groups[train]
            inner = list(GroupKFold(n_splits=3).split(data.features[train], data.labels[train], training_groups))
            inner_audit = []
            for inner_train, inner_valid in inner:
                train_groups, valid_groups = set(training_groups[inner_train]), set(training_groups[inner_valid])
                if train_groups & valid_groups or held_out in train_groups | valid_groups:
                    raise ValueError("Acquisition-group leakage in inner folds")
                if any(set(data.labels[train][indices]) != {0, 1} for indices in (inner_train, inner_valid)):
                    raise ValueError("An inner partition lacks a diagnosis group")
                inner_audit.append({"train_groups": sorted(train_groups), "validation_groups": sorted(valid_groups)})
            for plan in plans:
                fold, rows = _fit_plan(data, plan, train, test, inner)
                fold.update(training_groups=sorted(set(training_groups)), inner_folds=inner_audit,
                            training_subject_ids_sha256=_array_digest(data.subject_ids[train]))
                folds.append(fold)
                predictions.extend(rows)
                if verbose:
                    print(f"{held_out}: {plan.name} balanced_accuracy={fold['metrics']['balanced_accuracy']:.3f} AUROC={fold['metrics']['auroc']:.3f}", flush=True)
                if on_fold is not None:
                    on_fold(fold, rows)
    report = {
        "purpose": "nested_group_evaluation", "seed": seed, "training_fraction": training_fraction,
        "outer_cv": "leave_one_acquisition_group_out", "inner_cv": "3_fold_GroupKFold",
        "selection_metric": "balanced_accuracy", "subjects": len(data.features),
        "acquisition_groups": len(outer), "feature_count": data.features.shape[1],
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__, "threads": 1,
        "system": platform.platform(), "processor": platform.processor(),
        "parameter_grids": {plan.name: plan.grid for plan in plans},
        "seconds": time.perf_counter() - started,
        "summaries": summarize_predictions(folds, predictions, seed), "folds": folds,
    }
    return report, predictions
