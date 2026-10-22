"""Training-fold groups of connectivity edges that co-vary across participants."""

from __future__ import annotations

import hashlib

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import AgglomerativeClustering
from sklearn.utils.validation import check_is_fitted


def signed_jaccard(first: dict[int, int], second: dict[int, int]) -> float:
    union = set(first) | set(second)
    if not union:
        return 1.0
    common = set(first) & set(second)
    same = sum(first[edge] == second[edge] for edge in common)
    flipped = sum(first[edge] == -second[edge] for edge in common)
    return max(same, flipped) / len(union)


def compare_modules(first: list[dict[int, int]], second: list[dict[int, int]]) -> tuple[float, float]:
    similarities = np.array([[signed_jaccard(a, b) for b in second] for a in first])
    rows, columns = linear_sum_assignment(-similarities)
    return float(similarities[rows, columns].mean()), float(similarities.max(axis=1).mean())


class EdgeCovariationGroups(TransformerMixin, BaseEstimator):
    """Summarize participant connectivity by unsupervised edge co-variation modules.

    Candidate edges are selected by training-fold variance. Edges are then grouped by
    absolute cross-participant correlation, oriented to a within-module medoid, and
    averaged per participant. Neither labels nor held-out rows are used during fit.
    """

    def __init__(self, dimensions: int = 4, candidate_edges: int = 256):
        self.dimensions = dimensions
        self.candidate_edges = candidate_edges

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=np.float64)
        if (values.ndim != 2 or values.shape[0] < 4 or not np.isfinite(values).all()
                or not isinstance(self.dimensions, int) or self.dimensions < 2
                or not isinstance(self.candidate_edges, int) or self.candidate_edges < self.dimensions
                or self.candidate_edges > values.shape[1]):
            raise ValueError("Invalid edge matrix or co-variation grouping configuration")
        self.n_features_in_ = values.shape[1]
        variance = np.var(values, axis=0)
        candidates = np.argsort(variance, kind="stable")[-self.candidate_edges:]
        # Stable order makes module IDs and receipts independent of argsort ties.
        candidates = np.sort(candidates)
        center = values[:, candidates].mean(axis=0)
        scale = values[:, candidates].std(axis=0, ddof=0)
        keep = scale > np.finfo(np.float64).eps
        candidates, center, scale = candidates[keep], center[keep], scale[keep]
        if len(candidates) < self.dimensions:
            raise ValueError("Too few variable candidate edges for requested modules")
        standardized = (values[:, candidates] - center) / scale
        correlation = standardized.T @ standardized / max(1, len(values) - 1)
        correlation = np.clip(correlation, -1.0, 1.0)
        np.fill_diagonal(correlation, 1.0)
        distance = np.clip(1.0 - np.abs(correlation), 0.0, 1.0)
        labels = AgglomerativeClustering(
            n_clusters=self.dimensions, metric="precomputed", linkage="average"
        ).fit_predict(distance)

        modules = []
        signs = []
        within = []
        for module in range(self.dimensions):
            members = np.flatnonzero(labels == module)
            if not len(members):
                raise ValueError("Co-variation clustering produced an empty module")
            medoid = members[np.argmax(np.abs(correlation[np.ix_(members, members)]).sum(axis=1))]
            orient = np.sign(correlation[members, medoid])
            orient[orient == 0] = 1
            modules.append(candidates[members])
            signs.append(orient.astype(np.float64))
            if len(members) > 1:
                block = np.abs(correlation[np.ix_(members, members)])
                within.extend(block[np.triu_indices(len(members), k=1)].tolist())
        self.edge_mean_ = center
        self.edge_scale_ = scale
        self.candidate_indices_ = candidates
        self.modules_ = modules
        self.signs_ = signs
        assignment = np.full(len(candidates), -1, dtype=np.int32)
        for module, members in enumerate(modules):
            assignment[np.searchsorted(candidates, members)] = module
        assignment_bytes = assignment.astype("<i4", copy=False).tobytes()
        self.diagnostics_ = {
            "method": "training_variance_top_edges_abs_correlation_average_linkage_medoid_signed_mean",
            "candidate_edges": int(len(candidates)),
            "requested_candidate_edges": int(self.candidate_edges),
            "modules": int(self.dimensions),
            "module_sizes": [int(len(module)) for module in modules],
            "mean_within_module_absolute_correlation": float(np.mean(within)) if within else 1.0,
            "module_assignment_sha256": hashlib.sha256(assignment_bytes).hexdigest(),
            "training_participants": int(len(values)),
            "labels_used_for_learning": False,
        }
        return self

    def transform(self, X):
        check_is_fitted(self, "modules_")
        values = np.asarray(X, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.n_features_in_ or not np.isfinite(values).all():
            raise ValueError("Connectivity feature count changed or values are not finite")
        result = np.empty((len(values), self.dimensions), dtype=np.float64)
        for index, (members, signs) in enumerate(zip(self.modules_, self.signs_)):
            offsets = np.searchsorted(self.candidate_indices_, members)
            selected = values[:, members] - self.edge_mean_[offsets]
            result[:, index] = np.mean(selected / self.edge_scale_[offsets] * signs, axis=1)
        return result
