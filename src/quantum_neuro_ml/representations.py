"""Training-fold nuisance adjustment and shared compressed coordinates."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import StandardScaler


def _matrix(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 4 or not np.isfinite(values).all():
        raise ValueError("Expected age, sex, FD, and finite connectivity columns")
    if np.any(values[:, 0] <= 0) or not np.isin(values[:, 1], [1, 2]).all() or np.any(values[:, 2] < 0):
        raise ValueError("Invalid nuisance covariates")
    return values


class CovariateDesign(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.n_features_in_ = _matrix(X).shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self)
        values = _matrix(X)
        if values.shape[1] != self.n_features_in_:
            raise ValueError("Feature count changed")
        age, sex, fd = values[:, 0], values[:, 1], values[:, 2]
        return np.column_stack([age, age ** 2, sex == 2, fd])


class EdgeFeatures(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.n_features_in_ = _matrix(X).shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self)
        values = _matrix(X)
        if values.shape[1] != self.n_features_in_:
            raise ValueError("Feature count changed")
        return values[:, 3:]


class NuisanceResidualizer(TransformerMixin, BaseEstimator):
    """Regress edges on four covariates using training participants only."""

    def fit(self, X, y=None):
        values = _matrix(X)
        self.n_features_in_ = values.shape[1]
        self.covariates_ = CovariateDesign().fit(values)
        self.scale_ = StandardScaler().fit(self.covariates_.transform(values))
        design = self._design(values)
        self.coefficients_, _, rank, _ = np.linalg.lstsq(design, values[:, 3:], rcond=None)
        self.diagnostics_ = {"training_subjects": len(values), "design_rank": int(rank), "covariate_columns": 4}
        return self

    def _design(self, values):
        covariates = self.scale_.transform(self.covariates_.transform(values))
        return np.column_stack([np.ones(len(values)), covariates])

    def transform(self, X):
        check_is_fitted(self, "coefficients_")
        values = _matrix(X)
        if values.shape[1] != self.n_features_in_:
            raise ValueError("Feature count changed")
        return values[:, 3:] - self._design(values) @ self.coefficients_


class FirstComponents(TransformerMixin, BaseEstimator):
    def __init__(self, dimensions: int):
        self.dimensions = dimensions

    def fit(self, X, y=None):
        if self.dimensions < 1 or np.asarray(X).shape[1] < self.dimensions:
            raise ValueError("Requested components are unavailable")
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self)
        return np.asarray(X)[:, :self.dimensions]


class BoundedAngles(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self)
        values = np.asarray(X, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.n_features_in_ or not np.isfinite(values).all():
            raise ValueError("Invalid compressed coordinates")
        return np.pi * np.tanh(values / 2)
