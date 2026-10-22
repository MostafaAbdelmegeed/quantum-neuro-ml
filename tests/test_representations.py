import numpy as np
from quantum_neuro_ml.representations import CovariateDesign, NuisanceResidualizer, BoundedAngles


def test_residualization_removes_known_training_nuisance_and_keeps_coefficients_fixed() -> None:
    rng = np.random.default_rng(42)
    age = rng.uniform(10, 50, 80)
    sex = rng.choice([1, 2], 80)
    fd = rng.uniform(0, 0.2, 80)
    covariates = np.column_stack([age, sex, fd])
    design = np.column_stack([np.ones(80), age, age ** 2, sex == 2, fd])
    coefficients = rng.normal(size=(5, 4))
    edges = design @ coefficients
    values = np.column_stack([covariates, edges])
    transformer = NuisanceResidualizer().fit(values[:60])
    np.testing.assert_allclose(transformer.transform(values), 0, atol=1e-10)
    before = transformer.coefficients_.copy()
    changed = values[60:].copy()
    changed[:, 0] += 100
    transformer.transform(changed)
    np.testing.assert_array_equal(transformer.coefficients_, before)
    assert transformer.diagnostics_["training_subjects"] == 60


def test_covariate_design_and_angles_have_fixed_semantics() -> None:
    values = np.array([[20, 1, 0.1, 5], [30, 2, 0.2, 6]], dtype=float)
    np.testing.assert_array_equal(CovariateDesign().fit_transform(values), [[20, 400, 0, 0.1], [30, 900, 1, 0.2]])
    angles = BoundedAngles().fit_transform(np.array([[-100, 0, 100]], dtype=float))
    np.testing.assert_allclose(angles, [[-np.pi, 0, np.pi]])
