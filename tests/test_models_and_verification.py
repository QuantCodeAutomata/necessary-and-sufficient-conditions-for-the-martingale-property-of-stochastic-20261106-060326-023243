"""Smoke tests for the model factories in src/models.py and the shared
Monte Carlo verification routine in src/mc_verification.py, run on small
synthetic configurations (no external data, paper Section 6 examples)."""

from __future__ import annotations

import numpy as np

from src.mc_verification import compare_mass_survival, loss_of_mass_curve
from src.models import (
    cir_2d_model,
    critical_log_cir,
    model_32,
    power_volatility_isotropic,
)


def _small_mc_kwargs():
    return dict(T=1.0, n_steps=100, n_paths=800, seed=11)


def test_cir_2d_model_shapes_and_bounds():
    model = cir_2d_model(alpha=0.6, kappa=1.5, eta=1.0, theta_coef=0.4)
    result = compare_mass_survival(
        model, y0=np.array([1.0, 1.0]), dim=2, dim_w=2,
        positivity_floor=1e-6, **_small_mc_kwargs(),
    )
    assert 0.0 <= result.e_s_t <= 1.5
    assert 0.0 <= result.survival_prob <= 1.0
    assert result.n_paths == 800


def test_model_32_true_martingale_regime_keeps_mass_near_one():
    """alpha + rho*eta - 0.5*eta^2 >= 0 (Example 6.3 threshold): E[S_T]
    should stay close to 1 and far above the strict-local regime below."""
    alpha, kappa, eta, rho = 0.9, 1.0, 1.0, 0.1  # alpha+rho*eta=1.0 >= 0.5*eta^2=0.5
    model = model_32(alpha=alpha, kappa=kappa, eta=eta, rho=rho)
    result = compare_mass_survival(
        model, y0=np.array([1.0]), dim=1, dim_w=2, positivity_floor=1e-6,
        **_small_mc_kwargs(),
    )
    assert result.e_s_t > 0.9


def test_model_32_strict_local_regime_loses_mass():
    alpha, kappa, eta, rho = 0.1, 1.0, 1.0, -0.6  # alpha+rho*eta=-0.5 < 0.5*eta^2=0.5
    model = model_32(alpha=alpha, kappa=kappa, eta=eta, rho=rho)
    result = compare_mass_survival(
        model, y0=np.array([1.0]), dim=1, dim_w=2, positivity_floor=1e-6,
        **_small_mc_kwargs(),
    )
    assert result.e_s_t < 0.95


def test_power_volatility_explosive_regime_has_heavier_tail_than_benign():
    """Example 6.1: for rho>0, p>1 the shifted radial process can explode,
    which manifests on a short, coarse Monte Carlo run as a much heavier
    right tail (larger Monte Carlo standard error) of S_T than the benign
    rho<=0 case, whose S_T stays tightly concentrated near 1. The exact
    quantitative threshold v_{A,B}(infinity) is checked analytically and
    with a dedicated, larger simulation in experiment 3."""
    explosive = power_volatility_isotropic(dim=2, kappa=1.0, eta=1.0, p=2.0, rho=0.8)
    benign = power_volatility_isotropic(dim=2, kappa=1.0, eta=1.0, p=2.0, rho=-0.8)
    kwargs = dict(T=0.5, n_steps=150, n_paths=600, seed=21)
    r_exp = compare_mass_survival(explosive, y0=np.array([1.0, 0.5]), dim=2, dim_w=3, **kwargs)
    r_ben = compare_mass_survival(benign, y0=np.array([1.0, 0.5]), dim=2, dim_w=3, **kwargs)
    assert np.isclose(r_ben.e_s_t, 1.0, atol=0.1)
    assert r_exp.se_s_t > 3 * r_ben.se_s_t


def test_loss_of_mass_curve_is_nondecreasing():
    model = critical_log_cir(c=0.8)  # c > 1/2: strict local martingale regime
    curve = loss_of_mass_curve(
        model, y0=np.array([0.2]), T=1.0, n_steps=150, n_paths=500, seed=31,
        dim=1, dim_w=1, positivity_floor=1e-8,
    )
    m_t = curve["m_t"]
    n_knots = 5
    knots = np.array_split(m_t, n_knots)
    knot_means = np.array([k.mean() for k in knots])
    # Monte Carlo noise lets individual grid points dip slightly; the
    # block-averaged trend over the simulation horizon must not decrease.
    assert np.all(np.diff(knot_means) >= -0.03)
    assert knot_means[-1] > knot_means[0]


def test_compare_mass_survival_agrees_within_tolerance_true_martingale():
    model = cir_2d_model(alpha=0.6, kappa=1.5, eta=1.0, theta_coef=0.4)
    result = compare_mass_survival(
        model, y0=np.array([1.0, 1.0]), dim=2, dim_w=2, positivity_floor=1e-6,
        T=1.0, n_steps=200, n_paths=3000, seed=41,
    )
    assert abs(result.e_s_t - result.survival_prob) < 0.1
