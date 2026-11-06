"""Unit tests for src/martingale_criteria.py against known analytic facts:
the Feller boundary classification of CIR-type diffusions and the exact
threshold of the critical-CIR logarithmic boundary (paper Example 3.2)."""

from __future__ import annotations

import numpy as np

from src.martingale_criteria import (
    condition_T_holds,
    feller_integral_v_AB,
    fichera_drift,
    projection_beta,
    scale_integral_diverges_log_cir,
    v_AB_scalar_cir,
    weighted_distance_martingale,
)


def test_fichera_drift_sign_convention():
    # F = nu * (mu_tilde - 0.5 div a); nu = -1 at a face {y_i = 0}.
    assert np.isclose(fichera_drift(a_ii=1.0, div_a_i=1.0, mu_tilde_i=2.0, nu=-1.0), -1.5)
    assert np.isclose(fichera_drift(a_ii=0.0, div_a_i=0.0, mu_tilde_i=-0.3, nu=-1.0), 0.3)


def test_condition_T_holds_threshold():
    assert condition_T_holds(a_over_rho_minus_two_mu_tilde=0.05, rho=0.1, C=2.0)
    assert not condition_T_holds(a_over_rho_minus_two_mu_tilde=1.0, rho=0.1, C=2.0)


def test_weighted_distance_bessel_threshold():
    # Proposition 5.2: true martingale iff sup c <= (m-2)/2.
    assert weighted_distance_martingale(sup_c=0.4, m=3)
    assert weighted_distance_martingale(sup_c=0.5, m=3)  # equality case included
    assert not weighted_distance_martingale(sup_c=0.6, m=3)


def test_projection_beta_formula():
    assert np.isclose(projection_beta(A_pi=4.0, B_pi=6.0, C_pi=-8 * 0.2), 0.5 - 2 * 0.2)


def test_scale_integral_diverges_log_cir_exact_threshold():
    # Example 3.2: martingale (divergence of the scale integral) iff c <= 1/2.
    assert scale_integral_diverges_log_cir(c=0.3)
    assert scale_integral_diverges_log_cir(c=0.5)  # equality case is a true martingale
    assert not scale_integral_diverges_log_cir(c=0.7)


def test_v_AB_scalar_cir_feller_condition():
    """Classical Feller test for a CIR-type factor: the scale integral near
    0 diverges (0 inaccessible) iff 2*alpha >= eta^2. We do not need an
    automatic infinite/finite flag here: we check that shrinking y0 drives
    the integral to grow without bound in the Feller-satisfied regime and
    to stabilise in the Feller-violated regime."""
    eta, kappa, c = 1.0, 1.0, 0.0

    def growth_ratio(alpha):
        v_loose = v_AB_scalar_cir(alpha, eta, kappa, c, y0=1e-4, y_max=10.0)
        v_tight = v_AB_scalar_cir(alpha, eta, kappa, c, y0=1e-8, y_max=10.0)
        return v_tight / max(v_loose, 1e-12)

    # 2*alpha=2.0 >= eta^2=1.0: Feller condition holds, integral diverges as y0 -> 0.
    assert growth_ratio(alpha=1.0) > 5.0
    # 2*alpha=0.4 < eta^2=1.0: Feller condition fails, integral stays bounded.
    assert growth_ratio(alpha=0.2) < 1.5


def test_feller_integral_v_AB_plateauing_integrand_diverges():
    # A(r)=1, B(r)=b>0: outward push, scale integral grows without a decaying
    # tail, so the Feller test must report divergence (non-explosion).
    v = feller_integral_v_AB(A=lambda r: 1.0, B=lambda r: 0.5, r0=0.1,
                              r_inf_cutoff=200.0, n_points=2000)
    assert v == np.inf
