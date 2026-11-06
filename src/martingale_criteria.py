"""Analytic martingale criteria of Bayraktar and Xing.

Custom implementation of the paper's own formulas (Sections 3-5): these are
not standard quantitative-finance routines, so none of the researched
libraries (arch, hmmlearn, ruptures, skfolio, pyportfolioopt, copulae,
vectorbt, quantstats, ta, statsmodels) implement them; scipy.integrate is
used only as a generic numerical-quadrature backend, never as a source of
the formulas themselves.
"""

from __future__ import annotations

import numpy as np
from scipy import integrate


def fichera_drift(a_ii: float, div_a_i: float, mu_tilde_i: float, nu: float) -> float:
    """Fichera drift F_b(x) = <b(x) - 0.5 div a(x), nu(x)> at a boundary
    point with outward normal component nu along coordinate i (paper,
    Section 3.1, display preceding Theorem 3.1).

    For a face {y_i = 0} of the orthant the outward normal is nu = -1,
    so F = -(mu_tilde_i - 0.5 * div_a_i).
    """
    return nu * (mu_tilde_i - 0.5 * div_a_i)


def condition_T_holds(a_over_rho_minus_two_mu_tilde: float, rho: float, C: float) -> bool:
    """Condition (T) of the paper (Section 3.2): <a grad rho, grad rho>/rho
    - 2 L~rho <= C rho near the boundary, tested pointwise at distance rho
    from the boundary with a precomputed left-hand side.
    """
    return a_over_rho_minus_two_mu_tilde <= C * rho + 1e-12


def feller_integral_v_AB(A, B, r0: float, r_inf_cutoff: float = 1e4,
                          n_points: int = 4000) -> float:
    """Numerical value of Feller's test integral (paper eq. (4.3))

        v_{A,B}(infinity) = int_{r0}^infinity C(rho)^{-1}
                              int_{r0}^rho C(s)/A(s) ds  d rho,
        C(rho) = exp( int_{r0}^rho B(s) ds ).

    A finite but large cutoff r_inf_cutoff replaces infinity; the
    function reports np.inf when the integrand does not decay (i.e. the
    truncated integral keeps growing with the cutoff), by comparing the
    truncated value at r_inf_cutoff/2 and at r_inf_cutoff.
    """
    s_grid = np.linspace(r0, r_inf_cutoff, n_points)
    b_vals = np.array([B(s) for s in s_grid])
    log_C = integrate.cumulative_trapezoid(b_vals, s_grid, initial=0.0)
    C_vals = np.exp(log_C - log_C.max())  # rescaled for numerical stability (cancels below)

    inv_A = np.array([1.0 / A(s) for s in s_grid])
    inner = integrate.cumulative_trapezoid(C_vals * inv_A, s_grid, initial=0.0)
    integrand = inner / C_vals

    def _partial(upto_idx: int) -> float:
        return float(np.trapz(integrand[: upto_idx + 1], s_grid[: upto_idx + 1]))

    half_idx = n_points // 2
    v_half = _partial(half_idx)
    v_full = _partial(n_points - 1)
    # Divergence diagnostic: if the second half contributes at least as much
    # as the first, the integral has not started converging before the
    # cutoff, so report divergence (true martingale / non-explosion).
    if (v_full - v_half) > 0.05 * max(v_half, 1e-8):
        return np.inf
    return v_full


def scale_density_log_cir(y: np.ndarray, c: float) -> np.ndarray:
    """Scale density p(y) = 1/(y [log(1/y)]^{2c}) of the shifted critical
    CIR factor near the logarithmic boundary singularity (Example 3.2,
    paper Section 3.6)."""
    y = np.asarray(y)
    return 1.0 / (y * np.power(np.log(1.0 / y), 2.0 * c))


def scale_integral_diverges_log_cir(c: float, y_small: float = 1e-6,
                                     y_cut: float = np.exp(-2)) -> bool:
    """Whether int_0 p(y) dy diverges for the critical-CIR logarithmic
    scale density; analytically this holds iff 2c <= 1 (paper Example
    3.2). Verified here by numerical integration on (y_small, y_cut] and
    comparing to the shrinking-y_small asymptotic growth.
    """
    if 2.0 * c <= 1.0:
        return True
    grid = np.geomspace(y_small, y_cut, 4000)
    vals = scale_density_log_cir(grid, c)
    val = np.trapz(vals, grid)
    grid2 = np.geomspace(y_small / 100, y_cut, 4000)
    vals2 = scale_density_log_cir(grid2, c)
    val2 = np.trapz(vals2, grid2)
    # If shrinking y_small by two orders of magnitude barely changes the
    # integral, it converges; a diverging integral keeps growing.
    return (val2 - val) > 0.1 * val


def weighted_distance_martingale(sup_c: float, m: int) -> bool:
    """Proposition 5.2 Bessel threshold: true martingale iff sup c(z) <=
    (m - 2) / 2, where m is the codimension of the singular boundary
    stratum."""
    return sup_c <= (m - 2) / 2.0 + 1e-12


def projection_beta(A_pi: float, B_pi: float, C_pi: float) -> float:
    """Ramasubramanian projection rate beta = (B_pi - A_pi + C_pi)/A_pi
    used in the sufficient (not necessary) non-attainment test of
    Proposition 5.1."""
    return (B_pi - A_pi + C_pi) / A_pi


def v_AB_scalar_cir(alpha: float, eta: float, kappa: float, c: float,
                     y0: float = 1e-8, y_max: float = 60.0) -> float:
    """Feller scale/speed test at 0 for a 1D CIR-type factor with shifted
    drift mu_tilde(y) = alpha - kappa y - eta*c (affine, as in Example 6.3,
    6.5): accessibility of 0 is governed by
        integral_0 exp(-int 2 mu_tilde/a) dy,   a(y) = eta^2 y.
    Returns the value of that scale integral near 0 (finite => accessible,
    infinite => inaccessible, i.e. true martingale for the orthant model).
    """
    def integrand(y):
        # exact antiderivative of -2*mu_tilde/a for mu_tilde affine in y:
        # -2*(alpha - eta*c - kappa*y)/(eta^2*y) = -2*(alpha-eta*c)/(eta^2*y) + 2*kappa/eta^2
        return np.exp(
            -2.0 * (alpha - eta * c) / (eta ** 2) * np.log(y) + 2.0 * kappa / eta ** 2 * y
        )

    grid = np.geomspace(y0, y_max, 20000)
    vals = integrand(grid)
    return float(np.trapz(vals, grid))
