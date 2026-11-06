"""Unit tests for the shared Euler-Maruyama engine in data/generate_synthetic.py.

These are small, fast, synthetic-input tests (no external data, consistent
with the paper's "No external data" requirement for every experiment).
"""

from __future__ import annotations

import numpy as np

from data.generate_synthetic import simulate_diffusion


def test_zero_diffusion_matches_ode_solution():
    """With diffusion == 0, the Euler-Maruyama scheme reduces to the
    explicit Euler scheme for dY/dt = -kappa Y, whose exact solution is
    Y_T = y0 * exp(-kappa T)."""
    kappa = 1.3
    y0 = np.array([2.0])

    def drift(y, t):
        return -kappa * y

    def diffusion(y, t):
        return np.zeros((y.shape[0], 1, 1))

    result = simulate_diffusion(
        drift=drift, diffusion=diffusion, y0=y0, T=1.0, n_steps=2000,
        n_paths=4, dim=1, dim_w=1, seed=0,
    )
    analytic = y0[0] * np.exp(-kappa * 1.0)
    assert np.allclose(result.Y[:, -1, 0], analytic, atol=1e-2)


def test_domain_indicator_tracks_exit():
    """A process started safely inside a half-line domain with no exit
    condition never exits; one that drifts out must be flagged exited."""
    def drift_stay(y, t):
        return np.zeros_like(y)

    def diffusion_small(y, t):
        n = y.shape[0]
        return np.full((n, 1, 1), 1e-6)

    def domain(y):
        return y[:, 0] > 0.0

    result = simulate_diffusion(
        drift=drift_stay, diffusion=diffusion_small, y0=np.array([1.0]),
        T=1.0, n_steps=200, n_paths=50, dim=1, dim_w=1, seed=1,
        domain_indicator=domain,
    )
    assert not result.exited.any()

    def drift_down(y, t):
        return np.full_like(y, -5.0)

    result2 = simulate_diffusion(
        drift=drift_down, diffusion=diffusion_small, y0=np.array([1.0]),
        T=1.0, n_steps=200, n_paths=50, dim=1, dim_w=1, seed=2,
        domain_indicator=domain,
    )
    assert result2.exited.all()
    assert (result2.exit_step[result2.exited] > 0).all()


def test_stochastic_exponential_accumulation_zero_theta():
    """theta == 0 gives S_t == 1 identically (log S_t == 0), independent of
    the driving noise (paper eq. (1.3) with theta = 0)."""
    def drift(y, t):
        return np.zeros_like(y)

    def diffusion(y, t):
        n = y.shape[0]
        sig = np.zeros((n, 1, 1))
        sig[:, 0, 0] = 1.0
        return sig

    def theta_zero(y, t):
        return np.zeros((y.shape[0], 1))

    result = simulate_diffusion(
        drift=drift, diffusion=diffusion, y0=np.array([0.0]), T=1.0,
        n_steps=100, n_paths=10, dim=1, dim_w=1, seed=3, theta=theta_zero,
    )
    assert np.allclose(result.log_S, 0.0)


def test_positivity_floor_keeps_process_nonnegative():
    """A CIR-type square-root diffusion with the full-truncation floor
    never goes below the floor, preserving positivity of the factor."""
    alpha, kappa, eta = 0.1, 1.0, 2.0  # Feller condition violated on purpose

    def drift(y, t):
        return alpha - kappa * y

    def diffusion(y, t):
        n = y.shape[0]
        sig = np.zeros((n, 1, 1))
        sig[:, 0, 0] = eta * np.sqrt(np.maximum(y[:, 0], 0.0))
        return sig

    result = simulate_diffusion(
        drift=drift, diffusion=diffusion, y0=np.array([0.5]), T=1.0,
        n_steps=500, n_paths=200, dim=1, dim_w=1, seed=4,
        positivity_floor=1e-6,
    )
    assert result.Y.min() >= 1e-6 - 1e-12


def test_antithetic_paths_are_mirrored():
    """With antithetic=True (default), the first half of the Brownian
    increments used must be the negative of the second half, which is
    checked indirectly through perfectly anti-correlated pure-diffusion
    paths around a zero drift."""
    def drift(y, t):
        return np.zeros_like(y)

    def diffusion(y, t):
        n = y.shape[0]
        sig = np.zeros((n, 1, 1))
        sig[:, 0, 0] = 1.0
        return sig

    result = simulate_diffusion(
        drift=drift, diffusion=diffusion, y0=np.array([0.0]), T=1.0,
        n_steps=50, n_paths=10, dim=1, dim_w=1, seed=5, antithetic=True,
    )
    half = 5
    assert np.allclose(result.Y[:half, -1, 0], -result.Y[half:2 * half, -1, 0])
