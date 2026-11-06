"""Synthetic sample-path generator: Euler-Maruyama simulation of the
multi-dimensional diffusions studied in Bayraktar and Xing, "Necessary
and Sufficient Conditions for the Martingale Property of Stochastic
Exponentials Driven by Multi-Dimensional Diffusions".

No experiment in this repository uses external market data (every
data_requirements field in the experiment specification states "No
external data"): the only input is simulated Brownian motion. This
module owns the random-number generation; src/models.py and
src/stochastic_exponential.py hold the deterministic formulas applied
to the simulated paths.

Custom implementation (paper implementation notes: Euler-Maruyama with a
full-truncation floor for square-root-type diffusions, following Lord,
Koekkoek and Van Dijk's bias-reduced scheme for CIR factors): no library
inspected via Context7 (sdeint is not indexed; the project's SDE
toolbox -- arch, hmmlearn, vectorbt -- does not implement general
multi-dimensional Euler-Maruyama simulation with domain-exit tracking).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

DriftFn = Callable[[np.ndarray, float], np.ndarray]
DiffusionFn = Callable[[np.ndarray, float], np.ndarray]
ThetaFn = Callable[[np.ndarray, float], np.ndarray]
DomainFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class SimulationResult:
    times: np.ndarray            # (n_steps + 1,)
    Y: np.ndarray                 # (n_paths, n_steps + 1, dim)
    exited: np.ndarray            # (n_paths,) bool, True if the path left the domain by T
    exit_step: np.ndarray         # (n_paths,) int, first step index of exit, -1 if never
    log_S: Optional[np.ndarray] = None  # (n_paths, n_steps + 1) when theta is given


def _full_truncation(y: np.ndarray, floor: float) -> np.ndarray:
    """Full-truncation floor max(y, floor) applied to square-root-type
    diffusion coefficients, preventing negative arguments under the
    square root (Lord-Koekkoek-Van Dijk full truncation scheme)."""
    return np.maximum(y, floor)


def simulate_diffusion(
    drift: DriftFn,
    diffusion: DiffusionFn,
    y0: np.ndarray,
    T: float,
    n_steps: int,
    n_paths: int,
    dim: int,
    dim_w: int,
    seed: int,
    domain_indicator: Optional[DomainFn] = None,
    positivity_floor: Optional[float] = None,
    theta: Optional[ThetaFn] = None,
    antithetic: bool = True,
) -> SimulationResult:
    """Simulate n_paths trajectories of dY = drift(Y,t) dt + diffusion(Y,t) dW
    on [0, T] with an Euler-Maruyama scheme, optionally accumulating the
    stochastic exponential log S_t = int theta(Y) . dW - 0.5 int |theta(Y)|^2 dt
    along the same Brownian increments, and tracking the first exit from
    a domain D given by domain_indicator(y) -> bool array (True = inside D).

    Parameters
    ----------
    drift : callable(y, t) -> array (n_paths, dim)
    diffusion : callable(y, t) -> array (n_paths, dim, dim_w)
    y0 : array (dim,) starting point, identical for every path
    domain_indicator : callable(y) -> bool array (n_paths,), True while inside D
    positivity_floor : if given, apply full truncation to keep Y >= floor
        before evaluating drift/diffusion/theta (used for CIR-type factors)
    theta : callable(y, t) -> array (n_paths, dim_w), optional

    Returns
    -------
    SimulationResult
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)

    if antithetic:
        half = (n_paths + 1) // 2
        base = rng.standard_normal((half, n_steps, dim_w))
        Z = np.concatenate([base, -base], axis=0)[:n_paths]
    else:
        Z = rng.standard_normal((n_paths, n_steps, dim_w))
    dW = Z * sqrt_dt

    Y = np.zeros((n_paths, n_steps + 1, dim))
    Y[:, 0, :] = y0
    exited = np.zeros(n_paths, dtype=bool)
    exit_step = np.full(n_paths, -1, dtype=int)
    log_S = np.zeros((n_paths, n_steps + 1)) if theta is not None else None

    active = np.ones(n_paths, dtype=bool)
    for step in range(n_steps):
        t = step * dt
        y_cur = Y[:, step, :]
        y_eval = _full_truncation(y_cur, positivity_floor) if positivity_floor is not None else y_cur

        mu = drift(y_eval, t)
        sig = diffusion(y_eval, t)  # (n_paths, dim, dim_w)
        dW_t = dW[:, step, :]

        increment = mu * dt + np.einsum("pdw,pw->pd", sig, dW_t)
        y_raw = y_cur + increment

        # Domain-exit is checked on the raw (unfloored) step: the positivity
        # floor is only a numerical device to keep next-step coefficients
        # well defined and must not mask an exit from the true domain D.
        if domain_indicator is not None:
            still_inside = domain_indicator(y_raw)
            newly_exited = active & (~still_inside)
            exited |= newly_exited
            exit_step[newly_exited] = step + 1
            active &= still_inside

        y_next = np.maximum(y_raw, positivity_floor) if positivity_floor is not None else y_raw
        Y[:, step + 1, :] = y_next

        if theta is not None:
            theta_val = theta(y_eval, t)
            inc = np.einsum("pw,pw->p", theta_val, dW_t) - 0.5 * dt * np.einsum(
                "pw,pw->p", theta_val, theta_val
            )
            log_S[:, step + 1] = log_S[:, step] + inc

    times = np.linspace(0.0, T, n_steps + 1)
    return SimulationResult(times=times, Y=Y, exited=exited, exit_step=exit_step, log_S=log_S)
