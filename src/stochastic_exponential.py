"""Discretisation of the stochastic exponential S = E(int theta(Y)-dW).

Custom implementation (paper Section 1.1 / eq. (1.3)-(1.4)): no library in
the available quantitative-library set (arch, hmmlearn, ruptures, skfolio,
pyportfolioopt, copulae, vectorbt, quantstats, ta) implements pathwise
simulation of a stochastic exponential driven by a general diffusion; this
is a direct Euler discretisation of the defining SDE below.

The exact (Stratonovich-free) Euler-Maruyama discretisation of
    dS_t = S_t theta(Y_t) . dW_t,  S_0 = 1
is
    S_{t+dt} = S_t * exp( theta(Y_t) . dW_t - 0.5 |theta(Y_t)|^2 dt ).
Working with log S avoids overflow and keeps S >= 0 exactly.
"""

from __future__ import annotations

import numpy as np


def log_increment(theta_val: np.ndarray, dW: np.ndarray, dt: float) -> np.ndarray:
    """One Euler step of log S_t for the stochastic exponential.

    Parameters
    ----------
    theta_val : array (n_paths, dim_w)
        theta(Y_t) evaluated along each path at the current time step.
    dW : array (n_paths, dim_w)
        Brownian increments driving the asset over [t, t+dt], the same
        increments used to advance the factor process Y (so that the
        asset and factor stay driven by the same noise, as in the model
        dY = mu(Y) dt + sigma(Y) dW, dS = S theta(Y) . dW).
    dt : float
        Time step.

    Returns
    -------
    array (n_paths,)
        Increment of log S_t over the step.
    """
    inner = np.einsum("pw,pw->p", theta_val, dW)
    quad = np.einsum("pw,pw->p", theta_val, theta_val) * dt
    return inner - 0.5 * quad


def accumulate_log_S(theta_path: np.ndarray, dW: np.ndarray, dt: float) -> np.ndarray:
    """Accumulate log S_t along full paths.

    Parameters
    ----------
    theta_path : array (n_paths, n_steps, dim_w)
        theta(Y_t) at every step along every path.
    dW : array (n_paths, n_steps, dim_w)
        Brownian increments used to advance Y at every step.
    dt : float

    Returns
    -------
    array (n_paths, n_steps + 1)
        log S_t at every grid point, starting at log S_0 = 0.
    """
    n_paths, n_steps, _ = theta_path.shape
    increments = np.einsum("psw,psw->ps", theta_path, dW) - 0.5 * dt * np.einsum(
        "psw,psw->ps", theta_path, theta_path
    )
    log_S = np.zeros((n_paths, n_steps + 1))
    log_S[:, 1:] = np.cumsum(increments, axis=1)
    return log_S
