"""Wishart process simulation and Fichera-drift boundary criterion
(paper Section 7.2, Proposition 7.2): dSigma_t = (Omega Omega^T + M Sigma_t
+ Sigma_t M^T) dt + sqrt(Sigma_t) dB_t Q + Q^T dB_t^T sqrt(Sigma_t), on the
cone of positive semi-definite n x n matrices.

Custom implementation (paper eq. (7.2) and its Fichera-drift computation
at a rank-(n-1) boundary point): no researched library (arch, hmmlearn,
skfolio, pyportfolioopt) simulates matrix-valued Wishart diffusions with
boundary-exit tracking.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import sqrtm


def simulate_wishart(
    n: int,
    M: np.ndarray,
    Q: np.ndarray,
    Omega: np.ndarray,
    Sigma0: np.ndarray,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int,
    eps_floor: float = 1e-6,
):
    """Euler-Maruyama simulation of the Wishart SDE on S_n^+, tracking the
    minimum eigenvalue path to detect boundary exit (loss of positive
    definiteness, i.e. the matrix analogue of {y_i = 0})."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)

    Sigma = np.tile(Sigma0, (n_paths, 1, 1)).astype(float)
    min_eig_path = np.zeros((n_paths, n_steps + 1))
    min_eig_path[:, 0] = np.linalg.eigvalsh(Sigma0).min()
    exited = np.zeros(n_paths, dtype=bool)
    exit_step = np.full(n_paths, -1, dtype=int)

    OmOmT = Omega @ Omega.T

    for step in range(n_steps):
        dB = rng.standard_normal((n_paths, n, n)) * sqrt_dt
        # symmetrize sqrt(Sigma) @ dB @ Q + Q^T @ dB^T @ sqrt(Sigma)
        eigval, eigvec = np.linalg.eigh(Sigma)
        eigval_clipped = np.maximum(eigval, eps_floor)
        sqrt_Sigma = np.einsum("pij,pj,pkj->pik", eigvec, np.sqrt(eigval_clipped), eigvec)

        drift = OmOmT[None, :, :] + np.einsum("ij,pjk->pik", M, Sigma) + np.einsum(
            "pij,jk->pik", Sigma, M.T
        )
        diffusion_term = np.einsum("pij,pjk,kl->pil", sqrt_Sigma, dB, Q)
        diffusion_term = diffusion_term + np.transpose(diffusion_term, (0, 2, 1))

        Sigma = Sigma + drift * dt + diffusion_term
        Sigma = 0.5 * (Sigma + np.transpose(Sigma, (0, 2, 1)))  # enforce symmetry

        eig_now = np.linalg.eigvalsh(Sigma)
        min_eig = eig_now.min(axis=1)
        min_eig_path[:, step + 1] = min_eig

        newly_exited = (~exited) & (min_eig <= eps_floor * 2)
        exited |= newly_exited
        exit_step[newly_exited] = step + 1

    return dict(Sigma=Sigma, min_eig_path=min_eig_path, exited=exited, exit_step=exit_step)


def fichera_rank_boundary(n: int, Omega: np.ndarray, Q: np.ndarray) -> float:
    """F = -(v^T Omega Omega^T v - (n+1) v^T Q^T Q v) at a rank-(n-1)
    boundary point with kernel vector v (take v = e_n by symmetry when
    Omega Omega^T and Q^T Q are both proportional to the identity, as in
    the isotropic test case used here); returns the scalar Fichera drift
    (paper Proposition 7.2)."""
    v = np.zeros(n)
    v[-1] = 1.0
    quad_omega = v @ (Omega @ Omega.T) @ v
    quad_q = v @ (Q.T @ Q) @ v
    return -(quad_omega - (n + 1) * quad_q)
