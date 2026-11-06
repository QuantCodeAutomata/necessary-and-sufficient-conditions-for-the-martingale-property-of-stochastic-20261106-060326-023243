"""Uniqueness of the valuation PDE in the linear-growth class (Appendix B,
Theorem B.1): when S is a strict local martingale, the forward-contract
payoff phi(s,y) = s admits two distinct solutions u_1 = s and
u_2 = s * E_y[S_T] = s * (1 - m(T,y)); when S is a true martingale the two
coincide.

Custom implementation (paper Theorem B.1 and the Doob transform h = 1+s
preceding it): no researched library implements this PDE-uniqueness
argument.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from src.mc_verification import loss_of_mass_curve


def two_solutions_forward_payoff(
    model: Dict, y0: np.ndarray, T: float, n_steps: int, n_paths: int, seed: int,
    dim: int, dim_w: int, s0: float = 1.0, positivity_floor: float = None,
) -> Dict[str, float]:
    """Compute u_1 = s0 and u_2 = s0 * (1 - m(T, y0)) for the forward
    payoff phi(s, y) = s of Theorem B.1, with m(T, y0) estimated by Monte
    Carlo (Remark 2.2 / Appendix B).

    Returns a dict with u1, u2, loss_of_mass and their relative gap;
    u1 == u2 within Monte Carlo error iff S is a true martingale.
    """
    curve = loss_of_mass_curve(
        model, y0, T, n_steps, n_paths, seed, dim, dim_w, positivity_floor
    )
    m_T = float(curve["m_t"][-1])
    u1 = s0
    u2 = s0 * (1.0 - m_T)
    return {
        "u1": u1,
        "u2": u2,
        "loss_of_mass": m_T,
        "relative_gap": abs(u1 - u2) / max(abs(u1), 1e-8),
        "m_t_curve": curve["m_t"],
        "times": curve["times"],
    }
