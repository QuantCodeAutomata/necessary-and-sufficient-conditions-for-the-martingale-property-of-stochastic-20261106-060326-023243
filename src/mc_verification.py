"""Shared Monte Carlo verification routine used by every experiment:
compare the Monte Carlo average of S_T under the original measure with
the survival probability of the Girsanov-shifted diffusion under the
auxiliary measure, as in Lemma 2.1 (paper eq. (1.4)):

    E_y[S_T] = Qtilde_y(tau_D > T).

Custom implementation (paper Lemma 2.1 and Proposition 2.1): this is the
paper's own probabilistic identity, not a library routine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np

from data.generate_synthetic import simulate_diffusion


@dataclass
class MCComparisonResult:
    e_s_t: float
    se_s_t: float
    survival_prob: float
    se_survival: float
    agreement_within_ci: bool
    n_paths: int
    diagnostics: Dict = field(default_factory=dict)


def compare_mass_survival(
    model: Dict,
    y0: np.ndarray,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int,
    dim: int,
    dim_w: int,
    positivity_floor: Optional[float] = None,
    z_score: float = 1.96,
) -> MCComparisonResult:
    """Run the two simulations of Lemma 2.1 and compare.

    (a) Simulate the original diffusion Y under P with drift model['drift']
        and accumulate S_T via model['theta'] along the same Brownian path.
    (b) Simulate the Girsanov-shifted diffusion under Qtilde with drift
        model['shifted_drift'] (independent Brownian increments, since the
        two simulations represent different probability measures) and
        estimate the probability of staying in D up to T.
    """
    y0 = np.asarray(y0, dtype=float)

    sim_p = simulate_diffusion(
        drift=model["drift"], diffusion=model["diffusion"], y0=y0, T=T,
        n_steps=n_steps, n_paths=n_paths, dim=dim, dim_w=dim_w, seed=seed,
        domain_indicator=model.get("domain_indicator"),
        positivity_floor=positivity_floor, theta=model["theta"],
    )
    S_T = np.exp(sim_p.log_S[:, -1])
    e_s_t = float(S_T.mean())
    se_s_t = float(S_T.std(ddof=1) / np.sqrt(n_paths))

    sim_q = simulate_diffusion(
        drift=model["shifted_drift"], diffusion=model["diffusion"], y0=y0, T=T,
        n_steps=n_steps, n_paths=n_paths, dim=dim, dim_w=dim_w, seed=seed + 1,
        domain_indicator=model.get("domain_indicator"),
        positivity_floor=positivity_floor,
    )
    survival = ~sim_q.exited
    survival_prob = float(survival.mean())
    se_survival = float(np.sqrt(survival_prob * (1 - survival_prob) / n_paths))

    combined_se = np.sqrt(se_s_t ** 2 + se_survival ** 2)
    agreement = abs(e_s_t - survival_prob) <= z_score * combined_se + 0.02

    return MCComparisonResult(
        e_s_t=e_s_t, se_s_t=se_s_t, survival_prob=survival_prob,
        se_survival=se_survival, agreement_within_ci=agreement, n_paths=n_paths,
        diagnostics={"exit_fraction": float(sim_q.exited.mean())},
    )


def loss_of_mass_curve(
    model: Dict, y0: np.ndarray, T: float, n_steps: int, n_paths: int, seed: int,
    dim: int, dim_w: int, positivity_floor: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """Loss of mass m(t,y) = 1 - E_y[S_t] along the simulation grid
    (Remark 2.2), used to check monotonicity in t and the Cauchy problem
    partial_t m = Ltilde m qualitatively."""
    y0 = np.asarray(y0, dtype=float)
    sim_p = simulate_diffusion(
        drift=model["drift"], diffusion=model["diffusion"], y0=y0, T=T,
        n_steps=n_steps, n_paths=n_paths, dim=dim, dim_w=dim_w, seed=seed,
        domain_indicator=model.get("domain_indicator"),
        positivity_floor=positivity_floor, theta=model["theta"],
    )
    S_path = np.exp(sim_p.log_S)  # (n_paths, n_steps+1)
    m_t = 1.0 - S_path.mean(axis=0)
    return {"times": sim_p.times, "m_t": m_t}
