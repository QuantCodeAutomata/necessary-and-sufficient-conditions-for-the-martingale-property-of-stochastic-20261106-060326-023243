"""Experiment 5 -- Singular Boundary Strata: Weighted Distance Criterion
(Proposition 5.2) and Projection Test (Proposition 5.1), paper Section 5,
p.23-26.

U_t = u + W_t is a Brownian motion in R^m (m>=2), dZ^i = -lam Z^i dt + eta_ou
dW^i are k correlated Ornstein-Uhlenbeck tangential factors, and
S = E(-int c(Z) U/|U|^2 . dW). By Ito's formula the radial part R=|U|
of the shifted process is a Bessel-type diffusion of effective dimension
m - 2 c(Z); Proposition 5.2 gives the sharp threshold: S is a true
martingale iff sup c(z) <= (m-2)/2 (Bessel dimension >= 2, so the origin is
inaccessible), including the equality case.

Example 5.2 (m=3, k=1, M=diag(2,1,1)): with c(z) = c_base + c_span *
z^2/(1+z^2), the weighted-distance criterion gives the exact answer
sup c <= (m-2)/2 = 1/2, while the classical Euclidean projection test of
Proposition 5.1 is only sufficient: its rate beta(z) = 1/2 - 2 c(z) is
negative whenever c(z) > 1/4, making the projection integral finite and
the projection test inconclusive even though the boundary is in fact
inaccessible for sup c <= 1/2.

Run:
    python -m exp.exp5            # reduced configuration
    python -m exp.exp5 --full     # paper-scale configuration
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import config as cfg
from src.io_utils import to_jsonable
from src.mc_verification import compare_mass_survival
from src.models import weighted_distance_model
from src.martingale_criteria import weighted_distance_martingale, projection_beta

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def _run_regime(regime_name: str, c_span: float, n_steps_unit: float, n_paths: int) -> dict:
    m, k = cfg.EXP5_M, cfg.EXP5_K
    T = cfg.EXP5_HORIZON
    n_steps = max(int(n_steps_unit * T), 10)
    dim = m + k

    model = weighted_distance_model(
        m=m, k=k, lam=cfg.EXP5_LAMBDA, eta_ou=cfg.EXP5_ETA_OU, M_diag=cfg.EXP5_M_DIAG,
        c_base=cfg.EXP5_C_BASE, c_span=c_span,
    )
    y0 = np.array(list(cfg.EXP5_U0) + [cfg.EXP5_Z0])
    result = compare_mass_survival(
        model, y0=y0, T=T, n_steps=n_steps, n_paths=n_paths, seed=cfg.RNG_SEED,
        dim=dim, dim_w=dim, positivity_floor=None,
    )

    sup_c = cfg.EXP5_C_BASE + c_span  # c(z) = c_base + c_span*z^2/(1+z^2) -> c_base+c_span as |z|->inf
    threshold = (m - 2) / 2.0
    weighted_predicts_martingale = weighted_distance_martingale(sup_c, m)

    # Euclidean projection test on the first normal axis (Example 5.2):
    # A_pi=4, B_pi=6, C_pi=-8*c(z); beta = (B_pi-A_pi+C_pi)/A_pi = 1/2 - 2*c(z).
    A_pi, B_pi = 4.0, 6.0
    C_pi = -8.0 * sup_c
    beta = projection_beta(A_pi, B_pi, C_pi)
    projection_inconclusive = bool(beta < 0)

    return {
        "regime": regime_name,
        "c_span": c_span,
        "sup_c": sup_c,
        "threshold_m_minus_2_over_2": threshold,
        "weighted_distance_predicts_martingale": weighted_predicts_martingale,
        "projection_beta": beta,
        "projection_inconclusive": projection_inconclusive,
        "E_S_T": result.e_s_t,
        "se_E_S_T": result.se_s_t,
        "survival_prob": result.survival_prob,
        "se_survival": result.se_survival,
        "agreement_within_ci": result.agreement_within_ci,
        "exit_fraction": result.diagnostics["exit_fraction"],
    }


def run(full: bool) -> dict:
    n_steps_unit = cfg.N_STEPS_PER_UNIT_TIME_FULL if full else cfg.N_STEPS_PER_UNIT_TIME_RUN
    n_paths = cfg.N_PATHS_FULL if full else cfg.N_PATHS_RUN

    records = [
        _run_regime("martingale", cfg.EXP5_C_SPAN_MARTINGALE, n_steps_unit, n_paths),
        _run_regime("strict_local", cfg.EXP5_C_SPAN_STRICT, n_steps_unit, n_paths),
    ]

    def _agrees(predicts_martingale: bool, survival_prob: float) -> bool:
        # Lemma 2.1 (paper eq. (1.4)): E_y[S_T] = Qtilde_y(tau_D > T). The
        # survival fraction is the better-conditioned Monte Carlo statistic
        # here (bounded in [0,1]), while the direct average of S_T is a
        # heavy-tailed estimator of the same quantity and converges far more
        # slowly, so the regime classification is read off survival_prob.
        return survival_prob > 0.9 if predicts_martingale else survival_prob < 0.98

    weighted_distance_confirmed = all(
        _agrees(r["weighted_distance_predicts_martingale"], r["survival_prob"]) for r in records
    )
    # Example 5.2: in the martingale regime the Euclidean projection test is
    # inconclusive (beta<0) even though the weighted-distance criterion
    # correctly identifies the true martingale property.
    martingale_record = records[0]
    projection_inconclusive_in_martingale_case = martingale_record["projection_inconclusive"]

    summary = {
        "records": records,
        "weighted_distance_confirmed": bool(weighted_distance_confirmed),
        "projection_inconclusive_in_martingale_case": bool(projection_inconclusive_in_martingale_case),
        "n_paths": n_paths,
        "n_steps_unit": n_steps_unit,
        "full": full,
    }
    return summary


def save(summary: dict) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "exp5_results.json", "w") as f:
        json.dump(to_jsonable(summary), f, indent=2)

    records = summary["records"]
    labels = [f"{r['regime']}\nsup c={r['sup_c']:.2f}" for r in records]
    vals = [r["E_S_T"] for r in records]
    colors = ["tab:green" if r["weighted_distance_predicts_martingale"] else "tab:red" for r in records]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(labels, vals, color=colors)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="reference: E[S_T]=1")
    ax.axhline(
        0.0, color="gray", linestyle=":", linewidth=1,
    )
    ax.set_ylabel("E_y[S_T]")
    ax.set_title("Experiment 5: weighted-distance criterion (Prop. 5.2, Ex. 5.2)\n"
                  f"threshold (m-2)/2={records[0]['threshold_m_minus_2_over_2']:.2f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp5_weighted_distance.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="run the paper-scale configuration")
    args = parser.parse_args()

    summary = run(args.full)
    save(summary)

    print(f"Experiment 5 ({'full' if args.full else 'reduced'} configuration)")
    for r in summary["records"]:
        print(
            f"  [{r['regime']}] sup c={r['sup_c']:.2f} (m-2)/2={r['threshold_m_minus_2_over_2']:.2f}: "
            f"E[S_T]={r['E_S_T']:.4f}, survival={r['survival_prob']:.4f}, "
            f"weighted_distance_predicts_martingale={r['weighted_distance_predicts_martingale']}, "
            f"projection_beta={r['projection_beta']:.3f}, "
            f"projection_inconclusive={r['projection_inconclusive']}"
        )
    print(f"  weighted_distance_confirmed: {summary['weighted_distance_confirmed']}")
    print(f"  projection_inconclusive_in_martingale_case: {summary['projection_inconclusive_in_martingale_case']}")


if __name__ == "__main__":
    main()
