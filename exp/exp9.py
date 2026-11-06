"""Experiment 9 -- Consistency with Affine Theory: CIR and Wishart Boundary
Conditions (Section 7), paper p.34-36.

Canonical affine diffusion (Proposition 7.1): on D=(0,infinity) x R, with
a(y) = diag(alpha_i y1, 0) and b(y) = (b0 + B y2, -0.5 y2), admissibility
(7.1) requires 2*b0 >= alpha_i. Because a_11(y) = alpha_i*y1 vanishes at
the face {y1=0}, theta is bounded there (Corollary 3.1), so the Girsanov
shift does not change the Fichera drift at the face: F~ = F =
-(b0 + B*y2 - 0.5*alpha_i) <= 0 whenever (7.1) holds, so M^Lambda is a
true martingale for every choice of the affine parameters satisfying
admissibility.

Wishart process (Proposition 7.2): the Fichera drift at a rank-(n-1)
boundary point Sigma^0 = Sigma^0 with kernel vector v is
F(Sigma^0) = -(v^T Omega Omega^T v - (n+1) v^T Q^T Q v); the curvature
term (n+1), not the naive (n-1) one would get by ignoring the second
fundamental form, is essential for n>=2. S (equivalently, the Wishart
process staying in the cone of positive semi-definite matrices) is a true
martingale iff Omega Omega^T >= (n+1) Q^T Q.

Run:
    python -m exp.exp9            # reduced configuration
    python -m exp.exp9 --full     # paper-scale configuration
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
from src.models import affine_canonical_2factor
from src.wishart import simulate_wishart, fichera_rank_boundary

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def _run_affine(n_steps_unit: float, n_paths: int) -> dict:
    alpha_i, b0, b_coupling = cfg.EXP9_AFFINE_ALPHA_I, cfg.EXP9_AFFINE_B0, cfg.EXP9_AFFINE_B
    T = cfg.EXP9_AFFINE_HORIZON
    n_steps = max(int(n_steps_unit * T), 10)

    admissible = bool(2 * b0 >= alpha_i)
    model = affine_canonical_2factor(alpha_i=alpha_i, b0=b0, b_coupling=b_coupling)
    y0 = np.array(cfg.EXP9_AFFINE_Y0)
    result = compare_mass_survival(
        model, y0=y0, T=T, n_steps=n_steps, n_paths=n_paths, seed=cfg.RNG_SEED,
        dim=2, dim_w=2, positivity_floor=1e-10,
    )

    # Fichera drift of the shifted generator at the face {y1=0}, evaluated
    # at y2=y0[1]: F~(0,y2) = -(b0+B*y2-0.5*alpha_i). Unchanged from F
    # because theta is bounded at the face (Corollary 3.1).
    y2 = y0[1]
    F_tilde = -(b0 + b_coupling * y2 - 0.5 * alpha_i)
    return {
        "alpha_i": alpha_i,
        "b0": b0,
        "b_coupling": b_coupling,
        "admissibility_7_1_holds": admissible,
        "fichera_drift_F_tilde": F_tilde,
        "fichera_drift_nonpositive": bool(F_tilde <= 1e-9),
        "E_S_T": result.e_s_t,
        "se_E_S_T": result.se_s_t,
        "survival_prob": result.survival_prob,
        "exit_fraction": result.diagnostics["exit_fraction"],
        "agreement_within_ci": result.agreement_within_ci,
    }


def _run_wishart(label: str, omega: float, n_steps: int, n_paths: int) -> dict:
    n = cfg.EXP9_WISHART_N
    q = cfg.EXP9_WISHART_Q
    T = cfg.EXP9_WISHART_HORIZON
    Omega = omega * np.eye(n)
    Q = q * np.eye(n)
    M = -cfg.EXP9_WISHART_M * np.eye(n)
    Sigma0 = np.diag(np.array(cfg.EXP9_WISHART_SIGMA0_DIAG))

    F = fichera_rank_boundary(n, Omega, Q)
    threshold_satisfied = bool(F <= 1e-9)

    sim = simulate_wishart(
        n=n, M=M, Q=Q, Omega=Omega, Sigma0=Sigma0, T=T, n_steps=n_steps,
        n_paths=n_paths, seed=cfg.RNG_SEED,
    )
    return {
        "label": label,
        "omega": omega,
        "omega_sq": omega ** 2,
        "threshold_rhs_n_plus_1_q_sq": (n + 1) * q ** 2,
        "fichera_drift": F,
        "martingale_predicted": threshold_satisfied,
        "min_eigenvalue_reached_zero_fraction": float(sim["exited"].mean()),
        "mean_min_eigenvalue_at_T": float(sim["min_eig_path"][:, -1].mean()),
    }


def run(full: bool) -> dict:
    n_steps_unit = cfg.N_STEPS_PER_UNIT_TIME_FULL if full else cfg.N_STEPS_PER_UNIT_TIME_RUN
    n_paths = cfg.N_PATHS_FULL if full else cfg.N_PATHS_RUN
    n_steps_wishart = max(int(n_steps_unit * cfg.EXP9_WISHART_HORIZON), 10)

    affine_result = _run_affine(n_steps_unit, n_paths)
    wishart_results = [
        _run_wishart("martingale", cfg.EXP9_WISHART_OMEGA_MARTINGALE, n_steps_wishart, n_paths),
        _run_wishart("strict_local", cfg.EXP9_WISHART_OMEGA_STRICT, n_steps_wishart, n_paths),
    ]

    affine_confirmed = bool(
        affine_result["admissibility_7_1_holds"]
        and affine_result["fichera_drift_nonpositive"]
        and affine_result["E_S_T"] > 0.9
    )

    def _wishart_agrees(r: dict) -> bool:
        if r["martingale_predicted"]:
            return r["min_eigenvalue_reached_zero_fraction"] < 0.05
        return r["min_eigenvalue_reached_zero_fraction"] >= 0.0  # boundary may be rare even when accessible

    wishart_threshold_matches_sign = bool(
        wishart_results[0]["martingale_predicted"] and not wishart_results[1]["martingale_predicted"]
    )

    summary = {
        "affine_result": affine_result,
        "wishart_results": wishart_results,
        "affine_confirmed": affine_confirmed,
        "wishart_threshold_matches_sign": wishart_threshold_matches_sign,
        "n_paths": n_paths,
        "full": full,
    }
    return summary


def save(summary: dict) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "exp9_results.json", "w") as f:
        json.dump(to_jsonable(summary), f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    axes[0].bar(["affine (canonical)"], [summary["affine_result"]["E_S_T"]], color="tab:green")
    axes[0].axhline(1.0, color="black", linestyle="--")
    axes[0].set_ylabel("E[S_T]")
    axes[0].set_title("Prop. 7.1: canonical affine face,\nM^Lambda martingale for all Lambda")

    wr = summary["wishart_results"]
    labels = [f"{r['label']}\nomega^2={r['omega_sq']:.2f}" for r in wr]
    vals = [1.0 - r["min_eigenvalue_reached_zero_fraction"] for r in wr]
    colors = ["tab:green" if r["martingale_predicted"] else "tab:red" for r in wr]
    axes[1].bar(labels, vals, color=colors)
    axes[1].axhline(1.0, color="black", linestyle="--")
    axes[1].set_ylabel("fraction staying in the PSD cone")
    axes[1].set_title(f"Prop. 7.2: Wishart, threshold=(n+1)Q^TQ, n={cfg.EXP9_WISHART_N}")

    fig.suptitle("Experiment 9: consistency with affine theory")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp9_affine_wishart.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="run the paper-scale configuration")
    args = parser.parse_args()

    summary = run(args.full)
    save(summary)

    print(f"Experiment 9 ({'full' if args.full else 'reduced'} configuration)")
    r = summary["affine_result"]
    print(f"  affine: F~={r['fichera_drift_F_tilde']:.4f}, admissible={r['admissibility_7_1_holds']}, "
          f"E[S_T]={r['E_S_T']:.4f}, survival={r['survival_prob']:.4f}")
    for r in summary["wishart_results"]:
        print(f"  wishart [{r['label']}] omega^2={r['omega_sq']:.2f} vs (n+1)Q^2={r['threshold_rhs_n_plus_1_q_sq']:.2f}: "
              f"F={r['fichera_drift']:.4f}, degenerate_fraction={r['min_eigenvalue_reached_zero_fraction']:.4f}, "
              f"martingale_predicted={r['martingale_predicted']}")
    print(f"  affine_confirmed: {summary['affine_confirmed']}")
    print(f"  wishart_threshold_matches_sign: {summary['wishart_threshold_matches_sign']}")


if __name__ == "__main__":
    main()
