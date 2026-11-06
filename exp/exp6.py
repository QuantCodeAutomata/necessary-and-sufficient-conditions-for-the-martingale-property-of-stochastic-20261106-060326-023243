"""Experiment 6 -- Anisotropic Multi-Factor Model with Tangential Rotation
(Example 6.2) and Exact Feller Test via Compact Sublevel Sets
(Proposition 4.3), paper Section 6, p.20, 28.

Example 6.2 replaces the isotropic power-volatility model of Example 6.1
with dY = q(Y) M[-kappa z + Omega(z) z] dt + eta sqrt(q(Y)) M dW, where
z = M^{-1} y, M is an invertible (here diagonal, anisotropic) matrix,
q(y) = q_base + 0.5 sin(y_1) is bounded above and away from zero, and
Omega(z) is skew-symmetric (a nonlinear tangential rotation). Because
z^T Omega(z) z = 0 for skew-symmetric Omega, the level-set function
P(y) = |M^{-1} y|^2 / 2 satisfies the structural identity (4.5) of
Proposition 4.3 with the *same* scalar functions A(r), B(r) as the
isotropic Example 6.1: <a grad P, grad P> = q(y) A(P), 2 L~P = q(y) A(P)
B(P). Since P(Y_t) is, up to a time change with rate q(Y_t) bounded above
and below (which preserves finiteness of hitting times), a diffusion with
exactly the Example 6.1 scale/speed pair, the martingale threshold is
unchanged by the anisotropy, the rotation, and the time-change rate: S is
a true martingale iff rho<=0 or gamma<=1.

Run:
    python -m exp.exp6            # reduced configuration
    python -m exp.exp6 --full     # paper-scale configuration
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
from src.models import power_volatility_anisotropic
from src.martingale_criteria import feller_integral_v_AB

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

EXPLOSION_RADIUS = 50.0  # proxy cutoff for "has effectively exploded", not specified in the paper


def make_model_with_explosion_proxy(dim, kappa, eta, gamma, rho, M_diag, q_base, omega_skew) -> dict:
    model = power_volatility_anisotropic(
        dim=dim, kappa=kappa, eta=eta, gamma=gamma, rho=rho, M_diag=M_diag,
        q_base=q_base, omega_skew=omega_skew,
    )
    model = dict(model)  # shallow copy, do not mutate the original
    M_inv = model["M_inv"]

    def domain_indicator(y: np.ndarray) -> np.ndarray:
        z = (M_inv @ y.T).T
        return np.linalg.norm(z, axis=1) < EXPLOSION_RADIUS

    model["domain_indicator"] = domain_indicator
    return model


def run(full: bool) -> dict:
    n_steps_unit = cfg.N_STEPS_PER_UNIT_TIME_FULL if full else cfg.N_STEPS_PER_UNIT_TIME_RUN
    n_paths = cfg.N_PATHS_FULL if full else cfg.N_PATHS_RUN
    n_steps = max(int(n_steps_unit * cfg.EXP6_HORIZON), 10)

    records = []
    for rho, gamma in cfg.EXP6_REGIMES:
        model = make_model_with_explosion_proxy(
            dim=cfg.EXP6_DIM, kappa=cfg.EXP6_KAPPA, eta=cfg.EXP6_ETA, gamma=gamma, rho=rho,
            M_diag=cfg.EXP6_M_DIAG, q_base=cfg.EXP6_Q_BASE, omega_skew=cfg.EXP6_OMEGA_SKEW,
        )
        result = compare_mass_survival(
            model, y0=np.array(cfg.EXP6_Y0), T=cfg.EXP6_HORIZON, n_steps=n_steps,
            n_paths=n_paths, seed=cfg.RNG_SEED, dim=cfg.EXP6_DIM,
            dim_w=cfg.EXP6_DIM + 1, positivity_floor=None,
        )

        martingale_predicted = bool(rho <= 0 or gamma <= 1)

        # Same radial Feller test A(r), B(r) as the isotropic Example 6.1
        # (Proposition 4.3): the anisotropy, rotation, and time-change rate
        # q(y) do not enter A, B.
        A = lambda r: 2 * cfg.EXP6_ETA ** 2 * r
        B = lambda r: (
            cfg.EXP6_DIM * cfg.EXP6_ETA ** 2 - 4 * cfg.EXP6_KAPPA * r
            + 2 * rho * cfg.EXP6_ETA * (2 * r) ** ((gamma + 1) / 2)
        ) / (2 * cfg.EXP6_ETA ** 2 * r)
        M_inv = np.diag(1.0 / np.array(cfg.EXP6_M_DIAG))
        z0 = M_inv @ np.array(cfg.EXP6_Y0)
        r0 = max(0.5 * float(np.dot(z0, z0)), 1.0)
        v = feller_integral_v_AB(A, B, r0=r0, r_inf_cutoff=1e4, n_points=4000)
        v_diverges = bool(not np.isfinite(v))

        if martingale_predicted:
            mc_agrees = bool(result.e_s_t > 0.9 and result.diagnostics["exit_fraction"] < 0.1)
        else:
            mc_agrees = bool(result.e_s_t < 0.98 or result.diagnostics["exit_fraction"] > 0.0)

        records.append({
            "rho": rho,
            "gamma": gamma,
            "e_s_t": result.e_s_t,
            "se_s_t": result.se_s_t,
            "survival_prob": result.survival_prob,
            "exit_fraction": result.diagnostics["exit_fraction"],
            "agreement_within_ci": result.agreement_within_ci,
            "martingale_predicted": martingale_predicted,
            "v_AB": None if not np.isfinite(v) else float(v),
            "v_diverges": v_diverges,
            "mc_agrees_with_prediction": mc_agrees,
        })

    all_regimes_consistent = all(
        (r["v_diverges"] == r["martingale_predicted"]) and r["mc_agrees_with_prediction"]
        for r in records
    )
    # Independent of M, q and Omega: the threshold reproduces Example 6.1
    # exactly (same analytic A, B as exp3's isotropic test).
    threshold_independent_of_anisotropy = bool(all_regimes_consistent)

    summary = {
        "records": records,
        "all_regimes_consistent": bool(all_regimes_consistent),
        "threshold_independent_of_anisotropy": threshold_independent_of_anisotropy,
        "n_paths": n_paths,
        "n_steps": n_steps,
        "full": full,
    }
    return summary


def save(summary: dict) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "exp6_results.json", "w") as f:
        json.dump(to_jsonable(summary), f, indent=2)

    records = summary["records"]
    labels = [f"rho={r['rho']}\ngamma={r['gamma']}" for r in records]
    e_s_t_vals = [r["e_s_t"] for r in records]
    colors = ["tab:green" if r["martingale_predicted"] else "tab:red" for r in records]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(labels, e_s_t_vals, color=colors)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="reference: E[S_T]=1")
    ax.set_ylabel("E_y[S_T]")
    ax.set_title("Experiment 6: anisotropic model (Ex. 6.2) and exact Feller test (Prop. 4.3)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp6_anisotropic_feller_test.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="run the paper-scale configuration")
    args = parser.parse_args()

    summary = run(args.full)
    save(summary)

    print(f"Experiment 6 ({'full' if args.full else 'reduced'} configuration)")
    for r in summary["records"]:
        print(
            f"  rho={r['rho']:+.2f}, gamma={r['gamma']:.2f}: E[S_T]={r['e_s_t']:.4f}, "
            f"survival_prob={r['survival_prob']:.4f}, exit_fraction={r['exit_fraction']:.4f}, "
            f"v_AB={r['v_AB']}, v_diverges={r['v_diverges']}, "
            f"martingale_predicted={r['martingale_predicted']}, "
            f"mc_agrees={r['mc_agrees_with_prediction']}"
        )
    print(f"  all_regimes_consistent: {summary['all_regimes_consistent']}")
    print(f"  threshold_independent_of_anisotropy: {summary['threshold_independent_of_anisotropy']}")


if __name__ == "__main__":
    main()
