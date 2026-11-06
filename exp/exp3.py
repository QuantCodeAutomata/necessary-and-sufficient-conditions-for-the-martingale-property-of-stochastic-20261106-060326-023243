"""Experiment 3 -- Explosion at Infinity: Radial Khasminskii Tests and
Feller's Test (Propositions 4.1-4.2), verified by Monte Carlo simulation of
the isotropic power-volatility model (Example 6.1, paper Section 4, p.18-19,
27-28).

Example 6.1 sets dY = -kappa Y dt + eta dZ (Z a dim-dimensional Brownian
motion), dS = S |Y|^p dB with d<B,Z_i> = rho Y_i/|Y| dt. The Girsanov-shifted
drift is mu_tilde(y) = (-kappa + rho*eta*|y|^(p-1)) y. Propositions 4.1-4.2
reduce the martingale question for S to a radial (Feller) scale-function test
on |Y|, which here collapses to the simple analytic dichotomy: S is a true
martingale iff rho<=0 or p<=1 (no superlinear self-reinforcing drift), and a
strict local martingale (explosion to infinity with positive probability)
iff rho>0 and p>1.

D = R^dim has no literal boundary, so power_volatility_isotropic's own
domain_indicator always returns True. For a finite-step Monte Carlo proxy of
"explosion to infinity" we substitute our own domain_indicator that marks a
path as exited once |y| crosses EXPLOSION_RADIUS (a simulation cutoff, not a
quantity from the paper).

Run:
    python -m exp.exp3            # reduced configuration
    python -m exp.exp3 --full     # paper-scale configuration
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
from src.models import power_volatility_isotropic
from src.martingale_criteria import feller_integral_v_AB

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

EXPLOSION_RADIUS = 50.0  # proxy cutoff for "has effectively exploded"; not specified in the paper


def make_model_with_explosion_proxy(dim: int, kappa: float, eta: float, p: float, rho: float) -> dict:
    model = power_volatility_isotropic(dim=dim, kappa=kappa, eta=eta, p=p, rho=rho)
    model = dict(model)  # shallow copy, do not mutate the original
    model["domain_indicator"] = lambda y: np.linalg.norm(y, axis=1) < EXPLOSION_RADIUS
    return model


def run(full: bool) -> dict:
    n_steps_unit = cfg.N_STEPS_PER_UNIT_TIME_FULL if full else cfg.N_STEPS_PER_UNIT_TIME_RUN
    n_paths = cfg.N_PATHS_FULL if full else cfg.N_PATHS_RUN
    n_steps = max(int(n_steps_unit * cfg.EXP3_HORIZON), 10)

    records = []
    for rho, p in cfg.EXP3_REGIMES:
        model = make_model_with_explosion_proxy(
            dim=cfg.EXP3_DIM, kappa=cfg.EXP3_KAPPA, eta=cfg.EXP3_ETA, p=p, rho=rho,
        )
        result = compare_mass_survival(
            model, y0=np.array(cfg.EXP3_Y0), T=cfg.EXP3_HORIZON, n_steps=n_steps,
            n_paths=n_paths, seed=cfg.RNG_SEED, dim=cfg.EXP3_DIM,
            dim_w=cfg.EXP3_DIM + 1, positivity_floor=None,
        )

        martingale_predicted = bool(rho <= 0 or p <= 1)

        # Feller scale-integral cross-check (Proposition 4.1-4.2, eq. (4.3))
        # on the radial coefficients A(r), B(r) of Example 6.1, with
        # r = |y|^2/2.
        A = lambda r: 2 * cfg.EXP3_ETA ** 2 * r
        B = lambda r: (
            cfg.EXP3_DIM * cfg.EXP3_ETA ** 2 - 4 * cfg.EXP3_KAPPA * r
            + 2 * rho * cfg.EXP3_ETA * (2 * r) ** ((p + 1) / 2)
        ) / (2 * cfg.EXP3_ETA ** 2 * r)
        y0_arr = np.array(cfg.EXP3_Y0)
        r0 = max(0.5 * float(np.dot(y0_arr, y0_arr)), 1.0)
        v = feller_integral_v_AB(A, B, r0=r0, r_inf_cutoff=1e4, n_points=4000)
        v_diverges = bool(not np.isfinite(v))

        if martingale_predicted:
            mc_agrees = bool(result.e_s_t > 0.9 and result.diagnostics["exit_fraction"] < 0.1)
        else:
            mc_agrees = bool(result.e_s_t < 0.98 or result.diagnostics["exit_fraction"] > 0.0)

        records.append({
            "rho": rho,
            "p": p,
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

    summary = {
        "records": records,
        "all_regimes_consistent": bool(all_regimes_consistent),
        "n_paths": n_paths,
        "n_steps": n_steps,
        "full": full,
    }
    return summary


def save(summary: dict) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "exp3_results.json", "w") as f:
        json.dump(to_jsonable(summary), f, indent=2)

    records = summary["records"]
    labels = [f"rho={r['rho']}\np={r['p']}" for r in records]
    e_s_t_vals = [r["e_s_t"] for r in records]
    colors = ["tab:green" if r["martingale_predicted"] else "tab:red" for r in records]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(labels, e_s_t_vals, color=colors)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="reference: E[S_T]=1")
    ax.set_ylabel("E_y[S_T]")
    ax.set_title("Experiment 3: Feller/Khasminskii explosion test (Prop. 4.1-4.2, Ex. 6.1)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp3_explosion_feller_test.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="run the paper-scale configuration")
    args = parser.parse_args()

    summary = run(args.full)
    save(summary)

    print(f"Experiment 3 ({'full' if args.full else 'reduced'} configuration)")
    for r in summary["records"]:
        print(
            f"  rho={r['rho']:+.2f}, p={r['p']:.2f}: E[S_T]={r['e_s_t']:.4f}, "
            f"survival_prob={r['survival_prob']:.4f}, exit_fraction={r['exit_fraction']:.4f}, "
            f"v_AB={r['v_AB']}, v_diverges={r['v_diverges']}, "
            f"martingale_predicted={r['martingale_predicted']}, "
            f"mc_agrees={r['mc_agrees_with_prediction']}"
        )
    print(f"  all_regimes_consistent: {summary['all_regimes_consistent']}")


if __name__ == "__main__":
    main()
