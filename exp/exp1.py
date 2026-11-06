"""Experiment 1 -- Loss-of-mass identity (Lemma 2.1) and the finite-horizon
martingale dichotomy (Proposition 2.1), verified by Monte Carlo simulation of
a two-dimensional independent CIR model (paper Section 2, p.7-8).

Lemma 2.1 states E_y[S_T] = Qtilde_y(tau_D > T): the expectation of the
stochastic exponential under the original measure equals the survival
probability of the Girsanov-shifted diffusion. Proposition 2.1(ii) states
the all-or-nothing dichotomy under locally Hoelder continuous coefficients:
either S is a true martingale on [0, infinity) for every starting point, or
S is a strict local martingale on [0, T] for every T > 0 and every y.

Run:
    python -m exp.exp1            # reduced configuration
    python -m exp.exp1 --full     # paper-scale configuration
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
from src.mc_verification import compare_mass_survival, loss_of_mass_curve
from src.models import cir_2d_model

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def run(full: bool) -> dict:
    n_steps_unit = cfg.N_STEPS_PER_UNIT_TIME_FULL if full else cfg.N_STEPS_PER_UNIT_TIME_RUN
    n_paths = cfg.N_PATHS_FULL if full else cfg.N_PATHS_RUN

    y0_list = [np.array(cfg.EXP1_Y0), np.array([2.0, 0.5])]
    regimes = {
        "martingale": cfg.EXP1_THETA_COEF_MARTINGALE,
        "strict_local": cfg.EXP1_THETA_COEF_STRICT,
    }

    identity_checks = []
    dichotomy_checks = []
    for regime_name, theta_coef in regimes.items():
        model = cir_2d_model(
            alpha=cfg.EXP1_ALPHA, kappa=cfg.EXP1_KAPPA, eta=cfg.EXP1_ETA,
            theta_coef=theta_coef,
        )
        for y0 in y0_list:
            for T in cfg.EXP1_HORIZONS:
                n_steps = max(int(n_steps_unit * T), 10)
                result = compare_mass_survival(
                    model, y0=y0, T=T, n_steps=n_steps, n_paths=n_paths,
                    seed=cfg.RNG_SEED, dim=2, dim_w=2, positivity_floor=1e-6,
                )
                record = {
                    "regime": regime_name,
                    "theta_coef": theta_coef,
                    "y0": y0.tolist(),
                    "T": T,
                    "E_S_T": result.e_s_t,
                    "se_E_S_T": result.se_s_t,
                    "survival_prob": result.survival_prob,
                    "se_survival": result.se_survival,
                    "agreement_within_ci": result.agreement_within_ci,
                }
                identity_checks.append(record)
                dichotomy_checks.append(record)

    # Loss-of-mass curve m(t,y) = 1 - E_y[S_t], one regime of each kind,
    # checked for monotonicity (Remark 2.2 / Cauchy problem partial_t m = Ltilde m)
    curves = {}
    for regime_name, theta_coef in regimes.items():
        model = cir_2d_model(
            alpha=cfg.EXP1_ALPHA, kappa=cfg.EXP1_KAPPA, eta=cfg.EXP1_ETA,
            theta_coef=theta_coef,
        )
        T_curve = max(cfg.EXP1_HORIZONS)
        n_steps = max(int(n_steps_unit * T_curve), 10)
        curve = loss_of_mass_curve(
            model, y0=np.array(cfg.EXP1_Y0), T=T_curve, n_steps=n_steps,
            n_paths=n_paths, seed=cfg.RNG_SEED, dim=2, dim_w=2,
            positivity_floor=1e-6,
        )
        m_t = curve["m_t"]
        # Downsample to curb Monte Carlo sampling noise before checking
        # monotonicity of m(t,y) (Remark 2.2): average over windows of
        # roughly n_steps/20 grid points.
        window = max(len(m_t) // 20, 1)
        m_t_smooth = np.array([
            m_t[i:i + window].mean() for i in range(0, len(m_t), window)
        ])
        non_decreasing = bool(np.all(np.diff(m_t_smooth) >= -0.01))
        curves[regime_name] = {
            "times": curve["times"].tolist(),
            "m_t": m_t.tolist(),
            "non_decreasing": non_decreasing,
            "final_loss_of_mass": float(m_t[-1]),
        }

    summary = {
        "identity_checks": identity_checks,
        "loss_of_mass_curves": curves,
        "all_identities_agree": all(r["agreement_within_ci"] for r in identity_checks),
        "strict_regime_loses_mass": all(
            r["E_S_T"] < 1.0 for r in identity_checks if r["regime"] == "strict_local"
        ),
        "martingale_regime_mass_near_one": all(
            r["E_S_T"] > 0.9 for r in identity_checks if r["regime"] == "martingale"
        ),
        "n_paths": n_paths,
        "full": full,
    }
    return summary


def save(summary: dict) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "exp1_results.json", "w") as f:
        json.dump(to_jsonable(summary), f, indent=2)

    fig, ax = plt.subplots(figsize=(6, 4))
    for regime_name, curve in summary["loss_of_mass_curves"].items():
        ax.plot(curve["times"], curve["m_t"], label=regime_name)
    ax.set_xlabel("t")
    ax.set_ylabel("m(t, y) = 1 - E_y[S_t]")
    ax.set_title("Experiment 1: loss of mass m(t,y) (Lemma 2.1 / Remark 2.2)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp1_loss_of_mass.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="run the paper-scale configuration")
    args = parser.parse_args()

    summary = run(args.full)
    save(summary)

    print(f"Experiment 1 ({'full' if args.full else 'reduced'} configuration)")
    print(f"  all Lemma 2.1 identities agree within CI: {summary['all_identities_agree']}")
    print(f"  strict-local regime loses mass (E[S_T]<1): {summary['strict_regime_loses_mass']}")
    print(f"  martingale regime keeps mass near 1:       {summary['martingale_regime_mass_near_one']}")
    for regime, curve in summary["loss_of_mass_curves"].items():
        print(f"  {regime}: m(T) = {curve['final_loss_of_mass']:.4f}, non-decreasing = {curve['non_decreasing']}")


if __name__ == "__main__":
    main()
