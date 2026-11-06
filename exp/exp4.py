"""Experiment 4 -- Critical logarithmic boundary (Theorem 3.2 and Example
3.2), verified by Monte Carlo simulation of the critical CIR factor with a
logarithmic drift correction.

Example 3.2 (paper Section 3.7, p.15-17): the critical CIR factor dY =
0.5 dt + sqrt(Y) dZ (kappa=0, alpha=1/2, so 2*alpha=eta^2, the Feller
boundary case) with theta(y) = -c/(sqrt(y) log(1/y)) for y near 0 gives a
shifted drift mu_tilde(y) = 0.5 - c/log(1/y) whose scale density p(y) =
1/(y [log(1/y)]^{2c}) has int_0 p(y) dy = infinity iff c <= 1/2: S is a
true martingale iff c <= 1/2 (the equality case included), even though
condition (T) of Section 3.2 fails for every c > 0 (a(y)/y - 2 mu_tilde(y)
= 2c/log(1/y) is not O(y) for any c), so only the Feller/Khasminskii test,
not the Fichera-drift sufficient condition, gives the exact threshold.

Theorem 3.2 extends this to a critical CIR factor X coupled to correlated
Ornstein-Uhlenbeck factors Z through c(z) = a + b z^2/(1+z^2): the sharp
threshold becomes sup c(z) = a + b <= 1/2.

Run:
    python -m exp.exp4            # reduced configuration
    python -m exp.exp4 --full     # paper-scale configuration
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
from src.models import critical_log_cir, coupled_log_cir
from src.martingale_criteria import scale_integral_diverges_log_cir

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def _run_single_factor(c: float, n_steps_unit: float, n_paths: int) -> dict:
    """One Monte Carlo regime of the 1D critical CIR with logarithmic
    correction (Example 3.2): true martingale iff c <= 1/2."""
    T = cfg.EXP4_HORIZON
    n_steps = max(int(n_steps_unit * T), 10)

    model = critical_log_cir(c=c)
    result = compare_mass_survival(
        model, y0=np.array([cfg.EXP4_Y0]), T=T, n_steps=n_steps, n_paths=n_paths,
        seed=cfg.RNG_SEED, dim=1, dim_w=1, positivity_floor=1e-10,
    )

    # Analytic Feller/Khasminskii criterion (scale density p(y) = 1/(y
    # [log(1/y)]^{2c}), paper Example 3.2): the scale integral at 0
    # diverges iff 2c <= 1, which is the exact martingale threshold here,
    # even though condition (T) fails for all c > 0 (checked separately).
    scale_diverges = scale_integral_diverges_log_cir(c)

    # Condition (T) check (Section 3.2): a(y)/y - 2 mu_tilde(y) =
    # eta^2 - 2*(0.5 - c/log(1/y)) = 2c/log(1/y), which the paper shows is
    # NOT O(y) as y -> 0 for any c > 0, so (T) fails identically here.
    y_test = np.array([1e-4, 1e-6, 1e-8])
    lhs = 2.0 * c / np.log(1.0 / y_test)
    # (T) would require lhs <= C*y; the ratio lhs/y -> infinity, so (T) fails.
    ratio_growth = float((lhs / y_test)[-1] / (lhs / y_test)[0])
    condition_T_fails = bool(c > 0 and ratio_growth > 10.0)

    return {
        "c": c,
        "E_S_T": result.e_s_t,
        "se_E_S_T": result.se_s_t,
        "survival_prob": result.survival_prob,
        "se_survival": result.se_survival,
        "agreement_within_ci": result.agreement_within_ci,
        "exit_fraction": result.diagnostics["exit_fraction"],
        "scale_integral_diverges": scale_diverges,
        "scale_criterion_predicts_martingale": scale_diverges,
        "condition_T_fails": condition_T_fails,
        "analytic_martingale_threshold": c <= 0.5,
    }


def _run_coupled(regime_name: str, b: float, n_steps_unit: float, n_paths: int) -> dict:
    """One Monte Carlo regime of the coupled critical CIR / OU system
    (Theorem 3.2): true martingale iff a + b <= 1/2 (sup c(z) = a + b
    since c(z) = a + b*z^2/(1+z^2) increases to a+b as |z| -> infinity)."""
    a = cfg.EXP4_COUPLED_A
    T = cfg.EXP4_HORIZON
    n_steps = max(int(n_steps_unit * T), 10)

    model = coupled_log_cir(
        a=a, b=b, lam=cfg.EXP4_COUPLED_LAMBDA, eta_ou=cfg.EXP4_COUPLED_ETA,
        rho=cfg.EXP4_COUPLED_RHO, kappa=cfg.EXP4_COUPLED_KAPPA,
    )
    y0 = np.array([cfg.EXP4_Y0, cfg.EXP4_COUPLED_Z0])
    # The X-coordinate is CIR-type (must stay positive) but Z is an
    # unconstrained Ornstein-Uhlenbeck factor: a uniform positivity floor
    # would wrongly clip Z to stay non-negative, so no global floor is
    # applied here; the model's own functions locally floor X where needed
    # (src/models.py coupled_log_cir) and domain_indicator tracks only X.
    result = compare_mass_survival(
        model, y0=y0, T=T, n_steps=n_steps, n_paths=n_paths,
        seed=cfg.RNG_SEED, dim=2, dim_w=2, positivity_floor=None,
    )

    sup_c = a + b
    return {
        "regime": regime_name,
        "a": a,
        "b": b,
        "sup_c": sup_c,
        "E_S_T": result.e_s_t,
        "se_E_S_T": result.se_s_t,
        "survival_prob": result.survival_prob,
        "se_survival": result.se_survival,
        "agreement_within_ci": result.agreement_within_ci,
        "exit_fraction": result.diagnostics["exit_fraction"],
        "analytic_martingale_threshold": sup_c <= 0.5,
    }


def run(full: bool) -> dict:
    # Finer time grid than the other experiments: the critical CIR
    # boundary makes Euler-Maruyama converge only logarithmically slowly.
    n_steps_unit = cfg.EXP4_N_STEPS_PER_UNIT_TIME_FULL if full else cfg.EXP4_N_STEPS_PER_UNIT_TIME_RUN
    n_paths = cfg.N_PATHS_FULL if full else cfg.N_PATHS_RUN

    single_factor_results = [
        _run_single_factor(c, n_steps_unit, n_paths) for c in cfg.EXP4_C_VALUES
    ]

    coupled_results = [
        _run_coupled("martingale", cfg.EXP4_COUPLED_B_MARTINGALE, n_steps_unit, n_paths),
        _run_coupled("strict_local", cfg.EXP4_COUPLED_B_STRICT, n_steps_unit, n_paths),
    ]

    def _agrees(predicts_martingale: bool, e_s_t: float) -> bool:
        return e_s_t > 0.9 if predicts_martingale else e_s_t < 0.98

    threshold_confirmed = all(
        _agrees(r["analytic_martingale_threshold"], r["E_S_T"]) for r in single_factor_results
    )
    coupled_threshold_confirmed = all(
        _agrees(r["analytic_martingale_threshold"], r["E_S_T"]) for r in coupled_results
    )
    condition_T_fails_for_all_c = all(
        r["condition_T_fails"] for r in single_factor_results if r["c"] > 0
    )

    summary = {
        "single_factor_results": single_factor_results,
        "coupled_results": coupled_results,
        "threshold_c_half_confirmed": threshold_confirmed,
        "coupled_threshold_confirmed": coupled_threshold_confirmed,
        "condition_T_fails_for_all_c": condition_T_fails_for_all_c,
        "n_paths": n_paths,
        "full": full,
    }
    return summary


def save(summary: dict) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "exp4_results.json", "w") as f:
        json.dump(to_jsonable(summary), f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    cs = [r["c"] for r in summary["single_factor_results"]]
    vals = [r["E_S_T"] for r in summary["single_factor_results"]]
    colors = ["tab:green" if c <= 0.5 else "tab:red" for c in cs]
    axes[0].bar([f"c={c}" for c in cs], vals, color=colors)
    axes[0].axhline(1.0, color="black", linestyle="--")
    axes[0].set_ylabel("E[S_T]")
    axes[0].set_title("1D critical CIR (Example 3.2): martingale iff c<=1/2")

    labels = [f"{r['regime']}\n(a+b={r['sup_c']:.2f})" for r in summary["coupled_results"]]
    vals2 = [r["E_S_T"] for r in summary["coupled_results"]]
    colors2 = ["tab:green" if r["sup_c"] <= 0.5 else "tab:red" for r in summary["coupled_results"]]
    axes[1].bar(labels, vals2, color=colors2)
    axes[1].axhline(1.0, color="black", linestyle="--")
    axes[1].set_title("Coupled critical CIR/OU (Theorem 3.2)")

    fig.suptitle("Experiment 4: critical logarithmic boundary")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp4_critical_log_boundary.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="run the paper-scale configuration")
    args = parser.parse_args()

    summary = run(args.full)
    save(summary)

    print(f"Experiment 4 ({'full' if args.full else 'reduced'} configuration)")
    for r in summary["single_factor_results"]:
        print(f"  c={r['c']}: E[S_T]={r['E_S_T']:.4f}, survival={r['survival_prob']:.4f}, "
              f"scale integral diverges={r['scale_integral_diverges']}, (T) fails={r['condition_T_fails']}")
    for r in summary["coupled_results"]:
        print(f"  coupled [{r['regime']}] a+b={r['sup_c']:.2f}: E[S_T]={r['E_S_T']:.4f}, "
              f"survival={r['survival_prob']:.4f}")
    print(f"  threshold c<=1/2 confirmed:          {summary['threshold_c_half_confirmed']}")
    print(f"  coupled threshold a+b<=1/2 confirmed: {summary['coupled_threshold_confirmed']}")
    print(f"  condition (T) fails for all c>0:      {summary['condition_T_fails_for_all_c']}")


if __name__ == "__main__":
    main()
