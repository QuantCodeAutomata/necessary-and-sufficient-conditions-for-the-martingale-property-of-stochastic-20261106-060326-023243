"""Experiment 2 -- Fichera drift criterion at smooth boundary points
(Theorem 3.1) and condition (T) (Section 3.2), verified by Monte Carlo
simulation of the 3/2 model (Example 6.3) and the two-factor inverse-CIR
model (Example 6.4).

Theorem 3.1 states that at a smooth boundary point of D the stochastic
exponential is (locally) a true martingale iff the Fichera drift
F(x) = <mu_tilde(x) - 0.5 div a(x), nu(x)> is non-positive (the diffusion
does not flow out of D), together with condition (T), a local integral
test near the boundary controlling how fast the drift may become positive.

For the 3/2 model (Example 6.3), dY=(alpha-kappa*Y)dt+eta*sqrt(Y)dZ on
D=(0,infinity), the shifted drift is mu_tilde(y)=alpha+rho*eta-kappa*y and
the criterion reduces to alpha+rho*eta >= 0.5*eta^2, i.e. the Fichera
drift F=-(alpha+rho*eta-0.5*eta^2) at y=0 is <=0.

For the two-factor inverse-CIR model (Example 6.4), the analogous
criterion on the Y^1-face is 2*alpha1-eta1^2 >= 2*eta1*(rho)^- * sup(g).

Run:
    python -m exp.exp2            # reduced configuration
    python -m exp.exp2 --full     # paper-scale configuration
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
from src.models import model_32, two_factor_inverse_cir
from src.martingale_criteria import fichera_drift, condition_T_holds, v_AB_scalar_cir

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def _run_32_model(regime_name: str, alpha: float, n_steps_unit: float, n_paths: int) -> dict:
    """One Monte Carlo regime of the 3/2 model (Example 6.3)."""
    eta, kappa, rho = cfg.EXP2_ETA, cfg.EXP2_KAPPA, cfg.EXP2_RHO
    T = cfg.EXP2_HORIZON
    n_steps = max(int(n_steps_unit * T), 10)

    model = model_32(alpha=alpha, kappa=kappa, eta=eta, rho=rho)
    result = compare_mass_survival(
        model, y0=np.array([cfg.EXP2_Y0]), T=T, n_steps=n_steps, n_paths=n_paths,
        seed=cfg.RNG_SEED, dim=1, dim_w=2, positivity_floor=1e-6,
    )

    # Fichera drift at the boundary y=0 (Theorem 3.1): a(y)=eta^2*y so
    # div a = eta^2 at y=0, mu_tilde(0) = alpha + rho*eta, outward normal
    # of the face {y=0} of the half-line is nu=-1.
    F = fichera_drift(a_ii=0.0, div_a_i=eta ** 2, mu_tilde_i=alpha + rho * eta, nu=-1.0)

    # Condition (T) (Section 3.2) evaluated at a small distance rho_dist
    # from the boundary: a(y)/y - 2*mu_tilde(y) = eta^2 - 2*(alpha+rho*eta-kappa*y).
    rho_dist = 1e-3
    lhs = eta ** 2 - 2.0 * (alpha + rho * eta) + 2.0 * kappa * rho_dist
    C = 2.0 * kappa
    t_holds = condition_T_holds(a_over_rho_minus_two_mu_tilde=lhs, rho=rho_dist, C=C)

    # Auxiliary Feller scale-integral cross-check (not the primary
    # criterion): compare the integral at two shrinking inner cutoffs to
    # decide whether it diverges (true martingale) or stays finite
    # (strict local martingale), see Example 6.3.
    v_outer = v_AB_scalar_cir(alpha=alpha, eta=eta, kappa=kappa, c=-rho, y0=1e-8)
    v_inner = v_AB_scalar_cir(alpha=alpha, eta=eta, kappa=kappa, c=-rho, y0=1e-10)
    feller_diverges = v_inner > 10.0 * v_outer

    return {
        "regime": regime_name,
        "alpha": alpha,
        "E_S_T": result.e_s_t,
        "se_E_S_T": result.se_s_t,
        "survival_prob": result.survival_prob,
        "se_survival": result.se_survival,
        "agreement_within_ci": result.agreement_within_ci,
        "exit_fraction": result.diagnostics["exit_fraction"],
        "fichera_drift_F": F,
        "fichera_predicts_martingale": F <= 0.0,
        "condition_T_holds": t_holds,
        "feller_integral_diverges": feller_diverges,
    }


def _run_two_factor_model(regime_name: str, rho: float, n_steps_unit: float, n_paths: int) -> dict:
    """One Monte Carlo regime of the two-factor inverse-CIR model (Example 6.4)."""
    alpha1, kappa1, eta1 = cfg.EXP2_TWO_FACTOR_ALPHA1, cfg.EXP2_TWO_FACTOR_KAPPA1, cfg.EXP2_TWO_FACTOR_ETA1
    alpha2, kappa2, eta2 = cfg.EXP2_TWO_FACTOR_ALPHA2, cfg.EXP2_TWO_FACTOR_KAPPA2, cfg.EXP2_TWO_FACTOR_ETA2
    g_bar = cfg.EXP2_TWO_FACTOR_G_BAR
    T = cfg.EXP2_HORIZON
    n_steps = max(int(n_steps_unit * T), 10)

    model = two_factor_inverse_cir(
        alpha1=alpha1, kappa1=kappa1, eta1=eta1,
        alpha2=alpha2, kappa2=kappa2, eta2=eta2,
        rho=rho, g_bar=g_bar,
    )
    result = compare_mass_survival(
        model, y0=np.array(cfg.EXP2_TWO_FACTOR_Y0), T=T, n_steps=n_steps, n_paths=n_paths,
        seed=cfg.RNG_SEED, dim=2, dim_w=3, positivity_floor=1e-6,
    )

    # Fichera-type criterion on the Y^1-face (Example 6.4): true martingale
    # iff 2*alpha1-eta1^2 >= 2*eta1*(rho)^- * sup(g), (rho)^- = max(-rho,0).
    rho_minus = max(-rho, 0.0)
    lhs = 2.0 * alpha1 - eta1 ** 2
    rhs = 2.0 * eta1 * rho_minus * g_bar
    inequality_holds = lhs >= rhs

    return {
        "regime": regime_name,
        "rho": rho,
        "E_S_T": result.e_s_t,
        "se_E_S_T": result.se_s_t,
        "survival_prob": result.survival_prob,
        "se_survival": result.se_survival,
        "agreement_within_ci": result.agreement_within_ci,
        "exit_fraction": result.diagnostics["exit_fraction"],
        "lhs_2alpha1_minus_eta1sq": lhs,
        "rhs_threshold": rhs,
        "inequality_predicts_martingale": inequality_holds,
    }


def run(full: bool) -> dict:
    n_steps_unit = cfg.N_STEPS_PER_UNIT_TIME_FULL if full else cfg.N_STEPS_PER_UNIT_TIME_RUN
    n_paths = cfg.N_PATHS_FULL if full else cfg.N_PATHS_RUN

    # Threshold alpha0 = 0.5*eta^2 - rho*eta; two regimes symmetric around it.
    alpha0 = 0.5 * cfg.EXP2_ETA ** 2 - cfg.EXP2_RHO * cfg.EXP2_ETA
    alpha_above = alpha0 + cfg.EXP2_MARGIN   # true martingale
    alpha_below = alpha0 - cfg.EXP2_MARGIN   # strict local martingale

    model_32_results = [
        _run_32_model("martingale", alpha_above, n_steps_unit, n_paths),
        _run_32_model("strict_local", alpha_below, n_steps_unit, n_paths),
    ]

    two_factor_results = [
        _run_two_factor_model("strict_local", cfg.EXP2_TWO_FACTOR_RHO_STRICT, n_steps_unit, n_paths),
        _run_two_factor_model("martingale", cfg.EXP2_TWO_FACTOR_RHO_MARTINGALE, n_steps_unit, n_paths),
    ]

    # Loss-of-mass curves for the 3/2 model, one per regime, for a visual
    # sanity check of monotone mass loss in the strict-local regime.
    curves = {}
    for regime_name, alpha in (("martingale", alpha_above), ("strict_local", alpha_below)):
        model = model_32(alpha=alpha, kappa=cfg.EXP2_KAPPA, eta=cfg.EXP2_ETA, rho=cfg.EXP2_RHO)
        T = cfg.EXP2_HORIZON
        n_steps = max(int(n_steps_unit * T), 10)
        curve = loss_of_mass_curve(
            model, y0=np.array([cfg.EXP2_Y0]), T=T, n_steps=n_steps, n_paths=n_paths,
            seed=cfg.RNG_SEED, dim=1, dim_w=2, positivity_floor=1e-6,
        )
        curves[regime_name] = {"times": curve["times"].tolist(), "m_t": curve["m_t"].tolist()}

    martingale_regime_ok = all(
        r["E_S_T"] > 0.9 for r in model_32_results if r["regime"] == "martingale"
    )
    strict_regime_ok = all(
        r["E_S_T"] < 0.98 for r in model_32_results if r["regime"] == "strict_local"
    )

    def _criterion_agrees_with_mc(predicts_martingale: bool, e_s_t: float) -> bool:
        if predicts_martingale:
            return e_s_t > 0.9
        return e_s_t < 0.98

    fichera_sign_predicts_mc = all(
        _criterion_agrees_with_mc(r["fichera_predicts_martingale"], r["E_S_T"])
        for r in model_32_results
    ) and all(
        _criterion_agrees_with_mc(r["inequality_predicts_martingale"], r["E_S_T"])
        for r in two_factor_results
    )

    summary = {
        "model_32_results": model_32_results,
        "two_factor_results": two_factor_results,
        "loss_of_mass_curves": curves,
        "martingale_regime_mass_near_one": martingale_regime_ok,
        "strict_regime_loses_mass": strict_regime_ok,
        "fichera_sign_predicts_mc": fichera_sign_predicts_mc,
        "n_paths": n_paths,
        "full": full,
    }
    return summary


def save(summary: dict) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "exp2_results.json", "w") as f:
        json.dump(to_jsonable(summary), f, indent=2)

    labels = [f"3/2 {r['regime']}" for r in summary["model_32_results"]]
    labels += [f"2-factor {r['regime']}" for r in summary["two_factor_results"]]
    values = [r["E_S_T"] for r in summary["model_32_results"]]
    values += [r["E_S_T"] for r in summary["two_factor_results"]]

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["tab:green" if v > 0.9 else "tab:red" for v in values]
    ax.bar(labels, values, color=colors)
    ax.axhline(1.0, color="black", linestyle="--", label="E[S_T]=1 (full mass)")
    ax.set_ylabel("E[S_T]")
    ax.set_title("Experiment 2: Fichera drift criterion (Theorem 3.1)")
    ax.legend()
    fig.autofmt_xdate(rotation=20)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp2_fichera_drift.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="run the paper-scale configuration")
    args = parser.parse_args()

    summary = run(args.full)
    save(summary)

    print(f"Experiment 2 ({'full' if args.full else 'reduced'} configuration)")
    for r in summary["model_32_results"]:
        print(f"  3/2 model [{r['regime']}]: E[S_T]={r['E_S_T']:.4f}, survival_prob={r['survival_prob']:.4f}, "
              f"F={r['fichera_drift_F']:.4f}, condition(T)={r['condition_T_holds']}")
    for r in summary["two_factor_results"]:
        print(f"  2-factor model [{r['regime']}]: E[S_T]={r['E_S_T']:.4f}, survival_prob={r['survival_prob']:.4f}, "
              f"inequality_holds={r['inequality_predicts_martingale']}")
    print(f"  martingale regime mass near 1: {summary['martingale_regime_mass_near_one']}")
    print(f"  strict regime loses mass:      {summary['strict_regime_loses_mass']}")
    print(f"  fichera_sign_predicts_mc:      {summary['fichera_sign_predicts_mc']}")


if __name__ == "__main__":
    main()
