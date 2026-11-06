"""Experiment 7 -- 4/2 Stochastic Volatility Model: Martingale Criterion and
Multi-Factor Extension (Example 6.5, Theorem 6.2), paper Section 6, p.29-32.

Single-factor 4/2 model (Example 6.5): dV = (alpha-kappa V)dt + eta sqrt(V)
dZ, dS = S(a sqrt(V) + b/sqrt(V)) dB, d<B,Z>=rho dt. The Heston part
a*sqrt(V) is bounded near V=0, so by Corollary 3.1 it does not change the
Fichera drift at the boundary; only the 3/2 part b/sqrt(V) (unbounded near
0) matters. The Fichera drift F(0) = -(alpha+rho*eta*b - 0.5*eta^2), so S
is a true martingale iff alpha + rho*eta*b >= 0.5*eta^2, for every value of
the Heston coefficient a.

Multi-factor correlated extension (Theorem 6.2 / Example 6.6): ell(y) =
u*sqrt(y1) + v*g(y2)/sqrt(y1) with g bounded; S is a true martingale iff
2*alpha1 - eta1^2 >= 2*eta1*(rho1*v)^- * sup g, where (x)^- = max(-x,0).
With the parameters of Example 6.6 (eta1=eta2=kappa1=kappa2=1, rho1=-1/2,
v=1, g(z)=1+z/(1+z) so sup g=2), the threshold is exactly alpha1 >= 3/2.

Run:
    python -m exp.exp7            # reduced configuration
    python -m exp.exp7 --full     # paper-scale configuration
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
from src.models import model_4_2, model_4_2_multi

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def _b_threshold(alpha: float, kappa: float, eta: float, rho: float) -> float:
    """Critical b_32 solving alpha + rho*eta*b = 0.5*eta^2 (Example 6.5)."""
    return (0.5 * eta ** 2 - alpha) / (rho * eta)


def _run_single_factor(label: str, b_32: float, a_heston: float,
                        n_steps_unit: float, n_paths: int) -> dict:
    alpha, kappa, eta, rho = cfg.EXP7_ALPHA, cfg.EXP7_KAPPA, cfg.EXP7_ETA, cfg.EXP7_RHO
    T = cfg.EXP7_HORIZON
    n_steps = max(int(n_steps_unit * T), 10)

    model = model_4_2(alpha=alpha, kappa=kappa, eta=eta, rho=rho, a_heston=a_heston, b_32=b_32)
    result = compare_mass_survival(
        model, y0=np.array([cfg.EXP7_V0]), T=T, n_steps=n_steps, n_paths=n_paths,
        seed=cfg.RNG_SEED, dim=1, dim_w=2, positivity_floor=1e-10,
    )
    lhs = alpha + rho * eta * b_32
    threshold = 0.5 * eta ** 2
    martingale_predicted = bool(lhs >= threshold - 1e-9)
    return {
        "label": label,
        "a_heston": a_heston,
        "b_32": b_32,
        "alpha_plus_rho_eta_b": lhs,
        "half_eta_sq": threshold,
        "martingale_predicted": martingale_predicted,
        "E_S_T": result.e_s_t,
        "se_E_S_T": result.se_s_t,
        "survival_prob": result.survival_prob,
        "exit_fraction": result.diagnostics["exit_fraction"],
        "agreement_within_ci": result.agreement_within_ci,
    }


def _g_bar_fun(z: np.ndarray) -> np.ndarray:
    """Example 6.6 modulation g(z) = 1 + z/(1+z), sup g = 2 as z -> infinity."""
    return 1.0 + z / (1.0 + z)


def _run_multi_factor(label: str, alpha1: float, n_steps_unit: float, n_paths: int) -> dict:
    eta1, eta2 = cfg.EXP7_MULTI_ETA1, cfg.EXP7_MULTI_ETA2
    kappa1, kappa2 = cfg.EXP7_MULTI_KAPPA1, cfg.EXP7_MULTI_KAPPA2
    alpha2 = cfg.EXP7_MULTI_ALPHA2
    rho1, rho2 = cfg.EXP7_MULTI_RHO1, cfg.EXP7_MULTI_RHO2
    zz_corr = cfg.EXP7_MULTI_ZZ_CORR
    u, v = cfg.EXP7_MULTI_U, cfg.EXP7_MULTI_V
    T = cfg.EXP7_MULTI_HORIZON
    n_steps = max(int(n_steps_unit * T), 10)

    model = model_4_2_multi(
        eta1=eta1, eta2=eta2, kappa1=kappa1, kappa2=kappa2, alpha1=alpha1, alpha2=alpha2,
        rho1=rho1, rho2=rho2, zz_corr=zz_corr, u=u, v=v, g_bar_fun=_g_bar_fun,
    )
    result = compare_mass_survival(
        model, y0=np.array(cfg.EXP7_MULTI_Y0), T=T, n_steps=n_steps, n_paths=n_paths,
        seed=cfg.RNG_SEED, dim=2, dim_w=3, positivity_floor=1e-10,
    )

    sup_g = 2.0  # sup_z g(z) = 1 + lim_{z->inf} z/(1+z) = 2
    rho1_v_minus = max(-rho1 * v, 0.0)  # (rho1*v)^-
    lhs = 2 * alpha1 - eta1 ** 2
    rhs = 2 * eta1 * rho1_v_minus * sup_g
    martingale_predicted = bool(lhs >= rhs - 1e-9)

    return {
        "label": label,
        "alpha1": alpha1,
        "lhs_2alpha1_minus_eta1sq": lhs,
        "rhs_threshold": rhs,
        "martingale_predicted": martingale_predicted,
        "E_S_T": result.e_s_t,
        "se_E_S_T": result.se_s_t,
        "survival_prob": result.survival_prob,
        "exit_fraction": result.diagnostics["exit_fraction"],
        "agreement_within_ci": result.agreement_within_ci,
    }


def run(full: bool) -> dict:
    n_steps_unit = cfg.N_STEPS_PER_UNIT_TIME_FULL if full else cfg.N_STEPS_PER_UNIT_TIME_RUN
    n_paths = cfg.N_PATHS_FULL if full else cfg.N_PATHS_RUN

    alpha, kappa, eta, rho = cfg.EXP7_ALPHA, cfg.EXP7_KAPPA, cfg.EXP7_ETA, cfg.EXP7_RHO
    b_crit = _b_threshold(alpha, kappa, eta, rho)
    # rho*eta < 0 here, so the condition alpha+rho*eta*b>=0.5*eta^2 reads b<=b_crit
    b_martingale = b_crit - cfg.EXP7_B_MARGIN
    b_strict = b_crit + cfg.EXP7_B_MARGIN

    single_factor_results = [
        _run_single_factor("martingale", b_martingale, cfg.EXP7_A_HESTON, n_steps_unit, n_paths),
        _run_single_factor("strict_local", b_strict, cfg.EXP7_A_HESTON, n_steps_unit, n_paths),
        # Corollary 3.1: the Heston part a does not affect the threshold --
        # rerun the martingale regime with a=0 and compare to a=EXP7_A_HESTON.
        _run_single_factor("martingale_a_zero", b_martingale, 0.0, n_steps_unit, n_paths),
    ]

    multi_factor_results = [
        _run_multi_factor("martingale", cfg.EXP7_MULTI_ALPHA1_MARTINGALE, n_steps_unit, n_paths),
        _run_multi_factor("strict_local", cfg.EXP7_MULTI_ALPHA1_STRICT, n_steps_unit, n_paths),
    ]

    def _agrees(predicts_martingale: bool, e_s_t: float) -> bool:
        return e_s_t > 0.9 if predicts_martingale else e_s_t < 0.98

    single_factor_confirmed = all(
        _agrees(r["martingale_predicted"], r["E_S_T"]) for r in single_factor_results[:2]
    )
    heston_part_irrelevant = bool(
        abs(single_factor_results[0]["E_S_T"] - single_factor_results[2]["E_S_T"])
        < 4 * max(single_factor_results[0]["se_E_S_T"], single_factor_results[2]["se_E_S_T"], 1e-6)
    )
    multi_factor_confirmed = all(
        _agrees(r["martingale_predicted"], r["E_S_T"]) for r in multi_factor_results
    )

    summary = {
        "single_factor_results": single_factor_results,
        "multi_factor_results": multi_factor_results,
        "single_factor_confirmed": bool(single_factor_confirmed),
        "heston_part_irrelevant": heston_part_irrelevant,
        "multi_factor_confirmed": bool(multi_factor_confirmed),
        "b_crit": b_crit,
        "n_paths": n_paths,
        "full": full,
    }
    return summary


def save(summary: dict) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "exp7_results.json", "w") as f:
        json.dump(to_jsonable(summary), f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    r1 = summary["single_factor_results"][:2]
    labels1 = [f"{r['label']}\nb={r['b_32']:.2f}" for r in r1]
    vals1 = [r["E_S_T"] for r in r1]
    colors1 = ["tab:green" if r["martingale_predicted"] else "tab:red" for r in r1]
    axes[0].bar(labels1, vals1, color=colors1)
    axes[0].axhline(1.0, color="black", linestyle="--")
    axes[0].set_ylabel("E[S_T]")
    axes[0].set_title("4/2 model (Ex. 6.5): alpha+rho*eta*b>=0.5*eta^2")

    r2 = summary["multi_factor_results"]
    labels2 = [f"{r['label']}\nalpha1={r['alpha1']:.2f}" for r in r2]
    vals2 = [r["E_S_T"] for r in r2]
    colors2 = ["tab:green" if r["martingale_predicted"] else "tab:red" for r in r2]
    axes[1].bar(labels2, vals2, color=colors2)
    axes[1].axhline(1.0, color="black", linestyle="--")
    axes[1].set_title("Multi-factor (Thm 6.2, Ex. 6.6): alpha1>=3/2")

    fig.suptitle("Experiment 7: 4/2 model and multi-factor extension")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp7_4_2_model.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="run the paper-scale configuration")
    args = parser.parse_args()

    summary = run(args.full)
    save(summary)

    print(f"Experiment 7 ({'full' if args.full else 'reduced'} configuration)")
    for r in summary["single_factor_results"]:
        print(f"  [{r['label']}] a={r['a_heston']}, b={r['b_32']:.3f}: E[S_T]={r['E_S_T']:.4f}, "
              f"survival={r['survival_prob']:.4f}, martingale_predicted={r['martingale_predicted']}")
    for r in summary["multi_factor_results"]:
        print(f"  [{r['label']}] alpha1={r['alpha1']:.3f}: E[S_T]={r['E_S_T']:.4f}, "
              f"survival={r['survival_prob']:.4f}, martingale_predicted={r['martingale_predicted']}")
    print(f"  single_factor_confirmed: {summary['single_factor_confirmed']}")
    print(f"  heston_part_irrelevant:  {summary['heston_part_irrelevant']}")
    print(f"  multi_factor_confirmed:  {summary['multi_factor_confirmed']}")


if __name__ == "__main__":
    main()
