"""Experiment 10 -- Uniqueness of PDE Solutions in the Linear-Growth Class
(Appendix B, Theorem B.1), paper p.39-40.

For the valuation equation (B.3) on G=(0,infinity) x D with payoff
phi(s,y)=s, u_1(tau,s,y)=s is always a solution (s is L-harmonic). When S
is a true martingale, Theorem B.1(ii) gives uniqueness in the linear-growth
class C_lin; when S is a strict local martingale, u_2(tau,s,y) =
s*E_y[S_tau] = s*(1-m(tau,y)) is a second, distinct solution, exhibiting
non-uniqueness. The Doob transform h=1+s turns L into L^h = L +
(s/(1+s)) <sigma theta, grad_y .>, a bounded perturbation that makes the
probabilistic argument (Lemma 2.1) the uniqueness mechanism; no separate
finite-difference discretization of (B.3) is attempted here, since the
paper specifies no boundary condition at the point where volatility is
unbounded and no discretization scheme (Appendix B, uncertainties).

Tested on the 4/2 model (Example 6.5, same coefficients as Experiment 7)
and its multi-factor extension (Theorem 6.2, Example 6.6 parameters).

Run:
    python -m exp.exp10            # reduced configuration
    python -m exp.exp10 --full     # paper-scale configuration
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
from src.models import model_4_2, model_4_2_multi
from src.pde_uniqueness import two_solutions_forward_payoff

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def _run_single_factor(label: str, alpha: float, n_steps_unit: float, n_paths: int) -> dict:
    kappa, eta, rho, b = cfg.EXP10_KAPPA, cfg.EXP10_ETA, cfg.EXP10_RHO, cfg.EXP10_B
    T = cfg.EXP10_HORIZON
    n_steps = max(int(n_steps_unit * T), 10)

    model = model_4_2(alpha=alpha, kappa=kappa, eta=eta, rho=rho, a_heston=0.0, b_32=b)
    out = two_solutions_forward_payoff(
        model, y0=np.array([cfg.EXP10_V0]), T=T, n_steps=n_steps, n_paths=n_paths,
        seed=cfg.RNG_SEED, dim=1, dim_w=2, s0=cfg.EXP10_S0, positivity_floor=1e-10,
    )
    martingale_predicted = bool(alpha + rho * eta * b >= 0.5 * eta ** 2 - 1e-9)
    return {
        "label": label,
        "alpha": alpha,
        "alpha_plus_rho_eta_b": alpha + rho * eta * b,
        "half_eta_sq": 0.5 * eta ** 2,
        "martingale_predicted": martingale_predicted,
        "u1": out["u1"],
        "u2": out["u2"],
        "loss_of_mass": out["loss_of_mass"],
        "relative_gap": out["relative_gap"],
        "unique_in_c_lin": bool(out["relative_gap"] < 0.02),
    }


def _g_bar_fun(z: np.ndarray) -> np.ndarray:
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
    out = two_solutions_forward_payoff(
        model, y0=np.array(cfg.EXP7_MULTI_Y0), T=T, n_steps=n_steps, n_paths=n_paths,
        seed=cfg.RNG_SEED, dim=2, dim_w=3, s0=cfg.EXP10_S0, positivity_floor=1e-10,
    )
    sup_g = 2.0
    rho1_v_minus = max(-rho1 * v, 0.0)
    martingale_predicted = bool(2 * alpha1 - eta1 ** 2 >= 2 * eta1 * rho1_v_minus * sup_g - 1e-9)
    return {
        "label": label,
        "alpha1": alpha1,
        "martingale_predicted": martingale_predicted,
        "u1": out["u1"],
        "u2": out["u2"],
        "loss_of_mass": out["loss_of_mass"],
        "relative_gap": out["relative_gap"],
        "unique_in_c_lin": bool(out["relative_gap"] < 0.02),
    }


def run(full: bool) -> dict:
    n_steps_unit = cfg.N_STEPS_PER_UNIT_TIME_FULL if full else cfg.N_STEPS_PER_UNIT_TIME_RUN
    n_paths = cfg.N_PATHS_FULL if full else cfg.N_PATHS_RUN

    single_factor_results = [
        _run_single_factor("martingale", cfg.EXP10_ALPHA_MARTINGALE, n_steps_unit, n_paths),
        _run_single_factor("strict_local", cfg.EXP10_ALPHA_STRICT, n_steps_unit, n_paths),
    ]
    multi_factor_results = [
        _run_multi_factor("martingale", cfg.EXP7_MULTI_ALPHA1_MARTINGALE, n_steps_unit, n_paths),
        _run_multi_factor("strict_local", cfg.EXP7_MULTI_ALPHA1_STRICT, n_steps_unit, n_paths),
    ]

    def _consistent(r: dict) -> bool:
        return r["unique_in_c_lin"] == r["martingale_predicted"]

    single_factor_confirmed = all(_consistent(r) for r in single_factor_results)
    multi_factor_confirmed = all(_consistent(r) for r in multi_factor_results)

    summary = {
        "single_factor_results": single_factor_results,
        "multi_factor_results": multi_factor_results,
        "single_factor_confirmed": bool(single_factor_confirmed),
        "multi_factor_confirmed": bool(multi_factor_confirmed),
        "n_paths": n_paths,
        "full": full,
    }
    return summary


def save(summary: dict) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "exp10_results.json", "w") as f:
        json.dump(to_jsonable(summary), f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for ax, results, title in (
        (axes[0], summary["single_factor_results"], "4/2 model (Ex. 6.5)"),
        (axes[1], summary["multi_factor_results"], "Multi-factor (Thm 6.2)"),
    ):
        labels = [r["label"] for r in results]
        u1 = [r["u1"] for r in results]
        u2 = [r["u2"] for r in results]
        x = np.arange(len(results))
        width = 0.35
        ax.bar(x - width / 2, u1, width, label="u1 = s", color="tab:blue")
        ax.bar(x + width / 2, u2, width, label="u2 = s(1-m)", color="tab:orange")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(title)
        ax.legend()

    fig.suptitle("Experiment 10: uniqueness in C_lin (Theorem B.1)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp10_pde_uniqueness.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="run the paper-scale configuration")
    args = parser.parse_args()

    summary = run(args.full)
    save(summary)

    print(f"Experiment 10 ({'full' if args.full else 'reduced'} configuration)")
    for r in summary["single_factor_results"]:
        print(f"  [{r['label']}] alpha={r['alpha']:.3f}: u1={r['u1']:.4f}, u2={r['u2']:.4f}, "
              f"relative_gap={r['relative_gap']:.4f}, unique_in_C_lin={r['unique_in_c_lin']}")
    for r in summary["multi_factor_results"]:
        print(f"  [{r['label']}] alpha1={r['alpha1']:.3f}: u1={r['u1']:.4f}, u2={r['u2']:.4f}, "
              f"relative_gap={r['relative_gap']:.4f}, unique_in_C_lin={r['unique_in_c_lin']}")
    print(f"  single_factor_confirmed: {summary['single_factor_confirmed']}")
    print(f"  multi_factor_confirmed:  {summary['multi_factor_confirmed']}")


if __name__ == "__main__":
    main()
