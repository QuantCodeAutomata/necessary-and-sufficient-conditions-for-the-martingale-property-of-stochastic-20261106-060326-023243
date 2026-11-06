"""Experiment 8 -- Absorbed Factor with Correlated Modulation (Theorem 6.3),
paper Section 6.2, p.32-34.

X_t = x + W^0_{t and tau0} is a Brownian motion absorbed at 0 (tau0 = inf{t:
X_t=0}), Z^i are correlated Ornstein-Uhlenbeck factors, and
S_t = exp(int_0^t c(Z)/X dW^0 - 0.5 int_0^t c(Z)^2/X^2 ds) for t<tau0, with
S_t := 0 for t>=tau0 (the continuation by zero is forced by the paper's
argument that int_0^{tau0} c(Z)^2/X^2 ds = infinity on {tau0<infinity}, so
S_t -> 0 as t -> tau0 along the time-changed Brownian-motion representation
S_t = exp(beta_{A_t} - A_t/2)). Theorem 6.3: S is a true martingale iff
inf c(z) >= 1/2. With c(z) = a + b/(1+z^2) (b>0), inf c = a (approached as
|z| -> infinity, not attained), so the exact condition is a >= 1/2.

Because the paper's zero-continuation after absorption is essential to the
statement (Theorem 6.3's whole point is that S_t does not merely stop but
is identically zero afterwards), this experiment does not reuse the shared
compare_mass_survival helper, which only reads S_T=exp(log S_T) without
applying that convention; the zero-continuation is applied explicitly below
before averaging.

Run:
    python -m exp.exp8            # reduced configuration
    python -m exp.exp8 --full     # paper-scale configuration
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
from src.models import absorbed_factor_model
from data.generate_synthetic import simulate_diffusion

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def _run_regime(label: str, a: float, n_steps_unit: float, n_paths: int) -> dict:
    lam, eta_ou, rho, b = cfg.EXP8_LAMBDA, cfg.EXP8_ETA_OU, cfg.EXP8_RHO, cfg.EXP8_B
    T = cfg.EXP8_HORIZON
    n_steps = max(int(n_steps_unit * T), 10)
    y0 = np.array([cfg.EXP8_X0, cfg.EXP8_Z0])

    model = absorbed_factor_model(lam=lam, eta_ou=eta_ou, rho=rho, a=a, b=b)

    # Original measure P: X_t = x + W^0_{t and tau0}, S_t as above with
    # zero-continuation applied explicitly after the absorption time tau0.
    sim_p = simulate_diffusion(
        drift=model["drift"], diffusion=model["diffusion"], y0=y0, T=T, n_steps=n_steps,
        n_paths=n_paths, dim=2, dim_w=2, seed=cfg.RNG_SEED,
        domain_indicator=model["domain_indicator"], positivity_floor=None, theta=model["theta"],
    )
    S_T_raw = np.exp(sim_p.log_S[:, -1])
    S_T = np.where(sim_p.exited, 0.0, S_T_raw)  # S_t := 0 for t >= tau0 (Theorem 6.3)
    e_s_t = float(S_T.mean())
    se_s_t = float(S_T.std(ddof=1) / np.sqrt(n_paths))
    absorbed_fraction = float(sim_p.exited.mean())

    # Shifted system (6.10) under Qtilde: dX = c(Z)/X dt + dW~^0,
    # dZ = (-lam Z + eta_ou*rho*c(Z)/X) dt + eta_ou dW~^1. True martingale
    # iff 0 is inaccessible under Qtilde, i.e. iff inf c >= 1/2 (effective
    # Bessel dimension 1+2 inf c >= 2).
    sim_q = simulate_diffusion(
        drift=model["shifted_drift"], diffusion=model["diffusion"], y0=y0, T=T, n_steps=n_steps,
        n_paths=n_paths, dim=2, dim_w=2, seed=cfg.RNG_SEED + 1,
        domain_indicator=model["domain_indicator"], positivity_floor=None,
    )
    survival_prob = float((~sim_q.exited).mean())

    martingale_predicted = bool(a >= 0.5 - 1e-9)
    return {
        "label": label,
        "a": a,
        "b": b,
        "inf_c": a,
        "martingale_predicted": martingale_predicted,
        "E_S_T": e_s_t,
        "se_E_S_T": se_s_t,
        "absorbed_fraction": absorbed_fraction,
        "survival_prob_shifted": survival_prob,
    }


def run(full: bool) -> dict:
    n_steps_unit = cfg.N_STEPS_PER_UNIT_TIME_FULL if full else cfg.N_STEPS_PER_UNIT_TIME_RUN
    n_paths = cfg.N_PATHS_FULL if full else cfg.N_PATHS_RUN

    records = [
        _run_regime("martingale", cfg.EXP8_A_MARTINGALE, n_steps_unit, n_paths),
        _run_regime("strict_local", cfg.EXP8_A_STRICT, n_steps_unit, n_paths),
    ]

    def _agrees(predicts_martingale: bool, e_s_t: float) -> bool:
        return e_s_t > 0.9 if predicts_martingale else e_s_t < 0.98

    threshold_confirmed = all(_agrees(r["martingale_predicted"], r["E_S_T"]) for r in records)
    # Lemma 2.1 cross-check: E_P[S_T] (with zero-continuation) should agree
    # with the Qtilde survival probability of the shifted system (6.10).
    lemma_2_1_agrees = all(
        abs(r["E_S_T"] - r["survival_prob_shifted"]) < 0.1 for r in records
    )

    summary = {
        "records": records,
        "threshold_confirmed": bool(threshold_confirmed),
        "lemma_2_1_agrees": bool(lemma_2_1_agrees),
        "n_paths": n_paths,
        "full": full,
    }
    return summary


def save(summary: dict) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "exp8_results.json", "w") as f:
        json.dump(to_jsonable(summary), f, indent=2)

    records = summary["records"]
    labels = [f"{r['label']}\ninf c={r['inf_c']:.2f}" for r in records]
    vals = [r["E_S_T"] for r in records]
    colors = ["tab:green" if r["martingale_predicted"] else "tab:red" for r in records]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(labels, vals, color=colors)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="reference: E[S_T]=1")
    ax.set_ylabel("E[S_T] (zero-continuation after absorption)")
    ax.set_title("Experiment 8: absorbed factor (Theorem 6.3): martingale iff inf c>=1/2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp8_absorbed_factor.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="run the paper-scale configuration")
    args = parser.parse_args()

    summary = run(args.full)
    save(summary)

    print(f"Experiment 8 ({'full' if args.full else 'reduced'} configuration)")
    for r in summary["records"]:
        print(
            f"  [{r['label']}] inf c={r['inf_c']:.2f}: E[S_T]={r['E_S_T']:.4f}, "
            f"absorbed_fraction={r['absorbed_fraction']:.4f}, "
            f"survival_prob_shifted={r['survival_prob_shifted']:.4f}, "
            f"martingale_predicted={r['martingale_predicted']}"
        )
    print(f"  threshold_confirmed: {summary['threshold_confirmed']}")
    print(f"  lemma_2_1_agrees:    {summary['lemma_2_1_agrees']}")


if __name__ == "__main__":
    main()
