# AGENTS.md

Guide for coding agents and contributors working in this repository.

## What this repository implements

A numerical replication of ten Monte Carlo experiments verifying the
martingale criteria of Bayraktar, E. and Xing, H., *Necessary and Sufficient
Conditions for the Martingale Property of Stochastic Exponentials Driven by
Multi-Dimensional Diffusions*: the loss-of-mass identity and dichotomy
(Lemma 2.1, Proposition 2.1), the Fichera drift test at smooth boundaries
(Theorem 3.1) and its critical logarithmic extension (Theorem 3.2), the
radial Khasminskii/Feller tests for explosion at infinity (Propositions
4.1–4.3), the weighted-distance and projection tests at singular boundary
strata (Propositions 5.1–5.2), the paper's own worked examples (3/2, 4/2,
power-volatility, anisotropic, two-factor inverse-CIR, absorbed-factor
models, Section 6), consistency with affine and Wishart boundary theory
(Section 7), and the uniqueness of the associated linear-growth PDE
(Appendix B, Theorem B.1). Every experiment simulates diffusion sample paths
directly; no market or other external data is used anywhere.

## Layout

```
src/     core formulas: the Euler-Maruyama-driven stochastic exponential
         (src/stochastic_exponential.py), the paper's model coefficients
         (src/models.py), the Fichera drift / condition (T) / Feller /
         weighted-distance criteria (src/martingale_criteria.py), the
         Monte Carlo verification harness (src/mc_verification.py), the
         Wishart boundary test (src/wishart.py), the PDE-uniqueness
         comparison (src/pde_uniqueness.py), result I/O helpers
         (src/io_utils.py) and all experiment parameters
         (src/config.py)
data/    generate_synthetic.py: the shared Euler-Maruyama path generator
         (original and Girsanov-shifted diffusions, with domain-exit /
         explosion tracking) used by every experiment; no market data is
         loaded anywhere in this repository
exp/     one script per experiment, exp1.py .. exp10.py, run as
         `python -m exp.expN` (add `--full` for the paper-scale Monte
         Carlo configuration)
tests/   unit tests of src/ and data/ on small synthetic configurations
results/ metrics (expN_results.json), plots (expN_*.png) and RESULTS.md
```

## Commands

Setup:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

No API keys are needed: every experiment's data requirement is "No external
data" (Monte Carlo simulation only).

Tests:

```bash
python -m pytest -m "not market_data"
```

(there are no `market_data`-marked tests in this repository, since no market
data is used; the flag is kept for consistency with other replications in
this family).

Experiments (reduced Monte Carlo configuration by default; add `--full` for
the paper-scale configuration from `src/config.py`):

```bash
python -m exp.exp1    # Lemma 2.1 identity and Proposition 2.1 dichotomy
python -m exp.exp2    # Theorem 3.1 Fichera drift and condition (T)
python -m exp.exp3    # Propositions 4.1-4.2 explosion at infinity
python -m exp.exp4    # Theorem 3.2 / Example 3.2 critical logarithmic boundary
python -m exp.exp5    # Propositions 5.1-5.2 weighted-distance criterion
python -m exp.exp6    # Example 6.2 anisotropic model and Proposition 4.3
python -m exp.exp7    # Example 6.5 / Theorem 6.2 4/2 model
python -m exp.exp8    # Theorem 6.3 absorbed factor
python -m exp.exp9    # Section 7 affine and Wishart consistency
python -m exp.exp10   # Appendix B Theorem B.1 PDE uniqueness
```

`reproduce.sh` runs the full test suite followed by all ten experiments in
their reduced (default) configuration.

## Conventions

- All experiment parameters live in `src/config.py` as module-level
  `UPPER_CASE` variables, grouped by experiment, each with a one-line
  comment giving its meaning, unit and source (a paper section/equation, or
  "not specified in the paper"); there are no parameter classes or dicts.
- Model coefficients (drift, diffusion, integrand) for each of the paper's
  Section 6 examples live in `src/models.py` as plain functions, shared by
  every experiment that uses that model; experiment scripts in `exp/` only
  call `src/` and `data/generate_synthetic.py` functions and must not
  reimplement simulation or evaluation logic.
- `data/generate_synthetic.py` is the only place that draws Brownian
  increments; it implements the Euler-Maruyama scheme (with a
  full-truncation floor for square-root-type factors) for both the original
  diffusion under `P` and the Girsanov-shifted diffusion under `Q̃`, and
  tracks each path's exit/explosion time from its stated domain.
- Every experiment writes its metrics to `results/expN_results.json` and its
  plot(s) to `results/expN_*.png`; rerunning an experiment command
  overwrites these files. The numbers and verdicts in `results/RESULTS.md`
  are not regenerated automatically: after a rerun, check the new files and
  update `RESULTS.md` by hand.
- No API keys or secrets are used by this repository (no external data of
  any kind); nothing is read from the environment.
