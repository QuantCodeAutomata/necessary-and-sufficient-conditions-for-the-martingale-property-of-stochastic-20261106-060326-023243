# Necessary and Sufficient Conditions for the Martingale Property of Stochastic Exponentials Driven by Multi-Dimensional Diffusions

Replication of the numerical experiments accompanying Bayraktar, E. and Xing,
H., *Necessary and Sufficient Conditions for the Martingale Property of
Stochastic Exponentials Driven by Multi-Dimensional Diffusions*.

The paper gives purely analytic, measure-theoretic criteria (Fichera drift,
Feller's test, weighted-distance / Khasminskii tests) for deciding whether a
stochastic exponential `S` is a true martingale or a strict local martingale.
No market data of any kind is used anywhere in this repository: every
experiment verifies one of these criteria by Monte Carlo simulation of the
multi-dimensional diffusions defined in the paper's own examples (Section 6)
and compares the simulated loss of mass with the criterion's analytic
prediction.

## Layout

```
src/    core formulas: the Euler-Maruyama-driven stochastic exponential,
        the paper's model coefficients (Section 6 examples), the Fichera
        drift / Feller / weighted-distance criteria, the Wishart boundary
        test, and the PDE-uniqueness comparison (Appendix B)
data/   generate_synthetic.py: the shared Euler-Maruyama path generator used
        by every experiment to produce its simulated diffusion paths
exp/    one script per experiment, exp1.py .. exp10.py
tests/  unit tests of src/ and data/ on small synthetic configurations
results/ saved metrics, tables and plots, and RESULTS.md
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

No API keys are required: the paper's experiments use no external data.

## Tests

```bash
python -m pytest
```

## Running the experiments

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

Each script writes its metrics, and any plot, directly into `results/`.
`reproduce.sh` runs the full test suite followed by every experiment.

See `results/RESULTS.md` for the comparison against the paper's qualitative
and quantitative claims, deviations from the paper, and what is not
reproduced.
