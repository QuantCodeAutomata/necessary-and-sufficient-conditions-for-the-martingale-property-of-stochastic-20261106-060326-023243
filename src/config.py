"""Experiment parameters for the replication of Bayraktar and Xing,
"Necessary and Sufficient Conditions for the Martingale Property of
Stochastic Exponentials Driven by Multi-Dimensional Diffusions".

No external market data is used anywhere in this repository: every
experiment verifies an analytic criterion of the paper by Monte Carlo
simulation of the diffusions defined in the paper itself (Section 6 and
the examples cited below). Parameters not given a numeric value by the
paper are chosen so that the two branches of each dichotomy (true
martingale / strict local martingale) are both exercised; this choice
is noted as "not specified in the paper" below.

All experiments share the simulation engine in data/generate_synthetic.py
and the formulas in src/models.py, src/stochastic_exponential.py and
src/martingale_criteria.py.
"""

# --------------------------------------------------------------------
# Common Monte Carlo settings (not specified in the paper; the paper
# works at a theoretical level and gives no simulation parameters)
# --------------------------------------------------------------------
RNG_SEED = 20240521            # base seed for reproducibility, not specified in the paper
N_STEPS_PER_UNIT_TIME_RUN = 200   # Euler steps per unit of time, reduced configuration
N_STEPS_PER_UNIT_TIME_FULL = 1000  # Euler steps per unit of time, paper-scale configuration
N_PATHS_RUN = 20_000            # Monte Carlo paths, reduced configuration
N_PATHS_FULL = 200_000          # Monte Carlo paths, paper-scale configuration

# --------------------------------------------------------------------
# Experiment 1: Lemma 2.1 loss-of-mass identity and Proposition 2.1
# dichotomy, two-dimensional independent CIR factors (Section 2, p.7-8)
# --------------------------------------------------------------------
EXP1_ALPHA = 0.6        # CIR level alpha_i, chosen so 2*alpha>=eta^2 (P itself satisfies Feller)
EXP1_KAPPA = 1.5        # CIR mean-reversion speed kappa_i, not specified in the paper
EXP1_ETA = 1.0          # CIR volatility eta_i, not specified in the paper
EXP1_Y0 = (1.0, 1.0)    # starting point y, not specified in the paper
# shifted Feller parameter: 2*(alpha+eta*c) vs eta^2, i.e. c vs 0.5*eta^2/eta - alpha/eta = -0.1
EXP1_THETA_COEF_MARTINGALE = 0.4   # alpha+eta*c=1.0 >= 0.5*eta^2=0.5 (true martingale)
EXP1_THETA_COEF_STRICT = -0.5      # alpha+eta*c=0.1 < 0.5*eta^2=0.5 (strict local martingale)
EXP1_HORIZONS = (0.5, 1.0, 2.0)    # time horizons T tested for the dichotomy (Prop. 2.1(ii))

# --------------------------------------------------------------------
# Experiment 2: Theorem 3.1 Fichera-drift criterion and condition (T),
# Example 6.3 (3/2 model) and Example 6.4 (two-factor inverse-CIR)
# --------------------------------------------------------------------
EXP2_ALPHA = 0.75       # CIR level alpha for the 3/2 model, chosen with 2*alpha >= eta^2
EXP2_KAPPA = 1.0        # CIR speed kappa
EXP2_ETA = 1.0          # CIR volatility eta
EXP2_RHO = -0.5         # correlation rho between the asset and the CIR factor
EXP2_Y0 = 1.0           # starting value of the CIR factor
# threshold: alpha + rho*eta >= 0.5*eta^2  <=>  alpha >= 0.5*eta^2 - rho*eta
EXP2_ALPHA_MARTINGALE = 0.9   # alpha + rho*eta - 0.5*eta^2 = 0.9 - 0.5 - 0.5 = -0.1 -> adjust below
# Two regimes are built explicitly in exp/exp2.py from EXP2_RHO, EXP2_ETA via
# alpha_above = 0.5*eta^2 - rho*eta + 0.2 (true martingale) and
# alpha_below = 0.5*eta^2 - rho*eta - 0.2 (strict local martingale), both >0.
EXP2_MARGIN = 0.2             # distance from the critical alpha used for the two regimes

EXP2_TWO_FACTOR_ALPHA1 = 0.6   # alpha_1 for Example 6.4, 2*alpha_1 >= eta_1^2
EXP2_TWO_FACTOR_ALPHA2 = 0.6   # alpha_2 for Example 6.4, 2*alpha_2 >= eta_2^2
EXP2_TWO_FACTOR_KAPPA1 = 1.0
EXP2_TWO_FACTOR_KAPPA2 = 1.0
EXP2_TWO_FACTOR_ETA1 = 1.0
EXP2_TWO_FACTOR_ETA2 = 1.0
EXP2_TWO_FACTOR_RHO_STRICT = -0.6     # correlation rho<0 so (rho)^-=-rho>0 drives the threshold
EXP2_TWO_FACTOR_RHO_MARTINGALE = 0.6  # rho>0 gives (rho)^-=0, trivially satisfying the criterion
EXP2_TWO_FACTOR_G_BAR = 1.5     # sup g, with g(y2) = g_bar*y2/(1+y2) bounded by g_bar
EXP2_TWO_FACTOR_Y0 = (1.0, 1.0)
EXP2_HORIZON = 1.0

# --------------------------------------------------------------------
# Experiment 3: Propositions 4.1-4.2, isotropic power-volatility model
# (Example 6.1), explosion at infinity (Section 4, p.18-19, 27-28)
# --------------------------------------------------------------------
EXP3_DIM = 2            # dimension d >= 2 of the factor Y
EXP3_KAPPA = 1.0         # mean-reversion speed kappa
EXP3_ETA = 1.0           # volatility eta
EXP3_Y0 = (1.0, 0.5)     # starting point y
EXP3_HORIZON = 1.0
# four regimes of (rho, p): martingale iff rho<=0 or p<=1
EXP3_REGIMES = (
    (0.7, 2.0),   # rho>0, p>1  -> strict local martingale (explosion)
    (-0.7, 2.0),  # rho<=0      -> true martingale
    (0.7, 0.8),   # p<=1        -> true martingale
    (0.0, 2.0),   # rho=0       -> true martingale (boundary case rho<=0)
)

# --------------------------------------------------------------------
# Experiment 4: Theorem 3.2 / Example 3.2, critical CIR with logarithmic
# boundary correction (Section 3.7, p.15-17)
# --------------------------------------------------------------------
EXP4_Y0 = 0.2             # starting value of the critical CIR factor X0 in (0, e^-2]... y0 chosen in (0,1)
EXP4_HORIZON = 1.0
EXP4_C_VALUES = (0.2, 0.5, 0.8)   # scalar c in Example 3.2: martingale iff c<=1/2
EXP4_KAPPA_CUTOFF = 2.0           # cutoff scale for theta/h beyond y=e^-2, not specified in the paper

# coupled Theorem 3.2 case: c(z) = a + b*z^2/(1+z^2), threshold a+b<=1/2
EXP4_COUPLED_A = 0.3
EXP4_COUPLED_B_MARTINGALE = 0.1    # a+b = 0.40 <= 0.5  -> true martingale (away from the critical value for faster MC convergence)
EXP4_COUPLED_B_STRICT = 0.4        # a+b = 0.70 > 0.5   -> strict local martingale
EXP4_COUPLED_LAMBDA = 1.0          # OU mean reversion lambda_1
EXP4_COUPLED_ETA = 0.5             # OU volatility eta_1
EXP4_COUPLED_RHO = 0.3             # correlation between W^0 and W^1
EXP4_COUPLED_KAPPA = 1.0           # CIR mean-reversion speed kappa of X
EXP4_COUPLED_Z0 = 0.0

# --------------------------------------------------------------------
# Experiment 5: Proposition 5.2 weighted-distance criterion, Example 5.2
# (Section 5, p.23-26)
# --------------------------------------------------------------------
EXP5_M = 3                      # ambient dimension m of the normal Brownian motion U
EXP5_K = 1                      # number of tangential OU factors
EXP5_LAMBDA = 1.0                # OU mean-reversion speed
EXP5_ETA_OU = 0.5                 # OU volatility
EXP5_M_DIAG = (2.0, 1.0, 1.0)      # M = diag(2,1,1) of Example 5.2
EXP5_C_BASE = 0.2                 # c(z) = C_BASE + C_SPAN * z^2/(1+z^2)
EXP5_C_SPAN_MARTINGALE = 0.2      # sup c = 0.4 < (m-2)/2 = 0.5 -> true martingale
EXP5_C_SPAN_STRICT = 0.5          # sup c = 0.7 > 0.5 -> strict local martingale
EXP5_U0 = (1.0, 1.0, 1.0)         # starting point of U
EXP5_Z0 = 0.0
EXP5_HORIZON = 1.0

# --------------------------------------------------------------------
# Experiment 6: Example 6.2 anisotropic multi-factor model and
# Proposition 4.3 exact Feller test via compact sublevel sets (p.20,28)
# --------------------------------------------------------------------
EXP6_DIM = 2
EXP6_KAPPA = 1.0
EXP6_ETA = 1.0
EXP6_M_DIAG = (2.0, 1.0)          # invertible diagonal matrix M
EXP6_Q_BASE = 1.0                 # q(y) = Q_BASE + 0.5*sin(y1), bounded above/below away from 0
EXP6_OMEGA_SKEW = 0.8              # skew-symmetric rotation strength
EXP6_Y0 = (1.0, 0.5)
EXP6_HORIZON = 1.0
EXP6_REGIMES = (
    (0.6, 2.0),    # rho>0, gamma>1 -> strict local martingale
    (-0.6, 2.0),   # rho<=0          -> true martingale
    (0.6, 0.8),    # gamma<=1        -> true martingale
)

# --------------------------------------------------------------------
# Experiment 7: Example 6.5 (4/2 model) and Theorem 6.2 multi-factor
# extension (p.29-32)
# --------------------------------------------------------------------
EXP7_ALPHA = 0.75      # single-factor alpha, 2*alpha>=eta^2
EXP7_KAPPA = 1.0
EXP7_ETA = 1.0
EXP7_RHO = -0.5
EXP7_A_HESTON = 0.3    # Heston part a (should not affect the threshold, Corollary 3.1)
EXP7_B_MARGIN = 0.2    # alpha + rho*eta*b vs 0.5*eta^2: solved for b in exp script
EXP7_V0 = 1.0
EXP7_HORIZON = 1.0

# multi-factor Theorem 6.2 / Example 6.6
EXP7_MULTI_ETA1 = 1.0
EXP7_MULTI_ETA2 = 1.0
EXP7_MULTI_KAPPA1 = 1.0
EXP7_MULTI_KAPPA2 = 1.0
EXP7_MULTI_ALPHA2 = 1.0
EXP7_MULTI_RHO1 = -0.5        # correlation between B and Z^1
EXP7_MULTI_RHO2 = -0.3        # correlation between B and Z^2 (does not affect the threshold)
EXP7_MULTI_ZZ_CORR = 0.6      # correlation between Z^1 and Z^2
EXP7_MULTI_V = 1.0
EXP7_MULTI_U = 0.0
EXP7_MULTI_ALPHA1_MARTINGALE = 1.5   # Example 6.6 critical threshold alpha_1 >= 3/2
EXP7_MULTI_ALPHA1_STRICT = 1.0        # 1/2 <= alpha_1 < 3/2 -> strict local martingale
EXP7_MULTI_Y0 = (1.0, 1.0)
EXP7_MULTI_HORIZON = 1.0

# --------------------------------------------------------------------
# Experiment 8: Theorem 6.3 absorbed factor with correlated modulation
# (Section 6.2, p.32-34)
# --------------------------------------------------------------------
EXP8_X0 = 1.0
EXP8_LAMBDA = 1.0
EXP8_ETA_OU = 0.5
EXP8_RHO = 0.5          # correlation rho_1 between W^0 and W^1
EXP8_A_MARTINGALE = 0.7   # c(z) = a + b/(1+z^2), inf c = a >= 1/2 -> true martingale
EXP8_A_STRICT = 0.3       # a < 1/2 -> strict local martingale
EXP8_B = 0.2
EXP8_Z0 = 0.0
EXP8_HORIZON = 1.0

# --------------------------------------------------------------------
# Experiment 9: Section 7, canonical affine diffusions and Wishart
# processes (p.34-36)
# --------------------------------------------------------------------
EXP9_AFFINE_ALPHA_I = 0.6     # CIR-type face coefficient (alpha_i)_ii, 2*b0_i >= (alpha_i)_ii
EXP9_AFFINE_B0 = 0.5          # b0_i, chosen to satisfy the admissibility condition (7.1)
EXP9_AFFINE_B = 0.3           # off-diagonal B_{ij} >= 0 coupling term
EXP9_AFFINE_Y0 = (1.0, 1.0)
EXP9_AFFINE_HORIZON = 1.0

EXP9_WISHART_N = 2            # size n of the Wishart matrix
EXP9_WISHART_Q = 1.0          # scalar diffusion coefficient Q = q*I_n
EXP9_WISHART_M = 0.5          # mean-reversion M = -m*I_n (m>0)
EXP9_WISHART_OMEGA_MARTINGALE = 2.4   # Omega Omega^T = omega^2 I_n with omega^2 >= (n+1)*Q^2
EXP9_WISHART_OMEGA_STRICT = 1.2       # omega^2 < (n+1)*Q^2
EXP9_WISHART_SIGMA0_DIAG = (1.0, 1.0)
EXP9_WISHART_HORIZON = 1.0

# --------------------------------------------------------------------
# Experiment 10: Appendix B, Theorem B.1 uniqueness in the linear-growth
# class; 4/2 model and two-factor inverse-CIR as test cases (p.39-40)
# --------------------------------------------------------------------
EXP10_ALPHA_MARTINGALE = 0.95   # alpha + rho*eta*b >= 0.5*eta^2  (reuses EXP7 4/2 coefficients)
EXP10_ALPHA_STRICT = 0.55
EXP10_ETA = 1.0
EXP10_KAPPA = 1.0
EXP10_RHO = -0.5
EXP10_B = 0.5
EXP10_V0 = 1.0
EXP10_S0 = 1.0
EXP10_HORIZON = 1.0
EXP10_PDE_Y_GRID_POINTS = 400     # spatial grid points for the Doob-transformed PDE, not specified in the paper
EXP10_PDE_Y_MAX = 8.0             # truncated domain for the factor variable
EXP10_PDE_N_TIME_STEPS = 2000     # time steps for the finite-difference scheme
