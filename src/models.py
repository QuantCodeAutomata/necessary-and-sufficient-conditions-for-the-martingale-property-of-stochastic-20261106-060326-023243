"""Coefficient functions (drift mu, diffusion sigma, integrand theta, and
shifted drift mu_tilde = mu + sigma theta) for the diffusion models used
in Sections 2, 3, 4, 6 of Bayraktar and Xing. Each factory function
returns a dict of callables with the common signature f(y, t) -> array,
consumed by data/generate_synthetic.py and src/martingale_criteria.py.

Custom implementation of the paper's own model definitions (Examples
6.1-6.6 and Theorems 3.2, 6.2, 6.3): these are not standard models
offered by arch, hmmlearn or any other researched library.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------
# Experiment 1: two independent CIR factors, theta linking them (Section
# 2, Lemma 2.1 and Proposition 2.1 verification)
# ---------------------------------------------------------------------
def cir_2d_model(alpha: float, kappa: float, eta: float, theta_coef: float):
    """dY^i = (alpha - kappa Y^i) dt + eta sqrt(Y^i) dZ^i, i=1,2, independent
    factors on D = (0, infinity)^2. theta(y) = theta_coef/sqrt(y_i) along
    each factor (3/2-type integrand, unbounded at the boundary so that the
    Girsanov shift genuinely changes the Fichera drift there, Example 6.3):
    sigma_i theta_i = eta*theta_coef is constant, giving the affine shifted
    drift mu_tilde_i(y) = alpha + eta*theta_coef - kappa*y_i, with Feller
    accessibility of 0 governed by 2*(alpha + eta*theta_coef) vs eta^2.
    """
    def drift(y, t):
        return alpha - kappa * y

    def diffusion(y, t):
        n = y.shape[0]
        sig = np.zeros((n, 2, 2))
        sig[:, 0, 0] = eta * np.sqrt(np.maximum(y[:, 0], 0.0))
        sig[:, 1, 1] = eta * np.sqrt(np.maximum(y[:, 1], 0.0))
        return sig

    def theta(y, t):
        y_safe = np.maximum(y, 1e-8)
        return theta_coef / np.sqrt(y_safe)

    def domain_indicator(y):
        return (y[:, 0] > 1e-10) & (y[:, 1] > 1e-10)

    def shifted_drift(y, t):
        return alpha + eta * theta_coef - kappa * y

    return dict(drift=drift, diffusion=diffusion, theta=theta,
                domain_indicator=domain_indicator, shifted_drift=shifted_drift)


# ---------------------------------------------------------------------
# Experiment 2a: the 3/2 model recovered from the Fichera drift (Example
# 6.3). Y is a CIR factor, theta(y) = y^{-1/2}(rho, sqrt(1-rho^2)).
# ---------------------------------------------------------------------
def model_32(alpha: float, kappa: float, eta: float, rho: float):
    def drift(y, t):
        return alpha - kappa * y

    def diffusion(y, t):
        n = y.shape[0]
        sig = np.zeros((n, 1, 2))
        sig[:, 0, 0] = eta * np.sqrt(np.maximum(y[:, 0], 0.0))
        return sig

    def theta(y, t):
        y0 = np.maximum(y[:, 0], 1e-8)
        inv_sqrt = 1.0 / np.sqrt(y0)
        out = np.zeros((y.shape[0], 2))
        out[:, 0] = inv_sqrt * rho
        out[:, 1] = inv_sqrt * np.sqrt(max(1.0 - rho ** 2, 0.0))
        return out

    def domain_indicator(y):
        return y[:, 0] > 1e-10

    def shifted_drift(y, t):
        # sigma*theta = rho*eta (constant); mu_tilde(y) = alpha + rho*eta - kappa*y
        return alpha + rho * eta - kappa * y

    return dict(drift=drift, diffusion=diffusion, theta=theta,
                domain_indicator=domain_indicator, shifted_drift=shifted_drift)


# ---------------------------------------------------------------------
# Experiment 2b: two-factor inverse-CIR model (Example 6.4)
# ---------------------------------------------------------------------
def two_factor_inverse_cir(alpha1, kappa1, eta1, alpha2, kappa2, eta2, rho, g_bar):
    """dY^i = (alpha_i - kappa_i Y^i)dt + eta_i sqrt(Y^i) dZ^i, i=1,2,
    dS = S (Y^1)^{-1/2} g(Y^2) dB, B = rho Z^1 + sqrt(1-rho^2) B_perp.
    g(y2) = g_bar * y2 / (1+y2) is continuous, bounded by g_bar, positive.
    """
    def drift(y, t):
        out = np.zeros_like(y)
        out[:, 0] = alpha1 - kappa1 * y[:, 0]
        out[:, 1] = alpha2 - kappa2 * y[:, 1]
        return out

    def diffusion(y, t):
        n = y.shape[0]
        sig = np.zeros((n, 2, 3))  # driving (Z1, Z2, B_perp)
        sig[:, 0, 0] = eta1 * np.sqrt(np.maximum(y[:, 0], 0.0))
        sig[:, 1, 1] = eta2 * np.sqrt(np.maximum(y[:, 1], 0.0))
        return sig

    def g(y2):
        return g_bar * y2 / (1.0 + y2)

    def theta(y, t):
        y1 = np.maximum(y[:, 0], 1e-8)
        out = np.zeros((y.shape[0], 3))
        coef = g(y[:, 1]) / np.sqrt(y1)
        out[:, 0] = rho * coef
        out[:, 2] = np.sqrt(max(1.0 - rho ** 2, 0.0)) * coef
        return out

    def domain_indicator(y):
        return (y[:, 0] > 1e-10) & (y[:, 1] > 1e-10)

    def shifted_drift(y, t):
        out = np.zeros_like(y)
        out[:, 0] = alpha1 + rho * eta1 * g(y[:, 1]) - kappa1 * y[:, 0]
        out[:, 1] = alpha2 - kappa2 * y[:, 1]
        return out

    return dict(drift=drift, diffusion=diffusion, theta=theta,
                domain_indicator=domain_indicator, shifted_drift=shifted_drift, g=g)


# ---------------------------------------------------------------------
# Experiment 3 & 6: isotropic / anisotropic multi-factor power-volatility
# models (Examples 6.1 and 6.2)
# ---------------------------------------------------------------------
def power_volatility_isotropic(dim, kappa, eta, p, rho):
    """D = R^d, dY = -kappa Y dt + eta dZ, dS = S|Y|^p dB,
    dB = rho <Y/|Y|, dZ> + sqrt(1-rho^2) dB_perp."""
    def drift(y, t):
        return -kappa * y

    def diffusion(y, t):
        n = y.shape[0]
        sig = np.zeros((n, dim, dim + 1))
        sig[:, :, :dim] = eta * np.eye(dim)
        return sig

    def theta(y, t):
        norm = np.maximum(np.linalg.norm(y, axis=1), 1e-10)
        p_norm = norm ** p
        out = np.zeros((y.shape[0], dim + 1))
        out[:, :dim] = rho * (p_norm / norm)[:, None] * y
        out[:, dim] = np.sqrt(max(1.0 - rho ** 2, 0.0)) * p_norm
        return out

    def domain_indicator(y):
        return np.ones(y.shape[0], dtype=bool)  # D = R^d, never exits

    def shifted_drift(y, t):
        norm = np.maximum(np.linalg.norm(y, axis=1), 1e-10)
        return (-kappa + rho * eta * norm ** (p - 1))[:, None] * y

    return dict(drift=drift, diffusion=diffusion, theta=theta,
                domain_indicator=domain_indicator, shifted_drift=shifted_drift)


def power_volatility_anisotropic(dim, kappa, eta, gamma, rho, M_diag, q_base, omega_skew):
    """Example 6.2: dY = q(Y) M[-kappa z + Omega(z) z] dt + eta sqrt(q(Y)) M dW,
    z = M^{-1} y, Omega skew-symmetric, q(y) = q_base + 0.5*sin(y_1) > 0.
    P(y) = |M^{-1} y|^2 / 2 reduces to the isotropic radial test of Example 6.1
    with the same A(r), B(r) (paper Section 6, Example 6.2).
    """
    M = np.diag(M_diag)
    M_inv = np.diag(1.0 / np.array(M_diag))

    def q(y):
        return q_base + 0.5 * np.sin(y[:, 0])

    def omega_matrix(z):
        # 2D skew-symmetric rotation generator scaled by |z|^2 (tangential,
        # does not affect P(y) = |z|^2/2 since z^T Omega(z) z = 0)
        n = z.shape[0]
        Om = np.zeros((n, dim, dim))
        if dim == 2:
            norm2 = np.einsum("pd,pd->p", z, z)
            Om[:, 0, 1] = omega_skew * norm2
            Om[:, 1, 0] = -omega_skew * norm2
        return Om

    def drift(y, t):
        z = (M_inv @ y.T).T
        Om = omega_matrix(z)
        rotated = np.einsum("pij,pj->pi", Om, z)
        lin = -kappa * z + rotated
        qy = q(y)
        return qy[:, None] * (M @ lin.T).T

    def diffusion(y, t):
        n = y.shape[0]
        qy = q(y)
        sig = np.zeros((n, dim, dim + 1))
        base = eta * np.sqrt(np.maximum(qy, 1e-10))
        sig[:, :, :dim] = base[:, None, None] * M[None, :, :]
        return sig

    def theta(y, t):
        z = (M_inv @ y.T).T
        norm_z = np.maximum(np.linalg.norm(z, axis=1), 1e-10)
        p_norm = norm_z ** gamma
        qy = np.maximum(q(y), 1e-10)
        out = np.zeros((y.shape[0], dim + 1))
        tang = rho * (p_norm / norm_z)[:, None] * z / np.sqrt(qy)[:, None]
        out[:, :dim] = tang
        out[:, dim] = np.sqrt(max(1.0 - rho ** 2, 0.0)) * p_norm / np.sqrt(qy)
        return out

    def domain_indicator(y):
        return np.ones(y.shape[0], dtype=bool)

    def shifted_drift(y, t):
        # mu_tilde = mu + sigma(y) theta(y). Here sigma(y) theta(y) =
        # eta*sqrt(q(y))*M @ (tangential/sqrt(q(y))) = eta*rho*|z|^(gamma-1)*(M z),
        # independent of q(y): the sqrt(q(y)) loading of the diffusion exactly
        # cancels the 1/sqrt(q(y)) built into theta's tangential part, so the
        # correction must NOT be multiplied again by q(y) outside (paper
        # Section 6, Example 6.2: the threshold is the isotropic one of
        # Example 6.1, unaffected by the time-change rate q).
        z = (M_inv @ y.T).T
        norm_z = np.maximum(np.linalg.norm(z, axis=1), 1e-10)
        correction = (rho * eta * norm_z ** (gamma - 1))[:, None] * (M @ z.T).T
        return drift(y, t) + correction

    return dict(drift=drift, diffusion=diffusion, theta=theta,
                domain_indicator=domain_indicator, shifted_drift=shifted_drift, M_inv=M_inv)


# ---------------------------------------------------------------------
# Experiment 4: critical CIR with logarithmic boundary correction
# (Example 3.2) and its coupled extension (Theorem 3.2)
# ---------------------------------------------------------------------
def critical_log_cir(c: float, y_cut: float = np.exp(-2)):
    """dY = 0.5 dt + sqrt(Y) dZ (critical CIR, kappa=0, alpha=1/2),
    theta(y) = -c/(sqrt(y) log(1/y)) for y <= y_cut, smoothly cut off
    above y_cut (paper Example 3.2)."""
    def h(y):
        y = np.asarray(y)
        out = np.zeros_like(y)
        small = y <= y_cut
        out[small] = 1.0 / np.log(1.0 / np.maximum(y[small], 1e-12))
        return out

    def drift(y, t):
        return np.full_like(y, 0.5)

    def diffusion(y, t):
        n = y.shape[0]
        sig = np.zeros((n, 1, 1))
        sig[:, 0, 0] = np.sqrt(np.maximum(y[:, 0], 0.0))
        return sig

    def theta(y, t):
        y0 = np.maximum(y[:, 0], 1e-10)
        out = np.zeros((y.shape[0], 1))
        out[:, 0] = -c * h(y0) / np.sqrt(y0)
        return out

    def domain_indicator(y):
        return y[:, 0] > 1e-12

    def shifted_drift(y, t):
        y0 = np.maximum(y[:, 0], 1e-10)
        out = np.zeros_like(y)
        out[:, 0] = 0.5 - c * h(y0)
        return out

    return dict(drift=drift, diffusion=diffusion, theta=theta,
                domain_indicator=domain_indicator, shifted_drift=shifted_drift, h=h)


def coupled_log_cir(a, b, lam, eta_ou, rho, kappa, y_cut=np.exp(-2)):
    """Theorem 3.2: dX = (0.5 - kappa X)dt + sqrt(X) dW^0,
    dZ = -lam Z dt + eta_ou dW^1, c(z) = a + b*z^2/(1+z^2),
    S = E(-int c(Z) h(X) X^{-1/2} dW^0)."""
    def h(x):
        x = np.asarray(x)
        out = np.zeros_like(x)
        small = x <= y_cut
        out[small] = 1.0 / np.log(1.0 / np.maximum(x[small], 1e-12))
        return out

    def c_fun(z):
        return a + b * z ** 2 / (1.0 + z ** 2)

    def drift(y, t):
        # y[:,0]=X, y[:,1]=Z
        out = np.zeros_like(y)
        out[:, 0] = 0.5 - kappa * y[:, 0]
        out[:, 1] = -lam * y[:, 1]
        return out

    def diffusion(y, t):
        n = y.shape[0]
        sig = np.zeros((n, 2, 2))  # driving (W0, W1)
        sig[:, 0, 0] = np.sqrt(np.maximum(y[:, 0], 0.0))
        sig[:, 1, 0] = rho * eta_ou
        sig[:, 1, 1] = eta_ou * np.sqrt(max(1.0 - rho ** 2, 0.0))
        return sig

    def theta(y, t):
        x = np.maximum(y[:, 0], 1e-10)
        out = np.zeros((y.shape[0], 2))
        out[:, 0] = -c_fun(y[:, 1]) * h(x) / np.sqrt(x)
        return out

    def domain_indicator(y):
        return y[:, 0] > 1e-12

    def shifted_drift(y, t):
        x = np.maximum(y[:, 0], 1e-10)
        out = np.zeros_like(y)
        out[:, 0] = 0.5 - kappa * y[:, 0] - c_fun(y[:, 1]) * h(x)
        out[:, 1] = -lam * y[:, 1] - eta_ou * rho * c_fun(y[:, 1]) * h(x) / np.sqrt(x)
        return out

    return dict(drift=drift, diffusion=diffusion, theta=theta,
                domain_indicator=domain_indicator, shifted_drift=shifted_drift,
                h=h, c_fun=c_fun)


# ---------------------------------------------------------------------
# Experiment 5: weighted-distance / inverse-distance singular integrand
# (Proposition 5.2, Example 5.2)
# ---------------------------------------------------------------------
def weighted_distance_model(m, k, lam, eta_ou, M_diag, c_base, c_span, exit_eps=0.05):
    """U_t = u + W_t in R^m, dZ^i = -lam Z^i dt + eta_ou dW^i (k OU
    factors), S = E(-int c(Z) U/|U|^2 . dW). Coordinates transformed by
    X = M(Z) U with M constant diagonal (Example 5.2, M=diag(2,1,1)).

    The radial part R=|U| of the shifted process is, by Ito's formula,
    a Bessel-type diffusion of effective dimension m-2c(Z) driven by
    *unit*-intensity noise (unlike the CIR factors elsewhere in this
    repository, whose diffusion coefficient vanishes at the boundary and
    therefore lets an Euler scheme approach an arbitrarily small floor).
    Resolving R down to machine-scale tolerances would require an
    impractically small step size, so `exit_eps` is the numerical
    surrogate for "R=0" used by `domain_indicator`: a finite but small
    radius, analogous to the floor used for CIR-type boundaries."""
    M = np.diag(M_diag)

    def c_fun(z):
        z0 = z[:, 0] if z.ndim == 2 else z
        return c_base + c_span * z0 ** 2 / (1.0 + z0 ** 2)

    dim = m + k

    def drift(y, t):
        out = np.zeros_like(y)
        out[:, m:] = -lam * y[:, m:]
        return out

    def diffusion(y, t):
        n = y.shape[0]
        sig = np.zeros((n, dim, dim))
        sig[:, :m, :m] = np.eye(m)
        sig[:, m:, m:] = eta_ou * np.eye(k)
        return sig

    def theta(y, t):
        u = y[:, :m]
        z = y[:, m:]
        norm2 = np.maximum(np.einsum("pd,pd->p", u, u), 1e-12)
        c_val = c_fun(z)
        out = np.zeros((y.shape[0], dim))
        out[:, :m] = -c_val[:, None] * u / norm2[:, None]
        return out

    def domain_indicator(y):
        u = y[:, :m]
        return np.linalg.norm(u, axis=1) > exit_eps

    def shifted_drift(y, t):
        u = y[:, :m]
        z = y[:, m:]
        norm2 = np.maximum(np.einsum("pd,pd->p", u, u), 1e-12)
        c_val = c_fun(z)
        out = np.zeros_like(y)
        out[:, :m] = -c_val[:, None] * u / norm2[:, None]
        out[:, m:] = -lam * z
        return out

    return dict(drift=drift, diffusion=diffusion, theta=theta,
                domain_indicator=domain_indicator, shifted_drift=shifted_drift,
                c_fun=c_fun, M=M, m=m, k=k)


# ---------------------------------------------------------------------
# Experiment 7: 4/2 model (Example 6.5) and its correlated multi-factor
# extension (Theorem 6.2 / Example 6.6)
# ---------------------------------------------------------------------
def model_4_2(alpha, kappa, eta, rho, a_heston, b_32):
    """dV = (alpha - kappa V)dt + eta sqrt(V) dZ,
    dS = S(a sqrt(V) + b/sqrt(V)) dB, d<B,Z> = rho dt."""
    def drift(y, t):
        return alpha - kappa * y

    def diffusion(y, t):
        n = y.shape[0]
        sig = np.zeros((n, 1, 2))
        sig[:, 0, 0] = eta * np.sqrt(np.maximum(y[:, 0], 0.0))
        return sig

    def theta(y, t):
        v = np.maximum(y[:, 0], 1e-10)
        coef = a_heston * np.sqrt(v) + b_32 / np.sqrt(v)
        out = np.zeros((y.shape[0], 2))
        out[:, 0] = coef * rho
        out[:, 1] = coef * np.sqrt(max(1.0 - rho ** 2, 0.0))
        return out

    def domain_indicator(y):
        return y[:, 0] > 1e-10

    def shifted_drift(y, t):
        return alpha + rho * eta * b_32 - (kappa - rho * eta * a_heston) * y

    return dict(drift=drift, diffusion=diffusion, theta=theta,
                domain_indicator=domain_indicator, shifted_drift=shifted_drift)


def model_4_2_multi(eta1, eta2, kappa1, kappa2, alpha1, alpha2, rho1, rho2,
                     zz_corr, u, v, g_bar_fun):
    """Theorem 6.2 / Example 6.6: dY^i = (alpha_i - kappa_i Y^i)dt + eta_i
    sqrt(Y^i) dZ^i, correlation matrix with d<Z1,Z2>=zz_corr dt,
    d<B,Z1>=rho1 dt, d<B,Z2>=rho2 dt; ell(y) = u*sqrt(y1) + v*g(y2)/sqrt(y1),
    S = E(int ell(Y) dB)."""
    def drift(y, t):
        out = np.zeros_like(y)
        out[:, 0] = alpha1 - kappa1 * y[:, 0]
        out[:, 1] = alpha2 - kappa2 * y[:, 1]
        return out

    def diffusion(y, t):
        n = y.shape[0]
        # 3 independent Brownian drivers (Z1, Z2, and the residual driver of
        # B that is independent of Z1, Z2); Y only loads on the first two.
        sig = np.zeros((n, 2, 3))
        sig[:, 0, 0] = eta1 * np.sqrt(np.maximum(y[:, 0], 0.0))
        sig[:, 1, 0] = eta2 * zz_corr * np.sqrt(np.maximum(y[:, 1], 0.0))
        sig[:, 1, 1] = eta2 * np.sqrt(max(1.0 - zz_corr ** 2, 0.0)) * np.sqrt(np.maximum(y[:, 1], 0.0))
        return sig

    def g(y2):
        return g_bar_fun(y2)

    def theta(y, t):
        y1 = np.maximum(y[:, 0], 1e-10)
        ell = u * np.sqrt(y1) + v * g(y[:, 1]) / np.sqrt(y1)
        # B correlated with Z1, Z2 at rho1, rho2 and a third independent driver
        rho_resid = max(1.0 - rho1 ** 2 - rho2 ** 2, 1e-10)
        out = np.zeros((y.shape[0], 3))
        out[:, 0] = ell * rho1
        out[:, 1] = ell * rho2
        out[:, 2] = ell * np.sqrt(rho_resid)
        return out

    def domain_indicator(y):
        return (y[:, 0] > 1e-10) & (y[:, 1] > 1e-10)

    def shifted_drift(y, t):
        # Girsanov shift mu_tilde = mu + sigma(y) theta(y) (paper eq. preceding
        # Lemma 2.1), applied to the full 2-factor system, not only to Y^1:
        # the correlation between Z^2 and (B, Z^1) also tilts Y^2's drift.
        #
        # (sigma theta)_1 = eta1*sqrt(y1)*rho1*ell(y) = eta1*rho1*(u*y1+v*g(y2));
        # the sqrt(y1) loading of Y^1 on its own driver Z^1 cancels the
        # 1/sqrt(y1) singularity of ell(y) = u*sqrt(y1)+v*g(y2)/sqrt(y1),
        # leaving the affine drift mu_tilde_1 = c(z) - k*x of the methodology
        # note (c(z)=alpha1+rho1*eta1*v*g(z), k=kappa1-rho1*eta1*u). Dropping
        # the sqrt(y1) factor here would wrongly keep an O(1/sqrt(y1))
        # singular pull that the model does not have.
        y1 = np.maximum(y[:, 0], 1e-10)
        ell = u * np.sqrt(y1) + v * g(y[:, 1]) / np.sqrt(y1)
        out = np.zeros_like(y)
        out[:, 0] = alpha1 + rho1 * eta1 * (u * y[:, 0] + v * g(y[:, 1])) - kappa1 * y[:, 0]
        cross2 = zz_corr * rho1 + np.sqrt(max(1.0 - zz_corr ** 2, 0.0)) * rho2
        out[:, 1] = alpha2 - kappa2 * y[:, 1] + eta2 * np.sqrt(np.maximum(y[:, 1], 0.0)) * cross2 * ell
        return out

    return dict(drift=drift, diffusion=diffusion, theta=theta,
                domain_indicator=domain_indicator, shifted_drift=shifted_drift, g=g)


# ---------------------------------------------------------------------
# Experiment 8: absorbed factor with correlated modulation (Theorem 6.3)
# ---------------------------------------------------------------------
def absorbed_factor_model(lam, eta_ou, rho, a, b):
    """X_t = x + W^0_{t and tau0} (absorbed Brownian motion), dZ = -lam Z dt
    + eta_ou dW^1, c(z) = a + b/(1+z^2), S_t = exp(int c(Z)/X dW^0 - 0.5
    int c(Z)^2/X^2 ds) for t<tau0, 0 afterwards."""
    def c_fun(z):
        return a + b / (1.0 + z ** 2)

    def drift(y, t):
        out = np.zeros_like(y)
        out[:, 1] = -lam * y[:, 1]
        return out

    def diffusion(y, t):
        n = y.shape[0]
        sig = np.zeros((n, 2, 2))
        sig[:, 0, 0] = 1.0
        sig[:, 1, 0] = rho * eta_ou
        sig[:, 1, 1] = eta_ou * np.sqrt(max(1.0 - rho ** 2, 0.0))
        return sig

    def theta(y, t):
        x = np.maximum(y[:, 0], 1e-10)
        out = np.zeros((y.shape[0], 2))
        out[:, 0] = c_fun(y[:, 1]) / x
        return out

    def domain_indicator(y):
        return y[:, 0] > 1e-10

    def shifted_drift(y, t):
        # Paper's shifted system (6.10): dX = c(Z)/X dt + dW~^0,
        # dZ^i = (-lam_i Z^i + eta_i rho_i c(Z)/X) dt + eta_i dW~^i (PLUS sign,
        # since sigma(y) theta(y) = (c/x, rho*eta_ou*c/x) with the correlation
        # row of sigma carrying a positive rho*eta_ou coefficient).
        x = np.maximum(y[:, 0], 1e-10)
        out = np.zeros_like(y)
        out[:, 0] = c_fun(y[:, 1]) / x
        out[:, 1] = -lam * y[:, 1] + eta_ou * rho * c_fun(y[:, 1]) / x
        return out

    return dict(drift=drift, diffusion=diffusion, theta=theta,
                domain_indicator=domain_indicator, shifted_drift=shifted_drift, c_fun=c_fun)


# ---------------------------------------------------------------------
# Experiment 9: canonical affine diffusion face and Wishart process
# ---------------------------------------------------------------------
def affine_canonical_2factor(alpha_i, b0, b_coupling):
    """D = (0,infinity) x R, a(y) = diag(alpha_i * y1, 0) (one CIR face
    coordinate, one unconstrained OU-like coordinate), b(y) = (b0 + B*y2,
    lambda*y2) with B = b_coupling >= 0 (admissibility (7.1): 2*b0 >=
    alpha_i). theta is bounded near the face {y1=0} (Corollary 3.1), so
    the Girsanov shift leaves the Fichera drift at the face unchanged."""
    def drift(y, t):
        out = np.zeros_like(y)
        out[:, 0] = b0 + b_coupling * y[:, 1]
        out[:, 1] = -0.5 * y[:, 1]
        return out

    def diffusion(y, t):
        n = y.shape[0]
        sig = np.zeros((n, 2, 2))
        sig[:, 0, 0] = np.sqrt(alpha_i * np.maximum(y[:, 0], 0.0))
        sig[:, 1, 1] = 0.3
        return sig

    def theta(y, t):
        # bounded near the face: theta(y) = (const, 0), Corollary 3.1 applies
        out = np.zeros((y.shape[0], 2))
        out[:, 0] = 0.2
        return out

    def domain_indicator(y):
        return y[:, 0] > 1e-10

    def shifted_drift(y, t):
        out = np.zeros_like(y)
        out[:, 0] = b0 + b_coupling * y[:, 1] + 0.2 * np.sqrt(alpha_i * np.maximum(y[:, 0], 1e-10))
        out[:, 1] = -0.5 * y[:, 1]
        return out

    return dict(drift=drift, diffusion=diffusion, theta=theta,
                domain_indicator=domain_indicator, shifted_drift=shifted_drift)
