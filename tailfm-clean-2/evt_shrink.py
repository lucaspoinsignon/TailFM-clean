"""Lower-variance GPD tail estimation for `tailfm.evt`, as a post-fit step.

Additive: nothing in evt.py changes.  `MarginalEnsemble.fit` is run as usual,
then `shrink_ensemble(marg, x)` replaces (xi_lo_, beta_lo_) and (xi_hi_,
beta_hi_) on every SemiParametricMarginal in place.  The empirical body
(_body_x, _body_p) is untouched -- only the two GPD branches of cdf/ppf move --
so the piecewise CDF stays continuous at the thresholds by construction.

Two changes, both aimed at the k ~ 62 regime:

1. ESTIMATOR.  scipy's genpareto MLE is replaced by Zhang & Stephens (2009),
   an empirical-Bayes profile in b = xi/beta.  Measured at k=62 over 3000
   replicates it removes the MLE's downward bias (-0.04 -> +0.01 at xi=0.267)
   and cuts sd by ~7%, for ~9% lower RMSE.  It also always returns a value:
   there is no optimiser to fail.

2. POOLING.  The f features are f parallel small-sample problems with a common
   structure, so xi_j is shrunk toward the cross-sectional mean by the
   James-Stein / empirical-Bayes weight

       s_j^2 = (1 + xi_j)^2 / k_j        (Smith 1987 asymptotic variance)
       tau^2 = max(0, Var_j(xi_hat) - mean_j(s_j^2))
       xi_j~ = mu + tau^2 / (tau^2 + s_j^2) * (xi_hat_j - mu).

   tau^2 is the heterogeneity that survives after subtracting estimation noise,
   so the procedure is self-limiting: if the features really do have different
   tail indices, lambda -> 1 and nothing is shrunk.  Measured at f=235, k=62,
   mu=0.267:

       tau_true   lambda_hat   RMSE(xi) MLE -> ZS+EB   RMSE(log q_0.999)
         0.00        0.03         0.178 -> 0.014         0.299 -> 0.161
         0.10        0.26         0.179 -> 0.089         0.304 -> 0.209
         0.20        0.59         0.181 -> 0.133         0.317 -> 0.261
         0.35        0.81         0.187 -> 0.158         0.349 -> 0.315

   beta is NOT shrunk: after xi is fixed it is re-estimated by the profile MLE
   (the same objective evt.fit_gpd already uses on its xi_max path), which keeps
   the pair (xi~, beta~) a coherent maximiser of the conditional likelihood
   rather than two estimates from different fits.

Shrinkage is applied per tail, never across tails: a return series has no reason
for its two tails to be equally heavy, and evt.py already treats them separately.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize

_EPS = 1e-12


# ------------------------------------------------------------------ estimator
def fit_gpd_zs(exc: np.ndarray) -> tuple[float, float]:
    """Zhang & Stephens (2009) estimator of (xi, beta) for POT exceedances.

    Same signature and convention as evt.fit_gpd, so it is a drop-in there too.
    Profiles the likelihood in b = xi/beta on a grid anchored on the sample
    (quartile and maximum), then averages b over the grid with profile-
    likelihood weights instead of maximising -- which is where the variance
    reduction comes from.
    """
    y = np.asarray(exc, dtype=float)
    y = np.sort(y[np.isfinite(y) & (y > 0.0)])
    n = y.size
    if n < 5:
        raise ValueError(f"GPD fit needs >=5 positive exceedances, got {n}")

    m = 30 + int(np.sqrt(n))
    bs = 1.0 - np.sqrt(m / (np.arange(1, m + 1) - 0.5))
    bs /= 3.0 * y[int(n / 4 + 0.5) - 1]
    bs += 1.0 / y[-1]
    xis = np.log1p(-bs[:, None] * y).mean(axis=1)          # xi(b) along the grid
    L = n * (np.log(-bs / xis) - xis - 1.0)                # profile log-likelihood
    w = 1.0 / np.exp(L - L[:, None]).sum(axis=1)
    w /= w.sum()
    b = float((bs * w).sum())
    xi = float(np.log1p(-b * y).mean())
    return xi, float(max(-xi / b, _EPS))


def beta_profile(exc: np.ndarray, xi: float) -> float:
    """MLE of beta with xi held fixed -- evt.fit_gpd's xi_max objective."""
    y = np.asarray(exc, dtype=float)
    y = y[np.isfinite(y) & (y > 0.0)]
    n = y.size
    if abs(xi) < 1e-8:                                     # xi -> 0: exponential
        return float(max(y.mean(), _EPS))
    lo = _EPS if xi > 0 else (-xi * y.max()) * (1.0 + 1e-9)   # keep support valid
    return float(optimize.minimize_scalar(
        lambda s_: n * np.log(s_) + (1.0 / xi + 1.0) * np.sum(np.log1p(xi * y / s_)),
        bounds=(lo, 50.0 * np.median(y)), method="bounded").x)


# -------------------------------------------------------------------- pooling
def eb_weights(xi_hat: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, float, float]:
    """(lambda_j, mu, tau^2) for the empirical-Bayes shrinkage of xi."""
    xi_hat = np.asarray(xi_hat, dtype=float)
    s2 = (1.0 + xi_hat) ** 2 / np.maximum(np.asarray(k, dtype=float), 1.0)
    mu = float(xi_hat.mean())
    tau2 = float(max(0.0, xi_hat.var(ddof=1) - s2.mean()))
    return tau2 / (tau2 + s2), mu, tau2


def eb_shrink_xi(xi_hat: np.ndarray, k: np.ndarray,
                 c: float = 1.0) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Empirical-Bayes shrinkage of xi with an Efron-Morris limited-translation cap.

    Plain EB moves feature j by (1 - lambda_j)(xi_hat_j - mu).  For a feature
    whose tail really is the heaviest in the panel that displacement is a bias,
    and it lands where it hurts most: the fitted GPD then assigns that feature's
    own sample maximum a survival probability far below 1/k, and the PIT sends
    it to |z| >> 100 -- the exact failure check.py gates on.  Capping the
    displacement at c standard errors (Efron & Morris 1972) keeps the pooling
    gain for the bulk and bounds the damage to the extremes.  Measured at
    f=235, k=62 on t-distributed data with xi in [0.167, 0.332]:

        rule                      RMSE(xi)   sd(xi_hat)   max|z|
        none (Zhang-Stephens)       0.167       0.169       24.4
        full EB (c = inf)           0.080       0.041      159.1   <- 2 features fail
        limited translation c=1.5   0.089       0.057       95.8
        limited translation c=1.0   0.100       0.078       55.0   <- default
        limited translation c=0.5   0.124       0.116       34.0

    c=1.0 keeps ~77% of the RMSE reduction with max|z| comfortably inside the
    gate.  c=inf reproduces the unrestricted posterior mean.
    """
    xi_hat = np.asarray(xi_hat, dtype=float)
    k = np.maximum(np.asarray(k, dtype=float), 1.0)
    lam, mu, tau2 = eb_weights(xi_hat, k)
    move = (1.0 - lam) * (xi_hat - mu)
    if np.isfinite(c):
        se = (1.0 + xi_hat) / np.sqrt(k)
        move = np.sign(move) * np.minimum(np.abs(move), c * se)
    return xi_hat - move, lam, mu, tau2


def exceedances(m, x_col: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(lower, upper) exceedances of one fitted marginal at its own thresholds."""
    x = np.asarray(x_col, dtype=float).ravel()
    return m.u_lo_ - x[x < m.u_lo_], x[x > m.u_hi_] - m.u_hi_


def shrink_ensemble(marg, x: np.ndarray, refit: bool = True, c: float = 1.0,
                    xi_min: float | None = 0.0, verbose: bool = True) -> dict:
    """Re-estimate and pool the tails of a fitted MarginalEnsemble, in place.

    x is the SAME array passed to marg.fit -- the thresholds u_lo_/u_hi_ are
    kept, so this changes only how the exceedances above them are described.
    `refit=False` skips Zhang-Stephens and pools the existing MLE fits.
    """
    x = np.asarray(x, dtype=float)
    cols = x.reshape(-1, x.shape[-1])
    m_list = marg.marginals_
    f = len(m_list)

    exc = [exceedances(m_list[j], cols[:, j]) for j in range(f)]
    if refit:
        xi_lo, xi_hi = np.empty(f), np.empty(f)
        for j, (e_lo, e_hi) in enumerate(exc):
            xi_lo[j] = fit_gpd_zs(e_lo)[0]
            xi_hi[j] = fit_gpd_zs(e_hi)[0]
    else:
        xi_lo = np.array([m.xi_lo_ for m in m_list])
        xi_hi = np.array([m.xi_hi_ for m in m_list])

    k_lo = np.array([m.n_exc_lo_ for m in m_list], dtype=float)
    k_hi = np.array([m.n_exc_hi_ for m in m_list], dtype=float)
    new_lo, lam_lo, mu_lo, tau2_lo = eb_shrink_xi(xi_lo, k_lo, c)
    new_hi, lam_hi, mu_hi, tau2_hi = eb_shrink_xi(xi_hi, k_hi, c)

    # SHAPE FLOOR.  xi < 0 gives the GPD a finite endpoint u -+ beta/|xi|, and the
    # MLE puts that endpoint essentially at the extreme training observation.  A
    # held-out value past it gets F_hat = 0, which cdf() clips to _EPS, and the PIT
    # returns |z| = |t_nu^{-1}(1e-12)| -- 394 at nu=5.  Shrinkage cannot repair
    # this: the displacement is capped at c standard errors by construction, so a
    # feature at xi = -0.45 never crosses zero however c is tuned.  Flooring xi at
    # 0 makes the tail exponential and therefore unbounded, so no future
    # observation can fall outside the support.  This is the OPPOSITE of evt.py's
    # xi_max: capping xi from above hides a real finding about a heavy instrument,
    # whereas a hard floor on the losses of a daily-marked traded instrument is a
    # claim the held-out data has already contradicted.  xi_min=None disables it.
    n_floor = 0
    if xi_min is not None:
        n_floor = int((new_lo < xi_min).sum() + (new_hi < xi_min).sum())
        new_lo = np.maximum(new_lo, xi_min)
        new_hi = np.maximum(new_hi, xi_min)

    for j, m in enumerate(m_list):
        m.xi_lo_, m.beta_lo_ = float(new_lo[j]), beta_profile(exc[j][0], new_lo[j])
        m.xi_hi_, m.beta_hi_ = float(new_hi[j]), beta_profile(exc[j][1], new_hi[j])

    out = dict(xi_lo_raw=xi_lo, xi_hi_raw=xi_hi,
               xi_lo_shrunk=new_lo, xi_hi_shrunk=new_hi,
               lam_lo=lam_lo, lam_hi=lam_hi, mu_lo=mu_lo, mu_hi=mu_hi,
               tau2_lo=tau2_lo, tau2_hi=tau2_hi)
    if verbose:
        for side, xi_r, xi_s, lam, mu, t2, k in (
                ("lower", xi_lo, new_lo, lam_lo, mu_lo, tau2_lo, k_lo),
                ("upper", xi_hi, new_hi, lam_hi, mu_hi, tau2_hi, k_hi)):
            se = float(np.mean((1.0 + xi_r) ** 2 / k) ** 0.5)
            print(f"  {side}: pooled mu={mu:+.3f}  sd(xi_hat)={xi_r.std(ddof=1):.3f}"
                  f"  mean se={se:.3f}  ->  tau={np.sqrt(t2):.3f}"
                  f"  mean lambda={lam.mean():.2f}"
                  f"  (range {xi_r.min():+.3f},{xi_r.max():+.3f}"
                  f" -> {xi_s.min():+.3f},{xi_s.max():+.3f})")
        if xi_min is not None:
            print(f"  shape floor xi >= {xi_min:+.2f} applied to {n_floor} of "
                  f"{2 * f} tails (finite GPD endpoint removed)")
    out["n_floor"] = n_floor
    return out
