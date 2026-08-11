"""Semi-parametric marginal models via Extreme Value Theory (peaks-over-threshold).

For each feature, the marginal CDF is modeled as

    F(x) = q_tail * GPD_sf((u_lo - x); xi_lo, beta_lo)          for x <  u_lo   (lower tail)
    F(x) = empirical (interpolated)                              for u_lo <= x <= u_hi
    F(x) = 1 - q_tail * GPD_sf((x - u_hi); xi_hi, beta_hi)       for x >  u_hi   (upper tail)

where GPD_sf(y; xi, beta) = (1 + xi * y / beta)^(-1/xi) is the survival function of the
Generalized Pareto Distribution, justified by the Pickands-Balkema-de Haan theorem:
exceedances over a high threshold of any distribution in a max-domain of attraction are
asymptotically GPD.

The probability integral transform (PIT) then maps data to a Student-t_nu reference scale:

    z = T_nu^{-1}(F(x)),

so that in z-space every marginal is *exactly* t_nu (up to fitting error). Combined with a
t_nu flow-matching base (see base.py), the source and target marginals coincide in z-space
and the flow only has to transport the *copula* (temporal + cross-sectional dependence) --
a bounded-Lipschitz task. Mapping generated samples back through F^{-1} guarantees GPD
tails, hence closed-form tail risk beyond the threshold:

    VaR_a = u + (beta/xi) * [ ((1-a)/q_tail)^(-xi) - 1 ]
    ES_a  = VaR_a / (1 - xi) + (beta - xi * u) / (1 - xi)         (valid for xi < 1)

Conventions: 'lower'/'upper' refer to tails of the *raw variable* (e.g. returns). Risk of
losses L = -r corresponds to the *lower* tail of returns.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

_EPS = 1e-12


def hill_estimator(x: np.ndarray, k_frac: float = 0.05, tail: str = "lower") -> float:
    """Hill estimator of the tail index alpha (heavier tail <=> smaller alpha).

        alpha_hat^{-1} = (1/k) * sum_{i=1}^{k} log( X_(n-i+1) / X_(n-k) )

    computed on the positive exceedances of the requested tail (x -> -x for 'lower').
    Returns np.inf if the tail has too few positive observations (treated as light).
    """
    y = -np.asarray(x, dtype=float) if tail == "lower" else np.asarray(x, dtype=float)
    y = np.sort(y[y > 0.0])
    k = max(10, int(k_frac * y.size))
    if y.size < k + 1:
        return np.inf
    top, x_k = y[-k:], y[-k - 1]
    inv_alpha = np.mean(np.log(top / x_k))
    return 1.0 / max(inv_alpha, _EPS)


class SemiParametricMarginal:
    """Marginal model for one feature: empirical body + GPD tails + t_nu PIT."""

    def __init__(self, q_tail: float = 0.05, nu: float = 4.0):
        assert 0.0 < q_tail < 0.5
        self.q_tail, self.nu = q_tail, nu

    # ----------------------------------------------------------------- fitting
    def fit(self, x: np.ndarray) -> "SemiParametricMarginal":
        x = np.sort(np.asarray(x, dtype=float).ravel())
        self.n_ = x.size
        self.u_lo_ = np.quantile(x, self.q_tail)
        self.u_hi_ = np.quantile(x, 1.0 - self.q_tail)

        # GPD MLE on exceedances, location fixed at 0 (POT).
        exc_hi = x[x > self.u_hi_] - self.u_hi_
        exc_lo = self.u_lo_ - x[x < self.u_lo_]
        self.xi_hi_, _, self.beta_hi_ = stats.genpareto.fit(exc_hi, floc=0.0)
        self.xi_lo_, _, self.beta_lo_ = stats.genpareto.fit(exc_lo, floc=0.0)

        # Interpolated empirical body on plotting positions (i - 0.5)/n, pinned to the
        # thresholds so the piecewise CDF is continuous: F(u_lo)=q_tail, F(u_hi)=1-q_tail.
        p = (np.arange(1, self.n_ + 1) - 0.5) / self.n_
        mask = (p > self.q_tail) & (p < 1.0 - self.q_tail)
        xs = np.concatenate([[self.u_lo_], x[mask], [self.u_hi_]])
        ps = np.concatenate([[self.q_tail], p[mask], [1.0 - self.q_tail]])
        xs, idx = np.unique(xs, return_index=True)  # strictly increasing for interp
        self._body_x, self._body_p = xs, ps[idx]
        return self

    # --------------------------------------------------------------- cdf / ppf
    def cdf(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        out = np.interp(x, self._body_x, self._body_p)
        lo, hi = x < self.u_lo_, x > self.u_hi_
        if lo.any():
            out[lo] = self.q_tail * stats.genpareto.sf(
                self.u_lo_ - x[lo], c=self.xi_lo_, scale=self.beta_lo_)
        if hi.any():
            out[hi] = 1.0 - self.q_tail * stats.genpareto.sf(
                x[hi] - self.u_hi_, c=self.xi_hi_, scale=self.beta_hi_)
        return np.clip(out, _EPS, 1.0 - _EPS)

    def ppf(self, p: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=float), _EPS, 1.0 - _EPS)
        out = np.interp(p, self._body_p, self._body_x)
        lo, hi = p < self.q_tail, p > 1.0 - self.q_tail
        if lo.any():
            out[lo] = self.u_lo_ - stats.genpareto.isf(
                p[lo] / self.q_tail, c=self.xi_lo_, scale=self.beta_lo_)
        if hi.any():
            out[hi] = self.u_hi_ + stats.genpareto.isf(
                (1.0 - p[hi]) / self.q_tail, c=self.xi_hi_, scale=self.beta_hi_)
        return out

    # ------------------------------------------------------------------- PIT
    def transform(self, x: np.ndarray) -> np.ndarray:
        """x -> z with z ~ t_nu marginally."""
        return stats.t.ppf(self.cdf(x), df=self.nu)

    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        return self.ppf(stats.t.cdf(np.asarray(z, dtype=float), df=self.nu))

    # ------------------------------------------------------------- tail risk
    def var_es(self, alpha: float, tail: str = "lower") -> tuple[float, float]:
        """Closed-form (VaR, ES) of the *loss* implied by this marginal.

        tail='lower': loss L = -X, risk in the lower tail of X (the usual case for
        returns). tail='upper': loss L = +X. Requires alpha >= 1 - q_tail so that the
        quantile lies in the GPD region, and xi < 1 for ES to exist.
        """
        assert alpha >= 1.0 - self.q_tail, "alpha must lie in the GPD tail region"
        if tail == "lower":
            u, xi, beta = -self.u_lo_, self.xi_lo_, self.beta_lo_
        else:
            u, xi, beta = self.u_hi_, self.xi_hi_, self.beta_hi_
        assert xi < 1.0, "ES undefined (infinite mean of exceedances) for xi >= 1"
        ratio = (1.0 - alpha) / self.q_tail
        if abs(xi) > 1e-8:
            var = u + (beta / xi) * (ratio ** (-xi) - 1.0)
        else:  # xi -> 0 limit (exponential tail)
            var = u - beta * np.log(ratio)
        es = var / (1.0 - xi) + (beta - xi * u) / (1.0 - xi)
        return float(var), float(es)


class MarginalEnsemble:
    """Per-feature SemiParametricMarginal for arrays shaped (..., f).

    If nu='auto', the reference degrees of freedom are set to the median Hill index over
    features and tails, clipped to [2.5, 10]; a base at least as heavy as the data is
    required for the Lipschitz-flow argument to go through (the constraint is one-sided).
    """

    def __init__(self, q_tail: float = 0.05, nu: float | str = "auto"):
        self.q_tail, self.nu = q_tail, nu

    def fit(self, x: np.ndarray) -> "MarginalEnsemble":
        x = np.asarray(x, dtype=float)
        f = x.shape[-1]
        cols = x.reshape(-1, f)
        if self.nu == "auto":
            hills = [hill_estimator(cols[:, j], tail=t)
                     for j in range(f) for t in ("lower", "upper")]
            self.nu_ = float(np.clip(np.median([h for h in hills if np.isfinite(h)]),
                                     2.5, 10.0))
        else:
            self.nu_ = float(self.nu)
        self.marginals_ = [SemiParametricMarginal(self.q_tail, self.nu_).fit(cols[:, j])
                           for j in range(f)]
        return self

    def _apply(self, x: np.ndarray, method: str) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        shape, f = x.shape, x.shape[-1]
        flat = x.reshape(-1, f)
        out = np.stack([getattr(self.marginals_[j], method)(flat[:, j])
                        for j in range(f)], axis=-1)
        return out.reshape(shape)

    def transform(self, x):          return self._apply(x, "transform")
    def inverse_transform(self, z):  return self._apply(z, "inverse_transform")
