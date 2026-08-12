"""Semi-parametric marginal models via Extreme Value Theory (peaks-over-threshold).

For each feature and each tail, the marginal CDF is

    F(x) = q_lo * GPD_sf((u_lo - x); xi_lo, beta_lo)          for x <  u_lo
    F(x) = empirical (interpolated)                            for u_lo <= x <= u_hi
    F(x) = 1 - q_hi * GPD_sf((x - u_hi); xi_hi, beta_hi)       for x >  u_hi

with GPD_sf(y; xi, beta) = (1 + xi y / beta)^(-1/xi), justified by
Pickands-Balkema-de Haan: exceedances over a high threshold of any distribution in
a max-domain of attraction are asymptotically GPD.  The PIT

    z = T_nu^{-1}(F(x))

then makes every marginal exactly t_nu in z-space, matching the flow-matching base
(base.py) so the flow only has to transport the copula.

THRESHOLD SELECTION (q_tail="auto").  A single global q is the wrong object.  The
asymptotics hold only as u -> x_F, so too low a threshold fits the *body*; too high
a one leaves too few exceedances.  Simulation at T=1236 makes the sizes concrete:

  - Data whose tail really is GPD (t_3, true xi = 1/3): bias is flat in q
    (-0.08 to -0.13 across q = 0.02..0.20) while sd falls monotonically with k.
    There is NO bias-variance trade-off when the model is right -- more
    exceedances is strictly better.
  - Data with volatility regimes (normal mixture, true xi = 0, Gaussian tail):
    q = 0.05 gives xi_hat = -0.26, but q = 0.10 gives +0.53 and q = 0.15 gives
    +0.76.  The GPD is fitting the mixture, not the tail.

So the trade-off exists only through misspecification, and the rule is: take the
LOWEST threshold (largest k, smallest variance) at which the GPD still fits.
Thresholds are scanned from large q downward and the first to pass an
Anderson-Darling goodness-of-fit test on its own exceedances is kept -- per tail,
since the two tails of a return series need not be equally heavy.  The AD null is
obtained by parametric bootstrap, which is exact up to Monte Carlo error and
avoids the xi-dependent critical-value tables.

NO SHAPE CONSTRAINT.  xi is estimated freely (xi_max defaults to inf).  Capping it
is neither necessary nor sufficient: a badly-fitted tail produces |z| ~ 1e3 at
xi = 0.48, and a correctly-fitted one keeps |z| small at any xi.  Simulated with
data that really is GPD, k = 247 exceedances, over 400 replicates:

    true xi   median max|z|   p99   worst
      0.3         12.9        42      62
      2.0         12.6        44      89

i.e. the transformed extreme is bounded around 100 whatever the tail index, because
a valid fit assigns the largest of k exceedances a survival probability near 1/k by
construction.  So max|z| >> 100 is a goodness-of-fit diagnostic, and xi >= 1
(infinite mean of exceedances) is a finding about an instrument rather than a
number to suppress.  xi_max is retained only to run the constrained variant as an
ablation.

WHAT THIS STILL DOES NOT FIX.  The AD test loses power as the fitted tail gets
heavier: simulating the null from a GPD with xi ~ 2 gives an A^2 distribution wide
enough to accept almost anything.  A column can therefore pass selection and still
transform to |z| ~ 1e4.  Threshold selection is the right primary fix -- it repairs
the misspecification case cleanly, picking q = 0.20 for a genuinely GPD tail and
q = 0.04 for a regime mixture whose forced q = 0.15 fit would have returned
xi = +0.82 against a true 0 -- but max|z| must still be checked afterwards, which
is what check.py does.

Conventions: 'lower'/'upper' refer to tails of the *raw variable* (e.g. returns).
Risk of losses L = -r corresponds to the *lower* tail of returns.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize, stats

_EPS = 1e-12
Q_GRID = (0.20, 0.15, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03)
MIN_EXC = 25          # below this a GPD fit is not worth attempting


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


# ----------------------------------------------------------------- GPD fitting
def fit_gpd(exc: np.ndarray, xi_max: float = np.inf) -> tuple[float, float]:
    """MLE of (xi, beta) for exceedances over a threshold, location fixed at 0 (POT).

    scipy's two-parameter MLE, which is accurate across the whole range that matters
    here: against known truth at k = 124 it recovers -0.216 / +0.194 / +0.399 /
    +0.693 / +1.989 for true xi of -0.2 / 0.2 / 0.4 / 0.7 / 2.0.  A hand-rolled
    profile in Grimshaw's theta = xi/beta agrees to 1e-3 below xi ~ 0.7 but needs a
    search interval anchored on the mean exceedance, which is infinite for xi >= 1 --
    exactly the regime worth measuring.

    Note what this does NOT do: bound xi.  A large xi_hat is a statement that the
    exceedances at this threshold are very heavy, and if that is an artefact it is
    an artefact of the *threshold*, fixed by select_threshold, not by clamping the
    parameter.  xi_max exists only to run the constrained variant as an ablation.

    Raises ValueError on a degenerate exceedance set -- the threshold coinciding
    with the sample extremum, e.g. an accrual series that never falls.
    """
    y = np.asarray(exc, dtype=float)
    y = y[np.isfinite(y) & (y > 0.0)]
    n = y.size
    if n < 5:
        raise ValueError(
            f"GPD fit needs >=5 positive exceedances, got {n}; the threshold "
            "coincides with the sample extremum (a point mass there, e.g. an "
            "accrual series that never falls)")

    xi, _, beta = stats.genpareto.fit(y, floc=0.0)
    if not (np.isfinite(xi) and np.isfinite(beta) and beta > 0.0):
        raise ValueError(f"GPD MLE did not converge on {n} exceedances")

    if xi > xi_max:                       # ablation path only; off by default
        xi = float(xi_max)
        beta = float(optimize.minimize_scalar(
            lambda s_: n * np.log(s_) + (1.0 / xi + 1.0) * np.sum(np.log1p(xi * y / s_)),
            bounds=(_EPS, 50.0 * np.median(y)), method="bounded").x)
    return float(xi), float(max(beta, _EPS))


# -------------------------------------------------------- goodness of fit
def gpd_ad_statistic(y: np.ndarray, xi: float, beta: float) -> float:
    """Anderson-Darling A^2 of exceedances against the fitted GPD.

    AD rather than KS because its 1/(u(1-u)) weight puts the power in the tails,
    which is the region the threshold decision is about.
    """
    u = np.sort(stats.genpareto.cdf(np.asarray(y, float), c=xi, scale=beta))
    u = np.clip(u, _EPS, 1.0 - _EPS)
    n = u.size
    i = np.arange(1, n + 1)
    return float(-n - np.mean((2 * i - 1) * (np.log(u) + np.log1p(-u[::-1]))))


def gpd_gof_pvalue(y: np.ndarray, xi: float, beta: float, n_boot: int = 199,
                   rng: np.random.Generator | None = None,
                   xi_max: float = np.inf) -> float:
    """Parametric-bootstrap p-value for the AD statistic.

    The null distribution of A^2 depends on xi (and weakly on n) once the
    parameters are estimated, so tabulated critical values would need
    interpolation in xi; simulating from the fitted GPD and refitting each replicate
    is exact up to Monte Carlo error and costs ~1 ms per replicate here.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    a_obs = gpd_ad_statistic(y, xi, beta)
    n, hits, used = np.asarray(y).size, 0, 0
    for _ in range(n_boot):
        s = stats.genpareto.rvs(xi, scale=beta, size=n, random_state=rng)
        try:
            xb, bb = fit_gpd(s, xi_max)
        except ValueError:
            continue
        used += 1
        if gpd_ad_statistic(s, xb, bb) >= a_obs:
            hits += 1
    return (hits + 1) / (used + 1) if used else np.nan


def select_threshold(x: np.ndarray, tail: str, q_grid=Q_GRID, alpha: float = 0.05,
                     n_boot: int = 199, rng: np.random.Generator | None = None,
                     xi_max: float = np.inf) -> dict:
    """Lowest threshold (largest q) whose exceedances still pass the AD test.

    Scans q downward from the largest candidate and returns the first that passes
    at level `alpha`, maximising the exceedance count subject to the GPD being
    tenable.  If none passes, returns the candidate with the largest p-value and
    flags it, so a feature that is nowhere GPD is visible rather than silently
    fitted at an arbitrary q.

    Caveat worth stating in any write-up: this is a sequential test over a grid, so
    `alpha` is a per-comparison level, not a family-wise one.  Bader, Yan and Zhang
    (2018) apply a ForwardStop correction to the same scan; the ordering of the
    selected thresholds is unaffected in practice, only the nominal level.
    """
    x = np.asarray(x, dtype=float).ravel()
    best = None
    for q in sorted(q_grid, reverse=True):
        if tail == "lower":
            u = float(np.quantile(x, q))
            exc = u - x[x < u]
        else:
            u = float(np.quantile(x, 1.0 - q))
            exc = x[x > u] - u
        if exc.size < MIN_EXC:
            continue
        try:
            xi, beta = fit_gpd(exc, xi_max)
        except ValueError:
            continue
        p = gpd_gof_pvalue(exc, xi, beta, n_boot, rng, xi_max)
        cand = dict(q=q, u=u, xi=xi, beta=beta, n_exc=int(exc.size), p=float(p),
                    passed=bool(p >= alpha))
        if best is None or (np.isfinite(p) and p > best["p"]):
            best = cand
        if cand["passed"]:
            return cand
    if best is None:
        raise ValueError(f"no usable {tail} threshold: fewer than {MIN_EXC} "
                         "exceedances at every candidate q")
    return best


class SemiParametricMarginal:
    """Marginal model for one feature: empirical body + GPD tails + t_nu PIT.

    q_tail may be a float (that threshold for both tails, the classical setup) or
    "auto", in which case each tail gets its own threshold from select_threshold.
    The two tails are treated separately throughout: q_lo_ and q_hi_ can differ,
    and the body is pinned to F(u_lo) = q_lo_ and F(u_hi) = 1 - q_hi_ so the
    piecewise CDF stays continuous either way.
    """

    def __init__(self, q_tail: float | str = 0.05, nu: float = 4.0,
                 xi_max: float = np.inf, q_grid=Q_GRID, alpha: float = 0.05,
                 n_boot: int = 199):
        if not isinstance(q_tail, str):
            assert 0.0 < float(q_tail) < 0.5
        self.q_tail, self.nu, self.xi_max = q_tail, nu, xi_max
        self.q_grid, self.alpha, self.n_boot = q_grid, alpha, n_boot

    # ----------------------------------------------------------------- fitting
    def fit(self, x: np.ndarray,
            rng: np.random.Generator | None = None) -> "SemiParametricMarginal":
        x = np.sort(np.asarray(x, dtype=float).ravel())
        self.n_ = x.size

        if isinstance(self.q_tail, str) and self.q_tail == "auto":
            lo = select_threshold(x, "lower", self.q_grid, self.alpha,
                                  self.n_boot, rng, self.xi_max)
            hi = select_threshold(x, "upper", self.q_grid, self.alpha,
                                  self.n_boot, rng, self.xi_max)
        else:
            q = float(self.q_tail)
            u_lo, u_hi = np.quantile(x, q), np.quantile(x, 1.0 - q)
            e_lo, e_hi = u_lo - x[x < u_lo], x[x > u_hi] - u_hi
            xl, bl = fit_gpd(e_lo, self.xi_max)
            xh, bh = fit_gpd(e_hi, self.xi_max)
            lo = dict(q=q, u=float(u_lo), xi=xl, beta=bl, n_exc=int(e_lo.size),
                      p=np.nan, passed=True)
            hi = dict(q=q, u=float(u_hi), xi=xh, beta=bh, n_exc=int(e_hi.size),
                      p=np.nan, passed=True)

        self.q_lo_, self.u_lo_, self.xi_lo_, self.beta_lo_ = (
            lo["q"], lo["u"], lo["xi"], lo["beta"])
        self.q_hi_, self.u_hi_, self.xi_hi_, self.beta_hi_ = (
            hi["q"], hi["u"], hi["xi"], hi["beta"])
        self.n_exc_lo_, self.n_exc_hi_ = lo["n_exc"], hi["n_exc"]
        self.gof_p_lo_, self.gof_p_hi_ = lo["p"], hi["p"]
        self.gof_ok_lo_, self.gof_ok_hi_ = lo["passed"], hi["passed"]

        # Interpolated empirical body on plotting positions (i - 0.5)/n, pinned to
        # the thresholds so the piecewise CDF is continuous.
        p = (np.arange(1, self.n_ + 1) - 0.5) / self.n_
        mask = (p > self.q_lo_) & (p < 1.0 - self.q_hi_)
        xs = np.concatenate([[self.u_lo_], x[mask], [self.u_hi_]])
        ps = np.concatenate([[self.q_lo_], p[mask], [1.0 - self.q_hi_]])
        xs, idx = np.unique(xs, return_index=True)   # strictly increasing for interp
        self._body_x, self._body_p = xs, ps[idx]
        return self

    # --------------------------------------------------------------- cdf / ppf
    def cdf(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        out = np.interp(x, self._body_x, self._body_p)
        lo, hi = x < self.u_lo_, x > self.u_hi_
        if lo.any():
            out[lo] = self.q_lo_ * stats.genpareto.sf(
                self.u_lo_ - x[lo], c=self.xi_lo_, scale=self.beta_lo_)
        if hi.any():
            out[hi] = 1.0 - self.q_hi_ * stats.genpareto.sf(
                x[hi] - self.u_hi_, c=self.xi_hi_, scale=self.beta_hi_)
        return np.clip(out, _EPS, 1.0 - _EPS)

    def ppf(self, p: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=float), _EPS, 1.0 - _EPS)
        out = np.interp(p, self._body_p, self._body_x)
        lo, hi = p < self.q_lo_, p > 1.0 - self.q_hi_
        if lo.any():
            out[lo] = self.u_lo_ - stats.genpareto.isf(
                p[lo] / self.q_lo_, c=self.xi_lo_, scale=self.beta_lo_)
        if hi.any():
            out[hi] = self.u_hi_ + stats.genpareto.isf(
                (1.0 - p[hi]) / self.q_hi_, c=self.xi_hi_, scale=self.beta_hi_)
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
        returns). tail='upper': loss L = +X. Requires alpha beyond that tail's own
        threshold so the quantile lies in the GPD region, and xi < 1 for ES to
        exist -- xi >= 1 means the exceedances have infinite mean, so the ES is
        genuinely undefined and should be reported as such rather than fitted away.
        """
        if tail == "lower":
            q, u, xi, beta = self.q_lo_, -self.u_lo_, self.xi_lo_, self.beta_lo_
        else:
            q, u, xi, beta = self.q_hi_, self.u_hi_, self.xi_hi_, self.beta_hi_
        assert alpha >= 1.0 - q, f"alpha must exceed 1 - q_{tail} = {1 - q:.3f}"
        assert xi < 1.0, (f"ES undefined: xi_{tail} = {xi:.3f} >= 1 (infinite mean "
                          "of exceedances)")
        ratio = (1.0 - alpha) / q
        if abs(xi) > 1e-8:
            var = u + (beta / xi) * (ratio ** (-xi) - 1.0)
        else:  # xi -> 0 limit (exponential tail)
            var = u - beta * np.log(ratio)
        es = var / (1.0 - xi) + (beta - xi * u) / (1.0 - xi)
        return float(var), float(es)


class MarginalEnsemble:
    """Per-feature SemiParametricMarginal for arrays shaped (..., f).

    If nu='auto', the reference degrees of freedom are set to the median Hill index
    over features and tails, clipped to [2.5, 10]; a base at least as heavy as the
    data is required for the Lipschitz-flow argument to go through (the constraint
    is one-sided).
    """

    def __init__(self, q_tail: float | str = 0.05, nu: float | str = "auto",
                 xi_max: float = np.inf, q_grid=Q_GRID, alpha: float = 0.05,
                 n_boot: int = 199, seed: int = 0):
        self.q_tail, self.nu, self.xi_max = q_tail, nu, xi_max
        self.q_grid, self.alpha, self.n_boot, self.seed = q_grid, alpha, n_boot, seed

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

        rng = np.random.default_rng(self.seed)
        self.marginals_ = []
        for j in range(f):
            try:
                self.marginals_.append(
                    SemiParametricMarginal(self.q_tail, self.nu_, self.xi_max,
                                           self.q_grid, self.alpha,
                                           self.n_boot).fit(cols[:, j], rng))
            except ValueError as e:      # name the column instead of a bare traceback
                raise ValueError(f"feature index {j}: {e}") from None
        return self

    def summary(self) -> dict:
        """Per-feature selection outcome, for check.py / evtdiag to report."""
        m = self.marginals_
        return dict(
            q_lo=np.array([x.q_lo_ for x in m]),
            q_hi=np.array([x.q_hi_ for x in m]),
            xi_lo=np.array([x.xi_lo_ for x in m]),
            xi_hi=np.array([x.xi_hi_ for x in m]),
            n_exc_lo=np.array([x.n_exc_lo_ for x in m]),
            n_exc_hi=np.array([x.n_exc_hi_ for x in m]),
            gof_p_lo=np.array([x.gof_p_lo_ for x in m]),
            gof_p_hi=np.array([x.gof_p_hi_ for x in m]),
            gof_ok_lo=np.array([x.gof_ok_lo_ for x in m]),
            gof_ok_hi=np.array([x.gof_ok_hi_ for x in m]),
        )

    def _apply(self, x: np.ndarray, method: str) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        shape, f = x.shape, x.shape[-1]
        flat = x.reshape(-1, f)
        out = np.stack([getattr(self.marginals_[j], method)(flat[:, j])
                        for j in range(f)], axis=-1)
        return out.reshape(shape)

    def transform(self, x):          return self._apply(x, "transform")
    def inverse_transform(self, z):  return self._apply(z, "inverse_transform")
