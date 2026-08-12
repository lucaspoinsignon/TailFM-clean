"""Threshold selection: Hill plots and xi(q) stability, per feature.

    python -m evtdiag.hill --data data/returns.csv --out fig/hill.png

The problem this answers: --q-tail is currently one global number picked by fiat.
It controls k = q*T exceedances per tail, and that is a bias-variance trade, not a
convention.  Pickands-Balkema-de Haan gives the GPD limit only as u -> x_F, so a
high threshold (small q) is asymptotically right but leaves few points --
Var(xi_hat) ~ (1+xi)^2 / k, so at q = 0.05 and T = 1236 the standard error is
~0.17 and an unconstrained MLE can run away to xi = 5.8, which maps observed
exceedances to p ~ 1e-12 and hence |z| ~ 1e4.  A low threshold gives more points
but fits the body rather than the tail, biasing xi.

Two classical diagnostics, computed per feature and per tail:

  Hill plot     alpha_hat(k) = 1 / mean(log X_(n-i+1) / X_(n-k)), i = 1..k, whose
                reciprocal estimates xi for heavy tails.  Read off the region
                where the curve is flat in k.
  xi(q) plot    the GPD MLE itself as a function of the threshold.  Same idea,
                but for the estimator actually used downstream.

The plateau is where the GPD approximation holds and the estimate is still
stable.  --pick reports, per feature, the q minimising the local roughness of
xi(q) over the scanned grid, as a starting point rather than an answer: threshold
choice is a judgement call and the plots are the evidence for it.
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from csvio import load_returns, feature_names_from_csv
from tailfm import hill_estimator


def gpd_xi(x: np.ndarray, q: float, tail: str) -> float:
    """Unconstrained GPD MLE of xi at threshold quantile q, one tail."""
    if tail == "lower":
        u = np.quantile(x, q)
        exc = u - x[x < u]
    else:
        u = np.quantile(x, 1 - q)
        exc = x[x > u] - u
    if exc.size < 10:
        return np.nan
    try:
        return float(stats.genpareto.fit(exc, floc=0.0)[0])
    except Exception:
        return np.nan


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--prices", action="store_true")
    p.add_argument("--test-frac", type=float, default=0.2,
                   help="fit on the training rows only, as fit_returns does")
    p.add_argument("--valors", default=None,
                   help="comma-separated names; default = the most problematic")
    p.add_argument("--n-panels", type=int, default=9)
    p.add_argument("--q-grid", default="0.02,0.15,14",
                   help="qmin,qmax,npoints for the xi(q) scan")
    p.add_argument("--out", default="fig/hill.png")
    p.add_argument("--csv", default="evt_threshold.csv")
    a = p.parse_args()

    r = load_returns(a.data, a.prices)
    T, f = r.shape
    names = feature_names_from_csv(a.data, f)
    tr = r[:int((1 - a.test_frac) * T)]
    n = tr.shape[0]
    q0, q1, nq = a.q_grid.split(",")
    qs = np.linspace(float(q0), float(q1), int(nq))
    print(f"{a.data}: {n} training rows x {f} features; "
          f"q grid {qs[0]:.3f}..{qs[-1]:.3f} ({len(qs)} points, "
          f"k = {int(qs[0]*n)}..{int(qs[-1]*n)} exceedances)")

    # xi(q) for every feature and tail
    XI = {t: np.array([[gpd_xi(tr[:, j], q, t) for q in qs] for j in range(f)])
          for t in ("lower", "upper")}

    rough = np.nansum(np.abs(np.diff(XI["lower"], axis=1)), axis=1)
    best_q = np.array([qs[int(np.nanargmin(np.abs(np.gradient(XI["lower"][j]))))]
                       if np.isfinite(XI["lower"][j]).any() else np.nan
                       for j in range(f)])
    d = pd.DataFrame({
        "VALOR": names,
        "xi_lo@0.05": [gpd_xi(tr[:, j], 0.05, "lower") for j in range(f)],
        "xi_lo@0.10": [gpd_xi(tr[:, j], 0.10, "lower") for j in range(f)],
        "xi_hi@0.05": [gpd_xi(tr[:, j], 0.05, "upper") for j in range(f)],
        "xi_hi@0.10": [gpd_xi(tr[:, j], 0.10, "upper") for j in range(f)],
        "hill_lo": [hill_estimator(tr[:, j], 0.05, "lower") for j in range(f)],
        "hill_hi": [hill_estimator(tr[:, j], 0.05, "upper") for j in range(f)],
        "roughness": rough,
        "q_flattest": best_q,
    })
    d.to_csv(a.csv, index=False)
    print(f"\nxi_lower > 0.5 at q=0.05: {int((d['xi_lo@0.05'] > 0.5).sum())} of {f}"
          f"   at q=0.10: {int((d['xi_lo@0.10'] > 0.5).sum())}")
    print(f"median xi_lower  q=0.05 {d['xi_lo@0.05'].median():.3f}   "
          f"q=0.10 {d['xi_lo@0.10'].median():.3f}")
    print(f"suggested q (flattest xi): median {np.nanmedian(best_q):.3f}")
    print(f"per-feature table -> {a.csv}")

    if a.valors:
        sel = [names.index(s.strip()) for s in a.valors.split(",")]
    else:
        sel = list(np.argsort(-rough)[:a.n_panels])       # least stable first
        print(f"\nshowing the {len(sel)} least stable features "
              f"(pass --valors to choose)")

    ncol = min(3, len(sel)); nrow = math.ceil(len(sel) / ncol)
    fig, axes = plt.subplots(nrow, 2 * ncol, figsize=(5.4 * ncol, 3.4 * nrow),
                             squeeze=False)
    for p_i, j in enumerate(sel):
        rr, cc = divmod(p_i, ncol)
        # Hill plot: alpha_hat as a function of k
        ax = axes[rr][2 * cc]
        ks = np.unique(np.linspace(10, int(0.25 * n), 60).astype(int))
        for tail, col in (("lower", "C0"), ("upper", "C1")):
            al = [hill_estimator(tr[:, j], k / n, tail) for k in ks]
            ax.plot(ks, 1.0 / np.array(al), color=col, lw=1.2, label=tail)
        ax.axvline(0.05 * n, ls="--", color="grey", lw=.8)
        ax.axvline(0.10 * n, ls=":", color="grey", lw=.8)
        ax.set_title(f"{names[j]}: Hill", fontsize=9)
        ax.set_xlabel("k"); ax.set_ylabel(r"$1/\hat\alpha$")
        ax.legend(fontsize=7)
        # xi(q) from the GPD MLE actually used downstream
        ax = axes[rr][2 * cc + 1]
        for tail, col in (("lower", "C0"), ("upper", "C1")):
            ax.plot(qs, XI[tail][j], "o-", ms=3, color=col, lw=1.2, label=tail)
        ax.axhline(0.5, ls="--", color="r", lw=.8)
        ax.axvline(0.05, ls="--", color="grey", lw=.8)
        ax.axvline(0.10, ls=":", color="grey", lw=.8)
        ax.set_title(f"{names[j]}: GPD " + r"$\hat\xi(q)$", fontsize=9)
        ax.set_xlabel("q"); ax.set_ylabel(r"$\hat\xi$")
        ax.legend(fontsize=7)
    for ax in axes.ravel()[2 * len(sel):]:
        ax.axis("off")
    fig.suptitle("Threshold stability -- flat region = GPD valid and estimate stable "
                 "(dashed q=0.05, dotted q=0.10, red line xi=0.5)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=140); plt.close(fig)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
