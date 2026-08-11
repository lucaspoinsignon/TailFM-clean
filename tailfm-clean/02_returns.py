"""Step 2.  Price panel -> log returns, filtered so the EVT stage can run.

    python 02_returns.py --data data/prices.csv --out data/returns.csv

Four filters, each removing series that break a specific downstream assumption:

  coverage      a column missing too much of the calendar drags every other
                column down when rows are inner-joined
  zero fraction stale or coarsely-quantised quotes; their "empirical body" is a
                handful of knots and the interpolated marginal is meaningless
  EVT feasible  evt.SemiParametricMarginal fits a GPD to x[x < u_lo] with u_lo
                the q-quantile.  A point mass at the minimum -- accrual NAVs
                that never fall -- makes that set empty and genparetoicfit raises
                "zero-size array to reduction operation minimum".  This is the
                one filter that is not optional.
  near-duplicate  columns with |rho_S| ~ 1 are the same instrument twice.  A CFM
                pushes a full-support base through a diffeomorphism, so its law
                is absolutely continuous and cannot concentrate on the comonotone
                curve; these pairs are unreachable by construction and dominate
                the worst tail-dependence panels while carrying no information.
                |rho_S| rather than rho_S, so inverted quotes collapse too.

Clustering uses the training rows only, so the reduction never sees the test
period.  Filters are reported one line each; nothing is dropped silently.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def pseudo_obs(x: np.ndarray) -> np.ndarray:
    n, f = x.shape
    u = np.empty((n, f))
    for j in range(f):
        u[:, j] = (np.argsort(np.argsort(x[:, j])) + 1.0) / (n + 1.0)
    return u


def drop(px, ret, cols, why):
    if len(cols) == 0:
        return px, ret
    print(f"  drop {len(cols):4d}  {why:<22s} e.g. {list(cols[:4])}")
    return px.drop(columns=cols), ret.drop(columns=cols)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="wide price panel from 01_panel.py")
    p.add_argument("--out", required=True)
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--ffill-limit", type=int, default=2,
                   help="patch holes up to this many days; 0 = pure inner join")
    p.add_argument("--min-coverage", type=float, default=0.98)
    p.add_argument("--max-zero-frac", type=float, default=0.05)
    p.add_argument("--q-tail", type=float, default=0.05,
                   help="must match fit_returns.py --q-tail")
    p.add_argument("--min-exceedances", type=int, default=30)
    p.add_argument("--min-unique-frac", type=float, default=0.5)
    p.add_argument("--dedup-rho", type=float, default=0.99,
                   help="|Spearman| at or above this merges two columns; 1 = off")
    p.add_argument("--dedup-lam", type=float, default=0.90)
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--max-assets", type=int, default=0, help="0 = keep all")
    a = p.parse_args()

    px = pd.read_csv(a.data, index_col=0, parse_dates=True).sort_index()
    if a.start:
        px = px[px.index >= pd.Timestamp(a.start)]
    if a.end:
        px = px[px.index <= pd.Timestamp(a.end)]
    print(f"{a.data}: {px.shape[0]} dates x {px.shape[1]} VALOR  "
          f"{px.index[0].date()} -> {px.index[-1].date()}")

    cov = px.notna().mean()
    px = px.drop(columns=cov.index[cov < a.min_coverage])
    print(f"  drop {int((cov < a.min_coverage).sum()):4d}  "
          f"coverage < {a.min_coverage}")

    if a.ffill_limit > 0:
        px = px.ffill(limit=a.ffill_limit)
    before = len(px)
    px = px.dropna(how="any")
    print(f"  rows {before} -> {len(px)} after ffill(limit={a.ffill_limit}) + dropna")
    px = px.loc[:, (px > 0).all(axis=0)]

    ret = np.log(px / px.shift(1)).dropna(how="any")
    print(f"returns: {ret.shape[0]} x {ret.shape[1]}")

    zf = (ret == 0.0).mean()
    px, ret = drop(px, ret, ret.columns[zf > a.max_zero_frac],
                   f"zero frac > {a.max_zero_frac}")

    q = a.q_tail
    lo = (ret < ret.quantile(q)).sum()
    hi = (ret > ret.quantile(1 - q)).sum()
    nu = ret.nunique() / len(ret)
    px, ret = drop(px, ret,
                   ret.columns[(lo < a.min_exceedances) | (hi < a.min_exceedances)
                               | (nu < a.min_unique_frac)],
                   "EVT infeasible")
    if ret.shape[1] == 0:
        raise SystemExit("everything filtered out; loosen the thresholds")

    if a.dedup_rho < 1.0:
        tr = ret.iloc[:int((1 - a.test_frac) * len(ret))].to_numpy()
        u = pseudo_obs(tr)
        uc = (u - u.mean(0)) / u.std(0)
        rho = (uc.T @ uc) / len(tr)
        b = (u < q).astype(float)
        lam = (b.T @ b) / (q * len(tr))
        adj = (np.abs(rho) >= a.dedup_rho) | (lam >= a.dedup_lam)
        np.fill_diagonal(adj, False)
        ncomp, label = connected_components(csr_matrix(adj), directed=False)
        nuniq = np.array([ret.iloc[:, j].nunique() for j in range(ret.shape[1])])
        var = tr.var(0)
        keep = sorted(idx[np.lexsort((var[idx], nuniq[idx]))[-1]]
                      for idx in (np.flatnonzero(label == c) for c in range(ncomp)))
        px, ret = drop(px, ret,
                       ret.columns[np.setdiff1d(np.arange(ret.shape[1]), keep)],
                       f"near-duplicate (|rho|>={a.dedup_rho})")

    if a.max_assets and ret.shape[1] > a.max_assets:
        keep = (ret == 0.0).mean().nsmallest(a.max_assets).index
        keep = [c for c in ret.columns if c in set(keep)]
        print(f"  cap  {ret.shape[1] - len(keep):4d}  --max-assets {a.max_assets}")
        px, ret = px[keep], ret[keep]

    ret.index.name = "Date"
    ret.to_csv(a.out, float_format="%.8e")
    T, f = ret.shape
    tr = int((1 - a.test_frac) * T)
    print(f"\nwrote {a.out}: T={T}  f={f}  {ret.index[0].date()} -> {ret.index[-1].date()}")
    print(f"train rows {tr}  exceedances/tail {int(a.q_tail * tr)}  "
          f"n*f at n=24 is {24 * f} vs {tr // 24} non-overlapping windows")


if __name__ == "__main__":
    main()
