"""Validate a CSV before handing it to fit_returns.py.

    python check.py --data data/prices.csv --prices     # after step 1
    python check.py --data data/returns.csv             # after step 2

Exits non-zero if anything would break the run, so it can gate a script.

Checks, in the order they would bite:

  header      A bare numeric header (VALOR ids with no prefix) parses as floats,
              so np.loadtxt reads it as row 0 and every id becomes a return of
              ~1e6.  No error is raised.  feature_names_from_csv falling back to
              "feat0, feat1, ..." is the tell.
  positivity  load_returns takes np.log, so a non-positive price yields -inf or
              nan.  Reported per column, since one bad column is worth dropping
              and a hundred means the file is not prices.
  finiteness  after differencing.
  EVT         evt.SemiParametricMarginal fits a GPD to the exceedances strictly
              beyond the q-quantile of each column.  A point mass at the minimum
              -- accrual NAVs that never fall, coarsely quantised quotes -- makes
              that set empty and scipy raises "zero-size array to reduction
              operation minimum".  Rather than infer this, the fit is actually
              run, on the same training rows the model will use.

The GPD fits also give the xi estimates before committing to a run: xi > 0.5
means infinite variance and a threshold worth revisiting, xi < 0 asserts a hard
floor on losses, which for a daily-marked instrument usually means the price is
smoothed or appraisal-based rather than traded.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from scipy import stats

from csvio import load_returns, feature_names_from_csv

FAIL: list[str] = []
WARN: list[str] = []


def fail(msg: str) -> None:
    FAIL.append(msg)
    print(f"  FAIL  {msg}")


def warn(msg: str) -> None:
    WARN.append(msg)
    print(f"  WARN  {msg}")


def ok(msg: str) -> None:
    print(f"  ok    {msg}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--prices", action="store_true",
                   help="--data holds prices, not returns (same flag as fit_returns)")
    p.add_argument("--q-tail", type=float, default=0.05,
                   help="must match fit_returns.py --q-tail")
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--min-exceedances", type=int, default=30)
    p.add_argument("--no-fit", action="store_true",
                   help="skip the actual genpareto fits (structure checks only)")
    a = p.parse_args()
    print(f"checking {a.data}  (--prices={a.prices}, q_tail={a.q_tail})\n")

    # ---- header -------------------------------------------------------------
    # Load the raw columns first (prices=False never takes a log), so a
    # non-positive price is reported by name instead of surfacing as a bare
    # "non-finite after loading" from inside np.log.
    print("header")
    try:
        lv = load_returns(a.data, False)
    except SystemExit as e:
        fail(str(e))
        sys.exit(1)
    f = lv.shape[1]
    names = feature_names_from_csv(a.data, f)
    if names and names[0].startswith("feat") and not a.data.endswith(".npy"):
        fail("no usable header: names came back as feat0, feat1, ... .  If the "
             "header is bare VALOR numbers it was parsed as DATA row 0 -- every "
             "id became a return of ~1e6.  Prefix the names (V4156860) and rebuild.")
    else:
        ok(f"{f} named columns, e.g. {names[:3]}")
        if all(nm.replace(".", "").replace("-", "").isdigit() for nm in names):
            warn("column names are purely numeric.  They parse here only because "
                 "a non-numeric Date column is present; without it np.loadtxt "
                 "would read the header as data.  Prefix them (V4156860).")

    # ---- levels -------------------------------------------------------------
    if a.prices:
        print("\nprices")
        nonpos = np.flatnonzero((lv <= 0).any(axis=0))
        if nonpos.size:
            fail(f"{nonpos.size} columns contain a non-positive price, so log is "
                 f"undefined: {[names[j] for j in nonpos[:6]]}")
            print(f"\n{'-' * 60}\nFAILED: fix or drop those columns, then re-run.")
            sys.exit(1)
        ok(f"all {f} columns strictly positive (min {lv.min():.6g})")
        if np.abs(lv).mean() < 0.5:
            warn(f"mean |level| is {np.abs(lv).mean():.3g} -- this looks like "
                 "returns, not prices.  Drop --prices?")

    r = load_returns(a.data, a.prices)
    T, f = r.shape
    names = feature_names_from_csv(a.data, f)
    ok(f"loads as (T={T}, f={f})")

    # ---- returns -----------------------------------------------------------
    print("\nreturns")
    if not np.isfinite(r).all():
        fail(f"{int((~np.isfinite(r)).sum())} non-finite returns")
    else:
        ok(f"all finite, sd {r.std():.4g}, range [{r.min():.4g}, {r.max():.4g}]")
    if np.abs(r).mean() > 1.0:
        fail(f"mean |return| is {np.abs(r).mean():.3g}.  Log returns are ~1e-3; "
             "this file is probably price levels -- add --prices.")

    tr = r[:int((1.0 - a.test_frac) * T)]
    ok(f"training rows {tr.shape[0]}  ->  {int(a.q_tail * tr.shape[0])} "
       f"exceedances per tail per column")

    # ---- EVT ---------------------------------------------------------------
    print("\nEVT feasibility (on the training rows, as fit_returns sees them)")
    u_lo = np.quantile(tr, a.q_tail, axis=0)
    u_hi = np.quantile(tr, 1.0 - a.q_tail, axis=0)
    n_lo = (tr < u_lo).sum(axis=0)
    n_hi = (tr > u_hi).sum(axis=0)
    empty = np.flatnonzero((n_lo == 0) | (n_hi == 0))
    thin = np.flatnonzero(((n_lo < a.min_exceedances) | (n_hi < a.min_exceedances))
                          & ~np.isin(np.arange(f), empty))
    if empty.size:
        fail(f"{empty.size} columns have an EMPTY exceedance set -- "
             f"genpareto.fit will raise: {[names[j] for j in empty[:6]]}")
    else:
        ok(f"every column has >=1 exceedance per tail "
           f"(min lower {n_lo.min()}, min upper {n_hi.min()})")
    if thin.size:
        warn(f"{thin.size} columns have <{a.min_exceedances} exceedances in a "
             f"tail; the GPD fit will be unstable: {[names[j] for j in thin[:6]]}")

    nuniq = np.array([len(np.unique(tr[:, j])) for j in range(f)])
    coarse = np.flatnonzero(nuniq < 0.5 * tr.shape[0])
    if coarse.size:
        warn(f"{coarse.size} columns have <50% distinct return values "
             f"(quantised/stale): {[names[j] for j in coarse[:6]]}")

    if not a.no_fit and not empty.size:
        print("\nrunning the actual GPD fits")
        xi_lo, xi_hi, errs = [], [], []
        for j in range(f):
            for side, thr, sgn in ((xi_lo, u_lo[j], -1.0), (xi_hi, u_hi[j], 1.0)):
                exc = sgn * (tr[:, j] - thr)
                exc = exc[exc > 0]
                try:
                    side.append(stats.genpareto.fit(exc, floc=0.0)[0])
                except Exception as e:
                    errs.append(f"{names[j]}: {type(e).__name__}: {e}")
                    side.append(np.nan)
        if errs:
            fail(f"{len(errs)} GPD fits raised, e.g. {errs[0]}")
        else:
            ok(f"{2 * f} GPD fits succeeded")
        xi_lo, xi_hi = np.array(xi_lo), np.array(xi_hi)
        for nm, xi in (("lower", xi_lo), ("upper", xi_hi)):
            print(f"    xi_{nm}: median {np.nanmedian(xi):+.3f}  "
                  f"[{np.nanmin(xi):+.3f}, {np.nanmax(xi):+.3f}]  "
                  f"| xi>0.5: {int((xi > 0.5).sum())}  xi<0: {int((xi < 0).sum())}")
        if (xi_lo > 0.5).any():
            warn(f"{int((xi_lo > 0.5).sum())} columns with xi_lower > 0.5 "
                 "(infinite variance) -- check the data or raise --q-tail")
        if (xi_lo < -0.1).any():
            warn(f"{int((xi_lo < -0.1).sum())} columns with xi_lower < -0.1: the "
                 "fitted GPD has a finite left endpoint, i.e. a hard floor on "
                 "losses.  Usually smoothed or appraisal-based pricing.")

    print(f"\n{'-' * 60}")
    if FAIL:
        print(f"FAILED: {len(FAIL)} blocking issue(s), {len(WARN)} warning(s).")
        print("fit_returns.py will crash or produce invalid output on this file.")
        sys.exit(1)
    print(f"PASSED with {len(WARN)} warning(s).  Ready for fit_returns.py.")


if __name__ == "__main__":
    main()
