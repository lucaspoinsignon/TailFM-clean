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
    p.add_argument("--q-tail", default="0.05",
                   help="float, or 'auto' for per-feature threshold selection; "
                        "must match fit_returns.py --q-tail")
    p.add_argument("--nu", default="5.0",
                   help="t_nu latent space, or 'auto'; must match fit_returns.py")
    p.add_argument("--no-shrink", action="store_true",
                   help="disable empirical-Bayes pooling of xi (match fit_returns)")
    p.add_argument("--shrink-c", type=float, default=1.0,
                   help="Efron-Morris limited-translation cap, in standard errors")
    p.add_argument("--n-boot", type=int, default=99,
                   help="bootstrap replicates for the auto goodness-of-fit test")
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--min-exceedances", type=int, default=30)
    p.add_argument("--no-fit", action="store_true",
                   help="skip the actual genpareto fits (structure checks only)")
    a = p.parse_args()
    q_tail = a.q_tail if a.q_tail == "auto" else float(a.q_tail)
    nu = a.nu if a.nu == "auto" else float(a.nu)
    print(f"checking {a.data}  (--prices={a.prices}, q_tail={q_tail}, nu={nu}, "
          f"shrink={'off' if a.no_shrink else f'c={a.shrink_c}'})\n")

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
    if q_tail == "auto":
        ok(f"training rows {tr.shape[0]}  ->  threshold chosen per feature and tail")
    else:
        ok(f"training rows {tr.shape[0]}  ->  {int(q_tail * tr.shape[0])} "
           f"exceedances per tail per column")

    # ---- EVT ----------------------------------------------------------------
    # Fit the marginals for real, on the training rows, exactly as fit_returns
    # does.  Inferring feasibility from exceedance counts is not enough: the fit
    # can succeed and still transform an observation to |z| ~ 1e4, which is what
    # destroys training, and only fitting reveals it.
    print("\nEVT: fitting marginals on the training rows")
    from tailfm.evt import MarginalEnsemble
    from evt_shrink import shrink_ensemble
    try:
        marg = MarginalEnsemble(q_tail=q_tail, nu=nu, n_boot=a.n_boot).fit(tr)
    except ValueError as e:
        fail(str(e))
        print(f"\n{'-' * 60}\nFAILED: fit_returns.py will raise on this file.")
        sys.exit(1)
    ok(f"{2 * f} GPD fits succeeded  (nu = {marg.nu_:.3f})")
    if not a.no_shrink:
        # Same pooling fit_returns.py applies, so this gate validates the
        # marginals the model is actually trained on.
        print("    pooling xi across features (empirical Bayes):")
        shrink_ensemble(marg, tr, c=a.shrink_c)
    su = marg.summary()

    if q_tail == "auto":
        for side in ("lo", "hi"):
            qv, cnt = np.unique(su[f"q_{side}"], return_counts=True)
            print(f"    q_{side} selected: "
                  + "  ".join(f"{v:.2f}x{c}" for v, c in zip(qv, cnt)))
        nfail = int((~su["gof_ok_lo"]).sum() + (~su["gof_ok_hi"]).sum())
        if nfail:
            warn(f"{nfail} tails where no threshold passed the AD test; the "
                 "best-fitting q was used and those marginals are suspect")
    print(f"    exceedances: lower {su['n_exc_lo'].min()}-{su['n_exc_lo'].max()}, "
          f"upper {su['n_exc_hi'].min()}-{su['n_exc_hi'].max()}")
    for side, xi in (("lower", su["xi_lo"]), ("upper", su["xi_hi"])):
        print(f"    xi_{side}: median {np.median(xi):+.3f}  "
              f"[{xi.min():+.3f}, {xi.max():+.3f}]  "
              f"| xi>1: {int((xi > 1).sum())}  xi<0: {int((xi < 0).sum())}")

    # ---- the operational check ----------------------------------------------
    z = marg.transform(tr)
    mz = np.abs(z).max(axis=0)
    if not np.isfinite(z).all():
        fail(f"{int((~np.isfinite(z)).sum())} non-finite values after the PIT")
    bad = np.flatnonzero(mz > 100.0)
    if bad.size:
        fail(f"{bad.size} features transform to |z| > 100.  A correct fit keeps "
             f"max|z| under ~100 at any tail index (p99 = 44 in simulation), so "
             f"these marginals are mis-fitted and will dominate the training "
             f"gradient: "
             + ", ".join(f"{names[j]}({mz[j]:.0f})" for j in bad[:6]))
    else:
        ok(f"max|z| = {mz.max():.1f} over all features (correct fits stay under ~100)")
    warnz = np.flatnonzero((mz > 50.0) & (mz <= 100.0))
    if warnz.size:
        warn(f"{warnz.size} features with 50 < max|z| <= 100: plausible but on the "
             "edge of what a correct fit produces")

    print(f"\n{'-' * 60}")
    if FAIL:
        print(f"FAILED: {len(FAIL)} blocking issue(s), {len(WARN)} warning(s).")
        print("fit_returns.py will crash or produce invalid output on this file.")
        sys.exit(1)
    print(f"PASSED with {len(WARN)} warning(s).  Ready for fit_returns.py.")


if __name__ == "__main__":
    main()
