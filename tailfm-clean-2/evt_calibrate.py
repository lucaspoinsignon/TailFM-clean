"""Is `149 of 235 rejected` evidence of bad marginals, or a broken test?

    python evt_calibrate.py --data data/returns_clean.csv --nu 5.0

`evtdiag.ks` scores each feature with `stats.kstest(u_te, "uniform")`, whose null
assumes the PIT values are iid.  They are not: u_t = F(r_t) inherits the
volatility clustering of r_t, so the empirical CDF of u wanders far more than the
iid null allows and the p-value is anti-conservative.  Simulated with a
CORRECTLY specified marginal on n=309 rows of a stochastic-volatility t_5
process, nominal 5% rejects at:

    log-vol persistence     median KS   95th pct   rejected @5%
      none (iid)              0.047      0.077        5.3%
      phi=0.90 s=0.20         0.057      0.093       16.8%
      phi=0.95 s=0.15         0.063      0.104       28.2%
      phi=0.98 s=0.12         0.080      0.150       53.8%

so a 60%-ish rejection rate is what a perfect marginal produces on realistic
daily data.  Nothing can be concluded from the p-value alone.

Two things are computed instead.

PLACEBO.  The train window is itself split 80/20; marginals are fitted on the
first part and scored on the second, exactly as the real split is scored.  Both
halves come from the same period, so any excess of the real rejection rate over
the placebo rate is attributable to the held-out period, and the placebo rate is
a valid empirical null that already contains the serial dependence.

TAIL COVERAGE.  KS and AD are dominated by the body, which is the interpolated
empirical CDF of the training rows and is untouched by anything done to the GPD
parameters.  To decide a question about the TAILS, count how many held-out
observations fall beyond each fitted quantile:

    N_p = #{ t in test, j : F_j(r_tj) < p },     E[N_p] = p * n_test * f

pooled over features, with a binomial reference interval.  Ratios above 1 mean
the fitted tail is too thin (more exceedances than promised), below 1 too fat.
This is the criterion the shrinkage choice should be made on.
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy import stats

from csvio import load_returns, feature_names_from_csv
from tailfm.evt import MarginalEnsemble
from evt_shrink import shrink_ensemble

LEVELS = (0.01, 0.025, 0.05, 0.10)


def fit(x, q_tail, nu, shrink_c):
    m = MarginalEnsemble(q_tail=q_tail, nu=nu).fit(x)
    if shrink_c is not None:
        shrink_ensemble(m, x, c=shrink_c, verbose=False)
    return m


def ks_rejection_rate(marg, te) -> tuple[float, float]:
    ks = np.array([stats.kstest(m.cdf(te[:, j]), "uniform").statistic
                   for j, m in enumerate(marg.marginals_)])
    p = np.array([stats.kstest(marg.marginals_[j].cdf(te[:, j]), "uniform").pvalue
                  for j in range(len(marg.marginals_))])
    return float(np.median(ks)), float((p < 0.05).mean())


def tail_coverage(marg, te) -> dict:
    u = np.stack([m.cdf(te[:, j]) for j, m in enumerate(marg.marginals_)], axis=1)
    n = u.size
    out = {}
    for p in LEVELS:
        for side, cnt in (("lower", int((u < p).sum())),
                          ("upper", int((u > 1.0 - p).sum()))):
            exp = p * n
            lo, hi = stats.binom.ppf([0.025, 0.975], n, p)
            out[(side, p)] = (cnt, exp, cnt / exp, lo / exp, hi / exp)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--prices", action="store_true")
    ap.add_argument("--q-tail", default="0.05")
    ap.add_argument("--nu", default="5.0")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--shrink-c", type=float, default=1.0)
    a = ap.parse_args()
    q_tail = a.q_tail if a.q_tail == "auto" else float(a.q_tail)
    nu = a.nu if a.nu == "auto" else float(a.nu)

    r = load_returns(a.data, a.prices)
    T, f = r.shape
    split = int((1.0 - a.test_frac) * T)
    tr, te = r[:split], r[split:]
    print(f"{a.data}: train {tr.shape}  test {te.shape}  nu={nu}  q_tail={q_tail}\n")

    # ---------------------------------------------------------------- placebo
    isplit = int((1.0 - a.test_frac) * split)
    m_pl = fit(tr[:isplit], q_tail, nu, None)
    ks_pl, rej_pl = ks_rejection_rate(m_pl, tr[isplit:])
    m_re = fit(tr, q_tail, nu, None)
    ks_re, rej_re = ks_rejection_rate(m_re, te)
    print("KS on held-out rows (no shrinkage):")
    print(f"  placebo  fit {tr[:isplit].shape[0]} -> score {split - isplit:4d} rows"
          f"   median KS {ks_pl:.3f}   rejected@5% {100*rej_pl:5.1f}%")
    print(f"  real     fit {split:4d} -> score {T - split:4d} rows"
          f"   median KS {ks_re:.3f}   rejected@5% {100*rej_re:5.1f}%")
    print("  -> the placebo rate is the empirical null; only the EXCESS is a finding.\n")

    # ------------------------------------------------------- scale shift check
    s_tr = np.median(np.abs(tr - np.median(tr, axis=0)), axis=0)
    s_te = np.median(np.abs(te - np.median(te, axis=0)), axis=0)
    ratio = s_te / np.maximum(s_tr, 1e-300)
    print(f"robust scale ratio test/train over {f} features: "
          f"median {np.median(ratio):.2f}  q05 {np.quantile(ratio, .05):.2f}  "
          f"q95 {np.quantile(ratio, .95):.2f}")
    print("  (a common factor far from 1.0 means the split straddles a volatility "
          "regime, which no unconditional marginal can absorb)\n")

    # --------------------------------------------------------- tail coverage
    for tag, c in (("no shrinkage", None), (f"shrunk c={a.shrink_c}", a.shrink_c)):
        m = fit(tr, q_tail, nu, c)
        cov = tail_coverage(m, te)
        print(f"tail coverage on held-out rows -- {tag}")
        print(f"  {'level':>7} {'side':>6} {'obs':>7} {'exp':>8} {'obs/exp':>9}"
              f"   {'95% band':>16}")
        for p in LEVELS:
            for side in ("lower", "upper"):
                cnt, exp, rat, lo, hi = cov[(side, p)]
                flag = "" if lo <= rat <= hi else "   <-- outside"
                print(f"  {p:7.3f} {side:>6} {cnt:7d} {exp:8.0f} {rat:9.2f}"
                      f"   [{lo:.2f}, {hi:.2f}]{flag}")
        print()
    print("Pick the variant whose obs/exp sits closer to 1 at p = 0.01 and 0.025.")


if __name__ == "__main__":
    main()
