"""Fit tail-aware flow matching on your own multivariate return series.

Usage:
    python fit_returns.py --data returns.csv --n 24 --steps 20000 --gen 50000

Input: a CSV (one column per feature, optional header, rows = time steps) or a .npy
array of shape (T, f). Values are log returns; pass --prices if the file contains
prices instead (log returns are then computed as diff(log(P))).

Pipeline: temporal train/test split -> EVT marginals on train -> PIT -> CFM training
(GPU if available) -> sampling -> inverse PIT -> tail diagnostics, portfolio VaR/CVaR
with bootstrap CIs, Kupiec backtest on the held-out period -> figures + saved artifacts.

Everything printed is also written to {outdir}/report.log (override with --log), and
the figures are one PNG per diagnostic (figures.py): qq_lower_tail.png,
tail_dependence.png, portfolio_loss_survival.png, empirical_distributions.png.
"""

from __future__ import annotations

import argparse
import os
import pickle

import numpy as np
import torch

from evt_shrink import shrink_ensemble
from figures import save_all_figures
from run_logging import tee_output
from tailfm import (MarginalEnsemble, VelocityField, train_cfm, sample,
                    estimate_risk, kupiec_test, portfolio_losses, make_windows,
                    print_report)


def load_returns(path: str, prices: bool) -> np.ndarray:
    if path.endswith(".npy"):
        arr = np.load(path)
    else:
        try:
            arr = np.loadtxt(path, delimiter=",")
        except ValueError:                       # header row and/or Date column
            def _is_float(tok: str) -> bool:
                try:
                    float(tok)
                    return True
                except ValueError:
                    return False
            with open(path) as fh:
                rows = [ln.strip().split(",") for ln in fh if ln.strip()]
            if not all(_is_float(t) for t in rows[0]):
                rows = rows[1:]                  # drop header
            keep = [j for j, t in enumerate(rows[0]) if _is_float(t)]
            arr = np.array([[float(r[j]) for j in keep] for r in rows])
    arr = np.atleast_2d(np.asarray(arr, dtype=float))
    if arr.shape[0] < arr.shape[1]:
        arr = arr.T
    if prices:
        arr = np.diff(np.log(arr), axis=0)
    if not np.isfinite(arr).all():
        raise ValueError("Non-finite values in the return matrix; clean the data first.")
    return arr


def feature_names_from_csv(path: str, f: int) -> list[str]:
    """Use the CSV header for labels if one is present."""
    if path.endswith(".npy"):
        return [f"feat{j}" for j in range(f)]
    with open(path) as fh:
        first = fh.readline().strip().split(",")
    try:
        [float(v) for v in first if v]                    # no header
        return [f"feat{j}" for j in range(f)]
    except ValueError:
        names = [c for c in first if c.lower() != "date"]
        return names if len(names) == f else [f"feat{j}" for j in range(f)]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CSV or .npy of shape (T, f)")
    ap.add_argument("--prices", action="store_true", help="input is prices, not returns")
    ap.add_argument("--n", type=int, default=24, help="window length")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--q-tail", default="0.05",
                    help="float, or 'auto' for per-feature threshold selection")
    ap.add_argument("--nu", default="5.0",
                    help="degrees of freedom of the t_nu latent space, or 'auto' "
                         "(Hill median).  The PIT makes every marginal exactly "
                         "t_nu for ANY nu, so this is a design parameter, not an "
                         "estimate: nu <= 4 gives the CFM target x1-x0 an "
                         "infinite fourth moment and hence infinite-variance "
                         "gradients.  nu=5 is the smallest value with E z^4 < inf.")
    ap.add_argument("--no-shrink", action="store_true",
                    help="disable empirical-Bayes pooling of the per-feature xi")
    ap.add_argument("--shrink-c", type=float, default=1.0,
                    help="Efron-Morris limited-translation cap, in standard "
                         "errors; inf = unrestricted posterior mean")
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--gen", type=int, default=50_000, help="# generated windows")
    ap.add_argument("--ode-steps", type=int, default=100)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--weights", type=str, default=None,
                    help="comma-separated portfolio weights (default: equal)")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--no-recalibrate", action="store_true",
                    help="disable rank-recalibration of generated marginals")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-figures", action="store_true",
                    help="skip save_all_figures.  The PNGs are the only part of a "
                         "run that nothing downstream reads: 05_diagnose.py and "
                         "06_compare.py need generated_windows.npy alone.  Useful "
                         "for a sweep -- draw figures afterwards for the winner.")
    ap.add_argument("--no-report", action="store_true",
                    help="skip print_report (Hill / tail dependence / VaR / ACF "
                         "tables).  Same rationale: nothing downstream reads them.")
    ap.add_argument("--outdir", type=str, default="run_out")
    ap.add_argument("--log", type=str, default=None,
                    help="text file receiving a copy of everything printed "
                         "(default: {outdir}/report.log)")
    return ap.parse_args()


def run(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.outdir, exist_ok=True)
    alphas = (0.95, 0.99, 0.995)

    # ------------------------------------------------------------------- data
    r = load_returns(args.data, args.prices)
    T, f = r.shape
    names = feature_names_from_csv(args.data, f)
    split = int((1.0 - args.test_frac) * T)
    train_r, test_r = r[:split], r[split:]
    real = make_windows(train_r, args.n, args.stride)
    print(f"data: T={T}, f={f} ({', '.join(names)}) | "
          f"train windows {real.shape} | device={device}")

    w = (np.array([float(v) for v in args.weights.split(",")])
         if args.weights else np.full(f, 1.0 / f))
    assert w.size == f, "--weights length must equal the number of features"

    # -------------------------------------------------- EVT marginals + PIT
    q_tail = args.q_tail if args.q_tail == "auto" else float(args.q_tail)
    nu = args.nu if args.nu == "auto" else float(args.nu)
    # Fit on train_r, NOT on `real`: make_windows overlaps at stride 1, so every
    # interior date appears n times in the pooled (N*n, f) array.  That leaves the
    # MLE roughly unchanged but multiplies the exceedance count by ~n, which
    # invalidates the goodness-of-fit bootstrap, breaks the (1+xi)^2/k variance
    # used for pooling, and makes these marginals differ from the ones check.py
    # and evtdiag.ks validate.
    marg = MarginalEnsemble(q_tail=q_tail, nu=nu).fit(train_r)
    if not args.no_shrink:
        print("EVT: pooling xi across features (empirical Bayes)")
        shrink_ensemble(marg, train_r, c=args.shrink_c)
    xi_lo = [round(m.xi_lo_, 3) for m in marg.marginals_]
    print(f"EVT: nu={marg.nu_:.2f}, xi_lower={xi_lo}  "
          f"(daily equities typically ~0.1-0.4; >~0.5 => check data/threshold)")
    z = torch.tensor(marg.transform(real), dtype=torch.float32)

    # --------------------------------------------------------------- training
    model = VelocityField(f=f, n_max=args.n, d_model=args.d_model, depth=args.depth)
    ema, _ = train_cfm(model, z, nu=marg.nu_, steps=args.steps,
                       batch_size=args.batch, device=device, seed=args.seed)
    torch.save(ema.shadow.state_dict(), f"{args.outdir}/model_ema.pt")
    pickle.dump(marg, open(f"{args.outdir}/marginals.pkl", "wb"))

    # --------------------------------------------------------------- sampling
    z_gen = sample(ema.shadow, args.gen, args.n, f, nu=marg.nu_,
                   n_steps=args.ode_steps, device=device, seed=args.seed)
    gen = marg.inverse_transform(z_gen.numpy())
    del z_gen                       # 1.13 GB at --gen 50000; held through diagnostics
    if not args.no_recalibrate:
        # Rank-recalibration: replace each feature's pooled marginal by exactly F_hat_j
        # via the increasing map x -> F_hat_j^{-1}(rank/(K+1)). Leaves the learned copula
        # invariant (Sklar), so the flow keeps only the dependence while the EVT
        # marginals keep the tails -- restoring the marginal guarantee exactly.
        flat = gen.reshape(-1, f)
        K = flat.shape[0]
        rank = np.arange(1, K + 1, dtype=float) / (K + 1.0)
        for a in range(0, f, 32):                      # column blocks, cache-friendly
            b = min(a + 32, f)
            blk = np.ascontiguousarray(flat[:, a:b].T)
            idx = np.argsort(blk, axis=1)
            u = np.empty(blk.shape)
            np.put_along_axis(u, idx, np.broadcast_to(rank, blk.shape), axis=1)
            for j in range(a, b):
                flat[:, j] = marg.marginals_[j].ppf(u[j - a])
        gen = flat.reshape(args.gen, args.n, f)
    np.save(f"{args.outdir}/generated_windows.npy", gen)

    # ------------------------------------------------------------ diagnostics
    if not args.no_report:
        print_report(real, gen, feature_names=names)

    # ------------------------------------------------------------ risk report
    report = estimate_risk(gen, alphas=alphas, weights=w, horizon=args.horizon,
                           n_boot=200, seed=args.seed)
    L_test = portfolio_losses(make_windows(test_r, args.horizon, stride=args.horizon),
                              weights=w, horizon=args.horizon)
    print(f"\n=== Portfolio risk (h={args.horizon}) and Kupiec backtest "
          f"(held-out N={L_test.size}) ===")
    for a in alphas:
        rp, k = report[a], kupiec_test(L_test, report[a]["var_gpd"], a)
        print(f"a={a:5.3f}: VaR {rp['var_gpd']:.5f} "
              f"[{rp['var_ci'][0]:.5f},{rp['var_ci'][1]:.5f}]  "
              f"CVaR {rp['cvar_gpd']:.5f} "
              f"[{rp['cvar_ci'][0]:.5f},{rp['cvar_ci'][1]:.5f}]  | "
              f"exceed {k['exceedances']}/{k['expected']:.1f}  p={k['p_value']:.3f}")

    # ----------------------------------------------------------------- figures
    # One PNG per diagnostic, same conventions as run_baselines.py (figures.py).
    paths = ([] if args.no_figures else
             save_all_figures(real, {"tailfm": gen}, names, args.outdir,
                              weights=w, horizon=args.horizon))
    print(f"\nSaved: {args.outdir}/{{model_ema.pt, marginals.pkl, "
          f"generated_windows.npy}}"
          + ("" if not paths else ", " + ", ".join(os.path.basename(p) for p in paths)))


def main():
    args = parse_args()
    log_path = args.log or f"{args.outdir}/report.log"
    with tee_output(log_path, header="fit_returns.py"):
        run(args)
    print(f"Terminal report saved to {log_path}")


if __name__ == "__main__":
    main()
