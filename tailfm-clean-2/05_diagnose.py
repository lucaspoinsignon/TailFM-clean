"""Step 5.  Is the generated tail dependence wrong, or is the sample just too small?

Compares three lambda_hat(q0) distributions over all f(f-1)/2 pairs:

    real     train rows of --data (the same rows fit_returns fitted on)
    gen      run dir's generated_windows.npy
    null     real split in half; lambda(half A) vs lambda(half B)

`null` is the key column: it is what a *perfect* model would score, because it
contains only sampling noise.  If |gen - real| is comparable to |A - B|, the
model is as accurate as the sample can resolve and no amount of training or
capacity will improve the picture.  If |gen - real| is much larger, the gap is
real and worth chasing.

Also reports the cross-pair relation between lambda_real and lambda_gen:
correlation says whether the model ranks pairs correctly at all, the regression
slope says whether it systematically shrinks (slope < 1) or inflates (> 1) joint
extremes.

    python 05_diagnose.py --data data/returns.csv --run runs/main
"""

from __future__ import annotations

import argparse

import numpy as np

from csvio import load_returns


def uniform_scores(x: np.ndarray) -> np.ndarray:
    n, f = x.shape
    u = np.empty((n, f), dtype=np.float32)
    for j in range(f):
        u[:, j] = (np.argsort(np.argsort(x[:, j])) + 1.0) / (n + 1.0)
    return u


def lam(u: np.ndarray, q0: float) -> np.ndarray:
    """f x f matrix of lambda_hat(q0) = P(U_i<q0, U_j<q0)/q0."""
    b = (u < q0).astype(np.float32)
    return (b.T @ b) / (q0 * b.shape[0])


def desc(name: str, v: np.ndarray) -> None:
    q = np.quantile(v, [0.05, 0.5, 0.95])
    print(f"  {name:22s} mean {v.mean():6.3f}  sd {v.std():6.3f}  "
          f"q05 {q[0]:6.3f}  med {q[1]:6.3f}  q95 {q[2]:6.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--run", required=True, help="outdir holding generated_windows.npy")
    p.add_argument("--q0", type=float, default=0.05)
    p.add_argument("--prices", action="store_true",
                   help="--data holds prices, not returns (same flag as fit_returns)")
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--eval", choices=["train", "test"], default="train",
                   help="rows to score the generator against.  'train' measures "
                        "fit, 'test' measures generalisation -- the one that "
                        "separates capacity from overfitting")
    p.add_argument("--max-rows", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    r = load_returns(a.data, a.prices)
    T, f = r.shape
    split = int((1.0 - a.test_frac) * T)              # same split as fit_returns
    train = r[:split] if a.eval == "train" else r[split:]
    g = np.load(f"{a.run}/generated_windows.npy").reshape(-1, f)
    rng = np.random.default_rng(a.seed)
    if g.shape[0] > a.max_rows:
        g = g[np.sort(rng.choice(g.shape[0], a.max_rows, replace=False))]
    print(f"{a.eval} rows {train.shape}  generated rows {g.shape}  q0={a.q0}\n")

    iu = np.triu_indices(f, 1)
    Lr = lam(uniform_scores(train), a.q0)[iu]
    Lg = lam(uniform_scores(g), a.q0)[iu]

    # noise floor: two independent halves of the same real sample
    idx = rng.permutation(train.shape[0])
    h = train.shape[0] // 2
    La = lam(uniform_scores(train[idx[:h]]), a.q0)[iu]
    Lb = lam(uniform_scores(train[idx[h:2 * h]]), a.q0)[iu]

    print(f"lambda_hat({a.q0}) over {iu[0].size} pairs "
          f"(independence would give {a.q0:.3f}):")
    desc(f"real ({a.eval})", Lr)
    desc("generated", Lg)
    print("\nabsolute error per pair:")
    desc("|gen - real|", np.abs(Lg - Lr))
    desc("|halfA - halfB| (null)", np.abs(La - Lb))
    ratio = np.abs(Lg - Lr).mean() / max(np.abs(La - Lb).mean(), 1e-12)
    print(f"\n  error / noise floor = {ratio:.2f}   "
          f"({'at the resolution limit' if ratio < 1.5 else 'real gap'})")

    sd_r, sd_g = Lr.std(), Lg.std()
    rho = np.corrcoef(Lr, Lg)[0, 1]
    slope = np.polyfit(Lr, Lg, 1)[0]
    print(f"\ncross-pair agreement:  corr(real, gen) = {rho:.3f}   "
          f"slope = {slope:.3f}   sd_gen/sd_real = {sd_g / sd_r:.3f}")
    print("  slope < 1  => joint extremes systematically shrunk toward the base")
    print("  corr ~ 0   => pair-specific structure not learned at all")


if __name__ == "__main__":
    main()
