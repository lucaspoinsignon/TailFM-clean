"""The loader fit_returns.py uses, without the torch import.

04/05/06 only read CSVs, so importing fit_returns for its loader would pull in
torch and the whole model package for nothing.  These two functions must stay
behaviourally identical to fit_returns.load_returns / feature_names_from_csv --
if they diverge, the diagnostics silently score against different rows than the
ones the model was fitted on.
"""

from __future__ import annotations

import numpy as np


def _is_float(tok: str) -> bool:
    try:
        float(tok)
        return True
    except ValueError:
        return False


def load_returns(path: str, prices: bool) -> np.ndarray:
    """(T, f) log returns.  Drops a header row and any non-numeric Date column."""
    if path.endswith(".npy"):
        arr = np.load(path)
    else:
        try:
            arr = np.loadtxt(path, delimiter=",")
        except ValueError:                       # header row and/or Date column
            with open(path) as fh:
                rows = [ln.strip().split(",") for ln in fh if ln.strip()]
            if not all(_is_float(t) for t in rows[0]):
                rows = rows[1:]
            keep = [j for j, t in enumerate(rows[0]) if _is_float(t)]
            arr = np.array([[float(r[j]) for j in keep] for r in rows])
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if prices:
        arr = np.diff(np.log(arr), axis=0)
    if not np.isfinite(arr).all():
        raise SystemExit(f"{path}: non-finite values after loading")
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
