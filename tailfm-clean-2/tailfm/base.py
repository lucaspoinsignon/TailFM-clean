"""Heavy-tailed source distribution for flow matching.

Multivariate Student-t via normal variance mixing:

    x_0 = z * sqrt(nu / W),   z ~ N(0, I),   W ~ chi^2_nu,

with W SHARED across a group of coordinates. Sharing the mixing variable makes the
group jointly elliptically t-distributed, which has strictly positive tail-dependence
coefficient

    lambda = 2 * t_{nu+1}( -sqrt((nu+1)(1-rho)/(1+rho)) ) > 0   for any rho > -1,

in contrast to the Gaussian, whose tail dependence is exactly 0. The flow therefore
only has to *modulate* joint-extreme intensity per pair (up or down, including down to
near-independence), instead of manufacturing it from a source that has none.

TEMPORAL STRUCTURE OF W (mix_phi).  The same "modulate, don't manufacture" argument
applies to volatility clustering, and the original two options both fail it -- for the
same reason, from opposite ends:

    mix_dim="window"  one W for the whole window.  All n timesteps are scaled by the
                      SAME constant, and evaluate.acf de-means and normalises within
                      each window, so a common scale factor cancels exactly.
    mix_dim="time"    one W per timestep, independent.  No persistence at all.

Both give a within-window squared-return ACF of exactly the iid null, -1/(n-1).
Measured on the base alone at n=24, nu=5 (lag 1):

    phi   0.00    0.50    0.80    0.90    0.95    0.99    1.00
    ACF  -0.044  +0.005  +0.019  +0.009  -0.006  -0.032  -0.043
          ^ mix_dim="time"                               ^ mix_dim="window"

Only the INTERIOR of the range produces clustering: the scale has to vary within the
window *and* be persistent.  mix_phi makes that a continuum, with the two original
options as its endpoints.  W_t is coupled by a Gaussian copula,

    g_t   AR(1) with corr(g_t, g_s) = phi^|t-s|,     W_t = F^{-1}_{chi^2_nu}(Phi(g_t)),

so every W_t is EXACTLY chi^2_nu marginally and every coordinate of x_0 is exactly
t_nu.  The z-space marginal match with the EVT PIT is therefore untouched -- mix_phi
changes only the dependence the base donates, never its marginals.

W is shared across features at each timestep, so cross-sectional tail dependence is
retained at every phi; phi controls how much of it also persists through time.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy import stats

_CHOL: dict[tuple[int, float], torch.Tensor] = {}


def _ar1_chol(n: int, phi: float, device, dtype) -> torch.Tensor:
    """Cholesky factor of the AR(1) correlation matrix phi^|t-s|, cached per (n, phi)."""
    key = (n, round(float(phi), 6))
    L = _CHOL.get(key)
    if L is None:
        t = np.arange(n)
        L = torch.tensor(np.linalg.cholesky(phi ** np.abs(t[:, None] - t[None, :])),
                         dtype=torch.float64)
        _CHOL[key] = L
    return L.to(device=device, dtype=dtype)


def resolve_phi(mix_dim: str = "window", mix_phi: float | None = None) -> float:
    """mix_phi if given, else the phi implied by mix_dim (1.0 window, 0.0 time)."""
    if mix_phi is not None:
        return float(np.clip(mix_phi, 0.0, 1.0))
    return 1.0 if mix_dim == "window" else 0.0


def sample_base(batch: int, n: int, f: int, nu: float,
                mix_dim: str = "window", mix_phi: float | None = None,
                device: torch.device | str = "cpu",
                generator: torch.Generator | None = None) -> torch.Tensor:
    z = torch.randn(batch, n, f, device=device, generator=generator)
    phi = resolve_phi(mix_dim, mix_phi)

    if phi >= 1.0:                      # one W per window  (mix_dim="window")
        # chi^2_nu = Gamma(shape=nu/2, rate=1/2)
        w = torch.distributions.Gamma(nu / 2.0, 0.5).sample((batch, 1, 1)).to(device)
    elif phi <= 0.0:                    # one W per timestep (mix_dim="time")
        w = torch.distributions.Gamma(nu / 2.0, 0.5).sample((batch, n, 1)).to(device)
    else:                               # AR(1)-coupled W_t, exact chi^2_nu margins
        e = torch.randn(batch, n, device=device, generator=generator)
        g = e @ _ar1_chol(n, phi, device, e.dtype).T          # corr = phi^|t-s|
        u = 0.5 * (1.0 + torch.erf(g / np.sqrt(2.0)))         # -> U(0,1) marginally
        u = u.clamp(1e-9, 1.0 - 1e-9).cpu().numpy()
        # scipy is used for the chi^2 inverse CDF only; all randomness is torch-side
        # above, so this stays reproducible under torch.manual_seed.  ~1.6 ms per
        # training batch, ~32 s over a 20k-step run.
        w = torch.tensor(stats.chi2.ppf(u, nu), dtype=z.dtype,
                         device=device).unsqueeze(-1)

    return z * torch.sqrt(torch.as_tensor(nu, device=device) / w.clamp_min(1e-8))
