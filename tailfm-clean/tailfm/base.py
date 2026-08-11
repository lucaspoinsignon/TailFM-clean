"""Heavy-tailed source distribution for flow matching.

Multivariate Student-t via normal variance mixing:

    x_0 = z * sqrt(nu / W),   z ~ N(0, I),   W ~ chi^2_nu,

with a SINGLE draw of W shared across a group of coordinates. Sharing the mixing
variable makes the group jointly elliptically t-distributed, which has strictly
positive tail-dependence coefficient

    lambda = 2 * t_{nu+1}( -sqrt((nu+1)(1-rho)/(1+rho)) ) > 0   for any rho > -1,

in contrast to the Gaussian, whose tail dependence is exactly 0. The flow therefore
only has to *modulate* joint-extreme intensity per pair (up or down, including down to
near-independence), instead of manufacturing it from a source that has none.

mix_dim:
  "window": one W per sample (B, 1, 1) -- tail dependence across features AND time
            (extremes cluster within a window; consistent with volatility clustering).
  "time":   one W per timestep (B, n, 1) -- cross-feature tail dependence only.
"""

from __future__ import annotations

import torch


def sample_base(batch: int, n: int, f: int, nu: float,
                mix_dim: str = "window",
                device: torch.device | str = "cpu",
                generator: torch.Generator | None = None) -> torch.Tensor:
    z = torch.randn(batch, n, f, device=device, generator=generator)
    w_shape = (batch, 1, 1) if mix_dim == "window" else (batch, n, 1)
    # chi^2_nu = Gamma(shape=nu/2, rate=1/2)
    w = torch.distributions.Gamma(nu / 2.0, 0.5).sample(w_shape).to(device)
    return z * torch.sqrt(torch.as_tensor(nu, device=device) / w.clamp_min(1e-8))
