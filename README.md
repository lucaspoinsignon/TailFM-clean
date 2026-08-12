```bash

python -c "
import torch
from tailfm import sample_base
for nu in (2.5, 3.0, 4.0):
    m = 0.0; bad = 0
    for _ in range(200):
        x = sample_base(128, 24, 251, nu, mix_dim='window', device='cpu')
        m = max(m, x.abs().max().item()); bad += int((~torch.isfinite(x)).sum())
    print(f'nu={nu}: max|x0| over 25600 draws = {m:.4g}, non-finite {bad}')
"


```
