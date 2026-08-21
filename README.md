```bash

cd /repos/quail
python -c "
import torch
sd = torch.load('runs/nu5/model_ema.pt', map_location='cpu')
p = sd['pos'][0]
print('pos std:', p.std().item(), ' (init was 0.02)')
print('per-step norms:', [round(v,3) for v in p.norm(dim=-1).tolist()])
"



```
