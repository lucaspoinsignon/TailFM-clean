```bash

python summarize_runs.py --data data/returns_clean.csv \
    runs/nu5 runs/nu5_phi08 runs/nu5_phi095 \
    runs/ab_old runs/ab_pos runs/ab_phi runs/ab_both


python -c "
import numpy as np
from csvio import load_returns
from tailfm import make_windows
from tailfm.evaluate import acf
r = load_returns('data/returns_clean.csv', False)
real = make_windows(r[:int(0.8*len(r))], 24, 1)
A = lambda W: np.array([acf(W[:,:,j]**2,3) for j in range(W.shape[2])]).mean(0)+1/23
print('real          ', np.round(A(real),4))
for nm in ('nu5','nu5_phi08','nu5_phi095'):
    print(f'{nm:14s}', np.round(A(np.load(f'runs/{nm}/generated_windows.npy')),4))
"



```
