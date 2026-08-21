```bash

python fit_returns.py --data data/returns_clean.csv --nu 5 --q-tail 0.05 \
    --n 24 --test-frac 0.2 --horizon 10 --seed 0 --steps 20000 --gen 20000 \
    --d-model 512 --ode-steps 100 --no-figures --outdir runs/nu5_phi08


python -c "
import numpy as np
from csvio import load_returns
from tailfm import make_windows
from tailfm.evaluate import acf
r = load_returns('data/returns_clean.csv', False)
real = make_windows(r[:int(0.8*len(r))], 24, 1)
rng = np.random.default_rng(0); sh = real.copy()
for i in range(sh.shape[0]): sh[i] = sh[i][rng.permutation(24)]
old = np.load('runs/nu5/generated_windows.npy')
new = np.load('runs/nu5_phi08/generated_windows.npy')
A = lambda W: np.array([acf(W[:,:,j]**2,3) for j in range(W.shape[2])]).mean(0)
for nm, W in (('real',real),('shuffled',sh),('old (phi=1)',old),('new (phi=0.8)',new)):
    print(f'{nm:16s} {np.round(A(W),4)}')
"



```
