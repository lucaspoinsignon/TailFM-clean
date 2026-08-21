```bash

python -c "
import numpy as np
from csvio import load_returns
from tailfm import make_windows
from tailfm.evaluate import acf
r = load_returns('data/returns_clean.csv', False)
real = make_windows(r[:int(0.8*len(r))], 24, 1)
gen  = np.load('runs/nu5/generated_windows.npy')
rng = np.random.default_rng(0)
sh = real.copy()
for i in range(sh.shape[0]):
    sh[i] = sh[i][rng.permutation(24)]      # destroy time order, keep everything else
for j in (0, 10, 43, 100):
    print(f'feat {j:3d}  real {np.round(acf(real[:,:,j]**2,3),3)}'
          f'  shuffled {np.round(acf(sh[:,:,j]**2,3),3)}'
          f'  gen {np.round(acf(gen[:,:,j]**2,3),3)}')
"



```
