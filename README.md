```bash

python -c "
import numpy as np, pandas as pd
from csvio import load_returns, feature_names_from_csv
r = load_returns('data/returns.csv', False); f = r.shape[1]
n = feature_names_from_csv('data/returns.csv', f)
sd = r.std(0)
d = pd.DataFrame({'VALOR': n, 'sd': sd, 'n_uniq': [len(np.unique(r[:,j])) for j in range(f)]})
print(d.nsmallest(10, 'sd').to_string(index=False, float_format='%.4g'))
print('features with sd < 1e-4:', int((sd < 1e-4).sum()))
"

```
