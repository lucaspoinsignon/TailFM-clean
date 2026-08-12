```bash

python -c "
import numpy as np, pandas as pd
r = pd.read_csv('data/returns_clean.csv', index_col=0, parse_dates=True)
c = 'PUT_VALOR_HERE'
tr = r[c].iloc[:1236].to_numpy()
u = np.quantile(tr, 0.10)
exc = np.sort(u - tr[tr < u])[::-1]
print(f'threshold {u:.4g}, {exc.size} exceedances')
print('largest 10:', np.round(exc[:10], 5))
print('ratio top/2nd:', round(exc[0]/exc[1], 2))
"
```
