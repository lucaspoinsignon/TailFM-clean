```bash

python -c "
import numpy as np, pandas as pd
from csvio import load_returns, feature_names_from_csv
r = load_returns('data/returns.csv', False); T,f = r.shape
n = feature_names_from_csv('data/returns.csv', f)
s = int(0.8*T); tr, te = r[:s], r[s:]
d = pd.DataFrame({'VALOR': n,
  'sd_train': tr.std(0), 'sd_test': te.std(0),
  'ratio': te.std(0)/tr.std(0),
  'n_beyond': [(te[:,j]<tr[:,j].min()).sum()+(te[:,j]>tr[:,j].max()).sum() for j in range(f)]})
print(d.sort_values('ratio', ascending=False).head(15).to_string(index=False, float_format='%.4g'))
print(f'\nfeatures with sd_test/sd_train > 3: {(d.ratio>3).sum()}   > 10: {(d.ratio>10).sum()}')
print(f'median ratio {d.ratio.median():.2f}   median n_beyond {d.n_beyond.median():.0f} of {len(te)}')
"

```
