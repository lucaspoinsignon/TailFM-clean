```bash

python -c "
import pandas as pd
p = pd.read_csv('data/prices.csv', index_col=0, parse_dates=True)
bad = p.columns[(p <= 0).any(axis=0)]
for c in bad:
    z = p[c][p[c] <= 0]
    print(f'{c}  n_nonpos {len(z)}  min {p[c].min():.6g}  dates {[d.date() for d in z.index[:5]]}')
print(','.join(bad))
"

```
