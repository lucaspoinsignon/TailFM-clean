```bash

python -c "
import pandas as pd
p = pd.read_csv('data/prices.csv', index_col=0, parse_dates=True)
c = p.notna().mean()
print('per-column coverage: min %.3f  q25 %.3f  median %.3f  max %.3f' % (c.min(), c.quantile(.25), c.median(), c.max()))
n = p.notna().sum(axis=1)
print('rows with <50%% of VALOR quoted: %d of %d' % (int((n < 0.5*p.shape[1]).sum()), len(p)))
print(n.describe())
"

```
