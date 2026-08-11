```bash

python -c "
import pandas as pd, numpy as np
px = pd.read_csv('data/prices.csv', index_col=0, parse_dates=True).sort_index()
per = px.notna().sum(axis=1); cal = per[per >= 0.5*px.shape[1]].index
px = px.loc[cal]

cov = px.notna().mean()
low = cov[cov < 0.98].sort_values()
print('=== COVERAGE (%d dropped) ===' % len(low))
print(low.to_string())

ok = px[cov[cov >= 0.98].index].ffill(limit=2).dropna(how='any')
neg = ok.columns[(ok <= 0).any(axis=0)]
print('\n=== NON-POSITIVE PRICE (%d dropped) ===' % len(neg))
for c in neg:
    bad = ok[c][ok[c] <= 0]
    print(f'  {c}: {len(bad)} cells, min {ok[c].min():.6g}, first {bad.index[0].date()}')

ok = ok.drop(columns=neg)
r = np.log(ok/ok.shift(1)).dropna()
zf = (r==0).mean().sort_values(ascending=False)
print('\n=== ZERO FRACTION > 0.05 (%d dropped) ===' % int((zf>0.05).sum()))
print(zf[zf>0.05].to_string())
" 2>&1 | head -120

```
