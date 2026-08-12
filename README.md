```bash

python -c "
import numpy as np, pickle, pandas as pd
from csvio import load_returns, feature_names_from_csv
m = pickle.load(open('runs/dim512_200feat/marginals.pkl','rb'))
p = 'data/returns.csv'
r = load_returns(p, False); f = r.shape[1]
names = feature_names_from_csv(p, f)
tr = r[:int(0.8*len(r))]
z = m.transform(tr)
d = pd.DataFrame({
    'VALOR': names,
    'xi_lo': [mm.xi_lo_ for mm in m.marginals_],
    'xi_hi': [mm.xi_hi_ for mm in m.marginals_],
    'max_abs_z': np.abs(z).max(axis=0),
    'sd': tr.std(0), 'min': tr.min(0), 'max': tr.max(0),
    'n_uniq': [len(np.unique(tr[:,j])) for j in range(f)],
})
d = d.sort_values('xi_lo', ascending=False)
print(d.head(25).to_string(index=False, float_format='%.4g'))
d.to_csv('evt_fits.csv', index=False)
print(f'\n{(d.xi_lo>0.5).sum()} of {f} with xi_lo>0.5 -- full table in evt_fits.csv')
"

python -c "
import numpy as np, pandas as pd, pickle
from csvio import load_returns, feature_names_from_csv
m = pickle.load(open('runs/dim512_200feat/marginals.pkl','rb'))
p='data/returns.csv'; r=load_returns(p,False); names=feature_names_from_csv(p,r.shape[1])
j = names.index('PUT_THE_VALOR_HERE')
tr = r[:int(0.8*len(r))][:,j]
mm = m.marginals_[j]
exc = mm.u_lo_ - tr[tr < mm.u_lo_]
print(f'threshold {mm.u_lo_:.4g}, {len(exc)} exceedances, xi {mm.xi_lo_:.3f}')
print('largest exceedances:', np.sort(exc)[-8:])
dates = pd.read_csv(p, index_col=0, parse_dates=True).index[:len(tr)]
k = np.argsort(tr)[:5]
print('worst days:', [(str(dates[i].date()), round(float(tr[i]),5)) for i in k])
"

```
