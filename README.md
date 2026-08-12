```bash

python -c "
import numpy as np, pickle
from csvio import load_returns
m = pickle.load(open('runs/dim512_200feat/marginals.pkl','rb'))
r = load_returns('data/returns.csv', False)
z = m.transform(r[:int(0.8*len(r))])
print('nu =', m.nu_, ' max|z| =', np.abs(z).max())
mx = np.abs(z).max(axis=0)
o = np.argsort(-mx)[:10]
print('worst columns by max|z|:', [(int(j), round(float(mx[j]),1)) for j in o])
xi = [(mm.xi_lo_, mm.xi_hi_) for mm in m.marginals_]
lo = np.array([a for a,_ in xi]); hi = np.array([b for _,b in xi])
print('xi_lower: max %.3f  n>0.5: %d' % (lo.max(), (lo>0.5).sum()))
print('xi_upper: max %.3f  n>0.5: %d' % (hi.max(), (hi>0.5).sum()))
"

```
