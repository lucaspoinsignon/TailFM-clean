```bash

python -c "
import numpy as np, pickle
from csvio import load_returns
m = pickle.load(open('runs/main/marginals.pkl','rb'))
r = load_returns('data/returns.csv', False)
z = m.transform(r[:int(0.8*len(r))])
print('non-finite z:', int((~np.isfinite(z)).sum()), ' max |z|:', np.abs(z[np.isfinite(z)]).max())
bad = np.argwhere(~np.isfinite(z))
print('rows/cols:', bad[:10])
"

```
