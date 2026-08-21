```bash


python -m evtdiag.ks --data data/returns_clean.csv --q-tail 0.05 --nu 5.0 \
    --csv evt_ks_floor.csv --out fig/ks_floor.png

python -c "
import pandas as pd
for nm, f in (('c=1.0 no floor','evt_ks_shrunk.csv'), ('c=1.0 + floor ','evt_ks_floor.csv')):
    d = pd.read_csv(f)
    print(f'{nm}  AD_test median {d.AD_test.median():6.3f}  max|z| {d.max_abs_z.max():7.1f}'
          f'  (>50: {int((d.max_abs_z>50).sum())})  xi_lo<0: {int((d.xi_lo<0).sum())}'
          f'  xi_hi<0: {int((d.xi_hi<0).sum())}')
"

```
