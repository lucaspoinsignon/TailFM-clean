```bash


python -m evtdiag.ks --data data/returns_clean.csv --q-tail 0.05 --nu 5.0 \
    --shrink-c 0.4 --csv evt_ks_c04.csv --out fig/ks_c04.png

python -c "
import pandas as pd
for nm, f in (('c=1.0','evt_ks_shrunk.csv'), ('c=0.4','evt_ks_c04.csv')):
    d = pd.read_csv(f)
    print(f'{nm}  AD_test median {d.AD_test.median():6.3f}  max|z| {d.max_abs_z.max():7.1f}'
          f'  (>50: {int((d.max_abs_z>50).sum())})  xi_lo<0: {int((d.xi_lo<0).sum())}')
"

```
