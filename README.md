```bash

pip uninstall -y pyarrow

python -m evtdiag.ks --data data/returns_clean.csv --q-tail 0.05 --nu 5.0 \
    --no-shrink --csv evt_ks_base.csv --out fig/ks_base.png

python -m evtdiag.ks --data data/returns_clean.csv --q-tail 0.05 --nu 5.0 \
    --csv evt_ks_shrunk.csv --out fig/ks_shrunk.png

python -c "
import pandas as pd
for nm, f in (('unpooled','evt_ks_base.csv'), ('pooled  ','evt_ks_shrunk.csv')):
    d = pd.read_csv(f)
    print(f'{nm}  AD_test median {d.AD_test.median():6.3f}  mean {d.AD_test.mean():7.3f}'
          f'  rejected@5% {int((d.p_test<0.05).sum()):3d}/{len(d)}'
          f'  max|z| {d.max_abs_z.max():7.1f}  (>50: {int((d.max_abs_z>50).sum())})')
"

python evt_calibrate.py --data data/returns_clean.csv --nu 5.0

python -c "
import pandas as pd
d = pd.read_csv('evt_ks_base.csv')
print(d.loc[d.max_abs_z > 50, ['VALOR','xi_lo','xi_hi','n_exc_lo','n_exc_hi','max_abs_z','n_beyond']].to_string(index=False))
"

```
