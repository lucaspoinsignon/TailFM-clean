```bash

python -m evtdiag.ks --data data/returns_clean.csv --q-tail 0.10 --csv evt_ks_q10.csv --out fig/ks_q10.png
python -m evtdiag.ks --data data/returns_clean.csv --q-tail auto --n-boot 99 --csv evt_ks_auto.csv --out fig/ks_auto.png

python -c "
import pandas as pd
a = pd.read_csv('evt_ks_q10.csv').set_index('VALOR')
b = pd.read_csv('evt_ks_auto.csv').set_index('VALOR')
for nm, d in (('q=0.10', a), ('auto', b)):
    print(f'{nm:7s}  AD_test median {d.AD_test.median():6.3f}  mean {d.AD_test.mean():7.3f}  max {d.AD_test.max():8.1f}'
          f'   rejected@5% {int((d.p_test<0.05).sum()):4d}/{len(d)}'
          f'   max|z| {d.max_abs_z.max():8.1f}  (>100: {int((d.max_abs_z>100).sum())})')
imp = (a.AD_test - b.AD_test)
print(f'\nauto better on {int((imp>0).sum())}/{len(a)} features, worse on {int((imp<0).sum())}')
print('\nbiggest improvements:'); print((imp.sort_values(ascending=False).head(8)).to_string(float_format='%.2f'))
print('\nbiggest regressions:');  print((imp.sort_values().head(5)).to_string(float_format='%.2f'))
"

python -c "
import pandas as pd
a = pd.read_csv('evt_ks_q10.csv'); b = pd.read_csv('evt_ks_auto.csv')
for nm, d in (('q=0.10', a), ('auto', b)):
    print(f'{nm:7s}  xi_lo median {d.xi_lo.median():+.3f}  max {d.xi_lo.max():+.3f}  |  xi>1: {int((d.xi_lo>1).sum())}  xi>0.5: {int((d.xi_lo>0.5).sum())}')
"
```
