```bash

# 1. baseline for comparison (your current marginals, unpooled)
python check.py --data data/returns_clean.csv --nu 5.0 --no-shrink
python -m evtdiag.ks --data data/returns_clean.csv --q-tail 0.05 \
    --nu 5.0 --csv evt_ks_base.csv --out fig/ks_base.png

# 2. with pooling
python check.py --data data/returns_clean.csv --nu 5.0

# 3. the decisive test: out-of-sample AD, which shrinkage cannot game
python -m evtdiag.ks --data data/returns_clean.csv --q-tail 0.05 \
    --nu 5.0 --csv evt_ks_shrunk.csv --out fig/ks_shrunk.png

python -c "
import pandas as pd
a = pd.read_csv('evt_ks_base.csv'); b = pd.read_csv('evt_ks_shrunk.csv')
for nm, d in (('unpooled', a), ('pooled  ', b)):
    print(f'{nm}  AD_test median {d.AD_test.median():6.3f}  mean {d.AD_test.mean():7.3f}'
          f'  rejected@5% {int((d.p_test<0.05).sum()):3d}/{len(d)}'
          f'  max|z| {d.max_abs_z.max():7.1f}')
"

# 4. fit with the settings that won
python fit_returns.py --data data/returns_clean.csv --nu 5.0 --q-tail 0.05 \
    --n 24 --test-frac 0.2 --horizon 10 --seed 0 --steps 20000 --gen 50000 \
    --d-model 512 --ode-steps 100 --outdir runs/nu5

# 5. the nu ablation
for NU in 3 5 8; do
  python fit_returns.py --data data/returns_clean.csv --nu $NU --q-tail 0.05 \
      --n 24 --test-frac 0.2 --horizon 10 --seed 0 --steps 20000 --gen 50000 \
      --d-model 512 --ode-steps 100 --outdir runs/nu$NU
  echo "=== nu=$NU ==="
  python 05_diagnose.py --data data/returns_clean.csv --run runs/nu$NU --eval test
done

```
