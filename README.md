```bash

for NU in 3 5 8; do
  python fit_returns.py --data data/returns_clean.csv --nu $NU --q-tail 0.05 \
      --n 24 --test-frac 0.2 --horizon 10 --seed 0 --steps 20000 --gen 20000 \
      --d-model 512 --ode-steps 100 --outdir runs/nu$NU
  echo "=== nu=$NU ==="
  python 05_diagnose.py --data data/returns_clean.csv --run runs/nu$NU --eval test
done

```
