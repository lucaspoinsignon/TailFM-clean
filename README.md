```bash

python fit_returns.py --data data/returns_clean.csv \
    --nu 5 --q-tail 0.05 --mix-phi 1.0 --pos-std 0.1 \
    --n 24 --test-frac 0.2 --horizon 10 --seed 0 \
    --steps 20000 --gen 20000 --d-model 512 --ode-steps 100 \
    --outdir runs/final

```
