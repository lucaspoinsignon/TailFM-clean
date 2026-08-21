```bash

cd /repos/quail
for PS in 0.1 0.2 0.3; do
  python fit_returns.py --data data/returns_clean.csv --nu 5 --q-tail 0.05 \
      --n 24 --test-frac 0.2 --horizon 10 --seed 0 --steps 20000 --gen 20000 \
      --d-model 512 --ode-steps 100 --mix-phi 1.0 --pos-std $PS \
      --no-figures --no-report --outdir runs/ps$PS
done
python summarize_runs.py --data data/returns_clean.csv \
    runs/ab_old runs/ps0.1 runs/ps0.2 runs/ps0.3 runs/ab_pos runs/ab_both



```
