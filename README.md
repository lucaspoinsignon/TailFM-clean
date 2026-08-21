```bash

cd /repos/quail
python fit_returns.py --data data/returns_clean.csv --nu 5 --q-tail 0.05 \
    --n 24 --test-frac 0.2 --horizon 10 --seed 0 --steps 20000 --gen 20000 \
    --d-model 512 --ode-steps 100 --mix-phi 0.95 --no-figures --outdir runs/nu5_phi095


for CFG in "1.0 0.02 old" "1.0 0.5 pos" "0.95 0.02 phi" "0.95 0.5 both"; do
  set -- $CFG
  python fit_returns.py --data data/returns_clean.csv --nu 5 --q-tail 0.05 \
      --n 24 --test-frac 0.2 --horizon 10 --seed 0 --steps 20000 --gen 20000 \
      --d-model 512 --ode-steps 100 --mix-phi $1 --pos-std $2 \
      --no-figures --no-report --outdir runs/ab_$3
done


python 05_diagnose.py --data data/returns_clean.csv --run runs/nu5_phi08 --eval train


```
