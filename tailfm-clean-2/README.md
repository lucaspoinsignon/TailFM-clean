# TailFM pipeline

Each step is one command. Steps 1–4 need only numpy/pandas/scipy/matplotlib;
steps 5–7 also need torch.

```bash
pip install numpy pandas scipy matplotlib torch
```

---

## 1. Raw extract → one price series per VALOR

Input: semicolon-separated `VALOR;FI_ID;PRICE_TYPE;PRICE_DATE;PRICE;CURRENCY`.
A VALOR appearing under several PRICE_TYPEs or CURRENCYs is collapsed to one
series (the combination with the most observations).

```bash
python 01_panel.py --raw data/raw.csv --out data/prices.csv
```

```bash
python 01_panel.py --raw data/raw.csv --out data/prices.csv --currency GBP --price-type 90122 --start 2019-12-23 --end 2026-01-05
```

Check the printed currency mix. Mixed currencies mean the columns are in
different units and their returns are not comparable.
Writes `data/prices_selected.csv` recording which quote was kept per VALOR.

Validate the panel — strictly positive prices, and every column actually
GPD-fittable. Exits non-zero if `fit_returns.py` would crash on it:

```bash
python check.py --data data/prices.csv --prices
```

## 2. Prices → log returns

Drops series that break the EVT stage (point mass at the minimum → empty GPD
exceedance set), stale quotes, and near-duplicate columns.

```bash
python 02_returns.py --data data/prices.csv --out data/returns.csv
```

```bash
python 02_returns.py --data data/prices.csv --out data/returns.csv --max-zero-frac 0.05 --min-exceedances 30 --dedup-rho 0.99 --max-assets 30
```

`--dedup-rho 1` turns deduplication off.

Re-validate. This must pass before step 5:

```bash
python check.py --data data/returns.csv
```

## 3. Plot the series

```bash
python 03_plot.py --data data/returns.csv --out fig/returns.png
```

```bash
python 03_plot.py --data data/prices.csv --out fig/prices.png
python 03_plot.py --data data/prices.csv --out fig/two.png --valors V4156860,V4156861
python 03_plot.py --data data/prices.csv --out fig/r.png --to-returns --overlay
```

## 4. Dependence structure

Rank correlation, eigenvalue spectrum with effective rank, distribution of
`lambda(q0)` over all pairs, most tail-dependent pairs.

```bash
python 04_analyse.py --data data/returns.csv --out fig/dependence.png
```

## 5. Fit the model

```bash
python fit_returns.py --data data/returns.csv --n 24 --test-frac 0.2 --horizon 10 --seed 0 --steps 20000 --gen 50000 --outdir runs/main
```

```bash
python fit_returns.py --data data/returns.csv --n 24 --test-frac 0.2 --horizon 10 --seed 0 --steps 20000 --gen 50000 --d-model 512 --q-tail 0.05 --ode-steps 100 --outdir runs/main
```

Add `--prices` if `--data` holds price levels instead of returns.
Writes `report.log`, `generated_windows.npy`, `model_ema.pt`, `marginals.pkl`
and the diagnostic PNGs into `--outdir`.

## 6. Did it get the tail dependence right?

Compares generated vs real `lambda(q0)` against a resampling noise floor.

```bash
python 05_diagnose.py --data data/returns.csv --run runs/main --q0 0.05
```

```bash
python 05_diagnose.py --data data/returns.csv --run runs/main --eval test
```

Read `slope` and `sd_gen/sd_real`, not the per-pair error — at ~1200 rows the
per-pair error is swamped by noise even when the generator is badly wrong.

## 7. Real vs generated windows

```bash
python 06_compare.py --data data/returns.csv --run runs/main --out fig/windows.png
```

```bash
python 06_compare.py --data data/returns.csv --run runs/main --out fig/worst.png --valors V4156860,V4156861,V4156862,V4156863 --n-windows 2 --rank 1.0 --cumulative
```

`--rank 1.0` shows the worst window in each sample; the default spreads picks
across matched loss quantiles.

## 8. Baselines (optional)

```bash
python run_baselines.py --data data/returns.csv --n 24 --test-frac 0.2 --horizon 10 --seed 0 --outdir runs/baselines
```

```bash
python run_baselines.py --data data/returns.csv --models timevae,timegan,tailgan --tailfm-gen runs/main/generated_windows.npy --outdir runs/baselines
```

---

## Files

| | |
|---|---|
| `01_panel.py` … `06_compare.py` | pipeline steps |
| `check.py` | validates a prices or returns CSV; run after steps 1 and 2 |
| `csvio.py` | CSV loader shared by steps 4–7, mirrors `fit_returns.load_returns` |
| `fit_returns.py` | model runner (step 5) |
| `run_baselines.py`, `baselines/` | baseline generators (step 8) |
| `figures.py` | diagnostic PNGs, shared by steps 5 and 8 |
| `run_logging.py` | tees stdout to `report.log` |
| `tailfm/` | the model: `evt`, `base`, `model`, `cfm`, `risk`, `evaluate`, `data` |

`figures.tail_dependence_figure` ranks pairs by
`|lambda_gen(q0) - lambda_real(q0)|` and plots only the worst `max_pairs` (200).
At f = 300 there are 44 850 pairs and the unranked version does not terminate.
