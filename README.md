```bash

python check.py --data data/returns_clean.csv --q-tail 0.10

python -m evtdiag.ks --data data/returns_clean.csv --q-tail 0.10 --out fig/ks2.png

python 04_analyse.py --data data/returns_clean.csv --out fig/dep2.png

```
