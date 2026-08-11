```bash

python -c "
import pandas as pd, subprocess
p = pd.read_csv('data/prices.csv', index_col=0, parse_dates=True)
bad = ','.join(p.columns[(p <= 0).any(axis=0)])
subprocess.run(['python','03_plot.py','--data','data/prices.csv','--out','fig/nonpos.png','--valors',bad])
"

```
