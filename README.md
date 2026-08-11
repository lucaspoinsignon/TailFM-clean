```bash

# Reference calendar: dates where a quorum of VALOR is quoted.  Weekend and
    # holiday rows survive the pivot in 01_panel.py whenever a single VALOR
    # reports on them, and they count as missing against all the others -- 70
    # such rows out of 1646 caps every column at 0.957 coverage and the
    # --min-coverage filter then removes the entire panel.
    per_date = px.notna().sum(axis=1)
    cal = per_date[per_date >= a.calendar_quorum * px.shape[1]].index
    print(f"  dates {len(px)} -> {len(cal)} with >= {a.calendar_quorum:.0%} "
          f"of VALOR quoted")
    px = px.loc[cal]

    cov = px.notna().mean()
    px = px.drop(columns=cov.index[cov < a.min_coverage])
    print(f"  drop {int((cov < a.min_coverage).sum()):4d}  "
          f"coverage < {a.min_coverage}")
    if px.shape[1] == 0:
        raise SystemExit("--min-coverage dropped every column; see the "
                         "calendar line above")


p.add_argument("--calendar-quorum", type=float, default=0.5)

```
