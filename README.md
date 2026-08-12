```bash

cd "$(git rev-parse --show-toplevel)"
git rm -r --cached '*.csv'

git ls-files '*.csv'        # what will be affected — should list all three folders
git rm -r --cached '*.csv'
git status                  # every csv shows as "deleted"; check nothing else does
git commit -m "Stop tracking CSV files"
git push

```
