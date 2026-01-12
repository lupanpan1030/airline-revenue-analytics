# data/raw (not in git)

This project does not ship the SQLite database.
Source: Kaggle dataset datalearn/airlines-db (https://www.kaggle.com/datasets/datalearn/airlines-db).
Use is subject to the Kaggle dataset terms; data is not redistributed here.

Recommended (Kaggle API, dataset: datalearn/airlines-db):
1) pip install kaggle
2) Place your token at ~/.kaggle/kaggle.json (chmod 600)
3) python scripts/fetch_data.py
   - Optional: add --force to re-download, or --output to change the target dir
   - Manual steps: python scripts/fetch_data.py --manual

Expected output:
- data/raw/airlines_db.sqlite

Manual option:
- Place the DB at data/raw/airlines_db.sqlite, or set AIRLINE_DB_PATH=/path/to/airlines_db.sqlite
