# Segment Notebooks (Legacy Pipeline) (旧线)

This folder keeps the legacy segment-level notebooks for method comparison and supplemental explanation. (用于方法对比与补充解释)

## Notebooks (拆分后)
- `segment_02_data_understanding.ipynb`
- `segment_03_preparation_features.ipynb`
- `segment_04_modeling.ipynb`
- `segment_06_evaluation.ipynb`

## Data & Outputs (数据与输出)
Notebooks use `get_paths("segment")` for data and outputs:
- DB defaults to `data/raw/airlines_db.sqlite`, or `AIRLINE_DB_PATH`. (数据库路径)
- Outputs go to `outputs/segment/figures`, `outputs/segment/tables`, `outputs/segment/artifacts`. (输出目录)
