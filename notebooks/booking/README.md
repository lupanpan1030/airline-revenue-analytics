# Booking Notebooks (Mainline) (主线)

Narrative notebooks for the primary pipeline, run in order 00–06. (主线叙事型 notebooks，按 00–06 顺序运行)

## Notebooks
- `00_setup_and_data_intake.ipynb`
- `01_data_understanding.ipynb`
- `02_data_preparation.ipynb`
- `03_transformation_and_split.ipynb`
- `04_modeling_loopA.ipynb`
- `05_modeling_loopB.ipynb`
- `06_interpretation_compare.ipynb`

## Paths (Standardized) (路径)
- Database: `data/raw/airlines_db.sqlite` or `AIRLINE_DB_PATH`. (数据库路径)
- Outputs: `outputs/booking/` (figures/tables/artifacts). (输出目录)

## Notes
- The first cell auto-detects repo root and output dirs. (首个 cell 会自动定位 repo 根目录与输出目录)
