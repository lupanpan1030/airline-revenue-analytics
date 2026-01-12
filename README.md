# Airline Revenue Analytics – Dual-Track Entry (双线并行入口)

This repo provides a **primary deliverable** and an **optional appendix**. (本仓库提供主交付与可选附录)
- **Primary: Booking-level pipeline (recommended)** — Main deliverable for operational/business summary. (主交付：面向履约/业务汇总，推荐复现路径)
- **Segment pipeline: feature building / optional appendix** — Segment-level exploration only. (附录：单航段探索，用于特征工程对比)

> The database is not committed; see Data Setup. (数据库不入库，请按下方 Data Setup 配置)

## Core Results (Booking Pipeline) (核心结果)
| Model | Role | R2 (log) | RMSE (log) | MAE (log) |
| --- | --- | --- | --- | --- |
| LinearRegression | Baseline | 0.921 | 0.242 | 0.182 |
| DecisionTree | Best | 0.977 | 0.130 | 0.081 |

Conclusion: DecisionTree clearly outperforms the linear baseline on log scale (R2 +0.056, RMSE -0.113, MAE -0.101), driven by **nonlinear effects of itinerary duration and cabin/route complexity**. (结论：DecisionTree 在 log 尺度上明显优于线性基线，主要来自对行程时长与舱位/路线复杂度的非线性关系捕捉)

## Robustness Check (Route Hold-out) (稳健性检验)
| Split | R2 (log) | RMSE (log) | MAE (log) | Notes |
| --- | --- | --- | --- | --- |
| Random split (by `book_ref`) | 0.977 | 0.130 | 0.081 | DecisionTree (best) |
| Route hold-out (unseen `primary_route_code`) | 0.966 | 0.146 | 0.094 | 91 routes held out (~30% rows) |

Interpretation: Only a small drop on route hold-out suggests the model is not just memorizing routes and still generalizes to unseen routes. (解读：路线留出下降幅度小，说明模型对未见航线仍具较强泛化)

## Business Insights (业务洞察)
- **Itinerary duration is the main price driver**: `sum_sched_duration_min` dominates importance, consistent with duration-based revenue management. (航程时长是定价主驱动)
- **Cabin mix and trip complexity are actionable levers**: `share_premium_cabin` / `max_cabin_index` plus `n_tickets` / `n_flights` / `n_unique_routes` capture price structure for upsell and connection strategies. (舱位结构与行程复杂度是可操作杠杆)

## Validity / Leakage Controls (有效性与泄漏控制)
- **Split**: group-wise by `book_ref` to avoid leakage across train/test. (按 `book_ref` 分组切分)
- **Features**: only schedule/cabin/route observables; no price fields as inputs. (仅使用可观测字段，不用价格字段)
- **Metric scale**: reported on `log_total_amount`; use `exp` to interpret real amounts. (log 尺度指标)
- **High R2 note**: duration dominance is expected but monitored; leakage is prevented by feature design. (高 R2 解释)

## Quickstart (快速开始)
1. `pip install -e .` (安装依赖)
2. Download data (not in repo): `pip install kaggle` then `python scripts/fetch_data.py` (下载数据；或手动放到 `data/raw/airlines_db.sqlite`)
3. `python pipelines/run_booking.py` (run the primary pipeline / 运行主线)

Optional: `pip install -e .[parquet]` to enable parquet output, or pass `--no-parquet` to skip. (可选启用 parquet)

## Repo Structure (结构总览)
```
repo_root/
  README.md
  .gitignore
  pyproject.toml

  src/
    airline_revenue_analytics/
      __init__.py
      config.py
      io.py
      features/
        segment.py
        booking.py
      modeling/
        train.py
        eval.py
      viz/
        charts.py

  notebooks/
    booking/
    segment/

  pipelines/
    run_booking.py
    run_segment.py

  data/raw/                  # gitignored
  outputs/
    booking/
    segment/
```

## Primary vs Appendix (摘要)
| Dimension | Booking pipeline (Primary) | Segment pipeline (Appendix) |
| --- | --- | --- |
| Granularity | Booking-level (aggregated) | Segment-level (single leg) |
| Target | `total_amount` / `log_total_amount` | `amount` (segment fare) |
| Features | Itinerary aggregates (counts/durations/routes) | Segment features (time/cabin/aircraft/route) |
| Models | Linear + Tree baselines | Linear + Tree baselines |
| Outputs | `outputs/booking/` | `outputs/segment/` |
| Use case | Business summary & interpretation | Feature engineering appendix |

> The two tracks use different definitions and are not directly comparable in absolute metrics. (两条线口径不同，不直接做数值对比)

## Reproducibility (口径与可复现)
- Target: `log_total_amount` (log of `total_amount`). (目标变量)
- Metrics: reported on log scale. (指标尺度)
- Split: fixed random seed, grouped by `book_ref`. (切分策略)
- Randomness: `numpy/random=42`, `DecisionTree random_state=42`. (随机性控制)

## Notebooks (叙事型)
- `notebooks/booking/`: main narrative notebooks (recommended). (主线叙事型)
- `notebooks/segment/`: legacy segment notebooks (appendix). (旧线附录)
- Notebook deps: `pip install -e .[notebook]` or `pip install -r requirements-notebook.txt`. (notebook 依赖)
- Use `pip install -r requirements.txt` only when editable install is not available. (仅在不可 editable 时使用)

## Data Source & License (数据来源与许可)
- Dataset: Airlines_DB (Kaggle `datalearn/airlines-db`). (数据集)
- URL: https://www.kaggle.com/datasets/datalearn/airlines-db
- License/terms: follow the Kaggle dataset page and Kaggle API terms; data is not redistributed here. (遵循 Kaggle 条款)
- Processing: SQLite ingestion, basic cleaning (`\\N` -> NaN), feature engineering, and booking-level aggregation. (处理流程)

## Data Setup (数据准备)
- Data is not included; source is Kaggle `datalearn/airlines-db`. (数据不随仓库提供)
- Recommended: `pip install kaggle` + `python scripts/fetch_data.py` (token at `~/.kaggle/kaggle.json`, `chmod 600`). (推荐脚本下载)
- Manual guide: `python scripts/fetch_data.py --manual`. (手动指引)
- Manual placement: `data/raw/airlines_db.sqlite`, or set `AIRLINE_DB_PATH=/path/to/airlines_db.sqlite`. (手动放置)

## Key Figures (主线节选)
![Actual vs predicted](outputs/booking/figures/actual_vs_pred.png)
![R2 compare](outputs/booking/figures/r2_compare.png)
![Feature importance](outputs/booking/figures/feature_importance.png)

## Core Metrics (指标表)
- Full metrics: `outputs/booking/tables/metrics.csv`. (完整指标表)

## Clean (清理)
- Clean caches before packaging: `scripts/clean_repo.sh`. (打包前清理缓存)

## License
- MIT (see `LICENSE`)
