# Airline Revenue Analytics – 双线并行入口

**[English](README.md) | 中文**

本仓库提供**主交付**与**可选附录**：
- **Primary: Booking-level pipeline（推荐）**：面向履约/业务汇总视角，是本项目的主交付与推荐复现路径。
- **Segment pipeline（附录）**：单航段探索，仅作为特征工程中间步骤或对比附录。

> 数据库不入库，请按下方 Data Setup 配置。

## Core Results（Booking 主线）
| Model | Role | R2 (log) | RMSE (log) | MAE (log) |
| --- | --- | --- | --- | --- |
| LinearRegression | Baseline | 0.921 | 0.242 | 0.182 |
| RandomForest | Best | 0.988 | 0.095 | 0.043 |
| DecisionTree | Comparator | 0.977 | 0.130 | 0.081 |

结论：RandomForest 在 log 尺度上明显优于线性基线（R2 +0.067、RMSE -0.147、MAE -0.139），主要来自对**行程时长与舱位/路线复杂度的非线性关系**的捕捉。

## Robustness Check（路线留出）
| Split | R2 (log) | RMSE (log) | MAE (log) | Notes |
| --- | --- | --- | --- | --- |
| Random split (by `book_ref`) | 0.988 | 0.095 | 0.043 | RandomForest (best) |
| Route hold-out (unseen `primary_route_code`) | 0.966 | 0.149 | 0.077 | 91 routes held out (~30% rows) |

解读：路线留出仅小幅下降，说明模型不是简单记忆路线，对未见航线仍有较强泛化能力。

## Business Insights（业务洞察）
- **航程时长是定价主驱动**：`sum_sched_duration_min` 在特征重要性中显著领先，可用于按航程分层的收益管理框架。
- **舱位结构与行程复杂度是可操作杠杆**：`share_premium_cabin` / `max_cabin_index` 与 `n_tickets` / `n_flights` / `n_unique_routes` 共同刻画价格结构，可用于 upsell 与连接行程的收益优化。

## Validity / Leakage Controls（有效性与泄漏控制）
- **切分口径**：按 `book_ref` 分组切分，避免同一订单跨训练/测试。
- **特征来源**：仅使用 schedule/cabin/route 等可观测字段，不使用价格字段作为输入。
- **指标尺度**：所有指标在 `log_total_amount` 尺度上报告；真实金额需做 `exp` 回转换理解。
- **高 R2 解释**：时长主导是业务规律，但通过特征口径约束避免直接泄漏。

## Quickstart（最短复现）
1. `pip install -e .`
2. 下载数据（数据不入库）：`pip install kaggle` 然后 `python scripts/fetch_data.py`（或手动放到 `data/raw/airlines_db.sqlite`）
3. `python pipelines/run_booking.py`

可选：`pip install -e .[parquet]` 以启用 parquet 输出，或用 `--no-parquet` 跳过。

## Repo 结构
```
repo_root/
  README.md
  README.zh.md
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

## 主交付 vs 附录
| 维度 | Booking pipeline（主交付） | Segment pipeline（附录） |
| --- | --- | --- |
| 粒度 | Booking-level（聚合） | Segment-level（单航段） |
| 目标 | `total_amount` / `log_total_amount` | `amount`（单航段票价） |
| 主要特征 | 旅程汇总指标（次数/时长/路线） | 航段特征（时间/舱位/机型/路线） |
| 模型 | Linear + Tree + RandomForest 基线 | Linear + Tree 基线 |
| 输出 | `outputs/booking/` | `outputs/segment/` |
| 适用场景 | 业务汇总与可解释展示 | 特征工程附录 |

> 两条线口径不同，结果不做数值对比。

## Reproducibility（口径与可复现）
- 目标变量：`log_total_amount`（对 `total_amount` 取对数）
- 指标尺度：log 尺度
- 切分策略：按 `book_ref` 分组，固定随机种子
- 随机性控制：`numpy/random=42`，`DecisionTree/RandomForest random_state=42`

## Notebooks（叙事型）
- `notebooks/portfolio_report.ipynb`：只读主展示报告（从 `outputs/booking/*` 读取；先跑 `python pipelines/run_booking.py`）
- `notebooks/booking/`：主线叙事型 notebooks（推荐）
- `notebooks/segment/`：旧线单体 notebooks（附录）
- Notebook 依赖：`pip install -e .[notebook]` 或 `pip install -r requirements-notebook.txt`
- 仅在不可 editable 环境时使用：`pip install -r requirements.txt`

## Data Source & License（数据来源与许可）
- 数据集：Airlines_DB（Kaggle `datalearn/airlines-db`）
- URL：https://www.kaggle.com/datasets/datalearn/airlines-db
- 许可/条款：遵循 Kaggle 数据集页面与 API 条款；本仓库不再分发数据
- 处理流程：SQLite 读取、基础清洗（`\\N` -> NaN）、特征工程与 booking 级聚合

## Data Setup（数据准备）
- 数据不随仓库提供；来源为 Kaggle `datalearn/airlines-db`
- 推荐脚本下载：`pip install kaggle` + `python scripts/fetch_data.py`（token 放置 `~/.kaggle/kaggle.json`，并 `chmod 600`）
- 手动指引：`python scripts/fetch_data.py --manual`
- 手动放置也可：`data/raw/airlines_db.sqlite` 或设置 `AIRLINE_DB_PATH=/path/to/airlines_db.sqlite`

## Key Figures（主线节选）
![Actual vs predicted](outputs/booking/figures/actual_vs_pred.png)
![R2 compare](outputs/booking/figures/r2_compare.png)
![Feature importance](outputs/booking/figures/feature_importance.png)

## Core Metrics（指标表）
- 完整指标表：`outputs/booking/tables/metrics.csv`

## Clean（清理）
- 打包前清理缓存：`scripts/clean_repo.sh`

## License
- MIT（见 `LICENSE`）
