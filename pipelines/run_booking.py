"""Run the booking-level pipeline end-to-end (no notebooks). (运行 booking 主线全流程，不依赖 notebook)"""

from __future__ import annotations

import argparse
import importlib.util
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

from airline_revenue_analytics.config import get_paths
from airline_revenue_analytics.io import load_core_tables, replace_literal_N
from airline_revenue_analytics.features.segment import (
    add_route_code,
    add_segment_features,
)
from airline_revenue_analytics.features.booking import (
    add_booking_lead_time,
    aggregate_to_booking,
)
from airline_revenue_analytics.modeling.train import (
    make_preprocess,
    linear_pipeline,
    tree_pipeline,
)
from airline_revenue_analytics.modeling.eval import evaluate
from airline_revenue_analytics.viz.charts import apply_style, PLOT_COLORS


REQUIRED = {"bookings", "tickets", "ticket_flights", "flights"}


def _ensure_tables(tables: dict):
    """Validate that required tables are present. (校验必需表是否存在)"""
    missing = sorted(REQUIRED - set(tables))
    if missing:
        raise ValueError(f"Missing required tables: {missing}")


def _has_pyarrow() -> bool:
    return importlib.util.find_spec("pyarrow") is not None


def main(write_parquet: bool = True):
    """Execute the full booking pipeline and save outputs. (执行完整 booking 管线并保存输出)"""
    random.seed(42)
    np.random.seed(42)
    apply_style()

    paths = get_paths("booking")
    table_cols = {
        "bookings": ["book_ref", "book_date", "total_amount"],
        "tickets": ["ticket_no", "book_ref"],
        "ticket_flights": ["ticket_no", "flight_id", "fare_conditions"],
        "flights": [
            "flight_id",
            "scheduled_departure",
            "scheduled_arrival",
            "departure_airport",
            "arrival_airport",
            "aircraft_code",
        ],
    }
    tables = load_core_tables(
        paths.db_path,
        table_names=REQUIRED,
        columns=table_cols,
    )
    _ensure_tables(tables)

    bookings = replace_literal_N(tables["bookings"])
    tickets = replace_literal_N(tables["tickets"])
    ticket_flights = replace_literal_N(tables["ticket_flights"])
    flights = replace_literal_N(tables["flights"])

    tf = ticket_flights.merge(
        tickets[["ticket_no", "book_ref"]], on="ticket_no", how="left"
    )
    tf = tf.merge(
        flights[
            [
                "flight_id",
                "scheduled_departure",
                "scheduled_arrival",
                "departure_airport",
                "arrival_airport",
                "aircraft_code",
            ]
        ],
        on="flight_id",
        how="left",
    )

    tf = add_route_code(tf, "departure_airport", "arrival_airport")
    tf = add_segment_features(tf)
    tf = add_booking_lead_time(tf, bookings)

    booking_df = aggregate_to_booking(tf, bookings)
    if "book_ref" in booking_df.columns:
        booking_df = booking_df.sort_values("book_ref").reset_index(drop=True)
        assert booking_df[
            "book_ref"
        ].is_unique, "book_ref must be unique after aggregation"

    booking_df = booking_df.replace([np.inf, -np.inf], np.nan)
    booking_df = booking_df.dropna(subset=["log_total_amount"]).copy()

    train_mask = None
    if "book_ref" in booking_df.columns:
        book_refs = sorted(booking_df["book_ref"].dropna().unique().tolist())
        train_refs, test_refs = train_test_split(
            book_refs, test_size=0.3, random_state=42
        )
        train_mask = booking_df["book_ref"].isin(train_refs)

    # Create a capped route category from train data only / 仅使用训练集生成 Top-K 路线类别
    top_routes = []
    if "primary_route_code" in booking_df.columns:
        route_source = (
            booking_df.loc[train_mask, "primary_route_code"]
            if train_mask is not None
            else booking_df["primary_route_code"]
        )
        vc = route_source.value_counts(dropna=True).reset_index()
        vc.columns = ["route", "count"]
        vc = vc.sort_values(["count", "route"], ascending=[False, True])
        top_routes = vc.head(15)["route"].tolist()
        booking_df["primary_route_code_top"] = booking_df["primary_route_code"].where(
            booking_df["primary_route_code"].isin(top_routes), "Other"
        )

    # Save dataset preview / 保存数据预览
    booking_df.head(50).to_csv(
        paths.tables / "booking_model_df_preview.csv", index=False
    )
    if write_parquet and _has_pyarrow():
        booking_df.to_parquet(paths.tables / "booking_model_df.parquet", index=False)
    elif write_parquet:
        print("[WARN] pyarrow not installed; skipping parquet output.")

    # Modeling / 模型训练与评估
    num_cols = [
        "n_segments",
        "n_tickets",
        "n_flights",
        "itinerary_duration_sum",
        "sum_sched_duration_min",
        "avg_sched_duration_min",
        "max_sched_duration_min",
        "share_premium_cabin",
        "max_cabin_index",
        "has_longhaul",
        "n_unique_routes",
        "avg_booking_lead_days",
    ]
    num_cols = [c for c in num_cols if c in booking_df.columns]
    cat_cols = (
        ["primary_route_code_top"]
        if "primary_route_code_top" in booking_df.columns
        else []
    )

    X_all = booking_df[num_cols + cat_cols].copy()
    y_all = booking_df["log_total_amount"].copy()
    if train_mask is not None:
        X_train, X_test = X_all[train_mask], X_all[~train_mask]
        y_train, y_test = y_all[train_mask], y_all[~train_mask]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=0.3, random_state=42
        )
    pre = make_preprocess(num_cols, cat_cols)

    models = {
        "LinearRegression": linear_pipeline(pre),
        "DecisionTree": tree_pipeline(pre),
    }

    rows = []
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)
        rows.append({"model": name, **metrics})
        fitted[name] = model

    metrics_df = pd.DataFrame(rows).sort_values(
        ["R2", "model"], ascending=[False, True]
    )
    metrics_df["target"] = "log_total_amount"
    metrics_df["scale"] = "log"
    metrics_df.to_csv(paths.tables / "metrics.csv", index=False, float_format="%.3f")

    baseline_name = "LinearRegression"
    best_name = metrics_df.iloc[0]["model"]
    best_model = fitted[best_name]

    # Robustness check: route hold-out split (unseen routes) / 稳健性检验：路线留出（未见航线）
    robust_rows = []
    if "primary_route_code" in booking_df.columns:
        route_counts = booking_df["primary_route_code"].value_counts(dropna=True)
        routes = route_counts.index.tolist()
        rng = np.random.RandomState(42)
        rng.shuffle(routes)
        test_routes = []
        test_rows = 0
        total_rows = len(booking_df)
        for r in routes:
            test_routes.append(r)
            test_rows += int(route_counts[r])
            if test_rows / total_rows >= 0.30:
                break
        route_mask = booking_df["primary_route_code"].isin(test_routes)
        # Rebuild top routes on train side only to keep hold-out strictly unseen
        top_routes_r = []
        cat_cols_r = []
        route_source_r = booking_df.loc[~route_mask, "primary_route_code"]
        vc_r = route_source_r.value_counts(dropna=True).reset_index()
        vc_r.columns = ["route", "count"]
        vc_r = vc_r.sort_values(["count", "route"], ascending=[False, True])
        top_routes_r = vc_r.head(15)["route"].tolist()
        booking_df["primary_route_code_top_r"] = booking_df["primary_route_code"].where(
            booking_df["primary_route_code"].isin(top_routes_r), "Other"
        )
        cat_cols_r = ["primary_route_code_top_r"]

        X_all_r = booking_df[num_cols + cat_cols_r].copy()
        X_train_r, X_test_r = X_all_r[~route_mask], X_all_r[route_mask]
        y_train_r, y_test_r = y_all[~route_mask], y_all[route_mask]

        # Rebuild the best model on the route-holdout split / 在路线留出切分上重训最佳模型
        pre_r = make_preprocess(num_cols, cat_cols_r)
        if best_name == "LinearRegression":
            model_r = linear_pipeline(pre_r)
        elif best_name == "DecisionTree":
            model_r = tree_pipeline(pre_r)
        else:
            model_r = linear_pipeline(pre_r)
        model_r.fit(X_train_r, y_train_r)
        metrics_r = evaluate(model_r, X_test_r, y_test_r)
        robust_rows.append(
            {
                "split": "route_holdout",
                "model": best_name,
                "R2": metrics_r["R2"],
                "RMSE": metrics_r["RMSE"],
                "MAE": metrics_r["MAE"],
                "target": "log_total_amount",
                "scale": "log",
                "test_routes": len(test_routes),
                "test_rows": int(route_mask.sum()),
            }
        )

    if robust_rows:
        pd.DataFrame(robust_rows).to_csv(
            paths.tables / "metrics_robust.csv", index=False, float_format="%.3f"
        )

    # R2 compare plot (baseline vs best) / R2 对比图（基线 vs 最优）
    baseline_r2 = float(
        metrics_df.loc[metrics_df["model"] == baseline_name, "R2"].iloc[0]
    )
    best_r2 = float(metrics_df.loc[metrics_df["model"] == best_name, "R2"].iloc[0])
    plt.figure()
    plt.bar(
        [baseline_name, f"Best ({best_name})"],
        [baseline_r2, best_r2],
        color=[PLOT_COLORS.muted, PLOT_COLORS.primary],
    )
    plt.ylabel("R2")
    plt.title("R2 Comparison (Baseline vs Best)")
    plt.tight_layout()
    plt.savefig(paths.figures / "r2_compare.png", dpi=150)
    plt.close()

    # Actual vs predicted (best model) / 真实值 vs 预测值（最佳模型）
    y_pred = best_model.predict(X_test)
    plt.figure()
    plt.scatter(
        y_test, y_pred, s=12, alpha=0.65, color=PLOT_COLORS.primary, edgecolors="none"
    )
    min_v = min(float(y_test.min()), float(np.min(y_pred)))
    max_v = max(float(y_test.max()), float(np.max(y_pred)))
    plt.plot([min_v, max_v], [min_v, max_v], color=PLOT_COLORS.accent, linewidth=1)
    plt.xlabel("Actual (log_total_amount)")
    plt.ylabel("Predicted")
    plt.title(f"Actual vs Predicted ({best_name})")
    plt.tight_layout()
    plt.savefig(paths.figures / "actual_vs_pred.png", dpi=150)
    plt.close()

    # Feature importance (permutation on original columns) / 特征重要性（基于原始列的置换重要性）
    X_perm = X_test
    y_perm = y_test
    if len(X_perm) > 5000:
        X_perm = X_perm.sample(5000, random_state=42)
        y_perm = y_perm.loc[X_perm.index]
    perm = permutation_importance(
        best_model, X_perm, y_perm, n_repeats=5, random_state=42, scoring="r2"
    )
    importances = pd.Series(perm.importances_mean, index=X_perm.columns).sort_values(
        ascending=False
    )
    top = importances.head(15)
    plt.figure()
    top.sort_values().plot(kind="barh", color=PLOT_COLORS.primary)
    plt.xlabel("Permutation Importance (R2 decrease)")
    plt.title(f"Feature Importance ({best_name})")
    plt.tight_layout()
    plt.savefig(paths.figures / "feature_importance.png", dpi=150)
    plt.close()

    print("[OK] booking pipeline finished")
    print("Outputs:", paths.outputs_root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run booking pipeline")
    parser.add_argument("--no-parquet", action="store_true", help="Skip parquet output")
    args = parser.parse_args()
    main(write_parquet=not args.no_parquet)
