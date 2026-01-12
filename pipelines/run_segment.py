"""Run the segment-level pipeline end-to-end (no notebooks). (运行 segment 管线全流程，不依赖 notebook)"""
from __future__ import annotations

import os
import pandas as pd

from airline_revenue_analytics.config import get_paths
from airline_revenue_analytics.io import load_core_tables, replace_literal_N
from airline_revenue_analytics.features.segment import (
    parse_json_en,
    add_route_code,
    add_segment_features,
)
from airline_revenue_analytics.features.booking import add_booking_lead_time
from airline_revenue_analytics.modeling.train import (
    split_xy,
    split_xy_grouped,
    make_preprocess,
    linear_pipeline,
    tree_pipeline,
)
from airline_revenue_analytics.modeling.eval import evaluate


REQUIRED = {"bookings", "tickets", "ticket_flights", "flights", "aircrafts_data"}


def _ensure_tables(tables: dict):
    """Validate that required tables are present. (校验必需表是否存在)"""
    missing = sorted(REQUIRED - set(tables))
    if missing:
        raise ValueError(f"Missing required tables: {missing}")


def _is_true(val: str) -> bool:
    """Parse common truthy strings. (解析常见的真值字符串)"""
    return str(val).strip().lower() in {"1", "true", "yes", "y"}


def _cap_top_k(s: pd.Series, k: int, other_label: str = "Other") -> pd.Series:
    """Cap a categorical series to top-k values. (将类别特征裁剪为 Top-K，其余归为 Other)"""
    if k <= 0:
        return s
    vc = s.value_counts(dropna=True)
    keep = set(vc.head(k).index)
    return s.where(s.isin(keep), other_label)


def main():
    """Execute the full segment pipeline and save outputs. (执行完整 segment 管线并保存输出)"""
    light_mode = _is_true(os.getenv("SEGMENT_LIGHT_MODE", "0"))
    sample_rows = int(os.getenv("SEGMENT_SAMPLE_ROWS", "200000"))
    route_topk = int(os.getenv("SEGMENT_ROUTE_TOPK", "40"))
    aircraft_topk = int(os.getenv("SEGMENT_AIRCRAFT_TOPK", "20"))

    paths = get_paths("segment")
    table_cols = {
        "bookings": ["book_ref", "book_date"],
        "tickets": ["ticket_no", "book_ref"],
        "ticket_flights": ["ticket_no", "flight_id", "fare_conditions", "amount"],
        "flights": [
            "flight_id",
            "scheduled_departure",
            "scheduled_arrival",
            "departure_airport",
            "arrival_airport",
            "aircraft_code",
        ],
        "aircrafts_data": ["aircraft_code", "model"],
    }
    # Limit the largest table early in light mode / 轻量模式下提前限制最大表
    row_limit = {"ticket_flights": sample_rows} if light_mode else None
    tables = load_core_tables(
        paths.db_path,
        table_names=REQUIRED,
        columns=table_cols,
        row_limit=row_limit,
    )
    _ensure_tables(tables)

    bookings = replace_literal_N(tables["bookings"])
    tickets = replace_literal_N(tables["tickets"])
    ticket_flights = replace_literal_N(tables["ticket_flights"])
    flights = replace_literal_N(tables["flights"])
    aircrafts = replace_literal_N(tables["aircrafts_data"])

    if light_mode and len(ticket_flights) > sample_rows:
        ticket_flights = ticket_flights.sample(n=sample_rows, random_state=42)

    if "model" in aircrafts.columns:
        aircrafts = aircrafts.copy()
        aircrafts["Aircraft_Model_EN"] = parse_json_en(aircrafts["model"])

    tf = ticket_flights.merge(tickets[["ticket_no", "book_ref"]], on="ticket_no", how="left")
    tf = tf.merge(
        flights[["flight_id", "scheduled_departure", "scheduled_arrival", "departure_airport", "arrival_airport", "aircraft_code"]],
        on="flight_id", how="left",
    )
    tf = tf.merge(aircrafts[["aircraft_code", "Aircraft_Model_EN"]], on="aircraft_code", how="left")

    tf = add_route_code(tf, "departure_airport", "arrival_airport")
    tf = add_segment_features(tf)
    tf = add_booking_lead_time(tf, bookings)
    tf["cabin_index"] = tf["fare_class_ord"]

    # Minimal dataset (target = amount) / 最小建模数据集（目标为 amount）
    seg_df = tf.dropna(subset=["amount"]).copy()
    if light_mode and len(seg_df) > sample_rows:
        seg_df = seg_df.sample(n=sample_rows, random_state=42)

    if light_mode:
        # Reduce high-cardinality categories / 降低高基数类别
        if "route_code" in seg_df.columns:
            seg_df["route_code"] = _cap_top_k(seg_df["route_code"].astype("string"), route_topk)
        if "Aircraft_Model_EN" in seg_df.columns:
            seg_df["Aircraft_Model_EN"] = _cap_top_k(seg_df["Aircraft_Model_EN"].astype("string"), aircraft_topk)

    # Save a small preview for inspection / 保存少量预览用于检查
    seg_df.head(50).to_csv(paths.tables / "segment_model_df_preview.csv", index=False)

    num_cols = [
        "sched_flight_duration_minutes", "booking_lead_time_days", "cabin_index",
        "is_premium_cabin", "departure_hour", "departure_dow",
    ]
    num_cols = [c for c in num_cols if c in seg_df.columns]
    cat_cols = [c for c in ["fare_conditions", "route_code", "Aircraft_Model_EN"] if c in seg_df.columns]

    if "book_ref" in seg_df.columns:
        # Group-wise split to avoid leakage / 按订单分组切分避免泄漏
        X_train, X_test, y_train, y_test = split_xy_grouped(seg_df, "amount", "book_ref")
    else:
        X_train, X_test, y_train, y_test = split_xy(seg_df, "amount")
    pre = make_preprocess(num_cols, cat_cols)

    models = {
        "LinearRegression": linear_pipeline(pre),
        "DecisionTree": tree_pipeline(pre),
    }

    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)
        rows.append({"model": name, **metrics})

    pd.DataFrame(rows).to_csv(
        paths.tables / "model_metrics_segment.csv", index=False, float_format="%.3f"
    )
    print("[OK] segment pipeline finished")
    print("Outputs:", paths.outputs_root)


if __name__ == "__main__":
    main()
