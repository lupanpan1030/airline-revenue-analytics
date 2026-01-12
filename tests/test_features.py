import pandas as pd
import numpy as np
from airline_revenue_analytics.features.segment import (
    minutes_between,
    days_between,
    add_segment_features,
    add_route_code,
)
from airline_revenue_analytics.features.booking import aggregate_to_booking
from airline_revenue_analytics.modeling.train import split_xy_grouped

def test_minutes_between_non_negative():
    s = pd.Series(pd.to_datetime(["2024-01-01T00:00Z", "2024-01-01T01:00Z"]))
    e = pd.Series(pd.to_datetime(["2024-01-01T00:30Z", "2023-12-31T23:00Z"]))
    out = minutes_between(s, e)
    assert out.iloc[0] == 30.0
    assert np.isnan(out.iloc[1])  # negative → NaN

def test_add_segment_features_columns():
    df = pd.DataFrame({
        "scheduled_departure": ["2024-01-01T00:00Z"],
        "scheduled_arrival":   ["2024-01-01T02:00Z"],
        "fare_conditions":     ["Economy"],
    })
    out = add_segment_features(df)
    for col in ["sched_flight_duration_minutes","departure_dow","departure_hour","fare_class","fare_class_ord","is_premium_cabin"]:
        assert col in out.columns


def test_split_xy_grouped_no_overlap():
    df = pd.DataFrame({
        "book_ref": ["A", "A", "B", "B", "C", "C"],
        "amount": [10, 11, 20, 21, 30, 31],
        "feature": [1, 2, 3, 4, 5, 6],
    })
    X_train, X_test, _, _ = split_xy_grouped(df, "amount", "book_ref", test_size=0.33, seed=42)
    train_groups = set(df.loc[X_train.index, "book_ref"])
    test_groups = set(df.loc[X_test.index, "book_ref"])
    assert train_groups.isdisjoint(test_groups)


def test_aggregate_to_booking_counts_and_itinerary_duration():
    seg_master = pd.DataFrame({
        "book_ref": ["A", "A", "A", "B"],
        "ticket_no": ["t1", "t2", "t2", "t3"],
        "flight_id": [1, 1, 2, 3],
        "sched_flight_duration_minutes": [60, 60, 120, 90],
        "is_premium_cabin": [0, 0, 1, 0],
        "fare_class_ord": [0, 0, 2, 0],
        "route_code": ["X-Y", "X-Y", "Y-Z", "Z-W"],
    })
    bookings = pd.DataFrame({
        "book_ref": ["A", "B"],
        "total_amount": [100.0, 200.0],
    })
    out = aggregate_to_booking(seg_master, bookings)
    row_a = out.loc[out["book_ref"] == "A"].iloc[0]
    assert row_a["n_segments"] == 3
    assert row_a["n_tickets"] == 2
    assert row_a["n_flights"] == 2
    assert row_a["itinerary_duration_sum"] == 180


def test_add_route_code_missing_values():
    df = pd.DataFrame({"dep": ["AAA", None], "arr": ["BBB", "CCC"]})
    out = add_route_code(df, "dep", "arr")
    assert out.loc[0, "route_code"] == "AAA-BBB"
    assert pd.isna(out.loc[1, "route_code"])


def test_aggregate_to_booking_log_total_amount_non_positive():
    seg_master = pd.DataFrame({
        "book_ref": ["A"],
        "ticket_no": ["t1"],
        "flight_id": [1],
        "sched_flight_duration_minutes": [60],
        "is_premium_cabin": [0],
        "fare_class_ord": [0],
        "route_code": ["X-Y"],
    })
    bookings = pd.DataFrame({
        "book_ref": ["A"],
        "total_amount": [0.0],
    })
    out = aggregate_to_booking(seg_master, bookings)
    assert pd.isna(out.loc[0, "log_total_amount"])
