"""Booking-level aggregation from segment data. (基于航段数据的订单级聚合)"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .segment import to_utc, days_between


def add_booking_lead_time(seg_master: pd.DataFrame, bookings: pd.DataFrame) -> pd.DataFrame:
    """Attach booking lead time (days) to each segment row. (为每条航段追加订票提前期)"""
    d = seg_master.copy()
    if "book_date" not in bookings.columns:
        return d

    d = d.merge(bookings[["book_ref", "book_date"]], on="book_ref", how="left")
    d["book_date"] = to_utc(d["book_date"])
    if "scheduled_departure" in d.columns:
        d["booking_lead_time_days"] = days_between(d["book_date"], d["scheduled_departure"])
    return d


def _primary_route(s: pd.Series) -> str | None:
    """Deterministic primary route (mode with stable tie-break). (确定性的主路线，众数平局时稳定排序)"""
    m = s.dropna().mode()
    if m.empty:
        return None
    return sorted(m.astype(str).tolist())[0]


def aggregate_to_booking(seg_master: pd.DataFrame, bookings: pd.DataFrame) -> pd.DataFrame:
    """Aggregate segment rows to booking-level table; attach target. (将航段聚合为订单级特征，并附加目标变量)"""
    agg_spec = dict(
        n_segments=("ticket_no", "count"),
        n_tickets=("ticket_no", "nunique"),
        n_flights=("flight_id", "nunique"),
        sum_sched_duration_min=("sched_flight_duration_minutes", "sum"),
        avg_sched_duration_min=("sched_flight_duration_minutes", "mean"),
        max_sched_duration_min=("sched_flight_duration_minutes", "max"),
        share_premium_cabin=("is_premium_cabin", "mean"),
        max_cabin_index=("fare_class_ord", "max"),
        has_longhaul=("sched_flight_duration_minutes", lambda s: int((s >= 240).any())),
        n_unique_routes=("route_code", "nunique"),
        primary_route_code=("route_code", _primary_route),
    )

    if "booking_lead_time_days" in seg_master.columns:
        agg_spec["avg_booking_lead_days"] = ("booking_lead_time_days", "mean")

    g = seg_master.groupby("book_ref").agg(**agg_spec).reset_index()

    if {"book_ref", "flight_id", "sched_flight_duration_minutes"}.issubset(seg_master.columns):
        # Sum unique flights to avoid double-counting multi-ticket rows / 按唯一航班求和，避免多票重复计数
        unique_flights = seg_master.dropna(subset=["book_ref", "flight_id"]).drop_duplicates(["book_ref", "flight_id"])
        itinerary_sum = (
            unique_flights.groupby("book_ref")["sched_flight_duration_minutes"]
            .sum()
            .rename("itinerary_duration_sum")
        )
        g = g.merge(itinerary_sum, on="book_ref", how="left")

    out = g.merge(
        bookings[["book_ref", "total_amount"]],
        on="book_ref", how="left", validate="one_to_one",
    )
    # Log-transform target for modeling stability / 对目标取对数以增强建模稳定性
    total_amount = pd.to_numeric(out["total_amount"], errors="coerce")
    total_amount = total_amount.where(total_amount > 0, np.nan)
    out["log_total_amount"] = np.log(total_amount)
    return out
