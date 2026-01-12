"""Segment-level feature engineering. (航段级特征工程)"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd


def parse_json_en(series: pd.Series) -> pd.Series:
    """Extract English text from a JSON-like multilingual field. (从多语言 JSON 字段中提取英文文本)"""
    def _pick(x):
        try:
            if isinstance(x, str) and x.strip().startswith("{"):
                obj = json.loads(x)
                return obj.get("en")
        except Exception:
            return np.nan
        return x
    return series.apply(_pick)


def to_utc(s: pd.Series) -> pd.Series:
    """Convert timestamps to UTC, coercing invalid values. (将时间戳转换为 UTC，非法值强制为 NaT)"""
    return pd.to_datetime(s, utc=True, errors="coerce")


def minutes_between(a: pd.Series, b: pd.Series) -> pd.Series:
    """Compute minute-level deltas and drop negative durations. (计算分钟级时长并将负值置为 NaN)"""
    delta = (b - a).dt.total_seconds() / 60.0
    # Guard against negative durations / 防止出现负时长
    delta[delta < 0] = np.nan
    return delta


def days_between(a: pd.Series, b: pd.Series) -> pd.Series:
    """Compute day-level deltas and drop negative durations. (计算天级时长并将负值置为 NaN)"""
    delta = (b - a).dt.total_seconds() / (3600.0 * 24.0)
    # Guard against negative durations / 防止出现负时长
    delta[delta < 0] = np.nan
    return delta


def add_route_code(df: pd.DataFrame, dep_col: str, arr_col: str, out_col: str = "route_code") -> pd.DataFrame:
    """Build a route code as DEP-ARR. (构建 DEP-ARR 形式的路线编码)"""
    d = df.copy()
    dep = d[dep_col].astype("string")
    arr = d[arr_col].astype("string")
    d[out_col] = dep + "-" + arr
    d.loc[dep.isna() | arr.isna(), out_col] = pd.NA
    return d


def add_segment_features(tf_fl: pd.DataFrame) -> pd.DataFrame:
    """Derive segment-level features (schedule-based, no leakage). (衍生航段级特征，基于时刻表避免泄漏)"""
    d = tf_fl.copy()
    d["scheduled_departure"] = to_utc(d["scheduled_departure"])
    d["scheduled_arrival"] = to_utc(d["scheduled_arrival"])
    d["sched_flight_duration_minutes"] = minutes_between(
        d["scheduled_departure"], d["scheduled_arrival"]
    )
    d["departure_dow"] = d["scheduled_departure"].dt.dayofweek
    d["departure_hour"] = d["scheduled_departure"].dt.hour
    d["is_weekend"] = d["departure_dow"].isin([5, 6]).astype(int)

    # Normalize fare class text and derive ordinal/premium flags / 规范舱位文本并生成序数与高端标记
    d["fare_class"] = d["fare_conditions"].astype("string").str.strip().str.title()
    d["fare_class_ord"] = d["fare_class"].map({"Economy": 0, "Comfort": 1, "Business": 2}).astype("Int64")
    d["is_premium_cabin"] = d["fare_class"].isin(["Comfort", "Business"]).astype(int)
    return d
