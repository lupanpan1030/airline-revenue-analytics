"""Training helpers and pipeline builders. (训练辅助函数与管线构建器)"""
from __future__ import annotations

from typing import List
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


def split_xy(df: pd.DataFrame, y_col: str, test_size: float = 0.3, seed: int = 42):
    """Split features/target with basic leakage prevention. (切分特征与目标，并移除潜在泄漏字段)"""
    y = df[y_col].copy()
    # Drop direct price/ID fields to avoid leakage / 移除直接价格或标识字段以避免泄漏
    X = df.drop(columns=[y_col, "total_amount", "book_ref"], errors="ignore")
    return train_test_split(X, y, test_size=test_size, random_state=seed)


def split_xy_grouped(
    df: pd.DataFrame,
    y_col: str,
    group_col: str,
    test_size: float = 0.3,
    seed: int = 42,
):
    """Split by group keys to avoid cross-group leakage. (按组切分，避免同一组跨训练/测试泄漏)"""
    y = df[y_col].copy()
    # Remove group column from features / 将分组键从特征中移除
    X = df.drop(columns=[y_col, "total_amount", group_col], errors="ignore")
    if group_col not in df.columns:
        return train_test_split(X, y, test_size=test_size, random_state=seed)
    groups = df[group_col].dropna().unique().tolist()
    if not groups:
        return train_test_split(X, y, test_size=test_size, random_state=seed)
    train_groups, test_groups = train_test_split(groups, test_size=test_size, random_state=seed)
    train_mask = df[group_col].isin(train_groups)
    return X[train_mask], X[~train_mask], y[train_mask], y[~train_mask]


def make_preprocess(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    """Build preprocessing with imputation, scaling, and one-hot encoding. (构建包含缺失值填补、标准化与独热编码的预处理器)"""
    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]), cat_cols),
    ])


def linear_pipeline(pre: ColumnTransformer) -> Pipeline:
    """Linear regression baseline pipeline. (线性回归基线管线)"""
    return Pipeline([("pre", pre), ("model", LinearRegression())])


def tree_pipeline(pre: ColumnTransformer, **params) -> Pipeline:
    """Decision tree pipeline with conservative defaults. (决策树管线，保守默认参数)"""
    params = {"random_state": 42, "max_depth": 6, "min_samples_leaf": 50, **params}
    return Pipeline([("pre", pre), ("model", DecisionTreeRegressor(**params))])


def forest_pipeline(pre: ColumnTransformer, **params) -> Pipeline:
    """Random forest pipeline with conservative defaults. (随机森林管线，保守默认参数)"""
    params = {
        "random_state": 42,
        "n_estimators": 200,
        "max_depth": 12,
        "min_samples_leaf": 20,
        "n_jobs": -1,
        **params,
    }
    return Pipeline([("pre", pre), ("model", RandomForestRegressor(**params))])
