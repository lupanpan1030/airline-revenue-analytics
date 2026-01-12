"""Evaluation helpers. (评估指标工具)"""
from __future__ import annotations

from typing import Dict
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline


def evaluate(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: pd.Series | None = None,
) -> Dict[str, float]:
    """Evaluate model performance with R2/RMSE/MAE. (使用 R2/RMSE/MAE 评估模型表现)"""
    if y_pred is None:
        y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return {
        "R2": float(r2_score(y_test, y_pred)),
        # Use sqrt(MSE) for RMSE to keep sklearn-version compatibility / 使用 sqrt(MSE) 计算 RMSE，兼容旧版 sklearn
        "RMSE": float(mse ** 0.5),
        "MAE": float(mean_absolute_error(y_test, y_pred)),
    }
