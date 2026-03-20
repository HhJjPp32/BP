"""
preprocessor.py — 数据预处理模块

特征预处理：
  - 中位数插补缺失值（Nexp/kN 数据无缺失，作为保险措施保留）
  - StandardScaler 标准化（BP 网络对尺度敏感，必须标准化）

目标变量预处理（关键优化）：
  - log1p 变换：Nexp/kN 范围 228~15850（std=2703），log 变换后
    分布更均匀，MSE 不再被大值主导，显著改善 COV 和 MAPE
  - 预测后使用 expm1 逆变换回原始尺度
"""
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class Preprocessor:
    """
    数据预处理流水线：
      特征侧：中位数插补 → StandardScaler
      目标侧：log1p 变换（可选，由 config 控制）
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.data_cfg = config.get("data", {})
        self.log_transform_target: bool = self.data_cfg.get(
            "log_transform_target", True
        )
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self._is_fitted: bool = False
        self.feature_names_: list = []

    # ── 特征预处理 ─────────────────────────────

    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> np.ndarray:
        """在训练集上拟合并变换特征矩阵。"""
        self.feature_names_ = list(X.columns)
        missing_before = int(X.isnull().sum().sum())

        X_imputed = self.imputer.fit_transform(X)
        if missing_before > 0:
            logger.info(f"中位数插补：修复 {missing_before} 个缺失值")

        X_scaled = self.scaler.fit_transform(X_imputed)
        self._is_fitted = True
        logger.info(
            f"特征标准化完成：{X.shape[1]} 个特征，"
            f"均值范围 [{self.scaler.mean_.min():.3f}, {self.scaler.mean_.max():.3f}]"
        )
        return X_scaled

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """使用已拟合的参数变换新数据。"""
        if not self._is_fitted:
            raise RuntimeError("Preprocessor 尚未拟合，请先调用 fit_transform。")
        X_imputed = self.imputer.transform(X)
        return self.scaler.transform(X_imputed)

    # ── 目标变量变换 ───────────────────────────

    def fit_transform_y(self, y: np.ndarray) -> np.ndarray:
        """
        对目标变量应用 log1p 变换。

        适用场景：
          Nexp/kN 范围 228~15850，直接用 MSE 训练时，
          大值（如 15000+）的梯度主导训练，导致小值预测差。
          log1p 变换后范围约 5.4~9.7，分布均匀，训练更稳定。

        Returns:
            log1p(y) 变换后的数组
        """
        y_arr = np.asarray(y, dtype=np.float64)
        if self.log_transform_target:
            if np.any(y_arr <= 0):
                raise ValueError(
                    f"目标变量含非正值（min={y_arr.min():.4f}），"
                    "无法进行 log1p 变换。请检查数据或关闭 log_transform_target。"
                )
            y_transformed = np.log1p(y_arr)
            logger.info(
                f"目标变量 log1p 变换：原始范围 [{y_arr.min():.1f}, {y_arr.max():.1f}] "
                f"→ 变换后 [{y_transformed.min():.4f}, {y_transformed.max():.4f}]"
            )
            return y_transformed
        return y_arr

    def inverse_transform_y(self, y_transformed: np.ndarray) -> np.ndarray:
        """
        将网络输出逆变换回原始尺度（expm1）。
        用于计算 COV、RMSE 等工程指标（必须在原始尺度计算）。
        """
        y_arr = np.asarray(y_transformed, dtype=np.float64)
        if self.log_transform_target:
            return np.expm1(y_arr)
        return y_arr

    # ── 序列化 ────────────────────────────────

    def save(self, path: str) -> None:
        """序列化到磁盘。"""
        from src.utils import ensure_dirs
        import os
        ensure_dirs([os.path.dirname(path)])
        joblib.dump(
            {
                "imputer": self.imputer,
                "scaler": self.scaler,
                "feature_names": self.feature_names_,
                "log_transform_target": self.log_transform_target,
            },
            path,
        )
        logger.info(f"预处理器已保存: {path}")

    @classmethod
    def load(cls, path: str, config: dict) -> "Preprocessor":
        """从磁盘加载预处理器。"""
        if not Path(path).exists():
            raise FileNotFoundError(f"预处理器文件未找到: {path}")
        obj = cls(config)
        data = joblib.load(path)
        obj.imputer = data["imputer"]
        obj.scaler = data["scaler"]
        obj.feature_names_ = data.get("feature_names", [])
        obj.log_transform_target = data.get("log_transform_target", True)
        obj._is_fitted = True
        logger.info(
            f"预处理器已加载: {path}，"
            f"log_transform_target={obj.log_transform_target}"
        )
        return obj
