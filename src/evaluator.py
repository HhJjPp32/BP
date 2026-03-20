"""
evaluator.py — 模型评估模块
实现 R², RMSE, MAE, MAPE 及核心工程指标 COV
"""
import logging
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """决定系数 R²。"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """均方根误差 RMSE。"""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """平均绝对误差 MAE。"""
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """平均绝对百分比误差 MAPE (%)。"""
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)


def cov_metric(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    变异系数 COV（核心工程评估指标）。

    定义：
        ξ   = y_pred / y_true  （预测比）
        μ_ξ = mean(ξ)
        σ_ξ = std(ξ)
        COV = σ_ξ / μ_ξ

    COV 越小，预测稳定性越高。
    μ_ξ 越接近 1.0，系统偏差越小。

    返回字典包含: cov, mean_ratio, std_ratio
    """
    xi = y_pred / (y_true + 1e-10)       # 防止除零
    mu_xi = float(np.mean(xi))
    sigma_xi = float(np.std(xi, ddof=1))  # 样本标准差
    cov = sigma_xi / (abs(mu_xi) + 1e-10)
    return {
        "cov": cov,
        "mean_ratio": mu_xi,   # μ_ξ，理想值≈1.0
        "std_ratio": sigma_xi, # σ_ξ
    }


class Evaluator:
    """
    统一评估接口，聚合所有指标并输出报告。
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.eval_cfg = config.get("evaluation", {})

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        dataset_name: str = "Test",
    ) -> Dict[str, float]:
        """
        计算并记录所有评估指标。

        Args:
            y_true: 真实值数组
            y_pred: 预测值数组
            dataset_name: 数据集名称（用于日志）

        Returns:
            包含所有指标的字典
        """
        y_true = np.asarray(y_true, dtype=np.float64).ravel()
        y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

        metrics: Dict[str, float] = {}
        metrics["r2"]   = r2_score(y_true, y_pred)
        metrics["rmse"] = rmse(y_true, y_pred)
        metrics["mae"]  = mae(y_true, y_pred)
        metrics["mape"] = mape(y_true, y_pred)

        cov_dict = cov_metric(y_true, y_pred)
        metrics.update(cov_dict)

        self._log_metrics(metrics, dataset_name)
        return metrics

    def _log_metrics(self, metrics: Dict[str, float], name: str) -> None:
        sep = "=" * 52
        logger.info(sep)
        logger.info(f"  评估结果 [{name}]")
        logger.info(sep)
        logger.info(f"  R²          : {metrics['r2']:.6f}")
        logger.info(f"  RMSE        : {metrics['rmse']:.4f}")
        logger.info(f"  MAE         : {metrics['mae']:.4f}")
        logger.info(f"  MAPE        : {metrics['mape']:.4f} %")
        logger.info("─" * 52)
        logger.info(f"  COV         : {metrics['cov']:.6f}  ← 核心工程指标")
        logger.info(f"  μ_ξ (均值比): {metrics['mean_ratio']:.6f}  (理想≈1.0)")
        logger.info(f"  σ_ξ (标准差): {metrics['std_ratio']:.6f}")
        logger.info(sep)

    def save_report(
        self,
        metrics: Dict[str, float],
        dataset_name: str = "Test",
        report_path: str = "output/evaluation_report.txt",
    ) -> None:
        """将评估报告追加写入文本文件。"""
        from src.utils import ensure_dirs
        import os
        ensure_dirs([os.path.dirname(report_path)])
        with open(report_path, "a", encoding="utf-8") as f:
            f.write(f"\n{'=' * 52}\n")
            f.write(f"  评估结果 [{dataset_name}]\n")
            f.write(f"{'=' * 52}\n")
            for k, v in metrics.items():
                f.write(f"  {k:15s}: {v:.6f}\n")
        logger.info(f"报告已追加至: {report_path}")
