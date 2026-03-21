"""
shap_analyzer.py — SHAP 可解释性分析模块

支持图表：
  1. Summary Plot (蜂群图)  — 全局特征重要性 + 方向
  2. Bar Plot               — 平均 |SHAP| 排名
  3. Waterfall Plot         — 单样本预测分解（最佳/最差/中位数预测）
  4. Dependence Plot        — 前 N 个特征的 SHAP 依赖图
  5. Heatmap                — 所有样本 × 所有特征的 SHAP 热力图
"""
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def _make_wrapper(model):
    """
    将 BPNet 包装为输出 (batch, 1) 的 nn.Module。
    shap.DeepExplainer 要求输出 shape[1] 存在（即至少 2D），
    而 BPNet.forward 使用了 .squeeze(-1)，输出为 1D (batch,)。
    使用真正的 nn.Module 子类，确保 shap 能正确遍历层结构。
    """
    import torch
    import torch.nn as nn

    class _Wrapper(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.inner(x).unsqueeze(-1)   # (batch,) → (batch, 1)

    return _Wrapper(model)


def compute_shap_values(
    model,
    X_train: np.ndarray,
    X_explain: np.ndarray,
    device,
    n_background: int = 100,
    seed: int = 42,
) -> Tuple[np.ndarray, object]:
    """
    使用 shap.DeepExplainer 计算 SHAP 值。

    Args:
        model        : 已训练的 BPNet（PyTorch）
        X_train      : 训练集特征（标准化后），用于构建背景分布
        X_explain    : 需要解释的样本（通常为测试集）
        device       : torch.device
        n_background : 背景样本数（越多越准，越慢）
        seed         : 随机种子

    Returns:
        shap_values  : ndarray, shape (n_samples, n_features)
        explainer    : shap.DeepExplainer 对象
    """
    try:
        import shap
        import torch
    except ImportError as e:
        raise ImportError(f"请安装依赖: pip install shap torch  ({e})")

    model.eval()
    wrapped = _make_wrapper(model)   # 包装为 2D 输出，兼容 DeepExplainer
    wrapped.eval()

    # 随机抽取背景样本
    rng = np.random.default_rng(seed)
    n_bg = min(n_background, len(X_train))
    bg_idx = rng.choice(len(X_train), n_bg, replace=False)
    background = torch.tensor(X_train[bg_idx], dtype=torch.float32).to(device)

    # 构建 DeepExplainer
    explainer = shap.DeepExplainer(wrapped, background)

    # 分批计算 SHAP 值（避免 GPU OOM）
    batch_size = 64
    shap_list = []
    for i in range(0, len(X_explain), batch_size):
        batch = torch.tensor(
            X_explain[i: i + batch_size], dtype=torch.float32
        ).to(device)
        sv = explainer.shap_values(batch)
        # DeepExplainer 回归单输出返回 list[ndarray(n,f,1)] 或 ndarray(n,f,1)
        if isinstance(sv, list):
            sv = sv[0]
        sv = np.array(sv)
        # 压缩尾部多余维度：(n,f,1) → (n,f)
        if sv.ndim == 3 and sv.shape[-1] == 1:
            sv = sv.squeeze(-1)
        shap_list.append(sv)

    shap_values = np.concatenate(shap_list, axis=0)
    logger.info(
        f"SHAP 值计算完成: shape={shap_values.shape}, "
        f"背景样本={n_bg}, 解释样本={len(X_explain)}"
    )
    return shap_values, explainer


def get_shap_summary(
    shap_values: np.ndarray,
    feature_names: List[str],
) -> dict:
    """
    计算全局 SHAP 统计摘要（用于报告）。

    Returns:
        dict: feature → mean_abs_shap, 按重要性降序排列
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    return {
        feature_names[i]: float(mean_abs[i])
        for i in order
    }


def save_shap_report(
    shap_values: np.ndarray,
    feature_names: List[str],
    output_path: str,
) -> None:
    """将 SHAP 摘要统计保存为文本报告。"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    summary = get_shap_summary(shap_values, feature_names)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("SHAP 全局特征重要性报告\n")
        f.write("=" * 50 + "\n")
        f.write(f"{'排名':>4}  {'特征名':>12}  {'平均|SHAP|':>12}\n")
        f.write("-" * 50 + "\n")
        for rank, (feat, val) in enumerate(summary.items(), 1):
            f.write(f"{rank:>4}  {feat:>12}  {val:>12.6f}\n")
        f.write("=" * 50 + "\n")
    logger.info(f"SHAP 报告已保存: {output_path}")
