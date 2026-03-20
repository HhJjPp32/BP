"""
visualizer.py — 可视化模块
生成：散点图、残差图、特征重要性图、训练曲线
"""
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # 非交互后端，避免无 GUI 环境报错
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# ── 中文字体自动配置 ─────────────────────────────────────────
def _setup_chinese_font() -> None:
    """
    配置中文字体。
    核心原则：所有含中文的字符串不使用 $...$ LaTeX 数学模式，
    直接用 Unicode 纯文本，避免 mathtext 渲染器不支持中文的问题。
    """
    chinese_fonts = [
        "Microsoft YaHei", "SimHei", "SimSun", "KaiTi", "FangSong", "STSong",
    ]
    fm._load_fontmanager(try_read_cache=False)
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = next((f for f in chinese_fonts if f in available), "DejaVu Sans")
    matplotlib.rcParams.update({
        "font.sans-serif":    [chosen, "DejaVu Sans"],
        "font.family":        "sans-serif",
        "axes.unicode_minus": False,
    })

_setup_chinese_font()

logger = logging.getLogger(__name__)


class Visualizer:
    """统一绘图接口，所有图表保存到 output/ 目录。"""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.viz_cfg = config.get("visualization", {})
        self.output_dir = config.get("paths", {}).get("output_dir", "output")
        self.dpi = self.viz_cfg.get("dpi", 150)
        self.figsize = tuple(self.viz_cfg.get("figure_size", [8, 6]))
        style = self.viz_cfg.get("style", "seaborn-v0_8-whitegrid")
        try:
            plt.style.use(style)
        except Exception:
            plt.style.use("default")
        # style.use() 会重置 rcParams，必须在其后重新应用中文字体
        _setup_chinese_font()
        self._ensure_output_dir()

    def _ensure_output_dir(self) -> None:
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def _save(self, fig: plt.Figure, filename: str) -> str:
        save_path = os.path.join(self.output_dir, filename)
        fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"图表已保存: {save_path}")
        return save_path

    # ── 1. 真实值 vs 预测值散点图 ─────────────

    def plot_pred_vs_true(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metrics: Optional[Dict] = None,
        dataset_name: str = "Test",
        filename: str = "pred_vs_true.png",
    ) -> str:
        """
        绘制预测值 vs 真实值散点图，附对角参考线与指标注释。
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        ax.scatter(y_true, y_pred, alpha=0.7, edgecolors="k", linewidths=0.5,
                   color="#4C72B0", s=50, label="样本点")

        # 对角线
        lo = min(y_true.min(), y_pred.min()) * 0.95
        hi = max(y_true.max(), y_pred.max()) * 1.05
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="理想线 (y=x)")

        # ±10% 误差带
        ax.fill_between([lo, hi], [lo * 0.9, hi * 0.9], [lo * 1.1, hi * 1.1],
                        alpha=0.1, color="red", label="±10% 误差带")

        ax.set_xlabel("真实值  Nexp  (kN)", fontsize=12)
        ax.set_ylabel("预测值  Npred  (kN)", fontsize=12)
        ax.set_title(f"真实值 vs 预测值  [{dataset_name}]", fontsize=13)
        ax.legend(fontsize=10)

        if metrics:
            info = (
                f"R² = {metrics.get('r2', 0):.4f}\n"
                f"RMSE = {metrics.get('rmse', 0):.2f}\n"
                f"COV = {metrics.get('cov', 0):.4f}"
            )
            ax.text(0.05, 0.95, info, transform=ax.transAxes,
                    verticalalignment="top", fontsize=10,
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        ax.set_aspect("equal", adjustable="box")
        return self._save(fig, filename)

    # ── 2. 残差分布图 ──────────────────────────

    def plot_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        dataset_name: str = "Test",
        filename: str = "residuals.png",
    ) -> str:
        """
        绘制残差分布图（残差 vs 预测值 + 直方图）。
        """
        residuals = y_pred - y_true
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 左：残差 vs 预测值
        ax = axes[0]
        ax.scatter(y_pred, residuals, alpha=0.7, edgecolors="k",
                   linewidths=0.4, color="#DD8452", s=40)
        ax.axhline(0, color="r", linestyle="--", linewidth=1.5)
        ax.set_xlabel("预测值 (kN)", fontsize=11)
        ax.set_ylabel("残差 = 预测 - 真实 (kN)", fontsize=11)
        ax.set_title(f"残差分布 [{dataset_name}]", fontsize=12)

        # 右：残差直方图
        ax2 = axes[1]
        ax2.hist(residuals, bins=25, edgecolor="black", color="#4878D0", alpha=0.8)
        ax2.axvline(0, color="r", linestyle="--", linewidth=1.5)
        ax2.axvline(np.mean(residuals), color="green", linestyle="-",
                    linewidth=1.5, label=f"均值={np.mean(residuals):.2f}")
        ax2.set_xlabel("残差 (kN)", fontsize=11)
        ax2.set_ylabel("频数", fontsize=11)
        ax2.set_title("残差直方图", fontsize=12)
        ax2.legend(fontsize=9)

        plt.tight_layout()
        return self._save(fig, filename)

    # ── 3. 特征重要性图 ────────────────────────

    def plot_feature_importance(
        self,
        feature_names: List[str],
        importances: np.ndarray,
        title: str = "特征重要性（排列重要性）",
        filename: str = "feature_importance.png",
        top_n: int = 20,
    ) -> str:
        """
        绘制特征重要性水平条形图，按重要性降序排列。
        """
        # 按重要性降序
        idx = np.argsort(importances)[::-1][:top_n]
        sorted_names = [feature_names[i] for i in idx]
        sorted_imp = importances[idx]

        fig, ax = plt.subplots(figsize=(self.figsize[0], max(6, len(sorted_names) * 0.4)))
        bars = ax.barh(range(len(sorted_names)), sorted_imp[::-1],
                       color="#4C72B0", edgecolor="black", alpha=0.8)
        ax.set_yticks(range(len(sorted_names)))
        ax.set_yticklabels(sorted_names[::-1], fontsize=9)
        ax.set_xlabel("重要性得分", fontsize=11)
        ax.set_title(title, fontsize=12)

        # 数值标签
        for bar, val in zip(bars, sorted_imp[::-1]):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", ha="left", fontsize=8)

        plt.tight_layout()
        return self._save(fig, filename)

    # ── 4. 训练损失曲线 ────────────────────────

    def plot_training_history(
        self,
        histories: List[Dict],
        filename: str = "training_curves.png",
    ) -> str:
        """
        绘制各折的训练/验证损失曲线。
        """
        n_folds = len(histories)
        cols = min(n_folds, 3)
        rows = (n_folds + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
        axes = np.array(axes).flatten() if n_folds > 1 else [axes]

        for i, history in enumerate(histories):
            ax = axes[i]
            ax.plot(history["train_loss"], label="训练损失", color="#4C72B0")
            ax.plot(history["val_loss"], label="验证损失", color="#DD8452", linestyle="--")
            ax.set_xlabel("Epoch", fontsize=10)
            ax.set_ylabel("MSE Loss", fontsize=10)
            ax.set_title(f"Fold {i + 1}", fontsize=11)
            ax.legend(fontsize=9)
            ax.set_yscale("log")  # 对数坐标更易观察

        # 隐藏多余子图
        for j in range(n_folds, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle("各折训练/验证损失曲线", fontsize=13, y=1.01)
        plt.tight_layout()
        return self._save(fig, filename)

    # ── 5. 预测比 ξ 分布图 ─────────────────────

    def plot_ratio_distribution(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        dataset_name: str = "Test",
        filename: str = "ratio_distribution.png",
    ) -> str:
        """
        绘制预测比 ξ = y_pred/y_true 的分布，
        用于直观展示 COV 指标。
        """
        xi = y_pred / (y_true + 1e-10)
        mu = np.mean(xi)
        sigma = np.std(xi, ddof=1)
        cov = sigma / abs(mu)

        fig, ax = plt.subplots(figsize=self.figsize)
        ax.hist(xi, bins=25, edgecolor="black", color="#55A868", alpha=0.8, density=True)
        ax.axvline(1.0, color="r", linestyle="--", linewidth=1.5, label="理想值 ξ=1.0")
        ax.axvline(mu, color="navy", linestyle="-", linewidth=1.5,
                   label=f"μ_ξ = {mu:.4f}")
        ax.axvspan(mu - sigma, mu + sigma, alpha=0.15, color="navy",
                   label=f"μ ± σ  (σ={sigma:.4f})")
        ax.set_xlabel("预测比  ξ = Npred / Nexp", fontsize=11)
        ax.set_ylabel("概率密度", fontsize=11)
        ax.set_title(f"预测比分布  [{dataset_name}]  COV = {cov:.4f}", fontsize=12)
        ax.legend(fontsize=9)
        plt.tight_layout()
        return self._save(fig, filename)

    # ── 6. 特征选择肘部图 ─────────────────────

    def plot_feature_selection_elbow(
        self,
        n_features_list: List[int],
        r2_list: List[float],
        rmse_list: List[float],
        cov_list: List[float],
        filename: str = "feature_selection_elbow.png",
    ) -> str:
        """绘制特征选择各指标随特征数变化的折线图（肘部图）。"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for ax, values, label, color in zip(
            axes,
            [r2_list, rmse_list, cov_list],
            ["R²", "RMSE", "COV"],
            ["#4C72B0", "#DD8452", "#55A868"],
        ):
            ax.plot(n_features_list, values, "o-", color=color,
                    linewidth=2, markersize=7)
            ax.set_xlabel("特征数量", fontsize=11)
            ax.set_ylabel(label, fontsize=11)
            ax.set_title(f"{label} vs 特征数", fontsize=12)
            ax.grid(True, alpha=0.4)
            ax.invert_xaxis()

        plt.suptitle("递归特征消除 - 指标变化（肘部图）", fontsize=13)
        plt.tight_layout()
        return self._save(fig, filename)
