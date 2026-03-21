"""
train.py — 主训练入口
运行：python train.py [--config config/config.yaml]

完整流程：
  数据加载 → 预处理（特征标准化 + 目标 log1p 变换）
  → KFold 训练 + Optuna 优化 → 原始尺度评估 → 可视化

输出目录：
  每次完整训练自动创建 output/YYYYMMDD_HHMMSS/ 目录保存所有结果。
  Optuna 中间过程训练仅写入 logs/training.log，不单独建文件夹。
"""
import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from src.data_loader import DataLoader
from src.evaluator import Evaluator
from src.model_trainer import ModelTrainer
from src.preprocessor import Preprocessor
from src.shap_analyzer import compute_shap_values, save_shap_report
from src.utils import ensure_dirs, load_config, setup_logging
from src.visualizer import Visualizer

logger = logging.getLogger(__name__)


def _setup_run_dir(cfg: dict) -> str:
    """
    创建以训练时间命名的输出目录，并将 config 中所有输出路径
    重定向到该目录。日志和 Optuna 数据库路径保持不变。

    Returns:
        run_dir: 本次训练的输出目录路径（字符串）
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output = cfg.get("paths", {}).get("output_dir", "output")
    run_dir = str(Path(base_output) / timestamp)
    Path(run_dir).mkdir(parents=True, exist_ok=True)

    # 覆盖所有"最终输出"相关路径，日志/Optuna 路径不变
    cfg["paths"]["output_dir"]      = run_dir
    cfg["paths"]["model_save_path"] = str(Path(run_dir) / "best_model.pth")
    cfg["paths"]["scaler_path"]     = str(Path(run_dir) / "scaler.pkl")
    cfg["paths"]["report_path"]     = str(Path(run_dir) / "evaluation_report.txt")

    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BP 神经网络训练 - CFDST 承载力预测"
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml",
        help="配置文件路径",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── 1. 加载配置 ───────────────────────────────────────────
    cfg = load_config(args.config)
    log_cfg = cfg.get("logging", {})
    setup_logging(
        log_level=log_cfg.get("level", "INFO"),
        log_file=log_cfg.get("log_file", "logs/training.log"),
        log_format=log_cfg.get("format"),
    )

    # ── 创建本次训练的时间戳输出目录 ─────────────────────────
    run_dir = _setup_run_dir(cfg)
    paths = cfg.get("paths", {})
    ensure_dirs([paths.get("log_dir", "logs")])

    logger.info("=" * 62)
    logger.info("  BP 神经网络训练启动（CFDST 极限承载力预测）")
    logger.info(f"  本次输出目录: {run_dir}")
    logger.info("=" * 62)

    # ── 2. 加载数据 ───────────────────────────────────────────
    loader = DataLoader(cfg)
    df = loader.load()
    loader.summary(df)
    X_df, y = loader.get_features_target(df)

    # ── 3. 划分训练/测试集 ────────────────────────────────────
    data_cfg = cfg.get("data", {})
    X_train_df, X_test_df, y_train_s, y_test_s = train_test_split(
        X_df, y,
        test_size=data_cfg.get("test_size", 0.2),
        random_state=data_cfg.get("random_state", 42),
    )
    logger.info(f"训练集: {len(X_train_df)} 条 | 测试集: {len(X_test_df)} 条")

    y_train_np = y_train_s.to_numpy(dtype=float)
    y_test_np  = y_test_s.to_numpy(dtype=float)

    # ── 4. 数据预处理 ─────────────────────────────────────────
    preprocessor = Preprocessor(cfg)

    # 特征：中位数插补 + StandardScaler
    X_train = preprocessor.fit_transform(X_train_df)
    X_test  = preprocessor.transform(X_test_df)

    # 目标：log1p 变换（训练在 log 空间，评估在原始空间）
    y_train_log = preprocessor.fit_transform_y(y_train_np)
    # y_test 不需要 log 变换版本，评估全程使用原始尺度 y_test_np

    # 保存预处理器（含 scaler 和变换参数）
    preprocessor.save(paths.get("scaler_path", "output/scaler.pkl"))

    # ── 5. 模型训练（KFold + 可选 Optuna）───────────────────
    trainer = ModelTrainer(cfg)
    # 注入 preprocessor，使 Optuna 内部可逆变换计算 COV
    best_model = trainer.fit_cv(
        X_train, y_train_log, preprocessor=preprocessor
    )
    trainer.save_model(best_model)

    # ── 6. 测试集评估（原始尺度）────────────────────────────
    device = trainer.device
    best_model.eval()

    with torch.no_grad():
        y_pred_log_test = best_model(
            torch.tensor(X_test, dtype=torch.float32).to(device)
        ).cpu().numpy()
        y_pred_log_train = best_model(
            torch.tensor(X_train, dtype=torch.float32).to(device)
        ).cpu().numpy()

    # 逆变换 → 原始尺度（kN）
    y_pred_test  = preprocessor.inverse_transform_y(y_pred_log_test)
    y_pred_train = preprocessor.inverse_transform_y(y_pred_log_train)

    # 防止预测值为负（极少数情况）
    y_pred_test  = np.clip(y_pred_test,  0, None)
    y_pred_train = np.clip(y_pred_train, 0, None)

    evaluator = Evaluator(cfg)
    logger.info("\n【测试集评估（原始尺度 kN）】")
    test_metrics = evaluator.evaluate(y_test_np, y_pred_test, dataset_name="Test")
    evaluator.save_report(
        test_metrics, "Test",
        paths.get("report_path", "output/evaluation_report.txt"),
    )

    logger.info("\n【训练集评估（原始尺度 kN）】")
    train_metrics = evaluator.evaluate(y_train_np, y_pred_train, dataset_name="Train")
    evaluator.save_report(
        train_metrics, "Train",
        paths.get("report_path", "output/evaluation_report.txt"),
    )

    # 过拟合检查
    r2_gap = train_metrics["r2"] - test_metrics["r2"]
    if r2_gap > 0.05:
        logger.warning(
            f"过拟合迹象：Train R²={train_metrics['r2']:.4f} "
            f"vs Test R²={test_metrics['r2']:.4f}（差距={r2_gap:.4f}）"
        )

    # ── 7. 计算排列重要性 ────────────────────────────────────
    logger.info("计算排列重要性（Permutation Importance）...")
    feature_names = list(X_df.columns)
    importances = _permutation_importance(
        best_model, X_test, y_test_np,
        preprocessor, device, n_repeats=10,
    )

    # ── 8. 可视化 ────────────────────────────────────────────
    viz = Visualizer(cfg)
    viz.plot_pred_vs_true(
        y_test_np, y_pred_test, test_metrics,
        dataset_name="Test", filename="pred_vs_true_test.png",
    )
    viz.plot_residuals(y_test_np, y_pred_test, dataset_name="Test")
    viz.plot_feature_importance(feature_names, importances)
    viz.plot_training_history(trainer.training_history_)
    viz.plot_ratio_distribution(y_test_np, y_pred_test, dataset_name="Test")

    # ── 9. SHAP 可解释性分析 ─────────────────────────────────
    logger.info("=" * 62)
    logger.info("  SHAP 可解释性分析")
    logger.info("=" * 62)
    try:
        shap_values, _ = compute_shap_values(
            best_model,
            X_train=X_train,
            X_explain=X_test,
            device=device,
            n_background=min(100, len(X_train)),
        )
        # 估算 base_value（背景预测均值，逆变换回 kN）
        import torch as _torch
        with _torch.no_grad():
            bg_pred_log = best_model(
                _torch.tensor(X_train[:100], dtype=_torch.float32).to(device)
            ).cpu().numpy()
        base_value = float(preprocessor.inverse_transform_y(bg_pred_log).mean())

        # 保存 SHAP 文本报告
        save_shap_report(
            shap_values, feature_names,
            paths.get("output_dir", "output") + "/shap_report.txt",
        )

        # 绘制所有 SHAP 图表
        viz.plot_shap_summary(shap_values, X_test, feature_names)
        viz.plot_shap_bar(shap_values, feature_names)
        viz.plot_shap_waterfall(
            shap_values, X_test, feature_names,
            y_pred=y_pred_test, y_true=y_test_np,
            base_value=base_value,
        )
        viz.plot_shap_dependence(shap_values, X_test, feature_names, top_n=4)
        viz.plot_shap_heatmap(shap_values, feature_names)
        logger.info("SHAP 分析完成，图表已保存至 output/")
    except Exception as e:
        logger.warning(f"SHAP 分析失败（不影响主流程）: {e}")

    # ── 10. KFold 指标汇总 ──────────────────────────────────
    logger.info("\n" + "=" * 62)
    logger.info("  KFold 各折汇总")
    logger.info("=" * 62)
    for r in trainer.cv_results_:
        logger.info(
            f"  Fold {r['fold']} | RMSE={r['rmse']:.2f} kN | COV={r['cov']:.4f}"
        )
    avg_rmse = np.mean([r["rmse"] for r in trainer.cv_results_])
    avg_cov  = np.mean([r["cov"]  for r in trainer.cv_results_])
    logger.info(f"  平均     | RMSE={avg_rmse:.2f} kN | COV={avg_cov:.4f}")

    logger.info("\n" + "=" * 62)
    logger.info("  训练完成！")
    logger.info(f"  Test R²   : {test_metrics['r2']:.6f}")
    logger.info(f"  Test RMSE : {test_metrics['rmse']:.2f} kN")
    logger.info(f"  Test COV  : {test_metrics['cov']:.6f}  ← 工程核心指标")
    logger.info(f"  模型保存  : {paths.get('model_save_path', 'output/best_model.pth')}")
    logger.info(f"  图表保存  : {paths.get('output_dir', 'output')}/")
    logger.info("=" * 62)


def _permutation_importance(
    model,
    X: np.ndarray,
    y_orig: np.ndarray,       # 原始尺度目标（用于计算 RMSE）
    preprocessor: Preprocessor,
    device: torch.device,
    n_repeats: int = 10,
) -> np.ndarray:
    """
    排列重要性：逐列打乱 → 测量原始尺度 RMSE 增量。
    增量越大说明该特征越重要。
    评估全程在原始尺度（kN）进行。
    """
    model.eval()
    with torch.no_grad():
        base_pred_log = model(
            torch.tensor(X, dtype=torch.float32).to(device)
        ).cpu().numpy()
    base_pred = preprocessor.inverse_transform_y(base_pred_log)
    base_rmse = float(np.sqrt(np.mean((base_pred - y_orig) ** 2)))

    n_features = X.shape[1]
    importances = np.zeros(n_features)
    rng = np.random.default_rng(42)

    for i in range(n_features):
        scores = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            X_perm[:, i] = rng.permutation(X_perm[:, i])
            with torch.no_grad():
                perm_pred_log = model(
                    torch.tensor(X_perm, dtype=torch.float32).to(device)
                ).cpu().numpy()
            perm_pred = preprocessor.inverse_transform_y(perm_pred_log)
            perm_rmse = float(np.sqrt(np.mean((perm_pred - y_orig) ** 2)))
            scores.append(perm_rmse - base_rmse)
        importances[i] = float(np.mean(scores))
        logger.debug(f"  特征 [{i}] 重要性: {importances[i]:.2f}")

    return importances


if __name__ == "__main__":
    main()
