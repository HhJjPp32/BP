"""
feature_selection.py — 递归特征消除（RFE）管道
运行：python feature_selection.py [--config config/config.yaml] [--min-features 2]

逻辑：
  训练模型（全特征）→ 评估 → 删除重要性最低的特征 → 重复
  输出各阶段 R², RMSE, COV 对比报告 + 肘部图
"""
import argparse
import logging
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from src.data_loader import DataLoader
from src.evaluator import Evaluator
from src.model_trainer import BPNet, ModelTrainer, train_model
from src.preprocessor import Preprocessor  # noqa: F401 (used in type hint)
from src.utils import ensure_dirs, load_config, setup_logging
from src.visualizer import Visualizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="递归特征消除（RFE）管道")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--min-features", type=int, default=2,
        help="最少保留的特征数（默认2）"
    )
    parser.add_argument(
        "--remove-n", type=int, default=1,
        help="每轮删除的特征数（默认1）"
    )
    return parser.parse_args()


def compute_permutation_importance(
    model: BPNet,
    X: np.ndarray,
    y: np.ndarray,             # 原始尺度 kN
    device: torch.device,
    n_repeats: int = 5,
    preprocessor=None,
) -> np.ndarray:
    """排列重要性（在原始尺度 kN 计算 RMSE 增量）。"""
    model.eval()

    def _predict_orig(X_arr):
        with torch.no_grad():
            X_t = torch.tensor(X_arr, dtype=torch.float32).to(device)
            preds_log = model(X_t).cpu().numpy()
        if preprocessor is not None:
            return np.clip(preprocessor.inverse_transform_y(preds_log), 0, None)
        return preds_log

    base_pred = _predict_orig(X)
    base_rmse = float(np.sqrt(np.mean((base_pred - y) ** 2)))

    rng = np.random.default_rng(0)
    importances = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        scores = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            X_perm[:, i] = rng.permutation(X_perm[:, i])
            perm_pred = _predict_orig(X_perm)
            scores.append(float(np.sqrt(np.mean((perm_pred - y) ** 2))) - base_rmse)
        importances[i] = float(np.mean(scores))
    return importances


def train_and_evaluate(
    cfg: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,       # log1p 变换后的训练目标
    X_test: np.ndarray,
    y_test: np.ndarray,        # 原始尺度测试目标（kN）
    feature_names: List[str],
    preprocessor: "Preprocessor" = None,
) -> Dict:
    """
    用当前特征子集训练模型并返回评估指标。
    为加速特征选择，此处使用单次训练（非KFold）。
    """
    # 简化配置：减少 epoch 和 trials 加速迭代
    fast_cfg = deepcopy(cfg)
    fast_cfg["training"]["epochs"] = 300
    fast_cfg["training"]["early_stopping_patience"] = 30
    fast_cfg["optuna"]["use_optuna"] = False  # RFE 阶段不再跑 Optuna

    # 从已有 best_params 获取参数（若有）
    trainer = ModelTrainer(fast_cfg)
    trainer._X_cache = X_train
    trainer._y_cache = y_train
    params = trainer._resolve_params()

    device = trainer.device

    # 训练（简单 80/20 内部验证）
    split = int(0.8 * len(X_train))
    model = BPNet(
        input_dim=X_train.shape[1],
        hidden_layers=params.get("hidden_layers", [64, 32]),
        activation=params.get("activation", "relu"),
        dropout_rate=params.get("dropout_rate", 0.1),
        batch_norm=params.get("batch_norm", True),
    )
    tr_cfg = {
        **fast_cfg["training"],
        "learning_rate": params.get("learning_rate", 0.001),
        "batch_size": params.get("batch_size", 32),
        "optimizer": params.get("optimizer", "adam"),
        "weight_decay": params.get("weight_decay", 1e-4),
    }
    model, _ = train_model(
        model,
        X_train[:split], y_train[:split],
        X_train[split:], y_train[split:],
        tr_cfg, device,
    )

    # 评估（逆变换回原始尺度 kN）
    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_pred_log = model(X_test_t).cpu().numpy()

    if preprocessor is not None:
        y_pred = preprocessor.inverse_transform_y(y_pred_log)
    else:
        y_pred = y_pred_log
    y_pred = np.clip(y_pred, 0, None)

    evaluator = Evaluator(cfg)
    metrics = evaluator.evaluate(y_test, y_pred, dataset_name=f"n_feat={len(feature_names)}")

    # 排列重要性（在原始尺度计算，更有工程意义）
    importances = compute_permutation_importance(
        model, X_test, y_test, device, preprocessor=preprocessor
    )

    return {
        "metrics": metrics,
        "importances": importances,
        "feature_names": feature_names,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    log_cfg = cfg.get("logging", {})
    setup_logging(
        log_level=log_cfg.get("level", "INFO"),
        log_file="logs/feature_selection.log",
    )
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("  递归特征消除（RFE）启动")
    logger.info("=" * 60)

    paths = cfg.get("paths", {})
    ensure_dirs([paths.get("output_dir", "output"), paths.get("log_dir", "logs")])

    # ── 1. 加载数据 ───────────────────────────
    loader = DataLoader(cfg)
    df = loader.load()
    X_df, y = loader.get_features_target(df)
    feature_names: List[str] = list(X_df.columns)
    logger.info(f"初始特征数: {len(feature_names)}")

    data_cfg = cfg.get("data", {})
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df, y,
        test_size=data_cfg.get("test_size", 0.2),
        random_state=data_cfg.get("random_state", 42),
    )
    y_train_np = y_train.to_numpy(dtype=float)
    y_test_np = y_test.to_numpy(dtype=float)

    # ── 2. RFE 主循环 ─────────────────────────
    results: List[Dict] = []
    current_features = feature_names.copy()

    while len(current_features) >= args.min_features:
        logger.info(f"\n当前特征数: {len(current_features)} → {current_features}")

        # 用当前特征子集预处理
        preprocessor = Preprocessor(cfg)
        X_train_sub = preprocessor.fit_transform(X_train_df[current_features])
        X_test_sub = preprocessor.transform(X_test_df[current_features])

        # 目标变量 log1p 变换（与主训练管道一致）
        y_train_log = preprocessor.fit_transform_y(y_train_np)

        # 训练 & 评估（y_test 用原始尺度评估，y_train 用 log 空间训练）
        result = train_and_evaluate(
            cfg,
            X_train_sub, y_train_log,       # 训练用 log 空间
            X_test_sub, y_test_np,           # 评估用原始尺度
            feature_names=current_features,
            preprocessor=preprocessor,       # 传入预处理器做逆变换
        )

        results.append({
            "n_features": len(current_features),
            "features": current_features.copy(),
            "r2": result["metrics"]["r2"],
            "rmse": result["metrics"]["rmse"],
            "cov": result["metrics"]["cov"],
            "importances": result["importances"].tolist(),
        })

        logger.info(
            f"  R²={result['metrics']['r2']:.4f} | "
            f"RMSE={result['metrics']['rmse']:.4f} | "
            f"COV={result['metrics']['cov']:.4f}"
        )

        # 找到重要性最低的特征并删除
        if len(current_features) <= args.min_features:
            break

        imp = result["importances"]
        remove_n = min(args.remove_n, len(current_features) - args.min_features)
        remove_idx = np.argsort(imp)[:remove_n]
        remove_feats = [current_features[i] for i in remove_idx]
        logger.info(f"  删除特征: {remove_feats}（重要性最低）")
        for f in remove_feats:
            current_features.remove(f)

    # ── 3. 保存 & 输出报告 ─────────────────────
    report_path = paths.get("output_dir", "output") + "/feature_selection_report.txt"
    _save_report(results, report_path)

    # ── 4. 可视化肘部图 ───────────────────────
    n_features_list = [r["n_features"] for r in results]
    r2_list = [r["r2"] for r in results]
    rmse_list = [r["rmse"] for r in results]
    cov_list = [r["cov"] for r in results]

    viz = Visualizer(cfg)
    viz.plot_feature_selection_elbow(n_features_list, r2_list, rmse_list, cov_list)

    logger.info("=" * 60)
    logger.info(f"  RFE 完成！报告保存于: {report_path}")
    logger.info("=" * 60)


def _save_report(results: List[Dict], path: str) -> None:
    """将 RFE 结果保存为文本报告。"""
    ensure_dirs([str(Path(path).parent)])
    with open(path, "w", encoding="utf-8") as f:
        f.write("递归特征消除（RFE）结果报告\n")
        f.write("=" * 70 + "\n")
        f.write(f"{'特征数':>6}  {'R²':>10}  {'RMSE':>10}  {'COV':>10}  特征列表\n")
        f.write("-" * 70 + "\n")
        for r in results:
            f.write(
                f"{r['n_features']:>6}  "
                f"{r['r2']:>10.6f}  "
                f"{r['rmse']:>10.4f}  "
                f"{r['cov']:>10.6f}  "
                f"{r['features']}\n"
            )
        f.write("=" * 70 + "\n\n")

        # 找到 COV 最小的特征子集（推荐）
        best_idx = int(np.argmin([r["cov"] for r in results]))
        best = results[best_idx]
        f.write(f"【推荐子集】COV最小 → 特征数={best['n_features']}\n")
        f.write(f"  特征: {best['features']}\n")
        f.write(f"  R²={best['r2']:.6f}  RMSE={best['rmse']:.4f}  COV={best['cov']:.6f}\n")

    logging.getLogger(__name__).info(f"RFE 报告已保存: {path}")


if __name__ == "__main__":
    main()
