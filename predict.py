"""
predict.py — 主推理入口
运行：python predict.py --input data/new_data.csv [--output output/predictions.csv]
"""
import argparse
import logging
import sys

import numpy as np
import pandas as pd

from src.data_loader import DataLoader
from src.predictor import Predictor
from src.utils import ensure_dirs, load_config, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BP 神经网络推理 - CFDST 承载力预测"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入 CSV 文件路径（包含特征列，可不含目标列）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/predictions.csv",
        help="预测结果输出路径",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── 1. 加载配置 ───────────────────────────
    cfg = load_config(args.config)
    log_cfg = cfg.get("logging", {})
    setup_logging(
        log_level=log_cfg.get("level", "INFO"),
        log_file=log_cfg.get("log_file", "logs/predict.log"),
    )
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("  BP 神经网络推理启动")
    logger.info("=" * 60)

    ensure_dirs([cfg.get("paths", {}).get("output_dir", "output")])

    # ── 2. 加载输入数据 ───────────────────────
    # 临时修改配置中的数据路径
    cfg_temp = dict(cfg)
    cfg_temp["paths"] = dict(cfg.get("paths", {}))
    cfg_temp["paths"]["raw_data"] = args.input

    loader = DataLoader(cfg_temp)
    df = loader.load()

    # 目标列名在 DataLoader 中会经过清洗（如 "Nexp/kN" → "Nexp_per_kN"）
    # 使用 loader 清洗后的实际目标列名做判断
    target_col = loader.target_column  # 已同步为清洗后的列名
    has_target = target_col in df.columns

    # 获取特征列
    if has_target:
        X_df, y_true = loader.get_features_target(df)
    else:
        feature_cols = cfg.get("data", {}).get("feature_columns", [])
        X_df = df[feature_cols] if feature_cols else df
        y_true = None

    input_dim = X_df.shape[1]
    logger.info(f"输入特征数: {input_dim}，样本数: {len(X_df)}")

    # ── 3. 加载预测器 ─────────────────────────
    predictor = Predictor(cfg)
    predictor.load(input_dim)

    # ── 4. 推理 ───────────────────────────────
    y_pred = predictor.predict(X_df)
    logger.info(f"预测完成，共 {len(y_pred)} 条结果")

    # ── 5. 保存结果 ───────────────────────────
    result_df = X_df.copy()
    result_df["Nexp_predicted"] = y_pred

    if y_true is not None:
        result_df["Nexp_true"] = y_true.values
        result_df["ratio"] = y_pred / (y_true.values + 1e-10)

        # 打印简要评估
        from src.evaluator import Evaluator
        evaluator = Evaluator(cfg)
        metrics = evaluator.evaluate(y_true.to_numpy(), y_pred, "Prediction")
        logger.info(f"推理评估指标: {metrics}")

    ensure_dirs([args.output.rsplit("/", 1)[0]])
    result_df.to_csv(args.output, index=False, encoding="utf-8-sig")
    logger.info(f"预测结果已保存: {args.output}")

    # 打印前几行结果
    logger.info(f"\n{result_df[['Nexp_predicted']].head(10).to_string()}")


if __name__ == "__main__":
    main()
