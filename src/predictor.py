"""
predictor.py — 推理/预测模块
加载已保存的模型和预处理器，对新数据进行推理
"""
import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch

from src.model_trainer import BPNet, ModelTrainer
from src.preprocessor import Preprocessor

logger = logging.getLogger(__name__)


class Predictor:
    """
    加载训练好的 BP 神经网络及预处理器，
    对新样本进行推理，返回预测的承载力（kN）。
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.paths_cfg = config.get("paths", {})
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[BPNet] = None
        self.preprocessor: Optional[Preprocessor] = None

    def load(self, input_dim: int) -> None:
        """加载模型权重和预处理器。"""
        # 加载预处理器
        scaler_path = self.paths_cfg.get("scaler_path", "output/scaler.pkl")
        self.preprocessor = Preprocessor.load(scaler_path, self.config)

        # 加载模型
        trainer = ModelTrainer(self.config)
        self.model = trainer.load_model(input_dim)
        self.model.eval()
        logger.info("预测器加载完成，可以进行推理。")

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
    ) -> np.ndarray:
        """
        对输入数据进行预测。

        Args:
            X: 特征矩阵（DataFrame 或 ndarray）

        Returns:
            预测的承载力数组（kN）
        """
        if self.model is None or self.preprocessor is None:
            raise RuntimeError("请先调用 Predictor.load() 加载模型和预处理器。")

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        X_scaled = self.preprocessor.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            preds_log = self.model(X_tensor).cpu().numpy()

        # 逆变换回原始尺度（kN），与训练管道保持一致
        preds = self.preprocessor.inverse_transform_y(preds_log)
        preds = np.clip(preds, 0, None)  # 承载力不能为负
        return preds

    def predict_single(self, feature_values: list) -> float:
        """
        对单个样本进行预测。

        Args:
            feature_values: 特征值列表（与训练时特征顺序一致）

        Returns:
            单个预测值（float）
        """
        X = pd.DataFrame([feature_values],
                         columns=self.preprocessor.feature_names_  # type: ignore
                         if self.preprocessor and hasattr(self.preprocessor, "feature_names_")
                         else None)
        result = self.predict(X)
        return float(result[0])
