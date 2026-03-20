"""
data_loader.py — 数据加载模块
负责读取 CSV、处理特殊列名（空格/斜杠）、拆分特征与目标变量
"""
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class DataLoader:
    """
    加载 CFDST 数据集 CSV 文件。

    特殊处理：
      - 列名含空格（如 "R e,out"）、斜杠（如 "Nexp/kN"）等特殊字符
      - 内部维护原始列名 → 清洗列名的映射，对外透明
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.data_cfg = config.get("data", {})
        self.paths_cfg = config.get("paths", {})
        self.raw_data_path: str = self.paths_cfg.get(
            "raw_data", "data/raw_data.csv"
        )
        self.target_column: str = self.data_cfg.get("target_column", "Nexp/kN")
        self.feature_columns: List[str] = self.data_cfg.get("feature_columns", [])
        # 原始列名 → 内部使用列名的映射（清洗后）
        self._col_map: Dict[str, str] = {}
        self._inv_col_map: Dict[str, str] = {}

    # ── 公共接口 ───────────────────────────────

    def load(self) -> pd.DataFrame:
        """
        读取 CSV 文件，清洗列名后返回 DataFrame。
        自动尝试 utf-8 / gbk / latin-1 编码。
        """
        path = Path(self.raw_data_path)
        if not path.exists():
            raise FileNotFoundError(f"数据文件未找到: {self.raw_data_path}")

        df: Optional[pd.DataFrame] = None
        for encoding in ["utf-8", "gbk", "utf-8-sig", "latin-1"]:
            try:
                df = pd.read_csv(path, encoding=encoding)
                logger.info(f"数据加载成功（编码: {encoding}）: {df.shape}")
                break
            except UnicodeDecodeError:
                continue

        if df is None:
            raise ValueError(f"无法读取文件（编码均失败）: {self.raw_data_path}")

        df = self._clean_columns(df)
        logger.info(f"清洗后列名: {list(df.columns)}")

        # 更新 target_column 为清洗后的名称
        cleaned_target = self._col_map.get(self.target_column, self.target_column)
        if cleaned_target != self.target_column:
            logger.info(
                f"目标列名映射: '{self.target_column}' → '{cleaned_target}'"
            )
            self.target_column = cleaned_target

        return df

    def get_features_target(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """拆分特征矩阵 X 和目标向量 y。"""
        if self.target_column not in df.columns:
            raise ValueError(
                f"目标列 '{self.target_column}' 不在数据集中。\n"
                f"可用列: {list(df.columns)}\n"
                f"提示：检查 config.yaml 中的 target_column 设置。"
            )

        if self.feature_columns:
            # 将配置中的列名也做清洗映射
            clean_feat_cols = [
                self._col_map.get(c, c) for c in self.feature_columns
            ]
            missing = [c for c in clean_feat_cols if c not in df.columns]
            if missing:
                raise ValueError(f"以下特征列不存在: {missing}")
            X = df[clean_feat_cols]
        else:
            X = df.drop(columns=[self.target_column])

        y = df[self.target_column]
        logger.info(f"特征数量: {X.shape[1]}，样本数: {X.shape[0]}")
        logger.info(f"特征列: {list(X.columns)}")
        logger.info(
            f"目标列统计: min={y.min():.2f}, max={y.max():.2f}, "
            f"mean={y.mean():.2f}, std={y.std():.2f}"
        )
        return X, y

    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """获取特征列名列表。"""
        X, _ = self.get_features_target(df)
        return list(X.columns)

    def summary(self, df: pd.DataFrame) -> None:
        """打印数据集摘要。"""
        logger.info("=" * 55)
        logger.info(f"  数据集形状: {df.shape}")
        missing = df.isnull().sum()
        if missing.sum() > 0:
            logger.info(f"  缺失值:\n{missing[missing > 0]}")
        else:
            logger.info("  缺失值: 无")
        logger.info(f"\n{df.describe().to_string()}")
        logger.info("=" * 55)

    # ── 列名清洗 ──────────────────────────────

    def _clean_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        对列名进行标准化处理：
          1. 去除首尾空格
          2. 将列名中的空格替换为下划线
          3. 将斜杠 / 替换为 _per_（保留语义）
          4. 将逗号替换为下划线
          5. 保存原始名 → 清洗名的映射

        示例：
          "R e,out"  → "R_e_out"
          "Nexp/kN"  → "Nexp_per_kN"
          "fy,out"   → "fy_out"
        """
        new_columns = []
        self._col_map = {}
        self._inv_col_map = {}

        for col in df.columns:
            original = col
            cleaned = col.strip()
            cleaned = cleaned.replace("/", "_per_")
            cleaned = re.sub(r"[\s,]+", "_", cleaned)
            cleaned = re.sub(r"_+", "_", cleaned)   # 合并连续下划线
            cleaned = cleaned.strip("_")

            self._col_map[original] = cleaned
            self._inv_col_map[cleaned] = original
            new_columns.append(cleaned)

        df = df.copy()
        df.columns = new_columns
        return df

    def get_original_column_name(self, cleaned_name: str) -> str:
        """根据清洗后的列名获取原始列名。"""
        return self._inv_col_map.get(cleaned_name, cleaned_name)
