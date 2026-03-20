"""
utils.py — 日志配置与通用辅助函数
"""
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
) -> logging.Logger:
    """配置根日志器，同时输出到控制台和文件（可选）。"""
    fmt = log_format or "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    level = getattr(logging, log_level.upper(), logging.INFO)

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        ensure_dirs([os.path.dirname(log_file)])
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=True)
    return logging.getLogger()


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """加载 YAML 配置文件，返回字典。"""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件未找到: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """将字典保存为 JSON 文件。"""
    ensure_dirs([os.path.dirname(filepath)])
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    logging.getLogger(__name__).info(f"JSON 已保存: {filepath}")


def load_json(filepath: str) -> Optional[Dict[str, Any]]:
    """加载 JSON 文件，文件不存在时返回 None。"""
    path = Path(filepath)
    if not path.exists():
        logging.getLogger(__name__).warning(f"JSON 文件不存在: {filepath}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dirs(paths: list) -> None:
    """批量创建目录（若不存在）。"""
    for p in paths:
        if p:
            Path(p).mkdir(parents=True, exist_ok=True)
