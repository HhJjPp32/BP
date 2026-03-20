"""
model_trainer.py — BP 神经网络定义 + Optuna 超参数优化 + 训练循环

优化点（相较初版）：
  1. Optuna 目标函数改为联合优化 COV + RMSE（各权重 0.5）
  2. 支持 optuna_objective: "cov_rmse" | "rmse" 配置切换
  3. 优化器参数、激活函数等针对小数据集调整
"""
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1. 网络结构定义
# ─────────────────────────────────────────────

def _get_activation(name: str) -> nn.Module:
    """根据字符串名称返回激活函数实例。"""
    mapping = {
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "leaky_relu": nn.LeakyReLU(0.1),
        "elu": nn.ELU(),
    }
    act = mapping.get(name.lower())
    if act is None:
        raise ValueError(f"不支持的激活函数: {name}。可选: {list(mapping.keys())}")
    return act


class BPNet(nn.Module):
    """
    全连接 BP 神经网络（回归）。
    结构：Input → [Linear → (BN) → Activation → Dropout] × n → Linear(1)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int],
        activation: str = "tanh",
        dropout_rate: float = 0.15,
        batch_norm: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        layers: List[nn.Module] = []
        in_dim = input_dim

        for out_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, out_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(_get_activation(activation))
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            in_dim = out_dim

        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


# ─────────────────────────────────────────────
# 2. 优化器工厂
# ─────────────────────────────────────────────

def _get_optimizer(
    name: str, params, lr: float, weight_decay: float
) -> optim.Optimizer:
    # 注意：必须惰性实例化，不能将所有优化器放入字典同时创建
    # （params 是生成器，被第一个优化器消费后后续均为空）
    key = name.lower()
    if key == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif key == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif key == "sgd":
        return optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif key == "rmsprop":
        return optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"不支持的优化器: {name}。可选: adam, adamw, sgd, rmsprop")


# ─────────────────────────────────────────────
# 3. 单次训练函数
# ─────────────────────────────────────────────

def train_model(
    model: BPNet,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    train_cfg: dict,
    device: torch.device,
) -> Tuple[BPNet, Dict[str, List[float]]]:
    """
    训练 BP 网络，包含早停 + ReduceLROnPlateau + 梯度裁剪。
    注意：y_train / y_val 已经过 log1p 变换，损失在 log 空间计算。
    """
    epochs   = train_cfg.get("epochs", 1000)
    batch_sz = train_cfg.get("batch_size", 16)
    lr       = train_cfg.get("learning_rate", 0.0005)
    opt_name = train_cfg.get("optimizer", "adamw")
    wd       = train_cfg.get("weight_decay", 5e-4)
    patience = train_cfg.get("early_stopping_patience", 80)
    use_sch  = train_cfg.get("lr_scheduler", True)
    sch_pat  = train_cfg.get("lr_scheduler_patience", 30)
    sch_fac  = train_cfg.get("lr_scheduler_factor", 0.5)
    min_lr   = train_cfg.get("min_lr", 1e-7)

    X_tr = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tr = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_v  = torch.tensor(X_val,   dtype=torch.float32).to(device)
    y_v  = torch.tensor(y_val,   dtype=torch.float32).to(device)

    loader = DataLoader(
        TensorDataset(X_tr, y_tr),
        batch_size=batch_sz, shuffle=True, drop_last=False,
    )

    optimizer = _get_optimizer(opt_name, model.parameters(), lr, wd)
    criterion = nn.MSELoss()
    scheduler = None
    if use_sch:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=sch_fac,
            patience=sch_pat, min_lr=min_lr,
        )

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state: Optional[dict] = None
    no_improve = 0

    model.to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        train_loss = epoch_loss / len(X_train)

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_v), y_v).item()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        if scheduler:
            scheduler.step(val_loss)

        if val_loss < best_val_loss - 1e-9:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(
                    f"早停于 epoch {epoch}，最佳验证 MSE={best_val_loss:.6f}"
                )
                break

        if epoch % 100 == 0:
            logger.info(
                f"Epoch {epoch:5d} | Train={train_loss:.6f} | "
                f"Val={val_loss:.6f} | "
                f"LR={optimizer.param_groups[0]['lr']:.2e}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


# ─────────────────────────────────────────────
# 4. COV 计算辅助函数（Optuna 内部使用）
# ─────────────────────────────────────────────

def _compute_cov(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算预测比 COV（在原始尺度）。
    ξ = y_pred / y_true，COV = std(ξ) / mean(ξ)
    """
    xi = y_pred / (y_true + 1e-10)
    mu = float(np.mean(xi))
    sigma = float(np.std(xi, ddof=1))
    return sigma / (abs(mu) + 1e-10)


# ─────────────────────────────────────────────
# 5. ModelTrainer 主类
# ─────────────────────────────────────────────

class ModelTrainer:
    """
    BP 神经网络训练器：
      - KFold 交叉验证
      - Optuna 超参数优化（支持联合 COV+RMSE 目标）
      - 自动保存/加载最优参数
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.paths_cfg  = config.get("paths", {})
        self.model_cfg  = config.get("model", {})
        self.train_cfg  = config.get("training", {})
        self.cv_cfg     = config.get("cross_validation", {})
        self.optuna_cfg = config.get("optuna", {})
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {self.device}")

        self.best_model_: Optional[BPNet] = None
        self.best_params_: Optional[Dict] = None
        self.cv_results_: List[Dict] = []
        self.training_history_: List[Dict] = []
        self._preprocessor = None  # 由 fit_cv 注入，供 Optuna 逆变换使用

    # ── 参数解析 ────────────────────────────────

    def _resolve_params(self) -> Dict[str, Any]:
        use_optuna = self.optuna_cfg.get("use_optuna", False)
        if use_optuna:
            logger.info("use_optuna=True，启动 Optuna 搜索...")
            return self._run_optuna(self._X_cache, self._y_cache)

        best_params_path = self.paths_cfg.get("best_params_path", "logs/best_params.json")
        if Path(best_params_path).exists():
            from src.utils import load_json
            params = load_json(best_params_path)
            if params:
                logger.info(f"从 {best_params_path} 加载最优参数: {params}")
                return params

        logger.info("使用配置文件预设超参数")
        return {
            "hidden_layers": self.model_cfg.get("hidden_layers", [128, 64, 32]),
            "activation":    self.model_cfg.get("activation", "tanh"),
            "dropout_rate":  self.model_cfg.get("dropout_rate", 0.15),
            "batch_norm":    self.model_cfg.get("batch_norm", True),
            "learning_rate": self.train_cfg.get("learning_rate", 0.0005),
            "optimizer":     self.train_cfg.get("optimizer", "adamw"),
            "weight_decay":  self.train_cfg.get("weight_decay", 5e-4),
            "batch_size":    self.train_cfg.get("batch_size", 16),
        }

    # ── Optuna 优化 ─────────────────────────────

    def _run_optuna(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Optuna 搜索最优超参数。

        目标函数：
          optuna_objective = "cov_rmse" → 0.5 * COV + 0.5 * norm_RMSE（联合优化）
          optuna_objective = "rmse"     → 仅最小化 RMSE
        """
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            raise ImportError("请安装 optuna: pip install optuna")

        ss = self.optuna_cfg.get("search_space", {})
        db_path    = self.paths_cfg.get("optuna_db", "logs/optuna_study.db")
        study_name = self.optuna_cfg.get("study_name", "bp_cfdst_study")
        n_trials   = self.optuna_cfg.get("n_trials", 150)
        timeout    = self.optuna_cfg.get("timeout", 3600)
        obj_type   = self.optuna_cfg.get("optuna_objective", "cov_rmse")

        from src.utils import ensure_dirs
        ensure_dirs([os.path.dirname(db_path)])

        # 计算 RMSE 归一化基准（用全数据均值的比例）
        y_mean = float(np.mean(y)) if np.mean(y) != 0 else 1.0

        def objective(trial: "optuna.Trial") -> float:
            n_layers = trial.suggest_int(
                "n_layers",
                ss.get("n_layers", {}).get("low", 2),
                ss.get("n_layers", {}).get("high", 4),
            )
            n_units = trial.suggest_categorical(
                "n_units",
                ss.get("n_units", {}).get("choices", [64, 128, 256]),
            )
            lr = trial.suggest_float(
                "learning_rate",
                ss.get("learning_rate", {}).get("low", 5e-5),
                ss.get("learning_rate", {}).get("high", 3e-3),
                log=True,
            )
            dropout_rate = trial.suggest_float(
                "dropout_rate",
                ss.get("dropout_rate", {}).get("low", 0.0),
                ss.get("dropout_rate", {}).get("high", 0.3),
            )
            batch_size = trial.suggest_categorical(
                "batch_size",
                ss.get("batch_size", {}).get("choices", [8, 16, 32]),
            )
            weight_decay = trial.suggest_float(
                "weight_decay",
                ss.get("weight_decay", {}).get("low", 1e-5),
                ss.get("weight_decay", {}).get("high", 1e-3),
                log=True,
            )
            activation = trial.suggest_categorical(
                "activation",
                ss.get("activation", {}).get("choices", ["relu", "tanh"]),
            )
            optimizer_name = trial.suggest_categorical(
                "optimizer",
                ss.get("optimizer", {}).get("choices", ["adam", "adamw"]),
            )
            hidden_layers = [n_units] * n_layers

            kf = KFold(
                n_splits=self.cv_cfg.get("n_splits", 5),
                shuffle=self.cv_cfg.get("shuffle", True),
                random_state=self.cv_cfg.get("random_state", 42),
            )
            fold_scores = []

            for tr_idx, val_idx in kf.split(X):
                X_tr, X_val = X[tr_idx], X[val_idx]
                y_tr, y_val = y[tr_idx], y[val_idx]

                model = BPNet(
                    input_dim=X.shape[1],
                    hidden_layers=hidden_layers,
                    activation=activation,
                    dropout_rate=dropout_rate,
                    batch_norm=True,
                )
                tr_cfg = {
                    **self.train_cfg,
                    "learning_rate": lr,
                    "batch_size": batch_size,
                    "optimizer": optimizer_name,
                    "weight_decay": weight_decay,
                }
                model, _ = train_model(
                    model, X_tr, y_tr, X_val, y_val, tr_cfg, self.device
                )
                model.eval()
                with torch.no_grad():
                    preds_log = model(
                        torch.tensor(X_val, dtype=torch.float32).to(self.device)
                    ).cpu().numpy()

                # 逆变换到原始尺度计算指标（确保 COV 有工程意义）
                if self._preprocessor is not None:
                    preds_orig = self._preprocessor.inverse_transform_y(preds_log)
                    y_val_orig = self._preprocessor.inverse_transform_y(y_val)
                else:
                    preds_orig = preds_log
                    y_val_orig = y_val

                rmse_val = float(np.sqrt(np.mean((preds_orig - y_val_orig) ** 2)))

                if obj_type == "cov_rmse":
                    cov_val = _compute_cov(y_val_orig, preds_orig)
                    # 归一化 RMSE（除以均值），使两个指标量纲一致
                    norm_rmse = rmse_val / (float(np.mean(y_val_orig)) + 1e-10)
                    score = 0.5 * cov_val + 0.5 * norm_rmse
                else:
                    score = rmse_val

                fold_scores.append(score)

            return float(np.mean(fold_scores))

        study = optuna.create_study(
            study_name=study_name,
            storage=f"sqlite:///{db_path}",
            direction="minimize",
            load_if_exists=True,
        )
        study.optimize(
            objective, n_trials=n_trials, timeout=timeout,
            show_progress_bar=True,
        )

        best = dict(study.best_params)
        n_units  = best.pop("n_units")
        n_layers = best.pop("n_layers")
        best["hidden_layers"] = [n_units] * n_layers
        best["batch_norm"] = True

        logger.info(f"Optuna 最优参数 ({obj_type}): {best}")
        logger.info(f"最优目标值: {study.best_value:.6f}")

        from src.utils import save_json
        save_json(best, self.paths_cfg.get("best_params_path", "logs/best_params.json"))
        return best

    # ── KFold 交叉验证训练 ──────────────────────

    def fit_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,          # 已经过 log1p 变换的目标
        preprocessor=None,      # 传入 preprocessor，供 Optuna 逆变换
    ) -> BPNet:
        """
        KFold 交叉验证训练，返回最优折的模型。
        y 应为 log1p 变换后的值，评估时逆变换回原始尺度。
        """
        self._X_cache = X
        self._y_cache = y
        self._preprocessor = preprocessor

        params = self._resolve_params()
        self.best_params_ = params
        logger.info(f"最终训练参数: {params}")

        kf = KFold(
            n_splits=self.cv_cfg.get("n_splits", 5),
            shuffle=self.cv_cfg.get("shuffle", True),
            random_state=self.cv_cfg.get("random_state", 42),
        )

        best_val_score = float("inf")
        best_model: Optional[BPNet] = None

        for fold, (tr_idx, val_idx) in enumerate(kf.split(X), 1):
            n_splits = self.cv_cfg.get("n_splits", 5)
            logger.info(f"── Fold {fold}/{n_splits} ──────────────────────────")
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]

            model = BPNet(
                input_dim=X.shape[1],
                hidden_layers=params.get("hidden_layers", [128, 64, 32]),
                activation=params.get("activation", "tanh"),
                dropout_rate=params.get("dropout_rate", 0.15),
                batch_norm=params.get("batch_norm", True),
            )
            tr_cfg = {
                **self.train_cfg,
                "learning_rate": params.get("learning_rate", 0.0005),
                "batch_size":    params.get("batch_size", 16),
                "optimizer":     params.get("optimizer", "adamw"),
                "weight_decay":  params.get("weight_decay", 5e-4),
            }

            model, history = train_model(
                model, X_tr, y_tr, X_val, y_val, tr_cfg, self.device
            )
            self.training_history_.append(history)

            # 在原始尺度计算验证指标
            model.eval()
            with torch.no_grad():
                preds_log = model(
                    torch.tensor(X_val, dtype=torch.float32).to(self.device)
                ).cpu().numpy()

            if preprocessor is not None:
                preds_orig = preprocessor.inverse_transform_y(preds_log)
                y_val_orig = preprocessor.inverse_transform_y(y_val)
            else:
                preds_orig = preds_log
                y_val_orig = y_val

            fold_rmse = float(np.sqrt(np.mean((preds_orig - y_val_orig) ** 2)))
            fold_cov  = _compute_cov(y_val_orig, preds_orig)
            self.cv_results_.append({
                "fold": fold, "rmse": fold_rmse, "cov": fold_cov
            })
            logger.info(
                f"Fold {fold} | RMSE={fold_rmse:.2f} kN | COV={fold_cov:.4f}"
            )

            # 以 COV（工程指标）为准选最优折
            if fold_cov < best_val_score:
                best_val_score = fold_cov
                best_model = model

        avg_rmse = np.mean([r["rmse"] for r in self.cv_results_])
        avg_cov  = np.mean([r["cov"]  for r in self.cv_results_])
        logger.info(f"KFold 平均 | RMSE={avg_rmse:.2f} | COV={avg_cov:.4f}")

        self.best_model_ = best_model
        return best_model  # type: ignore

    # ── 模型持久化 ──────────────────────────────

    def save_model(self, model: BPNet) -> None:
        from src.utils import ensure_dirs
        save_path = self.paths_cfg.get("model_save_path", "output/best_model.pth")
        ensure_dirs([os.path.dirname(save_path)])
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "input_dim": model.input_dim,
                "best_params": self.best_params_,
            },
            save_path,
        )
        logger.info(f"模型已保存: {save_path}")

    def load_model(self, input_dim: int) -> BPNet:
        save_path = self.paths_cfg.get("model_save_path", "output/best_model.pth")
        if not Path(save_path).exists():
            raise FileNotFoundError(f"模型文件未找到: {save_path}")
        checkpoint = torch.load(save_path, map_location=self.device)
        params = checkpoint.get("best_params", {})
        model = BPNet(
            input_dim=input_dim,
            hidden_layers=params.get("hidden_layers", [128, 64, 32]),
            activation=params.get("activation", "tanh"),
            dropout_rate=params.get("dropout_rate", 0.15),
            batch_norm=params.get("batch_norm", True),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()
        logger.info(f"模型已加载: {save_path}")
        return model
