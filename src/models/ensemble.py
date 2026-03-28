"""
앙상블 – Meta-Learner (LightGBM + MLP + Ridge) + Final Blending
docs/12_앙상블_심층설계.md 기반 구현
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.linear_model import RidgeClassifier

from ..config import LABEL_COLUMNS, NUM_LABELS, ENSEMBLE_CFG

logger = logging.getLogger(__name__)

# LightGBM 선택적 import
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    logger.warning("LightGBM not installed – meta-learner will use MLP + Ridge only")


# ──────────────────────────────────────────────────────────
#  Meta-Feature 생성 (294차원)
# ──────────────────────────────────────────────────────────
def create_meta_features(
    base_predictions: np.ndarray,
    texts: List[str],
    keyword_scores: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Meta-Learner 입력 특성 생성

    Args:
        base_predictions: (N, 25, 9) 25개 서브모델 × 9클래스 확률
        texts: N개 텍스트
        keyword_scores: (N, 9) 키워드 힌트 점수 (없으면 0)

    Returns:
        (N, ~294) 특성 행렬
    """
    N, n_models, n_classes = base_predictions.shape
    features = []

    # 1. 기본 확률 (25×9 = 225차원)
    flat_probs = base_predictions.reshape(N, -1)
    features.append(flat_probs)

    # 2. 클래스별 통계 (9×5 = 45차원)
    stats = []
    for c in range(n_classes):
        probs = base_predictions[:, :, c]
        stats.append(probs.mean(axis=1, keepdims=True))
        stats.append(probs.std(axis=1, keepdims=True))
        stats.append(probs.max(axis=1, keepdims=True))
        stats.append(probs.min(axis=1, keepdims=True))
        stats.append(np.median(probs, axis=1, keepdims=True))
    features.append(np.hstack(stats))

    # 3. 모델 간 합의도 (9차원)
    agreement = []
    for c in range(n_classes):
        binary = (base_predictions[:, :, c] > 0.5).astype(float)
        agreement.append(binary.mean(axis=1, keepdims=True))
    features.append(np.hstack(agreement))

    # 4. 예측 엔트로피 (1차원)
    mean_probs = base_predictions.mean(axis=1)
    eps = 1e-10
    entropy = -np.sum(
        mean_probs * np.log(mean_probs + eps)
        + (1 - mean_probs) * np.log(1 - mean_probs + eps),
        axis=1,
        keepdims=True,
    )
    features.append(entropy)

    # 5. 텍스트 통계 (5차원)
    text_features = np.zeros((N, 5))
    for i, t in enumerate(texts):
        text_features[i, 0] = len(t)
        text_features[i, 1] = len(t.split())
        text_features[i, 2] = _chosung_ratio(t)
        text_features[i, 3] = _special_char_ratio(t)
        text_features[i, 4] = 1.0 if "http" in t else 0.0
    features.append(text_features)

    # 6. 키워드 힌트 (9차원)
    if keyword_scores is not None:
        features.append(keyword_scores)
    else:
        features.append(np.zeros((N, n_classes)))

    return np.hstack(features)


def _chosung_ratio(text: str) -> float:
    if not text:
        return 0.0
    chosung = sum(1 for c in text if "\u3131" <= c <= "\u314e")
    return chosung / len(text)


def _special_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    special = sum(1 for c in text if not c.isalnum() and c != " ")
    return special / len(text)


# ──────────────────────────────────────────────────────────
#  MLP Meta-Learner
# ──────────────────────────────────────────────────────────
class MLPMetaLearner(nn.Module):
    """2-Layer MLP Meta-Learner"""

    def __init__(
        self,
        input_dim: int = 294,
        hidden_dims: List[int] = None,
        num_classes: int = NUM_LABELS,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = ENSEMBLE_CFG.mlp_hidden_dims

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(ENSEMBLE_CFG.mlp_dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[1], num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


# ──────────────────────────────────────────────────────────
#  Meta-Learner 학습
# ──────────────────────────────────────────────────────────
class StackingMetaLearner:
    """Level 1 Meta-Learner: LightGBM + MLP + Ridge"""

    def __init__(self, input_dim: int = 294, device: torch.device = None):
        self.input_dim = input_dim
        self.device = device or torch.device("cpu")
        self.lgbm_models = {}
        self.mlp_model = None
        self.ridge_models = {}

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ):
        """전체 Meta-Learner 학습"""
        logger.info(f"Training meta-learner: X={X_train.shape}, y={y_train.shape}")

        # 1. LightGBM (per-class)
        if HAS_LIGHTGBM:
            self._fit_lightgbm(X_train, y_train, X_val, y_val)

        # 2. MLP
        self._fit_mlp(X_train, y_train, X_val, y_val)

        # 3. Ridge (per-class)
        self._fit_ridge(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Meta-Learner 앙상블 예측"""
        preds = []
        weights = []

        if HAS_LIGHTGBM and self.lgbm_models:
            preds.append(self._predict_lightgbm(X))
            weights.append(ENSEMBLE_CFG.meta_lgbm_weight)

        if self.mlp_model is not None:
            preds.append(self._predict_mlp(X))
            weights.append(ENSEMBLE_CFG.meta_mlp_weight)

        if self.ridge_models:
            preds.append(self._predict_ridge(X))
            weights.append(ENSEMBLE_CFG.meta_ridge_weight)

        # 가중 평균
        w = np.array(weights)
        w = w / w.sum()
        result = sum(p * ww for p, ww in zip(preds, w))
        return result

    # ── LightGBM ───────────────────────────────────────
    def _fit_lightgbm(self, X_train, y_train, X_val, y_val):
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": ENSEMBLE_CFG.lgbm_num_leaves,
            "learning_rate": ENSEMBLE_CFG.lgbm_learning_rate,
            "feature_fraction": ENSEMBLE_CFG.lgbm_feature_fraction,
            "bagging_fraction": ENSEMBLE_CFG.lgbm_bagging_fraction,
            "bagging_freq": 5,
            "verbose": -1,
            "n_estimators": ENSEMBLE_CFG.lgbm_n_estimators,
        }

        for c in range(NUM_LABELS):
            train_data = lgb.Dataset(X_train, y_train[:, c])
            val_data = lgb.Dataset(X_val, y_val[:, c])
            model = lgb.train(
                params, train_data,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(ENSEMBLE_CFG.lgbm_early_stopping,
                                              verbose=False)],
            )
            self.lgbm_models[c] = model
        logger.info("LightGBM meta-learner trained")

    def _predict_lightgbm(self, X: np.ndarray) -> np.ndarray:
        preds = np.zeros((len(X), NUM_LABELS))
        for c, model in self.lgbm_models.items():
            preds[:, c] = model.predict(X)
        return preds

    # ── MLP ────────────────────────────────────────────
    def _fit_mlp(self, X_train, y_train, X_val, y_val):
        self.mlp_model = MLPMetaLearner(input_dim=self.input_dim).to(self.device)
        optimizer = Adam(self.mlp_model.parameters(), lr=ENSEMBLE_CFG.mlp_lr)
        criterion = nn.BCEWithLogitsLoss()

        X_t = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        X_v = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        y_v = torch.tensor(y_val, dtype=torch.float32).to(self.device)

        best_loss = float("inf")
        patience, counter = 20, 0

        for epoch in range(ENSEMBLE_CFG.mlp_epochs):
            self.mlp_model.train()
            logits = self.mlp_model(X_t)
            loss = criterion(logits, y_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.mlp_model.eval()
            with torch.no_grad():
                val_logits = self.mlp_model(X_v)
                val_loss = criterion(val_logits, y_v).item()

            if val_loss < best_loss:
                best_loss = val_loss
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    break

        logger.info(f"MLP meta-learner trained ({epoch + 1} epochs)")

    def _predict_mlp(self, X: np.ndarray) -> np.ndarray:
        self.mlp_model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            logits = self.mlp_model(X_t)
            return torch.sigmoid(logits).cpu().numpy()

    # ── Ridge ──────────────────────────────────────────
    def _fit_ridge(self, X_train, y_train):
        for c in range(NUM_LABELS):
            clf = RidgeClassifier(alpha=1.0)
            clf.fit(X_train, y_train[:, c])
            self.ridge_models[c] = clf
        logger.info("Ridge meta-learner trained")

    def _predict_ridge(self, X: np.ndarray) -> np.ndarray:
        preds = np.zeros((len(X), NUM_LABELS))
        for c, clf in self.ridge_models.items():
            decision = clf.decision_function(X)
            # decision → 확률 근사 (sigmoid)
            preds[:, c] = 1.0 / (1.0 + np.exp(-decision))
        return preds


# ──────────────────────────────────────────────────────────
#  Final Blending (Level 2)
# ──────────────────────────────────────────────────────────
def final_blend(
    p_meta: np.ndarray,
    p_best_single: np.ndarray,
    p_keyword: np.ndarray,
    p_ecn: Optional[np.ndarray] = None,
    weights: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    Level 2 최종 블렌딩

    P_final = α×P_meta + β×P_best + γ×P_keyword + δ×P_ecn
    """
    if weights is None:
        weights = {
            "meta": ENSEMBLE_CFG.blend_meta,
            "best_single": ENSEMBLE_CFG.blend_best_single,
            "keyword": ENSEMBLE_CFG.blend_keyword,
            "ecn": ENSEMBLE_CFG.blend_ecn,
        }

    result = (
        weights["meta"] * p_meta
        + weights["best_single"] * p_best_single
        + weights["keyword"] * p_keyword
    )

    if p_ecn is not None:
        result += weights["ecn"] * p_ecn
    else:
        # ECN 없으면 meta에 재분배
        result += weights["ecn"] * p_meta

    return np.clip(result, 0.0, 1.0)
