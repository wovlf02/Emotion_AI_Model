"""
Error Correction Network (ECN) – 앙상블 잔차 패턴 학습 보정
docs/10_Phase2_구현명세서.md Stage E, docs/12_앙상블_심층설계.md 기반 구현

입력:  앙상블 확률 (9) + 키워드 힌트 (9) + 텍스트 특성 (5) = 23차원
타겟:  잔차 = true_label - ensemble_prob
모델:  클래스별 LightGBM (regression)
출력:  보정된 확률 = ensemble + α × correction
"""
import logging
import os
import pickle
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import NUM_LABELS, LABEL_COLUMNS, ECN_CFG, MODELS_DIR

logger = logging.getLogger(__name__)


def _chosung_ratio(text: str) -> float:
    """한글 초성(자음) 비율"""
    if not text:
        return 0.0
    chosung = set("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")
    cnt = sum(1 for c in text if c in chosung)
    return cnt / len(text)


def _special_char_ratio(text: str) -> float:
    """특수문자 비율"""
    if not text:
        return 0.0
    special = sum(1 for c in text if not c.isalnum() and not c.isspace())
    return special / len(text)


class ErrorCorrectionNetwork:
    """앙상블 예측 잔차를 학습하여 보정하는 LightGBM 기반 ECN"""

    def __init__(
        self,
        num_classes: int = NUM_LABELS,
        correction_strength: float = ECN_CFG.correction_strength,
    ):
        self.num_classes = num_classes
        self.alpha = correction_strength
        self.lgb_models: Dict[int, object] = {}
        self._trained = False

    # ── 잔차 계산 ──────────────────────────────────────────
    @staticmethod
    def compute_residuals(
        ensemble_probs: np.ndarray,
        true_labels: np.ndarray,
    ) -> np.ndarray:
        """residual = true_label - ensemble_prob  →  (N, 9) in [-1, 1]"""
        return true_labels.astype(np.float64) - ensemble_probs.astype(np.float64)

    # ── 텍스트 특성 추출 ───────────────────────────────────
    @staticmethod
    def compute_text_features(texts: List[str]) -> np.ndarray:
        """텍스트 통계 특성 (N, 5):
        text_length, word_count, chosung_ratio, special_char_ratio, has_url
        """
        features = []
        for t in texts:
            features.append([
                len(t),
                len(t.split()),
                _chosung_ratio(t),
                _special_char_ratio(t),
                1.0 if re.search(r"https?://", t) else 0.0,
            ])
        return np.array(features, dtype=np.float32)

    # ── 특성 조합 ──────────────────────────────────────────
    @staticmethod
    def build_features(
        ensemble_probs: np.ndarray,
        keyword_hints: np.ndarray,
        text_features: np.ndarray,
    ) -> np.ndarray:
        """ECN 입력 특성 (N, 23) = ensemble(9) + keyword(9) + text(5)"""
        return np.hstack([ensemble_probs, keyword_hints, text_features])

    # ── 학습 ───────────────────────────────────────────────
    def train(
        self,
        ensemble_probs: np.ndarray,
        true_labels: np.ndarray,
        texts: List[str],
        keyword_hints: np.ndarray,
        val_ensemble_probs: Optional[np.ndarray] = None,
        val_true_labels: Optional[np.ndarray] = None,
        val_texts: Optional[List[str]] = None,
        val_keyword_hints: Optional[np.ndarray] = None,
    ) -> Dict:
        """클래스별 LightGBM 잔차 모델 학습"""
        try:
            import lightgbm as lgb
        except ImportError:
            logger.error("lightgbm not installed. Install with: pip install lightgbm")
            raise

        residuals = self.compute_residuals(ensemble_probs, true_labels)
        text_feat = self.compute_text_features(texts)
        X_train = self.build_features(ensemble_probs, keyword_hints, text_feat)
        y_train = residuals

        has_val = (val_ensemble_probs is not None and val_true_labels is not None
                   and val_texts is not None and val_keyword_hints is not None)
        if has_val:
            val_residuals = self.compute_residuals(val_ensemble_probs, val_true_labels)
            val_text_feat = self.compute_text_features(val_texts)
            X_val = self.build_features(val_ensemble_probs, val_keyword_hints, val_text_feat)
            y_val = val_residuals

        lgb_params = {
            "objective": "regression",
            "metric": "mae",
            "num_leaves": ECN_CFG.lgb_num_leaves,
            "learning_rate": ECN_CFG.lgb_learning_rate,
            "feature_fraction": ECN_CFG.lgb_feature_fraction,
            "bagging_fraction": ECN_CFG.lgb_bagging_fraction,
            "bagging_freq": 5,
            "verbose": -1,
            "seed": 42,
        }

        val_scores = {"mae": [], "r2": []}

        for c in range(self.num_classes):
            train_data = lgb.Dataset(X_train, label=y_train[:, c])
            callbacks = [lgb.early_stopping(ECN_CFG.lgb_early_stopping, verbose=False)]

            if has_val:
                val_data = lgb.Dataset(X_val, label=y_val[:, c], reference=train_data)
                model = lgb.train(
                    lgb_params, train_data,
                    num_boost_round=ECN_CFG.lgb_n_estimators,
                    valid_sets=[val_data],
                    callbacks=callbacks,
                )
                # Validation 메트릭
                pred_val = model.predict(X_val)
                mae = np.mean(np.abs(pred_val - y_val[:, c]))
                ss_res = np.sum((y_val[:, c] - pred_val) ** 2)
                ss_tot = np.sum((y_val[:, c] - y_val[:, c].mean()) ** 2)
                r2 = 1 - ss_res / max(ss_tot, 1e-8)
                val_scores["mae"].append(mae)
                val_scores["r2"].append(r2)
            else:
                model = lgb.train(
                    lgb_params, train_data,
                    num_boost_round=ECN_CFG.lgb_n_estimators,
                )

            self.lgb_models[c] = model
            logger.info("ECN class '%s' trained (trees=%d)", LABEL_COLUMNS[c], model.num_trees())

        self._trained = True
        logger.info("ECN training complete. Mean MAE=%.4f, Mean R²=%.4f",
                     np.mean(val_scores["mae"]) if val_scores["mae"] else 0,
                     np.mean(val_scores["r2"]) if val_scores["r2"] else 0)

        return {"val_mae": val_scores["mae"], "val_r2": val_scores["r2"]}

    # ── 보정값 예측 ────────────────────────────────────────
    def predict(
        self,
        ensemble_probs: np.ndarray,
        keyword_hints: np.ndarray,
        text_features: np.ndarray,
    ) -> np.ndarray:
        """보정값 예측 → (N, 9), 클리핑 적용"""
        if not self._trained:
            raise RuntimeError("ECN not trained yet")

        X = self.build_features(ensemble_probs, keyword_hints, text_features)
        corrections = np.zeros((len(X), self.num_classes), dtype=np.float64)

        for c in range(self.num_classes):
            corrections[:, c] = self.lgb_models[c].predict(X)

        # 보정 크기 제한
        corrections = np.clip(
            corrections,
            -ECN_CFG.max_correction_magnitude,
            ECN_CFG.max_correction_magnitude,
        )
        return corrections

    # ── 보정 적용 ──────────────────────────────────────────
    def apply_correction(
        self,
        ensemble_probs: np.ndarray,
        corrections: np.ndarray,
        keyword_hints: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """최종 보정: P_corrected = P_ensemble + α × correction

        안전장치:
        - |correction| < min_correction_abs → 무시
        - keyword hint와 방향이 반대 → α 감소
        """
        alpha_matrix = np.full_like(corrections, self.alpha)

        # 소규모 보정 무시
        small_mask = np.abs(corrections) < ECN_CFG.min_correction_abs
        alpha_matrix[small_mask] = 0.0

        # 키워드 힌트 방향 불일치 → alpha 감소
        if keyword_hints is not None:
            hint_direction = np.sign(keyword_hints)
            corr_direction = np.sign(corrections)
            opposite = (hint_direction != 0) & (hint_direction != corr_direction)
            alpha_matrix[opposite] = ECN_CFG.alpha_reduced

        corrected = ensemble_probs + alpha_matrix * corrections
        corrected = np.clip(corrected, 0.0, 1.0)
        return corrected

    # ── 저장/로드 ──────────────────────────────────────────
    def save(self, path: Optional[str] = None):
        if path is None:
            path = os.path.join(MODELS_DIR, "ecn_model.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "lgb_models": self.lgb_models,
                "alpha": self.alpha,
                "num_classes": self.num_classes,
            }, f)
        logger.info("ECN saved to %s", path)

    def load(self, path: Optional[str] = None):
        if path is None:
            path = os.path.join(MODELS_DIR, "ecn_model.pkl")
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.lgb_models = data["lgb_models"]
        self.alpha = data["alpha"]
        self.num_classes = data["num_classes"]
        self._trained = True
        logger.info("ECN loaded from %s (%d class models)", path, len(self.lgb_models))
