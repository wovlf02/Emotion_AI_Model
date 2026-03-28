"""
추론 파이프라인 – TTA + Temperature Scaling + 최종 예측
docs/10_Phase2_구현명세서.md Stage F 기반 구현
"""
import re
import logging
from typing import List, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer

from ..config import LABEL_COLUMNS, NUM_LABELS, TRAIN_CFG, PHASE2_MODELS
from ..models.model import MultiLabelClassifier
from ..data.preprocessing import TextNormalizer
from .rule_system import PostProcessingCorrector
from ..utils import load_checkpoint, get_device

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────
#  TTA (Test-Time Augmentation)
# ──────────────────────────────────────────────────────────
class TextTTA:
    """5종 텍스트 변환 기반 TTA"""

    AUGMENTATIONS = [
        lambda t: t,                                         # Original
        lambda t: t.rstrip("?!.~"),                          # 구두점 제거
        lambda t: re.sub(r"\s+", " ", t),                    # 공백 정규화
        lambda t: t + " .",                                  # 마침표 추가
        lambda t: TextTTA._swap_last_words(t),               # 마지막 2단어 교환
    ]

    @staticmethod
    def _swap_last_words(text: str) -> str:
        words = text.split()
        if len(words) >= 3:
            words[-1], words[-2] = words[-2], words[-1]
        return " ".join(words)

    def augment(self, text: str) -> List[str]:
        """텍스트 → 5개 변환 버전"""
        return [aug(text) for aug in self.AUGMENTATIONS]


# ──────────────────────────────────────────────────────────
#  Temperature Scaling (클래스별)
# ──────────────────────────────────────────────────────────
class ClasswiseTemperatureScaling:
    """클래스별 독립 Temperature 파라미터"""

    def __init__(self, num_classes: int = NUM_LABELS):
        self.temperatures = np.ones(num_classes)

    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        lr: float = 0.01,
        max_iter: int = 100,
    ):
        """Validation set으로 Temperature 학습"""
        for c in range(len(self.temperatures)):
            best_t, best_loss = 1.0, float("inf")
            for t_cand in np.arange(0.5, 2.0, 0.01):
                scaled = logits[:, c] / t_cand
                probs = 1.0 / (1.0 + np.exp(-scaled))
                eps = 1e-10
                loss = -np.mean(
                    labels[:, c] * np.log(probs + eps)
                    + (1 - labels[:, c]) * np.log(1 - probs + eps)
                )
                if loss < best_loss:
                    best_loss = loss
                    best_t = t_cand
            self.temperatures[c] = best_t

        logger.info(f"Temperature Scaling: {dict(zip(LABEL_COLUMNS, self.temperatures))}")

    def scale(self, logits: np.ndarray) -> np.ndarray:
        """Temperature 적용"""
        return logits / self.temperatures


# ──────────────────────────────────────────────────────────
#  추론 엔진
# ──────────────────────────────────────────────────────────
class InferenceEngine:
    """전체 추론 파이프라인"""

    def __init__(
        self,
        checkpoint_paths: List[str],
        thresholds: Optional[np.ndarray] = None,
        use_tta: bool = True,
        use_rule_system: bool = True,
        device: torch.device = None,
    ):
        self.device = device or get_device()
        self.thresholds = thresholds if thresholds is not None else np.full(NUM_LABELS, 0.5)
        self.use_tta = use_tta
        self.normalizer = TextNormalizer()
        self.tta = TextTTA() if use_tta else None
        self.corrector = PostProcessingCorrector() if use_rule_system else None
        self.models = []
        self.tokenizers = []

        self._load_models(checkpoint_paths)

    def _load_models(self, paths: List[str]):
        """체크포인트에서 모델 로드"""
        for path in paths:
            ckpt = load_checkpoint(path, self.device)
            model_name = ckpt["model_name"]
            model = MultiLabelClassifier(model_name).to(self.device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            self.models.append(model)
            self.tokenizers.append(tokenizer)
        logger.info(f"Loaded {len(self.models)} models for inference")

    @torch.no_grad()
    def predict_single(self, text: str) -> Dict:
        """
        단일 텍스트 추론

        Returns:
            {
                'probabilities': (9,) float,
                'labels': (9,) int,
                'categories': [str, ...],
            }
        """
        # 전처리
        text = self.normalizer.normalize(text)

        # TTA 변환
        if self.tta:
            texts = self.tta.augment(text)
        else:
            texts = [text]

        # 모든 모델 × 모든 TTA 변환 예측
        all_probs = []
        for model, tokenizer in zip(self.models, self.tokenizers):
            for t in texts:
                encoding = tokenizer(
                    t, max_length=TRAIN_CFG.max_length,
                    padding="max_length", truncation=True,
                    return_tensors="pt",
                )
                input_ids = encoding["input_ids"].to(self.device)
                attention_mask = encoding["attention_mask"].to(self.device)
                token_type_ids = encoding.get("token_type_ids")
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(self.device)

                logits = model(input_ids, attention_mask, token_type_ids)
                probs = torch.sigmoid(logits).cpu().numpy()[0]
                all_probs.append(probs)

        # 평균 앙상블
        mean_probs = np.mean(all_probs, axis=0)

        # 룰 시스템 보정
        if self.corrector:
            mean_probs = self.corrector.correct([text], mean_probs.reshape(1, -1))[0]

        # 임계값 적용
        binary = (mean_probs > self.thresholds).astype(int)
        categories = [LABEL_COLUMNS[i] for i in range(NUM_LABELS) if binary[i] == 1]

        return {
            "probabilities": mean_probs.tolist(),
            "labels": binary.tolist(),
            "categories": categories,
        }

    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """배치 추론"""
        return [self.predict_single(t) for t in texts]
