"""
Curriculum Learning – 난이도별 3-Stage 학습 스케줄링
docs/08_성능개선_로드맵.md, docs/10_Phase2_구현명세서.md 기반 구현

Stage 1 (Easy):   Epoch 1~15  – 고신뢰 쉬운 샘플 중심
Stage 2 (Medium): Epoch 16~35 – 중간 난이도 + 외부 데이터
Stage 3 (Hard):   Epoch 36~60 – 어려운 샘플 + 오분류 재주입
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..config import (
    LABEL_COLUMNS, NUM_LABELS, CURRICULUM_CFG, TRAIN_CFG, SOURCE_WEIGHTS,
)
from ..data.dataset import HateSpeechDataset, create_dataloader, get_tokenizer

logger = logging.getLogger(__name__)


class CurriculumScheduler:
    """에포크 기반 Curriculum Learning 스케줄러"""

    def __init__(
        self,
        total_epochs: int = CURRICULUM_CFG.total_epochs,
        easy_end: int = CURRICULUM_CFG.easy_end_epoch,
        medium_end: int = CURRICULUM_CFG.medium_end_epoch,
    ):
        self.total_epochs = total_epochs
        self.easy_end = easy_end
        self.medium_end = medium_end
        self._difficulty_scores: Optional[np.ndarray] = None

    # ── 난이도 단계 판별 ───────────────────────────────────
    def get_difficulty_stage(self, epoch: int) -> str:
        if epoch <= self.easy_end:
            return "easy"
        if epoch <= self.medium_end:
            return "medium"
        return "hard"

    # ── 샘플 난이도 계산 (초기 calibration) ─────────────────
    @torch.no_grad()
    def compute_sample_difficulty(
        self,
        dataloader: DataLoader,
        model: nn.Module,
        device: torch.device,
        num_epochs: int = CURRICULUM_CFG.difficulty_calibration_epochs,
    ) -> np.ndarray:
        """초기 num_epochs 에포크 동안의 평균 손실로 샘플 난이도 산출.

        Returns:
            (N,) 0~1 범위 난이도 점수 (높을수록 어려움)
        """
        model.eval()
        criterion = nn.BCEWithLogitsLoss(reduction="none")
        all_losses: list[np.ndarray] = []

        for _ in range(num_epochs):
            batch_losses = []
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                logits = model(input_ids, attention_mask)
                if isinstance(logits, dict):
                    logits = logits["logits"]
                loss_per_sample = criterion(logits, labels).mean(dim=1)
                batch_losses.append(loss_per_sample.cpu().numpy())
            all_losses.append(np.concatenate(batch_losses))

        avg_loss = np.mean(all_losses, axis=0)
        min_l, max_l = avg_loss.min(), avg_loss.max()
        if max_l - min_l < 1e-8:
            return np.zeros_like(avg_loss)
        scores = (avg_loss - min_l) / (max_l - min_l)
        self._difficulty_scores = scores
        logger.info(
            "Difficulty scores computed: easy(<%.2f)=%d, medium=%d, hard(>%.2f)=%d",
            CURRICULUM_CFG.easy_threshold,
            (scores <= CURRICULUM_CFG.easy_threshold).sum(),
            ((scores > CURRICULUM_CFG.easy_threshold) & (scores <= CURRICULUM_CFG.hard_threshold)).sum(),
            CURRICULUM_CFG.hard_threshold,
            (scores > CURRICULUM_CFG.hard_threshold).sum(),
        )
        return scores

    # ── Curriculum DataLoader 생성 ─────────────────────────
    def create_curriculum_dataloader(
        self,
        train_df: pd.DataFrame,
        difficulty_scores: np.ndarray,
        current_epoch: int,
        tokenizer,
        batch_size: int = 32,
        misclassified_indices: Optional[np.ndarray] = None,
    ) -> DataLoader:
        """현재 에포크의 난이도 단계에 맞는 DataLoader 생성"""
        stage = self.get_difficulty_stage(current_epoch)
        n = len(train_df)

        if stage == "easy":
            mask = difficulty_scores <= CURRICULUM_CFG.easy_threshold
            if "source" in train_df.columns:
                source_mask = train_df["source"].values == "unsmile"
                mask = mask | source_mask
        elif stage == "medium":
            mask = difficulty_scores <= CURRICULUM_CFG.hard_threshold
        else:
            mask = np.ones(n, dtype=bool)
            if misclassified_indices is not None and len(misclassified_indices) > 0:
                hard_mask = difficulty_scores > CURRICULUM_CFG.hard_threshold
                misc_mask = np.zeros(n, dtype=bool)
                valid_idx = misclassified_indices[misclassified_indices < n]
                misc_mask[valid_idx] = True
                mask = hard_mask | misc_mask

        subset_df = train_df.iloc[mask].reset_index(drop=True)
        if len(subset_df) == 0:
            logger.warning("Curriculum stage '%s' produced empty subset, using full data", stage)
            subset_df = train_df

        logger.info(
            "Curriculum [%s] epoch %d: %d/%d samples (%.1f%%)",
            stage, current_epoch, len(subset_df), n, 100.0 * len(subset_df) / n,
        )
        return create_dataloader(subset_df, tokenizer, batch_size=batch_size, shuffle=True)
