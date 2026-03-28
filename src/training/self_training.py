"""
Self-Training Pipeline – 3-Round Pseudo-Labeling + Noisy Student
docs/08_성능개선_로드맵.md Stage D, docs/10_Phase2_구현명세서.md 기반 구현

Round 1: confidence ≥ 0.95 → ~12,000 pseudo-labels
Round 2: confidence ≥ 0.92 → ~3,000 추가
Round 3: confidence ≥ 0.90 → ~1,500 추가
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..config import (
    LABEL_COLUMNS, NUM_LABELS, ST_CFG, TRAIN_CFG,
    PHASE2_MODELS, MODELS_DIR,
)
from ..data.dataset import create_dataloader, get_tokenizer, HateSpeechDataset

logger = logging.getLogger(__name__)


class SelfTrainingPipeline:
    """비레이블 데이터 Pseudo-labeling 반복 학습 파이프라인"""

    def __init__(
        self,
        num_rounds: int = ST_CFG.num_rounds,
        confidence_thresholds: Optional[List[float]] = None,
    ):
        self.num_rounds = num_rounds
        self.confidence_thresholds = confidence_thresholds or list(ST_CFG.confidence_thresholds)

    # ── Pseudo-label 부여 ──────────────────────────────────
    @torch.no_grad()
    def pseudo_label_data(
        self,
        unlabeled_texts: List[str],
        models: List[nn.Module],
        tokenizer,
        confidence_threshold: float,
        device: torch.device,
    ) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """Teacher 앙상블로 비레이블 데이터에 pseudo-label 부여.

        Returns:
            (selected_texts, pseudo_labels (M, 9), confidence_scores (M,))
        """
        dummy_labels = np.zeros((len(unlabeled_texts), NUM_LABELS))
        dataset = HateSpeechDataset(unlabeled_texts, dummy_labels, tokenizer)
        loader = DataLoader(dataset, batch_size=64, shuffle=False)

        all_probs_per_model = []
        for model in models:
            model.eval()
            model_preds = []
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attn = batch["attention_mask"].to(device)
                logits = model(input_ids, attn)
                if isinstance(logits, dict):
                    logits = logits["logits"]
                probs = torch.sigmoid(logits).cpu().numpy()
                model_preds.append(probs)
            all_probs_per_model.append(np.concatenate(model_preds, axis=0))

        # 모델 앙상블 평균
        ensemble_probs = np.mean(all_probs_per_model, axis=0)  # (N, 9)
        confidence = ensemble_probs.max(axis=1)  # (N,)

        # 모델 합의도
        num_models = len(models)
        agreement = np.zeros(len(unlabeled_texts))
        if num_models > 1:
            binary_preds = [(p > 0.5).astype(int) for p in all_probs_per_model]
            stacked = np.stack(binary_preds, axis=0)  # (M_models, N, 9)
            agreement = stacked.mean(axis=0).max(axis=1)  # 최대 합의율

        # 고신뢰 샘플 필터링
        mask = confidence >= confidence_threshold
        if num_models > 1:
            mask &= agreement >= ST_CFG.min_model_agreement

        selected_idx = np.where(mask)[0]
        selected_texts = [unlabeled_texts[i] for i in selected_idx]
        pseudo_labels = (ensemble_probs[selected_idx] > 0.5).astype(np.float32)
        selected_confidence = confidence[selected_idx]

        logger.info(
            "Pseudo-labeling (threshold=%.2f): %d/%d selected (%.1f%%)",
            confidence_threshold, len(selected_idx),
            len(unlabeled_texts), 100 * len(selected_idx) / max(len(unlabeled_texts), 1),
        )
        return selected_texts, pseudo_labels, selected_confidence

    # ── 품질 필터링 ────────────────────────────────────────
    def quality_filter(
        self,
        pseudo_labels: np.ndarray,
        confidence_scores: np.ndarray,
        texts: List[str],
    ) -> np.ndarray:
        """Pseudo-label 품질 검증. Returns keep mask (bool array)."""
        n = len(texts)
        keep = np.ones(n, dtype=bool)

        # 엔트로피 제약
        eps = 1e-8
        p = np.clip(pseudo_labels, eps, 1 - eps)
        entropy = -np.sum(p * np.log2(p) + (1 - p) * np.log2(1 - p), axis=1)
        keep &= entropy <= ST_CFG.max_entropy

        # 양성 비율 제약
        positive_ratio = pseudo_labels.sum(axis=1) / NUM_LABELS
        keep &= positive_ratio <= ST_CFG.max_positive_ratio

        # 빈 레이블 제거 (all 0 also valid → clean sample)
        # 최소 길이 검증
        keep &= np.array([len(t.strip()) >= 5 for t in texts])

        filtered = n - keep.sum()
        logger.info("Quality filter: %d/%d removed, %d kept", filtered, n, keep.sum())
        return keep

    # ── 3-Round Self-Training 실행 ─────────────────────────
    def run_self_training(
        self,
        unlabeled_texts: List[str],
        labeled_df: pd.DataFrame,
        models: List[nn.Module],
        tokenizer,
        device: torch.device,
    ) -> Dict:
        """3-round self-training 파이프라인 실행"""
        remaining_texts = list(unlabeled_texts)
        accumulated_pseudo_texts: List[str] = []
        accumulated_pseudo_labels: List[np.ndarray] = []
        round_stats = []

        for round_idx in range(self.num_rounds):
            if not remaining_texts:
                logger.info("No remaining unlabeled data, stopping at round %d", round_idx)
                break

            threshold = self.confidence_thresholds[round_idx]
            logger.info("═══ Self-Training Round %d (threshold=%.2f) ═══", round_idx + 1, threshold)

            # Pseudo-labeling
            selected_texts, pseudo_labels, confidence = self.pseudo_label_data(
                remaining_texts, models, tokenizer, threshold, device,
            )

            if len(selected_texts) == 0:
                logger.warning("Round %d: no samples met threshold, stopping", round_idx + 1)
                break

            # 품질 필터링
            keep_mask = self.quality_filter(pseudo_labels, confidence, selected_texts)
            filtered_texts = [t for t, k in zip(selected_texts, keep_mask) if k]
            filtered_labels = pseudo_labels[keep_mask]

            # 누적
            accumulated_pseudo_texts.extend(filtered_texts)
            accumulated_pseudo_labels.append(filtered_labels)

            # 남은 데이터 업데이트
            selected_set = set(selected_texts)
            remaining_texts = [t for t in remaining_texts if t not in selected_set]

            # 확장 학습 데이터 구성
            if accumulated_pseudo_labels:
                all_pseudo_labels = np.concatenate(accumulated_pseudo_labels, axis=0)
                pseudo_df = pd.DataFrame({
                    "text": accumulated_pseudo_texts,
                    **{col: all_pseudo_labels[:, i] for i, col in enumerate(LABEL_COLUMNS)},
                })
                pseudo_df["source"] = "pseudo"
                mixed_df = pd.concat([labeled_df, pseudo_df], ignore_index=True)
            else:
                mixed_df = labeled_df

            round_stats.append({
                "round": round_idx + 1,
                "threshold": threshold,
                "pseudo_added": len(filtered_texts),
                "total_pseudo": len(accumulated_pseudo_texts),
                "total_train_size": len(mixed_df),
                "remaining_unlabeled": len(remaining_texts),
            })

            logger.info(
                "Round %d: +%d pseudo, total=%d, train_size=%d, remaining=%d",
                round_idx + 1, len(filtered_texts),
                len(accumulated_pseudo_texts), len(mixed_df), len(remaining_texts),
            )

        # 최종 결과
        if accumulated_pseudo_labels:
            all_pseudo_labels = np.concatenate(accumulated_pseudo_labels, axis=0)
            pseudo_df = pd.DataFrame({
                "text": accumulated_pseudo_texts,
                **{col: all_pseudo_labels[:, i] for i, col in enumerate(LABEL_COLUMNS)},
            })
            pseudo_df["source"] = "pseudo"
            final_df = pd.concat([labeled_df, pseudo_df], ignore_index=True)
        else:
            final_df = labeled_df

        return {
            "final_train_df": final_df,
            "round_stats": round_stats,
            "total_pseudo_added": len(accumulated_pseudo_texts),
        }
