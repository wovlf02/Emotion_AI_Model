"""
Hard Negative Mining – False Negative 탐지 및 오버샘플링 + Specialist 모델
docs/08_성능개선_로드맵.md Stage C 기반 구현

Round 1: FN 샘플 3배 오버샘플링 → 재학습
Round 2: Persistent FN → 취약 클래스 Specialist 이진 분류 모델
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..config import (
    LABEL_COLUMNS, NUM_LABELS, HNM_CFG, TRAIN_CFG, PHASE2_MODELS,
)
from ..data.dataset import create_dataloader, get_tokenizer
from ..models.model import MultiLabelClassifier

logger = logging.getLogger(__name__)


class HardNegativeMiner:
    """False Negative 샘플 탐지 및 오버샘플링"""

    def __init__(
        self,
        num_classes: int = NUM_LABELS,
        fn_oversample_ratio: float = HNM_CFG.fn_oversample_ratio,
        num_rounds: int = HNM_CFG.num_rounds,
    ):
        self.num_classes = num_classes
        self.fn_oversample_ratio = fn_oversample_ratio
        self.num_rounds = num_rounds

    # ── False Negative 식별 ────────────────────────────────
    def identify_hard_negatives(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        texts: List[str],
        threshold: float = HNM_CFG.fn_detection_threshold,
    ) -> Dict:
        """FN 샘플 탐지: 정답에 양성 레이블이 있는데 모델이 전부 음성으로 예측한 경우"""
        n = len(y_true)
        y_binary = (y_pred > threshold).astype(int)

        fn_indices = []
        fn_confidence = []

        for i in range(n):
            has_positive = y_true[i].sum() > 0
            predicted_all_negative = y_binary[i].sum() == 0
            if has_positive and predicted_all_negative:
                fn_indices.append(i)
                fn_confidence.append(1.0 - y_pred[i].max())

        fn_indices = np.array(fn_indices)
        fn_confidence = np.array(fn_confidence) if fn_indices.size > 0 else np.array([])

        class_wise_fns: Dict[str, int] = {}
        if fn_indices.size > 0:
            fn_labels = y_true[fn_indices]
            for c, col in enumerate(LABEL_COLUMNS):
                class_wise_fns[col] = int(fn_labels[:, c].sum())
        else:
            fn_labels = np.empty((0, self.num_classes))

        fn_rate = len(fn_indices) / n if n > 0 else 0.0
        logger.info("Hard Negatives: %d/%d (%.2f%%)", len(fn_indices), n, fn_rate * 100)
        for cls, cnt in sorted(class_wise_fns.items(), key=lambda x: -x[1])[:5]:
            logger.info("  %s: %d FN samples", cls, cnt)

        return {
            "fn_indices": fn_indices.tolist(),
            "fn_texts": [texts[i] for i in fn_indices] if fn_indices.size > 0 else [],
            "fn_labels": fn_labels,
            "fn_confidence": fn_confidence,
            "class_wise_fns": class_wise_fns,
            "total_fn": len(fn_indices),
            "fn_rate": fn_rate,
        }

    # ── FN 오버샘플링 DataLoader ───────────────────────────
    def create_hard_negative_dataloader(
        self,
        train_df: pd.DataFrame,
        fn_indices: List[int],
        tokenizer,
        batch_size: int = 32,
        oversample_ratio: float = None,
    ) -> DataLoader:
        """원본 + FN 오버샘플링 혼합 DataLoader"""
        if oversample_ratio is None:
            oversample_ratio = self.fn_oversample_ratio

        if not fn_indices:
            logger.warning("No FN indices provided, returning original dataloader")
            return create_dataloader(train_df, tokenizer, batch_size=batch_size, shuffle=True)

        fn_df = train_df.iloc[fn_indices]
        oversample_count = int(len(fn_df) * oversample_ratio)
        fn_oversampled = fn_df.sample(n=oversample_count, replace=True, random_state=42)

        mixed_df = pd.concat([train_df, fn_oversampled], ignore_index=True)
        mixed_df = mixed_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

        logger.info(
            "HNM DataLoader: original=%d + FN_oversampled=%d = %d",
            len(train_df), oversample_count, len(mixed_df),
        )
        return create_dataloader(mixed_df, tokenizer, batch_size=batch_size, shuffle=True)

    # ── Persistent FN 탐지 (Round 2) ──────────────────────
    @torch.no_grad()
    def identify_persistent_hard_negatives(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        y_true: np.ndarray,
        device: torch.device,
        threshold: float = HNM_CFG.persistent_fn_threshold,
    ) -> Dict:
        """Round 1 재학습 후에도 여전히 오분류되는 샘플"""
        model.eval()
        all_preds = []
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids, attention_mask)
            if isinstance(logits, dict):
                logits = logits["logits"]
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)
        y_pred = np.concatenate(all_preds, axis=0)

        persistent_indices = []
        for i in range(len(y_true)):
            if y_true[i].sum() > 0:
                positive_classes = np.where(y_true[i] == 1)[0]
                max_positive_prob = y_pred[i, positive_classes].max()
                if max_positive_prob < threshold:
                    persistent_indices.append(i)

        logger.info("Persistent FN: %d samples (threshold=%.2f)", len(persistent_indices), threshold)
        return {
            "persistent_indices": persistent_indices,
            "total_persistent": len(persistent_indices),
            "y_pred": y_pred,
        }


class SpecialistModel:
    """취약 클래스 전문가 이진 분류 모델"""

    def __init__(
        self,
        target_class: str,
        model_name: str = "beomi/KcELECTRA-base",
        epochs: int = HNM_CFG.specialist_epochs,
    ):
        self.target_class = target_class
        self.model_name = model_name
        self.epochs = epochs
        self.target_idx = LABEL_COLUMNS.index(target_class)
        self.model: Optional[MultiLabelClassifier] = None

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        hard_negative_indices: List[int],
        device: torch.device,
    ) -> str:
        """Specialist 모델 학습 (이진 분류: target_class vs others)

        Returns:
            checkpoint_path
        """
        from ..training.trainer import train_one_epoch, evaluate
        from ..models.asymmetric_loss import AsymmetricLoss
        from ..utils import EarlyStopping, save_checkpoint

        binary_label = self.target_class
        logger.info("Training specialist for '%s' with %d hard negatives",
                     binary_label, len(hard_negative_indices))

        # 이진 레이블 데이터 구성
        def make_binary_df(df):
            bdf = df.copy()
            bdf["__target__"] = bdf[binary_label].values
            return bdf

        train_binary = make_binary_df(train_df)
        val_binary = make_binary_df(val_df)

        # HN 샘플 오버샘플링
        if hard_negative_indices:
            hn_df = train_binary.iloc[hard_negative_indices]
            hn_oversampled = hn_df.sample(
                n=min(len(hn_df) * 3, len(train_binary) // 2),
                replace=True, random_state=42,
            )
            train_binary = pd.concat([train_binary, hn_oversampled], ignore_index=True)

        self.model = MultiLabelClassifier(self.model_name, num_labels=1)
        self.model.to(device)

        tokenizer = get_tokenizer(self.model_name)
        train_loader = create_dataloader(train_binary, tokenizer, batch_size=32, shuffle=True)
        val_loader = create_dataloader(val_binary, tokenizer, batch_size=64, shuffle=False)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5, weight_decay=0.01)
        criterion = nn.BCEWithLogitsLoss()
        early_stop = EarlyStopping(patience=10)

        best_path = ""
        for epoch in range(self.epochs):
            self.model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(device)
                attn = batch["attention_mask"].to(device)
                labels = batch["labels"][:, self.target_idx:self.target_idx + 1].to(device)
                logits = self.model(input_ids, attn)
                if isinstance(logits, dict):
                    logits = logits["logits"]
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

            if early_stop(0):
                break

        import os
        from ..config import MODELS_DIR
        best_path = os.path.join(MODELS_DIR, f"specialist_{self.target_class}.pt")
        save_checkpoint({"model_state_dict": self.model.state_dict()}, best_path)
        logger.info("Specialist '%s' saved to %s", self.target_class, best_path)
        return best_path

    @torch.no_grad()
    def predict(self, texts: List[str], tokenizer, device: torch.device) -> np.ndarray:
        """이진 확률 예측 (N, 1)"""
        if self.model is None:
            raise RuntimeError("Specialist model not trained yet")
        self.model.eval()
        from ..data.dataset import HateSpeechDataset
        dummy_labels = np.zeros((len(texts), 1))
        dataset = HateSpeechDataset(texts, dummy_labels, tokenizer)
        loader = DataLoader(dataset, batch_size=64, shuffle=False)
        preds = []
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            logits = self.model(input_ids, attn)
            if isinstance(logits, dict):
                logits = logits["logits"]
            probs = torch.sigmoid(logits).cpu().numpy()
            preds.append(probs)
        return np.concatenate(preds, axis=0)
