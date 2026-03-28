"""
PyTorch Dataset + DataLoader 생성
"""
import logging
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
from transformers import AutoTokenizer

from ..config import LABEL_COLUMNS, TRAIN_CFG

logger = logging.getLogger(__name__)


class HateSpeechDataset(Dataset):
    """한국어 혐오 표현 다중 레이블 데이터셋"""

    def __init__(
        self,
        texts: list,
        labels: Optional[np.ndarray] = None,
        tokenizer=None,
        max_length: int = None,
        sample_weights: Optional[np.ndarray] = None,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length or TRAIN_CFG.max_length
        self.sample_weights = sample_weights

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }

        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)

        if self.sample_weights is not None:
            item["sample_weight"] = torch.tensor(
                self.sample_weights[idx], dtype=torch.float32
            )

        return item


def create_dataloader(
    df: pd.DataFrame,
    tokenizer,
    batch_size: int = 32,
    shuffle: bool = True,
    max_length: int = None,
    use_sample_weights: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """DataFrame → DataLoader 변환"""
    texts = df["text"].tolist()

    # 레이블 추출
    label_cols = [c for c in LABEL_COLUMNS if c in df.columns]
    labels = df[label_cols].values.astype(np.float32) if label_cols else None

    # 샘플 가중치
    weights = None
    if use_sample_weights and "sample_weight" in df.columns:
        weights = df["sample_weight"].values.astype(np.float32)

    dataset = HateSpeechDataset(
        texts=texts,
        labels=labels,
        tokenizer=tokenizer,
        max_length=max_length,
        sample_weights=weights,
    )

    # WeightedRandomSampler (학습용)
    sampler = None
    if use_sample_weights and weights is not None and shuffle:
        sampler = WeightedRandomSampler(
            weights=weights.tolist(),
            num_samples=len(weights),
            replacement=True,
        )
        shuffle = False  # sampler 사용 시 shuffle=False 필수

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return loader


def get_tokenizer(model_name: str):
    """사전학습 모델용 토크나이저 로드"""
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    logger.info(f"Tokenizer loaded: {model_name}")
    return tokenizer
