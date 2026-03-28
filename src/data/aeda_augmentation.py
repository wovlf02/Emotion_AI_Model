"""
AEDA 데이터 증강 – 소수 클래스 오버샘플링
docs/10_Phase2_구현명세서.md 기반 구현
"""
import random
import logging
from typing import List

import pandas as pd
import numpy as np

from ..config import LABEL_COLUMNS, TRAIN_CFG

logger = logging.getLogger(__name__)

# AEDA 삽입용 한국어 구두점
AEDA_PUNCTUATIONS = [".", ",", "!", "?", ";", ":", "…", "-", "~"]


def aeda_augment(text: str, punc_ratio: float = None) -> str:
    """
    AEDA (An Easier Data Augmentation) – 랜덤 위치에 구두점 삽입

    Args:
        text: 원본 텍스트
        punc_ratio: 삽입 비율 (단어 수 대비)
    """
    if punc_ratio is None:
        punc_ratio = TRAIN_CFG.aeda_punc_ratio

    words = text.split()
    if len(words) < 2:
        return text

    n_insert = max(1, int(len(words) * punc_ratio))
    new_words = list(words)

    for _ in range(n_insert):
        pos = random.randint(0, len(new_words))
        punc = random.choice(AEDA_PUNCTUATIONS)
        new_words.insert(pos, punc)

    return " ".join(new_words)


def augment_minority_classes(
    df: pd.DataFrame,
    target_per_class: int = None,
    seed: int = None,
) -> pd.DataFrame:
    """
    소수 클래스에 대해 AEDA 증강 수행

    각 레이블의 positive 샘플이 target_per_class 미만이면 증강.

    Args:
        df: 원본 데이터 (text + 레이블 컬럼 필수)
        target_per_class: 클래스당 목표 positive 샘플 수
        seed: 랜덤 시드
    """
    if target_per_class is None:
        target_per_class = TRAIN_CFG.aeda_target_per_class
    if seed is None:
        seed = TRAIN_CFG.seed

    random.seed(seed)
    np.random.seed(seed)

    augmented_rows = []

    for col in LABEL_COLUMNS:
        positive_df = df[df[col] == 1]
        current_count = len(positive_df)

        if current_count >= target_per_class:
            logger.info(f"  {col}: {current_count} ≥ {target_per_class}, skip")
            continue

        needed = target_per_class - current_count
        if current_count == 0:
            logger.warning(f"  {col}: no positive samples, cannot augment")
            continue

        # 필요한 만큼 반복 샘플링 + 증강
        n_repeats = (needed // current_count) + 1
        pool = pd.concat([positive_df] * n_repeats, ignore_index=True).head(needed)

        for _, row in pool.iterrows():
            new_row = row.copy()
            new_row["text"] = aeda_augment(row["text"])
            new_row["source"] = row.get("source", "augmented") + "_aeda"
            new_row["sample_weight"] = row.get("sample_weight", 1.0) * 0.8
            augmented_rows.append(new_row)

        logger.info(f"  {col}: {current_count} → {current_count + needed} (+{needed})")

    if augmented_rows:
        aug_df = pd.DataFrame(augmented_rows)
        result = pd.concat([df, aug_df], ignore_index=True)
        logger.info(f"AEDA augmentation: {len(df)} → {len(result)} (+{len(aug_df)})")
        return result

    logger.info("AEDA augmentation: no augmentation needed")
    return df
