"""
데이터 로더 – UnSmile 원본 데이터 로딩 + 테스트셋 분리 + 셔플링
"""
import os
import logging

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from ..config import RAW_DIR, PROCESSED_DIR, LABEL_COLUMNS, TRAIN_CFG

logger = logging.getLogger(__name__)


def load_unsmile(split: str = "train") -> pd.DataFrame:
    """
    UnSmile 데이터셋 로드

    Args:
        split: 'train' 또는 'valid'
    """
    path = os.path.join(RAW_DIR, f"unsmile_{split}.tsv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"UnSmile 데이터 미발견: {path}")

    df = pd.read_csv(path, sep="\t")

    # 레이블 컬럼 존재 확인 및 숫자 변환
    for col in LABEL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(float).astype(int)
        else:
            df[col] = 0

    # 'clean' 컬럼 제거 (있으면)
    if "clean" in df.columns:
        df = df.drop(columns=["clean"])

    df["source"] = "unsmile"
    df["mapping_confidence"] = 1.0
    logger.info(f"UnSmile {split} loaded: {len(df)} samples")
    return df


def load_unsmile_all() -> pd.DataFrame:
    """UnSmile train + valid 통합 로드"""
    train_df = load_unsmile("train")
    valid_df = load_unsmile("valid")
    df = pd.concat([train_df, valid_df], ignore_index=True)
    logger.info(f"UnSmile total: {len(df)} samples")
    return df


def separate_test_set(
    df: pd.DataFrame,
    test_ratio: float = 0.1,
    seed: int = None,
) -> tuple:
    """
    [중요] 테스트 데이터를 맨 처음에 분리하고 셔플링.
    테스트셋은 학습 과정에서 절대 사용하지 않음.

    Returns:
        (train_df, test_df)
    """
    if seed is None:
        seed = TRAIN_CFG.seed

    # 멀티레이블 → 합산으로 의사 레이블 생성 (Stratified 분리용)
    label_sums = df[LABEL_COLUMNS].sum(axis=1).astype(str)

    train_df, test_df = train_test_split(
        df, test_size=test_ratio, random_state=seed, stratify=label_sums
    )

    # 셔플링
    train_df = train_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_df = test_df.sample(frac=1.0, random_state=seed + 1).reset_index(drop=True)

    logger.info(
        f"Test set separated: train={len(train_df)}, test={len(test_df)} "
        f"(ratio={test_ratio})"
    )
    return train_df, test_df


def create_kfold_splits(
    df: pd.DataFrame,
    n_splits: int = None,
    seed: int = None,
) -> list:
    """
    Stratified K-Fold 분할 생성 (멀티레이블 대응)

    Returns:
        [(train_idx, val_idx), ...] – n_splits 개
    """
    if n_splits is None:
        n_splits = TRAIN_CFG.num_folds
    if seed is None:
        seed = TRAIN_CFG.seed

    # 멀티레이블 stratify: 레이블 조합 문자열 활용
    label_str = df[LABEL_COLUMNS].astype(str).agg("".join, axis=1)

    # 조합이 너무 적은 클래스 → 빈도 기반 그룹핑
    label_counts = label_str.value_counts()
    rare_mask = label_str.isin(label_counts[label_counts < n_splits].index)
    label_str[rare_mask] = "rare"

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(skf.split(df, label_str))

    for i, (train_idx, val_idx) in enumerate(folds):
        logger.info(f"Fold {i + 1}: train={len(train_idx)}, val={len(val_idx)}")

    return folds


def save_processed_data(df: pd.DataFrame, name: str):
    """전처리 완료 데이터 저장"""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    path = os.path.join(PROCESSED_DIR, f"{name}.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info(f"Saved: {path} ({len(df)} rows)")


def load_processed_data(name: str) -> pd.DataFrame:
    """전처리 완료 데이터 로드"""
    path = os.path.join(PROCESSED_DIR, f"{name}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed data not found: {path}")
    return pd.read_csv(path)
