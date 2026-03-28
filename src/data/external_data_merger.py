"""
외부 데이터셋 병합기 – 7개 외부 데이터셋 통합 + 레이블 매핑
docs/09_외부데이터셋_명세.md, docs/11_전처리_상세설계.md 기반 구현
"""
import os
import re
import logging
from typing import Optional

import pandas as pd
import numpy as np
from datasets import load_dataset

from ..config import (
    RAW_DIR, PROCESSED_DIR, LABEL_COLUMNS, NUM_LABELS,
    SOURCE_WEIGHTS, MAPPING_CONFIDENCE,
)

logger = logging.getLogger(__name__)

# ── K-MHaS 성별 분류 키워드 ────────────────────────────────
FEMALE_KW = ["김치녀", "맘충", "여자", "여성", "페미", "보슬", "한녀", "된장녀", "메갈",
             "김여사", "걸레", "암컷", "년"]
MALE_KW = ["한남", "남자", "남성", "자지", "남충", "틀딱", "재기"]


def _empty_labels() -> dict:
    return {col: 0 for col in LABEL_COLUMNS}


# ──────────────────────────────────────────────────────────
#  1. K-MHaS (109K)
# ──────────────────────────────────────────────────────────
def load_kmhas() -> Optional[pd.DataFrame]:
    """K-MHaS 데이터셋 로드 + 레이블 매핑"""
    try:
        ds = load_dataset("jeanlee/kmhas_korean_hate_speech")
    except Exception as e:
        logger.warning(f"K-MHaS 로드 실패: {e}")
        return None

    rows = []
    for split in ["train", "validation", "test"]:
        if split not in ds:
            continue
        for item in ds[split]:
            text = item.get("text", "")
            label_idx = item.get("label", -1)
            if not text or label_idx < 0:
                continue

            labels = _empty_labels()
            kmhas_map = {
                0: None,              # origin → clean
                1: "종교",            # religion
                2: None,              # gender → 키워드 분리
                3: "기타 혐오",       # age (매핑 변경: K-MHaS age ≠ UnSmile 연령)
                4: "기타 혐오",       # physical → 기타 혐오
                5: "인종/국적",       # race
                6: "기타 혐오",       # politics → 기타 혐오
            }

            if label_idx == 0:
                continue  # clean 건너뛰기

            if label_idx == 2:  # gender → 키워드 기반 분리
                text_lower = text.lower()
                is_female = any(kw in text_lower for kw in FEMALE_KW)
                is_male = any(kw in text_lower for kw in MALE_KW)
                if is_female:
                    labels["여성/가족"] = 1
                elif is_male:
                    labels["남성"] = 1
                else:
                    labels["기타 혐오"] = 1
            else:
                target = kmhas_map.get(label_idx)
                if target and target in labels:
                    labels[target] = 1

            labels["악플/욕설"] = 1  # K-MHaS는 혐오 텍스트 → 욕설 동반 가능
            row = {"text": text, "source": "kmhas",
                   "mapping_confidence": MAPPING_CONFIDENCE["kmhas_direct"]}
            row.update(labels)
            rows.append(row)

    df = pd.DataFrame(rows)
    logger.info(f"K-MHaS loaded: {len(df)} samples")
    return df


# ──────────────────────────────────────────────────────────
#  2. KOLD (40K)
# ──────────────────────────────────────────────────────────
KOLD_TARGET_MAP = {
    "여성": "여성/가족",
    "남성": "남성",
    "성소수자": "성소수자",
    "성 정체성-여성": "여성/가족",
    "성 정체성-남성": "남성",
    "성 정체성-성소수자": "성소수자",
    "인종/국적": "인종/국적",
    "국적": "인종/국적",
    "민족": "인종/국적",
    "연령": "연령",
    "지역": "지역",
    "종교": "종교",
}


def load_kold() -> Optional[pd.DataFrame]:
    """KOLD 데이터셋 로드 + 레이블 매핑"""
    try:
        ds = load_dataset("jeanlee/kold")
    except Exception as e:
        logger.warning(f"KOLD 로드 실패: {e}")
        return None

    rows = []
    for split in ds:
        for item in ds[split]:
            text = item.get("comment", "") or item.get("text", "")
            if not text:
                continue

            off = item.get("OFF", item.get("off", ""))
            if str(off).strip().upper() not in ["TRUE", "1", "YES"]:
                continue

            labels = _empty_labels()
            target = item.get("target_group", "") or ""

            matched = False
            for key, unsmile_label in KOLD_TARGET_MAP.items():
                if key in target:
                    labels[unsmile_label] = 1
                    matched = True

            if not matched:
                labels["기타 혐오"] = 1

            labels["악플/욕설"] = 1

            conf = MAPPING_CONFIDENCE["kold_direct" if matched else "kold_keyword"]
            row = {"text": text, "source": "kold", "mapping_confidence": conf}
            row.update(labels)
            rows.append(row)

    df = pd.DataFrame(rows)
    logger.info(f"KOLD loaded: {len(df)} samples")
    return df


# ──────────────────────────────────────────────────────────
#  3. Korean Hate Speech / BEEP! (9.3K)
# ──────────────────────────────────────────────────────────
def load_beep() -> Optional[pd.DataFrame]:
    """Korean Hate Speech (BEEP!) 로드 + 레이블 매핑"""
    try:
        ds = load_dataset("jeanlee/korean_hate_speech")
    except Exception as e:
        logger.warning(f"BEEP! 로드 실패: {e}")
        return None

    rows = []
    for split in ds:
        for item in ds[split]:
            text = item.get("comments", "") or item.get("text", "")
            if not text:
                continue

            hate = item.get("hate", "none")
            bias = item.get("contain_gender_bias", "none")

            if hate == "none" and bias in ["none", "False", False]:
                continue

            labels = _empty_labels()

            if hate in ["hate", "offensive"]:
                labels["악플/욕설"] = 1

            if bias in ["True", True, "gender_bias"]:
                labels["여성/가족"] = 1

            if hate == "hate":
                labels["기타 혐오"] = 1

            conf = MAPPING_CONFIDENCE["beep_direct"]
            row = {"text": text, "source": "beep", "mapping_confidence": conf}
            row.update(labels)
            rows.append(row)

    df = pd.DataFrame(rows)
    logger.info(f"BEEP! loaded: {len(df)} samples")
    return df


# ──────────────────────────────────────────────────────────
#  4. Korean Toxic Comments (8.4K)
# ──────────────────────────────────────────────────────────
KOREAN_TOXIC_VALID_CATS = {"01", "03", "04"}  # 비난혐오차별, 욕설, 폭력
KOREAN_TOXIC_MAP = {
    "01": "기타 혐오",
    "03": "악플/욕설",
    "04": "기타 혐오",
}


def load_korean_toxic() -> Optional[pd.DataFrame]:
    """Korean Toxic Comments 로드 + 레이블 매핑"""
    try:
        ds = load_dataset("captainnemo9292/korean_toxic_comments_dataset")
    except Exception as e:
        logger.warning(f"Korean Toxic 로드 실패: {e}")
        return None

    rows = []
    for split in ds:
        for item in ds[split]:
            text = item.get("text", "") or item.get("comment", "")
            if not text:
                continue

            # [placeholder] 제거
            text = re.sub(r"\[placeholder\]", "", text).strip()
            if not text:
                continue

            cat = str(item.get("level2_type", "") or item.get("category", ""))[:2]
            if cat not in KOREAN_TOXIC_VALID_CATS:
                continue

            labels = _empty_labels()
            target = KOREAN_TOXIC_MAP.get(cat)
            if target:
                labels[target] = 1
            labels["악플/욕설"] = 1

            row = {"text": text, "source": "korean_toxic",
                   "mapping_confidence": MAPPING_CONFIDENCE["korean_toxic"]}
            row.update(labels)
            rows.append(row)

    df = pd.DataFrame(rows)
    logger.info(f"Korean Toxic loaded: {len(df)} samples")
    return df


# ──────────────────────────────────────────────────────────
#  5. Korean Curse Filtering (1K)
# ──────────────────────────────────────────────────────────
def load_curse_filtering() -> Optional[pd.DataFrame]:
    """Korean Curse Filtering 로드"""
    try:
        ds = load_dataset("TheFrenchLeaf/KCF")
    except Exception as e:
        logger.warning(f"Curse Filtering 로드 실패: {e}")
        return None

    rows = []
    for split in ds:
        for item in ds[split]:
            text = item.get("text", "")
            label = item.get("label", 0)
            if not text or label == 0:
                continue

            labels = _empty_labels()
            labels["악플/욕설"] = 1

            row = {"text": text, "source": "curse_filtering",
                   "mapping_confidence": MAPPING_CONFIDENCE["curse_filtering"]}
            row.update(labels)
            rows.append(row)

    df = pd.DataFrame(rows)
    logger.info(f"Curse Filtering loaded: {len(df)} samples")
    return df


# ──────────────────────────────────────────────────────────
#  6. APEACH (11.6K) – pseudo-labeling 대상
# ──────────────────────────────────────────────────────────
def load_apeach() -> Optional[pd.DataFrame]:
    """APEACH 텍스트 로드 (레이블은 pseudo-labeling으로 추후 부여)"""
    try:
        ds = load_dataset("jason9693/APEACH")
    except Exception as e:
        logger.warning(f"APEACH 로드 실패: {e}")
        return None

    rows = []
    for split in ds:
        for item in ds[split]:
            text = item.get("text", "")
            if not text:
                continue
            labels = _empty_labels()
            row = {"text": text, "source": "apeach",
                   "mapping_confidence": MAPPING_CONFIDENCE["apeach_pseudo"],
                   "needs_pseudo_label": True}
            row.update(labels)
            rows.append(row)

    df = pd.DataFrame(rows)
    logger.info(f"APEACH loaded: {len(df)} samples (pseudo-label pending)")
    return df


# ──────────────────────────────────────────────────────────
#  7. Ko-HatefulMemes (8.5K) – pseudo-labeling 대상
# ──────────────────────────────────────────────────────────
def load_ko_hateful_memes() -> Optional[pd.DataFrame]:
    """Ko-HatefulMemes 텍스트 로드"""
    try:
        ds = load_dataset("sgunderscore/ko-HatefulMemes-converted")
    except Exception as e:
        logger.warning(f"Ko-HatefulMemes 로드 실패: {e}")
        return None

    rows = []
    for split in ds:
        for item in ds[split]:
            text = item.get("text", "")
            if not text or len(text.strip()) < 5:
                continue
            labels = _empty_labels()
            row = {"text": text, "source": "ko_hateful_memes",
                   "mapping_confidence": MAPPING_CONFIDENCE["hateful_pseudo"],
                   "needs_pseudo_label": True}
            row.update(labels)
            rows.append(row)

    df = pd.DataFrame(rows)
    logger.info(f"Ko-HatefulMemes loaded: {len(df)} samples (pseudo-label pending)")
    return df


# ──────────────────────────────────────────────────────────
#  통합 병합
# ──────────────────────────────────────────────────────────
def merge_all_datasets(
    base_df: pd.DataFrame,
    include_pseudo: bool = False,
) -> pd.DataFrame:
    """
    모든 외부 데이터셋 병합

    Args:
        base_df: UnSmile 원본 데이터
        include_pseudo: APEACH/Ko-HatefulMemes 포함 여부
    """
    loaders = [
        ("K-MHaS", load_kmhas),
        ("KOLD", load_kold),
        ("BEEP!", load_beep),
        ("Korean Toxic", load_korean_toxic),
        ("Curse Filtering", load_curse_filtering),
    ]

    if include_pseudo:
        loaders.extend([
            ("APEACH", load_apeach),
            ("Ko-HatefulMemes", load_ko_hateful_memes),
        ])

    all_dfs = [base_df]

    for name, loader_fn in loaders:
        logger.info(f"Loading {name}...")
        df_ext = loader_fn()
        if df_ext is not None and len(df_ext) > 0:
            all_dfs.append(df_ext)
            logger.info(f"  → {name}: {len(df_ext)} samples added")
        else:
            logger.warning(f"  → {name}: 로드 실패 또는 빈 데이터")

    merged = pd.concat(all_dfs, ignore_index=True)

    # source_weight 추가
    merged["sample_weight"] = merged["source"].map(SOURCE_WEIGHTS).fillna(0.5)

    # needs_pseudo_label 기본값
    if "needs_pseudo_label" not in merged.columns:
        merged["needs_pseudo_label"] = False
    merged["needs_pseudo_label"] = merged["needs_pseudo_label"].fillna(False)

    logger.info(
        f"Total merged: {len(merged)} samples\n"
        f"  Source distribution:\n"
        + "\n".join(
            f"    {k}: {v}" for k, v in merged["source"].value_counts().items()
        )
    )

    return merged
