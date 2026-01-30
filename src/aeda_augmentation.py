"""
AEDA (Adaptive Easier Data Augmentation) 구현
소수 클래스 데이터 증강을 위한 구두점 삽입 기법
"""

import random
import pandas as pd
import numpy as np
from typing import List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AEDAAugmenter:
    """AEDA 증강기 - 구두점 삽입으로 데이터 증강"""
    
    def __init__(self, punc_ratio: float = 0.3):
        """
        Args:
            punc_ratio: 단어당 구두점 삽입 확률
        """
        self.punc_ratio = punc_ratio
        # 한국어 문맥에서 자연스러운 구두점
        self.punctuations = ['.', ',', '!', '?', ';', ':']
    
    def augment(self, text: str) -> str:
        """단일 텍스트 증강"""
        words = text.split()
        
        if len(words) == 0:
            return text
        
        # 각 단어 뒤에 확률적으로 구두점 삽입
        augmented_words = []
        for word in words:
            augmented_words.append(word)
            if random.random() < self.punc_ratio:
                punc = random.choice(self.punctuations)
                augmented_words.append(punc)
        
        return ' '.join(augmented_words)
    
    def augment_batch(self, texts: List[str], num_aug: int = 1) -> List[str]:
        """배치 증강"""
        augmented_texts = []
        for text in texts:
            for _ in range(num_aug):
                augmented_texts.append(self.augment(text))
        return augmented_texts


def augment_minority_classes(
    train_df: pd.DataFrame,
    label_columns: List[str],
    target_size: int = 2500,  # 2000 → 2500으로 적절히 조정
    punc_ratio: float = 0.4,
    augment_all: bool = False  # True → False로 변경
) -> pd.DataFrame:
    """
    소수 클래스에 대해 AEDA 증강 적용 (밸런스 유지 버전)

    Args:
        train_df: 학습 데이터프레임
        label_columns: 레이블 컬럼 리스트
        target_size: 각 클래스의 목표 샘플 수
        punc_ratio: 구두점 삽입 비율
        augment_all: 모든 클래스 증강 여부 (False로 소수 클래스만)

    Returns:
        증강된 데이터프레임
    """
    logger.info("\n" + "="*80)
    logger.info("AEDA 데이터 증강 시작 (밸런스 유지 모드)")
    logger.info("="*80)
    
    augmenter = AEDAAugmenter(punc_ratio=punc_ratio)
    
    # 원본 데이터 유지
    augmented_dfs = [train_df.copy()]
    
    # 각 레이블별로 샘플 수 확인
    logger.info("\n=== 클래스별 샘플 수 (증강 전) ===")
    for label in label_columns:
        count = train_df[label].sum()
        logger.info(f"{label}: {count}")
    
    # 증강 임계값: 2000 미만인 클래스만 증강
    minority_threshold = 2000

    for label in label_columns:
        label_count = train_df[label].sum()
        
        # 소수 클래스만 증강 (2000 미만)
        if label_count < minority_threshold:
            # 해당 레이블이 1인 샘플들만 추출
            minority_samples = train_df[train_df[label] == 1].copy()
            
            if len(minority_samples) == 0:
                continue

            # 목표까지 필요한 샘플 수
            needed_samples = target_size - label_count

            # 증강 횟수 계산 (과도하지 않게)
            if needed_samples > 0:
                num_augmentations = max(1, min(3, needed_samples // len(minority_samples)))
            else:
                num_augmentations = 0

            if num_augmentations == 0:
                continue

            logger.info(f"\n{label}: {label_count} → {target_size} 목표")
            logger.info(f"  증강 횟수: {num_augmentations}회/샘플")
            
            # AEDA 증강 적용
            for aug_round in range(num_augmentations):
                augmented_texts = []
                for text in minority_samples['text']:
                    augmented_texts.append(augmenter.augment(text))
                
                # 새로운 데이터프레임 생성
                aug_df = minority_samples.copy()
                aug_df['text'] = augmented_texts
                augmented_dfs.append(aug_df)
    
    # 모든 데이터프레임 결합
    final_df = pd.concat(augmented_dfs, ignore_index=True)
    
    # 셔플
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    logger.info("\n=== 클래스별 샘플 수 (증강 후) ===")
    for label in label_columns:
        count = final_df[label].sum()
        logger.info(f"{label}: {count}")
    
    logger.info(f"\n총 샘플 수: {len(train_df)} → {len(final_df)}")
    logger.info(f"증강 비율: {len(final_df)/len(train_df):.2f}x")
    logger.info("="*80 + "\n")
    
    return final_df


if __name__ == "__main__":
    # 테스트
    augmenter = AEDAAugmenter(punc_ratio=0.3)
    
    test_text = "틀딱들은 집에나 있어라"
    augmented = augmenter.augment(test_text)
    
    print(f"원본: {test_text}")
    print(f"증강: {augmented}")
