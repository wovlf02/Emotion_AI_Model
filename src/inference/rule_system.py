"""
룰 시스템 – TextNormalizer + KeywordHintGenerator + PostProcessingCorrector
docs/10_Phase2_구현명세서.md Stage E 기반 구현
"""
import re
import logging
from typing import Dict, List, Optional

import numpy as np

from ..config import LABEL_COLUMNS, NUM_LABELS

logger = logging.getLogger(__name__)

# ── 카테고리별 키워드 사전 ─────────────────────────────────
# Strong Hints (가중치 0.6)
STRONG_KEYWORDS: Dict[str, List[str]] = {
    "여성/가족": [
        "김치녀", "맘충", "페미", "보슬", "한녀", "된장녀", "메갈", "김여사",
        "걸레", "암컷", "년", "페미나치", "꼴페미", "워마드",
    ],
    "남성": [
        "한남", "한남충", "재기", "자지", "남충", "틀딱남",
    ],
    "성소수자": [
        "게이", "레즈", "호모", "트랜스젠더", "동성애", "쉬메일", "성전환",
        "바이", "양성애", "퀴어", "이반",
    ],
    "인종/국적": [
        "짱깨", "쪽바리", "흑형", "깜둥이", "동남아", "조선족", "양키",
        "니거", "커리", "다문화", "난민",
    ],
    "연령": [
        "틀딱", "꼰대", "노인충", "급식충", "잼민이", "애새끼",
        "영감탱이", "할망구",
    ],
    "지역": [
        "홍어", "전라디언", "경상디언", "대구", "광주", "충청도",
        "전라도", "경상도",
    ],
    "종교": [
        "개독", "맘몬교", "예수쟁이", "불교충", "이슬람", "무슬림",
        "천주교", "신천지",
    ],
    "기타 혐오": [
        "장애인", "찐따", "미친놈", "정신병", "ㄴㄱㅁ", "병신",
        "불구", "지체장애",
    ],
    "악플/욕설": [
        "시발", "씨발", "시팔", "ㅅㅂ", "개새끼", "병신", "ㅂㅅ",
        "ㅈㄹ", "지랄", "닥쳐", "꺼져", "뒤져", "뒈져",
        "ㄲㅈ", "미친", "씹", "좆", "새끼", "ㅆㅂ",
    ],
}

# Weak Hints (가중치 0.25)
WEAK_KEYWORDS: Dict[str, List[str]] = {
    "여성/가족": ["여자", "여성", "엄마", "아줌마", "아내"],
    "남성": ["남자", "남성", "아빠", "아저씨"],
    "성소수자": ["성소수자", "LGBT"],
    "인종/국적": ["외국인", "이민", "다문화"],
    "연령": ["노인", "청소년", "어르신"],
    "지역": ["서울", "부산", "제주"],
    "종교": ["교회", "절", "사찰", "성당", "모스크"],
    "기타 혐오": ["혐오", "차별"],
    "악플/욕설": ["ㅋㅋ", "ㅎㅎ", "짜증", "열받"],
}

# 컨텍스트 억제 키워드
SELF_DEPRECATION_PREFIX = ["나는", "나 ", "저는", "제가", "내가"]
FOOD_CONTEXT_KW = ["맛있", "먹", "요리", "음식", "식당", "회"]

# 교차 레이블 강화 규칙
CROSS_REINFORCEMENT_RULES = [
    # (source_label, threshold, target_label, boost)
    ("여성/가족", 0.4, "악플/욕설", 0.15),
    ("남성", 0.4, "악플/욕설", 0.15),
    ("성소수자", 0.4, "악플/욕설", 0.15),
    ("인종/국적", 0.4, "악플/욕설", 0.15),
    ("연령", 0.4, "악플/욕설", 0.10),
    ("종교", 0.4, "악플/욕설", 0.10),
]


class KeywordHintGenerator:
    """텍스트 기반 키워드 힌트 점수 생성 (9차원)"""

    def __init__(
        self,
        strong_weight: float = 0.6,
        weak_weight: float = 0.25,
    ):
        self.strong_weight = strong_weight
        self.weak_weight = weak_weight

    def generate(self, text: str) -> np.ndarray:
        """단일 텍스트 → 9차원 힌트 점수"""
        text_lower = text.lower().strip()
        hints = np.zeros(NUM_LABELS, dtype=np.float32)

        for i, col in enumerate(LABEL_COLUMNS):
            score = 0.0

            # Strong 키워드
            strong_kw = STRONG_KEYWORDS.get(col, [])
            for kw in strong_kw:
                if kw in text_lower:
                    score += self.strong_weight
                    break  # 1개만 매칭해도 충분

            # Weak 키워드
            weak_kw = WEAK_KEYWORDS.get(col, [])
            for kw in weak_kw:
                if kw in text_lower:
                    score += self.weak_weight
                    break

            hints[i] = min(score, 1.0)

        # 컨텍스트 억제
        hints = self._apply_suppression(text_lower, hints)

        return hints

    def generate_batch(self, texts: List[str]) -> np.ndarray:
        """배치 텍스트 → (N, 9) 힌트 행렬"""
        return np.array([self.generate(t) for t in texts])

    @staticmethod
    def _apply_suppression(text: str, hints: np.ndarray) -> np.ndarray:
        """컨텍스트 억제 적용"""
        # 자기비하 컨텍스트
        if any(p in text[:10] for p in SELF_DEPRECATION_PREFIX):
            hints *= 0.5

        # 음식 컨텍스트 → 지역 힌트 억제
        if any(food in text for food in FOOD_CONTEXT_KW):
            region_idx = LABEL_COLUMNS.index("지역")
            hints[region_idx] *= 0.3

        # 인용 컨텍스트
        if '"' in text or "'" in text or "「" in text:
            hints *= 0.7

        return hints


class PostProcessingCorrector:
    """모델 예측 + 키워드 힌트 후처리 보정"""

    def __init__(self, blend_weight: float = 0.3):
        self.blend_weight = blend_weight
        self.hint_generator = KeywordHintGenerator()

    def correct(
        self,
        texts: List[str],
        model_probs: np.ndarray,
    ) -> np.ndarray:
        """
        모델 확률을 키워드 힌트와 블렌딩하여 보정

        Args:
            texts: N개 텍스트
            model_probs: (N, 9) 모델 출력 확률

        Returns:
            (N, 9) 보정된 확률
        """
        keyword_hints = self.hint_generator.generate_batch(texts)

        # 블렌딩: P_corrected = (1-w)×P_model + w×P_keyword
        corrected = (1.0 - self.blend_weight) * model_probs + self.blend_weight * keyword_hints

        # 교차 레이블 강화
        corrected = self._apply_cross_reinforcement(corrected)

        return np.clip(corrected, 0.0, 1.0)

    @staticmethod
    def _apply_cross_reinforcement(probs: np.ndarray) -> np.ndarray:
        """교차 레이블 강화 규칙 적용"""
        result = probs.copy()
        for src_label, threshold, tgt_label, boost in CROSS_REINFORCEMENT_RULES:
            src_idx = LABEL_COLUMNS.index(src_label)
            tgt_idx = LABEL_COLUMNS.index(tgt_label)
            mask = result[:, src_idx] > threshold
            result[mask, tgt_idx] = np.minimum(result[mask, tgt_idx] + boost, 1.0)
        return result
