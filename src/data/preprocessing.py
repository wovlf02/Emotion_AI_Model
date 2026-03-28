"""
텍스트 전처리 파이프라인 – TextNormalizer, TextCleaner
docs/11_전처리_상세설계.md 기반 구현
"""
import re
import unicodedata
import logging
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)

# ── 초성 복원 맵 ───────────────────────────────────────────
CHOSUNG_MAP = {
    "ㅋ": "크",  "ㅎ": "하",  "ㅉ": "짜",  "ㄷ": "다",
    "ㅂ": "바",  "ㅈ": "자",  "ㄱ": "가",  "ㅅ": "사",
    "ㅁ": "마",  "ㄴ": "나",  "ㅇ": "아",  "ㄹ": "라",
}

# ── 난독화 복원 맵 ─────────────────────────────────────────
DEOBFUSCATION_MAP = {
    "ㅇㅣ기": "이기",   "ㅂㅏ보": "바보",   "ㅁㅓ저리": "머저리",
    "ㄱㅓ지": "거지",   "ㅂㅕㅇ신": "병신",   "ㅅㅣ발": "시발",
    "ㅈㅏ살": "자살",   "ㄲㅓ져": "꺼져",   "ㄴㅏ가": "나가",
    "ㅈㅣ랄": "지랄",   "미ㅊ": "미친",     "ㅅㅐ끼": "새끼",
    "ㅆㅂ": "시발",    "ㄱㅐ": "개",       "ㅈㄹ": "지랄",
}

# ── 언마스킹 패턴 ──────────────────────────────────────────
UNMASK_PATTERNS = [
    (re.compile(r"[시씨쉬슈쓔쓰시][0-9@#*!]+[발빨팔]"), "시발"),
    (re.compile(r"[병뼝][0-9@#*!]*[신씬]"), "병신"),
    (re.compile(r"[새쌔][0-9@#*!]*[끼키]"), "새끼"),
    (re.compile(r"[지찌][0-9@#*!]*[랄럴]"), "지랄"),
    (re.compile(r"[개게][0-9@#*!]*[새세][0-9@#*!]*[끼키]"), "개새끼"),
    (re.compile(r"ㅅ[ㅂ빠발팔]"), "시발"),
    (re.compile(r"ㅂ[ㅅ신씬]"), "병신"),
]


class TextNormalizer:
    """텍스트 정규화 파이프라인 (전처리 Phase 1)"""

    def normalize(self, text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            return ""
        text = self._normalize_unicode(text)
        text = self._remove_html_url(text)
        text = self._remove_placeholders(text)
        text = self._restore_chosung(text)
        text = self._deobfuscate(text)
        text = self._unmask(text)
        text = self._normalize_repeats(text)
        return text.strip()

    def normalize_batch(self, texts: List[str]) -> List[str]:
        results, changed = [], 0
        for t in texts:
            n = self.normalize(t)
            if n != t:
                changed += 1
            results.append(n)
        logger.info(
            f"Normalization: {len(texts)} processed, "
            f"{changed} changed ({changed / max(len(texts), 1) * 100:.1f}%)"
        )
        return results

    # ── 개별 단계 ──────────────────────────────────────────
    @staticmethod
    def _normalize_unicode(text: str) -> str:
        return unicodedata.normalize("NFC", text)

    @staticmethod
    def _remove_html_url(text: str) -> str:
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"https?://\S+", " ", text)
        text = re.sub(r"www\.\S+", " ", text)
        return text

    @staticmethod
    def _remove_placeholders(text: str) -> str:
        text = re.sub(r"\[placeholder\]", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\[.*?\]", "", text)
        return text

    @staticmethod
    def _restore_chosung(text: str) -> str:
        result = []
        i = 0
        while i < len(text):
            c = text[i]
            if c in CHOSUNG_MAP:
                # 연속 초성 3개 이상이면 복원 대상
                j = i
                while j < len(text) and text[j] in CHOSUNG_MAP:
                    j += 1
                if j - i >= 3:
                    result.append("".join(CHOSUNG_MAP.get(text[k], text[k]) for k in range(i, j)))
                    i = j
                    continue
            result.append(c)
            i += 1
        return "".join(result)

    @staticmethod
    def _deobfuscate(text: str) -> str:
        for pattern, replacement in DEOBFUSCATION_MAP.items():
            text = text.replace(pattern, replacement)
        return text

    @staticmethod
    def _unmask(text: str) -> str:
        for pattern, replacement in UNMASK_PATTERNS:
            text = pattern.sub(replacement, text)
        return text

    @staticmethod
    def _normalize_repeats(text: str) -> str:
        # 같은 문자 3회 이상 반복 → 2회로
        text = re.sub(r"(.)\1{2,}", r"\1\1", text)
        # 같은 단어 2회 이상 반복 → 1회
        text = re.sub(r"(\b\w+\b)(\s+\1){2,}", r"\1", text)
        return text


class TextCleaner:
    """텍스트 정제 파이프라인 (전처리 Phase 2)"""

    def __init__(self, min_len: int = 5, max_len: int = 512, min_korean_ratio: float = 0.3):
        self.min_len = min_len
        self.max_len = max_len
        self.min_korean_ratio = min_korean_ratio

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.copy()

        # 1. 공백 정규화
        df["text"] = df["text"].apply(self._normalize_whitespace)

        # 2. 빈 텍스트 제거
        df = df[df["text"].str.strip().str.len() > 0]

        # 3. 길이 필터
        text_len = df["text"].str.len()
        df = df[(text_len >= self.min_len) & (text_len <= self.max_len)]

        # 4. 한국어 비율 필터
        df = df[df["text"].apply(self._korean_ratio) >= self.min_korean_ratio]

        after = len(df)
        logger.info(f"Cleaning: {before} → {after} ({before - after} removed)")
        return df.reset_index(drop=True)

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _korean_ratio(text: str) -> float:
        if not text:
            return 0.0
        korean = sum(1 for c in text if "\uac00" <= c <= "\ud7a3" or "\u3131" <= c <= "\u3163")
        return korean / len(text)


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """중복 제거 (exact + normalized)"""
    before = len(df)
    df = df.drop_duplicates(subset="text", keep="first")
    if "text_normalized" in df.columns:
        df = df.sort_values("mapping_confidence", ascending=False)
        df = df.drop_duplicates(subset="text_normalized", keep="first")
    after = len(df)
    dup_rate = (before - after) / max(before, 1) * 100
    logger.info(f"Deduplication: {before} → {after} ({dup_rate:.1f}% removed)")
    return df.reset_index(drop=True)
