"""
데이터셋 검증 스크립트 – 외부 데이터 가용성 + 레이블 분포 확인
"""
import logging
import sys
import os

from ..config import RAW_DIR, LABEL_COLUMNS, ensure_dirs
from ..utils import setup_logger

logger = logging.getLogger(__name__)


def verify_unsmile():
    """UnSmile 원본 데이터 존재 확인"""
    train_path = os.path.join(RAW_DIR, "unsmile_train.tsv")
    valid_path = os.path.join(RAW_DIR, "unsmile_valid.tsv")

    errors = []
    for path in [train_path, valid_path]:
        if os.path.exists(path):
            import pandas as pd
            df = pd.read_csv(path, sep="\t")
            logger.info(f"✓ {os.path.basename(path)}: {len(df)} rows")

            missing = [c for c in LABEL_COLUMNS if c not in df.columns]
            if missing:
                errors.append(f"Missing columns in {path}: {missing}")
        else:
            errors.append(f"File not found: {path}")

    return errors


def verify_external_datasets():
    """외부 데이터셋 HuggingFace 접근 확인"""
    datasets_to_check = [
        ("K-MHaS", "jeanlee/kmhas_korean_hate_speech"),
        ("KOLD", "jeanlee/kold"),
        ("BEEP!", "jeanlee/korean_hate_speech"),
        ("Korean Toxic", "captainnemo9292/korean_toxic_comments_dataset"),
        ("Curse Filtering", "TheFrenchLeaf/KCF"),
        ("APEACH", "jason9693/APEACH"),
        ("Ko-HatefulMemes", "sgunderscore/ko-HatefulMemes-converted"),
    ]

    results = {}
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets 라이브러리가 설치되지 않았습니다")
        return {"error": "datasets not installed"}

    for name, repo in datasets_to_check:
        try:
            ds = load_dataset(repo, split="train", streaming=True)
            first = next(iter(ds))
            results[name] = {"status": "OK", "columns": list(first.keys())}
            logger.info(f"✓ {name} ({repo}): accessible")
        except Exception as e:
            results[name] = {"status": "FAIL", "error": str(e)}
            logger.warning(f"✗ {name} ({repo}): {e}")

    return results


def run_verification():
    """전체 검증 실행"""
    setup_logger("verify", level=logging.INFO)
    ensure_dirs()

    logger.info("=" * 60)
    logger.info("데이터셋 검증 시작")
    logger.info("=" * 60)

    # 1. UnSmile
    logger.info("\n[1/2] UnSmile 데이터 확인")
    errors = verify_unsmile()
    if errors:
        for e in errors:
            logger.error(f"  ✗ {e}")
    else:
        logger.info("  → UnSmile 데이터 정상")

    # 2. 외부 데이터셋
    logger.info("\n[2/2] 외부 데이터셋 접근 확인")
    results = verify_external_datasets()

    # 요약
    logger.info("\n" + "=" * 60)
    ok_count = sum(1 for v in results.values() if isinstance(v, dict) and v.get("status") == "OK")
    total = len(results) if "error" not in results else 0
    logger.info(f"외부 데이터셋: {ok_count}/{total} accessible")
    logger.info("=" * 60)

    return len(errors) == 0 and ok_count >= 3


if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)
