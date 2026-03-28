"""
평가 지표 계산 – F1-Macro, Hamming Accuracy, Exact Match, 클래스별 F1
"""
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report
from typing import Dict, Tuple

from ..config import LABEL_COLUMNS


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
    thresholds: np.ndarray = None,
) -> Dict[str, float]:
    """
    종합 평가 지표 계산

    Args:
        y_true:  (N, 9) ground-truth binary
        y_pred:  (N, 9) 확률값 또는 이진값
        threshold: 단일 임계값 (thresholds 미지정 시)
        thresholds: (9,) 클래스별 임계값
    """
    # 확률 → 이진 변환
    if y_pred.dtype == np.float64 or y_pred.dtype == np.float32:
        if thresholds is not None:
            y_bin = (y_pred > thresholds).astype(int)
        else:
            y_bin = (y_pred > threshold).astype(int)
    else:
        y_bin = y_pred

    results = {}

    # F1-Macro (핵심 지표)
    results["f1_macro"] = f1_score(y_true, y_bin, average="macro", zero_division=0)

    # F1-Micro
    results["f1_micro"] = f1_score(y_true, y_bin, average="micro", zero_division=0)

    # 클래스별 F1
    per_class_f1 = f1_score(y_true, y_bin, average=None, zero_division=0)
    for i, col in enumerate(LABEL_COLUMNS):
        results[f"f1_{col}"] = per_class_f1[i]

    # Hamming Accuracy (element-wise)
    results["hamming_accuracy"] = accuracy_score(y_true.flatten(), y_bin.flatten())

    # Exact Match Ratio (전체 레이블 벡터 일치 비율)
    results["exact_match"] = np.mean(np.all(y_true == y_bin, axis=1))

    # Subset Accuracy (동일)
    results["subset_accuracy"] = results["exact_match"]

    return results


def find_optimal_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    search_range: Tuple[float, float] = (0.1, 0.9),
    step: float = 0.01,
) -> Tuple[np.ndarray, float]:
    """
    클래스별 최적 임계값 Grid Search

    Returns:
        (thresholds, best_f1_macro)
    """
    num_classes = y_true.shape[1]
    thresholds = np.full(num_classes, 0.5)
    candidates = np.arange(search_range[0], search_range[1] + step, step)

    for class_idx in range(num_classes):
        best_f1, best_t = 0.0, 0.5
        for t in candidates:
            pred = (y_prob[:, class_idx] > t).astype(int)
            f1 = f1_score(y_true[:, class_idx], pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        thresholds[class_idx] = best_t

    # 전체 F1-Macro
    y_bin = (y_prob > thresholds).astype(int)
    overall_f1 = f1_score(y_true, y_bin, average="macro", zero_division=0)

    return thresholds, overall_f1


def print_classification_report(
    y_true: np.ndarray, y_pred: np.ndarray, thresholds: np.ndarray = None
):
    """분류 리포트 출력"""
    if y_pred.dtype in (np.float32, np.float64):
        if thresholds is not None:
            y_bin = (y_pred > thresholds).astype(int)
        else:
            y_bin = (y_pred > 0.5).astype(int)
    else:
        y_bin = y_pred

    print(classification_report(y_true, y_bin, target_names=LABEL_COLUMNS, zero_division=0))
