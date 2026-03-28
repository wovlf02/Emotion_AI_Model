"""
임계값 최적화 – Grid Search + Bayesian (Optuna)
docs/10_Phase2_구현명세서.md Stage F 기반 구현
"""
import logging
from typing import Tuple, Optional

import numpy as np
from sklearn.metrics import f1_score

from ..config import LABEL_COLUMNS, NUM_LABELS, ENSEMBLE_CFG

logger = logging.getLogger(__name__)

# Optuna 선택적 import
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


def grid_search_thresholds(
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
    thresholds = np.full(NUM_LABELS, 0.5)
    candidates = np.arange(search_range[0], search_range[1] + step, step)

    for i in range(NUM_LABELS):
        best_f1, best_t = 0.0, 0.5
        for t in candidates:
            pred = (y_prob[:, i] > t).astype(int)
            f1 = f1_score(y_true[:, i], pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        thresholds[i] = best_t
        logger.info(f"  {LABEL_COLUMNS[i]}: threshold={best_t:.3f}, F1={best_f1:.4f}")

    y_pred = (y_prob > thresholds).astype(int)
    overall_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    logger.info(f"Grid Search F1-Macro: {overall_f1:.4f}")

    return thresholds, overall_f1


def bayesian_search_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_trials: int = None,
    seed: int = 42,
) -> Tuple[np.ndarray, float]:
    """
    Bayesian 최적화 (Optuna TPE) 기반 임계값 탐색

    Returns:
        (thresholds, best_f1_macro)
    """
    if not HAS_OPTUNA:
        logger.warning("Optuna not installed, falling back to grid search")
        return grid_search_thresholds(y_true, y_prob)

    if n_trials is None:
        n_trials = ENSEMBLE_CFG.threshold_n_trials

    def objective(trial):
        thresholds = []
        for i in range(NUM_LABELS):
            t = trial.suggest_float(f"threshold_{i}", 0.05, 0.95)
            thresholds.append(t)
        y_pred = (y_prob > np.array(thresholds)).astype(int)
        return f1_score(y_true, y_pred, average="macro", zero_division=0)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_thresholds = np.array([
        study.best_params[f"threshold_{i}"] for i in range(NUM_LABELS)
    ])

    logger.info(f"Bayesian Search F1-Macro: {study.best_value:.4f}")
    for i, col in enumerate(LABEL_COLUMNS):
        logger.info(f"  {col}: {best_thresholds[i]:.3f}")

    return best_thresholds, study.best_value


def optimize_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    method: str = "bayesian",
) -> Tuple[np.ndarray, float]:
    """
    임계값 최적화 통합 인터페이스

    Args:
        method: 'grid' 또는 'bayesian'
    """
    logger.info(f"Threshold optimization: method={method}")
    if method == "bayesian":
        return bayesian_search_thresholds(y_true, y_prob)
    return grid_search_thresholds(y_true, y_prob)
