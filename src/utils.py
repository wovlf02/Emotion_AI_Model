"""
유틸리티 함수 – 시드 고정, 로깅, 체크포인트, 디바이스 감지
"""
import os
import random
import logging
import json
from datetime import datetime

import numpy as np
import torch


def set_seed(seed: int = 42):
    """재현성을 위한 전역 시드 고정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    """사용 가능한 디바이스 반환"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def setup_logger(name: str, log_dir: str = None, level=logging.INFO) -> logging.Logger:
    """파일 + 콘솔 듀얼 로거 설정"""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)

    fmt = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 콘솔
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # 파일
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(
            os.path.join(log_dir, f"{name}_{ts}.log"), encoding="utf-8"
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def save_checkpoint(state: dict, path: str):
    """모델 체크포인트 저장"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, device: torch.device = None) -> dict:
    """모델 체크포인트 로드"""
    return torch.load(path, map_location=device or "cpu", weights_only=False)


def save_json(data: dict, path: str):
    """JSON 파일 저장"""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str) -> dict:
    """JSON 파일 로드"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class EarlyStopping:
    """Early Stopping – patience 기반 학습 조기 종료"""

    def __init__(self, patience: int = 12, min_delta: float = 0.001, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        improved = (
            score > self.best_score + self.min_delta
            if self.mode == "max"
            else score < self.best_score - self.min_delta
        )

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop
