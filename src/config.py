"""
전역 설정 및 하이퍼파라미터 관리
"""
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ── 프로젝트 경로 ──────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

# ── 레이블 정의 ────────────────────────────────────────────
LABEL_COLUMNS = [
    "여성/가족",
    "남성",
    "성소수자",
    "인종/국적",
    "연령",
    "지역",
    "종교",
    "기타 혐오",
    "악플/욕설",
]
NUM_LABELS = len(LABEL_COLUMNS)

# ── 5개 베이스 모델 ────────────────────────────────────────
@dataclass
class ModelConfig:
    name: str
    pretrained: str
    batch_size: int = 32
    lr: float = 2e-5
    epochs: int = 60
    early_stopping_patience: int = 12

PHASE2_MODELS: List[ModelConfig] = [
    ModelConfig("kcelectra",    "beomi/KcELECTRA-base"),
    ModelConfig("kcbert",       "beomi/kcbert-base"),
    ModelConfig("klue_bert",    "klue/bert-base"),
    ModelConfig("klue_roberta", "klue/roberta-base"),
    ModelConfig("kr_electra",   "snunlp/KR-ELECTRA-discriminator"),
]

# ── 학습 하이퍼파라미터 ────────────────────────────────────
@dataclass
class TrainConfig:
    # 기본 설정
    seed: int = 42
    num_folds: int = 5
    max_length: int = 128
    gradient_accumulation_steps: int = 4
    fp16: bool = True

    # Optimizer
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Scheduler: Cosine Annealing with Warm Restart
    scheduler_t0: int = 15
    scheduler_t_mult: int = 2

    # 과적합 방지 (6중 안전장치)
    dropout: float = 0.3
    multi_sample_dropout_k: int = 5
    label_smoothing: float = 0.05
    early_stopping_min_delta: float = 0.001

    # AsymmetricLoss
    asl_gamma_neg: float = 4.0
    asl_gamma_pos: float = 0.5
    asl_clip: float = 0.05

    # AWP (Adversarial Weight Perturbation)
    awp_adv_lr: float = 1e-4
    awp_adv_eps: float = 1e-2
    awp_start_epoch: int = 10

    # R-Drop
    rdrop_alpha: float = 4.0
    rdrop_start_epoch: int = 5

    # AEDA
    aeda_target_per_class: int = 10000
    aeda_punc_ratio: float = 0.3

    # Curriculum Learning 구간
    curriculum_easy_end: int = 15
    curriculum_medium_end: int = 35

# ── 데이터 소스 가중치 ─────────────────────────────────────
SOURCE_WEIGHTS: Dict[str, float] = {
    "unsmile":          2.0,
    "kmhas":            1.0,
    "kold":             0.9,
    "beep":             0.8,
    "korean_toxic":     0.7,
    "apeach":           0.6,
    "ko_hateful_memes": 0.5,
    "curse_filtering":  0.7,
}

# ── 매핑 신뢰도 ────────────────────────────────────────────
MAPPING_CONFIDENCE: Dict[str, float] = {
    "unsmile":         1.00,
    "kmhas_direct":    0.95,
    "kold_direct":     0.90,
    "kold_keyword":    0.80,
    "beep_direct":     0.85,
    "beep_keyword":    0.75,
    "korean_toxic":    0.80,
    "curse_filtering": 0.85,
    "apeach_pseudo":   0.70,
    "hateful_pseudo":  0.65,
}

# ── 앙상블 블렌딩 가중치 ───────────────────────────────────
@dataclass
class EnsembleConfig:
    # Level 1 Meta-Learner weights
    meta_lgbm_weight: float = 0.5
    meta_mlp_weight: float = 0.3
    meta_ridge_weight: float = 0.2

    # Level 2 Final Blending weights
    blend_meta: float = 0.60
    blend_best_single: float = 0.20
    blend_keyword: float = 0.10
    blend_ecn: float = 0.10

    # LightGBM 파라미터
    lgbm_num_leaves: int = 63
    lgbm_learning_rate: float = 0.05
    lgbm_feature_fraction: float = 0.8
    lgbm_bagging_fraction: float = 0.8
    lgbm_n_estimators: int = 1000
    lgbm_early_stopping: int = 50

    # MLP Meta-Learner
    mlp_hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    mlp_dropout: float = 0.3
    mlp_lr: float = 1e-3
    mlp_epochs: int = 100

    # Bayesian Threshold
    threshold_n_trials: int = 2000

# ── 기본 인스턴스 ──────────────────────────────────────────
TRAIN_CFG = TrainConfig()
ENSEMBLE_CFG = EnsembleConfig()


# ── Curriculum Learning 설정 ───────────────────────────────
@dataclass
class CurriculumConfig:
    enabled: bool = True
    total_epochs: int = 60
    easy_end_epoch: int = 15
    medium_end_epoch: int = 35
    difficulty_calibration_epochs: int = 3
    easy_threshold: float = 0.33
    hard_threshold: float = 0.67
    easy_confidence_min: float = 0.95
    medium_confidence_min: float = 0.70
    hard_confidence_min: float = 0.70


# ── Hard Negative Mining 설정 ──────────────────────────────
@dataclass
class HardNegativeMiningConfig:
    enabled: bool = True
    num_rounds: int = 2
    fn_detection_threshold: float = 0.5
    persistent_fn_threshold: float = 0.3
    fn_oversample_ratio: float = 3.0
    batch_fn_ratio: float = 0.4
    round1_epochs: int = 30
    specialist_epochs: int = 30
    specialist_targets: List[str] = field(
        default_factory=lambda: ["기타 혐오", "연령"]
    )


# ── Self-Training 설정 ─────────────────────────────────────
@dataclass
class SelfTrainingConfig:
    enabled: bool = True
    num_rounds: int = 3
    confidence_thresholds: List[float] = field(
        default_factory=lambda: [0.95, 0.92, 0.90]
    )
    max_entropy: float = 2.0
    min_model_agreement: float = 0.60
    max_positive_ratio: float = 0.5
    noisy_dropout: float = 0.4
    noisy_aeda_punc_ratio: float = 0.4
    training_epochs: int = 30
    training_batch_size: int = 32
    training_lr: float = 2e-5


# ── Error Correction Network 설정 ──────────────────────────
@dataclass
class ErrorCorrectionConfig:
    enabled: bool = True
    correction_strength: float = 0.1
    alpha_reduced: float = 0.05
    min_correction_abs: float = 0.1
    max_correction_magnitude: float = 0.5
    lgb_num_leaves: int = 31
    lgb_learning_rate: float = 0.05
    lgb_feature_fraction: float = 0.8
    lgb_bagging_fraction: float = 0.8
    lgb_n_estimators: int = 500
    lgb_early_stopping: int = 20


CURRICULUM_CFG = CurriculumConfig()
HNM_CFG = HardNegativeMiningConfig()
ST_CFG = SelfTrainingConfig()
ECN_CFG = ErrorCorrectionConfig()


def ensure_dirs():
    """필수 디렉토리 생성"""
    for d in [PROCESSED_DIR, MODELS_DIR, LOGS_DIR]:
        os.makedirs(d, exist_ok=True)
