# 06. API 레퍼런스

## 📚 소스 코드 모듈 상세 문서

---

## 1. 개요

본 문서는 `src/` 디렉토리의 각 모듈에 대한 상세 API 레퍼런스를 제공합니다.

```
src/
├── main.py                         # 6-Stage 파이프라인 진입점
├── config.py                       # 전역 설정 (Config dataclasses)
├── utils.py                        # 유틸리티 (시드, 로깅, 체크포인트)
│
├── data/                           # 데이터 처리
│   ├── data_loader.py              # UnSmile 데이터 로딩 / K-Fold 분할
│   ├── dataset.py                  # PyTorch Dataset (HateSpeechDataset)
│   ├── external_data_merger.py     # 7개 외부 데이터셋 병합
│   ├── preprocessing.py            # 텍스트 정규화/정제 파이프라인
│   ├── aeda_augmentation.py        # AEDA 데이터 증강
│   └── verify_datasets.py          # 데이터셋 사전 검증
│
├── models/                         # 모델 아키텍처
│   ├── model.py                    # MultiLabelClassifier (AWP, Multi-Sample Dropout)
│   ├── asymmetric_loss.py          # Asymmetric Loss
│   └── ensemble.py                 # Stacking Meta-Learner (LightGBM+MLP+Ridge)
│
├── training/                       # 학습 파이프라인
│   ├── trainer.py                  # K-Fold 학습 + AWP + R-Drop
│   ├── metrics.py                  # 평가 메트릭 (F1, Hamming Acc 등)
│   ├── optimize_thresholds.py      # Bayesian/Grid 임계값 최적화
│   ├── curriculum_learning.py      # Curriculum Learning 스케줄러
│   ├── hard_negative_mining.py     # Hard Negative Mining + Specialist
│   └── self_training.py            # Self-Training (3-Round Pseudo-Labeling)
│
└── inference/                      # 추론 파이프라인
    ├── inference.py                # TTA + Temperature Scaling + 추론 엔진
    ├── rule_system.py              # 키워드 힌트 + 후처리 보정
    └── error_correction.py         # Error Correction Network (LightGBM)
```

---

## 2. models/model.py

### 2.1 MultiLabelClassifier

다중 라벨 분류를 위한 기본 분류기 클래스입니다.

```python
class MultiLabelClassifier(nn.Module):
    """다중 라벨 분류를 위한 기본 분류기"""
    
    def __init__(
        self,
        model_name: str,           # Hugging Face 모델 이름
        num_labels: int,           # 레이블 개수 (기본: 9)
        dropout_rate: float = 0.3, # 드롭아웃 비율
        use_qlora: bool = False    # QLoRA 사용 여부
    )
```

**주요 메서드:**

| 메서드 | 설명 | 반환값 |
|--------|------|--------|
| `forward(input_ids, attention_mask, labels)` | 순전파 | `{'logits': Tensor}` |
| `freeze_encoder(num_layers_to_unfreeze)` | 인코더 동결 | None |

**사용 예시:**

```python
from src.models.model import MultiLabelClassifier

# 모델 생성
model = MultiLabelClassifier(
    model_name="beomi/KcELECTRA-base",
    num_labels=9,
    dropout_rate=0.3
)

# 순전파
outputs = model(input_ids, attention_mask)
logits = outputs['logits']  # [batch_size, 9]
```

### 2.2 HybridEnsemble

3-모델 하이브리드 앙상블 시스템 클래스입니다.

```python
class HybridEnsemble:
    """3-모델 하이브리드 앙상블 시스템"""
    
    def __init__(
        self,
        num_labels: int,        # 레이블 개수
        device: str = 'cuda'    # 디바이스
    )
```

**주요 메서드:**

| 메서드 | 설명 | 반환값 |
|--------|------|--------|
| `add_model(name, model, weight)` | 모델 추가 | None |
| `load_models()` | 3개 모델 로드 | None |
| `predict(input_ids, attention_mask)` | 앙상블 예측 | Tensor |
| `save_models(save_dir)` | 모델 저장 | None |
| `load_model_weights(load_dir)` | 가중치 로드 | None |

### 2.3 create_model

모델 생성 헬퍼 함수입니다.

```python
def create_model(
    model_name: str,           # Hugging Face 모델 이름
    num_labels: int,           # 레이블 개수
    use_qlora: bool = False,   # QLoRA 사용 여부
    dropout_rate: float = 0.3  # 드롭아웃 비율
) -> nn.Module
```

---

## 3. training/trainer.py

### 3.1 Trainer

모델 학습 및 평가 클래스입니다.

```python
class Trainer:
    """모델 학습 및 평가 클래스"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        device: str,
        model_name: str
    )
```

**주요 메서드:**

| 메서드 | 설명 | 반환값 |
|--------|------|--------|
| `train_epoch()` | 1 에포크 학습 | float (avg_loss) |
| `evaluate(threshold)` | 검증 데이터 평가 | dict (metrics) |
| `train(num_epochs, patience)` | 전체 학습 루프 | float (best_f1) |

**반환되는 메트릭:**

```python
{
    'f1_macro': float,
    'f1_micro': float,
    'exact_match': float,
    'hamming_accuracy': float,
    'probs': np.ndarray,
    'labels': np.ndarray
}
```

### 3.2 train_single_model

단일 모델 학습 함수입니다.

```python
def train_single_model(
    model_config: dict,        # 모델 설정
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    num_epochs: int = 80
) -> Tuple[nn.Module, dict]
```

### 3.3 optimize_ensemble_weights

앙상블 가중치 최적화 함수입니다.

```python
def optimize_ensemble_weights(
    all_probs: List[np.ndarray],  # 각 모델의 예측 확률
    true_labels: np.ndarray,       # 실제 레이블
    num_models: int                # 모델 개수
) -> np.ndarray                    # 최적 가중치 배열
```

### 3.4 optimize_thresholds_per_class

클래스별 최적 임계값 탐색 함수입니다.

```python
def optimize_thresholds_per_class(
    probs: np.ndarray,      # 예측 확률
    labels: np.ndarray,     # 실제 레이블
    num_classes: int        # 클래스 개수
) -> np.ndarray             # 최적 임계값 배열
```

### 3.5 main

메인 학습 파이프라인 함수입니다.

```python
def main() -> dict:
    """
    15시간 최대 정확도 전략 메인 함수
    
    Returns:
        results: 학습 결과 딕셔너리
    """
```

---

## 4. data/data_loader.py

### 4.1 UnsmileDataLoader

UnSmile 데이터셋 로더 클래스입니다.

```python
class UnsmileDataLoader:
    """UnSmile 데이터셋 로더"""
    
    def __init__(self, data_dir: str = "./data")
    
    # 레이블 컬럼 정의
    label_columns = [
        '여성/가족', '남성', '성소수자', '인종/국적',
        '연령', '지역', '종교', '기타 혐오', '악플/욕설'
    ]
```

**주요 메서드:**

| 메서드 | 설명 | 반환값 |
|--------|------|--------|
| `load_raw_data()` | 원본 TSV 로드 | Tuple[DataFrame, DataFrame, DataFrame] |
| `prepare_dataset()` | 데이터 전처리 및 저장 | None |
| `load_processed_data()` | 전처리된 데이터 로드 | Tuple[DataFrame, DataFrame, DataFrame] |

**사용 예시:**

```python
from src.data.data_loader import UnsmileDataLoader

# 데이터 로더 생성
loader = UnsmileDataLoader("./data")

# 전처리 수행
loader.prepare_dataset()

# 전처리된 데이터 로드
train_df, val_df, test_df = loader.load_processed_data()
```

---

## 5. data/dataset.py

### 5.1 UnsmileDataset

UnSmile 다중 라벨 분류 데이터셋 클래스입니다.

```python
class UnsmileDataset(Dataset):
    """UnSmile 다중 라벨 분류 데이터셋"""
    
    def __init__(
        self,
        texts: List[str],          # 텍스트 리스트
        labels: torch.Tensor,      # 레이블 텐서
        tokenizer,                 # Hugging Face 토크나이저
        max_length: int = 128      # 최대 시퀀스 길이
    )
```

**반환 형식:**

```python
{
    'input_ids': torch.Tensor,      # [max_length]
    'attention_mask': torch.Tensor, # [max_length]
    'labels': torch.Tensor          # [num_labels]
}
```

### 5.2 create_dataloaders

데이터로더 생성 함수입니다.

```python
def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer,
    label_columns: List[str],
    batch_size: int = 16,
    max_length: int = 128,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]
```

---

## 6. data/aeda_augmentation.py

### 6.1 AEDAAugmenter

AEDA 증강기 클래스입니다.

```python
class AEDAAugmenter:
    """AEDA 증강기 - 구두점 삽입으로 데이터 증강"""
    
    def __init__(self, punc_ratio: float = 0.3)
    
    # 사용 구두점
    punctuations = ['.', ',', '!', '?', ';', ':']
```

**주요 메서드:**

| 메서드 | 설명 | 반환값 |
|--------|------|--------|
| `augment(text)` | 단일 텍스트 증강 | str |
| `augment_batch(texts, num_aug)` | 배치 증강 | List[str] |

**사용 예시:**

```python
from src.data.aeda_augmentation import AEDAAugmenter

augmenter = AEDAAugmenter(punc_ratio=0.3)

# 단일 텍스트 증강
original = "틀딱들은 집에나 있어라"
augmented = augmenter.augment(original)
# 결과: "틀딱들은, 집에나. 있어라!"
```

### 6.2 augment_minority_classes

소수 클래스 데이터 증강 함수입니다.

```python
def augment_minority_classes(
    train_df: pd.DataFrame,
    label_columns: List[str],
    target_size: int = 2500,    # 목표 샘플 수
    punc_ratio: float = 0.4,    # 구두점 삽입 비율
    augment_all: bool = False   # 모든 클래스 증강 여부
) -> pd.DataFrame
```

---

## 7. models/asymmetric_loss.py

### 7.1 AsymmetricLoss

비대칭 손실 함수 클래스입니다.

```python
class AsymmetricLoss(nn.Module):
    """비대칭 손실 함수 (Asymmetric Loss)"""
    
    def __init__(
        self,
        gamma_neg: float = 4.0,   # 부정 샘플 감쇠율
        gamma_pos: float = 1.0,   # 긍정 샘플 감쇠율
        clip: float = 0.05,       # 부정 샘플 확률 하한값
        eps: float = 1e-8,
        disable_torch_grad_focal_loss: bool = True
    )
```

**사용 예시:**

```python
from src.models.asymmetric_loss import AsymmetricLoss

criterion = AsymmetricLoss(
    gamma_neg=4.0,
    gamma_pos=1.0,
    clip=0.05
)

loss = criterion(logits, labels)
```

### 7.2 AsymmetricLossOptimized

메모리 효율 개선된 비대칭 손실 함수입니다.

```python
class AsymmetricLossOptimized(nn.Module):
    """최적화된 비대칭 손실 함수 (메모리 효율 개선)"""
```

---

## 8. inference/inference.py

### 8.1 load_model_and_predict

단일 모델 로드 및 예측 함수입니다.

```python
def load_model_and_predict(
    model_config: dict,
    test_df: pd.DataFrame,
    label_columns: list,
    device: str
) -> Tuple[np.ndarray, np.ndarray]  # (probs, labels)
```

### 8.2 ensemble_predict

앙상블 예측 함수입니다.

```python
def ensemble_predict(
    models_config: list,
    test_df: pd.DataFrame,
    label_columns: list,
    device: str
) -> Tuple[np.ndarray, np.ndarray]  # (ensemble_probs, labels)
```

### 8.3 apply_optimal_thresholds

최적 임계값 적용 함수입니다.

```python
def apply_optimal_thresholds(
    probs: np.ndarray,
    thresholds: np.ndarray
) -> np.ndarray  # predictions
```

### 8.4 evaluate_predictions

예측 결과 평가 함수입니다.

```python
def evaluate_predictions(
    labels: np.ndarray,
    preds: np.ndarray,
    label_names: list
) -> dict
```

---

## 9. training/optimize_thresholds.py

### 9.1 load_model_and_predict_val

검증 데이터 예측 함수입니다.

```python
def load_model_and_predict_val(
    model_config: dict,
    val_df: pd.DataFrame,
    label_columns: list,
    device: str
) -> Tuple[np.ndarray, np.ndarray]
```

### 9.2 optimize_thresholds_per_class

클래스별 최적 임계값 탐색 함수입니다.

```python
def optimize_thresholds_per_class(
    ensemble_probs: np.ndarray,
    labels: np.ndarray,
    label_columns: list
) -> np.ndarray
```

### 9.3 optimize_thresholds_global

전역 최적 임계값 탐색 함수입니다.

```python
def optimize_thresholds_global(
    ensemble_probs: np.ndarray,
    labels: np.ndarray
) -> np.ndarray
```

---

## 10. 설정 및 하이퍼파라미터

### 10.1 모델 설정

```python
models_config = [
    {
        'name': 'kcelectra',
        'hf_name': 'beomi/KcELECTRA-base',
        'use_qlora': False,
        'role': '슬랭/욕설 전문가'
    },
    {
        'name': 'soongsil',
        'hf_name': 'snunlp/KR-SBERT-V40K-klueNLI-augSTS',
        'use_qlora': False,
        'role': '안정적 베이스라인'
    },
    {
        'name': 'roberta_base',
        'hf_name': 'klue/roberta-base',
        'use_qlora': False,
        'role': '고맥락 의미론 전문가'
    }
]
```

### 10.2 학습 하이퍼파라미터

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| `num_epochs` | 80 | 최대 에포크 수 |
| `batch_size` | 16~32 | 배치 크기 |
| `max_length` | 128 | 최대 시퀀스 길이 |
| `learning_rate` | 2e-5 | 학습률 |
| `weight_decay` | 0.01 | L2 정규화 |
| `dropout_rate` | 0.3 | 드롭아웃 비율 |
| `patience` | 10 | Early Stopping 인내 |
| `warmup_ratio` | 0.1 | Warmup 비율 |

---

**이전 문서**: [05_실험_결과.md](05_실험_결과.md)  
**다음 문서**: [07_트러블슈팅.md](07_트러블슈팅.md)
