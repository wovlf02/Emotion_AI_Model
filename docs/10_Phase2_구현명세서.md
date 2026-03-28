# 10. Phase 2 구현 명세서

## 📋 F1-Macro 97~98% 달성을 위한 구현 설계서

> 본 문서는 Phase 2의 **구현 명세(Specification)**입니다.  
> 전체 전략은 [08_성능개선_로드맵.md](08_성능개선_로드맵.md), 데이터셋 상세는 [09_외부데이터셋_명세.md](09_외부데이터셋_명세.md)를 참고하세요.  
> 전처리 상세 설계는 [11_전처리_상세설계.md](11_전처리_상세설계.md)를 참고하세요.

---

## 1. 구현 범위 및 아키텍처

### 1.1 구현해야 할 모듈

```
구현 대상 모듈 구조 (6-Stage 아키텍처)
═══════════════════════════════════════════════════════════════════
  src/
  ├── data/
  │   ├── external_data_merger.py    ← 외부 데이터셋 병합기 (7개 데이터셋)
  │   ├── preprocessing.py           ← 전처리 파이프라인 (정규화/정제)
  │   ├── aeda_augmentation.py       ← AEDA 데이터 증강
  │   └── verify_datasets.py         ← 데이터셋 사전 검증 스크립트
  │
  ├── models/
  │   ├── model.py                   ← MultiLabelClassifier (5-모델 × 5-Fold)
  │   ├── asymmetric_loss.py         ← Asymmetric Loss
  │   └── ensemble.py                ← Stacking Meta-Learner (LightGBM/MLP/Ridge)
  │
  ├── training/
  │   ├── trainer.py                 ← K-Fold 학습 + AWP + R-Drop
  │   ├── optimize_thresholds.py     ← Bayesian Threshold (Optuna)
  │   ├── curriculum_learning.py     ← Curriculum Learning 스케줄러
  │   ├── hard_negative_mining.py    ← Hard Negative Mining + Specialist
  │   └── self_training.py           ← Self-Training (3라운드 pseudo-labeling)
  │
  ├── inference/
  │   ├── inference.py               ← TTA + Temperature Scaling + 추론 엔진
  │   ├── rule_system.py             ← 룰 기반 보정 시스템
  │   └── error_correction.py        ← ECN (오류 보정 네트워크)
  │
  ├── main.py                        ← Phase 2 통합 6-Stage 파이프라인
  ├── config.py                      ← 전역 설정 (Config dataclasses)
  └── utils.py                       ← 유틸리티 (시드, 로깅, 체크포인트)
═══════════════════════════════════════════════════════════════════
```

### 1.2 전체 실행 흐름 (6-Stage)

```
6-Stage 파이프라인
═══════════════════════════════════════════════════════════════════

  [사전 검증] → [Stage A] → [Stage B] → [Stage C] → [Stage D] → [Stage E] → [Stage F] → [최종 평가]
       │            │            │            │            │            │            │           │
       ▼            ▼            ▼            ▼            ▼            ▼            ▼           ▼
   verify       전처리+병합   5×5 K-Fold   Curriculum   Self-Train   Rule+ECN   TTA+Calib    evaluate
   datasets     + 증강       + Stacking   + AWP+R-Drop  (3 rounds)  (보정)     + Bayesian   (test set)
               (190K건)     (25 모델)    + Multi-Task   (pseudo)    (3-level)  (최적화)

  예상 F1:    ─         87-89%     91-93%     94-96%     95-97%     97-98%     98-99%
═══════════════════════════════════════════════════════════════════
```

---

## 2. 모듈 ①: 데이터셋 사전 검증 (`verify_datasets.py`)

### 2.1 기능 명세

| 항목 | 내용 |
|------|------|
| **목적** | Phase 2 실행 전 필수 라이브러리 및 외부 데이터셋 접근 가능 여부 확인 |
| **입력** | 없음 (자동 검증) |
| **출력** | 콘솔 출력 (데이터셋별 접근 상태, 샘플 수, 컬럼 정보) |
| **종료 코드** | 0: 전체 성공, 1: 일부 실패 |

### 2.2 검증 항목

```python
# 1단계: 필수 라이브러리 버전 확인
REQUIRED_PACKAGES = {
    'torch': '2.0+',
    'transformers': '4.30+',
    'datasets': '2.14+',
    'pandas': '1.5+',
    'numpy': '1.24+',
    'scikit-learn': '1.2+',
    'scipy': '1.10+',
}

# 2단계: 외부 데이터셋 접근 확인 (streaming=True로 50건만 샘플)
DATASETS_TO_VERIFY = [
    ('jeanlee/kmhas_korean_hate_speech', None,    'K-MHaS'),
    ('nayohan/KOLD',                     None,    'KOLD'),
    ('nayohan/korean-hate-speech',       None,    'Korean Hate Speech'),
    ('jason9693/APEACH',                 None,    'APEACH'),
    ('josephnam/korean_toxic_datasets',  None,    'Korean Toxic Datasets'),
    ('2tle/korean-curse-filtering-dataset', None, 'Korean Curse Filtering'),
    # Ko-HatefulMemes는 멀티모달+영어혼재로 선택적 검증
]

# 3단계: UnSmile 원본 데이터 존재 확인
UNSMILE_FILES = [
    'data/raw/unsmile_train.tsv',
    'data/raw/unsmile_dev.tsv',
    'data/raw/unsmile_test.tsv',
]
```

### 2.3 예상 출력 형식

```
[1] 필수 라이브러리 확인
  ✅ torch: 2.x.x
  ✅ transformers: 4.x.x
  ✅ datasets: 2.x.x
  ...

[2] 외부 데이터셋 접근 확인 (각 50건 샘플)
  ✅ K-MHaS: 50건 로드 성공 | 컬럼: ['text', 'label']
  ✅ KOLD: 50건 로드 성공 | 컬럼: ['comment', 'OFF', 'GRP', ...]
  ✅ Korean Hate Speech: 50건 로드 성공 | 컬럼: ['comments', 'hate', ...]
  ✅ APEACH: 50건 로드 성공 | 컬럼: ['text', 'class']

[3] UnSmile 원본 데이터 확인
  ✅ data/raw/unsmile_train.tsv 존재

[4] 검증 요약
  🎉 모든 데이터셋 접근 가능! Phase 2 실행 준비 완료.
```

---

## 3. 모듈 ②: 외부 데이터 병합기 (`src/external_data_merger.py`)

### 3.1 기능 명세

| 항목 | 내용 |
|------|------|
| **목적** | 7개 외부 데이터셋을 UnSmile 9-label 형식으로 변환 후 병합 |
| **입력** | HuggingFace 데이터셋 (자동 다운로드) + UnSmile 원본 |
| **출력** | `data/processed/train_phase2.csv` (병합 완료 학습 데이터) |
| **핵심 인터페이스** | `ExternalDataMerger` 클래스 |

### 3.2 클래스 설계

```python
class ExternalDataMerger:
    """외부 데이터셋 → UnSmile 형식 변환 + 병합"""
    
    UNSMILE_LABELS = [
        '여성/가족', '남성', '성소수자', '인종/국적',
        '연령', '지역', '종교', '기타 혐오', '악플/욕설'
    ]
    
    def __init__(self, cache_dir: str = './data/hf_cache'):
        """HuggingFace 캐시 디렉토리 설정"""
    
    def load_and_convert_kmhas(self) -> pd.DataFrame:
        """K-MHaS 데이터셋 로드 + UnSmile 레이블 변환"""
    
    def load_and_convert_kold(self) -> pd.DataFrame:
        """KOLD 데이터셋 로드 + UnSmile 레이블 변환"""
    
    def load_and_convert_beep(self) -> pd.DataFrame:
        """Korean Hate Speech(BEEP!) 로드 + UnSmile 레이블 변환"""
    
    def load_and_convert_apeach(self, phase1_model=None) -> pd.DataFrame:
        """APEACH 로드 + pseudo-labeling으로 UnSmile 변환"""
    
    def load_and_convert_korean_toxic(self) -> pd.DataFrame:
        """Korean Toxic Datasets 로드 + src_path/level2_type 기반 UnSmile 변환"""
    
    def load_and_convert_curse_filtering(self) -> pd.DataFrame:
        """Korean Curse Filtering 로드 + 욕설 태그 파싱 + UnSmile 변환"""
    
    def merge_all(self, unsmile_train_path: str) -> pd.DataFrame:
        """전체 데이터셋 병합 + 중복 제거 + 품질 필터링"""
    
    def save(self, df: pd.DataFrame, output_path: str):
        """병합 결과 CSV 저장"""
```

### 3.3 레이블 매핑 상세 로직

#### K-MHaS → UnSmile

```python
KMHAS_LABEL_MAP = {
    0: '지역',       # origin
    1: '기타 혐오',  # physical (신체/외모 혐오)
    2: '기타 혐오',  # politics (정치 혐오)
    3: '악플/욕설',  # profanity
    4: '연령',       # age
    5: None,          # gender → 키워드 기반 여성/남성 분리
    6: '인종/국적',  # race
    7: '종교',       # religion
    8: None,          # not_hate_speech → Clean (모두 0)
}

def convert_kmhas_gender(text: str) -> str:
    """gender 레이블을 텍스트 키워드 기반으로 여성/남성 분리"""
    FEMALE_KW = ['김치녀', '맘충', '여자', '여성', '페미', '보슬', '한녀', '된장녀', '메갈']
    MALE_KW = ['한남', '남자', '남성', '자지', '남충']
    
    has_female = any(kw in text for kw in FEMALE_KW)
    has_male = any(kw in text for kw in MALE_KW)
    
    if has_female and not has_male:
        return '여성/가족'
    elif has_male and not has_female:
        return '남성'
    else:
        # 둘 다 감지 또는 미감지 → 양쪽 모두 활성화
        return 'both'
```

#### KOLD → UnSmile

```python
def convert_kold_target_group(raw_labels: list) -> dict:
    """KOLD raw_labels의 target_group 패턴 매칭으로 UnSmile 레이블 결정"""
    
    TARGET_GROUP_MAP = {
        '집단-성 정체성-여성':      '여성/가족',
        '집단-성 정체성-페미니스트': '여성/가족',
        '집단-성 정체성-남성':      '남성',
        '집단-성 정체성-성소수자':  '성소수자',
        '집단-국적/인종/민족':      '인종/국적',
        '집단-연령':                '연령',
        '집단-지역':                '지역',
        '집단-종교':                '종교',
    }
    
    # raw_labels에서 target_group 추출 후 패턴 매칭
    # OFF=False → 모두 0 (Clean)
    # OFF=True + 매핑 불가 → 기타 혐오
    # OFF=True + 욕설 키워드 탐지 → 악플/욕설 추가
```

#### Korean Hate Speech (BEEP!) → UnSmile

```python
def convert_beep(row: dict) -> dict:
    """BEEP! 데이터의 hate, gender_bias, bias 필드 조합으로 매핑"""
    
    labels = {label: 0 for label in UNSMILE_LABELS}
    
    if row['hate'] == 'none':
        return labels  # Clean
    
    # hate 또는 offensive → 악플/욕설
    if row['hate'] in ('hate', 'offensive'):
        labels['악플/욕설'] = 1
    
    # gender_bias → 텍스트 키워드로 여성/남성 분리
    if row['contain_gender_bias']:
        gender = detect_gender_from_text(row['comments'])
        if gender in ('여성/가족', 'both'):
            labels['여성/가족'] = 1
        if gender in ('남성', 'both'):
            labels['남성'] = 1
    
    # bias='others' + hate='hate' → 기타 혐오
    if row['bias'] == 'others' and row['hate'] == 'hate':
        labels['기타 혐오'] = 1
    
    return labels
```

#### APEACH → UnSmile (Pseudo-Labeling)

```python
def convert_apeach_pseudo(texts: list, phase1_model) -> pd.DataFrame:
    """
    APEACH pseudo-labeling 전략:
    1. class=1 (Spoiled) 샘플만 추출
    2. Phase 1 앙상블 모델로 9차원 확률 예측
    3. confidence ≥ 0.9인 레이블만 채택 (높은 신뢰도만)
    4. 9차원 모두 0이 되는 샘플은 제외
    
    주의: Phase 1 모델이 없으면 APEACH 데이터는 건너뜀
    """
```

#### Korean Toxic Datasets → UnSmile

```python
def convert_korean_toxic(row: dict) -> dict:
    """Korean Toxic Datasets의 src_path + level2_type 기반 매핑"""
    
    labels = {label: 0 for label in UNSMILE_LABELS}
    src = row['src_path']
    l2 = row.get('level2_type', '')
    text = row['instruct_text']
    
    # 활용 대상 카테고리만 처리
    if src not in ('/01.비난혐오차별/', '/03.욕설/', '/04.폭력/'):
        return None  # 범죄, 허위정보, 스팸 등 제외
    
    if src == '/03.욕설/':
        labels['악플/욕설'] = 1
    elif src == '/04.폭력/':
        labels['기타 혐오'] = 1
    elif src == '/01.비난혐오차별/':
        # level2_type 기반 세분류 매핑
        L2_MAP = {
            '성별': '여성/가족',     # 텍스트 키워드로 여성/남성 분리
            '인종': '인종/국적',
            '외국인': '인종/국적',
            '연령': '연령',
            '나이': '연령',
            '지역': '지역',
            '종교': '종교',
            '성소수자': '성소수자',
            '동성애': '성소수자',
            '장애': '기타 혐오',
            '신체': '기타 혐오',
        }
        mapped = False
        for keyword, target in L2_MAP.items():
            if keyword in l2:
                labels[target] = 1
                mapped = True
                break
        if not mapped:
            labels['기타 혐오'] = 1  # 매핑 불가 → 기타 혐오
    
    # 플레이스홀더 토큰 제거
    import re
    text = re.sub(r'\[[\w]+\]', '', text).strip()
    
    # 욕설 키워드 추가 체크
    if detect_profanity(text):
        labels['악플/욕설'] = 1
    
    return labels
```

#### Korean Curse Filtering → UnSmile

```python
def convert_curse_filtering(row: dict) -> tuple:
    """Korean Curse Filtering Dataset 파싱 + 매핑"""
    
    raw = row['text']
    parts = raw.rsplit('|', maxsplit=1)
    sentence = parts[0].strip()
    curse_tags = parts[1].strip() if len(parts) > 1 else ''
    
    labels = {label: 0 for label in UNSMILE_LABELS}
    
    if curse_tags:
        labels['악플/욕설'] = 1
        # 욕설 태그를 키워드 사전에 추가 (용도 2: 사전 확장)
        tags = [t.strip() for t in curse_tags.split(',')]
        # 각 태그에 대해 추가 카테고리 매핑 시도
        for tag in tags:
            category = classify_curse_tag(tag)
            if category:
                labels[category] = 1
    
    return sentence, labels

def classify_curse_tag(tag: str) -> str:
    """욕설 태그를 추가 혐오 카테고리로 분류 (해당 시)"""
    GENDER_F = ['김치녀', '보슬', '된장녀']
    GENDER_M = ['한남', '남충']
    if any(kw in tag for kw in GENDER_F):
        return '여성/가족'
    if any(kw in tag for kw in GENDER_M):
        return '남성'
    return None  # 순수 욕설 → 추가 카테고리 없음
```

### 3.4 품질 필터링

```python
def quality_filter(df: pd.DataFrame) -> pd.DataFrame:
    """병합 후 품질 필터링"""
    
    # 1. 빈 텍스트 제거
    df = df[df['text'].str.strip().str.len() > 0]
    
    # 2. 극단적 단문 제거 (5자 미만)
    df = df[df['text'].str.len() >= 5]
    
    # 3. 중복 텍스트 제거 (첫 번째 등장만 유지)
    df = df.drop_duplicates(subset='text', keep='first')
    
    # 4. 레이블 충돌 검사
    #    동일 텍스트에 서로 다른 레이블이 배정된 경우 → 다수결 또는 제거
    
    # 5. 최종 검증
    assert df['text'].notna().all()
    assert (df[UNSMILE_LABELS].sum(axis=1) >= 0).all()
    
    return df
```

### 3.5 데이터 분할 전략

```
병합 데이터 분할 원칙
─────────────────────────────────────────────────
  • UnSmile 원본 test.csv는 절대 변경하지 않음
    → Phase 1과 동일 기준 비교 보장
  
  • 외부 데이터는 train + val에만 투입
    → 외부 데이터의 90%: train_phase2
    → 외부 데이터의 10%: val_phase2
  
  • UnSmile 원본 train은 전체 포함 (가중치 ×2)
    → 타겟 도메인 분포 유지 목적
─────────────────────────────────────────────────
```

---

## 4. 모듈 ③: 5-모델 앙상블 학습

### 4.1 모델 구성

| # | 모델명 | HuggingFace ID | 파라미터 | 역할 | Phase 1 대비 |
|---|--------|---------------|----------|------|-------------|
| 1 | **KcELECTRA** | `beomi/KcELECTRA-base` | ~110M | 슬랭/욕설 전문가 | 유지 |
| 2 | **KcBERT** | `beomi/kcbert-base` | ~110M | 뉴스 댓글 전문가 | **신규** |
| 3 | **KLUE-BERT** | `klue/bert-base` | ~110M | 범용 베이스라인 | **신규** |
| 4 | **KLUE-RoBERTa** | `klue/roberta-base` | ~110M | 고맥락 의미론 | 유지 |
| 5 | **KR-ELECTRA** | `snunlp/KR-ELECTRA-discriminator` | ~110M | 판별 전문가 | **신규** |

> **SoongsilBERT** (`soongsil-ai/soongsil-bert-base`)는 Phase 1에서 사용되었으나,  
> Phase 2에서는 KcBERT, KLUE-BERT, KR-ELECTRA로 교체하여 **모델 다양성 강화**.

### 4.2 모델 선정 근거

```
모델 다양성 분석
─────────────────────────────────────────────────────────────
  아키텍처 다양성:
  ├── BERT 계열:  KcBERT, KLUE-BERT       (Masked LM)
  ├── RoBERTa 계열: KLUE-RoBERTa           (Dynamic Masking)
  └── ELECTRA 계열: KcELECTRA, KR-ELECTRA  (Discriminator)

  학습 데이터 다양성:
  ├── 온라인 댓글 특화: KcELECTRA (나무위키+뉴스+댓글)
  ├── 뉴스 댓글 특화:   KcBERT (네이버 뉴스 댓글 수억건)
  ├── 범용 한국어:       KLUE-BERT (다양한 한국어 벤치마크)
  ├── 대규모 코퍼스:     KLUE-RoBERTa (뉴스+위키+도서)
  └── 한국어 특화:       KR-ELECTRA (다양한 한국어 데이터)

  앙상블 효과 극대화:
  → 아키텍처 × 학습데이터 조합이 모두 다르면 앙상블 시 상보적 효과 ↑
  → Phase 1 (3모델 +3.41%p) → Phase 2 (5모델 +5~7%p 예상)
─────────────────────────────────────────────────────────────
```

### 4.3 학습 하이퍼파라미터

| 항목 | Phase 1 | Phase 2 | 변경 이유 |
|------|---------|---------|-----------|
| **학습 데이터** | 15,005건 | ~204,000건 | 14배 확대 |
| **모델 수** | 3개 | **5개** | 다양성 확대 |
| **Epoch** | 80 | **60** | 데이터 충분, 과적합 방지 |
| **Batch Size** | 32 | **32~64** | 큰 데이터에 맞춤 (GPU 여유 시 64) |
| **Learning Rate** | 2e-5 | **2e-5** | 유지 |
| **Weight Decay** | 0.01 | **0.01** | 유지 |
| **손실 함수** | BCEWithLogitsLoss | **AsymmetricLoss** | 불균형 처리 강화 |
| **AEDA 증강 목표** | 2,500 | **10,000** | 소수 클래스 추가 강화 |
| **Early Stopping** | patience=10 | **patience=12** | 충분한 수렴 대기 |
| **Max Length** | 128 | **128** | 유지 (UnSmile 특성 보존) |
| **Dropout** | 0.3 | **0.3** | 유지 |
| **Mixed Precision** | FP16 | **FP16** | 유지 |

### 4.4 AsymmetricLoss 명세

```python
class AsymmetricLossOptimized:
    """
    비대칭 손실 함수: 긍정/부정 샘플에 다른 감쇠 계수 적용
    
    특징:
    - 긍정 샘플(혐오): 낮은 gamma → 학습 강조
    - 부정 샘플(Clean): 높은 gamma → 쉬운 샘플 감쇠
    - 불균형 데이터에서 소수 클래스 학습 효과 극대화
    
    권장 하이퍼파라미터:
    - gamma_neg = 4.0  (부정 샘플 감쇠, 높을수록 쉬운 부정 샘플 무시)
    - gamma_pos = 0.5  (긍정 샘플 감쇠, 낮을수록 긍정 샘플 강조)
    - clip = 0.05      (부정 확률 하한 클리핑)
    """
    
    def __init__(self, gamma_neg=4.0, gamma_pos=0.5, clip=0.05):
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
    
    def forward(self, logits, targets):
        """
        logits:  (batch_size, 9) — 모델 raw output
        targets: (batch_size, 9) — 0/1 multi-hot labels
        
        수식:
          p = sigmoid(logits)
          L_pos = targets × (1-p)^gamma_pos × log(p)
          p_neg = max(p, clip)
          L_neg = (1-targets) × p_neg^gamma_neg × log(1-p_neg)
          Loss = -(L_pos + L_neg).mean()
        """
```

### 4.5 앙상블 추론 파이프라인

```
5-모델 앙상블 추론 흐름
═══════════════════════════════════════════════════════════

  입력 텍스트 "틀딱충들 한남충 꺼져"
       │
       ├──▶ KcELECTRA   → P₁ = [0.1, 0.7, 0.0, 0.1, 0.8, 0.2, 0.0, 0.2, 0.9]
       ├──▶ KcBERT       → P₂ = [0.1, 0.6, 0.0, 0.0, 0.7, 0.1, 0.0, 0.3, 0.8]
       ├──▶ KLUE-BERT    → P₃ = [0.0, 0.5, 0.0, 0.1, 0.6, 0.2, 0.1, 0.2, 0.7]
       ├──▶ KLUE-RoBERTa → P₄ = [0.1, 0.7, 0.0, 0.0, 0.9, 0.1, 0.0, 0.3, 0.9]
       └──▶ KR-ELECTRA   → P₅ = [0.0, 0.6, 0.0, 0.1, 0.7, 0.2, 0.0, 0.2, 0.8]
                │
                ▼
       가중 소프트 보팅: P_final = Σ(wᵢ × Pᵢ)
                │          가중치 wᵢ: Nelder-Mead 최적화로 결정
                ▼
       [룰 시스템 보정]  ← Stage C (선택적)
                │
                ▼
       클래스별 최적 임계값 적용
                │
                ▼
       최종 예측: [0, 1, 0, 0, 1, 0, 0, 0, 1]
                    남성    연령         악플/욕설
```

### 4.6 가중치 최적화

```python
def optimize_ensemble_weights(model_probs: list, true_labels: np.ndarray):
    """
    Nelder-Mead 알고리즘으로 최적 앙상블 가중치 탐색
    
    목적함수: -F1_Macro(weighted_sum(probs, weights), true_labels)
    제약조건: weights >= 0, sum(weights) = 1
    
    입력:
      model_probs: [P₁, P₂, P₃, P₄, P₅]  각 (N, 9)
      true_labels: (N, 9)
    
    출력:
      optimal_weights: [w₁, w₂, w₃, w₄, w₅]
    
    Phase 1 참고값 (3모델):
      KcELECTRA=0.40, SoongsilBERT=0.30, KLUE-RoBERTa=0.30
    
    Phase 2 초기 추정값 (5모델):
      KcELECTRA=0.25, KcBERT=0.20, KLUE-BERT=0.15,
      KLUE-RoBERTa=0.25, KR-ELECTRA=0.15
    """
```

---

## 5. 모듈 ④: 룰 시스템 (`src/rule_system.py`)

### 5.1 시스템 개요

룰 시스템은 모델 추론 **후** 확률을 보정하는 후처리 파이프라인입니다.

```
처리 흐름
═══════════════════════════════════════════════════════════

  원본 텍스트
      │
      ▼
  [TextNormalizer]  ── 난독화 표현 정규화
      │                 "10발" → "씨발", "ㅅㅂ" → "씨발"
      │                 "개*새" → "개새", "느금" → "느그엄마"
      ▼
  [KeywordHintGenerator]  ── 레이블별 키워드 스캔
      │                       강한 키워드(high confidence) → hint = 0.6
      │                       약한 키워드(low confidence)  → hint = 0.25
      │                       키워드 없음                    → hint = 0.0
      ▼
  [PostProcessingCorrector]  ── 모델 확률 + 힌트 혼합
      │
      │   final = (1 - blend_weight) × model_prob + blend_weight × hint
      │   기본 blend_weight = 0.3 (조절 가능)
      │
      │   상호 강화 규칙 (Cross-label Reinforcement):
      │   • 여성/가족 > 0.4 → 악플/욕설 += 0.15
      │   • 남성 > 0.4 → 악플/욕설 += 0.15
      │   • 인종/국적 > 0.4 → 악플/욕설 += 0.12
      │   (혐오 카테고리 탐지 시 동반되는 욕설도 함께 강화)
      ▼
  보정된 확률 → 임계값 적용 → 최종 예측
```

### 5.2 TextNormalizer 명세

```python
class TextNormalizer:
    """인터넷 난독화 표현을 원형으로 정규화"""
    
    NORMALIZATION_RULES = {
        # 초성 약어
        'ㅅㅂ': '씨발', 'ㅂㅅ': '병신', 'ㅈㄹ': '지랄',
        'ㅆㅂ': '씨발', 'ㅄ': '병신',
        
        # 숫자/기호 치환
        '10발': '씨발', '18놈': '씨발놈',
        'ㅅ1발': '씨발',
        
        # 자음 반복 (ㅋㅋㅋ, ㅎㅎㅎ 등은 유지)
        # 의미 변형만 정규화
        
        # 별표/특수문자 마스킹
        # '씨*' → '씨발', '개*끼' → '개새끼' (정규식 패턴)
    }
    
    def normalize(self, text: str) -> str:
        """텍스트 정규화 (원본 보존, 정규화 버전 반환)"""
```

### 5.3 KeywordHintGenerator 명세

```python
class KeywordHintGenerator:
    """레이블별 키워드 사전 기반 힌트 벡터 생성"""
    
    # 강한 키워드: 단독으로 해당 레이블을 강하게 시사
    # 약한 키워드: 맥락에 따라 해당 레이블을 약하게 시사
    
    KEYWORD_DICT = {
        '여성/가족': {
            'strong': ['김치녀', '맘충', '보슬아치', '페미충', '된장녀', '메갈리아', '워마드'],
            'weak':   ['페미', '여성혐오', '여자들', '아줌마'],
        },
        '남성': {
            'strong': ['한남충', '남충', '자지충', '한남유충'],
            'weak':   ['한남', '남성혐오', '남자들'],
        },
        '성소수자': {
            'strong': ['게이새끼', '레즈비언년', '트랜스젠더충'],
            'weak':   ['게이', '레즈', '트젠', '동성애', '호모'],
        },
        '인종/국적': {
            'strong': ['짱깨', '쪽발이', '왜놈', '죠센징', '깜둥이'],
            'weak':   ['조선족', '외국인노동자', '혼혈', '다문화'],
        },
        '연령': {
            'strong': ['틀딱충', '급식충', '노인충', '꼰대충'],
            'weak':   ['틀딱', '꼰대', '늙다리', '급식'],
        },
        '지역': {
            'strong': ['홍어새끼', '전라충', '경상충'],
            'weak':   ['홍어', '빨갱이', '촌놈', '쥐라도'],
        },
        '종교': {
            'strong': ['개독충', '예수충', '기독충'],
            'weak':   ['개독', '사이비', '이단', '맹신'],
        },
        '기타 혐오': {
            'strong': ['찐따새끼', '루저새끼', '장애충'],
            'weak':   ['찐따', '루저', '열폭', '헬조선', '인셀'],
        },
        '악플/욕설': {
            'strong': ['씨발', '개새끼', '병신', '미친놈', '지랄'],
            'weak':   ['쓰레기', '바보', '멍청이', '꺼져', '닥쳐'],
        },
    }
    
    STRONG_HINT = 0.6   # 강한 키워드 탐지 시 hint 값
    WEAK_HINT = 0.25    # 약한 키워드 탐지 시 hint 값
    
    def generate_hints(self, normalized_text: str) -> dict:
        """
        반환: {'여성/가족': 0.6, '남성': 0.0, ..., '악플/욕설': 0.25}
        규칙: 강한 > 약한, 복수 키워드 시 최대값 사용
        """
```

### 5.4 PostProcessingCorrector 명세

```python
class PostProcessingCorrector:
    """모델 확률 + 키워드 힌트 혼합 + 상호 강화 규칙"""
    
    # 상호 강화 규칙: 특정 혐오 카테고리 탐지 시 관련 카테고리 부스트
    CROSS_LABEL_RULES = [
        # (조건 레이블, 조건 임계값, 부스트 대상, 부스트 양)
        ('여성/가족', 0.4, '악플/욕설', 0.15),  # 여성혐오 → 욕설 동반 확률 높음
        ('남성',     0.4, '악플/욕설', 0.15),    # 남성혐오 → 욕설 동반 확률 높음
        ('인종/국적', 0.4, '악플/욕설', 0.12),   # 인종차별 → 욕설 동반 확률 높음
        ('성소수자', 0.4, '악플/욕설', 0.10),    # 성소수자 혐오 → 욕설 동반
        ('연령',     0.5, '악플/욕설', 0.10),    # 연령 혐오 → 욕설 동반
    ]
    
    def correct(self, texts: list, model_probs: np.ndarray,
                blend_weight: float = 0.3) -> np.ndarray:
        """
        1단계: 텍스트 정규화
        2단계: 키워드 힌트 생성
        3단계: (1-w) × model_prob + w × hint 혼합
        4단계: 상호 강화 규칙 적용
        5단계: [0, 1] 범위 클리핑
        
        반환: 보정된 확률 (N, 9)
        """
```

### 5.5 blend_weight 튜닝 가이드

| blend_weight | 동작 | 권장 상황 |
|-------------|------|----------|
| 0.0 | 룰 미사용 (모델 확률만) | Stage B 이후 F1 ≥ 96% |
| 0.1~0.2 | 약한 보정 | 모델 신뢰도 높지만 특정 클래스 취약 |
| **0.3** | **기본값 (권장)** | **일반적 상황** |
| 0.4~0.5 | 강한 보정 | 기타 혐오, 악플/욕설 F1이 여전히 낮을 때 |
| > 0.5 | 룰 우선 (비권장) | 모델 성능이 매우 낮은 초기 단계 |

### 5.6 적용 판단 기준

```
Stage B 완료 후 F1-Macro에 따른 Stage C 적용 전략
─────────────────────────────────────────────────
  F1 ≥ 96%  → 룰 시스템 불필요 (blend_weight=0)
  F1 93~96% → 취약 클래스만 선택적 적용
              (기타 혐오, 악플/욕설 중심 키워드만 활성화)
  F1 < 93%  → 전면 적용 (blend_weight=0.3)
─────────────────────────────────────────────────
```

---

## 6. 모듈 ⑤: Stacking Meta-Learner (`src/stacking_meta_learner.py`)

### 6.1 기능 명세

| 항목 | 설명 |
|------|------|
| **역할** | 25개 Base 모델의 OOF 예측을 비선형 결합하여 최종 확률 생성 |
| **입력** | OOF 예측 행렬 (N × 25 × 9) + 보조 특성 |
| **출력** | Meta 확률 (N × 9) |
| **학습** | Base 모델 학습 완료 후 OOF 예측으로 학습 |

### 6.2 OOF 예측 생성 프로세스

```
OOF (Out-of-Fold) 예측 생성
═══════════════════════════════════════════════════════════

  K-Fold 학습 결과물:
  ─────────────────────────────────────────────────
    models/fold_0/kcelectra.pt  → val 예측 (fold 0 val set)
    models/fold_0/kcbert.pt     → val 예측 (fold 0 val set)
    ...
    models/fold_4/kr_electra.pt → val 예측 (fold 4 val set)
    
  결합하면:
    oof_predictions.npy  shape: (N_train, 25, 9)
    → N_train 전체 학습 데이터에 대한 "미학습" 예측
    → Meta-Learner 학습용 X_train
═══════════════════════════════════════════════════════════
```

### 6.3 Meta-Feature 설계 (294차원)

```python
META_FEATURES = {
    # 카테고리 1: Base model 확률 (225차원)
    'base_probs':      '25 models × 9 classes = 225',
    
    # 카테고리 2: 통계 특성 (45차원)
    'per_class_stats':  '9 classes × 5 stats (mean/std/max/min/median) = 45',
    
    # 카테고리 3: 모델 합의도 (9차원)
    'agreement':       '9 classes × (>0.5 비율) = 9',
    
    # 카테고리 4: 예측 불확실성 (1차원)
    'entropy':         '전체 예측의 엔트로피 = 1',
    
    # 카테고리 5: 텍스트 통계 (5차원)
    'text_stats':      'length, word_count, chosung_ratio, special_ratio, has_url = 5',
    
    # 카테고리 6: 키워드 힌트 (9차원)
    'keyword_hints':   '9 classes × keyword_score = 9',
    
    # 총계: 294차원
}
```

### 6.4 3중 Meta-Learner 앙상블

```
Meta-Learner 학습 전략
═══════════════════════════════════════════════════════════

  1. LightGBM (Primary, 가중치 0.5)
     ─────────────────────────────────────────
     • 9개 독립 이진 분류기
     • num_leaves=63, lr=0.05, n_estimators=1000
     • early_stopping_rounds=50
     • feature_fraction=0.8, bagging_fraction=0.8
     • scale_pos_weight: 자동 계산

  2. 2-Layer MLP (Secondary, 가중치 0.3)
     ─────────────────────────────────────────
     • 294→256→128→9
     • BatchNorm + ReLU + Dropout(0.3, 0.2)
     • BCEWithLogitsLoss, lr=1e-3, epochs=100
     • Early Stopping patience=15

  3. Ridge Regression (Baseline, 가중치 0.2)
     ─────────────────────────────────────────
     • Ridge(alpha=1.0) per class
     • 안정적 기준선 역할
     • 과적합 방지 보험

  최종 Meta 확률:
    P_meta = 0.5 × P_lgb + 0.3 × P_mlp + 0.2 × P_ridge
    가중치는 Validation F1-Macro 기준 Nelder-Mead로 최적화
═══════════════════════════════════════════════════════════
```

---

## 7. 모듈 ⑥: Self-Training (`src/self_training.py`)

### 7.1 기능 명세

| 항목 | 설명 |
|------|------|
| **역할** | 비레이블 데이터(APEACH, Ko-HatefulMemes)에 pseudo-label 부여 후 반복 학습 |
| **입력** | 학습 완료 앙상블 모델 + 비레이블 데이터 |
| **출력** | pseudo-labeled 데이터 + 재학습된 모델 |
| **반복** | 3라운드 (confidence threshold: 0.95→0.92→0.90) |

### 7.2 Self-Training 프로세스

```
Self-Training 3라운드 프로세스
═══════════════════════════════════════════════════════════

  Round 0 (초기):
    Teacher = Stage B/C 학습 완료 앙상블
    Unlabeled Pool = APEACH (11,666건) + Ko-HatefulMemes (8,500건)
    ≈ 20,166건

  Round 1 (confidence ≥ 0.95):
  ─────────────────────────────────────────
    1. Teacher로 Unlabeled Pool 추론
    2. 클래스별 max(P) ≥ 0.95인 샘플 추출
    3. 예상 추출: ~5,000건 (25%)
    4. 추출 데이터 + 원본 데이터로 Student 모델 학습
       (Noisy Student: dropout↑, augmentation↑)
    5. Student가 새 Teacher가 됨

  Round 2 (confidence ≥ 0.92):
  ─────────────────────────────────────────
    1. 새 Teacher로 남은 Pool 재추론
    2. confidence ≥ 0.92 샘플 추가 추출 (~3,000건)
    3. 누적 pseudo-labeled: ~8,000건
    4. 재학습

  Round 3 (confidence ≥ 0.90):
  ─────────────────────────────────────────
    1. 최종 Teacher로 남은 Pool 재추론
    2. confidence ≥ 0.90 샘플 추가 추출 (~2,000건)
    3. 누적 pseudo-labeled: ~10,000건
    4. 최종 재학습

  Noisy Student 변형:
  • Student 학습 시 dropout 0.3→0.4, augmentation 강화
  • Teacher보다 더 강건한 모델 유도
═══════════════════════════════════════════════════════════
```

### 7.3 Pseudo-Label 품질 관리

```python
class PseudoLabelQualityFilter:
    """Pseudo-label 품질 검증"""
    
    QUALITY_CHECKS = {
        # 1. Confidence 기반 필터링
        'min_confidence': 0.90,      # 최소 신뢰도
        'max_entropy': 2.0,          # 최대 예측 엔트로피
        
        # 2. 일관성 검증
        'min_model_agreement': 0.6,  # 25모델 중 60% 이상 합의
        
        # 3. 분포 제약
        'max_positive_ratio': 0.5,   # pseudo 데이터 양성 비율 상한
        'min_clean_ratio': 0.2,      # clean 비율 하한
    }
```

---

## 8. 모듈 ⑦: Error Correction Network (`src/error_correction.py`)

### 8.1 기능 명세

| 항목 | 설명 |
|------|------|
| **역할** | 앙상블 모델의 잔차(residual) 패턴을 학습하여 체계적 오류 보정 |
| **입력** | 앙상블 예측 확률 + 정답 레이블 → 잔차 |
| **출력** | 보정 확률 (잔차 보정값) |
| **모델** | LightGBM (경량, 빠른 추론) |

### 8.2 ECN 학습 방법

```
Error Correction Network
═══════════════════════════════════════════════════════════

  학습 데이터:
    X = [앙상블 확률 (9차원), 텍스트 통계 (5차원), 키워드 힌트 (9차원)]
    y = 정답 레이블 - 앙상블 예측 = 잔차 (residual)

  예시:
    앙상블 예측: [0.7, 0.1, 0.0, ...]  → 이진: [1, 0, 0, ...]
    정답 레이블: [1,   1,   0,   ...]
    잔차:       [0,   1,   0,   ...]  → 기타 혐오를 놓침
    
    ECN은 이 "놓치는 패턴"을 학습
    → 특정 확률 분포 + 텍스트 특성 조합에서 FN 발생 경향 포착

  추론:
    P_corrected = P_ensemble + α × ECN_prediction
    α = 0.1 (보정 강도, Validation으로 튜닝)
═══════════════════════════════════════════════════════════
```

---

## 9. 모듈 ⑧: Inference Optimizer (`src/inference_optimizer.py`)

### 9.1 기능 명세

| 항목 | 설명 |
|------|------|
| **역할** | TTA + Temperature Scaling + Bayesian Threshold 통합 |
| **입력** | 최종 확률 (9차원) |
| **출력** | 최적화된 이진 예측 (9차원) |

### 9.2 TTA (Test-Time Augmentation) 명세

```python
class TextTTA:
    """텍스트 TTA — 추론 시 5개 변형 평균"""
    
    AUGMENTATIONS = [
        lambda t: t,                              # 원본
        lambda t: t.rstrip('?!.~'),               # 문장부호 제거
        lambda t: ' '.join(t.split()[::-1][-2:] + t.split()[:-2]),  # 끝 2어절 앞으로
        lambda t: re.sub(r'\s+', ' ', t),         # 공백 재정규화
        lambda t: t + ' .',                       # 마침표 추가
    ]
    
    def predict_with_tta(self, text, model_fn):
        predictions = []
        for aug_fn in self.AUGMENTATIONS:
            augmented = aug_fn(text)
            pred = model_fn(augmented)  # (9,) 확률
            predictions.append(pred)
        return np.mean(predictions, axis=0)
```

### 9.3 Bayesian Threshold Optimization (Optuna)

```python
import optuna

def optimize_thresholds_bayesian(y_true, y_prob, n_trials=2000):
    """
    Optuna TPE Sampler로 클래스별 최적 임계값 탐색
    기존 Grid Search (0.01 step) 대비 탐색 효율 ~10배
    """
    def objective(trial):
        thresholds = []
        for i in range(9):
            t = trial.suggest_float(f'threshold_{i}', 0.05, 0.95)
            thresholds.append(t)
        
        y_pred = (y_prob > np.array(thresholds)).astype(int)
        return f1_score(y_true, y_pred, average='macro')
    
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    best_thresholds = [
        study.best_params[f'threshold_{i}'] for i in range(9)
    ]
    return best_thresholds, study.best_value
```

---

## 10. 통합 파이프라인 (`run_train_phase2.py`)

### 10.1 명령행 인터페이스 설계

```
Usage: python run_train_phase2.py [OPTIONS]

Options:
  --skip-data           이미 병합된 데이터가 있으면 재다운로드 건너뜀
  --skip-train          저장된 모델이 있으면 재학습 건너뜀
  --models MODEL,...    특정 모델만 학습 (쉼표 구분)
                        유효값: kcelectra, kcbert, klue_bert, klue_roberta, kr_electra
  --rule-blend FLOAT    룰 시스템 혼합 비율 (기본: 0.3, 범위: 0.0~1.0)
  --data-cache DIR      HuggingFace 캐시 경로 (기본: ./data/hf_cache)
  --output-dir DIR      결과 저장 경로 (기본: ./results)
  --model-dir DIR       모델 저장 경로 (기본: ./models)
```

### 10.2 실행 흐름

```python
def main():
    """Phase 2 통합 파이프라인"""
    
    # ── Stage A: 데이터 병합 ──
    if not args.skip_data:
        merger = ExternalDataMerger(cache_dir=args.data_cache)
        merged_df = merger.merge_all(unsmile_train_path='data/processed/train.csv')
        merger.save(merged_df, 'data/processed/train_phase2.csv')
        # 예상 소요: ~30분
    
    # ── Stage B: 5-모델 앙상블 학습 ──
    if not args.skip_train:
        for model_config in PHASE2_MODELS:
            if args.models and model_config['name'] not in args.models:
                continue
            train_single_model(model_config, train_data, val_data)
            # 모델당 예상 소요: 12~24시간
        
        # 앙상블 가중치 최적화 (Nelder-Mead)
        optimize_ensemble_weights(all_model_probs, val_labels)
        
        # 클래스별 최적 임계값 탐색
        optimize_thresholds(ensemble_probs, val_labels)
    
    # ── Stage C: 룰 시스템 보정 ──
    if args.rule_blend > 0:
        rule_sys = RuleSystem(keyword_blend_weight=args.rule_blend)
        corrected_probs = rule_sys.correct(test_texts, ensemble_probs)
    
    # ── 최종 평가 ──
    # 반드시 UnSmile 원본 test.csv로 평가
    evaluate(corrected_probs, test_labels)
```

### 10.3 모델별 설정값

```python
PHASE2_MODELS = [
    {
        'name': 'kcelectra',
        'pretrained': 'beomi/KcELECTRA-base',
        'batch_size': 32,
        'lr': 2e-5,
        'epochs': 60,
        'early_stopping_patience': 12,
    },
    {
        'name': 'kcbert',
        'pretrained': 'beomi/kcbert-base',
        'batch_size': 32,
        'lr': 2e-5,
        'epochs': 60,
        'early_stopping_patience': 12,
    },
    {
        'name': 'klue_bert',
        'pretrained': 'klue/bert-base',
        'batch_size': 32,
        'lr': 2e-5,
        'epochs': 60,
        'early_stopping_patience': 12,
    },
    {
        'name': 'klue_roberta',
        'pretrained': 'klue/roberta-base',
        'batch_size': 32,
        'lr': 2e-5,
        'epochs': 60,
        'early_stopping_patience': 12,
    },
    {
        'name': 'kr_electra',
        'pretrained': 'snunlp/KR-ELECTRA-discriminator',
        'batch_size': 32,
        'lr': 2e-5,
        'epochs': 60,
        'early_stopping_patience': 12,
    },
]
```

### 10.4 결과 파일

```
학습 완료 시 생성되는 파일
─────────────────────────────────────────────────
  results/
  ├── phase2_results.json      ← 전체 성능 지표
  ├── phase2_thresholds.json   ← 최적 임계값 + 가중치
  └── phase2_training.log      ← 학습 로그

  models/
  ├── phase2_kcelectra.pt      ← 개별 모델 가중치
  ├── phase2_kcbert.pt
  ├── phase2_klue_bert.pt
  ├── phase2_klue_roberta.pt
  └── phase2_kr_electra.pt

  data/processed/
  └── train_phase2.csv         ← 병합된 학습 데이터
─────────────────────────────────────────────────
```

---

## 11. 예상 성능 분석

### 11.1 6-Stage 예상 F1-Macro

```
Phase 1 → Phase 2 (6-Stage) 예상 개선 경로
═══════════════════════════════════════════════════════════

  Phase 1 (현재)     ██████████████████████░░░░░░░░░  82.1%
                     ─────────────────────────────────
  Stage A 목표       ████████████████████████░░░░░░░  87~89%
  (데이터 14배 확장 + 정밀 전처리 + 증강)  (+5~7%p)
                     ─────────────────────────────────
  Stage B 목표       ██████████████████████████░░░░░  91~93%
  (5모델 × 5-Fold + Stacking Meta-Learner)  (+4~5%p)
                     ─────────────────────────────────
  Stage C 목표       ████████████████████████████░░░  94~96%
  (Curriculum + AWP + R-Drop + MultiTask)    (+2~3%p)
                     ─────────────────────────────────
  Stage D 목표       █████████████████████████████░░  95~97%
  (Self-Training 3라운드 + Noisy Student)    (+1~2%p)
                     ─────────────────────────────────
  Stage E 목표       ██████████████████████████████░  97~98%
  (3-Level 룰 시스템 + ECN 보정)             (+1~2%p)
                     ─────────────────────────────────
  Stage F 목표       ██████████████████████████████▓  98~99%
  (TTA + Calibration + Bayesian Threshold)   (+0.5~1%p)
                     ─────────────────────────────────
═══════════════════════════════════════════════════════════
```

### 11.2 클래스별 예상 개선

| 클래스 | Phase 1 F1 | 학습 샘플 (현재) | Phase 2 추가 | Phase 2 예상 F1 | 개선 근거 |
|--------|-----------|-----------------|-------------|----------------|----------|
| **기타 혐오** | 54.2% | 569 | +수만 건 (K-MHaS physical+politics + Korean Toxic 폭력/비난) | **90~95%** | 데이터 구조적 해결 |
| **악플/욕설** | 69.4% | 3,143 | +수만 건 (K-MHaS profanity + BEEP! + Korean Toxic 욕설 + Curse Filtering) | **93~97%** | 데이터 + 룰 시스템 |
| 여성/가족 | 83.7% | 3,759 | +수천 건 (KOLD + BEEP!) | 96~98% | KOLD 정밀 매핑 |
| 인종/국적 | 85.5% | 1,050 | +수천 건 (KOLD + K-MHaS) | 97~98% | 다중 소스 보강 |
| 연령 | 87.6% | 603 | +수천 건 (K-MHaS age) | 97~98% | 직접 매핑 |
| 지역 | 92.0% | 644 | +수천 건 (K-MHaS origin) | 97~99% | 직접 매핑 |
| 종교 | 90.2% | 422 | +수천 건 (K-MHaS religion) | 97~99% | 직접 매핑 |
| 남성 | 88.2% | 1,297 | +수천 건 (K-MHaS gender + KOLD) | 96~98% | 키워드 분리 매핑 |
| 성소수자 | 88.4% | 1,200 | +수백 건 (KOLD LGBT) | 96~98% | KOLD 직접 매핑 |

### 11.3 개선 효과의 이론적 근거

```
데이터 규모 효과 (실증 연구 기반)
─────────────────────────────────────────────────────────
  • Banko & Brill (2001): 학습 데이터 10배 → 정확도 5~15%p ↑
  • Sun et al. (2017): ImageNet 데이터 10배 → Top-5 에러 35% 감소
  • 본 프로젝트: 14배 확대 + 5-모델 앙상블 + 고급 학습 기법 + 룰 시스템
    → 보수적 추정: +9%p (Stage A+B)
    → 고급 기법 추가: +2~3%p (Stage B+)
    → 룰 보정 추가: +1~3%p (Stage C)
    → 합계: +12~15%p → 목표 97~98% 달성 가능

  핵심 가정:
  ① 레이블 매핑 품질 90%+ 달성 (수동 검증 필수)
  ② 도메인 불일치 최소화 (UnSmile 가중치 ×2)
  ③ 5-모델 앙상블이 3-모델 대비 +2%p 이상 (다양성 효과)
  ④ Curriculum Learning + Hard Negative Mining으로 +2%p (경계 사례 학습)
  ⑤ 룰 시스템의 욕설 키워드 커버리지 95%+ (Korean Curse 사전 확장)
─────────────────────────────────────────────────────────
```

---

## 12. 위험 분석 및 대응

### 12.1 위험 요소

| # | 위험 | 발생 확률 | 영향 | 대응 방안 |
|---|------|----------|------|----------|
| 1 | 레이블 매핑 오류 → 노이즈 유입 | 중 | F1 하락 | 데이터셋별 랜덤 100건 수동 검증 |
| 2 | 도메인 불일치 (뉴스댓글 vs 커뮤니티) | 중 | 일반화 저하 | UnSmile 데이터 가중치 ×2 부여 |
| 3 | APEACH pseudo-labeling 오류 | 중 | 노이즈 | confidence ≥ 0.9 필터링 |
| 4 | Ko-HatefulMemes 영어/이미지 혼재 | 높 | 노이즈 | **한국어 필터링 필수**, 우선순위 5순위 |
| 5 | GPU 메모리 부족 (5모델 학습) | 저 | 학습 불가 | batch_size 축소 + gradient accumulation |
| 6 | 학습 시간 초과 | 저 | 일정 지연 | 중간 체크포인트 저장, epoch 조절 |
| 7 | 룰 시스템 과보정 | 중 | Precision 하락 | blend_weight 세밀 조절 (0.1 단위) |

### 12.2 실패 시 대안 전략

```
F1 목표 미달 시 단계적 대안
─────────────────────────────────────────────────
  97~98% 미달, 95~97% 달성:
  → 룰 시스템 blend_weight 증가 (0.3 → 0.5)
  → 클래스별 키워드 사전 확대
  → 추가 외부 데이터 탐색

  95% 미달, 93~95% 달성:
  → 6~7번째 모델 추가 (DeBERTa-v3-base-ko 등)
  → Focal Loss 또는 Dice Loss 시도
  → 추가 데이터 증강 (BackTranslation, EDA)

  93% 미달:
  → 레이블 매핑 전수 재검증
  → 데이터 정제 (노이즈 샘플 제거)
  → 학습 전략 재설계 (Curriculum Learning 등)
─────────────────────────────────────────────────
```

---

## 13. 트러블슈팅 가이드

### 13.1 데이터셋 다운로드 실패

```bash
# HuggingFace datasets 라이브러리 업데이트
pip install -U datasets

# 특정 데이터셋 접근 테스트
python -c "from datasets import load_dataset; \
  ds = load_dataset('jeanlee/kmhas_korean_hate_speech', split='train[:10]'); \
  print(f'OK: {len(ds)}건')"

# 캐시 정리 후 재시도
# Windows: Remove-Item -Recurse -Force $HOME\.cache\huggingface\datasets\*
# Linux:   rm -rf ~/.cache/huggingface/datasets/*
```

### 13.2 GPU 메모리 부족

```
대응 방안 (우선순위 순):
1. batch_size 축소: 32 → 16 → 8
2. Mixed Precision 확인: fp16=True
3. Gradient Accumulation: accumulation_steps=2~4
4. Max Length 축소: 128 → 96 (비권장, 성능 저하 가능)
5. 모델 수 축소: 5개 → 3개 (핵심 모델만)
```

### 13.3 학습 시간 단축

```
전략:
1. 핵심 모델 2개만 선택: KcELECTRA + KLUE-RoBERTa
   → 예상 시간: 120시간 → ~50시간
2. epoch 축소: 60 → 30 (데이터 충분 시)
3. Learning Rate Warmup 후 Cosine Decay로 빠른 수렴
4. 중간 체크포인트로 이어서 학습
```

### 13.4 룰 시스템이 성능을 낮출 때

```
진단:
1. blend_weight=0.0 으로 설정하여 모델 단독 성능 확인
2. 클래스별로 룰 적용 전후 F1 비교
3. Precision이 크게 하락하면 → blend_weight 축소 또는 키워드 정리

조치:
- blend_weight 0.3 → 0.1 로 축소
- false positive가 많은 키워드 제거
- strong/weak 임계값 재조정
```

---

## 14. 구현 우선순위 체크리스트

| 순서 | 구현 항목 | 의존성 | Stage | 상태 |
|------|----------|--------|-------|------|
| 1 | `src/data/verify_datasets.py` 데이터셋 검증 | 없음 | 사전 | ✅ |
| 2 | `src/data/preprocessing.py` 전처리 파이프라인 | 없음 | A | ✅ |
| 3 | `src/data/external_data_merger.py` 데이터 병합 | #1, #2 | A | ✅ |
| 4 | `src/data/aeda_augmentation.py` AEDA 증강 | #3 | A | ✅ |
| 5 | `src/models/model.py` 5-모델 K-Fold 확장 | 없음 | B | ✅ |
| 6 | `src/training/trainer.py` AWP + R-Drop + Label Smoothing | #5 | B/C | ✅ |
| 7 | `src/models/ensemble.py` Stacking Meta-Learner | #5, #6 학습 완료 | B | ✅ |
| 8 | `src/training/self_training.py` Self-Training | #7 완료 | D | ✅ |
| 9 | `src/inference/rule_system.py` 룰 시스템 | 없음 | E | ✅ |
| 10 | `src/inference/error_correction.py` ECN | #7, #9 완료 | E | ✅ |
| 11 | `src/inference/inference.py` TTA+Calibration | #10 완료 | F | ✅ |
| 12 | `src/main.py` 통합 6-Stage 파이프라인 | 전체 | 통합 | ✅ |
| - | `src/training/curriculum_learning.py` | #5 | B | ✅ |
| - | `src/training/hard_negative_mining.py` | #6 | C | ✅ |

> **총 구현 소요**: 코드 작성 ~30시간 + 학습 ~80~120시간 (GPU) = **약 110~150시간**

---

**이전 문서**: [09_외부데이터셋_명세.md](09_외부데이터셋_명세.md)  
**다음 문서**: [11_전처리_상세설계.md](11_전처리_상세설계.md)  
**앙상블 설계**: [12_앙상블_심층설계.md](12_앙상블_심층설계.md)  
**로드맵**: [08_성능개선_로드맵.md](08_성능개선_로드맵.md)  
**처음으로**: [README.md](README.md)
