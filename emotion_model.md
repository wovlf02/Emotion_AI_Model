# UnSmile 한국어 혐오 표현 탐지 AI

URL: https://github.com/wovlf02/Emotion_AI_Model
기술: Python, PyTorch, Transformers, KcELECTRA, SoongsilBERT, KLUE-RoBERTa, Scikit-learn, NumPy, Pandas, AEDA, BCEWithLogitsLoss, AdamW, Cosine Scheduler
날짜: 2025년 1월
팀구성: 개인 프로젝트

<aside>

*🎭 온라인 커뮤니티와 소셜 미디어에서 급증하는 혐오 표현을 탐지하기 위해 개발된 다중 라벨 분류 AI 시스템. 3-모델 하이브리드 앙상블(KcELECTRA, SoongsilBERT, KLUE-RoBERTa)과 클래스별 임계값 최적화를 통해 9개 혐오 카테고리를 동시 탐지하며, Hamming Accuracy 96.72%, F1-Macro 82.91%를 달성하여 목표 정확도 95%를 초과 달성했습니다.*

</aside>

<aside>

> ***Information***
> 

**개발 기간:** 2025/01

**팀 구성:** 1인

**담당 업무:** 데이터 분석, 모델 설계, 학습 파이프라인 구축, 앙상블 최적화, 문서화

</aside>

<aside>

> ***Review***
> 

PyTorch와 Hugging Face Transformers를 활용한 딥러닝 모델 개발 역량을 강화했습니다. 다중 라벨 분류 문제의 특수성(Sigmoid + BCE), 극심한 클래스 불균형 해결을 위한 AEDA 데이터 증강, 3-모델 하이브리드 앙상블 설계 및 가중 소프트 보팅, 클래스별 임계값 최적화(Nelder-Mead), Early Stopping과 Cosine Annealing 스케줄러를 활용한 학습 전략 수립 경험을 쌓았습니다.

</aside>

### 01. GitHub

[https://github.com/wovlf02/Emotion_AI_Model](https://github.com/wovlf02/Emotion_AI_Model)

### 02. 프로젝트 배경 및 목적

- **해결 대상**
  - 온라인 혐오 표현 급증: 2021년 기준 사이버 폭력 신고 건수 전년 대비 40% 증가
  - 기존 해결책의 한계: 단순 욕설 필터링은 변형 표현("ㅅㅂ", "10발") 탐지 불가
  - 이진 분류 한계: 혐오 유형을 세분화하여 구분하지 못함
  - 키워드 기반 한계: 문맥을 이해하지 못해 오탐 발생
- **목표**
  - 9개 혐오 카테고리(여성/가족, 남성, 성소수자, 인종/국적, 연령, 지역, 종교, 기타 혐오, 악플/욕설) 동시 탐지
  - 95% 이상의 Hamming Accuracy 달성
  - 실제 서비스에 적용 가능한 고성능 모델 개발

### 03. 주요 구현 사항 및 기술적 성과

<aside>

**🎯 핵심 성과 지표**

1. Hamming Accuracy 96.72% 달성 - 목표 95% 초과 달성 (+1.72%p)
2. F1-Macro 82.91% 달성 - 개별 모델 대비 +3.4%p 향상 (앙상블 효과)
3. F1-Micro 81.08% 달성 - 전체 샘플 기반 균형 잡힌 성능
4. Exact Match 74.63% 달성 - 9개 레이블 완전 일치율
5. 클래스별 임계값 최적화 - 추가 +3.8%p F1 향상
6. 총 학습 시간 약 15시간 - 3개 모델 순차 학습 (각 80 epoch)
7. Early Stopping 적용 - 약 30% 학습 시간 절약 (평균 55 epoch에서 조기 종료)
</aside>

<aside>

**🧠 3-모델 하이브리드 앙상블 설계**

1. KcELECTRA-Base (슬랭/욕설 전문가) - 한국어 인터넷 댓글 기반 사전학습, 신조어/은어 이해도 우수
2. SoongsilBERT-Base (안정적 베이스라인) - 다양한 도메인에서 균형 잡힌 성능, 학술 기관 검증 품질
3. KLUE-RoBERTa-Base (고맥락 의미론 전문가) - 62GB KLUE 벤치마크 코퍼스 학습, 문맥 의존적 혐오 탐지
4. 가중 소프트 보팅 - 검증 데이터 F1-Macro 기반 모델별 가중치 할당
5. 모델별 역할 분담 - 각 모델의 전문성 활용으로 상호 보완적 예측
6. 앙상블 효과 +3.4%p - 개별 모델 평균 79.5% → 앙상블 82.91%
7. 예측 안정성 향상 - 다양한 관점 통합으로 불안정한 예측 방지
</aside>

<aside>

**📊 데이터 처리 및 증강**

1. UnSmile 데이터셋 활용 - Smilegate AI 제공, 전문가 레이블링, 약 18,000 샘플
2. 9개 혐오 카테고리 다중 라벨 분류 - 하나의 문장에 여러 레이블 동시 적용 가능
3. AEDA 데이터 증강 - 소수 클래스(연령 4%, 기타 혐오 3.7%) 2~3배 오버샘플링
4. 클래스 불균형 해결 - 다수:소수 = 6.5:1 불균형 문제 해결
5. 최소한의 텍스트 전처리 - 난독화 표현 보존 (특수문자 무조건 제거 X)
6. 토크나이저 모델별 적용 - KcELECTRA BPE, SoongsilBERT WordPiece, RoBERTa BPE
7. max_length 128 설정 - 95% 이상 텍스트 커버리지 확보
</aside>

<aside>

**🔧 학습 전략 및 최적화**

1. BCEWithLogitsLoss 손실 함수 - 다중 라벨 분류를 위한 Binary Cross-Entropy (Sigmoid 내장)
2. pos_weight 클래스 가중치 - 소수 클래스에 최대 10배 가중치 부여
3. AdamW 옵티마이저 - lr=2e-5, weight_decay=0.01, Decoupled Weight Decay
4. Cosine Annealing with Warmup - 500 warmup steps, 전체의 5%
5. Early Stopping (patience=10) - 10 epoch 연속 개선 없으면 조기 종료
6. Dropout 0.3 적용 - 분류 헤드에 30% 드롭아웃으로 과적합 방지
7. Gradient Clipping (max_norm=1.0) - 그래디언트 폭발 방지
</aside>

<aside>

**📈 클래스별 임계값 최적화**

1. Grid Search 기반 탐색 - 0.01~0.99 범위에서 0.01 간격으로 최적값 탐색
2. Nelder-Mead 전역 최적화 - F1-Macro 최대화를 위한 다차원 최적화
3. 클래스별 맞춤 임계값 - 여성/가족 0.59, 인종/국적 0.34, 악플/욕설 0.40 등
4. 소수 클래스 재현율 향상 - 낮은 임계값으로 미탐지 방지 (인종/국적 0.34)
5. 다수 클래스 정밀도 유지 - 적정 임계값으로 과탐지 방지
6. 검증 데이터 기반 최적화 - Val F1-Macro 82.91% 달성
7. 테스트 데이터 일반화 - Test Hamming Accuracy 96.72% 달성
</aside>

<aside>

**🏗️ 모델 아키텍처**

1. MultiLabelClassifier 클래스 - 사전학습 모델 + 분류 헤드 구조
2. [CLS] 토큰 임베딩 추출 - Transformer Encoder의 첫 번째 토큰 벡터(768차원) 활용
3. 2-Layer Dropout 강화 - dropout1(0.3) + dropout2(0.15)로 과적합 방지 강화
4. Linear 분류 레이어 - 768 → 9 차원 변환 (9개 레이블)
5. Xavier 초기화 적용 - 분류기 가중치 안정적 초기화
6. QLoRA 지원 (선택적) - Large 모델용 4비트 양자화 및 LoRA 어댑터
7. 모델 저장 형식 - state_dict 기반 .pt 파일 저장
</aside>

<aside>

**📁 프로젝트 구조 설계**

1. 루트 실행 스크립트 분리 - run_train.py, run_inference.py 등 진입점 분리
2. src 모듈 구조화 - train.py, model.py, dataset.py 등 기능별 모듈 분리
3. data 디렉토리 구조 - raw(원본 TSV) / processed(전처리 CSV) 분리
4. models 디렉토리 - 학습된 모델 가중치 저장 (kcelectra.pt, soongsil.pt, roberta_base.pt)
5. results 디렉토리 - 실험 결과 JSON, 예측 CSV 저장
6. docs 디렉토리 - 체계적 기술 문서화 (5개 핵심 문서 + 발표 자료)
7. Google Drive 모델 배포 - 대용량 모델 파일 외부 저장 및 다운로드 링크 제공
</aside>

<aside>

**🔍 추론 파이프라인**

1. 모델별 토크나이저 적용 - 각 모델에 맞는 토크나이저로 입력 처리
2. 배치 예측 - DataLoader 기반 효율적 배치 추론
3. Sigmoid 확률 변환 - Logits → 0~1 확률값 변환
4. 가중 소프트 보팅 - 3개 모델 예측 확률의 가중 평균
5. 최적 임계값 적용 - 클래스별로 다른 임계값으로 이진 예측
6. 평가 지표 계산 - F1-Macro, F1-Micro, Exact Match, Hamming Accuracy
7. 결과 저장 - JSON(메트릭) + CSV(예측 결과) 저장
</aside>

### 04. 기술 스택 상세

| 분류 | 기술 | 버전 | 설명 |
| --- | --- | --- | --- |
| Language | Python | 3.11.9 | 프로그래밍 언어 |
| Deep Learning | PyTorch | ≥2.0.0 | 딥러닝 프레임워크 |
|  | Transformers | ≥4.30.0 | Hugging Face 사전학습 모델 |
|  | Accelerate | ≥0.20.0 | 분산 학습 지원 |
|  | PEFT | ≥0.4.0 | LoRA/QLoRA 파인튜닝 |
|  | BitsAndBytes | ≥0.41.0 | 4비트 양자화 |
| Pre-trained Models | KcELECTRA-Base | - | 한국어 인터넷 언어 특화 |
|  | SoongsilBERT-Base | - | 균형 잡힌 범용 성능 |
|  | KLUE-RoBERTa-Base | - | 문맥 이해력 우수 |
| Data Processing | Pandas | ≥1.5.0 | 데이터프레임 처리 |
|  | NumPy | ≥1.23.0 | 수치 연산 |
|  | Scikit-learn | ≥1.2.0 | 평가 메트릭, 최적화 |
| Visualization | Matplotlib | ≥3.7.0 | 차트 시각화 |
|  | Seaborn | ≥0.12.0 | 통계 시각화 |
| Korean NLP | SoyNLP | ≥0.0.493 | 한국어 텍스트 처리 |
|  | KoNLPy | ≥0.6.0 | 한국어 형태소 분석 |
| Utilities | tqdm | ≥4.65.0 | 진행률 표시 |
|  | Datasets | ≥2.12.0 | 데이터셋 로딩 |

### 05. 프로젝트 구조

```
Emotion_AI_Model/
├── README.md                    # 프로젝트 소개 및 사용 가이드
├── LICENSE                      # MIT 라이선스
├── requirements.txt             # Python 의존성 패키지
├── emotion_model.md             # 프로젝트 상세 문서 (본 파일)
│
├── run_train.py                 # 학습 실행 스크립트 (진입점)
├── run_inference.py             # 추론 실행 스크립트 (진입점)
├── run_final_inference.py       # 최종 평가 실행 스크립트 (진입점)
├── run_optimize_thresholds.py   # 임계값 최적화 실행 스크립트 (진입점)
│
├── src/                         # 소스 코드 모듈
│   ├── train.py                 # 메인 학습 로직 (3-모델 앙상블 학습)
│   ├── inference.py             # 추론 로직 (앙상블 예측)
│   ├── final_inference.py       # 최종 평가 로직 (테스트 데이터 평가)
│   ├── optimize_thresholds.py   # 임계값 최적화 로직 (Grid Search + Nelder-Mead)
│   ├── data_loader.py           # 데이터 로딩 및 전처리 (TSV → CSV 변환)
│   ├── dataset.py               # PyTorch Dataset 클래스 (토크나이징, 배치 처리)
│   ├── model.py                 # 모델 아키텍처 (MultiLabelClassifier, HybridEnsemble)
│   ├── aeda_augmentation.py     # AEDA 데이터 증강 (구두점 삽입)
│   └── asymmetric_loss.py       # 비대칭 손실 함수 (ASL, 선택적)
│
├── data/                        # 데이터 디렉토리
│   ├── raw/                     # 원본 데이터 (TSV)
│   │   ├── unsmile_train.tsv    # 학습 데이터 (~15,000 샘플)
│   │   ├── unsmile_dev.tsv      # 검증 데이터 (~1,500 샘플)
│   │   └── unsmile_test.tsv     # 테스트 데이터 (~1,500 샘플)
│   └── processed/               # 전처리된 데이터 (CSV)
│       ├── train.csv            # 전처리된 학습 데이터
│       ├── val.csv              # 전처리된 검증 데이터
│       └── test.csv             # 전처리된 테스트 데이터
│
├── models/                      # 학습된 모델 가중치 (.pt)
│   ├── kcelectra.pt             # KcELECTRA 학습 모델 (~400MB)
│   ├── soongsil.pt              # SoongsilBERT 학습 모델 (~400MB)
│   └── roberta_base.pt          # RoBERTa-Base 학습 모델 (~400MB)
│
├── results/                     # 실험 결과
│   ├── final_results.json       # 최종 성능 지표
│   ├── optimal_thresholds.json  # 최적 임계값
│   ├── final_test_results.json  # 테스트 데이터 평가 결과
│   ├── final_test_predictions.csv  # 테스트 예측 결과
│   └── figures/                 # 시각화 결과
│
└── docs/                        # 기술 문서
    ├── README.md                # 문서 가이드
    ├── 01_프로젝트_개요.md       # 프로젝트 배경, 목표, 범위
    ├── 02_데이터_분석.md         # UnSmile 데이터셋 EDA, 전처리 전략
    ├── 03_모델_아키텍처.md       # 3-모델 앙상블 설계, 분류기 헤드
    ├── 04_학습_전략.md           # 손실 함수, 옵티마이저, 스케줄러
    ├── 05_실험_결과.md           # 성능 평가, 오류 분석
    └── presentation/            # 발표 자료
        ├── PRESENTATION.md      # 발표 슬라이드 (Markdown)
        └── log_screenshot/      # 학습 로그 스크린샷
```

### 06. 혐오 표현 카테고리 및 예시

| 카테고리 | 설명 | 예시 표현 |
| --- | --- | --- |
| 여성/가족 | 여성 및 가족 관련 혐오 | "김치녀", "맘충", "보슬" |
| 남성 | 남성 관련 혐오 | "한남충", "자지충" |
| 성소수자 | LGBTQ+ 관련 혐오 | - |
| 인종/국적 | 인종 및 국적 관련 혐오 | "조선족", "짱깨" |
| 연령 | 연령 관련 혐오 | "틀딱", "급식충", "젊친" |
| 지역 | 특정 지역 관련 혐오 | "홍어", "쥐라도" |
| 종교 | 종교 관련 혐오 | "개독", "빠구리" |
| 기타 혐오 | 기타 유형의 혐오 | 분류 불가 혐오 표현 |
| 악플/욕설 | 일반적인 악성 댓글 | 욕설, 비하 표현 |

### 07. 실행 방법

**환경 설정**
```bash
# 저장소 클론
git clone https://github.com/wovlf02/Emotion_AI_Model.git
cd Emotion_AI_Model

# 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate  # macOS/Linux

# 의존성 설치
pip install -r requirements.txt
```

**데이터 준비**
```bash
# UnSmile 데이터셋 다운로드 후 data/raw/ 에 배치
# 데이터 전처리 (TSV → CSV)
python -c "from src.data_loader import UnsmileDataLoader; loader = UnsmileDataLoader('./data'); loader.prepare_dataset()"
```

**모델 학습**
```bash
# 전체 학습 파이프라인 (약 15시간 소요)
python run_train.py
```

**임계값 최적화**
```bash
# 검증 데이터로 클래스별 최적 임계값 탐색
python run_optimize_thresholds.py
```

**최종 평가**
```bash
# 테스트 데이터로 최종 성능 평가
python run_final_inference.py
```

**추론 (새로운 데이터)**
```bash
# 학습된 모델로 추론
python run_inference.py
```

### 08. 모델 다운로드

학습된 모델 파일(.pt)은 용량이 커서 GitHub에 포함되지 않습니다.

**다운로드 링크**: [Google Drive](https://drive.google.com/drive/folders/1Noow6HkhI6hkAuggptroiNmbUVGDbu1u?usp=sharing)

| 파일 | 크기 | 설명 |
| --- | --- | --- |
| kcelectra.pt | ~400MB | KcELECTRA 학습 모델 |
| soongsil.pt | ~400MB | SoongsilBERT 학습 모델 |
| roberta_base.pt | ~400MB | RoBERTa-Base 학습 모델 |

다운로드 후 `models/` 폴더에 배치하세요.

### 09. 참고 자료

- **UnSmile 데이터셋**: [Smilegate AI GitHub](https://github.com/smilegate-ai/korean_unsmile_dataset)
- **KcELECTRA**: [Hugging Face](https://huggingface.co/beomi/KcELECTRA-base)
- **SoongsilBERT**: [Hugging Face](https://huggingface.co/soongsil-ai/soongsil-bert-base)
- **KLUE-RoBERTa**: [Hugging Face](https://huggingface.co/klue/roberta-base)
- **AEDA 논문**: [AEDA: An Easier Data Augmentation Technique for Text Classification](https://arxiv.org/abs/2108.13230)
- **Asymmetric Loss 논문**: [Asymmetric Loss For Multi-Label Classification (ICCV 2021)](https://arxiv.org/abs/2009.14119)

### 10. 라이선스 및 사용 정책

본 프로젝트는 **Portfolio Project License (Based on CC BY-NC-ND 4.0)** 라이선스를 따릅니다.

**⚠️ 포트폴리오 프로젝트 공지**

이 프로젝트는 개인 포트폴리오 및 학습 목적으로만 공개되었습니다.

**허용 사항:**
- ✅ 소스 코드 열람 (교육 목적)
- ✅ 프로젝트 참조 (포트폴리오, 이력서)
- ✅ 기술 분석 및 학습

**금지 사항:**
- ❌ 상업적 사용 금지
- ❌ 코드 복사/수정/재배포 금지
- ❌ 파생 작업 금지
- ❌ 학습된 모델(.pt) 사용 금지
- ❌ AI 학습 데이터로 사용 금지

**저작권 표시:**
```
UnSmile Korean Hate Speech Detection AI by wovlf02
Licensed under Portfolio Project License
GitHub: https://github.com/wovlf02/Emotion_AI_Model
```

**상업적 사용 문의:**  
상업적 이용, 라이선스 협의, 협업 제안 등은 별도 문의 바랍니다.
