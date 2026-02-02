# 🎭 UnSmile 한국어 혐오 표현 탐지 AI

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.11.9-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/🤗_Transformers-4.30+-FFD21E?style=for-the-badge)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-Portfolio-red?style=for-the-badge)](LICENSE)

**3-모델 하이브리드 앙상블을 통한 다중 라벨 혐오 표현 분류 시스템**

[📖 문서](docs/README.md) • [🚀 빠른 시작](#-빠른-시작) • [📊 성능](#-주요-성과) • [💾 모델 다운로드](#-사전-학습-모델-다운로드)

</div>

---

## 📋 목차

- [프로젝트 소개](#-프로젝트-소개)
- [주요 성과](#-주요-성과)
- [혐오 표현 카테고리](#-혐오-표현-카테고리)
- [시스템 요구사항](#-시스템-요구사항)
- [빠른 시작](#-빠른-시작)
- [프로젝트 구조](#-프로젝트-구조)
- [모델 아키텍처](#-모델-아키텍처)
- [사용 방법](#-사용-방법)
- [실험 결과](#-실험-결과)
- [기술 스택](#-기술-스택)
- [문서](#-문서)
- [사전 학습 모델 다운로드](#-사전-학습-모델-다운로드)
- [라이선스](#-라이선스)
- [참고 자료](#-참고-자료)

---

## 🎯 프로젝트 소개

### 배경

온라인 커뮤니티와 소셜 미디어에서 혐오 표현이 급증하고 있습니다. 2021년 기준 사이버 폭력 신고 건수가 **전년 대비 40% 증가**하며, 단순 욕설 필터링을 넘어 **혐오 유형을 세분화하여 탐지**할 수 있는 AI 시스템의 필요성이 대두되었습니다.

### 해결하고자 하는 문제

| 기존 방식 | 한계점 | 본 프로젝트 해결책 |
|-----------|--------|-------------------|
| 단순 욕설 필터링 | "ㅅㅂ", "10발" 등 변형 표현 탐지 불가 | 한국어 인터넷 언어 특화 모델 (KcELECTRA) |
| 이진 분류 | 혐오 유형 구분 불가 | 9개 카테고리 다중 라벨 분류 |
| 키워드 기반 | 문맥 이해 부족으로 오탐 발생 | 문맥 이해 전문가 모델 (RoBERTa) |
| 단일 모델 | 특정 패턴에 과적합 | 3-모델 하이브리드 앙상블 |

### 목표

- ✅ **9개 혐오 카테고리** 동시 탐지 (다중 라벨 분류)
- ✅ **95% 이상의 Hamming Accuracy** 달성
- ✅ 실제 서비스에 적용 가능한 **고성능 모델** 개발

---

## 🏆 주요 성과

<div align="center">

| 지표 | 성능 | 목표 | 달성 |
|:----:|:----:|:----:|:----:|
| **Hamming Accuracy** | **96.72%** | 95% | ✅ **+1.72%p** |
| **F1-Macro** | **82.91%** | 80% | ✅ **+2.91%p** |
| **F1-Micro** | **81.08%** | - | - |
| **Exact Match** | **74.63%** | - | - |

</div>

### 핵심 성과

- 🎯 **목표 정확도 95% 초과 달성** - Hamming Accuracy 96.72%
- 🔄 **3-모델 앙상블 효과** - 개별 모델 대비 +3.4%p F1-Macro 향상
- 📈 **클래스별 임계값 최적화** - 추가 +3.8%p F1 향상
- ⏱️ **효율적 학습** - Early Stopping으로 약 30% 학습 시간 절약

---

## 🏷️ 혐오 표현 카테고리

본 시스템은 **9개의 혐오 카테고리**를 동시에 탐지합니다.

| # | 카테고리 | 설명 | 예시 표현 |
|:-:|----------|------|-----------|
| 1 | **여성/가족** | 여성 및 가족 관련 혐오 | "김치녀", "맘충" |
| 2 | **남성** | 남성 관련 혐오 | "한남충" |
| 3 | **성소수자** | LGBTQ+ 관련 혐오 | - |
| 4 | **인종/국적** | 인종 및 국적 관련 혐오 | "조선족", "짱깨" |
| 5 | **연령** | 연령 관련 혐오 | "틀딱", "급식충" |
| 6 | **지역** | 특정 지역 관련 혐오 | "홍어" |
| 7 | **종교** | 종교 관련 혐오 | "개독" |
| 8 | **기타 혐오** | 기타 유형의 혐오 | - |
| 9 | **악플/욕설** | 일반적인 악성 댓글 | 욕설, 비하 표현 |

> **다중 라벨 분류**: 하나의 문장이 여러 혐오 유형을 동시에 포함할 수 있습니다.  
> 예: "김치녀는 한남이랑 꼭 닮았네" → 여성/가족 ✓, 남성 ✓, 악플/욕설 ✓

---

## 💻 시스템 요구사항

### 필수 환경

| 항목 | 요구사항 |
|------|----------|
| **Python** | 3.11.9 (권장) |
| **CUDA** | 11.8 이상 (GPU 사용 시) |
| **RAM** | 16GB 이상 권장 |
| **GPU VRAM** | 12GB 이상 권장 |
| **저장 공간** | 10GB 이상 (모델 포함) |

### 권장 하드웨어

- **GPU**: NVIDIA RTX 4070 Super 또는 동급 이상
- **CPU**: Intel i7 / AMD Ryzen 7 이상
- **SSD**: 모델 로딩 속도 향상

### 운영체제

- ✅ macOS (Apple Silicon 지원, QLoRA 제외)
- ✅ Linux (Ubuntu 20.04+)
- ✅ Windows 10/11

---

## 🚀 빠른 시작

### 1. 저장소 클론

```bash
git clone https://github.com/wovlf02/Emotion_AI_Model.git
cd Emotion_AI_Model
```

### 2. 가상환경 생성 및 활성화

```bash
# 가상환경 생성
python -m venv .venv

# 활성화 (macOS/Linux)
source .venv/bin/activate

# 활성화 (Windows)
.venv\Scripts\activate
```

### 3. 의존성 설치

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. 데이터셋 준비

[Smilegate AI GitHub](https://github.com/smilegate-ai/korean_unsmile_dataset)에서 UnSmile 데이터셋을 다운로드합니다.

```bash
# data/raw/ 디렉토리에 파일 배치
data/raw/
├── unsmile_train.tsv
├── unsmile_dev.tsv
└── unsmile_test.tsv
```

### 5. 데이터 전처리

```bash
python -c "from src.data_loader import UnsmileDataLoader; loader = UnsmileDataLoader('./data'); loader.prepare_dataset()"
```

### 6. 모델 학습 또는 다운로드

**옵션 A: 직접 학습** (약 15시간 소요)
```bash
python run_train.py
```

**옵션 B: 사전 학습 모델 다운로드** (권장)
1. [Google Drive](https://drive.google.com/drive/folders/1Noow6HkhI6hkAuggptroiNmbUVGDbu1u?usp=sharing)에서 모델 다운로드
2. `models/` 폴더에 배치

### 7. 추론 실행

```bash
python run_inference.py
```

---

## 📁 프로젝트 구조

```
Emotion_AI_Model/
├── 📄 README.md                    # 프로젝트 소개 (본 문서)
├── 📄 emotion_model.md             # 프로젝트 상세 문서 (포트폴리오용)
├── 📄 LICENSE                      # MIT 라이선스
├── 📄 requirements.txt             # Python 의존성
│
├── 🚀 run_train.py                 # 학습 실행 스크립트
├── 🚀 run_inference.py             # 추론 실행 스크립트
├── 🚀 run_final_inference.py       # 최종 평가 스크립트
├── 🚀 run_optimize_thresholds.py   # 임계값 최적화 스크립트
│
├── 📂 src/                         # 소스 코드 모듈
│   ├── train.py                    # 메인 학습 로직
│   ├── inference.py                # 추론 로직
│   ├── final_inference.py          # 최종 평가 로직
│   ├── optimize_thresholds.py      # 임계값 최적화
│   ├── data_loader.py              # 데이터 로딩/전처리
│   ├── dataset.py                  # PyTorch Dataset
│   ├── model.py                    # 모델 아키텍처
│   ├── aeda_augmentation.py        # AEDA 데이터 증강
│   └── asymmetric_loss.py          # 비대칭 손실 함수
│
├── 📂 data/                        # 데이터
│   ├── raw/                        # 원본 TSV 데이터
│   └── processed/                  # 전처리된 CSV 데이터
│
├── 📂 models/                      # 학습된 모델 가중치 (.pt)
│   ├── kcelectra.pt
│   ├── soongsil.pt
│   └── roberta_base.pt
│
├── 📂 results/                     # 실험 결과
│   ├── final_results.json
│   ├── optimal_thresholds.json
│   └── final_test_predictions.csv
│
└── 📂 docs/                        # 기술 문서
    ├── README.md                   # 문서 가이드
    ├── 01_프로젝트_개요.md
    ├── 02_데이터_분석.md
    ├── 03_모델_아키텍처.md
    ├── 04_학습_전략.md
    ├── 05_실험_결과.md
    ├── 06_API_레퍼런스.md
    ├── 07_트러블슈팅.md
    └── presentation/               # 발표 자료
```

---

## 🧠 모델 아키텍처

### 3-모델 하이브리드 앙상블

```
                        입력 텍스트
                            │
          ┌─────────────────┼─────────────────┐
          ▼                 ▼                 ▼
   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
   │  KcELECTRA  │   │ SoongsilBERT│   │   RoBERTa   │
   │             │   │             │   │             │
   │ 슬랭/욕설   │   │  안정적     │   │   고맥락    │
   │   전문가    │   │ 베이스라인  │   │ 의미론 전문가│
   └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
          │                 │                 │
          └─────────────────┼─────────────────┘
                            ▼
                 ┌────────────────────┐
                 │  가중 소프트 보팅   │
                 │ (Weighted Voting)  │
                 └─────────┬──────────┘
                           ▼
                 ┌────────────────────┐
                 │ 클래스별 임계값 적용 │
                 └─────────┬──────────┘
                           ▼
                      최종 예측
```

### 모델별 역할

| 모델 | 역할 | 특징 | 강점 |
|------|------|------|------|
| **KcELECTRA** | 슬랭/욕설 전문가 | 네이버 댓글 기반 사전학습 | 신조어, 은어 이해 |
| **SoongsilBERT** | 안정적 베이스라인 | 균형 잡힌 범용 성능 | 안정적 기준점 |
| **RoBERTa-Base** | 고맥락 의미론 전문가 | 62GB KLUE 코퍼스 학습 | 문맥 이해력 |

### 핵심 기술

| 기술 | 설명 | 효과 |
|------|------|------|
| **AEDA 데이터 증강** | 구두점 삽입으로 소수 클래스 오버샘플링 | 클래스 불균형 해소 |
| **가중 소프트 보팅** | 모델별 예측 확률의 가중 평균 | 앙상블 효과 +3.4%p |
| **클래스별 임계값 최적화** | 0.01~0.99 범위 Grid Search | 추가 +3.8%p 향상 |
| **Cosine Annealing** | Warmup + Cosine Decay 스케줄러 | 안정적 수렴 |

---

## 📖 사용 방법

### 전체 학습 파이프라인

```bash
# 1. 모델 학습 (약 15시간 소요)
python run_train.py

# 2. 임계값 최적화
python run_optimize_thresholds.py

# 3. 최종 평가
python run_final_inference.py

# 4. 추론
python run_inference.py
```

### Python 코드에서 사용

```python
import torch
from transformers import AutoTokenizer
from src.model import create_model

# 모델 로드
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = create_model("beomi/KcELECTRA-base", num_labels=9)
model.load_state_dict(torch.load("models/kcelectra.pt", map_location=device))
model.to(device)
model.eval()

# 토크나이저
tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base")

# 추론
text = "예시 텍스트입니다"
inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True, padding="max_length")
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(inputs['input_ids'], inputs['attention_mask'])
    probs = torch.sigmoid(outputs['logits'])
    
print(probs)
```

---

## 📊 실험 결과

### 모델별 성능 비교

| 모델 | F1-Macro | Hamming Acc | 역할 |
|------|:--------:|:-----------:|------|
| KcELECTRA | 80.2% | 95.8% | 슬랭/욕설 탐지 우수 |
| SoongsilBERT | 79.5% | 95.6% | 안정적 베이스라인 |
| RoBERTa-Base | 78.8% | 95.4% | 문맥 이해 우수 |
| **앙상블** | **82.91%** | **96.72%** | **모든 지표 최고** |

### 앙상블 효과

- **개별 모델 평균 F1-Macro**: 79.5%
- **앙상블 F1-Macro**: 82.91%
- **향상**: **+3.41%p** (상대적 4.3% 개선)

### 클래스별 F1-Score

| 클래스 | F1-Score | 분석 |
|--------|:--------:|------|
| 악플/욕설 | 89.9% | 데이터 풍부, 명확한 패턴 |
| 인종/국적 | 88.1% | 특정 키워드 의존 |
| 여성/가족 | 85.2% | 패턴 일관성 |
| 남성 | 83.4% | 양호 |
| 종교 | 81.2% | 양호 |
| 성소수자 | 79.8% | 보통 |
| 지역 | 76.5% | 보통 |
| 연령 | 72.3% | 데이터 부족 |
| 기타 혐오 | 69.8% | 정의 모호 |

### 최적 임계값

```python
optimal_thresholds = {
    '여성/가족': 0.59,
    '남성': 0.51,
    '성소수자': 0.36,
    '인종/국적': 0.34,  # 낮은 임계값 → 민감 탐지
    '연령': 0.34,
    '지역': 0.56,
    '종교': 0.36,
    '기타 혐오': 0.38,
    '악플/욕설': 0.40
}
```

---

## 🛠️ 기술 스택

### 핵심 기술

| 분류 | 기술 | 버전 |
|------|------|------|
| **언어** | Python | 3.11.9 |
| **딥러닝** | PyTorch | ≥2.0.0 |
| **NLP** | Transformers | ≥4.30.0 |
| **최적화** | Accelerate, PEFT | ≥0.20.0 |

### 사전학습 모델

| 모델 | Hugging Face | 용도 |
|------|--------------|------|
| KcELECTRA-Base | `beomi/KcELECTRA-base` | 슬랭/욕설 |
| SoongsilBERT | `snunlp/KR-SBERT-V40K-klueNLI-augSTS` | 베이스라인 |
| KLUE-RoBERTa | `klue/roberta-base` | 문맥 이해 |

### 데이터 처리

| 라이브러리 | 용도 |
|------------|------|
| Pandas | 데이터프레임 처리 |
| NumPy | 수치 연산 |
| Scikit-learn | 평가 메트릭, 최적화 |
| tqdm | 진행률 표시 |

---

## 📚 문서

상세 기술 문서는 `docs/` 폴더에서 확인할 수 있습니다.

| 문서 | 설명 |
|------|------|
| [📖 문서 가이드](docs/README.md) | 전체 문서 목차 |
| [01_프로젝트_개요](docs/01_프로젝트_개요.md) | 프로젝트 배경 및 목표 |
| [02_데이터_분석](docs/02_데이터_분석.md) | UnSmile 데이터셋 EDA |
| [03_모델_아키텍처](docs/03_모델_아키텍처.md) | 3-모델 앙상블 설계 |
| [04_학습_전략](docs/04_학습_전략.md) | 하이퍼파라미터 및 최적화 |
| [05_실험_결과](docs/05_실험_결과.md) | 성능 평가 및 분석 |
| [06_API_레퍼런스](docs/06_API_레퍼런스.md) | 코드 상세 문서 |
| [07_트러블슈팅](docs/07_트러블슈팅.md) | 문제 해결 가이드 |
| [📊 발표 자료](docs/presentation/PRESENTATION.md) | 프레젠테이션 |

---

## 💾 사전 학습 모델 다운로드

학습된 모델 파일(.pt)은 용량이 커서 GitHub에 포함되지 않습니다.

### 다운로드 링크

🔗 **[Google Drive](https://drive.google.com/drive/folders/1Noow6HkhI6hkAuggptroiNmbUVGDbu1u?usp=sharing)**

| 파일 | 크기 | 모델 |
|------|------|------|
| `kcelectra.pt` | ~400MB | KcELECTRA-Base |
| `soongsil.pt` | ~400MB | SoongsilBERT-Base |
| `roberta_base.pt` | ~400MB | KLUE-RoBERTa-Base |

### 설치 방법

1. 위 링크에서 모델 파일 다운로드
2. `models/` 폴더에 배치:

```
models/
├── kcelectra.pt
├── soongsil.pt
└── roberta_base.pt
```

3. 추론 실행:

```bash
python run_inference.py
```

---

## ⚠️ 주의사항

### 라이선스 제한

> **🚨 중요: 본 프로젝트는 포트폴리오 목적으로만 공개되었습니다.**
> 
> - ❌ 코드 복사 및 사용 금지
> - ❌ 상업적 이용 금지
> - ❌ 모델 파일(.pt) 사용 금지
> - ✅ 교육 목적 열람 및 참조만 가능
> 
> 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

### 기술적 요구사항

1. **GPU 권장**: CPU 환경에서도 추론 가능하나, 학습 시 GPU 필수 권장
2. **메모리**: 학습 시 12GB 이상 VRAM 권장
3. **데이터셋**: UnSmile 데이터셋은 별도 다운로드 필요
4. **모델 파일**: 용량 문제로 Git에 미포함, Google Drive에서 다운로드

### 데이터셋 저작권

- UnSmile 데이터셋은 Smilegate AI의 저작물입니다
- 데이터셋 사용 시 원저작자의 라이선스를 준수해야 합니다
- 상업적 사용 여부는 원저작자에게 확인 필요

---

## 📄 라이선스 및 사용 정책

### ⚠️ 포트폴리오 프로젝트

본 프로젝트는 **포트폴리오 및 학습 목적으로만 공개**된 프로젝트입니다.  
**[Portfolio Project License (Based on CC BY-NC-ND 4.0)](LICENSE)** 하에 배포됩니다.

### ✅ 허용 사항

| 용도 | 설명 |
|------|------|
| **소스 코드 열람** | 교육 및 학습 목적으로 코드 확인 가능 |
| **프로젝트 참조** | 포트폴리오, 이력서, 학술 논문 등에서 참조 가능 |
| **기술 분석** | 구현 방법론 및 기술 스택 학습 가능 |

### ❌ 금지 사항

| 제한 | 설명 |
|------|------|
| **상업적 사용** | 어떠한 형태의 상업적 이용도 금지 |
| **코드 복사/수정** | 코드를 복사하거나 수정하여 사용 금지 |
| **재배포** | 코드 또는 모델을 재배포 금지 |
| **파생 작업** | 본 프로젝트 기반 2차 저작물 제작 금지 |
| **모델 사용** | 학습된 모델(.pt 파일) 사용 금지 |
| **AI 학습 데이터** | 본 코드를 AI/ML 학습 데이터로 사용 금지 |

### 📌 사용 제한 사유

본 프로젝트는 개인의 연구 및 개발 노력의 결과물로, 다음과 같은 이유로 사용을 제한합니다:

1. **지적 재산권 보호** - 개인의 창작물에 대한 권리 보호
2. **포트폴리오 가치 유지** - 무분별한 복제 방지
3. **책임 소재 명확화** - 무단 사용으로 인한 법적 문제 예방
4. **상업적 이용 통제** - 영리 목적 사용에 대한 저작권자 통제

### 📖 인용 방법

본 프로젝트를 참조하거나 인용할 경우 다음과 같이 출처를 표시해야 합니다:

```
UnSmile Korean Hate Speech Detection AI
Author: wovlf02
GitHub: https://github.com/wovlf02/Emotion_AI_Model
License: Portfolio Project License (Based on CC BY-NC-ND 4.0)
Year: 2025
```

### 💼 상업적 사용 문의

상업적 사용, 라이선스 협의, 협업 제안 등은 별도 문의 바랍니다.

### ⚖️ 법적 고지

- 본 프로젝트는 "있는 그대로(AS IS)" 제공되며 어떠한 보증도 하지 않습니다
- 본 라이선스 위반 시 법적 조치가 취해질 수 있습니다
- 저작권법 및 관련 법규에 의해 보호됩니다

---

## 🔗 참고 자료

### 데이터셋

- **UnSmile 데이터셋**: [Smilegate AI GitHub](https://github.com/smilegate-ai/korean_unsmile_dataset)

### 사전학습 모델

- **KcELECTRA**: [Hugging Face](https://huggingface.co/beomi/KcELECTRA-base)
- **SoongsilBERT**: [Hugging Face](https://huggingface.co/soongsil-ai/soongsil-bert-base)
- **KLUE-RoBERTa**: [Hugging Face](https://huggingface.co/klue/roberta-base)

### 논문

- **AEDA**: [An Easier Data Augmentation Technique for Text Classification](https://arxiv.org/abs/2108.13230)
- **Asymmetric Loss**: [Asymmetric Loss For Multi-Label Classification (ICCV 2021)](https://arxiv.org/abs/2009.14119)
- **ELECTRA**: [Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555)

---

<div align="center">

**🎯 Hamming Accuracy 96.72% 달성! 목표 95% 초과! 🎉**

Made with ❤️ by [wovlf02](https://github.com/wovlf02)

</div>
