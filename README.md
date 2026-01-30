# 🎭 UnSmile 한국어 혐오 표현 탐지 AI

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **3-모델 하이브리드 앙상블을 통한 다중 라벨 혐오 표현 분류 시스템**

## 📋 목차

- [프로젝트 소개](#-프로젝트-소개)
- [주요 성과](#-주요-성과)
- [시스템 요구사항](#-시스템-요구사항)
- [설치 방법](#-설치-방법)
- [데이터셋 준비](#-데이터셋-준비)
- [사용 방법](#-사용-방법)
- [프로젝트 구조](#-프로젝트-구조)
- [모델 아키텍처](#-모델-아키텍처)
- [실험 결과](#-실험-결과)
- [문서](#-문서)
- [라이선스](#-라이선스)
- [참고 자료](#-참고-자료)

---

## 🎯 프로젝트 소개

### 배경

온라인 커뮤니티와 소셜 미디어에서 혐오 표현이 증가함에 따라, 단순 욕설 필터링을 넘어 **혐오 유형을 세분화하여 탐지**할 수 있는 AI 시스템의 필요성이 대두되었습니다.

### 목표

- **9개 혐오 카테고리** 동시 탐지 (다중 라벨 분류)
- **95% 이상의 정확도** 달성
- 실제 서비스에 적용 가능한 **고성능 모델** 개발

### 혐오 표현 카테고리

| 카테고리 | 설명 |
|---------|------|
| 여성/가족 | 여성 및 가족 관련 혐오 |
| 남성 | 남성 관련 혐오 |
| 성소수자 | LGBTQ+ 관련 혐오 |
| 인종/국적 | 인종 및 국적 관련 혐오 |
| 연령 | 연령 관련 혐오 |
| 지역 | 특정 지역 관련 혐오 |
| 종교 | 종교 관련 혐오 |
| 기타 혐오 | 기타 유형의 혐오 |
| 악플/욕설 | 일반적인 악성 댓글 |

---

## 🏆 주요 성과

| 지표 | 성능 |
|------|------|
| **Hamming Accuracy** | **96.72%** ✅ |
| **F1-Macro** | **82.91%** |
| **F1-Micro** | 81.08% |
| **Exact Match** | 74.63% |

- ✅ 목표 정확도 95% **초과 달성**
- ✅ 3개 전문화 모델 앙상블
- ✅ 클래스별 임계값 최적화 적용

---

## 💻 시스템 요구사항

### 필수 환경

- **Python**: 3.8 이상
- **CUDA**: 11.8 이상 (GPU 사용 시)
- **RAM**: 16GB 이상 권장
- **GPU VRAM**: 12GB 이상 권장

### 권장 하드웨어

- NVIDIA RTX 4070 Super 또는 동급 이상
- SSD 10GB 이상 여유 공간

---

## 🔧 설치 방법

### 1. 저장소 클론

```bash
git clone https://github.com/[username]/Emotion_Project.git
cd Emotion_Project
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
pip install -r requirements.txt
```

---

## 📦 데이터셋 준비

본 프로젝트는 **UnSmile 데이터셋**을 사용합니다.

### 1. 데이터셋 다운로드

[Smilegate AI GitHub](https://github.com/smilegate-ai/korean_unsmile_dataset)에서 다운로드

### 2. 파일 배치

```
data/raw/
├── unsmile_train.tsv
├── unsmile_dev.tsv
└── unsmile_test.tsv
```

### 3. 데이터 전처리

```bash
python -c "from src.data_loader import UnsmileDataLoader; loader = UnsmileDataLoader('./data'); loader.prepare_dataset()"
```

---

## 🚀 사용 방법

### 전체 학습 파이프라인

```bash
python run_train.py
```

**예상 소요 시간**: 약 15시간 (GPU 환경)

### 개별 실행

```bash
# 1. 모델 학습
python run_train.py

# 2. 임계값 최적화
python run_optimize_thresholds.py

# 3. 최종 평가
python run_final_inference.py

# 4. 테스트 추론
python run_inference.py
```

### 직접 실행 (src 폴더 내부)

```bash
cd src

# 모델 학습
python train.py

# 추론
python inference.py
```

---

## 📁 프로젝트 구조

```
Emotion_Project/
├── README.md                    # 프로젝트 소개
├── LICENSE                      # MIT 라이선스
├── requirements.txt             # 의존성 패키지
├── run_train.py                 # 학습 실행 스크립트
├── run_inference.py             # 추론 실행 스크립트
├── run_final_inference.py       # 최종 평가 실행 스크립트
├── run_optimize_thresholds.py   # 임계값 최적화 실행 스크립트
│
├── src/                         # 소스 코드
│   ├── train.py                 # 메인 학습 스크립트
│   ├── inference.py             # 추론 스크립트
│   ├── final_inference.py       # 최종 평가 스크립트
│   ├── optimize_thresholds.py   # 임계값 최적화
│   ├── data_loader.py           # 데이터 로딩 및 전처리
│   ├── dataset.py               # PyTorch Dataset 클래스
│   ├── model.py                 # 모델 아키텍처
│   ├── aeda_augmentation.py     # AEDA 데이터 증강
│   └── asymmetric_loss.py       # 비대칭 손실 함수
│
├── data/
│   ├── raw/                     # 원본 TSV 데이터
│   └── processed/               # 전처리된 CSV 데이터
│
├── models/                      # 학습된 모델 가중치
│   ├── kcelectra.pt
│   ├── soongsil.pt
│   └── roberta_base.pt
│
├── results/                     # 실험 결과
│   ├── final_results.json
│   ├── optimal_thresholds.json
│   └── final_test_predictions.csv
│
└── docs/                        # 문서
    ├── README.md
    ├── 01_프로젝트_개요.md
    ├── 02_데이터_분석.md
    ├── 03_모델_아키텍처.md
    ├── 04_학습_전략.md
    ├── 05_실험_결과.md
    └── presentation/
        └── FINAL_PRESENTATION.md
```

---

## 🧠 모델 아키텍처

### 3-모델 하이브리드 앙상블

| 모델 | 역할 | 특징 |
|------|------|------|
| **KcELECTRA** | 슬랭/욕설 전문가 | 한국어 인터넷 언어에 특화 |
| **SoongsilBERT** | 안정적 베이스라인 | 균형 잡힌 성능 |
| **RoBERTa-Base** | 고맥락 의미론 전문가 | 문맥 이해력 우수 |

### 핵심 기술

- **AEDA 데이터 증강**: 소수 클래스 오버샘플링
- **Asymmetric Loss**: 클래스 불균형 대응
- **클래스별 임계값 최적화**: F1-Score 최대화
- **가중 소프트 보팅**: 모델 예측 결합

---

## 📊 실험 결과

### 모델별 성능

| 모델 | F1-Macro | Hamming Acc |
|------|----------|-------------|
| KcELECTRA | 80.2% | 95.8% |
| SoongsilBERT | 79.5% | 95.6% |
| RoBERTa-Base | 78.8% | 95.4% |
| **앙상블** | **82.91%** | **96.72%** |

### 앙상블 효과

- 개별 모델 대비 **+2.7%p** F1-Macro 향상
- 임계값 최적화로 추가 **+1.2%p** 개선

---

## 📚 문서

| 문서 | 설명 |
|------|------|
| [프로젝트 개요](docs/01_프로젝트_개요.md) | 프로젝트 배경 및 목표 |
| [데이터 분석](docs/02_데이터_분석.md) | EDA 및 전처리 전략 |
| [모델 아키텍처](docs/03_모델_아키텍처.md) | 3-모델 앙상블 설계 |
| [학습 전략](docs/04_학습_전략.md) | 손실 함수 및 최적화 |
| [실험 결과](docs/05_실험_결과.md) | 성능 평가 및 분석 |
| [발표 자료](docs/presentation/FINAL_PRESENTATION.md) | 프레젠테이션 자료 |

### 다운로드 링크

- **학습된 모델**: [Google Drive](https://drive.google.com/drive/folders/1Noow6HkhI6hkAuggptroiNmbUVGDbu1u?usp=sharing) (kcelectra.pt, soongsil.pt, roberta_base.pt)

---

## ⚠️ 주의사항

- 학습된 모델 파일(`.pt`)은 용량이 커서 GitHub에 포함되지 않습니다.
- 모델을 사용하려면 다음 방법 중 하나를 선택하세요:
  1. **직접 학습**: `python run_train.py` (약 15시간 소요)
  2. **사전 학습 모델 다운로드**: [Google Drive](https://drive.google.com/drive/folders/1Noow6HkhI6hkAuggptroiNmbUVGDbu1u?usp=sharing)에서 다운로드
- GPU 환경에서 학습을 권장합니다 (CPU 학습 시 수일 소요).

### 사전 학습 모델 사용 방법

1. [Google Drive 링크](https://drive.google.com/drive/folders/1Noow6HkhI6hkAuggptroiNmbUVGDbu1u?usp=sharing)에서 모델 파일 다운로드
2. 다운로드한 파일들을 `models/` 폴더에 배치:
   ```
   models/
   ├── kcelectra.pt
   ├── soongsil.pt
   └── roberta_base.pt
   ```
3. 추론 실행: `python run_inference.py`

---

## 📄 라이선스

이 프로젝트는 [MIT License](LICENSE) 하에 배포됩니다.

---

## 🔗 참고 자료

- **UnSmile 데이터셋**: [Smilegate AI GitHub](https://github.com/smilegate-ai/korean_unsmile_dataset)
- **KcELECTRA**: [Hugging Face](https://huggingface.co/beomi/KcELECTRA-base)
- **SoongsilBERT**: [Hugging Face](https://huggingface.co/soongsil-ai/soongsil-bert-base)
- **KLUE-RoBERTa**: [Hugging Face](https://huggingface.co/klue/roberta-base)

---

## 👥 기여자

- 프로젝트 개발 및 문서화

---

**🎯 Hamming Accuracy 96.72% 달성! 🎉**
