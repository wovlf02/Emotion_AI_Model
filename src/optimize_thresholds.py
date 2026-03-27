"""
클래스별 최적 임계값 탐색
검증 데이터로 F1-Macro를 최대화하는 임계값 찾기
"""

import os
import torch
import numpy as np
import pandas as pd
import json
import logging
from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.metrics import f1_score
from scipy.optimize import minimize

from data_loader import UnsmileDataLoader
from dataset import create_dataloaders
from model import create_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model_and_predict_val(model_config: dict, val_df: pd.DataFrame, label_columns: list, device: str):
    """단일 모델로 검증 데이터 예측"""

    model_name = model_config['name']
    model_path = f"./models/{model_name}.pt"

    if not os.path.exists(model_path):
        logger.warning(f"⚠️ {model_path} 파일을 찾을 수 없습니다.")
        return None

    logger.info(f"📦 {model_name} 로딩 중...")

    # 토크나이저
    tokenizer = AutoTokenizer.from_pretrained(model_config['hf_name'])

    # 데이터로더
    _, val_loader, _ = create_dataloaders(
        train_df=val_df[:1],  # dummy
        val_df=val_df,
        test_df=val_df[:1],   # dummy
        tokenizer=tokenizer,
        label_columns=label_columns,
        batch_size=16,
        max_length=128
    )

    # 모델 생성
    model = create_model(
        model_name=model_config['hf_name'],
        num_labels=9,
        use_qlora=model_config.get('use_qlora', False)
    )

    # 가중치 로드
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    # 예측
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Predicting {model_name}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            logits = outputs['logits']
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return all_probs, all_labels


def optimize_thresholds_per_class(ensemble_probs: np.ndarray, labels: np.ndarray, label_columns: list):
    """클래스별 최적 임계값 탐색 (Grid Search)"""

    logger.info("\n" + "="*80)
    logger.info("🔍 클래스별 최적 임계값 탐색")
    logger.info("="*80)

    n_classes = ensemble_probs.shape[1]
    optimal_thresholds = np.zeros(n_classes)

    # 클래스별로 최적 임계값 탐색
    for i in range(n_classes):
        best_f1 = 0
        best_thresh = 0.5

        # 0.1부터 0.9까지 0.01 간격으로 탐색
        for thresh in np.arange(0.1, 0.91, 0.01):
            preds = (ensemble_probs[:, i] >= thresh).astype(int)
            f1 = f1_score(labels[:, i], preds, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh

        optimal_thresholds[i] = best_thresh
        logger.info(f"  {label_columns[i]:20s}: {best_thresh:.2f} (F1: {best_f1:.4f})")

    logger.info("="*80 + "\n")

    return optimal_thresholds


def optimize_thresholds_global(ensemble_probs: np.ndarray, labels: np.ndarray):
    """전역 최적 임계값 탐색 (F1-Macro 최대화)"""

    logger.info("\n" + "="*80)
    logger.info("🎯 전역 최적 임계값 탐색 (F1-Macro 최대화)")
    logger.info("="*80)

    def objective(thresholds):
        """F1-Macro를 최대화 (음수로 변환하여 최소화 문제로)"""
        thresholds = np.clip(thresholds, 0.05, 0.95)

        preds = np.zeros_like(ensemble_probs, dtype=int)
        for i in range(len(thresholds)):
            preds[:, i] = (ensemble_probs[:, i] >= thresholds[i]).astype(int)

        f1 = f1_score(labels, preds, average='macro', zero_division=0)
        return -f1  # 최소화 문제로 변환

    # 초기값: 0.5
    initial_thresholds = np.ones(ensemble_probs.shape[1]) * 0.5

    # Nelder-Mead 최적화
    result = minimize(
        objective,
        initial_thresholds,
        method='Nelder-Mead',
        options={'maxiter': 1000, 'disp': True}
    )

    optimal_thresholds = np.clip(result.x, 0.05, 0.95)
    best_f1 = -result.fun

    logger.info(f"\n✅ 최적화 완료!")
    logger.info(f"  최적 F1-Macro: {best_f1:.4f}")
    logger.info(f"  최적 임계값: {optimal_thresholds}")
    logger.info("="*80 + "\n")

    return optimal_thresholds


def main():
    """메인 임계값 최적화 파이프라인"""

    logger.info("\n" + "="*80)
    logger.info("🎯 클래스별 최적 임계값 탐색")
    logger.info("="*80)

    # 디바이스
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"\n💻 Device: {device}")

    # 1. 데이터 로드
    logger.info("\n📊 Step 1: 데이터 로딩")
    data_loader = UnsmileDataLoader(data_dir="./data")
    train_df, val_df, test_df = data_loader.load_processed_data()
    label_columns = data_loader.label_columns

    logger.info(f"✓ Val 샘플 수: {len(val_df)}")

    # 2. 모델 설정 (3개 모델)
    models_config = [
        {
            'name': 'kcelectra',
            'hf_name': 'beomi/KcELECTRA-base',
            'use_qlora': False
        },
        {
            'name': 'soongsil',
            'hf_name': 'snunlp/KR-SBERT-V40K-klueNLI-augSTS',
            'use_qlora': False
        },
        {
            'name': 'roberta_base',
            'hf_name': 'klue/roberta-base',
            'use_qlora': False
        }
    ]

    # 3. 각 모델로 검증 데이터 예측
    logger.info("\n📊 Step 2: 각 모델로 검증 데이터 예측")
    all_model_probs = []
    labels = None

    for config in models_config:
        result = load_model_and_predict_val(config, val_df, label_columns, device)

        if result is not None:
            probs, lbls = result
            all_model_probs.append(probs)
            if labels is None:
                labels = lbls

    if len(all_model_probs) == 0:
        raise ValueError("예측 가능한 모델이 없습니다!")

    # 4. 앙상블 (평균)
    logger.info("\n📊 Step 3: 앙상블 (단순 평균)")
    ensemble_probs = np.mean(all_model_probs, axis=0)
    logger.info(f"✓ 앙상블 완료: {len(all_model_probs)}개 모델")

    # 5. 클래스별 최적 임계값 탐색
    logger.info("\n📊 Step 4: 클래스별 최적 임계값 탐색")
    optimal_thresholds_class = optimize_thresholds_per_class(ensemble_probs, labels, label_columns)

    # 6. 전역 최적화 (F1-Macro 최대화)
    logger.info("\n📊 Step 5: 전역 최적화 (F1-Macro 최대화)")
    optimal_thresholds_global = optimize_thresholds_global(ensemble_probs, labels)

    # 7. 두 방법 비교
    logger.info("\n📊 Step 6: 두 방법 비교")

    # 클래스별 최적화 결과
    preds_class = np.zeros_like(ensemble_probs, dtype=int)
    for i in range(len(optimal_thresholds_class)):
        preds_class[:, i] = (ensemble_probs[:, i] >= optimal_thresholds_class[i]).astype(int)

    f1_class = f1_score(labels, preds_class, average='macro', zero_division=0)

    # 전역 최적화 결과
    preds_global = np.zeros_like(ensemble_probs, dtype=int)
    for i in range(len(optimal_thresholds_global)):
        preds_global[:, i] = (ensemble_probs[:, i] >= optimal_thresholds_global[i]).astype(int)

    f1_global = f1_score(labels, preds_global, average='macro', zero_division=0)

    logger.info(f"\n클래스별 최적화: F1-Macro {f1_class:.4f}")
    logger.info(f"전역 최적화:     F1-Macro {f1_global:.4f}")

    # 더 좋은 방법 선택
    if f1_global >= f1_class:
        logger.info(f"\n✅ 전역 최적화 방법 선택 (F1: {f1_global:.4f})")
        final_thresholds = optimal_thresholds_global
        final_f1 = f1_global
    else:
        logger.info(f"\n✅ 클래스별 최적화 방법 선택 (F1: {f1_class:.4f})")
        final_thresholds = optimal_thresholds_class
        final_f1 = f1_class

    # 8. 최종 임계값 출력
    logger.info("\n" + "="*80)
    logger.info("🎯 최종 최적 임계값")
    logger.info("="*80)

    for i, (name, thresh) in enumerate(zip(label_columns, final_thresholds)):
        logger.info(f"  {name:20s}: {thresh:.4f}")

    logger.info(f"\n✅ 최종 F1-Macro: {final_f1:.4f}")
    logger.info("="*80 + "\n")

    # 9. 결과 저장
    result_dict = {
        'optimal_thresholds': final_thresholds.tolist(),
        'threshold_dict': {name: float(thresh) for name, thresh in zip(label_columns, final_thresholds)},
        'val_f1_macro': float(final_f1),
        'method': 'global_optimization' if f1_global >= f1_class else 'class_wise_optimization'
    }

    output_path = "./results/optimal_thresholds.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, indent=2, ensure_ascii=False)

    logger.info(f"✅ 최적 임계값 저장: {output_path}")

    logger.info("\n" + "="*80)
    logger.info("✅ 임계값 최적화 완료!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
