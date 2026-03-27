"""
최적화된 임계값으로 테스트 데이터 최종 평가
"""

import os
import torch
import numpy as np
import pandas as pd
import json
import logging
from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, accuracy_score, hamming_loss

from data_loader import UnsmileDataLoader
from dataset import create_dataloaders
from model import create_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model_and_predict(model_config: dict, test_df: pd.DataFrame, label_columns: list, device: str):
    """단일 모델로 테스트 데이터 예측"""

    model_name = model_config['name']
    model_path = f"./models/{model_name}.pt"

    if not os.path.exists(model_path):
        logger.warning(f"⚠️ {model_path} 파일을 찾을 수 없습니다.")
        return None

    logger.info(f"📦 {model_name} 로딩 중...")

    # 토크나이저
    tokenizer = AutoTokenizer.from_pretrained(model_config['hf_name'])

    # 데이터로더
    _, _, test_loader = create_dataloaders(
        train_df=test_df[:1],  # dummy
        val_df=test_df[:1],    # dummy
        test_df=test_df,
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
        for batch in tqdm(test_loader, desc=f"Predicting {model_name}"):
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


def apply_optimal_thresholds(probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """최적 임계값 적용하여 예측"""
    preds = np.zeros_like(probs, dtype=int)

    for i in range(probs.shape[1]):
        preds[:, i] = (probs[:, i] >= thresholds[i]).astype(int)

    return preds


def evaluate_predictions(labels: np.ndarray, preds: np.ndarray, label_names: list):
    """예측 결과 평가"""

    logger.info("\n" + "="*80)
    logger.info("📊 테스트 데이터 최종 평가 결과")
    logger.info("="*80)

    # 전체 메트릭
    f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
    f1_micro = f1_score(labels, preds, average='micro', zero_division=0)
    exact_match = accuracy_score(labels, preds)
    hamming_acc = 1 - hamming_loss(labels, preds)

    logger.info(f"\n🏆 전체 성능:")
    logger.info(f"  F1-Macro:        {f1_macro:.4f} ({f1_macro*100:.2f}%)")
    logger.info(f"  F1-Micro:        {f1_micro:.4f} ({f1_micro*100:.2f}%)")
    logger.info(f"  Exact Match:     {exact_match:.4f} ({exact_match*100:.2f}%)")
    logger.info(f"  Hamming Acc:     {hamming_acc:.4f} ({hamming_acc*100:.2f}%) ⭐")

    # 클래스별 F1
    logger.info(f"\n📈 클래스별 F1-Score:")
    class_f1_scores = f1_score(labels, preds, average=None, zero_division=0)

    for i, (name, score) in enumerate(zip(label_names, class_f1_scores)):
        bar = "█" * int(score * 20)
        logger.info(f"  {name:20s}: {score:.4f} {bar}")

    logger.info("="*80 + "\n")

    # 목표 달성 여부
    target = 0.98
    if hamming_acc >= target:
        logger.info(f"🎉 목표 달성! Hamming Accuracy {hamming_acc:.4f} >= {target} ✅")
    elif hamming_acc >= 0.97:
        logger.info(f"👍 거의 달성! Hamming Accuracy {hamming_acc:.4f} (목표: {target})")
        logger.info(f"   부족: {(target - hamming_acc)*100:.2f}%p")
    else:
        logger.info(f"⚠️ 추가 개선 필요. Hamming Accuracy {hamming_acc:.4f} (목표: {target})")
        logger.info(f"   부족: {(target - hamming_acc)*100:.2f}%p")

    return {
        'f1_macro': float(f1_macro),
        'f1_micro': float(f1_micro),
        'exact_match': float(exact_match),
        'hamming_accuracy': float(hamming_acc),
        'class_f1_scores': {name: float(score) for name, score in zip(label_names, class_f1_scores)}
    }


def main():
    """메인 최종 평가 파이프라인"""

    logger.info("\n" + "="*80)
    logger.info("🎯 최적 임계값으로 테스트 데이터 최종 평가")
    logger.info("="*80)

    # 디바이스
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"\n💻 Device: {device}")

    # 1. 데이터 로드
    logger.info("\n📊 Step 1: 데이터 로딩")
    data_loader = UnsmileDataLoader(data_dir="./data")
    train_df, val_df, test_df = data_loader.load_processed_data()
    label_columns = data_loader.label_columns

    logger.info(f"✓ Test 샘플 수: {len(test_df)}")

    # 2. 최적 임계값 로드
    logger.info("\n📊 Step 2: 최적 임계값 로딩")

    threshold_path = "./results/optimal_thresholds.json"
    if not os.path.exists(threshold_path):
        logger.error(f"❌ {threshold_path} 파일을 찾을 수 없습니다.")
        logger.error("먼저 optimize_thresholds.py를 실행해주세요.")
        return

    with open(threshold_path, 'r', encoding='utf-8') as f:
        threshold_data = json.load(f)

    optimal_thresholds = np.array(threshold_data['optimal_thresholds'])

    logger.info(f"✓ 최적 임계값 로드 완료")
    for name, thresh in zip(label_columns, optimal_thresholds):
        logger.info(f"  {name:20s}: {thresh:.4f}")

    # 3. 모델 설정 (3개 모델)
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

    # 4. 각 모델로 테스트 데이터 예측
    logger.info("\n📊 Step 3: 각 모델로 테스트 데이터 예측")
    all_model_probs = []
    labels = None

    for config in models_config:
        result = load_model_and_predict(config, test_df, label_columns, device)

        if result is not None:
            probs, lbls = result
            all_model_probs.append(probs)
            if labels is None:
                labels = lbls

    if len(all_model_probs) == 0:
        raise ValueError("예측 가능한 모델이 없습니다!")

    # 5. 앙상블 (단순 평균)
    logger.info("\n📊 Step 4: 앙상블 (단순 평균)")

    # 단순 평균 (가중치 없음)
    ensemble_probs = np.mean(all_model_probs, axis=0)
    logger.info(f"✓ 앙상블 완료: {len(all_model_probs)}개 모델 (단순 평균)")

    # 6. 최적 임계값 적용
    logger.info("\n📊 Step 5: 최적 임계값 적용")
    ensemble_preds = apply_optimal_thresholds(ensemble_probs, optimal_thresholds)

    logger.info(f"✓ 예측 완료: {ensemble_preds.shape}")
    logger.info(f"✓ 긍정 예측 비율: {ensemble_preds.mean()*100:.2f}%")

    # 7. 최종 평가
    logger.info("\n📊 Step 6: 최종 평가")
    final_metrics = evaluate_predictions(labels, ensemble_preds, label_columns)

    # 8. 결과 저장
    logger.info("\n📊 Step 7: 결과 저장")

    final_metrics['optimal_thresholds'] = optimal_thresholds.tolist()
    final_metrics['models'] = [m['name'] for m in models_config]
    final_metrics['ensemble_method'] = 'simple_average'

    output_path = "./results/final_test_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_metrics, f, indent=2, ensure_ascii=False)

    logger.info(f"✅ 최종 결과 저장: {output_path}")

    # 9. 예측 결과 CSV 저장
    results_df = test_df[['text']].copy()

    # 예측 레이블 추가
    for i, col in enumerate(label_columns):
        results_df[f'pred_{col}'] = ensemble_preds[:, i]
        results_df[f'prob_{col}'] = ensemble_probs[:, i]

    # 실제 레이블 추가
    for i, col in enumerate(label_columns):
        results_df[f'true_{col}'] = test_df[col].values

    csv_path = "./results/final_test_predictions.csv"
    results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    logger.info(f"✅ 예측 결과 CSV 저장: {csv_path}")

    logger.info("\n" + "="*80)
    logger.info("✅ 최종 평가 완료!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
