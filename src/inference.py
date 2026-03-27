"""
학습된 앙상블 모델로 테스트 데이터 추론
최적 임계값 적용하여 최종 예측
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
    """단일 모델 로드 및 예측 (각 모델의 토크나이저 사용)"""

    model_name = model_config['name']
    model_path = f"./models/{model_name}.pt"

    if not os.path.exists(model_path):
        logger.warning(f"⚠️ {model_path} 파일을 찾을 수 없습니다.")
        return None

    logger.info(f"📦 {model_name} 로딩 중...")

    # 모델별 토크나이저 생성
    tokenizer = AutoTokenizer.from_pretrained(model_config['hf_name'])

    # 모델별 데이터로더 생성
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

    logger.info(f"✓ {model_name} 예측 완료: {all_probs.shape}")

    return all_probs, all_labels


def ensemble_predict(models_config: list, test_df: pd.DataFrame, label_columns: list, device: str):
    """앙상블 예측 (가중 평균)"""

    logger.info("\n" + "="*80)
    logger.info("🎯 앙상블 예측 시작")
    logger.info("="*80)

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

    # 앙상블 (단순 평균)
    ensemble_probs = np.mean(all_model_probs, axis=0)

    logger.info(f"\n✅ 앙상블 완료: {len(all_model_probs)}개 모델 평균")
    logger.info("="*80 + "\n")

    return ensemble_probs, labels


def apply_optimal_thresholds(probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """최적 임계값 적용하여 예측"""
    preds = np.zeros_like(probs, dtype=int)

    for i in range(probs.shape[1]):
        preds[:, i] = (probs[:, i] >= thresholds[i]).astype(int)

    return preds


def evaluate_predictions(labels: np.ndarray, preds: np.ndarray, label_names: list):
    """예측 결과 평가"""

    logger.info("\n" + "="*80)
    logger.info("📊 평가 결과")
    logger.info("="*80)

    # 전체 메트릭
    f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
    f1_micro = f1_score(labels, preds, average='micro', zero_division=0)
    exact_match = accuracy_score(labels, preds)
    hamming_acc = 1 - hamming_loss(labels, preds)

    logger.info(f"\n🏆 전체 성능:")
    logger.info(f"  F1-Macro: {f1_macro:.4f} ({f1_macro*100:.2f}%)")
    logger.info(f"  F1-Micro: {f1_micro:.4f} ({f1_micro*100:.2f}%)")
    logger.info(f"  Exact Match: {exact_match:.4f} ({exact_match*100:.2f}%)")
    logger.info(f"  Hamming Accuracy: {hamming_acc:.4f} ({hamming_acc*100:.2f}%)")

    # 클래스별 F1
    logger.info(f"\n📈 클래스별 F1-Score:")
    class_f1_scores = f1_score(labels, preds, average=None, zero_division=0)

    for i, (name, score) in enumerate(zip(label_names, class_f1_scores)):
        logger.info(f"  {name:20s}: {score:.4f}")

    logger.info("="*80 + "\n")

    return {
        'f1_macro': float(f1_macro),
        'f1_micro': float(f1_micro),
        'exact_match': float(exact_match),
        'hamming_accuracy': float(hamming_acc),
        'class_f1_scores': {name: float(score) for name, score in zip(label_names, class_f1_scores)}
    }


def save_predictions(test_df: pd.DataFrame, preds: np.ndarray, probs: np.ndarray, label_columns: list):
    """예측 결과 저장"""

    # 예측 결과 DataFrame 생성
    results_df = test_df[['text']].copy()

    # 예측 레이블 추가
    for i, col in enumerate(label_columns):
        results_df[f'pred_{col}'] = preds[:, i]
        results_df[f'prob_{col}'] = probs[:, i]

    # 실제 레이블 추가
    for i, col in enumerate(label_columns):
        results_df[f'true_{col}'] = test_df[col].values

    # 저장
    output_path = "./results/test_predictions.csv"
    os.makedirs("./results", exist_ok=True)
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')

    logger.info(f"✅ 예측 결과 저장: {output_path}")


def main():
    """메인 추론 파이프라인"""

    logger.info("\n" + "="*80)
    logger.info("🔮 UnSmile 테스트 데이터 추론")
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

    # 2. 학습 결과 로드
    logger.info("\n📊 Step 2: 학습 결과 로딩")

    # continue_results.json을 우선적으로 사용
    results_path = "./results/continue_results.json"
    if not os.path.exists(results_path):
        results_path = "./results/final_results.json"

    if not os.path.exists(results_path):
        logger.error(f"❌ 학습 결과 파일을 찾을 수 없습니다.")
        logger.error("먼저 train_continue.py를 실행하여 모델을 학습시켜주세요.")
        return

    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    logger.info(f"✓ 결과 파일 로드: {results_path}")
    optimal_thresholds = np.array(results['optimal_thresholds'])
    logger.info(f"✓ 최적 임계값: {optimal_thresholds}")

    # 3. 모델 설정 (3개 모델 앙상블)
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

    logger.info(f"✓ 앙상블 모델: {len(models_config)}개")
    for config in models_config:
        logger.info(f"  - {config['name']}: {config['hf_name']}")

    # 4. 테스트 데이터로더 생성 (첫 번째 모델 토크나이저 사용)
    logger.info("\n📊 Step 3: 데이터로더 생성")
    tokenizer = AutoTokenizer.from_pretrained(models_config[0]['hf_name'])

    _, _, test_loader = create_dataloaders(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        tokenizer=tokenizer,
        label_columns=label_columns,
        batch_size=16,
        max_length=128
    )

    # 5. 앙상블 예측
    logger.info("\n📊 Step 4: 앙상블 예측")
    ensemble_probs, test_labels = ensemble_predict(models_config, test_df, label_columns, device)

    # 6. 최적 임계값 적용
    logger.info("\n📊 Step 5: 최적 임계값 적용")
    ensemble_preds = apply_optimal_thresholds(ensemble_probs, optimal_thresholds)

    logger.info(f"✓ 예측 완료: {ensemble_preds.shape}")
    logger.info(f"✓ 긍정 예측 비율: {ensemble_preds.mean()*100:.2f}%")

    # 7. 평가
    logger.info("\n📊 Step 6: 최종 평가")
    final_metrics = evaluate_predictions(test_labels, ensemble_preds, label_columns)

    # 8. 결과 저장
    logger.info("\n📊 Step 7: 결과 저장")
    save_predictions(test_df, ensemble_preds, ensemble_probs, label_columns)

    # 최종 메트릭 저장
    final_metrics['optimal_thresholds'] = optimal_thresholds.tolist()

    with open("./results/test_results.json", 'w', encoding='utf-8') as f:
        json.dump(final_metrics, f, indent=2, ensure_ascii=False)

    logger.info(f"✅ 테스트 결과 저장: ./results/test_results.json")

    # 9. 목표 달성 여부 확인
    logger.info("\n" + "="*80)
    logger.info("🎯 목표 달성 여부")
    logger.info("="*80)

    target_accuracy = 0.95
    achieved = final_metrics['hamming_accuracy'] >= target_accuracy

    if achieved:
        logger.info(f"✅ 목표 달성! Hamming Accuracy {final_metrics['hamming_accuracy']:.4f} >= {target_accuracy}")
    else:
        gap = target_accuracy - final_metrics['hamming_accuracy']
        logger.info(f"⚠️ 목표 미달성. 부족: {gap:.4f} ({gap*100:.2f}%p)")
        logger.info(f"   현재: {final_metrics['hamming_accuracy']:.4f}")
        logger.info(f"   목표: {target_accuracy}")

    logger.info("="*80 + "\n")


if __name__ == "__main__":
    main()
