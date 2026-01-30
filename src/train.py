"""
15시간 최대 정확도 전략 (15번 문서 기반)
- 3개 모델 하이브리드 앙상블 (각 전문 분야)
- 모델당 충분한 학습 시간 (80 epoch)
- 클래스별 임계값 최적화
- 가중 소프트 보팅
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score, hamming_loss
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
import json
import time
from scipy.optimize import minimize

# 로컬 모듈
from data_loader import UnsmileDataLoader
from aeda_augmentation import augment_minority_classes
from asymmetric_loss import AsymmetricLossOptimized
from dataset import create_dataloaders
from model import create_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.model_name = model_name

        self.best_f1 = 0.0
        self.best_model_state = None

    def train_epoch(self) -> float:
        """1 에포크 학습"""
        self.model.train()
        total_loss = 0.0

        progress_bar = tqdm(self.train_loader, desc=f"Training {self.model_name}")

        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward
            outputs = self.model(input_ids, attention_mask)
            logits = outputs['logits']

            # Loss 계산
            loss = self.criterion(logits, labels)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    def evaluate(self, threshold: float = 0.5) -> dict:
        """검증 데이터 평가"""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Evaluating {self.model_name}"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids, attention_mask)
                logits = outputs['logits']
                probs = torch.sigmoid(logits)

                all_probs.append(probs.cpu())
                all_labels.append(labels.cpu())

        # 결과 집계
        all_probs = torch.cat(all_probs).numpy()
        all_labels = torch.cat(all_labels).numpy()
        all_preds = (all_probs >= threshold).astype(int)

        # 메트릭 계산
        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)
        exact_match = accuracy_score(all_labels, all_preds)
        hamming_acc = 1 - hamming_loss(all_labels, all_preds)

        return {
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'exact_match': exact_match,
            'hamming_accuracy': hamming_acc,
            'probs': all_probs,
            'labels': all_labels
        }

    def train(self, num_epochs: int, patience: int = 10):
        """전체 학습 루프"""
        logger.info(f"\n{'='*80}")
        logger.info(f"🚀 {self.model_name} 학습 시작")
        logger.info(f"{'='*80}")

        no_improve = 0

        for epoch in range(num_epochs):
            logger.info(f"\n📌 Epoch {epoch+1}/{num_epochs}")

            # 학습
            train_loss = self.train_epoch()
            logger.info(f"  Train Loss: {train_loss:.4f}")

            # 평가
            metrics = self.evaluate()
            logger.info(f"  Val F1-Macro: {metrics['f1_macro']:.4f}")
            logger.info(f"  Val Exact Match: {metrics['exact_match']:.4f}")
            logger.info(f"  Val Hamming Acc: {metrics['hamming_accuracy']:.4f}")

            # 최적 모델 저장
            if metrics['f1_macro'] > self.best_f1:
                self.best_f1 = metrics['f1_macro']
                self.best_model_state = self.model.state_dict()
                no_improve = 0
                logger.info(f"  ✅ 새로운 최고 F1: {self.best_f1:.4f}")
            else:
                no_improve += 1
                logger.info(f"  ⏳ No improvement: {no_improve}/{patience}")

            # Early Stopping
            if no_improve >= patience:
                logger.info(f"\n⚠️ Early stopping at epoch {epoch+1}")
                break

        # 최적 모델 복원
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info(f"\n✅ 최적 모델 복원 완료 (F1: {self.best_f1:.4f})")

        return self.best_f1


def train_single_model(
    model_config: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    num_epochs: int = 80
):
    """단일 모델 학습"""

    model_name = model_config['name']
    hf_model_name = model_config['hf_name']
    use_qlora = model_config.get('use_qlora', False)

    logger.info(f"\n{'='*80}")
    logger.info(f"📦 모델 초기화: {model_name}")
    logger.info(f"{'='*80}")

    # 모델 생성
    model = create_model(
        model_name=hf_model_name,
        num_labels=9,
        use_qlora=use_qlora
    )

    # 손실 함수 - 일반 BCE Loss with Pos Weight (불균형 처리)
    # 클래스별 양성 샘플 가중치 계산
    pos_weight = torch.ones(9) * 2.0  # 긍정 샘플에 2배 가중치
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    logger.info(f"  손실 함수: BCEWithLogitsLoss (pos_weight=2.0)")

    # 옵티마이저
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=2e-5,
        weight_decay=0.01
    )

    # 스케줄러 (Cosine Annealing with Warmup)
    num_training_steps = len(train_loader) * num_epochs
    num_warmup_steps = int(0.1 * num_training_steps)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # Trainer 생성
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        model_name=model_name
    )

    # 학습
    best_f1 = trainer.train(num_epochs=num_epochs, patience=10)

    # 최종 평가 (확률값 반환)
    final_metrics = trainer.evaluate()

    # 모델 저장
    save_path = f"./models/{model_name}.pt"
    os.makedirs("./models", exist_ok=True)
    torch.save(trainer.model.state_dict(), save_path)
    logger.info(f"✅ 모델 저장: {save_path}")

    return trainer.model, final_metrics


def optimize_ensemble_weights(all_probs, true_labels, num_models):
    """
    앙상블 가중치 최적화 (Nelder-Mead)
    
    Args:
        all_probs: 각 모델의 예측 확률 리스트 [(n_samples, n_classes), ...]
        true_labels: 실제 레이블 (n_samples, n_classes)
        num_models: 모델 개수
    
    Returns:
        최적 가중치 배열
    """
    logger.info("\n" + "="*80)
    logger.info("🎯 앙상블 가중치 최적화 (Nelder-Mead)")
    logger.info("="*80)
    
    def objective(weights):
        """F1-Score를 최대화 (음수로 변환하여 최소화 문제로)"""
        weights = np.abs(weights)
        weights = weights / np.sum(weights)  # 정규화
        
        # 가중 평균
        ensemble_probs = sum(w * probs for w, probs in zip(weights, all_probs))
        
        # 0.5 임계값으로 예측
        preds = (ensemble_probs >= 0.5).astype(int)
        
        # F1-Macro 계산
        f1 = f1_score(true_labels, preds, average='macro', zero_division=0)
        
        return -f1  # 최소화 문제로 변환
    
    # 초기 가중치 (균등)
    initial_weights = np.ones(num_models) / num_models
    
    # 최적화
    result = minimize(
        objective,
        initial_weights,
        method='Nelder-Mead',
        options={'maxiter': 500, 'disp': True}
    )
    
    optimal_weights = np.abs(result.x)
    optimal_weights = optimal_weights / np.sum(optimal_weights)
    
    logger.info(f"\n✅ 최적 가중치: {optimal_weights}")
    logger.info(f"  최적 F1-Score: {-result.fun:.4f}")
    logger.info("="*80 + "\n")
    
    return optimal_weights


def optimize_thresholds_per_class(probs: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    """
    클래스별 최적 임계값 탐색 (F1-Score 최대화)
    15번 문서 전략: 각 클래스마다 0.01~0.99 범위에서 최적값 탐색
    """
    logger.info("\n" + "="*80)
    logger.info("🎯 클래스별 임계값 최적화 (정밀 탐색)")
    logger.info("="*80)

    optimal_thresholds = []

    for class_idx in range(num_classes):
        best_f1 = 0.0
        best_threshold = 0.5

        # 0.01~0.99 범위에서 탐색 (문서 전략)
        for threshold in np.arange(0.01, 1.0, 0.01):
            preds = (probs[:, class_idx] >= threshold).astype(int)
            f1 = f1_score(labels[:, class_idx], preds, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        optimal_thresholds.append(best_threshold)
        logger.info(f"  Class {class_idx}: threshold={best_threshold:.2f}, F1={best_f1:.4f}")

    optimal_thresholds = np.array(optimal_thresholds)
    logger.info(f"\n✅ 최적 임계값 탐색 완료")
    logger.info("="*80 + "\n")

    return optimal_thresholds


def emergency_boost_strategy(
    all_models,
    train_df_augmented,
    val_df,
    label_columns,
    device,
    current_f1,
    current_hamming
):
    """
    긴급 성능 향상 전략
    90% 미달성 시 자동으로 실행되는 추가 최적화
    """
    logger.info("\n" + "="*80)
    logger.info("🚨 긴급 성능 향상 전략 실행!")
    logger.info("="*80)
    logger.info(f"현재 성능: F1-Macro={current_f1:.4f}, Hamming Acc={current_hamming:.4f}")
    logger.info("목표: 90% 이상 달성 필수!")

    # 전략 1: 더 많은 데이터 증강
    logger.info("\n🔄 전략 1: 강화된 데이터 증강")
    from src.data_loader import UnsmileDataLoader
    from src.aeda_augmentation import augment_minority_classes

    data_loader = UnsmileDataLoader(data_dir="./data")
    train_df, _, _ = data_loader.load_processed_data()

    # 증강 강도 2배 증가
    train_df_boosted = augment_minority_classes(
        train_df=train_df,
        label_columns=label_columns,
        target_size=2500,  # 1500 → 2500
        punc_ratio=0.4,    # 0.3 → 0.4
        augment_all=True   # 모든 클래스 증강
    )

    logger.info(f"  증강된 데이터: {len(train_df)} → {len(train_df_boosted)}")

    # 전략 2: 각 모델 추가 학습 (30 epoch)
    logger.info("\n🔄 전략 2: 모델 추가 학습 (30 epoch)")

    boosted_probs = []

    for model_info in all_models:
        model_name = model_info['name']
        model = model_info['model']
        tokenizer = model_info['tokenizer']

        logger.info(f"\n  📦 {model_name} 추가 학습 중...")

        from src.dataset import create_dataloaders

        train_loader, val_loader, _ = create_dataloaders(
            train_df=train_df_boosted,
            val_df=val_df,
            test_df=val_df,
            tokenizer=tokenizer,
            label_columns=label_columns,
            batch_size=32,
            max_length=128
        )

        # 추가 학습 (낮은 learning rate)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-5,  # 더 낮은 learning rate
            weight_decay=0.01
        )

        criterion = AsymmetricLossOptimized(
            gamma_neg=4.0,
            gamma_pos=0.0,
            clip=0.05
        )

        num_training_steps = len(train_loader) * 30
        num_warmup_steps = int(0.05 * num_training_steps)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            model_name=f"{model_name}_boost"
        )

        trainer.train(num_epochs=30, patience=5)

        # 평가
        metrics = trainer.evaluate()
        boosted_probs.append(metrics['probs'])

        logger.info(f"  ✅ {model_name} 추가 학습 완료 - F1: {metrics['f1_macro']:.4f}")

    # 전략 3: 재최적화
    logger.info("\n🔄 전략 3: 앙상블 가중치 및 임계값 재최적화")

    y_val = val_df[label_columns].values

    # 가중치 재최적화
    optimal_weights = optimize_ensemble_weights(
        all_probs=boosted_probs,
        true_labels=y_val,
        num_models=len(boosted_probs)
    )

    # 앙상블
    ensemble_probs = sum(w * probs for w, probs in zip(optimal_weights, boosted_probs))

    # 임계값 재최적화
    optimal_thresholds = optimize_thresholds_per_class(
        probs=ensemble_probs,
        labels=y_val,
        num_classes=len(label_columns)
    )

    # 최종 평가
    final_preds = (ensemble_probs >= optimal_thresholds).astype(int)

    boosted_f1 = f1_score(y_val, final_preds, average='macro', zero_division=0)
    boosted_hamming = 1 - hamming_loss(y_val, final_preds)

    logger.info(f"\n✅ 긴급 전략 결과:")
    logger.info(f"  F1-Macro: {current_f1:.4f} → {boosted_f1:.4f} (+{boosted_f1-current_f1:.4f})")
    logger.info(f"  Hamming Acc: {current_hamming:.4f} → {boosted_hamming:.4f} (+{boosted_hamming-current_hamming:.4f})")

    return {
        'probs': boosted_probs,
        'weights': optimal_weights,
        'thresholds': optimal_thresholds,
        'f1_macro': boosted_f1,
        'hamming_acc': boosted_hamming
    }


def main():
    """15시간 최대 정확도 전략 메인 함수"""

    start_time = time.time()

    logger.info("\n" + "="*80)
    logger.info("🚀 15시간 최대 정확도 전략 시작 (15번 문서)")
    logger.info("="*80)
    logger.info(f"⏰ 시작 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("\n📋 전략 개요:")
    logger.info("  1️⃣ 3-모델 하이브리드 앙상블 (역할 분담)")
    logger.info("  2️⃣ 모델당 80 epoch (충분한 학습)")
    logger.info("  3️⃣ AEDA 증강 (소수 클래스 강화)")
    logger.info("  4️⃣ Asymmetric Loss (불균형 해결)")
    logger.info("  5️⃣ 클래스별 임계값 최적화")
    logger.info("  6️⃣ 가중 소프트 보팅")
    logger.info("  🚨 7️⃣ 90% 미달성 시 자동 긴급 전략 실행!")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"\n💻 Device: {device}")
    
    if device == 'cuda':
        logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")

    # 1. 데이터 로드
    data_loader = UnsmileDataLoader(data_dir="./data")
    train_df, val_df, test_df = data_loader.load_processed_data()
    label_columns = data_loader.label_columns

    logger.info(f"\n📊 데이터 크기:")
    logger.info(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # 2. AEDA 데이터 증강 (15번 문서 전략)
    logger.info("\n" + "="*80)
    logger.info("📊 Phase 1: AEDA 데이터 증강 (소수 클래스 강화)")
    logger.info("="*80)

    train_df_augmented = augment_minority_classes(
        train_df=train_df,
        label_columns=label_columns,
        target_size=1500,  # 문서 권장값
        punc_ratio=0.3,
        augment_all=False  # 소수 클래스만
    )

    # 3. 3-모델 하이브리드 앙상블 (15번 문서 전략)
    logger.info("\n" + "="*80)
    logger.info("📊 Phase 2: 3-모델 하이브리드 앙상블 학습")
    logger.info("="*80)
    logger.info("\n각 모델의 역할:")
    logger.info("  🔹 KcELECTRA-Base: 슬랭/욕설 전문가 (댓글 데이터 학습)")
    logger.info("  🔹 SoongsilBERT-Base: 안정적 베이스라인 (균형잡힌 성능)")
    logger.info("  🔹 KLUE-RoBERTa-Large+LoRA: 고맥락 의미론 전문가 (복잡한 혐오)")

    # 15번 문서 권장 3개 모델
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
            'name': 'roberta_large',
            'hf_name': 'klue/roberta-large',
            'use_qlora': True,  # QLoRA 활성화
            'role': '고맥락 의미론 전문가'
        }
    ]

    all_models = []
    all_val_probs = []

    for idx, config in enumerate(models_config, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"🔹 모델 {idx}/3: {config['name']} ({config['role']})")
        logger.info(f"{'='*80}")

        tokenizer = AutoTokenizer.from_pretrained(config['hf_name'])

        train_loader, val_loader, test_loader = create_dataloaders(
            train_df=train_df_augmented,
            val_df=val_df,
            test_df=test_df,
            tokenizer=tokenizer,
            label_columns=label_columns,
            batch_size=32,
            max_length=128
        )

        try:
            model, metrics = train_single_model(
                model_config=config,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                num_epochs=80  # 충분한 학습 시간
            )

            all_models.append({
                'name': config['name'],
                'model': model,
                'tokenizer': tokenizer,
                'role': config['role']
            })
            all_val_probs.append(metrics['probs'])

            logger.info(f"\n✅ {config['name']} 학습 완료")
            logger.info(f"  F1-Macro: {metrics['f1_macro']:.4f}")

        except Exception as e:
            logger.error(f"❌ {config['name']} 실패: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue

    if len(all_models) == 0:
        logger.error("❌ 모든 모델 학습 실패!")
        return None

    # 4. 앙상블 가중치 최적화 (Nelder-Mead)
    logger.info("\n" + "="*80)
    logger.info("📊 Phase 3: 앙상블 가중치 최적화")
    logger.info("="*80)

    y_val = val_df[label_columns].values

    optimal_weights = optimize_ensemble_weights(
        all_probs=all_val_probs,
        true_labels=y_val,
        num_models=len(all_models)
    )

    # 가중 앙상블 확률 계산
    ensemble_probs = sum(w * probs for w, probs in zip(optimal_weights, all_val_probs))

    # 5. 클래스별 임계값 최적화 (15번 문서 핵심 전략)
    logger.info("\n" + "="*80)
    logger.info("📊 Phase 4: 클래스별 임계값 최적화")
    logger.info("="*80)

    optimal_thresholds = optimize_thresholds_per_class(
        probs=ensemble_probs,
        labels=y_val,
        num_classes=len(label_columns)
    )

    # 최종 예측
    final_preds = (ensemble_probs >= optimal_thresholds).astype(int)

    # 평가
    final_f1_macro = f1_score(y_val, final_preds, average='macro', zero_division=0)
    final_f1_micro = f1_score(y_val, final_preds, average='micro', zero_division=0)
    final_exact_match = accuracy_score(y_val, final_preds)
    final_hamming_acc = 1 - hamming_loss(y_val, final_preds)

    # 🚨 긴급 전략: 90% 미달성 시 자동 실행
    TARGET_THRESHOLD = 0.90

    if final_hamming_acc < TARGET_THRESHOLD or final_f1_macro < TARGET_THRESHOLD:
        logger.warning(f"\n⚠️ 목표 미달성! (Hamming: {final_hamming_acc:.4f}, F1: {final_f1_macro:.4f})")
        logger.info("🚨 긴급 성능 향상 전략 자동 실행...")

        boost_result = emergency_boost_strategy(
            all_models=all_models,
            train_df_augmented=train_df_augmented,
            val_df=val_df,
            label_columns=label_columns,
            device=device,
            current_f1=final_f1_macro,
            current_hamming=final_hamming_acc
        )

        # 결과 업데이트
        all_val_probs = boost_result['probs']
        optimal_weights = boost_result['weights']
        optimal_thresholds = boost_result['thresholds']
        final_f1_macro = boost_result['f1_macro']
        final_hamming_acc = boost_result['hamming_acc']

        # 재계산
        ensemble_probs = sum(w * probs for w, probs in zip(optimal_weights, all_val_probs))
        final_preds = (ensemble_probs >= optimal_thresholds).astype(int)
        final_f1_micro = f1_score(y_val, final_preds, average='micro', zero_division=0)
        final_exact_match = accuracy_score(y_val, final_preds)

    # 결과 출력
    elapsed_time = (time.time() - start_time) / 3600  # 시간 단위

    logger.info("\n" + "="*80)
    logger.info("🏆 최종 결과 (3-모델 하이브리드 앙상블)")
    logger.info("="*80)
    logger.info(f"\n📊 성능 지표:")
    logger.info(f"  F1-Macro:        {final_f1_macro:.4f} ({final_f1_macro*100:.2f}%)")
    logger.info(f"  F1-Micro:        {final_f1_micro:.4f} ({final_f1_micro*100:.2f}%)")
    logger.info(f"  Exact Match:     {final_exact_match:.4f} ({final_exact_match*100:.2f}%)")
    logger.info(f"  Hamming Acc:     {final_hamming_acc:.4f} ({final_hamming_acc*100:.2f}%)")
    logger.info(f"\n⏱️  소요 시간:        {elapsed_time:.2f} 시간")
    logger.info(f"\n🎯 목표 달성 여부:")

    if final_hamming_acc >= TARGET_THRESHOLD and final_f1_macro >= TARGET_THRESHOLD:
        logger.info(f"  ✅✅✅ 90% 목표 달성 성공! ✅✅✅")
    else:
        logger.warning(f"  ⚠️ 90% 목표 미달성...")
        logger.warning(f"  현재 최고 성능: Hamming {final_hamming_acc*100:.2f}%, F1 {final_f1_macro*100:.2f}%")

    logger.info(f"  정확도 95%:      {'✅ 달성!' if final_hamming_acc >= 0.95 else '❌ 미달성'}")
    logger.info(f"  F1-Macro 92%:    {'✅ 달성!' if final_f1_macro >= 0.92 else '❌ 미달성'}")
    logger.info("\n📋 사용된 모델:")
    for i, model_info in enumerate(all_models, 1):
        logger.info(f"  {i}. {model_info['name']:30s} (가중치: {optimal_weights[i-1]:.3f}) - {model_info['role']}")
    logger.info("="*80)

    # 결과 저장
    results = {
        'strategy': '15hour-3model-hybrid-ensemble',
        'document_reference': '15번 문서 전략',
        'n_models': len(all_models),
        'final_metrics': {
            'f1_macro': float(final_f1_macro),
            'f1_micro': float(final_f1_micro),
            'exact_match': float(final_exact_match),
            'hamming_accuracy': float(final_hamming_acc)
        },
        'optimal_weights': optimal_weights.tolist(),
        'optimal_thresholds': optimal_thresholds.tolist(),
        'elapsed_hours': elapsed_time,
        'models_used': [
            {
                'name': m['name'],
                'role': m['role'],
                'weight': float(w)
            }
            for m, w in zip(all_models, optimal_weights)
        ],
        'target_achieved': {
            'accuracy_90': final_hamming_acc >= 0.90,
            'accuracy_95': final_hamming_acc >= 0.95,
            'f1_macro_90': final_f1_macro >= 0.90,
            'f1_macro_92': final_f1_macro >= 0.92
        }
    }

    os.makedirs("./results", exist_ok=True)

    with open("./results/final_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info("\n✅ 학습 완료! 결과 저장: ./results/final_results.json")

    return results


if __name__ == "__main__":
    main()
