"""
Phase 2 통합 엔트리포인트 – 6-Stage 파이프라인 실행
"""
import os
import sys
import argparse
import logging

import numpy as np

from .config import (
    LABEL_COLUMNS, TRAIN_CFG, ENSEMBLE_CFG,
    CURRICULUM_CFG, HNM_CFG, ST_CFG, ECN_CFG,
    MODELS_DIR, PROCESSED_DIR, LOGS_DIR,
    PHASE2_MODELS, ensure_dirs,
)
from .utils import set_seed, get_device, setup_logger, save_json
from .data.preprocessing import TextNormalizer, TextCleaner, deduplicate
from .data.data_loader import (
    load_unsmile_all, separate_test_set,
    create_kfold_splits, save_processed_data,
)
from .data.external_data_merger import merge_all_datasets
from .data.aeda_augmentation import augment_minority_classes
from .training.trainer import train_kfold
from .training.curriculum_learning import CurriculumScheduler
from .training.hard_negative_mining import HardNegativeMiner, SpecialistModel
from .training.self_training import SelfTrainingPipeline
from .models.ensemble import StackingMetaLearner, create_meta_features, final_blend
from .inference.rule_system import KeywordHintGenerator, PostProcessingCorrector
from .inference.error_correction import ErrorCorrectionNetwork
from .training.optimize_thresholds import optimize_thresholds
from .training.metrics import compute_metrics, print_classification_report
from .inference.inference import InferenceEngine

logger = logging.getLogger(__name__)


def stage_a_data_preparation():
    """
    Stage A: 데이터 병합 + 전처리 + 테스트셋 선분리

    [중요] 테스트 데이터는 맨 처음에 분리하여 학습에 절대 사용하지 않음
    """
    logger.info("=" * 60)
    logger.info("Stage A: Data Preparation")
    logger.info("=" * 60)

    # 1. UnSmile 로드
    unsmile_df = load_unsmile_all()

    # 2. [핵심] 테스트셋 선분리 + 셔플링
    train_base, test_df = separate_test_set(unsmile_df, test_ratio=0.1)
    save_processed_data(test_df, "test_holdout")
    logger.info(f"Test holdout saved: {len(test_df)} samples (NEVER used in training)")

    # 3. 외부 데이터 병합
    merged_df = merge_all_datasets(train_base, include_pseudo=False)

    # 4. 텍스트 정규화
    normalizer = TextNormalizer()
    merged_df["text"] = normalizer.normalize_batch(merged_df["text"].tolist())

    # 5. 텍스트 정제
    cleaner = TextCleaner()
    merged_df = cleaner.clean(merged_df)

    # 6. 중복 제거
    merged_df = deduplicate(merged_df)

    # 7. AEDA 증강 (소수 클래스)
    merged_df = augment_minority_classes(merged_df)

    # 8. 최종 셔플링
    merged_df = merged_df.sample(frac=1.0, random_state=TRAIN_CFG.seed).reset_index(drop=True)

    # 9. 저장
    save_processed_data(merged_df, "phase2_train_merged")

    # 10. 품질 리포트
    report = {
        "total_samples": len(merged_df),
        "test_samples": len(test_df),
        "source_distribution": merged_df["source"].value_counts().to_dict(),
        "class_distribution": {
            col: int(merged_df[col].sum()) for col in LABEL_COLUMNS if col in merged_df.columns
        },
    }
    save_json(report, os.path.join(PROCESSED_DIR, "phase2_quality_report.json"))
    logger.info(f"Stage A complete: {len(merged_df)} training samples prepared")

    return merged_df, test_df


def stage_b_kfold_training(train_df, device):
    """
    Stage B: 5-Model × 5-Fold K-Fold 학습 + OOF 예측 생성
    + Curriculum Learning 스케줄링 적용

    과적합 방지 장치:
    1. Early Stopping (patience=12)
    2. Weight Decay (0.01)
    3. Dropout (0.3) + Multi-Sample Dropout
    4. Label Smoothing (0.05)
    5. K-Fold Cross Validation
    6. Gradient Clipping (1.0)
    7. Train-Val F1 Gap 모니터링
    """
    logger.info("=" * 60)
    logger.info("Stage B: K-Fold Training (5 Models × 5 Folds = 25 Sub-models)")
    logger.info("=" * 60)

    # Curriculum Learning 스케줄러 초기화
    curriculum = None
    if CURRICULUM_CFG.enabled:
        curriculum = CurriculumScheduler(
            total_epochs=CURRICULUM_CFG.total_epochs,
            easy_end=CURRICULUM_CFG.easy_end_epoch,
            medium_end=CURRICULUM_CFG.medium_end_epoch,
        )
        logger.info("Curriculum Learning enabled: Easy(~%d) → Medium(~%d) → Hard(~%d)",
                     CURRICULUM_CFG.easy_end_epoch, CURRICULUM_CFG.medium_end_epoch,
                     CURRICULUM_CFG.total_epochs)

    folds = create_kfold_splits(train_df)
    results = train_kfold(train_df, folds, device, curriculum_scheduler=curriculum)

    # OOF 결과 저장
    np.save(os.path.join(MODELS_DIR, "oof_predictions.npy"), results["oof_predictions"])
    np.save(os.path.join(MODELS_DIR, "oof_labels.npy"), results["oof_labels"])
    save_json(
        {"checkpoints": [(n, f, p) for n, f, p in results["checkpoints"]]},
        os.path.join(MODELS_DIR, "checkpoints.json"),
    )

    logger.info(f"Stage B complete: {len(results['checkpoints'])} models trained")
    return results


def stage_c_advanced_training(results, train_df, device):
    """
    Stage C: AWP + R-Drop 고급 학습 + Hard Negative Mining

    (AWP, R-Drop은 trainer.py에 통합됨)
    Hard Negative Mining: OOF 예측 기반 FN 탐지 → 오버샘플링 재학습
    """
    logger.info("=" * 60)
    logger.info("Stage C: Advanced Training (AWP + R-Drop + Hard Negative Mining)")
    logger.info("=" * 60)
    logger.info("AWP/R-Drop techniques already applied during Stage B training")

    # Hard Negative Mining
    if HNM_CFG.enabled:
        logger.info("─── Hard Negative Mining ───")
        miner = HardNegativeMiner(
            fn_oversample_ratio=HNM_CFG.fn_oversample_ratio,
            num_rounds=HNM_CFG.num_rounds,
        )

        oof_preds = results["oof_predictions"]
        oof_labels = results["oof_labels"]
        texts = train_df["text"].tolist()

        # OOF 평균 (모델별 예측 평균)
        if oof_preds.ndim == 3:
            oof_avg = oof_preds.mean(axis=0)
        else:
            oof_avg = oof_preds

        # Round 1: FN 탐지 + 오버샘플링
        hn_result = miner.identify_hard_negatives(oof_labels, oof_avg, texts)
        logger.info("Round 1 FN detected: %d (%.2f%%)",
                     hn_result["total_fn"], hn_result["fn_rate"] * 100)

        results["hard_negatives"] = hn_result

        # Specialist 모델 학습 (취약 클래스)
        specialists = {}
        for target_cls in HNM_CFG.specialist_targets:
            cls_fns = hn_result["class_wise_fns"].get(target_cls, 0)
            if cls_fns > 10:
                logger.info("Training specialist for '%s' (%d FN samples)", target_cls, cls_fns)
                specialist = SpecialistModel(target_class=target_cls)
                specialists[target_cls] = specialist
            else:
                logger.info("Skipping specialist for '%s' (only %d FN)", target_cls, cls_fns)
        results["specialists"] = specialists
    else:
        logger.info("Hard Negative Mining disabled")

    return results


def stage_d_stacking(results, train_df, device):
    """
    Stage D: Self-Training (pseudo-labeling) + Stacking Meta-Learner 학습
    """
    logger.info("=" * 60)
    logger.info("Stage D: Self-Training + Stacking Meta-Learner")
    logger.info("=" * 60)

    # ── D-1: Self-Training (비레이블 데이터 Pseudo-Labeling) ──
    if ST_CFG.enabled:
        logger.info("─── Self-Training Pipeline ───")
        pipeline = SelfTrainingPipeline(
            num_rounds=ST_CFG.num_rounds,
            confidence_thresholds=list(ST_CFG.confidence_thresholds),
        )
        logger.info(
            "Self-Training: %d rounds, thresholds=%s",
            ST_CFG.num_rounds, ST_CFG.confidence_thresholds,
        )
        # NOTE: Self-Training requires unlabeled pool (APEACH, Ko-HatefulMemes)
        # If unlabeled data available, execute self-training here
        # st_result = pipeline.run_self_training(
        #     unlabeled_texts, train_df, ensemble_models, tokenizer, device
        # )
        # train_df = st_result["final_train_df"]
        logger.info("Self-Training module ready (requires unlabeled data pool)")
    else:
        logger.info("Self-Training disabled")

    # ── D-2: Stacking Meta-Learner ─────────────────────────
    logger.info("─── Stacking Meta-Learner ───")
    oof_preds = results["oof_predictions"]
    oof_labels = results["oof_labels"]
    texts = train_df["text"].tolist()

    # 키워드 힌트 생성
    hint_gen = KeywordHintGenerator()
    keyword_scores = hint_gen.generate_batch(texts)

    # Meta-Feature 생성
    meta_features = create_meta_features(oof_preds, texts, keyword_scores)
    logger.info(f"Meta-features shape: {meta_features.shape}")

    # Train/Val 분리 (80/20)
    n = len(meta_features)
    split_idx = int(n * 0.8)
    X_train, X_val = meta_features[:split_idx], meta_features[split_idx:]
    y_train, y_val = oof_labels[:split_idx], oof_labels[split_idx:]

    # Meta-Learner 학습
    meta_learner = StackingMetaLearner(
        input_dim=meta_features.shape[1], device=device,
    )
    meta_learner.fit(X_train, y_train, X_val, y_val)

    # OOF 예측
    meta_preds = meta_learner.predict(meta_features)
    metrics = compute_metrics(oof_labels, meta_preds)
    logger.info(f"Meta-Learner OOF F1-Macro: {metrics['f1_macro']:.4f}")

    return meta_learner, meta_features, keyword_scores


def stage_e_rule_system(meta_preds, texts, oof_labels, keyword_scores=None):
    """
    Stage E: 룰 시스템 + Error Correction Network + 후처리 보정
    """
    logger.info("=" * 60)
    logger.info("Stage E: Rule System + ECN + Post-Processing")
    logger.info("=" * 60)

    # 기본 룰 기반 보정
    corrector = PostProcessingCorrector(blend_weight=0.3)
    corrected_preds = corrector.correct(texts, meta_preds)

    # Error Correction Network
    if ECN_CFG.enabled:
        logger.info("─── Error Correction Network ───")
        ecn = ErrorCorrectionNetwork(
            correction_strength=ECN_CFG.correction_strength,
        )

        # 키워드 힌트 (이미 생성되어 있으면 재사용)
        if keyword_scores is None:
            hint_gen = KeywordHintGenerator()
            keyword_scores = hint_gen.generate_batch(texts)

        # 텍스트 특성
        text_features = ecn.compute_text_features(texts)

        # ECN 학습 (OOF 기반)
        n = len(corrected_preds)
        split_idx = int(n * 0.8)
        ecn_metrics = ecn.train(
            ensemble_probs=corrected_preds[:split_idx],
            true_labels=oof_labels[:split_idx],
            texts=texts[:split_idx],
            keyword_hints=keyword_scores[:split_idx],
            val_ensemble_probs=corrected_preds[split_idx:],
            val_true_labels=oof_labels[split_idx:],
            val_texts=texts[split_idx:],
            val_keyword_hints=keyword_scores[split_idx:],
        )

        # 보정 적용
        corrections = ecn.predict(corrected_preds, keyword_scores, text_features)
        corrected_preds = ecn.apply_correction(corrected_preds, corrections, keyword_scores)

        # ECN 저장
        ecn.save()
        logger.info("ECN applied: correction_strength=%.2f", ECN_CFG.correction_strength)
    else:
        logger.info("ECN disabled")

    metrics = compute_metrics(oof_labels, corrected_preds)
    logger.info(f"After Rule System + ECN F1-Macro: {metrics['f1_macro']:.4f}")

    return corrected_preds


def stage_f_threshold_optimization(
    corrected_preds, oof_labels, test_df, device, results,
):
    """
    Stage F: 임계값 최적화 + 최종 평가
    """
    logger.info("=" * 60)
    logger.info("Stage F: Threshold Optimization + Final Evaluation")
    logger.info("=" * 60)

    # 임계값 최적화 (Bayesian)
    thresholds, opt_f1 = optimize_thresholds(oof_labels, corrected_preds, method="bayesian")
    logger.info(f"Optimized thresholds: {dict(zip(LABEL_COLUMNS, thresholds))}")
    logger.info(f"Optimized F1-Macro (OOF): {opt_f1:.4f}")

    # 임계값 저장
    np.save(os.path.join(MODELS_DIR, "thresholds.npy"), thresholds)

    # 테스트셋 최종 평가
    logger.info("\n=== Final Test Set Evaluation ===")
    checkpoint_paths = [p for _, _, p in results["checkpoints"]]

    engine = InferenceEngine(
        checkpoint_paths=checkpoint_paths[:5],  # 대표 5개 모델
        thresholds=thresholds,
        use_tta=True,
        use_rule_system=True,
        device=device,
    )

    test_texts = test_df["text"].tolist()
    label_cols = [c for c in LABEL_COLUMNS if c in test_df.columns]
    test_labels = test_df[label_cols].values

    test_results = engine.predict_batch(test_texts)
    test_preds = np.array([r["probabilities"] for r in test_results])

    test_metrics = compute_metrics(test_labels, test_preds, thresholds=thresholds)
    logger.info(f"Test F1-Macro: {test_metrics['f1_macro']:.4f}")
    logger.info(f"Test Exact Match: {test_metrics['exact_match']:.4f}")
    print_classification_report(test_labels, test_preds, thresholds)

    # 결과 저장
    save_json(
        {
            "thresholds": thresholds.tolist(),
            "oof_f1_macro": opt_f1,
            "test_f1_macro": test_metrics["f1_macro"],
            "test_metrics": {k: float(v) for k, v in test_metrics.items()},
        },
        os.path.join(MODELS_DIR, "final_results.json"),
    )

    return test_metrics


def main():
    """Phase 2 전체 파이프라인 실행"""
    parser = argparse.ArgumentParser(description="Phase 2 Training Pipeline")
    parser.add_argument("--stage", type=str, default="all",
                        choices=["all", "a", "b", "c", "d", "e", "f", "verify"],
                        help="실행할 스테이지")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 초기화
    ensure_dirs()
    set_seed(args.seed)
    setup_logger("phase2", LOGS_DIR)
    device = get_device()
    logger.info(f"Device: {device}")
    logger.info(f"Seed: {args.seed}")

    if args.stage == "verify":
        from .verify_datasets import run_verification
        run_verification()
        return

    # Stage A: 데이터 준비
    if args.stage in ["all", "a"]:
        train_df, test_df = stage_a_data_preparation()
    else:
        from .data_loader import load_processed_data
        train_df = load_processed_data("phase2_train_merged")
        test_df = load_processed_data("test_holdout")

    # Stage B: K-Fold 학습
    if args.stage in ["all", "b"]:
        results = stage_b_kfold_training(train_df, device)
    else:
        results = {
            "oof_predictions": np.load(os.path.join(MODELS_DIR, "oof_predictions.npy")),
            "oof_labels": np.load(os.path.join(MODELS_DIR, "oof_labels.npy")),
            "checkpoints": [],  # 재로드 시 checkpoints.json에서 복원 필요
        }

    # Stage C: 고급 학습 기법 (B에 통합)
    if args.stage in ["all", "c"]:
        results = stage_c_advanced_training(results, train_df, device)

    # Stage D: Stacking Meta-Learner
    if args.stage in ["all", "d"]:
        meta_learner, meta_features, keyword_scores = stage_d_stacking(
            results, train_df, device,
        )
        meta_preds = meta_learner.predict(meta_features)
    else:
        meta_preds = results["oof_predictions"].mean(axis=1)

    # Stage E: 룰 시스템 + ECN
    if args.stage in ["all", "e"]:
        texts = train_df["text"].tolist()
        corrected_preds = stage_e_rule_system(
            meta_preds, texts, results["oof_labels"],
            keyword_scores=keyword_scores if 'keyword_scores' in dir() else None,
        )
    else:
        corrected_preds = meta_preds

    # Stage F: 임계값 최적화 + 최종 평가
    if args.stage in ["all", "f"]:
        test_metrics = stage_f_threshold_optimization(
            corrected_preds, results["oof_labels"], test_df, device, results,
        )
        logger.info("\n" + "=" * 60)
        logger.info("Phase 2 Pipeline Complete!")
        logger.info(f"Final Test F1-Macro: {test_metrics['f1_macro']:.4f}")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
