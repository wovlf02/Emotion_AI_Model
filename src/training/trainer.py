"""
학습 루프 – K-Fold 학습 + 6중 과적합 방지 + AWP + R-Drop
docs/04_학습_전략.md, docs/10_Phase2_구현명세서.md 기반 구현
"""
import os
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from ..config import (
    LABEL_COLUMNS, TRAIN_CFG, MODELS_DIR,
    PHASE2_MODELS, ModelConfig,
)
from ..models.model import MultiLabelClassifier, AWP
from ..models.asymmetric_loss import AsymmetricLoss
from ..data.dataset import create_dataloader, get_tokenizer, HateSpeechDataset
from .metrics import compute_metrics, find_optimal_thresholds
from ..utils import (
    set_seed, get_device, EarlyStopping,
    save_checkpoint, setup_logger,
)

logger = logging.getLogger(__name__)


def rdrop_kl_loss(logits1: torch.Tensor, logits2: torch.Tensor) -> torch.Tensor:
    """R-Drop KL-Divergence 정규화 (양방향)"""
    p1 = torch.sigmoid(logits1)
    p2 = torch.sigmoid(logits2)

    # Bernoulli KL (다중 레이블)
    kl_1 = F.kl_div(
        torch.log(p1 + 1e-8), p2, reduction="batchmean", log_target=False
    )
    kl_2 = F.kl_div(
        torch.log(p2 + 1e-8), p1, reduction="batchmean", log_target=False
    )
    return (kl_1 + kl_2) / 2


def train_one_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer,
    scheduler,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    awp: Optional[AWP] = None,
    use_rdrop: bool = False,
    rdrop_alpha: float = None,
    gradient_accumulation_steps: int = None,
    max_grad_norm: float = None,
) -> Tuple[float, float]:
    """
    1 에폭 학습

    Returns:
        (avg_loss, avg_f1_macro)
    """
    if rdrop_alpha is None:
        rdrop_alpha = TRAIN_CFG.rdrop_alpha
    if gradient_accumulation_steps is None:
        gradient_accumulation_steps = TRAIN_CFG.gradient_accumulation_steps
    if max_grad_norm is None:
        max_grad_norm = TRAIN_CFG.max_grad_norm

    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        with autocast(enabled=TRAIN_CFG.fp16):
            logits1 = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(logits1, labels)

            # R-Drop: 두 번째 forward (dropout 차이)
            if use_rdrop:
                logits2 = model(input_ids, attention_mask, token_type_ids)
                loss2 = criterion(logits2, labels)
                kl = rdrop_kl_loss(logits1, logits2)
                loss = (loss + loss2) / 2 + rdrop_alpha * kl

            loss = loss / gradient_accumulation_steps

        scaler.scale(loss).backward()

        # AWP 적대적 섭동
        if awp is not None and epoch >= TRAIN_CFG.awp_start_epoch:
            awp.save()
            awp.attack_step()
            with autocast(enabled=TRAIN_CFG.fp16):
                logits_adv = model(input_ids, attention_mask, token_type_ids)
                loss_adv = criterion(logits_adv, labels) / gradient_accumulation_steps
            scaler.scale(loss_adv).backward()
            awp.restore()

        # Gradient Accumulation
        if (step + 1) % gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step(epoch + step / len(dataloader))

        total_loss += loss.item() * gradient_accumulation_steps

        with torch.no_grad():
            preds = torch.sigmoid(logits1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

    # 잔여 gradient 처리
    if len(dataloader) % gradient_accumulation_steps != 0:
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    avg_loss = total_loss / len(dataloader)
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    metrics = compute_metrics(all_labels, all_preds)

    return avg_loss, metrics["f1_macro"]


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    검증/평가

    Returns:
        (avg_loss, f1_macro, all_preds, all_labels)
    """
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        with autocast(enabled=TRAIN_CFG.fp16):
            logits = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(logits, labels)

        total_loss += loss.item()
        preds = torch.sigmoid(logits).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    metrics = compute_metrics(all_labels, all_preds)

    return avg_loss, metrics["f1_macro"], all_preds, all_labels


def train_single_model(
    model_cfg: ModelConfig,
    train_df,
    val_df,
    fold_idx: int,
    device: torch.device,
    save_dir: str = None,
) -> Tuple[str, np.ndarray, np.ndarray]:
    """
    단일 모델 학습 (1 Fold)

    Returns:
        (checkpoint_path, val_predictions, val_labels)
    """
    if save_dir is None:
        save_dir = MODELS_DIR

    model_name = model_cfg.pretrained
    tag = f"{model_cfg.name}_fold{fold_idx}"
    logger.info(f"=== Training {tag} ({model_name}) ===")

    # 토크나이저 + 모델
    tokenizer = get_tokenizer(model_name)
    model = MultiLabelClassifier(model_name).to(device)

    # DataLoader
    train_loader = create_dataloader(
        train_df, tokenizer, batch_size=model_cfg.batch_size,
        shuffle=True, use_sample_weights=True,
    )
    val_loader = create_dataloader(
        val_df, tokenizer, batch_size=model_cfg.batch_size * 2,
        shuffle=False,
    )

    # Loss
    criterion = AsymmetricLoss(
        gamma_neg=TRAIN_CFG.asl_gamma_neg,
        gamma_pos=TRAIN_CFG.asl_gamma_pos,
        clip=TRAIN_CFG.asl_clip,
        label_smoothing=TRAIN_CFG.label_smoothing,
    )

    # Optimizer + Scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=model_cfg.lr,
        weight_decay=TRAIN_CFG.weight_decay,
    )
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=TRAIN_CFG.scheduler_t0,
        T_mult=TRAIN_CFG.scheduler_t_mult,
    )

    scaler = GradScaler(enabled=TRAIN_CFG.fp16)
    awp = AWP(model)
    early_stopper = EarlyStopping(
        patience=model_cfg.early_stopping_patience,
        min_delta=TRAIN_CFG.early_stopping_min_delta,
    )

    best_f1 = 0.0
    best_path = os.path.join(save_dir, f"{tag}_best.pt")

    for epoch in range(model_cfg.epochs):
        use_rdrop = epoch >= TRAIN_CFG.rdrop_start_epoch

        train_loss, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            scaler, device, epoch, awp=awp, use_rdrop=use_rdrop,
        )

        val_loss, val_f1, val_preds, val_labels = evaluate(
            model, val_loader, criterion, device,
        )

        logger.info(
            f"[{tag}] Epoch {epoch + 1}/{model_cfg.epochs} | "
            f"Train Loss={train_loss:.4f} F1={train_f1:.4f} | "
            f"Val Loss={val_loss:.4f} F1={val_f1:.4f}"
        )

        # 과적합 감시: Train-Val F1 차이 경고
        f1_gap = train_f1 - val_f1
        if f1_gap > 0.10:
            logger.warning(
                f"[{tag}] Overfitting alert! "
                f"Train-Val F1 gap = {f1_gap:.4f} (> 0.10)"
            )

        # Best 모델 저장
        if val_f1 > best_f1:
            best_f1 = val_f1
            save_checkpoint(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": model_name,
                    "fold": fold_idx,
                    "epoch": epoch,
                    "val_f1": val_f1,
                },
                best_path,
            )
            logger.info(f"  → Best model saved: F1={val_f1:.4f}")

        # Early Stopping
        if early_stopper(val_f1):
            logger.info(
                f"  → Early stopping at epoch {epoch + 1} "
                f"(patience={model_cfg.early_stopping_patience})"
            )
            break

    # 최종 검증 예측
    model.load_state_dict(torch.load(best_path, weights_only=False)["model_state_dict"])
    _, _, final_preds, final_labels = evaluate(model, val_loader, criterion, device)

    logger.info(f"=== {tag} complete: Best Val F1={best_f1:.4f} ===")
    return best_path, final_preds, final_labels


def train_kfold(
    train_df,
    folds: list,
    device: torch.device,
) -> Dict[str, object]:
    """
    전체 K-Fold × 5-Model 학습

    Returns:
        {
            'oof_predictions': (N, 25, 9) OOF 예측,
            'oof_labels': (N, 9) 정답,
            'checkpoints': [(model_name, fold, path), ...],
        }
    """
    n_samples = len(train_df)
    n_models = len(PHASE2_MODELS)
    n_folds = len(folds)

    oof_preds = np.zeros((n_samples, n_models * n_folds, len(LABEL_COLUMNS)))
    oof_labels = np.zeros((n_samples, len(LABEL_COLUMNS)))
    checkpoints = []

    model_idx = 0
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        fold_train = train_df.iloc[train_idx].reset_index(drop=True)
        fold_val = train_df.iloc[val_idx].reset_index(drop=True)

        # OOF 레이블 저장
        label_cols = [c for c in LABEL_COLUMNS if c in train_df.columns]
        oof_labels[val_idx] = fold_val[label_cols].values

        for m_idx, model_cfg in enumerate(PHASE2_MODELS):
            ckpt_path, val_preds, _ = train_single_model(
                model_cfg, fold_train, fold_val, fold_idx, device,
            )

            col_idx = fold_idx * n_models + m_idx
            oof_preds[val_idx, col_idx, :] = val_preds
            checkpoints.append((model_cfg.name, fold_idx, ckpt_path))
            model_idx += 1

    return {
        "oof_predictions": oof_preds,
        "oof_labels": oof_labels,
        "checkpoints": checkpoints,
    }
