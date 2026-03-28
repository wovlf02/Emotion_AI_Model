"""
AsymmetricLoss – 다중 레이블 불균형 대응 손실 함수
docs/04_학습_전략.md 기반 구현
"""
import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for Multi-Label Classification
    - gamma_neg: negative focusing (FP 억제) → 높을수록 easy negative 무시
    - gamma_pos: positive focusing (FN 감소)
    - clip: probability clipping → negative easy sample 완전 제거
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 0.5,
        clip: float = 0.05,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (batch_size, num_labels) – raw logits
            targets: (batch_size, num_labels) – binary labels {0, 1}
        """
        # Label smoothing
        if self.label_smoothing > 0:
            targets = targets * (1.0 - self.label_smoothing) + self.label_smoothing * 0.5

        # Sigmoid probability
        probs = torch.sigmoid(logits)
        xs_pos = probs
        xs_neg = 1.0 - probs

        # Asymmetric clipping (negative 쪽)
        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        # Basic cross-entropy
        loss_pos = targets * torch.log(xs_pos.clamp(min=1e-8))
        loss_neg = (1 - targets) * torch.log(xs_neg.clamp(min=1e-8))

        loss = loss_pos + loss_neg

        # Asymmetric focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt0 = xs_pos * targets + xs_neg * (1 - targets)  # probability of true class
            gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
            one_sided = torch.pow(1 - pt0, gamma)
            loss *= one_sided

        loss = -loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
