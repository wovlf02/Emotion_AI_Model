"""
Asymmetric Loss (ASL) 구현
불균형 다중 라벨 분류를 위한 비대칭 손실 함수
"""

import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    """
    비대칭 손실 함수 (Asymmetric Loss)
    
    긍정 샘플과 부정 샘플에 서로 다른 감쇠율을 적용하여
    불균형 데이터셋에서 소수 클래스 학습을 강화
    
    Reference: Asymmetric Loss For Multi-Label Classification (ICCV 2021)
    """
    
    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        disable_torch_grad_focal_loss: bool = True
    ):
        """
        Args:
            gamma_neg: 부정 샘플 감쇠율 (높을수록 쉬운 부정 샘플 억제)
            gamma_pos: 긍정 샘플 감쇠율 (낮을수록 긍정 샘플 강조)
            clip: 부정 샘플 확률 하한값 (이하는 손실 0으로 처리)
            eps: 수치 안정성을 위한 작은 값
            disable_torch_grad_focal_loss: focal loss 기울기 최적화
        """
        super(AsymmetricLoss, self).__init__()
        
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: 모델 출력 (batch_size, num_labels)
            targets: 타겟 레이블 (batch_size, num_labels)
        
        Returns:
            loss: 스칼라 손실값
        """
        # 시그모이드로 확률 변환
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos
        
        # Probability Shifting (부정 샘플 클리핑)
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        
        # 긍정/부정 손실 계산
        los_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        
        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                # 메모리 효율적인 구현
                torch.set_grad_enabled(False)
            
            # 긍정 샘플: (1-p)^gamma_pos 가중치
            pt_pos = xs_pos * targets
            asymmetric_w_pos = (1 - pt_pos).pow(self.gamma_pos)
            
            # 부정 샘플: p^gamma_neg 가중치
            pt_neg = xs_neg * (1 - targets)
            asymmetric_w_neg = pt_neg.pow(self.gamma_neg)
            
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            
            los_pos *= asymmetric_w_pos
            los_neg *= asymmetric_w_neg
        
        # 최종 손실
        loss = -los_pos - los_neg
        
        return loss.mean()
    
    def __repr__(self):
        return (
            f"AsymmetricLoss(gamma_neg={self.gamma_neg}, "
            f"gamma_pos={self.gamma_pos}, clip={self.clip})"
        )


class AsymmetricLossOptimized(nn.Module):
    """
    최적화된 비대칭 손실 함수 (메모리 효율 개선)
    """
    
    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 0.0,
        clip: float = 0.05,
        eps: float = 1e-8
    ):
        super(AsymmetricLossOptimized, self).__init__()
        
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 로짓 (batch_size, num_classes)
            y: 타겟 (batch_size, num_classes)
        """
        # 확률 계산
        xs_pos = torch.sigmoid(x)
        xs_neg = 1.0 - xs_pos
        
        # Clipping
        if self.clip > 0:
            xs_neg.add_(self.clip).clamp_(max=1)
        
        # 기본 BCE loss
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        
        # Asymmetric focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            # Detach for efficiency
            pt0 = xs_pos.detach()
            pt1 = xs_neg.detach()
            
            pt0 = pt0 * y
            pt1 = pt1 * (1 - y)
            
            # Apply focusing
            los_pos *= (1 - pt0).pow(self.gamma_pos)
            los_neg *= pt1.pow(self.gamma_neg)
        
        loss = -(los_pos + los_neg)
        
        return loss.mean()


if __name__ == "__main__":
    # 테스트
    batch_size = 8
    num_labels = 9
    
    # 랜덤 데이터 생성
    logits = torch.randn(batch_size, num_labels)
    targets = torch.randint(0, 2, (batch_size, num_labels)).float()
    
    # ASL 테스트
    criterion = AsymmetricLoss(gamma_neg=4.0, gamma_pos=1.0, clip=0.05)
    loss = criterion(logits, targets)
    
    print(f"Asymmetric Loss: {loss.item():.4f}")
    print(f"Loss shape: {loss.shape}")
    
    # 최적화 버전 테스트
    criterion_opt = AsymmetricLossOptimized(gamma_neg=4.0, gamma_pos=0.0, clip=0.05)
    loss_opt = criterion_opt(logits, targets)
    
    print(f"Optimized ASL: {loss_opt.item():.4f}")
