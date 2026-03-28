"""
MultiLabelClassifier – 사전학습 모델 + Multi-Sample Dropout 분류 헤드
docs/03_모델_아키텍처.md, docs/12_앙상블_심층설계.md 기반 구현
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

from ..config import NUM_LABELS, TRAIN_CFG


class MultiSampleDropout(nn.Module):
    """Multi-Sample Dropout: K개 드롭아웃 마스크로 예측 후 평균"""

    def __init__(self, classifier: nn.Module, dropout_rate: float = 0.3, k: int = 5):
        super().__init__()
        self.classifier = classifier
        self.dropouts = nn.ModuleList([nn.Dropout(dropout_rate) for _ in range(k)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            logits = torch.stack([self.classifier(drop(x)) for drop in self.dropouts])
            return logits.mean(dim=0)
        else:
            return self.classifier(x)


class MultiLabelClassifier(nn.Module):
    """
    사전학습 BERT/ELECTRA 기반 다중 레이블 분류기

    - [CLS] 토큰 + Multi-Sample Dropout → 9차원 출력
    - R-Drop 학습 시 forward를 2회 호출하여 KL-divergence 계산
    """

    def __init__(
        self,
        pretrained_name: str,
        num_labels: int = NUM_LABELS,
        dropout: float = None,
        msd_k: int = None,
    ):
        super().__init__()
        self.num_labels = num_labels

        config = AutoConfig.from_pretrained(pretrained_name)
        self.encoder = AutoModel.from_pretrained(pretrained_name, config=config)
        hidden_size = config.hidden_size

        dropout = dropout or TRAIN_CFG.dropout
        msd_k = msd_k or TRAIN_CFG.multi_sample_dropout_k

        # 분류 헤드
        head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, num_labels),
        )

        # Multi-Sample Dropout 래핑
        self.classifier = MultiSampleDropout(head, dropout_rate=dropout, k=msd_k)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**kwargs)

        # [CLS] 토큰 표현
        cls_output = outputs.last_hidden_state[:, 0, :]

        logits = self.classifier(cls_output)
        return logits

    def get_cls_embedding(self, input_ids, attention_mask, token_type_ids=None):
        """[CLS] 임베딩 추출 (meta-feature용)"""
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        outputs = self.encoder(**kwargs)
        return outputs.last_hidden_state[:, 0, :]


class AWP:
    """
    Adversarial Weight Perturbation
    학습 안정성 향상 및 일반화 성능 개선
    """

    def __init__(self, model: nn.Module, adv_lr: float = None, adv_eps: float = None):
        self.model = model
        self.adv_lr = adv_lr or TRAIN_CFG.awp_adv_lr
        self.adv_eps = adv_eps or TRAIN_CFG.awp_adv_eps
        self.backup = {}
        self.backup_eps = {}

    def attack_step(self):
        """가중치에 적대적 섭동 추가"""
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data)
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]),
                        self.backup_eps[name][1],
                    )

    def save(self):
        """가중치 백업"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.backup[name] = param.data.clone()
                grad_eps = self.adv_eps * param.data.abs()
                self.backup_eps[name] = (
                    param.data - grad_eps,
                    param.data + grad_eps,
                )

    def restore(self):
        """백업된 가중치 복원"""
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}
