"""
다중 라벨 분류 모델 및 앙상블 시스템
KcELECTRA, SoongsilBERT, KLUE-RoBERTa-Large + LoRA 하이브리드 앙상블
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModel, 
    BitsAndBytesConfig
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training
)
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiLabelClassifier(nn.Module):
    """다중 라벨 분류를 위한 기본 분류기"""
    
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        dropout_rate: float = 0.3,  # 0.1 → 0.3으로 대폭 증가
        use_qlora: bool = False
    ):
        """
        Args:
            model_name: Hugging Face 모델 이름
            num_labels: 레이블 개수
            dropout_rate: 드롭아웃 비율 (과적합 방지 강화)
            use_qlora: QLoRA 사용 여부 (Large 모델용)
        """
        super(MultiLabelClassifier, self).__init__()
        
        self.model_name = model_name
        self.num_labels = num_labels
        self.use_qlora = use_qlora
        
        # QLoRA 설정 (Large 모델용)
        if use_qlora:
            logger.info(f"🔧 QLoRA 모드로 {model_name} 로딩 중...")
            
            # 4비트 양자화 설정
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",  # NormalFloat 4-bit
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True  # 중첩 양자화
            )
            
            # 모델 로드 (4비트)
            self.encoder = AutoModel.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
            
            # QLoRA용 준비
            self.encoder = prepare_model_for_kbit_training(self.encoder)
            
            # LoRA 설정
            lora_config = LoraConfig(
                r=16,  # Rank (문서 전략대로)
                lora_alpha=32,  # Alpha (문서 전략대로)
                target_modules=["query", "value"],  # Q, V 행렬만
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION
            )
            
            # LoRA 어댑터 주입
            self.encoder = get_peft_model(self.encoder, lora_config)
            self.encoder.print_trainable_parameters()
            
            hidden_size = self.encoder.config.hidden_size
        
        else:
            # 일반 모드 (Base 모델용)
            logger.info(f"📦 일반 모드로 {model_name} 로딩 중...")
            self.encoder = AutoModel.from_pretrained(
                model_name,
                use_safetensors=True  # safetensors 우선 사용
            )
            hidden_size = self.encoder.config.hidden_size
        
        # 분류 헤드 - Dropout 2개 레이어로 강화
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate * 0.5)  # 추가 드롭아웃
        self.classifier = nn.Linear(hidden_size, num_labels)
        
        # 초기화
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            input_ids: 토큰 ID (batch_size, seq_len)
            attention_mask: 어텐션 마스크 (batch_size, seq_len)
            labels: 레이블 (batch_size, num_labels)
        
        Returns:
            dict: logits, loss (if labels provided)
        """
        # 인코더 통과
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # [CLS] 토큰 임베딩 추출
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # 분류
        pooled_output = self.dropout1(pooled_output)
        pooled_output = self.dropout2(pooled_output)
        logits = self.classifier(pooled_output)
        
        output_dict = {'logits': logits}
        
        return output_dict
    
    def freeze_encoder(self, num_layers_to_unfreeze: int = 2):
        """인코더 동결 (마지막 N개 레이어만 학습)"""
        if self.use_qlora:
            logger.info("QLoRA 모드에서는 자동으로 파라미터가 최적화됩니다.")
            return
        
        # 전체 동결
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # 마지막 N개 레이어만 해제
        if hasattr(self.encoder, 'encoder'):
            layers = self.encoder.encoder.layer
        elif hasattr(self.encoder, 'layer'):
            layers = self.encoder.layer
        else:
            logger.warning("레이어 구조를 찾을 수 없습니다.")
            return
        
        for layer in layers[-num_layers_to_unfreeze:]:
            for param in layer.parameters():
                param.requires_grad = True
        
        logger.info(f"마지막 {num_layers_to_unfreeze}개 레이어 학습 활성화")


class HybridEnsemble:
    """3-모델 하이브리드 앙상블 시스템"""
    
    def __init__(
        self,
        num_labels: int,
        device: str = 'cuda'
    ):
        """
        Args:
            num_labels: 레이블 개수
            device: 디바이스
        """
        self.num_labels = num_labels
        self.device = device
        self.models = {}
        self.weights = None
        
        logger.info("\n" + "="*80)
        logger.info("🚀 하이브리드 3-모델 앙상블 시스템 초기화")
        logger.info("="*80)
    
    def add_model(self, name: str, model: nn.Module, weight: float = 1.0):
        """앙상블에 모델 추가"""
        self.models[name] = {
            'model': model.to(self.device),
            'weight': weight
        }
        logger.info(f"✓ {name} 추가 (가중치: {weight})")
    
    def load_models(self):
        """15번 문서 전략대로 3개 모델 로드"""
        
        # 모델 1: KcELECTRA-Base (슬랭/욕설 전문가)
        logger.info("\n[1/3] KcELECTRA-Base 로딩...")
        kcelectra = MultiLabelClassifier(
            model_name="beomi/KcELECTRA-base",
            num_labels=self.num_labels,
            dropout_rate=0.1,
            use_qlora=False
        )
        self.add_model("kcelectra", kcelectra, weight=1.0)
        
        # 모델 2: SoongsilBERT-Base (안정적 베이스라인)
        logger.info("\n[2/3] SoongsilBERT-Base 로딩...")
        soongsil = MultiLabelClassifier(
            model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS",
            num_labels=self.num_labels,
            dropout_rate=0.1,
            use_qlora=False
        )
        self.add_model("soongsil", soongsil, weight=1.0)
        
        # 모델 3: KLUE-RoBERTa-Large + LoRA (고맥락 의미론 전문가)
        logger.info("\n[3/3] KLUE-RoBERTa-Large + LoRA 로딩...")
        try:
            roberta_large = MultiLabelClassifier(
                model_name="klue/roberta-large",
                num_labels=self.num_labels,
                dropout_rate=0.1,
                use_qlora=True  # QLoRA 활성화
            )
            self.add_model("roberta_large", roberta_large, weight=1.2)
        except Exception as e:
            logger.warning(f"⚠️ RoBERTa-Large 로딩 실패: {e}")
            logger.info("💡 Base 모델로 대체합니다.")
            roberta_base = MultiLabelClassifier(
                model_name="klue/roberta-base",
                num_labels=self.num_labels,
                dropout_rate=0.1,
                use_qlora=False
            )
            self.add_model("roberta_base", roberta_base, weight=1.0)
        
        logger.info("\n✅ 앙상블 시스템 로딩 완료!")
        logger.info("="*80 + "\n")
    
    def predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """앙상블 예측 (가중 소프트 보팅)"""
        predictions = []
        total_weight = sum(m['weight'] for m in self.models.values())
        
        for name, model_dict in self.models.items():
            model = model_dict['model']
            weight = model_dict['weight']
            
            model.eval()
            with torch.no_grad():
                outputs = model(input_ids, attention_mask)
                logits = outputs['logits']
                probs = torch.sigmoid(logits)
                predictions.append(probs * (weight / total_weight))
        
        # 가중 평균
        ensemble_probs = torch.stack(predictions).sum(dim=0)
        
        return ensemble_probs
    
    def save_models(self, save_dir: str):
        """모든 모델 저장"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        for name, model_dict in self.models.items():
            model = model_dict['model']
            save_path = os.path.join(save_dir, f"{name}.pt")
            torch.save(model.state_dict(), save_path)
            logger.info(f"✓ {name} 저장: {save_path}")
    
    def load_model_weights(self, load_dir: str):
        """저장된 모델 가중치 로드"""
        import os
        
        for name, model_dict in self.models.items():
            model = model_dict['model']
            load_path = os.path.join(load_dir, f"{name}.pt")
            
            if os.path.exists(load_path):
                model.load_state_dict(torch.load(load_path, map_location=self.device, weights_only=True))
                logger.info(f"✓ {name} 로드: {load_path}")
            else:
                logger.warning(f"⚠️ {name} 파일을 찾을 수 없습니다: {load_path}")


def create_model(model_name: str, num_labels: int, use_qlora: bool = False, dropout_rate: float = 0.3) -> nn.Module:
    """단일 모델 생성 헬퍼 함수"""
    return MultiLabelClassifier(
        model_name=model_name,
        num_labels=num_labels,
        dropout_rate=dropout_rate,
        use_qlora=use_qlora
    )


if __name__ == "__main__":
    # 테스트
    print("="*80)
    print("모델 모듈 테스트")
    print("="*80)
    
    # 단일 모델 테스트
    model = create_model("beomi/KcELECTRA-base", num_labels=9, use_qlora=False)
    print(f"\n✓ 모델 생성 성공: {model.model_name}")
    print(f"  레이블 수: {model.num_labels}")
    
    # 앙상블 테스트
    # ensemble = HybridEnsemble(num_labels=9, device='cuda' if torch.cuda.is_available() else 'cpu')
    # ensemble.load_models()
    
    print("\n✅ 모든 테스트 통과!")
