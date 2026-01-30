"""
UnSmile 데이터셋 클래스
다중 라벨 분류를 위한 PyTorch Dataset
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, List
import pandas as pd


class UnsmileDataset(Dataset):
    """UnSmile 다중 라벨 분류 데이터셋"""
    
    def __init__(
        self,
        texts: List[str],
        labels: torch.Tensor,
        tokenizer,
        max_length: int = 128
    ):
        """
        Args:
            texts: 텍스트 리스트
            labels: 레이블 텐서 (num_samples, num_labels)
            tokenizer: Hugging Face 토크나이저
            max_length: 최대 시퀀스 길이
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # 토크나이징
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': label
        }


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer,
    label_columns: List[str],
    batch_size: int = 16,
    max_length: int = 128,
    num_workers: int = 0
):
    """
    데이터로더 생성
    
    Args:
        train_df, val_df, test_df: 데이터프레임
        tokenizer: 토크나이저
        label_columns: 레이블 컬럼 리스트
        batch_size: 배치 크기
        max_length: 최대 시퀀스 길이
        num_workers: 데이터 로딩 워커 수
    
    Returns:
        train_loader, val_loader, test_loader
    """
    from torch.utils.data import DataLoader
    
    # 레이블을 텐서로 변환
    train_labels = torch.tensor(train_df[label_columns].values, dtype=torch.float32)
    val_labels = torch.tensor(val_df[label_columns].values, dtype=torch.float32)
    test_labels = torch.tensor(test_df[label_columns].values, dtype=torch.float32)
    
    # 데이터셋 생성
    train_dataset = UnsmileDataset(
        train_df['text'].tolist(),
        train_labels,
        tokenizer,
        max_length
    )
    
    val_dataset = UnsmileDataset(
        val_df['text'].tolist(),
        val_labels,
        tokenizer,
        max_length
    )
    
    test_dataset = UnsmileDataset(
        test_df['text'].tolist(),
        test_labels,
        tokenizer,
        max_length
    )
    
    # 데이터로더 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # 테스트
    print("Dataset module loaded successfully!")
