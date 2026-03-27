"""
UnSmile 데이터셋 로더
TSV 파일 읽기 및 전처리
"""

import os
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnsmileDataLoader:
    """UnSmile 데이터셋 로더"""
    
    def __init__(self, data_dir: str = "./data"):
        """
        Args:
            data_dir: 데이터 디렉토리 경로
        """
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "raw")
        self.processed_dir = os.path.join(data_dir, "processed")
        
        # 레이블 컬럼 정의 (clean 제외 - 다중 라벨만)
        self.label_columns = [
            '여성/가족',
            '남성',
            '성소수자',
            '인종/국적',
            '연령',
            '지역',
            '종교',
            '기타 혐오',
            '악플/욕설'
        ]
    
    def load_raw_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """원본 TSV 파일 로드"""
        logger.info("📂 원본 데이터 로딩 중...")
        
        train_path = os.path.join(self.raw_dir, "unsmile_train.tsv")
        dev_path = os.path.join(self.raw_dir, "unsmile_dev.tsv")
        test_path = os.path.join(self.raw_dir, "unsmile_test.tsv")
        
        train_df = pd.read_csv(train_path, sep='\t')
        dev_df = pd.read_csv(dev_path, sep='\t')
        test_df = pd.read_csv(test_path, sep='\t')
        
        logger.info(f"  Train: {len(train_df)} 샘플")
        logger.info(f"  Dev: {len(dev_df)} 샘플")
        logger.info(f"  Test: {len(test_df)} 샘플")
        
        return train_df, dev_df, test_df
    
    def prepare_dataset(self):
        """데이터 전처리 및 저장"""
        logger.info("\n" + "="*80)
        logger.info("📊 데이터 전처리 시작")
        logger.info("="*80)
        
        # 원본 데이터 로드
        train_df, dev_df, test_df = self.load_raw_data()
        
        # 문장 컬럼 이름을 'text'로 변경하고 필요한 컬럼만 유지
        for df in [train_df, dev_df, test_df]:
            if '문장' in df.columns:
                df.rename(columns={'문장': 'text'}, inplace=True)
        
        # 필요한 컬럼만 선택 (text + label_columns)
        columns_to_keep = ['text'] + self.label_columns
        train_df = train_df[columns_to_keep]
        dev_df = dev_df[columns_to_keep]
        test_df = test_df[columns_to_keep]
        
        # processed 디렉토리 생성
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # CSV로 저장
        train_df.to_csv(os.path.join(self.processed_dir, "train.csv"), index=False, encoding='utf-8-sig')
        dev_df.to_csv(os.path.join(self.processed_dir, "val.csv"), index=False, encoding='utf-8-sig')
        test_df.to_csv(os.path.join(self.processed_dir, "test.csv"), index=False, encoding='utf-8-sig')
        
        logger.info("\n✅ 전처리 완료!")
        logger.info(f"  저장 경로: {self.processed_dir}")
        logger.info("="*80 + "\n")
    
    def load_processed_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """전처리된 데이터 로드"""
        train_path = os.path.join(self.processed_dir, "train.csv")
        val_path = os.path.join(self.processed_dir, "val.csv")
        test_path = os.path.join(self.processed_dir, "test.csv")
        
        # 파일이 없으면 전처리 수행
        if not all(os.path.exists(p) for p in [train_path, val_path, test_path]):
            logger.info("전처리된 데이터가 없습니다. 전처리를 수행합니다.")
            self.prepare_dataset()
        
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
        
        return train_df, val_df, test_df


if __name__ == "__main__":
    # 테스트
    loader = UnsmileDataLoader("./data")
    loader.prepare_dataset()
    
    train_df, val_df, test_df = loader.load_processed_data()
    print(f"\nTrain: {len(train_df)}")
    print(f"Val: {len(val_df)}")
    print(f"Test: {len(test_df)}")
    print(f"\nLabel columns: {loader.label_columns}")
