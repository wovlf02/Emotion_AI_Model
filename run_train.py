#!/usr/bin/env python
"""
모델 학습 실행 스크립트
"""
import sys
import os
# src 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
# src/train.py 실행
from train import main
if __name__ == "__main__":
    main()
