#!/usr/bin/env python
"""
모델 추론 실행 스크립트
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from inference import main
if __name__ == "__main__":
    main()
