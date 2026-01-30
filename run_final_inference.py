#!/usr/bin/env python
"""
최종 평가 실행 스크립트
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from final_inference import main
if __name__ == "__main__":
    main()
