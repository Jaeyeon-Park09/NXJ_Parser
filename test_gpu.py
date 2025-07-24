#!/usr/bin/env python3
"""GPU 테스트 스크립트"""

import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import time

def test_gpu():
    print("🔥 GPU 테스트 시작")
    
    # CUDA 확인
    if not torch.cuda.is_available():
        print("❌ CUDA 사용 불가")
        return False
    
    device = torch.device('cuda')
    print(f"✅ 사용 중인 디바이스: {device}")
    
    # GPU 메모리 확인
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"총 GPU 메모리: {total_memory:.1f}GB")
    
    # 간단한 텐서 연산 테스트
    print("텐서 연산 테스트...")
    x = torch.randn(1000, 1000).to(device)
    y = torch.randn(1000, 1000).to(device)
    start_time = time.time()
    z = torch.matmul(x, y)
    end_time = time.time()
    print(f"행렬 곱셈 시간: {end_time - start_time:.4f}초")
    
    # 메모리 사용량 확인
    allocated = torch.cuda.memory_allocated() / 1024**3
    print(f"할당된 메모리: {allocated:.2f}GB")
    
    return True

if __name__ == "__main__":
    test_gpu()
