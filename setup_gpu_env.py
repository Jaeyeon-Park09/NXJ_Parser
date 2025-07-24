#!/usr/bin/env python3
"""
NXJ 서버 GPU 환경 설정 체크 및 가이드
"""

import sys
import subprocess
import os
from pathlib import Path

def check_python_env():
    print("🐍 Python 환경 확인")
    print("=" * 50)
    print(f"Python 버전: {sys.version}")
    print(f"Python 경로: {sys.executable}")
    
    # 가상환경 확인
    if hasattr(sys, 'prefix') and hasattr(sys, 'base_prefix'):
        in_venv = sys.prefix != sys.base_prefix
        print(f"가상환경 사용 중: {in_venv}")
        if in_venv:
            print(f"가상환경 경로: {sys.prefix}")
    
    return True

def check_torch_cuda():
    print("\n🔥 PyTorch CUDA 지원 확인")
    print("=" * 50)
    
    try:
        import torch
        print(f"PyTorch 버전: {torch.__version__}")
        
        if hasattr(torch.version, 'cuda') and torch.version.cuda:
            print(f"CUDA 버전: {torch.version.cuda}")
        else:
            print("❌ CPU 전용 PyTorch 설치됨")
            return False
            
        cuda_available = torch.cuda.is_available()
        print(f"CUDA 사용 가능: {cuda_available}")
        
        if cuda_available:
            device_count = torch.cuda.device_count()
            print(f"GPU 개수: {device_count}")
            
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / (1024**3)
                print(f"  GPU {i}: {props.name} ({memory_gb:.1f}GB)")
        
        return cuda_available
        
    except ImportError:
        print("❌ PyTorch가 설치되지 않음")
        return False

def check_nvidia_driver():
    print("\n🚗 NVIDIA 드라이버 확인")
    print("=" * 50)
    
    try:
        # nvidia-smi 실행
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            for line in lines:
                if 'Driver Version' in line:
                    print(f"드라이버 버전 발견: {line.strip()}")
                    return True
            print("✅ nvidia-smi 실행 성공")
            return True
        else:
            print("❌ nvidia-smi 실행 실패")
            return False
    except FileNotFoundError:
        print("❌ nvidia-smi 명령어를 찾을 수 없음")
        return False
    except Exception as e:
        print(f"❌ nvidia-smi 오류: {e}")
        return False

def check_transformers():
    print("\n🤗 Transformers 라이브러리 확인")
    print("=" * 50)
    
    try:
        import transformers
        print(f"Transformers 버전: {transformers.__version__}")
        
        # BLIP-2 관련 모듈 확인
        try:
            from transformers import Blip2Processor, Blip2ForConditionalGeneration
            print("✅ BLIP-2 모듈 사용 가능")
            return True
        except ImportError as e:
            print(f"❌ BLIP-2 모듈 오류: {e}")
            return False
            
    except ImportError:
        print("❌ Transformers가 설치되지 않음")
        return False

def generate_setup_commands():
    print("\n🛠️  GPU 환경 설정 명령어")
    print("=" * 50)
    
    print("# 1. CUDA 지원 PyTorch 설치 (RTX 5090용)")
    print("pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    
    print("\n# 2. BLIP-2 필수 라이브러리")
    print("pip install transformers accelerate")
    
    print("\n# 3. 추가 GPU 최적화 라이브러리")
    print("pip install xformers  # 메모리 최적화")
    print("pip install bitsandbytes  # 양자화 지원")
    
    print("\n# 4. 환경변수 설정 (필요시)")
    print("export CUDA_VISIBLE_DEVICES=0")
    print("export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512")

def create_gpu_test_script():
    print("\n🧪 GPU 테스트 스크립트 생성")
    print("=" * 50)
    
    test_script = '''#!/usr/bin/env python3
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
'''
    
    with open('test_gpu.py', 'w') as f:
        f.write(test_script)
    
    print("✅ test_gpu.py 스크립트 생성 완료")
    print("실행: python test_gpu.py")

def main():
    print("🚀 NXJ 서버 GPU 환경 설정 체크")
    print("=" * 60)
    
    # 환경 체크
    python_ok = check_python_env()
    cuda_ok = check_torch_cuda()
    driver_ok = check_nvidia_driver()
    transformers_ok = check_transformers()
    
    print("\n" + "=" * 60)
    print("📋 체크 결과 요약")
    print("=" * 60)
    print(f"Python 환경: {'✅' if python_ok else '❌'}")
    print(f"PyTorch CUDA: {'✅' if cuda_ok else '❌'}")
    print(f"NVIDIA 드라이버: {'✅' if driver_ok else '❌'}")
    print(f"Transformers: {'✅' if transformers_ok else '❌'}")
    
    if all([python_ok, cuda_ok, driver_ok, transformers_ok]):
        print("\n🎉 모든 설정이 완료되었습니다!")
        print("바로 BLIP-2 + GMFT 멀티모달 파싱을 실행할 수 있습니다.")
        print("\n실행 명령어:")
        print("python main.py --force --single [파일명]")
    else:
        print("\n⚠️  추가 설정이 필요합니다.")
        generate_setup_commands()
    
    create_gpu_test_script()

if __name__ == "__main__":
    main() 