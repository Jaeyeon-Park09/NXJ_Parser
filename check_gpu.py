#!/usr/bin/env python3
"""
NXJ 서버 GPU 확인 스크립트
"""

import torch
import sys
import subprocess

def check_gpu_info():
    print("🔍 NXJ 서버 GPU 확인")
    print("=" * 50)
    
    # PyTorch CUDA 확인
    print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"GPU 개수: {gpu_count}")
        
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            print(f"\n🚀 GPU {i}: {props.name}")
            print(f"   메모리: {memory_gb:.1f}GB")
            print(f"   Compute Capability: {props.major}.{props.minor}")
            
            # RTX 5090 확인
            if "5090" in props.name:
                print("   ✅ RTX 5090 발견! BLIP-2 실행 가능")
                return True
        
        return True  # 다른 GPU라도 있으면 실행 가능
    else:
        print("❌ CUDA GPU를 찾을 수 없습니다")
        return False

def check_nvidia_smi():
    print("\n🔧 nvidia-smi 확인:")
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(result.stdout)
        else:
            print("nvidia-smi 실행 실패")
    except Exception as e:
        print(f"nvidia-smi 오류: {e}")

def main():
    has_gpu = check_gpu_info()
    check_nvidia_smi()
    
    if has_gpu:
        print("\n🎯 결론: GPU가 있습니다! BLIP-2 + GMFT 멀티모달 파싱 실행 가능")
        print("   명령어: python main.py --force --single [파일명]")
    else:
        print("\n⚠️  결론: GPU가 없습니다. CPU로만 실행 가능 (매우 느림)")
        
        # BLIP-2 비활성화 제안
        print("\n💡 CPU 전용 실행을 위한 설정:")
        print("   config.yaml에서 image_captioning.enabled: false로 설정")

if __name__ == "__main__":
    main() 