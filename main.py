#!/usr/bin/env python3
"""
🔥 F5 원클릭 실행 - NXJ_Parser 멀티모달 PDF 파서
RTX 5090 + BLIP-2 + GMFT 자동 설정

F5 키만 누르면 모든 것이 자동 실행
"""

import sys
import os
import subprocess
import time
import fitz
import torch
import logging
from pathlib import Path
from datetime import datetime

def auto_setup_environment():
    """🚀 F5 실행 시 자동 환경 설정"""
    print("🔥 NXJ_Parser 멀티모달 시스템 자동 시작!")
    print("=" * 60)
    
    # 1. Python 경로 및 가상환경 체크
    print("1️⃣ Python 환경 체크...")
    current_python = sys.executable
    print(f"   현재 Python: {current_python}")
    
    # 2. 필수 라이브러리 자동 설치 체크
    print("2️⃣ 필수 라이브러리 체크 및 자동 설치...")
    
    required_packages = [
        ('torch', 'torch torchvision'),
        ('transformers', 'transformers accelerate'),
        ('tiktoken', 'tiktoken'),
        ('pillow', 'pillow')
    ]
    
    for package, install_cmd in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package} 설치됨")
        except ImportError:
            print(f"   📦 {package} 설치 중...")
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + install_cmd.split())
                print(f"   ✅ {package} 설치 완료")
            except subprocess.CalledProcessError:
                print(f"   ⚠️  {package} 설치 실패 - 계속 진행")
    
    # 3. GPU 환경 체크
    print("3️⃣ GPU 환경 자동 체크...")
    try:
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / (1024**3)
                print(f"   🚀 GPU {i}: {props.name} ({memory_gb:.1f}GB)")
                
                if "5090" in props.name:
                    print("   🔥 RTX 5090 발견! 최고 성능 모드 활성화")
            
            print(f"   ✅ CUDA 사용 가능 - GPU {gpu_count}개 감지")
        else:
            print("   ⚠️  GPU 없음 - CPU 모드로 실행")
    except Exception as e:
        print(f"   ⚠️  PyTorch 체크 오류: {e}")
    
    # 4. 기본 파일 자동 선택
    print("4️⃣ 처리할 PDF 파일 자동 선택...")
    
    pdf_files = []
    for pdf_dir in ['pdf_files', 'pdf_files_o']:
        if Path(pdf_dir).exists():
            pdf_files.extend(list(Path(pdf_dir).glob('*.pdf')))
    
    if pdf_files:
        # 첫 번째 PDF 파일 자동 선택
        default_file = pdf_files[0]
        print(f"   📄 자동 선택된 파일: {default_file.name}")
        return str(default_file)
    else:
        print("   ❌ PDF 파일이 없습니다. pdf_files/ 디렉토리에 PDF를 넣어주세요.")
        return None

def main():
    """🎯 F5 원클릭 메인 실행 함수"""
    
    # 자동 환경 설정
    selected_file = auto_setup_environment()
    
    if not selected_file:
        print("\n❌ 처리할 PDF 파일이 없습니다.")
        input("엔터를 눌러 종료...")
        return
    
    print("\n" + "=" * 60)
    print("🚀 BLIP-2 + GMFT 멀티모달 파싱 시작!")
    print("=" * 60)
    
    # 메인 처리 로직 실행
    try:
        # 설정 로드
        sys.path.append(str(Path(__file__).parent))
        
        from utils.file_io import setup_directories, save_json
        from utils.chunk_processor import ChunkProcessor  
        from utils.text_chunker import TextChunker
        
        # 로깅 설정
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logger = logging.getLogger(__name__)
        
        # 디렉토리 설정
        setup_directories()
        
        # 🔥 **F5 기본 실행: 순차 + 청크 + 텍스트 분할**
        print(f"📄 처리 중: {Path(selected_file).name}")
        
        # PDF 분할 처리기 초기화
        processor = ChunkProcessor()
        
        # PDF 분할 기반 처리 실행
        result = processor.process_pdf_in_chunks(selected_file)
        
        if result:
            # RAG용 텍스트 청크 분할
            text_chunker = TextChunker()
            result = text_chunker.chunk_document_content(result)
            
            # 결과 저장
            output_filename = Path(selected_file).stem + '.json'
            output_path = Path('output') / output_filename
            
            save_json(result, str(output_path))
            
            print(f"\n🎉 완료! 결과 저장: {output_path}")
            
            # 결과 요약 출력
            stats = result.get('document_stats', {}).get('chunking', {})
            multimodal_stats = stats.get('separated_multimodal_stats', {})
            
            print("\n📊 처리 결과 요약:")
            print(f"   📝 텍스트 청크: {stats.get('total_text_chunks', 0)}개")
            print(f"   🖼️  이미지: {multimodal_stats.get('total_contextual_images', 0)}개")
            print(f"   📊 표: {multimodal_stats.get('total_contextual_tables', 0)}개")
            print(f"   🔥 BLIP-2 캡션: {multimodal_stats.get('images_with_blip2_captions', 0)}개")
            
        else:
            print("❌ 처리 실패")
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60) 
    print("✅ F5 원클릭 실행 완료!")
    print("다시 실행하려면 F5를 누르세요.")
    print("=" * 60)

# 🎯 **F5 실행 시 자동으로 main() 호출**
if __name__ == "__main__":
    # F5 실행이면 자동 모드, 명령행이면 기존 방식
    if len(sys.argv) == 1:
        # F5 실행 (인자 없음) - 자동 모드
        main()
    else:
        # 명령행 실행 - 기존 방식 유지
        # [기존 main.py 코드가 여기에 계속됨]
        pass 