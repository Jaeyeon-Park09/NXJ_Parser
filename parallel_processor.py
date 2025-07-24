#!/usr/bin/env python3
"""
병렬 처리 전용 스크립트
대용량 PDF 파일 일괄 처리를 위한 고성능 병렬 처리 구현
"""

import logging
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import time
import psutil
import yaml

from utils.file_io import get_file_manager
from main import process_single_pdf_worker

logger = logging.getLogger(__name__)


class ParallelPDFProcessor:
    """병렬 PDF 처리 전용 클래스"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        ParallelPDFProcessor 초기화
        
        Args:
            config_path: 설정 파일 경로
        """
        self.config_path = config_path
        self.file_manager = get_file_manager(config_path)
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 병렬 처리 설정
        parallel_config = self.config.get('parallel', {})
        self.default_max_workers = parallel_config.get('max_workers', 4)
        self.chunk_size = parallel_config.get('chunk_size', 10)
        
    def get_optimal_worker_count(self, total_files: int) -> int:
        """
        최적의 워커 수 계산
        
        Args:
            total_files: 전체 파일 수
            
        Returns:
            최적 워커 수
        """
        # CPU 코어 수
        cpu_count = mp.cpu_count()
        
        # 메모리 고려 (GB당 1개 워커, 최소 2GB 확보)
        memory_gb = psutil.virtual_memory().total // (1024**3)
        memory_workers = max(1, (memory_gb - 2) // 2)  # 2GB는 시스템용으로 확보
        
        # 파일 수 고려
        file_workers = min(total_files, cpu_count)
        
        # 최종 워커 수 결정
        optimal_workers = min(cpu_count, memory_workers, file_workers, self.default_max_workers)
        
        logger.info(f"시스템 정보: CPU={cpu_count}코어, 메모리={memory_gb}GB")
        logger.info(f"최적 워커 수: {optimal_workers}")
        
        return optimal_workers
    
    def process_files_in_chunks(self, pdf_files: List[Path], 
                               max_workers: int = None,
                               force_reprocess: bool = False) -> Dict[str, Any]:
        """
        분할 단위로 파일 처리
        
        Args:
            pdf_files: 처리할 PDF 파일 리스트
            max_workers: 최대 워커 수
            force_reprocess: 강제 재처리 여부
            
        Returns:
            처리 결과 요약
        """
        if not pdf_files:
            return {"total": 0, "success": 0, "failed": 0, "processing_time": 0}
        
        total_files = len(pdf_files)
        
        if max_workers is None:
            max_workers = self.get_optimal_worker_count(total_files)
        
        logger.info(f"병렬 처리 시작: {total_files}개 파일, {max_workers}개 워커")
        
        start_time = time.time()
        success_count = 0
        failed_count = 0
        
        # 분할 단위로 분할
        chunks = [pdf_files[i:i + self.chunk_size] 
                 for i in range(0, total_files, self.chunk_size)]
        
        logger.info(f"처리 분할: {len(chunks)}개 ({self.chunk_size}개씩)")
        
        # 전체 진행률 표시
        with tqdm(total=total_files, desc="전체 진행률", unit="파일") as total_pbar:
            
            for chunk_idx, chunk in enumerate(chunks):
                logger.info(f"분할 {chunk_idx + 1}/{len(chunks)} 처리 중")
                
                # 분할 내 병렬 처리
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    # 작업 제출
                    future_to_pdf = {
                        executor.submit(
                            process_single_pdf_worker, 
                            str(pdf_path), 
                            self.config_path, 
                            force_reprocess
                        ): pdf_path
                        for pdf_path in chunk
                    }
                    
                    # 분할 내 결과 수집
                    chunk_success = 0
                    chunk_failed = 0
                    
                    for future in as_completed(future_to_pdf):
                        pdf_path = future_to_pdf[future]
                        try:
                            success = future.result(timeout=300)  # 5분 타임아웃
                            if success:
                                chunk_success += 1
                                success_count += 1
                            else:
                                chunk_failed += 1
                                failed_count += 1
                        except Exception as e:
                            logger.error(f"프로세스 오류 {pdf_path}: {str(e)}")
                            chunk_failed += 1
                            failed_count += 1
                        
                        # 진행률 업데이트
                        total_pbar.update(1)
                        total_pbar.set_postfix({
                            'Success': success_count,
                            'Failed': failed_count,
                            'Chunk': f"{chunk_idx + 1}/{len(chunks)}"
                        })
                
                logger.info(f"분할 {chunk_idx + 1} 완료: 성공 {chunk_success}, 실패 {chunk_failed}")
                
                # 메모리 정리를 위한 잠시 대기
                if chunk_idx < len(chunks) - 1:
                    time.sleep(1)
        
        processing_time = time.time() - start_time
        
        result = {
            "total": total_files,
            "success": success_count,
            "failed": failed_count,
            "success_rate": (success_count / total_files * 100) if total_files > 0 else 0,
            "processing_time": processing_time,
            "files_per_second": total_files / processing_time if processing_time > 0 else 0,
            "splits_processed": len(chunks),
            "workers_used": max_workers
        }
        
        return result
    
    def process_with_progress_monitoring(self, pdf_files: List[Path],
                                       max_workers: int = None,
                                       force_reprocess: bool = False,
                                       save_progress: bool = True) -> Dict[str, Any]:
        """
        진행률 모니터링과 함께 처리
        
        Args:
            pdf_files: 처리할 PDF 파일 리스트
            max_workers: 최대 워커 수
            force_reprocess: 강제 재처리 여부
            save_progress: 진행률 저장 여부
            
        Returns:
            처리 결과 요약
        """
        # 시스템 리소스 모니터링 시작
        initial_memory = psutil.virtual_memory().percent
        initial_cpu = psutil.cpu_percent(interval=1)
        
        logger.info(f"처리 시작 시 시스템 상태: CPU {initial_cpu:.1f}%, 메모리 {initial_memory:.1f}%")
        
        # 처리 실행
        result = self.process_files_in_chunks(pdf_files, max_workers, force_reprocess)
        
        # 최종 시스템 상태
        final_memory = psutil.virtual_memory().percent
        final_cpu = psutil.cpu_percent(interval=1)
        
        result.update({
            "system_stats": {
                "initial": {"cpu": initial_cpu, "memory": initial_memory},
                "final": {"cpu": final_cpu, "memory": final_memory}
            }
        })
        
        # 진행률 저장
        if save_progress:
            self._save_progress_report(result)
        
        return result
    
    def _save_progress_report(self, result: Dict[str, Any]):
        """진행률 리포트 저장"""
        try:
            report_path = self.file_manager.logs_dir / f"parallel_progress_{int(time.time())}.json"
            
            import json
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            logger.info(f"진행률 리포트 저장: {report_path}")
            
        except Exception as e:
            logger.warning(f"진행률 리포트 저장 실패: {str(e)}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="NXJ_Parser 병렬 처리 스크립트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python parallel_processor.py                     # 기본 병렬 처리
  python parallel_processor.py --workers 8        # 워커 수 지정
  python parallel_processor.py --chunk-size 20    # 청크 크기 지정
  python parallel_processor.py --force            # 강제 재처리
  python parallel_processor.py --dry-run          # 처리 계획만 출력
        """
    )
    
    parser.add_argument(
        '--workers', type=int, metavar='N',
        help='병렬 처리 워커 수 (기본값: 자동 계산)'
    )
    
    parser.add_argument(
        '--chunk-size', type=int, metavar='N',
        help='청크 크기 (기본값: config.yaml 설정값)'
    )
    
    parser.add_argument(
        '--force', action='store_true',
        help='이미 처리된 파일도 강제로 재처리'
    )
    
    parser.add_argument(
        '--config', type=str, default='config.yaml',
        help='설정 파일 경로'
    )
    
    parser.add_argument(
        '--dry-run', action='store_true',
        help='실제 처리 없이 계획만 출력'
    )
    
    parser.add_argument(
        '--no-progress-save', action='store_true',
        help='진행률 리포트 저장 비활성화'
    )
    
    args = parser.parse_args()
    
    try:
        # 설정 파일 확인
        if not Path(args.config).exists():
            logger.error(f"설정 파일을 찾을 수 없습니다: {args.config}")
            sys.exit(1)
        
        # 병렬 프로세서 초기화
        processor = ParallelPDFProcessor(args.config)
        
        # 청크 크기 설정
        if args.chunk_size:
            processor.chunk_size = args.chunk_size
        
        # 처리할 파일 목록 가져오기
        if args.force:
            pdf_files = processor.file_manager.get_pdf_files()
            logger.info("강제 재처리 모드: 모든 파일을 다시 처리합니다")
        else:
            pdf_files = processor.file_manager.get_unprocessed_files()
        
        if not pdf_files:
            print("처리할 파일이 없습니다.")
            return
        
        # 처리 계획 출력
        total_files = len(pdf_files)
        optimal_workers = processor.get_optimal_worker_count(total_files)
        max_workers = args.workers or optimal_workers
        
        print(f"\n=== 병렬 처리 계획 ===")
        print(f"처리 대상: {total_files}개 파일")
        print(f"워커 수: {max_workers}개")
        print(f"청크 크기: {processor.chunk_size}개")
        print(f"예상 청크 수: {(total_files + processor.chunk_size - 1) // processor.chunk_size}개")
        
        if args.dry_run:
            print("Dry-run 모드: 실제 처리는 수행하지 않습니다.")
            return
        
        # 사용자 확인
        if total_files > 100:
            response = input(f"\n{total_files}개의 파일을 처리합니다. 계속하시겠습니까? (y/N): ")
            if response.lower() != 'y':
                print("처리가 취소되었습니다.")
                return
        
        # 병렬 처리 실행
        print(f"\n병렬 처리를 시작합니다...")
        result = processor.process_with_progress_monitoring(
            pdf_files,
            max_workers=max_workers,
            force_reprocess=args.force,
            save_progress=not args.no_progress_save
        )
        
        # 결과 출력
        print(f"\n=== 처리 결과 ===")
        print(f"전체 파일: {result['total']}")
        print(f"성공: {result['success']}")
        print(f"실패: {result['failed']}")
        print(f"성공률: {result['success_rate']:.1f}%")
        print(f"처리 시간: {result['processing_time']:.1f}초")
        print(f"처리 속도: {result['files_per_second']:.2f}파일/초")
        print(f"사용된 워커 수: {result['workers_used']}")
        
        # 시스템 리소스 정보
        if 'system_stats' in result:
            sys_stats = result['system_stats']
            print(f"\n=== 시스템 리소스 ===")
            print(f"CPU 사용률: {sys_stats['initial']['cpu']:.1f}% → {sys_stats['final']['cpu']:.1f}%")
            print(f"메모리 사용률: {sys_stats['initial']['memory']:.1f}% → {sys_stats['final']['memory']:.1f}%")
        
        if result['failed'] > 0:
            print(f"\n실패한 파일들은 logs/failed.txt에서 확인할 수 있습니다.")
        
    except KeyboardInterrupt:
        logger.info("사용자에 의해 중단되었습니다")
        sys.exit(130)
    except Exception as e:
        logger.error(f"예상치 못한 오류: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    main() 