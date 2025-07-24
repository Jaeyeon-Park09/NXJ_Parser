"""
파일 입출력 및 저장 관리 모듈
PDF 파일 검색, JSON 저장, 로그 관리 등
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from .config_loader import get_config, get_section, ensure_directories

logger = logging.getLogger(__name__)


class FileManager:
    """파일 입출력 및 저장 관리 클래스"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        FileManager 초기화
        
        Args:
            config_path: 설정 파일 경로
        """
        self.config = get_config(config_path)
        self.paths_config = get_section('paths', config_path)
        
        # 경로 설정
        self.pdf_input_dir = Path(self.paths_config.get('pdf_input', 'pdf_files'))
        self.pdf_input_other_dir = Path(self.paths_config.get('pdf_input_other', 'pdf_files_o'))
        self.output_dir = Path(self.paths_config.get('output', 'output'))
        self.image_output_dir = Path(self.paths_config.get('image_output', 'image_png'))
        self.logs_dir = Path(self.paths_config.get('logs', 'logs'))
        
        # 디렉토리 생성
        self._ensure_directories()
        
        # 로그 설정
        self._setup_logging()
    
    def _ensure_directories(self):
        """필요한 디렉토리들을 생성"""
        directories = [
            self.pdf_input_dir,
            self.pdf_input_other_dir,
            self.output_dir,
            self.image_output_dir,
            self.logs_dir
        ]
        
        ensure_directories(directories)
        logger.info("모든 필요 디렉토리가 준비되었습니다")
    
    def _setup_logging(self):
        """로깅 설정"""
        logging_config = self.config.get('logging', {})
        log_level = logging_config.get('level', 'INFO')
        log_format = logging_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # 파일 로거 설정
        log_file = self.logs_dir / f"nxj_parser_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(getattr(logging, log_level))
        file_handler.setFormatter(logging.Formatter(log_format))
        
        # 루트 로거에 핸들러 추가
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
    
    def get_pdf_files(self, include_other_dir: bool = False) -> List[Path]:
        """
        처리할 PDF 파일 목록 반환
        
        Args:
            include_other_dir: pdf_files_o 디렉토리 포함 여부 (기본값: False)
            
        Returns:
            PDF 파일 경로 리스트
        """
        pdf_files = []
        
        # 메인 PDF 디렉토리
        if self.pdf_input_dir.exists():
            pdf_files.extend(list(self.pdf_input_dir.glob('*.pdf')))
            pdf_files.extend(list(self.pdf_input_dir.glob('**/*.pdf')))  # 하위 디렉토리 포함
        
        # 기타 PDF 디렉토리 (선택적)
        if include_other_dir and self.pdf_input_other_dir.exists():
            pdf_files.extend(list(self.pdf_input_other_dir.glob('*.pdf')))
            pdf_files.extend(list(self.pdf_input_other_dir.glob('**/*.pdf')))
        
        # 중복 제거 및 정렬
        unique_files = list(set(pdf_files))
        unique_files.sort()
        
        logger.info(f"총 {len(unique_files)}개의 PDF 파일 발견")
        return unique_files
    
    def get_output_path(self, pdf_path: Path, extension: str = ".json") -> Path:
        """
        PDF 파일에 대응하는 출력 파일 경로 생성
        
        Args:
            pdf_path: 원본 PDF 파일 경로
            extension: 출력 파일 확장자
            
        Returns:
            출력 파일 경로
        """
        # PDF 파일명에서 확장자 제거하고 새 확장자 추가
        output_filename = pdf_path.stem + extension
        return self.output_dir / output_filename
    
    def save_json_result(self, result_data: Dict[str, Any], output_path: Path) -> bool:
        """
        결과 데이터를 JSON 파일로 저장
        
        Args:
            result_data: 저장할 결과 데이터
            output_path: 출력 파일 경로
            
        Returns:
            저장 성공 여부
        """
        try:
            # 메타데이터 추가
            result_data['metadata'] = {
                'processed_at': datetime.now().isoformat(),
                'processor_version': '1.0.0',
                'output_file': str(output_path)
            }
            
            # JSON 파일로 저장
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"결과 저장 완료: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"JSON 저장 오류: {output_path}, 오류: {str(e)}")
            return False
    
    def load_json_result(self, json_path: Path) -> Optional[Dict[str, Any]]:
        """
        JSON 결과 파일 로드
        
        Args:
            json_path: JSON 파일 경로
            
        Returns:
            로드된 데이터 또는 None
        """
        try:
            if not json_path.exists():
                logger.warning(f"JSON 파일이 존재하지 않습니다: {json_path}")
                return None
            
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"JSON 파일 로드 완료: {json_path}")
            return data
            
        except Exception as e:
            logger.error(f"JSON 로드 오류: {json_path}, 오류: {str(e)}")
            return None
    
    def is_already_processed(self, pdf_path: Path) -> bool:
        """
        PDF 파일이 이미 처리되었는지 확인
        
        Args:
            pdf_path: PDF 파일 경로
            
        Returns:
            처리 완료 여부
        """
        output_path = self.get_output_path(pdf_path)
        return output_path.exists()
    
    def log_failed_file(self, pdf_path: Path, error_message: str):
        """
        실패한 파일 정보를 로그에 기록
        
        Args:
            pdf_path: 실패한 PDF 파일 경로
            error_message: 오류 메시지
        """
        failed_log_path = self.logs_dir / "failed.txt"
        
        try:
            with open(failed_log_path, 'a', encoding='utf-8') as f:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"{timestamp}\t{pdf_path}\t{error_message}\n")
            
            logger.error(f"실패 로그 기록: {pdf_path} - {error_message}")
            
        except Exception as e:
            logger.error(f"실패 로그 기록 오류: {str(e)}")
    
    def get_processing_stats(self) -> Dict[str, int]:
        """
        처리 통계 정보 반환
        
        Returns:
            처리 통계 딕셔너리
        """
        pdf_files = self.get_pdf_files()
        total_files = len(pdf_files)
        
        processed_files = 0
        for pdf_path in pdf_files:
            if self.is_already_processed(pdf_path):
                processed_files += 1
        
        return {
            'total_files': total_files,
            'processed_files': processed_files,
            'remaining_files': total_files - processed_files,
            'completion_rate': (processed_files / total_files * 100) if total_files > 0 else 0
        }
    
    def cleanup_incomplete_files(self):
        """
        불완전한 출력 파일들을 정리
        """
        try:
            cleaned_count = 0
            
            for output_file in self.output_dir.glob('*.json'):
                try:
                    # JSON 파일 유효성 검사
                    data = self.load_json_result(output_file)
                    if data is None or not self._is_valid_result(data):
                        output_file.unlink()
                        cleaned_count += 1
                        logger.info(f"불완전한 파일 제거: {output_file}")
                except Exception as e:
                    logger.warning(f"파일 검사 중 오류: {output_file}, {str(e)}")
            
            logger.info(f"총 {cleaned_count}개의 불완전한 파일이 정리되었습니다")
            
        except Exception as e:
            logger.error(f"파일 정리 중 오류: {str(e)}")
    
    def _is_valid_result(self, data: Dict[str, Any]) -> bool:
        """
        결과 데이터의 유효성 검사
        
        Args:
            data: 검사할 데이터
            
        Returns:
            유효성 여부
        """
        required_fields = ['pdf_path', 'blocks', 'processing_info']
        return all(field in data for field in required_fields)
    
    def get_unprocessed_files(self) -> List[Path]:
        """
        아직 처리되지 않은 PDF 파일 목록 반환
        
        Returns:
            미처리 PDF 파일 경로 리스트
        """
        all_files = self.get_pdf_files()
        unprocessed_files = []
        
        for pdf_path in all_files:
            if not self.is_already_processed(pdf_path):
                unprocessed_files.append(pdf_path)
        
        logger.info(f"미처리 파일 수: {len(unprocessed_files)}")
        return unprocessed_files
    
    def create_processing_report(self) -> Dict[str, Any]:
        """
        처리 결과 리포트 생성
        
        Returns:
            처리 리포트 딕셔너리
        """
        stats = self.get_processing_stats()
        
        # 실패 파일 정보 수집
        failed_log_path = self.logs_dir / "failed.txt"
        failed_files = []
        
        if failed_log_path.exists():
            try:
                with open(failed_log_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    failed_files = [line.strip().split('\t') for line in lines if line.strip()]
            except Exception as e:
                logger.warning(f"실패 로그 읽기 오류: {str(e)}")
        
        report = {
            'processing_stats': stats,
            'failed_files_count': len(failed_files),
            'failed_files': failed_files[-10:],  # 최근 10개만
            'report_generated_at': datetime.now().isoformat(),
            'output_directory': str(self.output_dir),
            'logs_directory': str(self.logs_dir)
        }
        
        return report
    
    def save_processing_report(self) -> Path:
        """
        처리 리포트를 파일로 저장
        
        Returns:
            저장된 리포트 파일 경로
        """
        report = self.create_processing_report()
        report_path = self.logs_dir / f"processing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            logger.info(f"처리 리포트 저장 완료: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"리포트 저장 오류: {str(e)}")
            return None


def get_file_manager(config_path: str = "config.yaml") -> FileManager:
    """
    FileManager 인스턴스를 반환하는 편의 함수
    
    Args:
        config_path: 설정 파일 경로
        
    Returns:
        FileManager 인스턴스
    """
    return FileManager(config_path)


def setup_directories(config_path: str = "config.yaml") -> None:
    """
    필요한 디렉토리들을 설정에 따라 생성하는 편의 함수
    
    Args:
        config_path: 설정 파일 경로
    """
    try:
        fm = FileManager(config_path)
        logger.info("디렉토리 설정 완료")
    except Exception as e:
        logger.error(f"디렉토리 설정 실패: {e}")


def save_json(data: Dict[str, Any], file_path: str, config_path: str = "config.yaml") -> bool:
    """
    JSON 데이터를 파일로 저장하는 편의 함수
    
    Args:
        data: 저장할 데이터
        file_path: 저장할 파일 경로
        config_path: 설정 파일 경로
        
    Returns:
        저장 성공 여부
    """
    try:
        fm = FileManager(config_path)
        return fm.save_json_result(data, Path(file_path))
    except Exception as e:
        logger.error(f"JSON 저장 실패: {file_path}, 오류: {e}")
        return False


def load_json(file_path: str, config_path: str = "config.yaml") -> Optional[Dict[str, Any]]:
    """
    JSON 파일을 로드하는 편의 함수
    
    Args:
        file_path: 로드할 파일 경로
        config_path: 설정 파일 경로
        
    Returns:
        로드된 데이터 또는 None
    """
    try:
        fm = FileManager(config_path)
        return fm.load_json_result(Path(file_path))
    except Exception as e:
        logger.error(f"JSON 로드 실패: {file_path}, 오류: {e}")
        return None


if __name__ == "__main__":
    # 테스트용 코드
    fm = FileManager()
    
    print("파일 관리자 테스트")
    print(f"PDF 파일 수: {len(fm.get_pdf_files())}")
    
    stats = fm.get_processing_stats()
    print(f"처리 통계:")
    print(f"  전체: {stats['total_files']}")
    print(f"  처리완료: {stats['processed_files']}")
    print(f"  미처리: {stats['remaining_files']}")
    print(f"  완료율: {stats['completion_rate']:.1f}%")
    
    # 리포트 생성
    report_path = fm.save_processing_report()
    if report_path:
        print(f"리포트 저장: {report_path}") 