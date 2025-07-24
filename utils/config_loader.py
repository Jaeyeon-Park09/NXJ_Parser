"""
공통 설정 로더 모듈
YAML 설정 파일을 캐싱하여 효율적으로 관리
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ConfigLoader:
    """싱글톤 패턴 설정 로더 클래스"""
    
    _instance = None
    _config_cache = {}
    _config_timestamps = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    def get_config(cls, config_path: str = "config.yaml") -> Dict[str, Any]:
        """
        설정 파일을 로드하고 캐싱
        
        Args:
            config_path: 설정 파일 경로
            
        Returns:
            설정 딕셔너리
        """
        config_path = Path(config_path)
        config_key = str(config_path.absolute())
        
        # 파일 존재 여부 확인
        if not config_path.exists():
            logger.warning(f"설정 파일이 없습니다: {config_path}")
            return {}
        
        try:
            # 파일 수정 시간 확인
            current_mtime = config_path.stat().st_mtime
            
            # 캐시된 설정이 있고 최신인지 확인
            if (config_key in cls._config_cache and 
                config_key in cls._config_timestamps and 
                cls._config_timestamps[config_key] >= current_mtime):
                
                logger.debug(f"캐시된 설정 사용: {config_path}")
                return cls._config_cache[config_key]
            
            # 설정 파일 로드
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 캐시 업데이트
            cls._config_cache[config_key] = config
            cls._config_timestamps[config_key] = current_mtime
            
            logger.info(f"설정 파일 로드 완료: {config_path}")
            return config
            
        except Exception as e:
            logger.error(f"설정 파일 로드 실패: {config_path}, 오류: {e}")
            return {}
    
    @classmethod
    def clear_cache(cls):
        """설정 캐시 초기화"""
        cls._config_cache.clear()
        cls._config_timestamps.clear()
        logger.info("설정 캐시가 초기화되었습니다")
    
    @classmethod
    def get_section(cls, section_name: str, config_path: str = "config.yaml") -> Dict[str, Any]:
        """
        설정의 특정 섹션 반환
        
        Args:
            section_name: 섹션 이름
            config_path: 설정 파일 경로
            
        Returns:
            해당 섹션의 설정 딕셔너리
        """
        config = cls.get_config(config_path)
        return config.get(section_name, {})
    
    @classmethod
    def get_value(cls, key_path: str, default: Any = None, config_path: str = "config.yaml") -> Any:
        """
        점으로 구분된 키 경로로 설정값 반환
        
        Args:
            key_path: 점으로 구분된 키 경로 (예: "paths.output")
            default: 기본값
            config_path: 설정 파일 경로
            
        Returns:
            설정값 또는 기본값
        """
        config = cls.get_config(config_path)
        
        try:
            value = config
            for key in key_path.split('.'):
                value = value[key]
            return value
        except (KeyError, TypeError):
            logger.debug(f"설정 키를 찾을 수 없음: {key_path}, 기본값 사용: {default}")
            return default


# 편의 함수들
def get_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """ConfigLoader.get_config의 편의 함수"""
    return ConfigLoader.get_config(config_path)

def get_section(section_name: str, config_path: str = "config.yaml") -> Dict[str, Any]:
    """ConfigLoader.get_section의 편의 함수"""
    return ConfigLoader.get_section(section_name, config_path)

def get_value(key_path: str, default: Any = None, config_path: str = "config.yaml") -> Any:
    """ConfigLoader.get_value의 편의 함수"""
    return ConfigLoader.get_value(key_path, default, config_path)

def clear_config_cache():
    """ConfigLoader.clear_cache의 편의 함수"""
    ConfigLoader.clear_cache()


# 공통 유틸리티 함수들
def ensure_directories(directories: list) -> None:
    """
    디렉토리 리스트를 안전하게 생성
    
    Args:
        directories: 생성할 디렉토리 경로 리스트
    """
    for directory in directories:
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"디렉토리 생성 실패: {directory}, 오류: {e}")

def safe_bbox_conversion(bbox: Any) -> list:
    """
    다양한 타입의 bbox를 안전하게 리스트로 변환
    
    Args:
        bbox: bbox 객체 (리스트, 객체 등)
        
    Returns:
        [x0, y0, x1, y1] 형태의 리스트
    """
    if not bbox:
        return []
    
    # 이미 리스트인 경우
    if isinstance(bbox, list) and len(bbox) >= 4:
        return bbox[:4]
    
    # bbox 속성이 있는 객체
    if hasattr(bbox, 'bbox'):
        try:
            return list(bbox.bbox)[:4]
        except:
            pass
    
    # polygon 속성이 있는 객체
    if hasattr(bbox, 'polygon') and bbox.polygon:
        try:
            coords = bbox.polygon
            if len(coords) >= 2:
                return [coords[0][0], coords[0][1], coords[1][0], coords[1][1]]
        except:
            pass
    
    # 문자열로 변환 시도
    try:
        str_bbox = str(bbox)
        if '[' in str_bbox and ']' in str_bbox:
            # 문자열에서 숫자 추출 시도
            import re
            numbers = re.findall(r'-?\d+\.?\d*', str_bbox)
            if len(numbers) >= 4:
                return [float(n) for n in numbers[:4]]
    except:
        pass
    
    logger.warning(f"bbox 변환 실패: {type(bbox)}")
    return []

def safe_page_number(page: Any, offset: int = 0) -> int:
    """
    페이지 번호를 안전하게 정수로 변환
    
    Args:
        page: 페이지 번호 객체
        offset: 오프셋 값
        
    Returns:
        정수 페이지 번호
    """
    if isinstance(page, (int, float)):
        return int(page) + offset
    
    try:
        return int(page) + offset
    except (ValueError, TypeError):
        logger.debug(f"페이지 번호 변환 실패: {type(page)}, 기본값 {offset} 사용")
        return offset


if __name__ == "__main__":
    # 테스트용 코드
    print("ConfigLoader 테스트")
    
    config = get_config()
    print(f"설정 로드 완료: {len(config)} 개 섹션")
    
    # 섹션별 테스트
    paths = get_section('paths')
    print(f"paths 섹션: {paths}")
    
    # 키 경로 테스트
    output_dir = get_value('paths.output', 'output')
    print(f"출력 디렉토리: {output_dir}")
    
    # 캐시 테스트
    config2 = get_config()  # 캐시된 버전 사용
    print(f"캐시 테스트 완료: {config is config2}") 