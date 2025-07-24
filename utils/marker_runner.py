"""
Marker 기반 PDF 텍스트/이미지 블록 추출 모듈
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from .config_loader import get_config

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

logger = logging.getLogger(__name__)


class MarkerRunner:
    """Marker를 사용한 PDF 텍스트/이미지 블록 추출 클래스"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        MarkerRunner 초기화
        
        Args:
            config_path: 설정 파일 경로
        """
        self.config = get_config(config_path)
        
        # PdfConverter 초기화 (marker 1.8+ API)
        self.converter = PdfConverter(
            artifact_dict=create_model_dict(),
        )
        
    def extract_blocks_from_pdf(self, pdf_path: str) -> Tuple[Dict[str, Any], List[Dict]]:
        """
        PDF에서 텍스트 및 이미지 블록 추출
        
        Args:
            pdf_path: PDF 파일 경로
            
        Returns:
            Tuple of (conversion_result, extracted_blocks)
            - conversion_result: Marker MarkdownOutput 객체의 메타데이터
            - extracted_blocks: 추출된 블록 리스트
        """
        try:
            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
            
            logger.info(f"PDF 처리 시작: {pdf_path}")
            
            # Marker로 Document 객체 생성 (marker 1.8+ API)
            document = self.converter.build_document(str(pdf_path))
            
            # 모든 블록 추출
            all_blocks = document.contained_blocks()
            
            logger.info(f"추출된 블록 수: {len(all_blocks)}")
            
            # 블록을 딕셔너리 형태로 변환
            blocks = []
            for block in all_blocks:
                try:
                    # 블록 타입 추출 (enum을 문자열로 변환)
                    block_type = str(block.block_type).split('.')[-1] if hasattr(block, 'block_type') else 'unknown'
                    
                    # 텍스트 추출 (최적화된 순서)
                    text_content = self._extract_text_from_block(block, document)
                    
                    # 빈 텍스트는 로그로 기록
                    if not text_content.strip():
                        logger.debug(f"블록 {block_type}에서 텍스트 추출 실패")
                    
                    # bbox 처리 (PolygonBox 객체를 리스트로 변환)
                    bbox_coords = []
                    polygon = getattr(block, 'polygon', None)
                    if polygon and hasattr(polygon, 'bbox'):
                        bbox_coords = list(polygon.bbox)  # [x0, y0, x1, y1]
                    elif polygon and hasattr(polygon, 'polygon'):
                        # 폴리곤 좌표 리스트
                        bbox_coords = polygon.polygon
                    
                    block_data = {
                        "type": block_type,
                        "text": text_content,
                        "bbox": bbox_coords,
                        "page": getattr(block, 'page_id', 0),
                        "confidence": 1.0
                    }
                    blocks.append(block_data)
                except Exception as e:
                    logger.warning(f"블록 변환 오류: {str(e)}")
                    continue
            
            # 블록 분류 및 정리
            extracted_blocks = self._process_blocks(blocks)
            
            # 최종 렌더링
            rendered = self.converter(str(pdf_path))
            
            conversion_result = {
                "markdown": rendered.markdown if hasattr(rendered, 'markdown') else "",
                "metadata": rendered.metadata if hasattr(rendered, 'metadata') else {},
                "images": rendered.images if hasattr(rendered, 'images') else {},
                "pages": len(document.pages),
                "success": True
            }
            
            return conversion_result, extracted_blocks
            
        except Exception as e:
            logger.error(f"PDF 처리 중 오류 발생: {pdf_path}, 오류: {str(e)}")
            return {"success": False, "error": str(e)}, []
    
    def _extract_text_from_block(self, block, document) -> str:
        """
        블록에서 텍스트를 추출하는 최적화된 메서드
        
        Args:
            block: Marker 블록 객체
            document: Marker 문서 객체
            
        Returns:
            추출된 텍스트
        """
        # 추출 방법들을 우선순위 순으로 시도
        extraction_methods = [
            # 방법 1: raw_text 메서드 (document 인수 포함)
            lambda: block.raw_text(document) if hasattr(block, 'raw_text') and callable(block.raw_text) else None,
            # 방법 2: raw_text 속성
            lambda: str(block.raw_text) if hasattr(block, 'raw_text') and not callable(block.raw_text) else None,
            # 방법 3: text 속성
            lambda: str(block.text) if hasattr(block, 'text') and block.text else None,
            # 방법 4: content 속성
            lambda: str(block.content) if hasattr(block, 'content') and block.content else None,
        ]
        
        for method in extraction_methods:
            try:
                result = method()
                if result and result.strip():
                    return result.strip()
            except Exception:
                continue  # 다음 방법 시도
        
        return ""  # 모든 방법 실패시 빈 문자열 반환
    
    def _process_blocks(self, blocks: List[Dict]) -> List[Dict]:
        """
        Marker에서 추출된 블록들을 분류하고 정리
        
        Args:
            blocks: Marker에서 추출된 원본 블록 리스트
            
        Returns:
            처리된 블록 리스트
        """
        processed_blocks = []
        
        for block in blocks:
            try:
                block_type = block.get("type", "unknown")
                
                processed_block = {
                    "type": block_type,
                    "content": block.get("text", ""),  # 이제 올바른 텍스트가 들어있음
                    "bbox": block.get("bbox", []),
                    "page": block.get("page", 0),
                    "confidence": block.get("confidence", 1.0),
                    "source": "marker"
                }
                
                # 이미지 블록의 경우 추가 정보 저장
                if block_type == "image":
                    processed_block.update({
                        "image_path": block.get("image_path", ""),
                        "image_size": block.get("image_size", {}),
                        "needs_ocr": True
                    })
                
                # 표 블록의 경우 (Marker가 감지한 경우)
                elif block_type == "table":
                    processed_block.update({
                        "table_data": block.get("table_data", {}),
                        "needs_gmft": True
                    })
                
                processed_blocks.append(processed_block)
                
            except Exception as e:
                logger.warning(f"블록 처리 중 오류: {str(e)}")
                continue
        
        return processed_blocks
    
    def extract_text_blocks(self, blocks: List[Dict]) -> List[Dict]:
        """
        텍스트 블록만 필터링하여 반환
        
        Args:
            blocks: 처리된 블록 리스트
            
        Returns:
            텍스트 블록만 포함된 리스트
        """
        return [block for block in blocks if block["type"] in ["text", "paragraph", "heading"]]
    
    def extract_image_blocks(self, blocks: List[Dict]) -> List[Dict]:
        """
        이미지 블록만 필터링하여 반환
        
        Args:
            blocks: 처리된 블록 리스트
            
        Returns:
            이미지 블록만 포함된 리스트
        """
        image_block_types = ["Picture", "image"]  # Picture는 Marker enum에서 온 것
        return [block for block in blocks if block["type"] in image_block_types]
    
    def get_page_images(self, pdf_path: str, page_num: int) -> List[Dict]:
        """
        특정 페이지의 이미지 정보 추출
        
        Args:
            pdf_path: PDF 파일 경로
            page_num: 페이지 번호 (0부터 시작)
            
        Returns:
            해당 페이지의 이미지 블록 리스트
        """
        try:
            _, blocks = self.extract_blocks_from_pdf(pdf_path)
            page_images = [
                block for block in blocks 
                if block["type"] == "image" and block["page"] == page_num
            ]
            return page_images
        except Exception as e:
            logger.error(f"페이지 이미지 추출 오류: {str(e)}")
            return []


def process_single_pdf(pdf_path: str, config_path: str = "config.yaml") -> Tuple[Dict, List[Dict]]:
    """
    단일 PDF 파일 처리를 위한 편의 함수
    
    Args:
        pdf_path: PDF 파일 경로
        config_path: 설정 파일 경로
        
    Returns:
        Tuple of (conversion_result, blocks)
    """
    runner = MarkerRunner(config_path)
    return runner.extract_blocks_from_pdf(pdf_path)


if __name__ == "__main__":
    # 테스트용 코드
    import sys
    
    if len(sys.argv) > 1:
        pdf_file = sys.argv[1]
        result, blocks = process_single_pdf(pdf_file)
        
        print(f"처리 결과: {'성공' if result.get('success') else '실패'}")
        print(f"추출된 블록 수: {len(blocks)}")
        
        # 블록 타입별 통계
        type_counts = {}
        for block in blocks:
            block_type = block["type"]
            type_counts[block_type] = type_counts.get(block_type, 0) + 1
        
        print("블록 타입별 통계:")
        for block_type, count in type_counts.items():
            print(f"  {block_type}: {count}개")
    else:
        print("사용법: python marker_runner.py <pdf_file_path>") 