"""
블록 통합 및 마크다운/JSON 변환 모듈
텍스트, 이미지 OCR, 표 결과를 마크다운 기반 청크로 통합
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from .config_loader import get_config

logger = logging.getLogger(__name__)


class ContentFormatter:
    """컨텐츠 포맷팅 및 블록 통합 클래스"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        ContentFormatter 초기화
        
        Args:
            config_path: 설정 파일 경로
        """
        self.config = get_config(config_path)
    
    def format_text_block(self, block: Dict[str, Any]) -> Dict[str, Any]:
        """
        텍스트 블록을 마크다운 형태로 포맷팅
        
        Args:
            block: 텍스트 블록 데이터
            
        Returns:
            포맷팅된 블록 딕셔너리
        """
        content = block.get('content', '').strip()
        block_type = block.get('type', 'text')
        
        # 블록 타입에 따른 마크다운 포맷팅
        if block_type == 'heading':
            # 제목 블록
            markdown = f"## {content}\n"
        elif block_type == 'paragraph':
            # 단락 블록
            markdown = f"{content}\n\n"
        else:
            # 일반 텍스트 블록
            markdown = f"{content}\n\n"
        
        return {
            "type": "text",
            "markdown": markdown,
            "source": {
                "page": block.get('page', 0),
                "bbox": block.get('bbox', []),
                "confidence": block.get('confidence', 1.0),
                "original_type": block_type,
                "processor": block.get('source', 'marker')
            },
            "metadata": {
                "content_length": len(content),
                "word_count": len(content.split()) if content else 0
            }
        }
    
    def format_image_ocr_block(self, block: Dict[str, Any]) -> Dict[str, Any]:
        """
        이미지 OCR 결과를 마크다운 형태로 포맷팅
        
        Args:
            block: 이미지 OCR 블록 데이터
            
        Returns:
            포맷팅된 블록 딕셔너리
        """
        ocr_result = block.get('ocr_result', {})
        ocr_text = ocr_result.get('text', '').strip()
        image_path = block.get('cropped_image_path', '')
        confidence = ocr_result.get('confidence', 0)
        
        # 마크다운 형태로 포맷팅
        markdown_parts = []
        
        # 이미지 정보
        if image_path:
            markdown_parts.append(f"![이미지]({image_path})")
        
        # OCR 텍스트
        if ocr_text:
            markdown_parts.append("**이미지에서 추출된 텍스트:**")
            markdown_parts.append(f"> {ocr_text}")
        else:
            markdown_parts.append("*이미지에서 텍스트를 추출할 수 없었습니다.*")
        
        # 신뢰도 정보 (낮은 경우만 표시)
        if confidence < 70:
            markdown_parts.append(f"*OCR 신뢰도: {confidence:.1f}%*")
        
        markdown = "\n\n".join(markdown_parts) + "\n\n"
        
        return {
            "type": "image_ocr",
            "markdown": markdown,
            "source": {
                "page": block.get('page', 0),
                "bbox": block.get('bbox', []),
                "confidence": confidence,
                "image_path": image_path,
                "processor": "ocr"
            },
            "metadata": {
                "ocr_text_length": len(ocr_text),
                "ocr_word_count": len(ocr_text.split()) if ocr_text else 0,
                "ocr_confidence": confidence,
                "ocr_language": ocr_result.get('language', 'unknown')
            }
        }
    
    def format_table_block(self, block: Dict[str, Any]) -> Dict[str, Any]:
        """
        표 블록을 마크다운 형태로 포맷팅
        
        Args:
            block: 표 블록 데이터
            
        Returns:
            포맷팅된 블록 딕셔너리
        """
        table_markdown = block.get('markdown', '')
        table_id = block.get('table_id', 'unknown')
        dimensions = block.get('dimensions', {})
        confidence = block.get('confidence', 1.0)
        
        # 마크다운 형태로 포맷팅
        markdown_parts = []
        
        # 표 제목
        markdown_parts.append(f"### 표 {table_id}")
        
        # 표 정보
        if dimensions.get('rows', 0) > 0 and dimensions.get('columns', 0) > 0:
            markdown_parts.append(f"*크기: {dimensions['rows']}행 × {dimensions['columns']}열*")
        
        # 표 내용
        if table_markdown:
            markdown_parts.append(table_markdown)
        else:
            # 원본 텍스트 사용
            raw_text = block.get('content', '')
            if raw_text:
                markdown_parts.append("```")
                markdown_parts.append(raw_text)
                markdown_parts.append("```")
            else:
                markdown_parts.append("*표 내용을 추출할 수 없었습니다.*")
        
        # 신뢰도 정보 (낮은 경우만 표시)
        if confidence < 80:
            markdown_parts.append(f"*표 인식 신뢰도: {confidence:.1f}%*")
        
        markdown = "\n\n".join(markdown_parts) + "\n\n"
        
        return {
            "type": "table",
            "markdown": markdown,
            "source": {
                "page": block.get('page', 0),
                "bbox": block.get('bbox', []),
                "confidence": confidence,
                "table_id": table_id,
                "processor": "gmft"
            },
            "metadata": {
                "table_dimensions": dimensions,
                "table_confidence": confidence,
                "has_structured_data": bool(block.get('json_structure'))
            },
            "structured_data": block.get('json_structure', {})
        }
    
    def integrate_blocks(self, text_blocks: List[Dict], image_blocks: List[Dict], 
                        table_blocks: List[Dict]) -> List[Dict[str, Any]]:
        """
        모든 블록을 통합하여 페이지 순서대로 정렬
        
        Args:
            text_blocks: 텍스트 블록 리스트
            image_blocks: 이미지 OCR 블록 리스트  
            table_blocks: 표 블록 리스트
            
        Returns:
            통합되고 정렬된 블록 리스트
        """
        all_formatted_blocks = []
        
        # 텍스트 블록 포맷팅
        for block in text_blocks:
            try:
                formatted_block = self.format_text_block(block)
                all_formatted_blocks.append(formatted_block)
            except Exception as e:
                logger.warning(f"텍스트 블록 포맷팅 오류: {str(e)}")
                continue
        
        # 이미지 OCR 블록 포맷팅
        for block in image_blocks:
            try:
                formatted_block = self.format_image_ocr_block(block)
                all_formatted_blocks.append(formatted_block)
            except Exception as e:
                logger.warning(f"이미지 OCR 블록 포맷팅 오류: {str(e)}")
                continue
        
        # 표 블록 포맷팅
        for block in table_blocks:
            try:
                formatted_block = self.format_table_block(block)
                all_formatted_blocks.append(formatted_block)
            except Exception as e:
                logger.warning(f"표 블록 포맷팅 오류: {str(e)}")
                continue
        
        # 페이지와 bbox 좌표를 기준으로 정렬
        sorted_blocks = self._sort_blocks_by_position(all_formatted_blocks)
        
        logger.info(f"총 {len(sorted_blocks)}개 블록이 통합되고 정렬되었습니다")
        return sorted_blocks
    
    def _sort_blocks_by_position(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        블록들을 페이지와 위치를 기준으로 정렬
        
        Args:
            blocks: 정렬할 블록 리스트
            
        Returns:
            정렬된 블록 리스트
        """
        def get_sort_key(block):
            try:
                source = block.get('source', {})
                
                # 페이지 번호 안전하게 추출
                page = source.get('page', 0)
                if not isinstance(page, (int, float)):
                    try:
                        page = int(page) if page is not None else 0
                    except (ValueError, TypeError):
                        page = 0
                
                # bbox 안전하게 추출
                bbox = source.get('bbox', [0, 0, 0, 0])
                
                # bbox를 안전하게 리스트로 변환
                if not isinstance(bbox, list):
                    if hasattr(bbox, 'bbox'):
                        try:
                            bbox = list(bbox.bbox)
                        except Exception:
                            bbox = [0, 0, 0, 0]
                    elif hasattr(bbox, 'polygon') and bbox.polygon:
                        try:
                            coords = bbox.polygon
                            if len(coords) >= 2 and len(coords[0]) >= 2 and len(coords[1]) >= 2:
                                bbox = [coords[0][0], coords[0][1], coords[1][0], coords[1][1]]
                            else:
                                bbox = [0, 0, 0, 0]
                        except Exception:
                            bbox = [0, 0, 0, 0]
                    else:
                        bbox = [0, 0, 0, 0]
                
                # y 좌표 안전하게 추출
                if len(bbox) >= 2:
                    try:
                        y_position = float(bbox[1])
                    except (ValueError, TypeError):
                        y_position = 0.0
                else:
                    y_position = 0.0
                
                return (int(page), float(y_position))
                
            except Exception as e:
                logger.debug(f"정렬 키 생성 오류: {str(e)}, 기본값 사용")
                return (0, 0.0)
        
        try:
            return sorted(blocks, key=get_sort_key)
        except Exception as e:
            logger.warning(f"블록 정렬 오류: {str(e)}, 원본 순서 유지")
            return blocks
    
    def create_document_markdown(self, blocks: List[Dict[str, Any]], 
                               pdf_path: str) -> str:
        """
        전체 문서의 마크다운 생성
        
        Args:
            blocks: 통합된 블록 리스트
            pdf_path: 원본 PDF 경로
            
        Returns:
            전체 문서 마크다운
        """
        markdown_parts = []
        
        # 문서 헤더
        pdf_name = pdf_path.split('/')[-1] if '/' in pdf_path else pdf_path
        markdown_parts.append(f"# {pdf_name}")
        markdown_parts.append(f"*원본 파일: {pdf_path}*")
        markdown_parts.append(f"*처리 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        markdown_parts.append("---\n")
        
        # 블록별 마크다운 추가
        for i, block in enumerate(blocks):
            markdown = block.get('markdown', '')
            if markdown.strip():
                markdown_parts.append(markdown)
        
        return "\n".join(markdown_parts)
    
    def create_structured_json(self, blocks: List[Dict[str, Any]], 
                             pdf_path: str, processing_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        구조화된 JSON 문서 생성
        
        Args:
            blocks: 통합된 블록 리스트
            pdf_path: 원본 PDF 경로
            processing_info: 처리 정보
            
        Returns:
            구조화된 JSON 문서
        """
        # 블록 통계 계산
        block_stats = self._calculate_block_stats(blocks)
        
        # 전체 마크다운 생성
        full_markdown = self.create_document_markdown(blocks, pdf_path)
        
        return {
            "pdf_path": pdf_path,
            "processing_info": processing_info,
            "document_stats": block_stats,
            "full_markdown": full_markdown,
            "blocks": blocks,
            "generated_at": datetime.now().isoformat()
        }
    
    def _calculate_block_stats(self, blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        블록 통계 계산
        
        Args:
            blocks: 블록 리스트
            
        Returns:
            통계 정보 딕셔너리
        """
        stats = {
            "total_blocks": len(blocks),
            "text_blocks": 0,
            "image_ocr_blocks": 0,
            "table_blocks": 0,
            "total_words": 0,
            "total_characters": 0,
            "pages_processed": set()
        }
        
        for block in blocks:
            block_type = block.get('type', 'unknown')
            
            # 블록 타입별 카운트
            if block_type == 'text':
                stats["text_blocks"] += 1
            elif block_type == 'image_ocr':
                stats["image_ocr_blocks"] += 1
            elif block_type == 'table':
                stats["table_blocks"] += 1
            
            # 단어 및 문자 수 계산
            metadata = block.get('metadata', {})
            stats["total_words"] += metadata.get('word_count', 0)
            stats["total_characters"] += metadata.get('content_length', 0)
            
            # 페이지 정보 수집
            source = block.get('source', {})
            page = source.get('page')
            if page is not None:
                stats["pages_processed"].add(page)
        
        # 페이지 수를 리스트로 변환
        stats["pages_processed"] = sorted(list(stats["pages_processed"]))
        stats["total_pages"] = len(stats["pages_processed"])
        
        return stats
    
    def process_and_format(self, marker_result: Dict[str, Any], 
                          image_blocks: List[Dict], table_blocks: List[Dict],
                          pdf_path: str) -> Dict[str, Any]:
        """
        모든 결과를 처리하고 포맷팅하는 메인 함수
        
        Args:
            marker_result: Marker 처리 결과
            image_blocks: 이미지 OCR 블록 리스트
            table_blocks: 표 블록 리스트
            pdf_path: 원본 PDF 경로
            
        Returns:
            최종 포맷팅된 결과
        """
        try:
            # Marker에서 텍스트 블록 추출
            all_marker_blocks = marker_result.get('blocks', [])
            logger.info(f"Marker에서 받은 전체 블록 수: {len(all_marker_blocks)}")
            
            if all_marker_blocks:
                # 블록 타입 통계 출력
                type_counts = {}
                for block in all_marker_blocks:
                    block_type = block.get('type', 'None')
                    type_counts[block_type] = type_counts.get(block_type, 0) + 1
                logger.info(f"블록 타입 통계: {dict(sorted(type_counts.items()))}")
            
            # 실제 Marker enum 값들과 매칭 (enum에서 추출된 문자열)
            text_block_types = ['Text', 'Line', 'Span', 'SectionHeader', 'ListItem', 'Footnote', 'PageFooter', 'TextInlineMath', 'ListGroup']
            text_blocks = [block for block in all_marker_blocks 
                         if block.get('type') in text_block_types]
            logger.info(f"필터링된 텍스트 블록 수: {len(text_blocks)}")
            
            # 모든 블록 통합
            integrated_blocks = self.integrate_blocks(text_blocks, image_blocks, table_blocks)
            
            # 처리 정보 생성
            processing_info = {
                "marker_success": marker_result.get('success', False),
                "total_pages": marker_result.get('pages', 0),
                "processing_timestamp": datetime.now().isoformat(),
                "blocks_summary": {
                    "text_blocks": len(text_blocks),
                    "image_ocr_blocks": len(image_blocks),
                    "table_blocks": len(table_blocks),
                    "total_integrated": len(integrated_blocks)
                }
            }
            
            # 최종 JSON 문서 생성
            final_result = self.create_structured_json(
                integrated_blocks, pdf_path, processing_info
            )
            
            logger.info(f"문서 포맷팅 완료: {pdf_path}")
            return final_result
            
        except Exception as e:
            logger.error(f"문서 포맷팅 오류: {pdf_path}, 오류: {str(e)}")
            return {
                "pdf_path": pdf_path,
                "error": str(e),
                "success": False,
                "processing_info": {"error": str(e)}
            }


def format_document(marker_result: Dict[str, Any], image_blocks: List[Dict], 
                   table_blocks: List[Dict], pdf_path: str, 
                   config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    문서 포맷팅을 위한 편의 함수
    
    Args:
        marker_result: Marker 처리 결과
        image_blocks: 이미지 OCR 블록 리스트
        table_blocks: 표 블록 리스트
        pdf_path: 원본 PDF 경로
        config_path: 설정 파일 경로
        
    Returns:
        포맷팅된 문서 결과
    """
    formatter = ContentFormatter(config_path)
    return formatter.process_and_format(marker_result, image_blocks, table_blocks, pdf_path)


if __name__ == "__main__":
    # 테스트용 코드
    formatter = ContentFormatter()
    
    # 샘플 블록들
    sample_text_blocks = [
        {"type": "text", "content": "이것은 테스트 텍스트입니다.", "page": 0, "bbox": [0, 0, 100, 20]}
    ]
    
    sample_image_blocks = [
        {
            "type": "image",
            "page": 0,
            "bbox": [0, 30, 100, 80],
            "ocr_result": {"text": "이미지 속 텍스트", "confidence": 85.5}
        }
    ]
    
    sample_table_blocks = [
        {
            "type": "table",
            "page": 0,
            "bbox": [0, 90, 100, 150],
            "markdown": "| 헤더1 | 헤더2 |\n|-------|-------|\n| 값1 | 값2 |",
            "table_id": "table_0",
            "dimensions": {"rows": 2, "columns": 2}
        }
    ]
    
    # 블록 통합 테스트
    integrated = formatter.integrate_blocks(
        sample_text_blocks, sample_image_blocks, sample_table_blocks
    )
    
    print(f"통합된 블록 수: {len(integrated)}")
    for i, block in enumerate(integrated):
        print(f"블록 {i+1}: {block['type']}") 