"""
GMFT 기반 PDF 표 분석 모듈
PyPI gmft 모듈을 사용하여 PDF 전체에서 표를 감지하고 구조화
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from .config_loader import get_config, get_section

import gmft

logger = logging.getLogger(__name__)


def apply_pymupdf_compatibility_patch():
    """PyMuPDF와 GMFT 간의 호환성 문제를 해결하는 패치"""
    try:
        import fitz
        
        # get_image 메서드가 없는 경우에만 패치 적용
        if not hasattr(fitz.Page, 'get_image'):
            def page_get_image(self, *args, **kwargs):
                """PyMuPDF 호환성을 위한 get_image 메서드"""
                return self.get_pixmap(*args, **kwargs)
            fitz.Page.get_image = page_get_image
        
        if not hasattr(fitz.Document, 'get_image'):
            def doc_get_image(self, *args, **kwargs):
                """PyMuPDF Document 호환성을 위한 get_image 메서드"""
                if len(self) > 0:
                    return self[0].get_pixmap(*args, **kwargs)
                else:
                    raise ValueError("Document has no pages")
            fitz.Document.get_image = doc_get_image
        
        logger.debug("PyMuPDF 호환성 패치 적용 완료")
        return True
        
    except Exception as e:
        logger.warning(f"PyMuPDF 호환성 패치 실패: {e}")
        return False


class GMFTTableProcessor:
    """GMFT 기반 표 분석 처리 클래스"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        GMFTTableProcessor 초기화
        
        Args:
            config_path: 설정 파일 경로
        """
        self.gmft_config = get_section('gmft', config_path)
        self.detect_tables = self.gmft_config.get('detect_tables', True)
        self.output_formats = self.gmft_config.get('output_formats', ['markdown', 'json'])
        
        # GMFT 사용 가능 여부 확인
        self.gmft_available = self._initialize_gmft()
    
    def _initialize_gmft(self) -> bool:
        """GMFT 라이브러리 초기화 및 사용 가능 여부 확인"""
        if not self.detect_tables:
            logger.info("GMFT 테이블 감지가 비활성화됨")
            return False
        
        try:
            # PyMuPDF 호환성 패치 적용
            apply_pymupdf_compatibility_patch()
            
            # GMFT 사용 가능성 테스트
            test_detector = gmft.AutoTableDetector()
            logger.info("GMFT 라이브러리 초기화 완료")
            return True
            
        except Exception as e:
            logger.warning(f"GMFT 라이브러리 사용 불가: {e}")
            return False
        
    def extract_tables_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        PDF에서 모든 표를 감지하고 추출
        
        Args:
            pdf_path: PDF 파일 경로
            
        Returns:
            표 추출 결과 딕셔너리
        """
        try:
            # GMFT 사용 가능 여부 먼저 확인
            if not self.gmft_available:
                logger.info(f"GMFT 비활성화됨, 빈 결과 반환: {pdf_path}")
                return self._create_empty_result()
            
            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
            
            logger.info(f"GMFT 표 분석 시작: {pdf_path}")
            
            # GMFT AutoTableDetector로 PDF 처리
            doc = None
            try:
                import fitz
                
                # PyMuPDF Document 객체 생성
                doc = fitz.open(str(pdf_path))
                
                # GMFT 처리
                detector = gmft.AutoTableDetector()
                result = detector.extract(doc)
                
                if not result:
                    logger.warning(f"GMFT 결과가 없습니다: {pdf_path}")
                    return self._create_empty_result()
                
                # 결과 처리
                processed_result = self._process_gmft_result(result, str(pdf_path))
                
            except AttributeError as ae:
                # 'str' object has no attribute 'get_image' 등의 오류 처리
                logger.warning(f"GMFT 라이브러리 호환성 오류: {pdf_path}, {ae}")
                return self._create_empty_result()
            except Exception as gmft_error:
                # 기타 GMFT 관련 오류
                logger.warning(f"GMFT 내부 오류: {pdf_path}, {str(gmft_error)}")
                import traceback
                logger.debug(f"GMFT 상세 오류: {traceback.format_exc()}")
                return self._create_empty_result()
            finally:
                # Document 객체 정리
                if doc is not None:
                    try:
                        doc.close()
                    except Exception:
                        pass
            
            logger.info(f"GMFT 표 분석 완료: {len(processed_result.get('tables', []))}개 표 발견")
            return processed_result
            
        except Exception as e:
            logger.error(f"GMFT 처리 중 오류 발생: {pdf_path}, 오류: {str(e)}")
            import traceback
            logger.debug(f"GMFT 상세 오류: {traceback.format_exc()}")
            return self._create_error_result(str(e))
    
    def _process_gmft_result(self, gmft_result: Any, pdf_path: str) -> Dict[str, Any]:
        """
        GMFT 결과를 처리하여 구조화된 형태로 변환
        
        Args:
            gmft_result: GMFT auto() 함수의 결과
            pdf_path: PDF 파일 경로
            
        Returns:
            처리된 결과 딕셔너리
        """
        try:
            tables = []
            
            # GMFT 결과가 리스트인지 단일 객체인지 확인
            if isinstance(gmft_result, list):
                table_results = gmft_result
            else:
                table_results = [gmft_result] if gmft_result else []
            
            for i, table_result in enumerate(table_results):
                try:
                    # table_result가 유효한 객체인지 확인
                    if table_result is None or isinstance(table_result, str):
                        logger.warning(f"표 {i}: 유효하지 않은 결과 타입 - {type(table_result)}")
                        continue
                        
                    table_info = self._extract_table_info(table_result, i)
                    if table_info:
                        tables.append(table_info)
                except AttributeError as ae:
                    logger.warning(f"표 {i} 속성 오류: {str(ae)}")
                    continue
                except Exception as e:
                    logger.warning(f"표 {i} 처리 중 오류: {str(e)}")
                    continue
            
            return {
                "pdf_path": pdf_path,
                "total_tables": len(tables),
                "tables": tables,
                "success": True,
                "processor": "gmft"
            }
            
        except Exception as e:
            logger.error(f"GMFT 결과 처리 오류: {str(e)}")
            return self._create_error_result(str(e))
    
    def _extract_table_info(self, table_result: Any, table_index: int) -> Optional[Dict[str, Any]]:
        """
        개별 표 정보 추출 및 구조화
        
        Args:
            table_result: GMFT가 감지한 단일 표 결과
            table_index: 표 인덱스
            
        Returns:
            구조화된 표 정보 딕셔너리
        """
        try:
            # 안전한 속성 접근을 위한 헬퍼 함수
            def safe_getattr(obj, attr, default):
                try:
                    return getattr(obj, attr, default)
                except (AttributeError, TypeError):
                    return default
            
            table_info = {
                "table_id": f"table_{table_index}",
                "page": safe_getattr(table_result, 'page', 0),
                "bbox": safe_getattr(table_result, 'bbox', []),
                "confidence": safe_getattr(table_result, 'confidence', 1.0)
            }
            
            # 마크다운 형태 추출
            if 'markdown' in self.output_formats:
                markdown_content = self._extract_markdown(table_result)
                table_info['markdown'] = markdown_content
            
            # JSON 구조 추출
            if 'json' in self.output_formats:
                json_structure = self._extract_json_structure(table_result)
                table_info['json_structure'] = json_structure
            
            # 원본 텍스트 추출
            raw_text = self._extract_raw_text(table_result)
            table_info['raw_text'] = raw_text
            
            # 표 크기 정보
            table_info['dimensions'] = self._get_table_dimensions(table_result)
            
            return table_info
            
        except Exception as e:
            logger.warning(f"표 정보 추출 오류: {str(e)}")
            return None
    
    def _extract_markdown(self, table_result: Any) -> str:
        """
        표를 마크다운 형태로 변환
        
        Args:
            table_result: GMFT 표 결과
            
        Returns:
            마크다운 형태의 표
        """
        try:
            # GMFT 결과에서 마크다운 추출 시도
            if hasattr(table_result, 'to_markdown'):
                return table_result.to_markdown()
            elif hasattr(table_result, 'markdown'):
                return table_result.markdown
            elif hasattr(table_result, 'df'):
                # DataFrame이 있는 경우 마크다운으로 변환
                return table_result.df.to_markdown(index=False)
            else:
                # 기본적인 텍스트 기반 마크다운 생성
                return self._create_basic_markdown(table_result)
                
        except Exception as e:
            logger.warning(f"마크다운 생성 오류: {str(e)}")
            return f"<!-- 마크다운 생성 실패: {str(e)} -->"
    
    def _extract_json_structure(self, table_result: Any) -> Dict[str, Any]:
        """
        표를 JSON 구조로 변환
        
        Args:
            table_result: GMFT 표 결과
            
        Returns:
            JSON 구조의 표 데이터
        """
        try:
            # GMFT 결과에서 구조화된 데이터 추출
            if hasattr(table_result, 'to_dict'):
                return table_result.to_dict()
            elif hasattr(table_result, 'df'):
                # DataFrame을 딕셔너리로 변환
                return table_result.df.to_dict('records')
            elif hasattr(table_result, 'cells'):
                # 셀 기반 구조 추출
                return self._extract_cell_structure(table_result.cells)
            else:
                return {"error": "JSON 구조 추출 불가능"}
                
        except Exception as e:
            logger.warning(f"JSON 구조 생성 오류: {str(e)}")
            return {"error": str(e)}
    
    def _extract_raw_text(self, table_result: Any) -> str:
        """
        표의 원본 텍스트 추출
        
        Args:
            table_result: GMFT 표 결과
            
        Returns:
            원본 텍스트
        """
        try:
            if hasattr(table_result, 'text'):
                return table_result.text
            elif hasattr(table_result, 'content'):
                return str(table_result.content)
            elif hasattr(table_result, 'df'):
                return table_result.df.to_string()
            else:
                return str(table_result)
                
        except Exception as e:
            logger.warning(f"원본 텍스트 추출 오류: {str(e)}")
            return ""
    
    def _get_table_dimensions(self, table_result: Any) -> Dict[str, int]:
        """
        표의 크기 정보 추출
        
        Args:
            table_result: GMFT 표 결과
            
        Returns:
            행/열 수 정보
        """
        try:
            if hasattr(table_result, 'df'):
                df = table_result.df
                return {"rows": len(df), "columns": len(df.columns)}
            elif hasattr(table_result, 'shape'):
                rows, cols = table_result.shape
                return {"rows": rows, "columns": cols}
            else:
                return {"rows": 0, "columns": 0}
                
        except Exception as e:
            logger.warning(f"표 크기 정보 추출 오류: {str(e)}")
            return {"rows": 0, "columns": 0}
    
    def _create_basic_markdown(self, table_result: Any) -> str:
        """
        기본적인 마크다운 표 생성
        
        Args:
            table_result: GMFT 표 결과
            
        Returns:
            기본 마크다운 형태의 표
        """
        try:
            # 텍스트를 기반으로 간단한 마크다운 생성
            text = str(table_result)
            lines = text.split('\n')
            
            if len(lines) < 2:
                return f"| {text} |\n|---|\n"
            
            # 첫 번째 줄을 헤더로 사용
            header = f"| {lines[0]} |"
            separator = "|" + "---|" * lines[0].count('\t') + "---|\n" if '\t' in lines[0] else "|---|\n"
            
            rows = [f"| {line} |" for line in lines[1:] if line.strip()]
            
            return header + "\n" + separator + "\n".join(rows)
            
        except Exception as e:
            logger.warning(f"기본 마크다운 생성 오류: {str(e)}")
            return f"<!-- 표 내용: {str(table_result)} -->"
    
    def _extract_cell_structure(self, cells: Any) -> List[Dict[str, Any]]:
        """
        셀 구조에서 데이터 추출
        
        Args:
            cells: 표의 셀 데이터
            
        Returns:
            셀 구조 리스트
        """
        try:
            cell_data = []
            for cell in cells:
                cell_info = {
                    "text": getattr(cell, 'text', str(cell)),
                    "row": getattr(cell, 'row', 0),
                    "col": getattr(cell, 'col', 0),
                    "bbox": getattr(cell, 'bbox', [])
                }
                cell_data.append(cell_info)
            return cell_data
            
        except Exception as e:
            logger.warning(f"셀 구조 추출 오류: {str(e)}")
            return []
    
    def _create_empty_result(self) -> Dict[str, Any]:
        """
        빈 결과 생성
        
        Returns:
            빈 결과 딕셔너리
        """
        return {
            "pdf_path": "",
            "total_tables": 0,
            "tables": [],
            "success": True,
            "processor": "gmft",
            "message": "표가 감지되지 않음"
        }
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """
        오류 결과 생성
        
        Args:
            error_message: 오류 메시지
            
        Returns:
            오류 결과 딕셔너리
        """
        return {
            "pdf_path": "",
            "total_tables": 0,
            "tables": [],
            "success": False,
            "processor": "gmft",
            "error": error_message
        }
    
    def process_tables_for_blocks(self, tables_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        GMFT 결과를 블록 형태로 변환
        
        Args:
            tables_result: GMFT 처리 결과
            
        Returns:
            블록 형태의 표 리스트
        """
        blocks = []
        
        if not tables_result.get('success', False):
            return blocks
        
        for table in tables_result.get('tables', []):
            block = {
                "type": "table",
                "content": table.get('raw_text', ''),
                "markdown": table.get('markdown', ''),
                "json_structure": table.get('json_structure', {}),
                "bbox": table.get('bbox', []),
                "page": table.get('page', 0),
                "confidence": table.get('confidence', 1.0),
                "source": "gmft",
                "table_id": table.get('table_id', ''),
                "dimensions": table.get('dimensions', {})
            }
            blocks.append(block)
        
        return blocks


def process_pdf_tables(pdf_path: str, config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    PDF의 표들을 GMFT로 처리하는 편의 함수
    
    Args:
        pdf_path: PDF 파일 경로
        config_path: 설정 파일 경로
        
    Returns:
        GMFT 처리 결과
    """
    processor = GMFTTableProcessor(config_path)
    return processor.extract_tables_from_pdf(pdf_path)


if __name__ == "__main__":
    # 테스트용 코드
    import sys
    
    if len(sys.argv) > 1:
        pdf_file = sys.argv[1]
        result = process_pdf_tables(pdf_file)
        
        print(f"GMFT 처리 결과:")
        print(f"  성공: {'예' if result.get('success') else '아니오'}")
        print(f"  감지된 표 수: {result.get('total_tables', 0)}")
        
        if result.get('tables'):
            for i, table in enumerate(result['tables']):
                print(f"  표 {i+1}:")
                print(f"    페이지: {table.get('page', 'N/A')}")
                print(f"    크기: {table.get('dimensions', {})}")
                print(f"    신뢰도: {table.get('confidence', 0):.2f}")
    else:
        print("사용법: python gmft_runner.py <pdf_file_path>") 