"""
PDF 분할 기반 처리 모듈
PDF를 적당한 크기로 분할하여 독립적으로 처리한 후 병합
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import fitz  # PyMuPDF
import json
from datetime import datetime
from .config_loader import get_section, safe_page_number, ensure_directories

logger = logging.getLogger(__name__)


class ChunkProcessor:
    """PDF 분할 기반 처리 클래스"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        ChunkProcessor 초기화
        
        Args:
            config_path: 설정 파일 경로
        """
        self.chunk_config = get_section('pdf_split_processing', config_path)
        self.chunk_size = self.chunk_config.get('pages_per_split', 15)  # 기본 15페이지
        self.temp_dir = Path(self.chunk_config.get('temp_directory', 'temp_splits'))
        
        # 임시 디렉토리 생성
        ensure_directories([self.temp_dir])
        
        logger.info(f"PDF 분할 처리기 초기화: {self.chunk_size}페이지 단위")
    
    def analyze_pdf_structure(self, pdf_path: str) -> Dict[str, Any]:
        """
        PDF 구조 분석 및 청크 계획 수립
        
        Args:
            pdf_path: PDF 파일 경로
            
        Returns:
            PDF 구조 정보 및 청크 계획
        """
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            
            # 적응적 청크 크기 결정
            if total_pages <= 10:
                chunk_size = total_pages  # 작은 파일은 한번에
            elif total_pages <= 50:
                chunk_size = 10  # 중간 파일은 10페이지씩
            elif total_pages <= 200:
                chunk_size = 15  # 큰 파일은 15페이지씩
            else:
                chunk_size = 20  # 매우 큰 파일은 20페이지씩
            
            # PDF 분할 계획 생성
            chunks = []
            for start_page in range(0, total_pages, chunk_size):
                end_page = min(start_page + chunk_size - 1, total_pages - 1)
                chunks.append({
                    'chunk_id': len(chunks),
                    'start_page': start_page,
                    'end_page': end_page,
                    'page_count': end_page - start_page + 1
                })
            
            doc.close()
            
            structure = {
                'pdf_path': pdf_path,
                'total_pages': total_pages,
                'chunk_size': chunk_size,
                'total_chunks': len(chunks),
                'chunks': chunks,
                'analysis_time': datetime.now().isoformat()
            }
            
            logger.info(f"PDF 구조 분석 완료: {total_pages}페이지 -> {len(chunks)}개 분할")
            return structure
            
        except Exception as e:
            logger.error(f"PDF 구조 분석 실패: {str(e)}")
            return {}
    
    def create_chunk_pdf(self, pdf_path: str, chunk_info: Dict[str, Any]) -> str:
        """
        PDF 분할별 임시 PDF 파일 생성
        
        Args:
            pdf_path: 원본 PDF 경로
            chunk_info: 분할 정보
            
        Returns:
            생성된 분할 PDF 경로
        """
        try:
            doc = fitz.open(pdf_path)
            chunk_doc = fitz.open()  # 새 문서 생성
            
            # 지정된 페이지 범위 복사
            start_page = chunk_info['start_page']
            end_page = chunk_info['end_page']
            
            for page_num in range(start_page, end_page + 1):
                chunk_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
            
            # 분할 파일 저장
            pdf_name = Path(pdf_path).stem
            chunk_filename = f"{pdf_name}_chunk_{chunk_info['chunk_id']:03d}_{start_page}-{end_page}.pdf"
            chunk_path = self.temp_dir / chunk_filename
            
            chunk_doc.save(str(chunk_path))
            chunk_doc.close()
            doc.close()
            
            logger.info(f"PDF 분할 생성: {chunk_filename} (페이지 {start_page}-{end_page})")
            return str(chunk_path)
            
        except Exception as e:
            logger.error(f"청크 PDF 생성 실패: {str(e)}")
            return ""
    
    def process_chunk(self, chunk_pdf_path: str, chunk_info: Dict[str, Any], 
                     original_pdf_path: str) -> Dict[str, Any]:
        """
        개별 PDF 분할 처리
        
        Args:
            chunk_pdf_path: 분할 PDF 경로
            chunk_info: 분할 정보
            original_pdf_path: 원본 PDF 경로
            
        Returns:
            분할 처리 결과
        """
        try:
            from utils.marker_runner import MarkerRunner
            from utils.ocr_image_blocks import OCRImageProcessor
            from utils.gmft_runner import GMFTTableProcessor
            from utils.formatter import ContentFormatter
            
            logger.info(f"PDF 분할 {chunk_info['chunk_id']} 처리 시작")
            
            # 각 처리기 초기화
            marker = MarkerRunner()
            ocr_processor = OCRImageProcessor()
            gmft_processor = GMFTTableProcessor()
            formatter = ContentFormatter()
            
            # 1. Marker 처리
            marker_result, marker_blocks = marker.extract_blocks_from_pdf(chunk_pdf_path)
            
            if not marker_result.get('success', False):
                raise Exception(f"Marker 처리 실패: {marker_result.get('error', 'Unknown error')}")
            
            # 2. 이미지 OCR 처리
            image_blocks = marker.extract_image_blocks(marker_blocks)
            processed_image_blocks = []
            
            if image_blocks:
                processed_image_blocks = ocr_processor.process_image_blocks(
                    chunk_pdf_path, image_blocks
                )
            
            # 3. GMFT 표 처리
            gmft_result = gmft_processor.extract_tables_from_pdf(chunk_pdf_path)
            table_blocks = gmft_processor.process_tables_for_blocks(gmft_result)
            
            # 4. 포맷팅 (청크별로 독립적)
            marker_result['blocks'] = marker_blocks
            chunk_result = formatter.process_and_format(
                marker_result, processed_image_blocks, table_blocks, chunk_pdf_path
            )
            
            # 청크 메타데이터 추가
            chunk_result.update({
                'chunk_info': chunk_info,
                'original_pdf': original_pdf_path,
                'chunk_pdf': chunk_pdf_path,
                'processing_success': True
            })
            
            logger.info(f"PDF 분할 {chunk_info['chunk_id']} 처리 완료")
            return chunk_result
            
        except Exception as e:
            logger.error(f"PDF 분할 {chunk_info['chunk_id']} 처리 실패: {str(e)}")
            return {
                'chunk_info': chunk_info,
                'original_pdf': original_pdf_path,
                'chunk_pdf': chunk_pdf_path,
                'processing_success': False,
                'error': str(e)
            }
    
    def merge_chunks(self, chunk_results: List[Dict[str, Any]], 
                    original_pdf_path: str) -> Dict[str, Any]:
        """
        PDF 분할 결과들을 병합하여 최종 문서 생성
        
        Args:
            chunk_results: 분할 처리 결과 리스트
            original_pdf_path: 원본 PDF 경로
            
        Returns:
            병합된 최종 결과
        """
        try:
            logger.info(f"PDF 분할 병합 시작: {len(chunk_results)}개 분할")
            
            # 성공한 분할들만 필터링
            successful_chunks = [r for r in chunk_results if r.get('processing_success', False)]
            
            if not successful_chunks:
                raise Exception("성공한 분할이 없습니다")
            
            # 분할 순서대로 정렬
            successful_chunks.sort(key=lambda x: x['chunk_info']['chunk_id'])
            
            # 통합 결과 생성
            merged_result = {
                'pdf_path': original_pdf_path,
                'processing_info': {
                    'pdf_split_based_processing': True,
                    'total_splits': len(chunk_results),
                    'successful_splits': len(successful_chunks),
                    'failed_splits': len(chunk_results) - len(successful_chunks),
                    'processing_timestamp': datetime.now().isoformat()
                },
                'document_stats': {},
                'full_markdown': '',
                'blocks': [],
                'generated_at': datetime.now().isoformat()
            }
            
            # 청크별 데이터 병합
            total_blocks = 0
            total_words = 0
            total_characters = 0
            all_pages = set()
            markdown_parts = []
            
            # 문서 헤더 생성
            pdf_name = Path(original_pdf_path).name
            markdown_parts.append(f"# {pdf_name}")
            markdown_parts.append(f"*원본 파일: {original_pdf_path}*")
            markdown_parts.append(f"*처리 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
            markdown_parts.append(f"*PDF 분할 기반 처리: {len(successful_chunks)}개 분할*")
            markdown_parts.append("---\n")
            
            for chunk_result in successful_chunks:
                chunk_id = chunk_result['chunk_info']['chunk_id']
                start_page = chunk_result['chunk_info']['start_page']
                end_page = chunk_result['chunk_info']['end_page']
                
                # PDF 분할 구분선 추가
                markdown_parts.append(f"\n<!-- PDF 분할 {chunk_id}: 페이지 {start_page+1}-{end_page+1} -->")
                
                # 마크다운 병합
                chunk_markdown = chunk_result.get('full_markdown', '')
                if chunk_markdown:
                    # 분할의 헤더 제거 (첫 번째 --- 이후부터 사용)
                    if '---' in chunk_markdown:
                        chunk_content = chunk_markdown.split('---', 1)[1].strip()
                        markdown_parts.append(chunk_content)
                
                # 블록 병합 (페이지 번호 조정)
                chunk_blocks = chunk_result.get('blocks', [])
                for block in chunk_blocks:
                    # 페이지 번호를 원본 PDF 기준으로 조정
                    if 'source' in block and 'page' in block['source']:
                        current_page = block['source']['page']
                        block['source']['page'] = safe_page_number(current_page, start_page)
                    merged_result['blocks'].append(block)
                
                # 통계 누적
                chunk_stats = chunk_result.get('document_stats', {})
                total_blocks += chunk_stats.get('total_blocks', 0)
                total_words += chunk_stats.get('total_words', 0)
                total_characters += chunk_stats.get('total_characters', 0)
                
                # 페이지 정보 누적
                chunk_pages = chunk_stats.get('pages_processed', [])
                for page in chunk_pages:
                    all_pages.add(safe_page_number(page, start_page))
            
            # 최종 마크다운 생성
            merged_result['full_markdown'] = '\n'.join(markdown_parts)
            
            # 최종 통계 생성
            merged_result['document_stats'] = {
                'total_blocks': total_blocks,
                'text_blocks': len([b for b in merged_result['blocks'] if b.get('type') == 'text']),
                'image_ocr_blocks': len([b for b in merged_result['blocks'] if b.get('type') == 'image_ocr']),
                'table_blocks': len([b for b in merged_result['blocks'] if b.get('type') == 'table']),
                'total_words': total_words,
                'total_characters': total_characters,
                'pages_processed': sorted(list(all_pages)),
                'total_pages': len(all_pages)
            }
            
            logger.info(f"PDF 분할 병합 완료: {total_blocks}개 블록, {len(all_pages)}페이지")
            return merged_result
            
        except Exception as e:
            logger.error(f"PDF 분할 병합 실패: {str(e)}")
            return {
                'pdf_path': original_pdf_path,
                'error': str(e),
                'success': False,
                'processing_info': {'error': str(e)}
            }
    
    def cleanup_temp_files(self, keep_chunks: bool = False):
        """
        임시 파일 정리
        
        Args:
            keep_chunks: 분할 파일 보관 여부
        """
        try:
            if not keep_chunks and self.temp_dir.exists():
                for chunk_file in self.temp_dir.glob("*.pdf"):
                    chunk_file.unlink()
                logger.info("임시 PDF 분할 파일들이 정리되었습니다")
        except Exception as e:
            logger.warning(f"임시 파일 정리 중 오류: {str(e)}")
    
    def process_pdf_in_chunks(self, pdf_path: str, keep_temp_files: bool = False) -> Dict[str, Any]:
        """
        PDF를 분할 단위로 처리하는 메인 함수
        
        Args:
            pdf_path: 처리할 PDF 파일 경로
            keep_temp_files: 임시 파일 보관 여부
            
        Returns:
            최종 처리 결과
        """
        try:
            logger.info(f"PDF 분할 기반 처리 시작: {pdf_path}")
            
            # 1. PDF 구조 분석
            structure = self.analyze_pdf_structure(pdf_path)
            if not structure:
                raise Exception("PDF 구조 분석 실패")
            
            # 2. PDF 분할별 처리
            chunk_results = []
            for chunk_info in structure['chunks']:
                # 분할 PDF 생성
                chunk_pdf_path = self.create_chunk_pdf(pdf_path, chunk_info)
                if not chunk_pdf_path:
                    continue
                
                # 분할 처리
                chunk_result = self.process_chunk(chunk_pdf_path, chunk_info, pdf_path)
                chunk_results.append(chunk_result)
            
            # 3. PDF 분할 병합
            final_result = self.merge_chunks(chunk_results, pdf_path)
            
            # 4. 임시 파일 정리
            if not keep_temp_files:
                self.cleanup_temp_files()
            
            logger.info(f"PDF 분할 기반 처리 완료: {pdf_path}")
            return final_result
            
        except Exception as e:
            logger.error(f"PDF 분할 기반 처리 실패: {pdf_path}, 오류: {str(e)}")
            return {
                'pdf_path': pdf_path,
                'error': str(e),
                'success': False,
                'processing_info': {'error': str(e)}
            } 