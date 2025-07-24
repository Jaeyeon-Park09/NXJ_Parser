"""
RAG용 텍스트 토큰 청크 분할 모듈
OpenAI tiktoken을 사용한 정확한 토큰 기반 분할
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
from .config_loader import get_config, get_section

# BLIP-2 이미지 캡션 생성용 추가 import
try:
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    from PIL import Image
    import torch
    BLIP2_AVAILABLE = True
except ImportError:
    BLIP2_AVAILABLE = False

try:
    import tiktoken
except ImportError:
    tiktoken = None
    print("Warning: tiktoken not installed. Install with: pip install tiktoken")

logger = logging.getLogger(__name__)


class TextChunker:
    """텍스트를 토큰 기반으로 청크 분할하는 클래스"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        TextChunker 초기화
        
        Args:
            config_path: 설정 파일 경로
        """
        self.config = get_config(config_path)
        self.chunk_config = get_section('text_chunking', config_path)
        self.chunk_size = self.chunk_config.get('chunk_size', 500)
        self.overlap = self.chunk_config.get('overlap', 50)
        self.model_name = self.chunk_config.get('model_name', 'gpt-3.5-turbo')
        
        # tiktoken 인코더 초기화
        if tiktoken:
            try:
                self.encoding = tiktoken.encoding_for_model(self.model_name)
            except KeyError:
                logger.warning(f"Model {self.model_name} not found, using cl100k_base")
                self.encoding = tiktoken.get_encoding("cl100k_base")
        else:
            self.encoding = None
            logger.warning("tiktoken not available, using approximate token counting")
        
        # BLIP-2 모델 초기화 (지연 로딩)
        self.blip2_processor = None
        self.blip2_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"텍스트 청크 분할기 초기화: {self.chunk_size} 토큰, {self.overlap} 오버랩")
        if BLIP2_AVAILABLE:
            logger.info(f"BLIP-2 이미지 캡션 생성 가능, 디바이스: {self.device}")
    
    def _load_blip2_model(self):
        """BLIP-2 모델 지연 로딩 (최적화됨)"""
        if not BLIP2_AVAILABLE:
            logger.warning("BLIP-2 라이브러리가 설치되지 않음. 이미지 캡션 생성 불가")
            return False
            
        if self.blip2_processor is None:
            try:
                # 🔥 **성능 최적화**: 더 작은 모델 사용 옵션
                model_name = self.config.get('image_captioning', {}).get('model', 'Salesforce/blip2-opt-2.7b')
                
                logger.info(f"BLIP-2 모델 로딩 중: {model_name}")
                logger.info("⏳ 첫 실행 시 모델 다운로드에 시간이 걸릴 수 있습니다...")
                
                # 프로세서 로드 (캐시 활용)
                self.blip2_processor = Blip2Processor.from_pretrained(
                    model_name,
                    cache_dir=self.config.get('image_captioning', {}).get('cache_dir', None)
                )
                
                # 모델 로드 (메모리 최적화)
                self.blip2_model = Blip2ForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    cache_dir=self.config.get('image_captioning', {}).get('cache_dir', None)
                )
                
                if self.device == "cpu":
                    logger.warning("🐌 CPU에서 BLIP-2 실행 중 - 처리 속도가 매우 느릴 수 있습니다")
                else:
                    logger.info(f"🚀 GPU에서 BLIP-2 실행: {self.device}")
                
                self.blip2_model.to(self.device)
                
                # 메모리 사용량 체크
                if self.device == "cuda":
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    logger.info(f"GPU 메모리 사용량: {allocated:.1f}GB")
                
                logger.info("✅ BLIP-2 모델 로딩 완료")
                return True
                
            except Exception as e:
                logger.error(f"❌ BLIP-2 모델 로딩 실패: {e}")
                logger.info("💡 해결책: GPU 메모리 부족 시 CPU 사용 또는 더 작은 모델 사용")
                return False
        return True
    
    def _generate_image_caption(self, image_path: str) -> Dict[str, Any]:
        """
        BLIP-2를 사용한 이미지 캡션 생성
        
        Args:
            image_path: 이미지 파일 경로
            
        Returns:
            캡션 정보 딕셔너리
        """
        if not self._load_blip2_model():
            return {
                'caption': '',
                'caption_confidence': 0.0,
                'error': 'BLIP-2 모델 사용 불가'
            }
        
        try:
            # 이미지 로드
            if not Path(image_path).exists():
                return {
                    'caption': '',
                    'caption_confidence': 0.0,
                    'error': f'이미지 파일 없음: {image_path}'
                }
            
            image = Image.open(image_path).convert('RGB')
            
            # BLIP-2로 캡션 생성 (설정값 반영)
            inputs = self.blip2_processor(image, return_tensors="pt").to(self.device)
            
            captioning_config = self.config.get('image_captioning', {})
            max_length = captioning_config.get('max_length', 50)
            temperature = captioning_config.get('temperature', 0.7)
            
            with torch.no_grad():
                generated_ids = self.blip2_model.generate(
                    **inputs, 
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True if temperature > 0 else False
                )
                caption = self.blip2_processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )[0].strip()
            
            logger.info(f"이미지 캡션 생성 완료: {image_path} -> '{caption[:50]}...'")
            
            return {
                'caption': caption,
                'caption_confidence': 0.9,  # BLIP-2는 신뢰도를 직접 제공하지 않음
                'model_used': 'BLIP-2-OPT-2.7B',
                'generation_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"이미지 캡션 생성 오류 ({image_path}): {e}")
            return {
                'caption': '',
                'caption_confidence': 0.0,
                'error': str(e)
            }
    
    def count_tokens(self, text: str) -> int:
        """
        텍스트의 토큰 수 계산
        
        Args:
            text: 계산할 텍스트
            
        Returns:
            토큰 수
        """
        if not text:
            return 0
            
        if self.encoding:
            return len(self.encoding.encode(text))
        else:
            # tiktoken이 없는 경우 근사치 계산 (영어: 4자/토큰, 한국어: 2자/토큰)
            korean_chars = len(re.findall(r'[가-힣]', text))
            other_chars = len(text) - korean_chars
            return int(korean_chars / 2 + other_chars / 4)
    
    def split_text_by_sentences(self, text: str) -> List[str]:
        """
        텍스트를 문장 단위로 분할
        
        Args:
            text: 분할할 텍스트
            
        Returns:
            문장 리스트
        """
        # 한국어와 영어 문장 분할 패턴
        sentence_patterns = [
            r'[.!?]+\s+',  # 영어 문장 끝
            r'[.!?。！？]+\s*',  # 한국어 문장 끝
            r'\n\s*\n',  # 단락 구분
        ]
        
        sentences = [text]
        for pattern in sentence_patterns:
            new_sentences = []
            for sentence in sentences:
                split_sentences = re.split(pattern, sentence)
                new_sentences.extend([s.strip() for s in split_sentences if s.strip()])
            sentences = new_sentences
        
        return sentences
    
    def create_chunks_with_overlap(self, text: str) -> List[Dict[str, Any]]:
        """
        오버랩을 고려한 토큰 청크 생성
        
        Args:
            text: 청크로 분할할 텍스트
            
        Returns:
            청크 리스트 (메타데이터 포함)
        """
        if not text or not text.strip():
            return []
        
        sentences = self.split_text_by_sentences(text)
        chunks = []
        current_chunk = ""
        current_tokens = 0
        chunk_id = 0
        
        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            sentence_tokens = self.count_tokens(sentence)
            
            # 현재 청크에 문장을 추가할 수 있는지 확인
            if current_tokens + sentence_tokens <= self.chunk_size:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                current_tokens += sentence_tokens
                i += 1
            else:
                # 현재 청크 저장 (비어있지 않은 경우만)
                if current_chunk.strip():
                    chunks.append(self._create_chunk_metadata(
                        current_chunk, chunk_id, current_tokens
                    ))
                    chunk_id += 1
                
                # 오버랩 처리를 위한 백트래킹
                overlap_text = ""
                overlap_tokens = 0
                j = i - 1
                
                # 오버랩 크기만큼 이전 문장들을 포함
                while j >= 0 and overlap_tokens < self.overlap:
                    prev_sentence = sentences[j]
                    prev_tokens = self.count_tokens(prev_sentence)
                    
                    if overlap_tokens + prev_tokens <= self.overlap:
                        if overlap_text:
                            overlap_text = prev_sentence + " " + overlap_text
                        else:
                            overlap_text = prev_sentence
                        overlap_tokens += prev_tokens
                        j -= 1
                    else:
                        break
                
                # 새 청크 시작 (오버랩 포함)
                current_chunk = overlap_text
                current_tokens = overlap_tokens
                
                # 현재 문장이 청크 크기보다 큰 경우 강제 분할
                if sentence_tokens > self.chunk_size:
                    # 긴 문장을 강제로 분할
                    words = sentence.split()
                    word_chunk = ""
                    word_tokens = 0
                    
                    for word in words:
                        word_token_count = self.count_tokens(word + " ")
                        if word_tokens + word_token_count <= self.chunk_size - current_tokens:
                            word_chunk += word + " "
                            word_tokens += word_token_count
                        else:
                            # 현재 워드들을 청크에 추가
                            if word_chunk.strip():
                                current_chunk += " " + word_chunk.strip() if current_chunk else word_chunk.strip()
                                current_tokens += word_tokens
                            
                            # 청크 저장
                            if current_chunk.strip():
                                chunks.append(self._create_chunk_metadata(
                                    current_chunk, chunk_id, current_tokens
                                ))
                                chunk_id += 1
                            
                            # 새 청크 시작
                            current_chunk = word
                            current_tokens = word_token_count
                            word_chunk = ""
                            word_tokens = 0
                    
                    # 남은 단어들 추가
                    if word_chunk.strip():
                        current_chunk += " " + word_chunk.strip() if current_chunk else word_chunk.strip()
                        current_tokens += word_tokens
                    
                    i += 1
        
        # 마지막 청크 저장
        if current_chunk.strip():
            chunks.append(self._create_chunk_metadata(
                current_chunk, chunk_id, current_tokens
            ))
        
        logger.info(f"텍스트 청크 분할 완료: {len(chunks)}개 청크 생성")
        return chunks
    
    def _create_chunk_metadata(self, text: str, chunk_id: int, token_count: int) -> Dict[str, Any]:
        """
        청크 메타데이터 생성
        
        Args:
            text: 청크 텍스트
            chunk_id: 청크 ID
            token_count: 토큰 수
            
        Returns:
            메타데이터가 포함된 청크 딕셔너리
        """
        return {
            "chunk_id": chunk_id,
            "text": text.strip(),
            "token_count": token_count,
            "character_count": len(text.strip()),
            "word_count": len(text.strip().split()),
            "created_at": datetime.now().isoformat(),
            "chunk_size_limit": self.chunk_size,
            "overlap_size": self.overlap
        }
    
    def chunk_document_content(self, document_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        BLIP-2 + GMFT + OCR을 활용한 멀티모달 문서 청크 분할
        
        Args:
            document_data: 파싱된 문서 데이터  
            
        Returns:
            분리된 텍스트, 이미지, 표가 포함된 문서 데이터
        """
        try:
            # 전체 마크다운 텍스트에서 청크 생성
            full_markdown = document_data.get('full_markdown', '')
            
            if not full_markdown:
                logger.warning("마크다운 텍스트가 비어있습니다")
                self._create_empty_multimodal_structure(document_data)
                return document_data
            
            # 헤더 부분 제거 (--- 이후부터 청크 분할)
            if '---' in full_markdown:
                content_parts = full_markdown.split('---', 1)
                main_content = content_parts[1].strip() if len(content_parts) > 1 else full_markdown
            else:
                main_content = full_markdown
            
            # 기본 청크 생성 (텍스트 + 미디어 마크다운 포함)
            raw_chunks = self.create_chunks_with_overlap(main_content)
            
            # 🔥 **핵심: 멀티모달 미디어 분리 및 강화**
            media_registry = self._process_multimodal_content(raw_chunks, document_data)
            
            # 분리된 구조로 문서 데이터 구성
            document_data.update({
                'text_chunks': media_registry['text_only_chunks'],      # 순수 텍스트만
                'contextual_images': media_registry['images'],         # BLIP-2 캡션 + 컨텍스트
                'contextual_tables': media_registry['tables'],         # GMFT 구조 + OCR/텍스트
                'media_registry': media_registry,                      # 전체 미디어 레지스트리
                'multimodal_processing_stats': media_registry['processing_stats']
            })
            
            # 🔥 **분리된 멀티모달 구조 기반 통계 생성**
            text_chunks = media_registry['text_only_chunks']
            chunk_stats = {
                "total_text_chunks": len(text_chunks),
                "total_chunk_tokens": sum(chunk['token_count'] for chunk in text_chunks),
                "average_chunk_tokens": sum(chunk['token_count'] for chunk in text_chunks) / len(text_chunks) if text_chunks else 0,
                "chunk_size_limit": self.chunk_size,
                "overlap_size": self.overlap,
                "processing_timestamp": datetime.now().isoformat(),
                
                # **분리된 멀티모달 통계** 
                "separated_multimodal_stats": {
                    "total_contextual_images": len(media_registry['images']),
                    "images_with_blip2_captions": media_registry['processing_stats']['images_with_blip2_captions'],
                    "images_with_ocr": media_registry['processing_stats']['images_with_ocr'],
                    "total_contextual_tables": len(media_registry['tables']),
                    "tables_with_content": media_registry['processing_stats']['tables_with_content'],
                    "text_chunks_with_media_refs": media_registry['processing_stats']['chunks_with_media_references'],
                    "processing_approach": "BLIP-2_caption + GMFT_structure + OCR_extraction"
                }
            }
            
            # 기존 통계에 청크 정보 추가
            if 'document_stats' not in document_data:
                document_data['document_stats'] = {}
            
            document_data['document_stats']['chunking'] = chunk_stats
            
            logger.info(f"🔥 BLIP-2 + GMFT 멀티모달 분할 완료: "
                       f"{len(text_chunks)}개 텍스트 청크 (평균 {chunk_stats['average_chunk_tokens']:.1f} 토큰), "
                       f"{len(media_registry['images'])}개 이미지 "
                       f"(BLIP-2 캡션 {media_registry['processing_stats']['images_with_blip2_captions']}개), "
                       f"{len(media_registry['tables'])}개 표")
            
            return document_data
            
        except Exception as e:
            logger.error(f"문서 청크 분할 오류: {str(e)}")
            document_data['text_chunks'] = []
            return document_data
    
    def _add_multimodal_metadata(self, chunks: List[Dict[str, Any]], 
                                document_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        청크에 이미지/표 메타데이터 추가
        
        Args:
            chunks: 원본 청크 리스트
            document_data: 원본 문서 데이터
            
        Returns:
            메타데이터가 추가된 청크 리스트
        """
        try:
            blocks = document_data.get('blocks', [])
            image_blocks = [b for b in blocks if b.get('type') == 'image_ocr']
            table_blocks = [b for b in blocks if b.get('type') == 'table']
            
            enhanced_chunks = []
            
            for chunk in chunks:
                chunk_text = chunk.get('text', '')
                
                # 청크에 포함된 이미지 찾기
                chunk_images = []
                for img_block in image_blocks:
                    img_content = img_block.get('content', '')
                    img_path = img_block.get('source', {}).get('image_path', '')
                    
                    # 이미지 경로가 청크 텍스트에 포함되어 있는지 확인
                    if img_path and img_path in chunk_text:
                        chunk_images.append({
                            'image_path': img_path,
                            'ocr_text': img_content,
                            'page': img_block.get('source', {}).get('page', 0),
                            'bbox': img_block.get('source', {}).get('bbox', []),
                            'confidence': img_block.get('metadata', {}).get('ocr_confidence', 0)
                        })
                
                # 청크에 포함된 표 찾기 (마크다운 표 패턴도 감지)
                chunk_tables = []
                
                # GMFT/기타 감지된 표 블록
                for table_block in table_blocks:
                    table_content = table_block.get('content', '')
                    table_id = table_block.get('metadata', {}).get('table_id', '')
                    
                    if table_content and (table_content[:100] in chunk_text or table_id in chunk_text):
                        chunk_tables.append({
                            'table_id': table_id,
                            'type': 'structured_table',
                            'markdown_content': table_content,
                            'page': table_block.get('source', {}).get('page', 0),
                            'bbox': table_block.get('source', {}).get('bbox', []),
                            'dimensions': table_block.get('metadata', {}).get('dimensions', {}),
                            'confidence': table_block.get('metadata', {}).get('confidence', 1.0)
                        })
                
                # 마크다운 표 패턴 감지 (Marker가 변환한 표들)
                import re
                markdown_table_pattern = r'\|[^|\n]*\|[^|\n]*\|'
                table_matches = re.findall(markdown_table_pattern, chunk_text)
                
                for i, table_match in enumerate(table_matches):
                    # 마크다운 표의 줄 수 계산
                    table_lines = [line for line in table_match.split('\n') if '|' in line and '---' not in line]
                    if len(table_lines) >= 2:  # 헤더 + 최소 1개 데이터 행
                        chunk_tables.append({
                            'table_id': f'markdown_table_{chunk["chunk_id"]}_{i}',
                            'type': 'markdown_table',
                            'markdown_content': table_match,
                            'estimated_rows': len(table_lines) - 1,  # 헤더 제외
                            'estimated_cols': len(table_lines[0].split('|')) - 2 if table_lines else 0,
                            'extraction_method': 'marker_text_based'
                        })
                
                # 청크에 멀티모달 메타데이터 추가
                enhanced_chunk = chunk.copy()
                enhanced_chunk['multimodal_content'] = {
                    'has_images': len(chunk_images) > 0,
                    'has_tables': len(chunk_tables) > 0,
                    'images': chunk_images,
                    'tables': chunk_tables,
                    'total_media_items': len(chunk_images) + len(chunk_tables)
                }
                
                enhanced_chunks.append(enhanced_chunk)
            
            logger.info(f"멀티모달 메타데이터 추가 완료: "
                       f"{sum(len(c['multimodal_content']['images']) for c in enhanced_chunks)}개 이미지, "
                       f"{sum(len(c['multimodal_content']['tables']) for c in enhanced_chunks)}개 표")
            
            return enhanced_chunks
            
        except Exception as e:
            logger.error(f"멀티모달 메타데이터 추가 오류: {str(e)}")
            return chunks  # 오류 시 원본 청크 반환
    
    def get_chunk_statistics(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        청크 통계 정보 생성
        
        Args:
            chunks: 청크 리스트
            
        Returns:
            통계 정보
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "total_tokens": 0,
                "average_tokens": 0,
                "min_tokens": 0,
                "max_tokens": 0
            }
        
        token_counts = [chunk['token_count'] for chunk in chunks]
        
        return {
            "total_chunks": len(chunks),
            "total_tokens": sum(token_counts),
            "average_tokens": sum(token_counts) / len(token_counts),
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
            "chunk_size_limit": self.chunk_size,
            "overlap_size": self.overlap
        } 

    def _create_empty_multimodal_structure(self, document_data: Dict[str, Any]):
        """빈 멀티모달 구조 생성"""
        document_data.update({
            'text_chunks': [],
            'contextual_images': [],
            'contextual_tables': [],
            'media_registry': {
                'images': [], 'tables': [], 'text_only_chunks': [],
                'processing_stats': {'error': '마크다운 텍스트 없음'}
            },
            'multimodal_processing_stats': {'error': '마크다운 텍스트 없음'}
        })
    
    def _process_multimodal_content(self, raw_chunks: List[Dict[str, Any]], 
                                   document_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        🔥 **핵심 함수**: BLIP-2 + GMFT + OCR을 활용한 멀티모달 처리
        
        Args:
            raw_chunks: 기본 텍스트 청크들
            document_data: 문서 데이터
            
        Returns:
            처리된 미디어 레지스트리
        """
        try:
            blocks = document_data.get('blocks', [])
            full_markdown = document_data.get('full_markdown', '')
            
            media_registry = {
                'images': [],
                'tables': [],
                'text_only_chunks': [],
                'processing_stats': {
                    'total_images_processed': 0,
                    'images_with_blip2_captions': 0,
                    'images_with_ocr': 0,
                    'total_tables_processed': 0,
                    'tables_with_gmft_structure': 0,
                    'tables_with_content': 0,
                    'text_chunks_created': 0,
                    'chunks_with_media_references': 0
                }
            }
            
            # 🖼️ **1단계: 이미지 처리 (BLIP-2 캡션 + OCR + 컨텍스트)**
            self._process_images_with_blip2(blocks, media_registry, full_markdown)
            
            # 📊 **2단계: 표 처리 (GMFT 구조 + OCR/텍스트 추출)**
            self._process_tables_with_gmft(blocks, document_data, media_registry, full_markdown)
            
            # 📝 **3단계: 텍스트 청크 생성 (미디어 제거 + 참조 유지)**
            self._create_text_only_chunks(raw_chunks, media_registry)
            
            logger.info(f"멀티모달 처리 완료: "
                       f"이미지 {media_registry['processing_stats']['total_images_processed']}개 "
                       f"(BLIP-2 캡션 {media_registry['processing_stats']['images_with_blip2_captions']}개), "
                       f"표 {media_registry['processing_stats']['total_tables_processed']}개, "
                       f"텍스트 청크 {media_registry['processing_stats']['text_chunks_created']}개")
            
            return media_registry
            
        except Exception as e:
            logger.error(f"멀티모달 콘텐츠 처리 오류: {str(e)}")
            return {
                'images': [], 'tables': [], 'text_only_chunks': raw_chunks,
                'processing_stats': {'error': str(e)}
            }
    
    def _process_images_with_blip2(self, blocks: List[Dict], media_registry: Dict, full_markdown: str):
        """BLIP-2를 활용한 이미지 처리"""
        image_blocks = [b for b in blocks if b.get('type') == 'image_ocr']
        
        for img_block in image_blocks:
            page = img_block.get('source', {}).get('page', 0)
            image_path = img_block.get('source', {}).get('image_path', '')
            ocr_content = img_block.get('content', '')
            
            # 컨텍스트 추출
            context = self._extract_media_context(blocks, img_block)
            
            # 🔥 BLIP-2 캡션 생성 (핵심!)
            captioning_enabled = self.config.get('image_captioning', {}).get('enabled', True)
            
            if captioning_enabled and image_path:
                caption_info = self._generate_image_caption(image_path)
            else:
                reason = 'BLIP-2 disabled' if not captioning_enabled else 'No image path'
                caption_info = {
                    'caption': '', 
                    'caption_confidence': 0.0, 
                    'error': reason
                }
            
            image_item = {
                'media_id': f'img_{page}_{len(media_registry["images"])}',
                'type': 'image',
                'image_path': image_path,
                'page': page,
                'bbox': img_block.get('source', {}).get('bbox', []),
                
                # 🎯 **RAG용 의미있는 콘텐츠**
                'content_analysis': {
                    'ocr_text': ocr_content,
                    'ocr_confidence': img_block.get('metadata', {}).get('ocr_confidence', 0),
                    'blip2_caption': caption_info.get('caption', ''),
                    'caption_confidence': caption_info.get('caption_confidence', 0),
                    'combined_description': self._combine_ocr_and_caption(ocr_content, caption_info.get('caption', '')),
                    'has_meaningful_content': bool(ocr_content or caption_info.get('caption', ''))
                },
                
                # 주변 컨텍스트 (검색 시 활용)
                'context': context,
                'text_references': self._find_image_references(full_markdown, image_path)
            }
            
            media_registry['images'].append(image_item)
            
            # 통계 업데이트
            media_registry['processing_stats']['total_images_processed'] += 1
            if caption_info.get('caption'):
                media_registry['processing_stats']['images_with_blip2_captions'] += 1
            if ocr_content:
                media_registry['processing_stats']['images_with_ocr'] += 1
    
    def _process_tables_with_gmft(self, blocks: List[Dict], document_data: Dict, 
                                 media_registry: Dict, full_markdown: str):
        """GMFT + OCR 조합을 활용한 표 처리"""
        
        # 기존 Marker 감지 표 처리 (벡터 기반)
        table_blocks = [b for b in blocks if b.get('type') == 'table']
        for table_block in table_blocks:
            page = table_block.get('source', {}).get('page', 0)
            context = self._extract_media_context(blocks, table_block)
            
            table_item = {
                'media_id': f'table_{page}_{len(media_registry["tables"])}',
                'type': 'table',
                'detection_method': 'marker_vector_based',
                'page': page,
                'bbox': table_block.get('source', {}).get('bbox', []),
                
                # 표 콘텐츠 분석
                'content_analysis': {
                    'markdown_content': table_block.get('content', ''),
                    'content_source': 'pdf_vector_text',  # 벡터 기반 텍스트
                    'row_count': self._count_markdown_table_rows(table_block.get('content', '')),
                    'col_count': self._count_markdown_table_cols(table_block.get('content', '')),
                    'has_meaningful_content': bool(table_block.get('content', '').strip())
                },
                
                'context': context,
                'text_references': self._find_table_references(full_markdown)
            }
            
            media_registry['tables'].append(table_item)
            media_registry['processing_stats']['total_tables_processed'] += 1
            if table_block.get('content', '').strip():
                media_registry['processing_stats']['tables_with_content'] += 1
        
        # TODO: GMFT 감지 표 추가 처리 (이미지 기반 - 더 복잡한 구현 필요)
        # gmft_tables = self._get_gmft_detected_tables(document_data)
        # for gmft_table in gmft_tables:
        #     self._process_gmft_table_with_ocr(gmft_table, media_registry)
    
    def _create_text_only_chunks(self, raw_chunks: List[Dict], media_registry: Dict):
        """미디어 제거된 순수 텍스트 청크 생성"""
        for chunk in raw_chunks:
            original_text = chunk.get('text', '')
            
            # 미디어 마크다운 제거한 순수 텍스트
            clean_text = self._remove_media_markdown(original_text)
            
            if clean_text.strip():
                text_chunk = chunk.copy()
                text_chunk['text'] = clean_text
                text_chunk['is_text_only'] = True
                
                # 🔗 미디어 참조 연결 (RAG 검색 시 중요!)
                referenced_media = self._find_chunk_media_references(original_text, media_registry)
                text_chunk['referenced_media'] = referenced_media
                
                media_registry['text_only_chunks'].append(text_chunk)
                media_registry['processing_stats']['text_chunks_created'] += 1
                
                if referenced_media:
                    media_registry['processing_stats']['chunks_with_media_references'] += 1 

    # =============== 헬퍼 함수들 ===============
    
    def _extract_media_context(self, blocks: List[Dict], target_block: Dict) -> Dict[str, str]:
        """미디어 블록 주변의 컨텍스트 추출"""
        target_page = target_block.get('source', {}).get('page', 0)
        target_bbox = target_block.get('source', {}).get('bbox', [])
        
        if not target_bbox or len(target_bbox) < 4:
            return {
                'preceding_text': '',
                'following_text': '',
                'page_title': '',
                'section_header': ''
            }
        
        target_y = target_bbox[1]  # y 좌표
        
        preceding_texts = []
        following_texts = []
        
        # 같은 페이지의 텍스트 블록들에서 주변 텍스트 찾기
        for block in blocks:
            if (block.get('source', {}).get('page', -1) == target_page and 
                block.get('type') in ['Text', 'Span', 'Line', 'text', 'paragraph'] and
                block.get('source', {}).get('bbox')):
                
                block_bbox = block.get('source', {}).get('bbox', [])
                if len(block_bbox) >= 4:
                    block_y = block_bbox[1]
                    content = block.get('content', '').strip()
                    
                    if content:
                        if block_y < target_y:  # 타겟보다 위에 있는 텍스트
                            preceding_texts.append((block_y, content))
                        elif block_y > target_y:  # 타겟보다 아래에 있는 텍스트
                            following_texts.append((block_y, content))
        
        # 거리순 정렬 후 가장 가까운 텍스트들 선택
        preceding_texts.sort(key=lambda x: x[0], reverse=True)  # 가까운 순으로
        following_texts.sort(key=lambda x: x[0])  # 가까운 순으로
        
        return {
            'preceding_text': ' '.join([text for _, text in preceding_texts[:3]]),  # 앞 3개
            'following_text': ' '.join([text for _, text in following_texts[:3]]),   # 뒤 3개
            'page_title': self._extract_page_title(blocks, target_page),
            'section_header': self._extract_nearest_section_header(blocks, target_block)
        }
    
    def _extract_page_title(self, blocks: List[Dict], page: int) -> str:
        """페이지의 제목/헤더 추출"""
        for block in blocks:
            if (block.get('source', {}).get('page', -1) == page and 
                block.get('type') in ['SectionHeader', 'title', 'header']):
                return block.get('content', '').strip()
        return ''
    
    def _extract_nearest_section_header(self, blocks: List[Dict], target_block: Dict) -> str:
        """타겟 블록에 가장 가까운 섹션 헤더 찾기"""
        target_page = target_block.get('source', {}).get('page', 0)
        target_y = target_block.get('source', {}).get('bbox', [0, 0])[1]
        
        closest_header = ''
        min_distance = float('inf')
        
        for block in blocks:
            if (block.get('source', {}).get('page', -1) == target_page and 
                block.get('type') == 'SectionHeader'):
                
                block_bbox = block.get('source', {}).get('bbox', [])
                if len(block_bbox) >= 4:
                    block_y = block_bbox[1]
                    if block_y < target_y:  # 타겟보다 위에 있는 헤더
                        distance = target_y - block_y
                        if distance < min_distance:
                            min_distance = distance
                            closest_header = block.get('content', '').strip()
        
        return closest_header
    
    def _combine_ocr_and_caption(self, ocr_text: str, blip2_caption: str) -> str:
        """OCR 텍스트와 BLIP-2 캡션을 조합한 완전한 이미지 설명"""
        combined_parts = []
        
        if blip2_caption.strip():
            combined_parts.append(f"Image description: {blip2_caption}")
        
        if ocr_text.strip():
            combined_parts.append(f"Text in image: {ocr_text}")
        
        return ' | '.join(combined_parts) if combined_parts else ''
    
    def _find_image_references(self, full_markdown: str, image_path: str) -> List[str]:
        """텍스트에서 이미지 참조 표현 찾기"""
        import re
        references = []
        
        if image_path:
            filename = Path(image_path).stem  # 파일명 (확장자 제외)
            
            # 한국어 참조 표현 패턴
            korean_patterns = [
                r'위\s*(?:그림|이미지|사진|도표)',
                r'아래\s*(?:그림|이미지|사진|도표)',
                r'다음\s*(?:그림|이미지|사진|도표)',
                r'앞\s*(?:그림|이미지|사진|도표)',
                r'(?:그림|이미지|사진|도표)\s*[\d\-_]+',
                filename  # 파일명 기반
            ]
            
            for pattern in korean_patterns:
                matches = re.findall(pattern, full_markdown, re.IGNORECASE)
                references.extend(matches)
        
        return list(set(references))  # 중복 제거
    
    def _find_table_references(self, full_markdown: str) -> List[str]:
        """텍스트에서 표 참조 표현 찾기"""
        import re
        
        table_patterns = [
            r'위\s*표',
            r'아래\s*표',
            r'다음\s*표',
            r'앞\s*표',
            r'표\s*[\d\-_]+',
            r'다음과\s*같다',
            r'아래와\s*같다'
        ]
        
        references = []
        for pattern in table_patterns:
            matches = re.findall(pattern, full_markdown, re.IGNORECASE)
            references.extend(matches)
        
        return list(set(references))
    
    def _remove_media_markdown(self, text: str) -> str:
        """텍스트에서 이미지/표 마크다운 제거"""
        import re
        
        # 이미지 마크다운 제거: ![이미지](path)
        text = re.sub(r'!\[.*?\]\([^)]*\)', '[이미지 제거됨]', text)
        
        # 이미지 설명 섹션 제거: **이미지에서 추출된 텍스트:** > ...
        text = re.sub(r'\*\*이미지에서 추출된 텍스트:\*\*[^\n]*\n?', '', text)
        text = re.sub(r'>\s*[^\n]*\n?', '', text)  # > 로 시작하는 인용 텍스트
        
        # 표 마크다운 제거: | col1 | col2 |
        text = re.sub(r'\|[^|\n]*\|[^|\n]*\|[^\n]*\n?', '[표 제거됨]\n', text)
        text = re.sub(r'\|[-\s]*\|[-\s]*\|[^\n]*\n?', '', text)  # 표 구분선 제거
        
        # 연속된 빈 줄과 제거됨 표시 정리
        text = re.sub(r'\[이미지 제거됨\]\s*\[표 제거됨\]', '[미디어 제거됨]', text)
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        
        return text.strip()
    
    def _find_chunk_media_references(self, original_text: str, media_registry: Dict) -> List[str]:
        """청크 텍스트에서 참조하는 미디어 ID들 찾기"""
        referenced_ids = []
        
        # 이미지 참조 찾기
        for img in media_registry['images']:
            img_path = img.get('image_path', '')
            if img_path and img_path in original_text:
                referenced_ids.append(img['media_id'])
            
            # 텍스트 참조로도 확인
            for ref in img.get('text_references', []):
                if ref and ref.lower() in original_text.lower():
                    referenced_ids.append(img['media_id'])
                    break
        
        # 표 참조 찾기
        for table in media_registry['tables']:
            for ref in table.get('text_references', []):
                if ref and ref.lower() in original_text.lower():
                    referenced_ids.append(table['media_id'])
                    break
        
        return list(set(referenced_ids))  # 중복 제거
    
    def _count_markdown_table_rows(self, markdown_content: str) -> int:
        """마크다운 표의 행 수 계산"""
        if not markdown_content:
            return 0
        lines = [line.strip() for line in markdown_content.split('\n') if line.strip()]
        table_lines = [line for line in lines if '|' in line and '---' not in line]
        return max(0, len(table_lines) - 1)  # 헤더 제외
    
    def _count_markdown_table_cols(self, markdown_content: str) -> int:
        """마크다운 표의 열 수 계산"""
        if not markdown_content:
            return 0
        lines = [line.strip() for line in markdown_content.split('\n') if line.strip()]
        table_lines = [line for line in lines if '|' in line and '---' not in line]
        if table_lines:
            return len(table_lines[0].split('|')) - 2  # 양쪽 빈 부분 제외
        return 0 