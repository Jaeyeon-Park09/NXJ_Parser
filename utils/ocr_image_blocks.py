"""
이미지 블록 OCR 처리 모듈
Marker에서 감지된 이미지 블록을 crop하고 pytesseract로 OCR 수행
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from .config_loader import get_section, get_value, safe_bbox_conversion, ensure_directories

import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class OCRImageProcessor:
    """이미지 블록 OCR 처리 클래스"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        OCRImageProcessor 초기화
        
        Args:
            config_path: 설정 파일 경로
        """
        self.ocr_config = get_section('ocr', config_path)
        
        # 이미지 출력 디렉토리 생성
        image_output_path = get_value('paths.image_output', 'image_png', config_path)
        self.image_output_dir = Path(image_output_path)
        ensure_directories([self.image_output_dir])
        
        # OCR 설정
        self.ocr_language = self.ocr_config.get('language', 'kor+eng')
        self.ocr_psm = self.ocr_config.get('tesseract_config', '--psm 6')
        
    def crop_image_from_pdf(self, pdf_path: str, bbox: List[float], page_num: int, 
                           image_id: str) -> Optional[str]:
        """
        PDF에서 bbox 좌표를 사용하여 이미지 영역을 crop
        
        Args:
            pdf_path: PDF 파일 경로
            bbox: [x0, y0, x1, y1] 좌표
            page_num: 페이지 번호 (0부터 시작)
            image_id: 이미지 식별자
            
        Returns:
            crop된 이미지 파일 경로 (성공시) 또는 None (실패시)
        """
        try:
            # PDF 문서 열기
            doc = fitz.open(pdf_path)
            
            if page_num >= len(doc):
                logger.warning(f"페이지 번호가 범위를 벗어남: {page_num}, 전체 페이지: {len(doc)}")
                doc.close()
                return None
            
            page = doc[page_num]
            
            # bbox를 fitz.Rect로 변환
            rect = fitz.Rect(bbox)
            
            # 이미지 crop
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=rect)  # 2배 해상도
            
            # 이미지 파일 경로 생성
            pdf_name = Path(pdf_path).stem
            image_filename = f"{pdf_name}_page{page_num}_img{image_id}.png"
            image_path = self.image_output_dir / image_filename
            
            # 이미지 저장
            pix.save(str(image_path))
            pix = None  # 메모리 해제
            doc.close()
            
            logger.info(f"이미지 crop 완료: {image_path}")
            return str(image_path)
            
        except Exception as e:
            logger.error(f"이미지 crop 오류: {str(e)}")
            if 'doc' in locals():
                doc.close()
            return None
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        OCR 정확도 향상을 위한 이미지 전처리
        
        Args:
            image_path: 이미지 파일 경로
            
        Returns:
            전처리된 이미지 (numpy array)
        """
        try:
            # 이미지 로드
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
            
            # 그레이스케일 변환
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 노이즈 제거
            denoised = cv2.medianBlur(gray, 3)
            
            # 적응형 임계값 적용
            thresh = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # 모폴로지 연산 (선택적)
            kernel = np.ones((1, 1), np.uint8)
            processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            return processed
            
        except Exception as e:
            logger.warning(f"이미지 전처리 오류: {str(e)}, 원본 이미지 사용")
            # 전처리 실패시 원본 이미지 반환
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            return image if image is not None else np.array([])
    
    def extract_text_from_image(self, image_path: str, use_preprocessing: bool = True) -> Dict[str, any]:
        """
        이미지에서 OCR을 수행하여 텍스트 추출
        
        Args:
            image_path: 이미지 파일 경로
            use_preprocessing: 이미지 전처리 사용 여부
            
        Returns:
            OCR 결과 딕셔너리 (text, confidence, bbox_info 포함)
        """
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"이미지 파일이 존재하지 않습니다: {image_path}")
            
            # 이미지 전처리 (선택적)
            if use_preprocessing:
                processed_image = self.preprocess_image(image_path)
                if processed_image.size == 0:
                    raise ValueError("이미지 전처리 실패")
                
                # 임시 파일로 저장
                temp_path = str(Path(image_path).with_suffix('.temp.png'))
                cv2.imwrite(temp_path, processed_image)
                ocr_image_path = temp_path
            else:
                ocr_image_path = image_path
            
            # pytesseract OCR 수행
            custom_config = f'-l {self.ocr_language} {self.ocr_psm}'
            
            # 텍스트 추출
            text = pytesseract.image_to_string(
                Image.open(ocr_image_path), 
                config=custom_config
            ).strip()
            
            # 상세 정보 추출 (bbox, confidence 포함)
            data = pytesseract.image_to_data(
                Image.open(ocr_image_path), 
                config=custom_config,
                output_type=pytesseract.Output.DICT
            )
            
            # 신뢰도 계산
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            # 임시 파일 삭제
            if use_preprocessing and os.path.exists(temp_path):
                os.remove(temp_path)
            
            result = {
                "text": text,
                "confidence": avg_confidence,
                "word_count": len(text.split()),
                "language": self.ocr_language,
                "image_path": image_path,
                "bbox_data": data,
                "success": True
            }
            
            logger.info(f"OCR 완료: {image_path}, 텍스트 길이: {len(text)}, 신뢰도: {avg_confidence:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"OCR 처리 오류: {image_path}, 오류: {str(e)}")
            return {
                "text": "",
                "confidence": 0,
                "word_count": 0,
                "language": self.ocr_language,
                "image_path": image_path,
                "error": str(e),
                "success": False
            }
    
    def process_image_blocks(self, pdf_path: str, image_blocks: List[Dict]) -> List[Dict]:
        """
        이미지 블록 리스트를 일괄 처리
        
        Args:
            pdf_path: PDF 파일 경로
            image_blocks: Marker에서 추출된 이미지 블록 리스트
            
        Returns:
            OCR 결과가 포함된 이미지 블록 리스트
        """
        processed_blocks = []
        
        for i, block in enumerate(image_blocks):
            try:
                bbox = block.get("bbox", [])
                page_num = block.get("page", 0)
                
                # bbox 안전 변환
                bbox = safe_bbox_conversion(bbox)
                if len(bbox) < 4:
                    logger.warning(f"유효하지 않은 bbox")
                    continue
                
                # 이미지 crop
                image_path = self.crop_image_from_pdf(
                    pdf_path, bbox, page_num, str(i)
                )
                
                if image_path:
                    # OCR 수행
                    ocr_result = self.extract_text_from_image(image_path)
                    
                    # 원본 블록에 OCR 결과 추가
                    enhanced_block = block.copy()
                    enhanced_block.update({
                        "ocr_result": ocr_result,
                        "cropped_image_path": image_path,
                        "ocr_text": ocr_result.get("text", ""),
                        "ocr_confidence": ocr_result.get("confidence", 0),
                        "processed": True
                    })
                    
                    processed_blocks.append(enhanced_block)
                else:
                    # crop 실패시 원본 블록만 추가
                    block["processed"] = False
                    block["error"] = "이미지 crop 실패"
                    processed_blocks.append(block)
                    
            except Exception as e:
                logger.error(f"이미지 블록 처리 오류: {str(e)}")
                block["processed"] = False
                block["error"] = str(e)
                processed_blocks.append(block)
        
        logger.info(f"이미지 블록 처리 완료: {len(processed_blocks)}개 처리됨")
        return processed_blocks
    
    def cleanup_temp_files(self, keep_cropped_images: bool = True):
        """
        임시 파일 정리
        
        Args:
            keep_cropped_images: crop된 이미지 파일 보관 여부
        """
        try:
            if not keep_cropped_images:
                for image_file in self.image_output_dir.glob("*.png"):
                    image_file.unlink()
                logger.info("crop된 이미지 파일들이 정리되었습니다")
        except Exception as e:
            logger.warning(f"파일 정리 중 오류: {str(e)}")


def process_pdf_images(pdf_path: str, image_blocks: List[Dict], 
                      config_path: str = "config.yaml") -> List[Dict]:
    """
    PDF의 이미지 블록들을 일괄 OCR 처리하는 편의 함수
    
    Args:
        pdf_path: PDF 파일 경로
        image_blocks: 이미지 블록 리스트
        config_path: 설정 파일 경로
        
    Returns:
        OCR 결과가 포함된 이미지 블록 리스트
    """
    processor = OCRImageProcessor(config_path)
    return processor.process_image_blocks(pdf_path, image_blocks)


if __name__ == "__main__":
    # 테스트용 코드
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        processor = OCRImageProcessor()
        result = processor.extract_text_from_image(image_path)
        
        print(f"OCR 결과:")
        print(f"  텍스트: {result['text'][:100]}{'...' if len(result['text']) > 100 else ''}")
        print(f"  신뢰도: {result['confidence']:.2f}")
        print(f"  단어 수: {result['word_count']}")
    else:
        print("사용법: python ocr_image_blocks.py <image_file_path>") 