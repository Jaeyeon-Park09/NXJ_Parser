# NXJ_Parser 개발 작업 계획서

이 문서는 `/home/james4u1/NXJ_Parser` 디렉토리 내에 구현할 **Marker + GMFT 기반 PDF 파서 파이프라인**의 전체 작업 흐름과 개발 단계를 안내합니다.  

---

## ✅ 프로젝트 개요

- **목표**: 식약처에서 제공하는 약 3천 개의 PDF 가이드라인 문서를 자동 파싱하여 마크다운 + JSON 형태로 구조화함.
- **대상 문서**: 다양한 형태의 PDF (디지털 텍스트, 스캔 이미지, 복잡 표, 이미지 삽입 포함)
- **제약 사항**:
  - GPU 및 유료 API 사용 금지
  - marker, gmft 소스는 수정하지 않고 참조만 가능
  - Marker 1.8+ 기준으로 to_dict() 미사용, MarkdownOutput 사용
- **입력 디렉토리**: `/home/james4u1/NXJ_Parser/pdf_files`

---

## 📁 폴더 및 파일 구조

```
/home/james4u1/NXJ_Parser/
│
├── main.py                           # 전체 파이프라인 실행 스크립트
├── config.yaml                       # OCR 언어, 경로 설정 등
├── /pdf_files/                       # 원본 PDF 파일 저장 위치
├── /pdf_files_o/                     # 기타 PDF 파일 저장 위치
├── /output/                          # 결과 JSON 저장 디렉토리
├── /image_png/                       # 이미지 블록 crop 결과 저장 디렉토리
└── /utils/
    ├── marker_runner.py              # Marker 기반 텍스트/이미지 블록 추출
    ├── gmft_runner.py                # GMFT를 통한 테이블 분석 처리
    ├── ocr_image_blocks.py           # 이미지 블록 crop + pytesseract OCR 처리
    ├── file_io.py                    # 파일 입출력 및 저장 관리
    └── formatter.py                  # 블록 통합 + Markdown/JSON 변환
```

---

## 🛠️ 작업 단계 및 요청 프롬프트

### 1. 목적 인식 및 파이프라인 설명

> **프롬프트:**
이 프로젝트는 식약처의 다양한 유형의 PDF 가이드라인 문서를 자동으로 구조화하여 마크다운+JSON 형태로 변환하는 파서 파이프라인을 구현하는 것이다. 주요 툴로 marker와 gmft를 사용하며, 구조는 아래와 같다:
[PDF 파일] 
  │
  ├─ Marker 처리
  │   ├─ 일반 텍스트 블록 추출 (디지털 + OCR fallback)
  │   ├─ 전체 페이지가 이미지일 경우: OCR 수행하여 텍스트 추출
  │   ├─ PDF 내 이미지 감지 → 이미지 블록 crop
  │
  ├─ (Image) 이미지 블록 crop
  │   └─ pytesseract 등으로 OCR 수행 → 텍스트 블록으로 추가
  │
  ├─ (Table) 전체 페이지 대상 GMFT 처리
  │   └─ 모든 표 감지 + 셀 구조 + 텍스트 추출 (TATR 기반)
  │
  ├─ 모든 블록 통합
  │   └─ 텍스트, 이미지 OCR 결과, 표 구조를 병합
  │
  └─ 마크다운 형태로 감싸서 최종 JSON 저장

이 전체 구조를 기준으로 아래 단계별로 구현 작업을 분할할 예정이다.

---

### 2. 폴더 구조 및 책임 정의

> **프롬프트:**
다음 폴더 구조를 기준으로, 각 모듈이 수행해야 할 책임과 역할을 함께 설명해줘:

- /home/james4u1/NXJ_Parser/
│
├── main.py
├── config.yaml
├── pdf_files/
├── output/
├── image_png/
└── utils/
    ├── marker_runner.py
    ├── gmft_runner.py
    ├── ocr_image_blocks.py
    ├── file_io.py
    └── formatter.py

---

### 3. marker_runner.py: Marker 기반 텍스트/이미지 블록 추출

> **프롬프트:**
PyPI에서 설치한 marker-pdf 모듈을 불러와 PDF에서 텍스트 블록과 이미지 블록을 추출하는 함수를 작성해줘. 
반드시 Marker 1.8+ 기준에 맞춰 `PdfConverter.convert()`의 반환값인 `MarkdownOutput` 객체를 사용하고, `to_dict()`는 사용하지 말아야 해.  
텍스트/이미지 블록은 `result.metadata["blocks"]`에서 가져오고, 디지털 PDF는 그대로 추출하되 스캔본은 OCR fallback을 적용해줘.

---

### 4. ocr_image_blocks.py: 이미지 블록 OCR 처리

> **프롬프트:**
Marker로 감지된 이미지 블록을 기반으로 PDF를 crop하여 `image_png/` 폴더에 저장하고, pytesseract로 OCR을 수행하여 텍스트를 추출하는 함수를 작성해줘.  
bbox 좌표를 기반으로 PyMuPDF(fitz)를 활용하고, OCR 언어 설정은 config.yaml에서 불러오도록 구성해줘.

---

### 5. gmft_runner.py: 표 전용 GMFT 분석

> **프롬프트:**
PyPI에서 설치한 gmft 모듈을 사용해, gmft.auto() 기반으로 PDF 전체를 처리해서 표를 감지하고, Markdown 및 JSON으로 표 구조를 출력하는 함수를 작성해줘.  
표가 없거나 오류가 발생하면 로그를 남겨줘.

---

### 6. formatter.py: Markdown + JSON 구조 통합

> **프롬프트:**
텍스트 블록, 이미지 OCR 결과, 표 구조 결과를 받아서 각 항목을 마크다운 포맷으로 감싸고 아래와 같은 JSON 구조로 반환하는 함수를 작성해줘:

```json
{
  "type": "text" | "image_ocr" | "table",
  "markdown": "...",
  "source": {
    "page": 1,
    "bbox": [...]
  }
}
```

---

### 7. main.py: 전체 파이프라인 통합 실행

> **프롬프트:**
다음 흐름으로 동작하는 main.py를 작성해줘:

1. /pdf_files 디렉토리에서 PDF 파일 불러오기
2. marker_runner로 텍스트 및 이미지 블록 추출
3. ocr_image_blocks로 이미지 OCR 수행 및 image_png 저장
4. gmft_runner로 표 감지 및 구조화
5. formatter로 모든 블록을 마크다운 기반 청크로 통합
6. JSON 파일을 output/{파일명}.json으로 저장
7. tqdm으로 진행률 표시하고, 실패한 PDF는 logs/failed.txt에 기록

---

### 8. 병렬 처리 스크립트 (선택)

> **프롬프트:**
3천 개의 PDF를 multiprocessing으로 병렬 처리할 수 있도록 구성해줘.  
여러 프로세스로 main.py의 핵심 로직을 병렬 실행하고, 실패한 PDF는 logs에 따로 저장해줘.
