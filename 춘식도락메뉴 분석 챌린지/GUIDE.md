# 작업 가이드

## 폴더 구조

- `ai_top_100_menu/menu_images`: 원본 메뉴 이미지
- `data/vision`: OCR 결과 (`vision_YYYY-MM-DD.json`)
- `data/records`: 파싱 결과 (`records_YYYY-MM-DD.json`)
- `data/review`: OCR 정제 메모 (`ocr_review_YYYY-MM-DD.json`)
- `data/validation`: 규칙 검사 리포트 (`validation_YYYY-MM-DD.json`)
- `scripts`: 파이프라인 스크립트

## 준비 사항

- Python 3.10+
- Google Vision API 키
  - `VISION_API_KEY`(권장) 또는 `GOOGLE_API_KEY` 설정

예시 (PowerShell):

```powershell
$env:VISION_API_KEY="YOUR_API_KEY"
```

## 1단계. OCR (images -> vision_*.json)

폴더 전체:

```powershell
python scripts/run_vision_ocr.py --input ai_top_100_menu/menu_images --output-dir data/vision
```

단일 파일:

```powershell
python scripts/run_vision_ocr.py --input ai_top_100_menu/menu_images/2025-01-06.png --output-dir data/vision
```

## 2단계. 파싱 (vision_*.json -> records_*.json)

파서는 라이브러리 호출 형태다. 주차별로 짧은 파이썬 스니펫을 실행한다:

```powershell
@'
from pathlib import Path
import scripts.parse_vision_one as p

p.parse_one(
    Path("data/vision/vision_2025-01-13.json"),
    Path("data/records/records_2025-01-13.json"),
    Path("data/review/ocr_review_2025-01-13.json"),
)
'@ | python -
```

### 2025-01-13이 아닌 주차의 날짜 보정

현재 파서는 기본 시작일을 `2025-01-13`으로 가정한다. 다른 주차는
파싱 후 `date` 필드를 보정한다. (예: 2025-01-06 주차)

```powershell
@'
import json
from datetime import datetime
from pathlib import Path

out = Path("data/records/records_2025-01-06.json")
review = Path("data/review/ocr_review_2025-01-06.json")

base = datetime.strptime("2025-01-13", "%Y-%m-%d")
target = datetime.strptime("2025-01-06", "%Y-%m-%d")
offset = target - base

def shift_date(date_str):
    return (datetime.strptime(date_str, "%Y-%m-%d") + offset).strftime("%Y-%m-%d")

data = json.loads(out.read_text(encoding="utf-8"))
for r in data:
    r["date"] = shift_date(r["date"])
out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

review_data = json.loads(review.read_text(encoding="utf-8"))
for r in review_data:
    if "date" in r:
        r["date"] = shift_date(r["date"])
review.write_text(json.dumps(review_data, ensure_ascii=False, indent=2), encoding="utf-8")
'@ | python -
```

## 3단계. 검증 (records_*.json -> validation_*.json)

```powershell
python scripts/validate_records.py
```

리포트는 `data/validation`에 생성된다.

## 참고

- 레코드 파일이 비어 있으면 OCR JSON이 존재하는지, 헤더 날짜 토큰이
  인식되었는지 확인한다. 파서는 헤더의 일자 토큰을 컬럼 위치로 사용한다.
