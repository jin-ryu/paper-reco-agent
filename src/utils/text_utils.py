import re
from typing import List


def clean_text(text: str) -> str:
    """
    Removes HTML tags, Markdown image tags, and other noise from text.
    """
    if not text:
        return ""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Remove Markdown image tags and base64 data
    text = re.sub(r'!\[.*?\]\(data:image\/[a-zA-Z]+;base64,.*?\)', ' ', text)
    # Remove any remaining markdown image tags
    text = re.sub(r'!\[.*?\]\(.*?\)', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def clean_keywords(keywords: List[str]) -> List[str]:
    """키워드 중복 제거 및 전처리 (공통 함수)"""
    result = []
    seen = set()

    for kw in keywords:
        # 1. 앞뒤 공백 제거
        kw_clean = kw.strip()

        # 2. 불필요한 특수문자 제거 (점, 쉼표, 세미콜론 등)
        kw_clean = re.sub(r'^[.,;:!?\s]+|[.,;:!?\s]+$', '', kw_clean)

        # 3. 연속된 공백을 하나로
        kw_clean = re.sub(r'\s+', ' ', kw_clean)

        # 4. 빈 문자열이나 너무 짧은 키워드 제외 (1글자 제외)
        if len(kw_clean) < 2:
            continue

        # 5. 중복 체크 (대소문자 무시)
        kw_lower = kw_clean.lower()
        if kw_lower not in seen:
            result.append(kw_clean)
            seen.add(kw_lower)

    return result
