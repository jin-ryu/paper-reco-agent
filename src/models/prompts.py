import textwrap
from typing import Dict, Any, List

def create_reranking_prompt(context: Dict[str, Any], num_recommendations: int, previous_error: str = None) -> str:
    """
    Re-ranking을 위한 프롬프트 생성

    Args:
        task_description: 작업 설명
        context: 컨텍스트 정보
        num_recommendations: 추천할 개수
        previous_error: 이전 파싱 실패 에러 (재시도 시)

    Returns:
        프롬프트 문자열
    """
    source_title = context.get('source_title', '')
    source_description = context.get('source_description', '')
    source_keywords = context.get('source_keywords', '')

    candidates = context.get('candidates', [])
    candidate_type = context.get('candidate_type', '후보')

    # 후보 정보 포맷팅 (re-ranking에 필요한 정보만)
    candidates_text = ""
    for i, candidate in enumerate(candidates, 1):
        cand_id = candidate.get('id', '')
        title = candidate.get('title', '')
        desc = candidate.get('description', '')
        keywords = ", ".join(candidate.get('keywords', []))

        # 유사도 점수 (세부 점수 포함)
        semantic_score = candidate.get('semantic_score', 0.0)
        lexical_score = candidate.get('lexical_score', 0.0)
        hybrid_score = candidate.get('hybrid_score', 0.0)

        candidates_text += f"\n[{i}] ID: {cand_id}\n"
        candidates_text += f"- Title: {title}\n"
        candidates_text += f"- Description: {desc}\n"
        candidates_text += f"- Keywords: {keywords}\n"
        candidates_text += f"- Semantic Score (E5): {semantic_score:.3f}\n"
        candidates_text += f"- Lexical Score (BM25): {lexical_score:.3f}\n"
        candidates_text += f"- Final Score (Hybrid): {hybrid_score:.3f}\n"

    # 에러 피드백 추가
    error_feedback = ""
    if previous_error:
        error_feedback = f"""
**IMPORTANT - Previous Parsing Error**:
Your last response failed to parse with error: {previous_error}
Please avoid this error and output ONLY valid JSON format this time.
"""

    prompt = f"""# Task: Re-rank and Select Top {num_recommendations} {candidate_type} Recommendations
You are a research recommendation expert. Re-rank the candidates and select the top {num_recommendations} most relevant items.
## Source Dataset:
Title: {source_title}
Description: {source_description}
Keywords: {source_keywords}

## Top {len(candidates)} {candidate_type} Candidates (by E5+BM25 hybrid score):
{candidates_text}
{error_feedback}

## Your Task:
1. Analyze each candidate's title, description, and scores (semantic, lexical, final).
2. Select the TOP {num_recommendations} most relevant candidates.
3. Assign RANK (1 to {num_recommendations}) based on relevance.
4. Write a specific REASON in KOREAN. **Do not mention the numeric scores directly in the reason.** Instead, explain the relevance qualitatively (e.g., "의미적으로 매우 유사함", "핵심 키워드가 다수 일치함").
5. Determine LEVEL based on the final score.

## Score Information:
- Semantic Score (E5): Text embedding similarity (0.0-1.0).
- Lexical Score (BM25): Keyword matching score (0.0-1.0).
- Final Score (Hybrid): Combined score = 0.7 * Semantic + 0.3 * Lexical.

## Recommendation Level:
- 강추 (Strong): Score >= 0.70, highly relevant
- 추천 (Recommend): Score >= 0.55, moderately relevant
- 참고 (Reference): Score >= 0.40, somewhat relevant

## Output Format:
Output ONLY valid JSON (no markdown, no comments) with a single "recommendations" key:
{{
  "recommendations": [
    // ... {num_recommendations} recommendation objects here ...
  ]
}}

## Rules:
- Output EXACTLY {num_recommendations} recommendations.
- rank: Your ranking decision (1, 2, 3, ...).
- candidate_id: ID from the candidate list above.
- reason: Specific explanation in KOREAN (qualitative, no scores).
- level: "강추", "추천", or "참고".
- Start response with '{{' character.
""".strip()
    return prompt


def create_search_queries_prompt(source_data: Dict[str, Any]) -> str:
    """
    검색 쿼리 생성을 위한 프롬프트 생성

    Args:
        source_data: 소스 데이터셋 정보

    Returns:
        프롬프트 문자열
    """
    # 제목과 설명 선택 (한글 우선, 없으면 영어)
    title = source_data.get('title_ko') or source_data.get('title_en', '')
    description = source_data.get('description_ko') or source_data.get('description_en', '')

    # API에서 가져온 원본 키워드
    original_keywords = source_data.get('keywords', [])
    keywords_str = ', '.join(original_keywords[:10]) if original_keywords else ''

    # 언어 설정 (dataset_main_lang_pc, dataset_sub_lang_pc 필드 참고)
    main_lang = source_data.get('dataset_main_lang_pc', 'Korean')
    sub_lang = source_data.get('dataset_sub_lang_pc', 'English')

    # 프롬프트 생성
    prompt = f"""You are a research data search expert. Generate SHORT and PRECISE search keywords for optimal search results.
Input Dataset:
   - Title: {title}
   - Description: {description[:300]}...
   - Original Keywords (from API): {keywords_str}
   - Main Language: {main_lang}
   - Sub Language: {sub_lang}

Task:
1. Analyze original keywords and SELECT only SHORT, SPECIFIC ones (filter out too generic or too long phrases).
2. Generate NEW SHORT keywords (1-3 words maximum) based on title and description.
3. For dataset search: core topics, data types, domain names (SHORT terms).
4. For paper search: research methods, key concepts, technical terms (SHORT terms).
5. Use {main_lang} (main) and {sub_lang} (secondary).
6. Output 3 to 5 SHORT keywords per category.

KEYWORD LENGTH RULES:
- GOOD: "대사체", "NMR", "혈장", "농약", "독소" (1-2 words, concise)
- BAD: "NMR 기반 대사체 분석", "환경 오염물질과 대사 연관성" (too long, phrase-like)
- Keep it SHORT (maximum 3 words per keyword) for better search coverage.
- MUST output 3 to 5 keywords for dataset_queries and paper_queries.

IMPORTANT: Output ONLY valid JSON. No thinking, no explanations, no markdown.
Output this exact JSON structure (3 to 5 keywords each):
{{
  "dataset_queries": ["keyword1", "keyword2", "keyword3"],
  "paper_queries": ["keyword1", "keyword2", "keyword3", "keyword4"]
}}
""".strip()
    return prompt
