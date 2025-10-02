from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from clients.dataon_client import DataONClient
from clients.scienceon_client import ScienceONClient
import logging
import asyncio
import math
from collections import Counter

logger = logging.getLogger(__name__)

# 전역 클라이언트 인스턴스
dataon_client = DataONClient()
scienceon_client = ScienceONClient()

# 임베딩 모델 (한국어 최적화: RoBERTa 기반, 멀티태스크 학습)
# jhgan/ko-sroberta-multitask: KorSTS Spearman 85.60%, 논문/데이터셋 추천에 최적
embedding_model = SentenceTransformer('jhgan/ko-sroberta-multitask')

# SmolAgent용 도구 함수들

def get_dataon_dataset_metadata(svc_id: str) -> dict:
    """
    DataON API를 사용해서 특정 데이터셋의 메타데이터를 가져옵니다.

    Args:
        svc_id: DataON 서비스 ID

    Returns:
        데이터셋의 상세 메타데이터
    """
    try:
        # 비동기 함수를 동기적으로 실행
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(dataon_client.get_dataset_metadata(svc_id))
        loop.close()
        return result
    except Exception as e:
        logger.error(f"DataON 메타데이터 조회 실패: {e}")
        return {"error": str(e)}

def search_similar_dataon_datasets(keywords: List[str], limit: int = 10) -> list:
    """
    DataON에서 키워드로 유사한 데이터셋들을 검색합니다.

    Args:
        keywords: 검색할 키워드 리스트
        limit: 반환할 최대 결과 수

    Returns:
        유사한 데이터셋들의 리스트
    """
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(dataon_client.search_by_keywords(keywords, limit))
        loop.close()
        return result
    except Exception as e:
        logger.error(f"DataON 데이터셋 검색 실패: {e}")
        return []

def get_scienceon_access_token() -> str:
    """
    ScienceON API 접근을 위한 토큰을 발급받습니다.

    Returns:
        access_token
    """
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(scienceon_client.get_access_token())
        loop.close()
        return result
    except Exception as e:
        logger.error(f"ScienceON 토큰 발급 실패: {e}")
        return ""

def search_scienceon_papers(query: str, limit: int = 10) -> list:
    """
    ScienceON에서 논문을 검색합니다.

    Args:
        query: 검색 쿼리
        limit: 반환할 최대 결과 수

    Returns:
        검색된 논문들의 리스트
    """
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(scienceon_client.search_by_keywords([query], limit))
        loop.close()
        return result
    except Exception as e:
        logger.error(f"ScienceON 논문 검색 실패: {e}")
        return []

def get_scienceon_paper_details(paper_cn: str) -> dict:
    """
    ScienceON에서 특정 논문의 상세 정보를 가져옵니다.

    Args:
        paper_cn: 논문 CN (제어번호)

    Returns:
        논문의 상세 정보 (인용 정보, 참고문헌 등 포함)
    """
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(scienceon_client.get_paper_details(paper_cn))
        loop.close()
        return result
    except Exception as e:
        logger.error(f"ScienceON 논문 상세 조회 실패: {e}")
        return {"error": str(e)}

def generate_text_embedding(text: str) -> List[float]:
    """
    한국어/영어 텍스트를 다국어 임베딩으로 변환합니다.

    Args:
        text: 임베딩할 텍스트

    Returns:
        임베딩 벡터
    """
    try:
        if not text or text.strip() == "":
            return [0.0] * 768  # 빈 텍스트의 경우 영벡터 반환

        embedding = embedding_model.encode(text)
        return embedding.tolist()
    except Exception as e:
        logger.error(f"임베딩 생성 실패: {e}")
        return [0.0] * 768

def calculate_similarity_score(embedding1: List[float], embedding2: List[float]) -> float:
    """
    두 임베딩 벡터 간의 코사인 유사도를 계산합니다.

    Args:
        embedding1: 첫 번째 임베딩 벡터
        embedding2: 두 번째 임베딩 벡터

    Returns:
        0.0 ~ 1.0 사이의 유사도 점수
    """
    try:
        if len(embedding1) != len(embedding2):
            logger.warning("임베딩 벡터 크기가 다릅니다.")
            return 0.0

        # numpy 배열로 변환
        emb1 = np.array(embedding1).reshape(1, -1)
        emb2 = np.array(embedding2).reshape(1, -1)

        # 코사인 유사도 계산
        similarity = cosine_similarity(emb1, emb2)[0][0]

        # 0-1 범위로 정규화 (코사인 유사도는 -1~1 범위)
        normalized_similarity = (similarity + 1) / 2

        return float(normalized_similarity)
    except Exception as e:
        logger.error(f"유사도 계산 실패: {e}")
        return 0.0

def generate_korean_recommendation_reason(source_data: dict, candidate_data: dict, similarity_score: float) -> str:
    """
    한국어로 추천 이유를 생성합니다.

    Args:
        source_data: 소스 데이터셋 정보
        candidate_data: 후보 데이터/논문 정보
        similarity_score: 유사도 점수

    Returns:
        한국어 추천 이유 텍스트
    """
    try:
        reasons = []

        # 키워드 유사도 분석
        source_keywords = set(source_data.get('keywords', []))
        candidate_keywords = set(candidate_data.get('keywords', []) if isinstance(candidate_data.get('keywords'), list)
                                 else candidate_data.get('keywords', '').split(','))

        common_keywords = source_keywords.intersection(candidate_keywords)
        if common_keywords:
            keywords_text = ', '.join(list(common_keywords)[:3])  # 최대 3개만 표시
            reasons.append(f"공통 키워드 '{keywords_text}'로 높은 연관성")

        # 분류 유사도
        source_classification = source_data.get('classification_ko', '') or source_data.get('classification_en', '')
        candidate_classification = candidate_data.get('classification', '') or candidate_data.get('journal', '')

        if source_classification and candidate_classification:
            if source_classification in candidate_classification or candidate_classification in source_classification:
                reasons.append(f"동일 연구 분야({source_classification}) 소속")

        # 시간적 연관성
        source_year = source_data.get('pub_year', '')
        candidate_year = candidate_data.get('pub_year', '')

        if source_year and candidate_year:
            try:
                year_diff = abs(int(source_year) - int(candidate_year))
                if year_diff <= 3:
                    reasons.append("최근 연구로 시의성 높음")
                elif year_diff <= 5:
                    reasons.append("관련 시기의 연구")
            except ValueError:
                pass

        # 기관 연관성
        source_org = source_data.get('organization', '')
        candidate_org = candidate_data.get('affiliation', '') or candidate_data.get('publisher', '')

        if source_org and candidate_org:
            # 대학교, 연구소 등 공통 기관명 확인
            source_orgs = set(source_org.split())
            candidate_orgs = set(candidate_org.split())
            common_orgs = source_orgs.intersection(candidate_orgs)

            if common_orgs:
                reasons.append("관련 연구기관 소속으로 방법론 유사")

        # 유사도 점수 기반 평가
        if similarity_score >= 0.8:
            reasons.append("의미적 유사도 매우 높음")
        elif similarity_score >= 0.7:
            reasons.append("의미적 유사도 높음")
        elif similarity_score >= 0.6:
            reasons.append("의미적 연관성 존재")

        # 인용 정보 (논문의 경우)
        if candidate_data.get('citation_info'):
            citation_count = candidate_data['citation_info'].get('citation_count', 0)
            if citation_count > 10:
                reasons.append("높은 인용도의 주요 연구")

        # 추천 이유 조합
        if reasons:
            return '; '.join(reasons)
        else:
            return "주제 영역의 의미적 연관성"

    except Exception as e:
        logger.error(f"추천 이유 생성 실패: {e}")
        return "관련 연구로 판단됨"

def determine_recommendation_level(similarity_score: float, citation_score: float = 0.0) -> str:
    """
    추천 레벨을 결정합니다 (강추/추천/참고).

    Args:
        similarity_score: 유사도 점수
        citation_score: 인용 점수 (논문의 경우)

    Returns:
        추천 레벨 ("강추", "추천", "참고")
    """
    try:
        # 복합 점수 계산 (유사도 70%, 인용도 30%)
        combined_score = similarity_score * 0.7 + citation_score * 0.3

        if combined_score >= 0.8:
            return "강추"
        elif combined_score >= 0.65:
            return "추천"
        else:
            return "참고"

    except Exception as e:
        logger.error(f"추천 레벨 결정 실패: {e}")
        return "참고"

def extract_keywords_from_text(text: str, max_keywords: int = 5) -> List[str]:
    """
    텍스트에서 주요 키워드를 추출합니다.

    Args:
        text: 분석할 텍스트
        max_keywords: 최대 키워드 수

    Returns:
        추출된 키워드 리스트
    """
    try:
        if not text:
            return []

        # 간단한 키워드 추출 (실제로는 더 정교한 NLP 기법 사용 가능)
        import re

        # 한국어와 영어 단어 추출
        korean_words = re.findall(r'[가-힣]+', text)
        english_words = re.findall(r'[A-Za-z]+', text)

        # 길이 2자 이상인 단어만 선택
        keywords = [word for word in korean_words + english_words if len(word) >= 2]

        # 빈도수 기반 정렬 (간단한 구현)
        from collections import Counter
        word_counts = Counter(keywords)

        return [word for word, count in word_counts.most_common(max_keywords)]

    except Exception as e:
        logger.error(f"키워드 추출 실패: {e}")
        return []

def calculate_citation_importance(citation_info: dict) -> float:
    """
    인용 정보를 바탕으로 중요도 점수를 계산합니다.

    Args:
        citation_info: 인용 정보 딕셔너리

    Returns:
        0.0 ~ 1.0 사이의 중요도 점수
    """
    try:
        citation_count = citation_info.get('citation_count', 0)
        reference_count = citation_info.get('reference_count', 0)

        # 인용 수 기반 점수 (로그 스케일 적용)
        if citation_count > 0:
            citation_score = min(math.log(citation_count + 1) / math.log(100), 1.0)
        else:
            citation_score = 0.0

        # 참고문헌 수 기반 점수 (연구의 포괄성 지표)
        reference_score = min(reference_count / 50, 0.3)  # 최대 0.3점

        return citation_score + reference_score

    except Exception as e:
        logger.error(f"인용 중요도 계산 실패: {e}")
        return 0.0

# ===== BM25 + 임베딩 하이브리드 유사도 =====

def tokenize_korean_text(text: str) -> List[str]:
    """
    한국어 텍스트를 토큰화합니다 (간단한 구현)

    Args:
        text: 토큰화할 텍스트

    Returns:
        토큰 리스트
    """
    try:
        import re
        # 한국어 단어, 영어 단어, 숫자 추출
        tokens = []
        tokens.extend(re.findall(r'[가-힣]+', text))
        tokens.extend(re.findall(r'[A-Za-z]+', text.lower()))
        tokens.extend(re.findall(r'\d+', text))

        # 2글자 이상만 선택
        return [token for token in tokens if len(token) >= 2]
    except Exception as e:
        logger.error(f"토큰화 실패: {e}")
        return []

def calculate_bm25_score(query_text: str, document_text: str, k1: float = 1.2, b: float = 0.75) -> float:
    """
    BM25 점수를 계산합니다.

    Args:
        query_text: 질의 텍스트
        document_text: 문서 텍스트
        k1: BM25 파라미터 k1
        b: BM25 파라미터 b

    Returns:
        BM25 점수
    """
    try:
        # 텍스트 토큰화
        query_tokens = tokenize_korean_text(query_text)
        doc_tokens = tokenize_korean_text(document_text)

        if not query_tokens or not doc_tokens:
            return 0.0

        # 문서 길이
        doc_length = len(doc_tokens)
        if doc_length == 0:
            return 0.0

        # 단순화된 BM25 계산 (전체 코퍼스 통계 없이)
        # 실제로는 전체 문서 집합의 통계가 필요하지만, 여기서는 개별 문서 대 문서로 계산
        doc_token_counts = Counter(doc_tokens)
        avgdl = doc_length  # 평균 문서 길이를 현재 문서 길이로 근사

        bm25_score = 0.0

        for query_token in set(query_tokens):
            if query_token in doc_token_counts:
                tf = doc_token_counts[query_token]

                # IDF는 단순화 (실제로는 전체 코퍼스 필요)
                idf = 1.0  # 단순화된 IDF

                # BM25 공식
                score = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_length / avgdl)))
                bm25_score += score

        # 0-1 범위로 정규화
        normalized_score = min(bm25_score / len(set(query_tokens)), 1.0)
        return max(normalized_score, 0.0)

    except Exception as e:
        logger.error(f"BM25 점수 계산 실패: {e}")
        return 0.0

def calculate_hybrid_similarity(source_text: str, candidate_text: str) -> Dict[str, float]:
    """
    BM25 + 임베딩 하이브리드 유사도를 계산합니다 (Hybrid RAG 방식)

    Args:
        source_text: 소스 텍스트
        candidate_text: 후보 텍스트

    Returns:
        각종 유사도 점수와 최종 점수가 포함된 딕셔너리
    """
    try:
        # 1. 임베딩 기반 의미적 유사도 (Dense Retrieval)
        source_embedding = generate_text_embedding(source_text)
        candidate_embedding = generate_text_embedding(candidate_text)
        semantic_score = calculate_similarity_score(source_embedding, candidate_embedding)

        # 2. BM25 기반 어휘적 유사도 (Sparse Retrieval)
        lexical_score = calculate_bm25_score(source_text, candidate_text)

        # 3. 하이브리드 점수 계산
        # 일반적으로 의미적 유사도에 더 높은 가중치 부여
        alpha = 0.7  # 임베딩 가중치
        beta = 0.3   # BM25 가중치

        hybrid_score = alpha * semantic_score + beta * lexical_score

        # 4. 공통 키워드 보너스 (추가적인 신호)
        source_tokens = set(tokenize_korean_text(source_text))
        candidate_tokens = set(tokenize_korean_text(candidate_text))

        common_tokens = source_tokens.intersection(candidate_tokens)
        keyword_bonus = min(len(common_tokens) / max(len(source_tokens), 1), 0.2)  # 최대 0.2 보너스

        final_score = min(hybrid_score + keyword_bonus, 1.0)

        return {
            'semantic_score': semantic_score,
            'lexical_score': lexical_score,
            'keyword_bonus': keyword_bonus,
            'hybrid_score': hybrid_score,
            'final_score': final_score,
            'common_keywords': list(common_tokens)[:5]
        }

    except Exception as e:
        logger.error(f"하이브리드 유사도 계산 실패: {e}")
        return {
            'semantic_score': 0.0,
            'lexical_score': 0.0,
            'keyword_bonus': 0.0,
            'hybrid_score': 0.0,
            'final_score': 0.0,
            'common_keywords': []
        }

def generate_hybrid_recommendation_reason(
    source_data: dict,
    candidate_data: dict,
    similarity_result: Dict[str, Any]
) -> str:
    """
    하이브리드 유사도 분석 결과를 바탕으로 구체적인 추천 이유를 생성합니다.

    Args:
        source_data: 소스 데이터셋 정보
        candidate_data: 후보 데이터/논문 정보
        similarity_result: calculate_hybrid_similarity 결과

    Returns:
        구체적인 추천 이유 텍스트
    """
    try:
        reasons = []

        semantic_score = similarity_result.get('semantic_score', 0.0)
        lexical_score = similarity_result.get('lexical_score', 0.0)
        common_keywords = similarity_result.get('common_keywords', [])
        final_score = similarity_result.get('final_score', 0.0)

        # 의미적 유사도 기반 이유
        if semantic_score >= 0.8:
            reasons.append("의미적 유사도 매우 높음")
        elif semantic_score >= 0.7:
            reasons.append("주제적 연관성 높음")

        # 어휘적 유사도 기반 이유
        if lexical_score >= 0.6:
            reasons.append("핵심 용어 일치도 높음")
        elif lexical_score >= 0.4:
            reasons.append("관련 용어 공유")

        # 공통 키워드 언급
        if common_keywords:
            keywords_text = ', '.join(common_keywords[:3])
            reasons.append(f"공통 키워드 '{keywords_text}' 확인")

        # 전체적인 유사도 평가
        if final_score >= 0.8:
            reasons.append("종합적 연관성 매우 높음")
        elif final_score >= 0.7:
            reasons.append("다면적 분석결과 높은 관련성")

        # 기존 메타데이터 기반 이유도 포함
        source_keywords = set(source_data.get('keywords', []))
        candidate_keywords = set(candidate_data.get('keywords', []) if isinstance(candidate_data.get('keywords'), list)
                                else candidate_data.get('keywords', '').split(','))

        metadata_overlap = source_keywords.intersection(candidate_keywords)
        if metadata_overlap:
            meta_keywords = ', '.join(list(metadata_overlap)[:2])
            reasons.append(f"메타데이터 키워드 '{meta_keywords}' 일치")

        return '; '.join(reasons[:4]) if reasons else "하이브리드 분석을 통한 관련 연구"

    except Exception as e:
        logger.error(f"하이브리드 추천 이유 생성 실패: {e}")
        return "의미적·어휘적 분석을 통한 관련성 확인"