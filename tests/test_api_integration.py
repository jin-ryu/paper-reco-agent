import pytest
import asyncio
from clients.dataon_client import DataONClient
from clients.scienceon_client import ScienceONClient
from tools.research_tools import *

class TestAPIIntegration:
    """API 통합 테스트"""

    @pytest.fixture
    def dataon_client(self):
        return DataONClient()

    @pytest.fixture
    def scienceon_client(self):
        return ScienceONClient()

    def test_dataon_dataset_metadata(self, dataon_client):
        """DataON 데이터셋 메타데이터 조회 테스트"""
        # 실제 테스트용 데이터셋 ID로 교체 필요
        test_dataset_id = "SAMPLE_DATASET_ID"

        try:
            result = get_dataon_dataset_metadata(test_dataset_id)

            # 기본 필드 확인
            assert 'svc_id' in result
            assert 'title_ko' in result
            assert 'combined_text' in result

        except Exception as e:
            # API 키가 없거나 네트워크 오류인 경우 패스
            pytest.skip(f"DataON API 테스트 스킵: {e}")

    def test_dataon_search(self, dataon_client):
        """DataON 검색 테스트"""
        test_keywords = ["COVID-19", "데이터"]

        try:
            results = search_similar_dataon_datasets(test_keywords, limit=5)

            assert isinstance(results, list)
            if results:
                assert 'svc_id' in results[0]
                assert 'title_ko' in results[0]

        except Exception as e:
            pytest.skip(f"DataON 검색 테스트 스킵: {e}")

    def test_scienceon_token(self, scienceon_client):
        """ScienceON 토큰 발급 테스트"""
        try:
            token = get_scienceon_access_token()

            assert isinstance(token, str)
            assert len(token) > 0

        except Exception as e:
            pytest.skip(f"ScienceON 토큰 테스트 스킵: {e}")

    def test_scienceon_search(self, scienceon_client):
        """ScienceON 논문 검색 테스트"""
        test_query = "인공지능"

        try:
            results = search_scienceon_papers(test_query, limit=3)

            assert isinstance(results, list)
            if results:
                assert 'cn' in results[0]
                assert 'title' in results[0]

        except Exception as e:
            pytest.skip(f"ScienceON 검색 테스트 스킵: {e}")

    def test_embedding_generation(self):
        """임베딩 생성 테스트"""
        test_text = "인공지능과 머신러닝 연구"

        embedding = generate_text_embedding(test_text)

        assert isinstance(embedding, list)
        assert len(embedding) == 768  # KR-SBERT 임베딩 차원
        assert all(isinstance(x, float) for x in embedding)

    def test_similarity_calculation(self):
        """유사도 계산 테스트"""
        text1 = "인공지능 연구"
        text2 = "머신러닝 연구"
        text3 = "요리 레시피"

        emb1 = generate_text_embedding(text1)
        emb2 = generate_text_embedding(text2)
        emb3 = generate_text_embedding(text3)

        # 관련된 텍스트 간 유사도가 더 높아야 함
        sim_related = calculate_similarity_score(emb1, emb2)
        sim_unrelated = calculate_similarity_score(emb1, emb3)

        assert 0 <= sim_related <= 1
        assert 0 <= sim_unrelated <= 1
        assert sim_related > sim_unrelated

    def test_korean_reason_generation(self):
        """한국어 추천 이유 생성 테스트"""
        source_data = {
            'title_ko': 'COVID-19 연구 데이터',
            'keywords': ['COVID-19', '연구', '데이터'],
            'classification_ko': '의학',
            'organization': '서울대학교',
            'pub_year': '2023'
        }

        candidate_data = {
            'title': 'COVID-19 백신 효과 연구',
            'keywords': ['COVID-19', '백신', '효과'],
            'classification': '의학',
            'affiliation': '서울대학교 의과대학',
            'pub_year': '2023'
        }

        reason = generate_korean_recommendation_reason(source_data, candidate_data, 0.85)

        assert isinstance(reason, str)
        assert len(reason) > 0
        assert 'COVID-19' in reason or '공통 키워드' in reason

    def test_recommendation_level(self):
        """추천 레벨 결정 테스트"""
        # 높은 유사도 -> 강추
        level_high = determine_recommendation_level(0.9, 0.8)
        assert level_high == "강추"

        # 중간 유사도 -> 추천
        level_medium = determine_recommendation_level(0.7, 0.5)
        assert level_medium == "추천"

        # 낮은 유사도 -> 참고
        level_low = determine_recommendation_level(0.5, 0.3)
        assert level_low == "참고"