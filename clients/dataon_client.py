import httpx
import asyncio
from typing import Dict, List, Optional
from config.settings import settings
import logging
import json

logger = logging.getLogger(__name__)

class DataONClient:
    def __init__(self):
        self.base_url = settings.DATAON_BASE_URL
        self.search_key = settings.DATAON_SEARCH_KEY
        self.meta_key = settings.DATAON_META_KEY
        self.timeout = httpx.Timeout(30.0)

    async def get_dataset_metadata(self, svc_id: str) -> Dict:
        """
        DataON API를 사용해서 특정 데이터셋의 메타데이터를 가져옵니다.

        Args:
            svc_id: DataON 서비스 ID

        Returns:
            데이터셋 메타데이터
        """
        url = f"{self.base_url}/rest/api/search/dataset/{svc_id}"
        params = {"key": self.meta_key}

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()

                data = response.json()
                logger.info(f"Successfully retrieved metadata for dataset {svc_id}")

                # 응답 데이터 구조 로깅
                logger.info(f"API Response:\n{json.dumps(data, indent=2, ensure_ascii=False)}")

                # 데이터 정제 및 구조화
                if 'records' in data:
                    # 단일 레코드 응답
                    dataset = data['records']
                    return self._process_dataset_metadata(dataset)
                elif 'result' in data and len(data['result']) > 0:
                    # 리스트 응답
                    dataset = data['result'][0]
                    return self._process_dataset_metadata(dataset)
                else:
                    raise ValueError(f"No data found for dataset {svc_id}")

        except httpx.HTTPError as e:
            logger.error(f"HTTP error getting dataset {svc_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error getting dataset {svc_id}: {e}")
            raise

    async def search_datasets(self, query: str, page: int = 1, size: int = 20) -> List[Dict]:
        """
        DataON에서 데이터셋을 검색합니다.

        Args:
            query: 검색 쿼리
            page: 페이지 번호
            size: 페이지 크기

        Returns:
            검색된 데이터셋 리스트
        """
        url = f"{self.base_url}/rest/api/search/dataset/"
        params = {
            "key": self.search_key,
            "query": query,
            "page": page,
            "size": size
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()

                data = response.json()
                logger.info(f"Search completed: {data.get('totalCount', 0)} results for '{query}'")

                datasets = []
                for item in data.get('result', []):
                    datasets.append(self._process_dataset_metadata(item))

                return datasets

        except httpx.HTTPError as e:
            logger.error(f"HTTP error searching datasets with query '{query}': {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error searching datasets with query '{query}': {e}")
            raise

    def _process_dataset_metadata(self, raw_data: Dict) -> Dict:
        """
        원시 DataON API 응답을 구조화된 메타데이터로 변환합니다.
        """
        # 한글 제목/설명 우선, 없으면 영어 사용
        title_ko = raw_data.get('dataset_title_kor', '') or raw_data.get('titl_nm', '')
        title_en = raw_data.get('dataset_title_etc_main', '') or raw_data.get('titl_nm_en', '')
        desc_ko = raw_data.get('dataset_expl_kor', '') or raw_data.get('desc_cn', '')
        desc_en = raw_data.get('dataset_expl_etc_main', '') or raw_data.get('desc_cn_en', '')

        # 키워드 처리 (다양한 형식 지원)
        keywords = []
        if raw_data.get('dataset_kywd_etc_main'):
            keywords = [k.strip() for k in raw_data['dataset_kywd_etc_main'].split(';') if k.strip()]
        elif raw_data.get('keywrd'):
            keywords = [k.strip() for k in raw_data['keywrd'].split(',') if k.strip()]

        return {
            'svc_id': raw_data.get('svc_id', ''),
            'title_ko': title_ko,
            'title_en': title_en,
            'description_ko': desc_ko,
            'description_en': desc_en,
            'keywords': keywords,
            'organization': raw_data.get('cltfm_kor', '') or raw_data.get('org_nm', ''),
            'classification_ko': raw_data.get('dataset_mnsb_pc', [''])[0] if isinstance(raw_data.get('dataset_mnsb_pc'), list) else raw_data.get('clsfctn_nm', ''),
            'classification_en': raw_data.get('clsfctn_nm_en', ''),
            'pub_year': raw_data.get('dataset_pub_dt_pc', '') or raw_data.get('pub_year', ''),
            'url': raw_data.get('dataset_lndgpg', '') or raw_data.get('url', ''),
            'doi': raw_data.get('dataset_doi', '') or raw_data.get('doi', ''),
            'data_format': raw_data.get('file_frmt_pc', [''])[0] if isinstance(raw_data.get('file_frmt_pc'), list) else raw_data.get('data_format', ''),
            'file_size': raw_data.get('file_size', ''),
            'download_count': raw_data.get('download_count', 0),
            # 의미적 검색을 위한 결합된 텍스트
            'combined_text': self._create_combined_text(raw_data),
            'combined_text_en': self._create_combined_text_en(raw_data)
        }

    def _create_combined_text(self, data: Dict) -> str:
        """한국어 통합 텍스트 생성"""
        parts = []

        # 제목
        title = data.get('dataset_title_kor', '') or data.get('titl_nm', '')
        if title:
            parts.append(title)

        # 설명
        desc = data.get('dataset_expl_kor', '') or data.get('desc_cn', '')
        if desc:
            parts.append(desc)

        # 키워드
        if data.get('dataset_kywd_kor'):
            parts.append(data['dataset_kywd_kor'])
        elif data.get('keywrd'):
            parts.append(data['keywrd'])

        # 조직
        org = data.get('cltfm_kor', '') or data.get('org_nm', '')
        if org:
            parts.append(org)

        return ' '.join(parts)

    def _create_combined_text_en(self, data: Dict) -> str:
        """영어 통합 텍스트 생성"""
        parts = []

        # 제목
        title = data.get('dataset_title_etc_main', '') or data.get('titl_nm_en', '')
        if title:
            parts.append(title)

        # 설명
        desc = data.get('dataset_expl_etc_main', '') or data.get('desc_cn_en', '')
        if desc:
            parts.append(desc)

        # 키워드
        if data.get('dataset_kywd_etc_main'):
            parts.append(data['dataset_kywd_etc_main'])

        # 분류
        if data.get('clsfctn_nm_en'):
            parts.append(data['clsfctn_nm_en'])

        return ' '.join(parts)

    async def search_by_keywords(self, keywords: List[str], limit: int = 10) -> List[Dict]:
        """
        키워드 리스트로 데이터셋을 검색합니다.

        Args:
            keywords: 검색할 키워드 리스트
            limit: 반환할 최대 결과 수

        Returns:
            검색된 데이터셋 리스트
        """
        # 키워드를 공백으로 연결하여 검색
        query = ' '.join(keywords)
        results = await self.search_datasets(query, size=limit)
        return results[:limit]