import httpx
import asyncio
from typing import Dict, List, Optional
from src.config.settings import settings
from src.utils.text_utils import clean_text
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
                logger.info(f"Search completed: {len(data.get('records', []))} datasets for '{query}'")

                datasets = []
                for item in data.get('records', []):
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
        # 제목 (한글 우선, 없으면 영문)
        title = raw_data.get('dataset_title_kor', '') or raw_data.get('titl_nm', '') or \
                raw_data.get('dataset_title_etc_main', '') or raw_data.get('titl_nm_en', '')

        # 설명 (한글 우선, 없으면 영문)
        description = raw_data.get('dataset_expl_kor', '') or raw_data.get('desc_cn', '') or \
                      raw_data.get('dataset_expl_etc_main', '') or raw_data.get('desc_cn_en', '')
        description = clean_text(description)

        # 키워드 처리 (중복 제거)
        keywords = set()
        if raw_data.get('dataset_kywd_kor'):
            keywords.update([k.strip() for k in raw_data['dataset_kywd_kor'].split(';') if k.strip()])
        if raw_data.get('dataset_kywd_etc_main'):
            keywords.update([k.strip() for k in raw_data['dataset_kywd_etc_main'].split(';') if k.strip()])
        if raw_data.get('keywrd'):
            keywords.update([k.strip() for k in raw_data['keywrd'].split(',') if k.strip()])
        
        processed_keywords = list(keywords)

        # Create a dictionary with the processed data
        processed_data = {
            'svc_id': raw_data.get('svc_id', ''),
            'title': title,
            'description': description,
            'keywords': processed_keywords,
            'organization': raw_data.get('cltfm_kor', '') or raw_data.get('org_nm', ''),
            'classification_ko': raw_data.get('dataset_mnsb_pc', [''])[0] if isinstance(raw_data.get('dataset_mnsb_pc'), list) else raw_data.get('clsfctn_nm', ''),
            'classification_en': raw_data.get('clsfctn_nm_en', ''),
            'pub_year': raw_data.get('dataset_pub_dt_pc', '') or raw_data.get('pub_year', ''),
            'url': raw_data.get('dataset_lndgpg', '') or raw_data.get('url', ''),
            'doi': raw_data.get('dataset_doi', '') or raw_data.get('doi', ''),
            'data_format': raw_data.get('file_frmt_pc', [''])[0] if isinstance(raw_data.get('file_frmt_pc'), list) else raw_data.get('data_format', ''),
            'file_size': raw_data.get('file_size', ''),
            'download_count': raw_data.get('download_count', 0),
        }
        
        # Add combined_text using the processed data
        processed_data['combined_text'] = self._create_combined_text(processed_data)

        return processed_data

    def _create_combined_text(self, data: Dict) -> str:
        """통합 텍스트 생성"""
        parts = []
        if data.get('title'):
            parts.append(data['title'])
        if data.get('description'):
            parts.append(data['description'])
        if data.get('keywords'):
            parts.append(' '.join(data['keywords']))
        return ' '.join(parts)

    async def search_by_keywords(self, keywords: List[str], limit: int = 10) -> List[Dict]:
        """
        키워드 리스트로 데이터셋을 검색합니다.
        각 키워드별로 검색 후 결과를 합칩니다.

        Args:
            keywords: 검색할 키워드 리스트
            limit: 반환할 최대 결과 수

        Returns:
            검색된 데이터셋 리스트 (중복 제거됨)
        """
        all_results = []
        seen_ids = set()  # 중복 제거용 (dataset_id 기준)

        # 각 키워드당 검색 개수
        per_keyword_limit = max(5, limit // len(keywords)) if keywords else 10

        # 각 키워드별로 검색
        for keyword in keywords:
            try:
                results = await self.search_datasets(keyword.strip(), size=per_keyword_limit)

                # 중복 제거하며 추가
                for dataset in results:
                    dataset_id = dataset.get('svc_id', '')
                    if dataset_id and dataset_id not in seen_ids:
                        seen_ids.add(dataset_id)
                        all_results.append(dataset)

                        # limit 도달하면 종료
                        if len(all_results) >= limit:
                            return all_results[:limit]

            except Exception as e:
                logger.warning(f"키워드 '{keyword}' 검색 실패: {e}")
                continue

        return all_results[:limit]