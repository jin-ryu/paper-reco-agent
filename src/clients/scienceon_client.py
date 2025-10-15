import httpx
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from src.config.settings import settings
from src.tools.utils import clean_text
import logging
import asyncio
import re

logger = logging.getLogger(__name__)

class ScienceONClient:
    def __init__(self):
        self.base_url = settings.SCIENCEON_BASE_URL
        self.client_id = settings.SCIENCEON_CLIENT_ID
        self.accounts = settings.SCIENCEON_ACCOUNTS
        self.access_token: Optional[str] = None
        self.token_expires: Optional[datetime] = None
        self.timeout = httpx.Timeout(30.0)

    async def get_access_token(self) -> str:
        """
        ScienceON API 접근 토큰을 발급받습니다.

        Returns:
            access_token
        """
        if self.access_token and self.token_expires and datetime.now() < self.token_expires:
            return self.access_token

        url = f"{self.base_url}/tokenrequest.do"
        params = {
            "accounts": self.accounts,
            "client_id": self.client_id
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()

                data = response.json()
                self.access_token = data["access_token"]

                # 토큰 만료 시간 설정 (응답에서 받은 시간보다 10분 일찍 만료 처리)
                expire_str = data["access_token_expire"]
                expire_time = datetime.strptime(expire_str, "%Y-%m-%d %H:%M:%S.%f")
                self.token_expires = expire_time - timedelta(minutes=10)

                logger.info("Successfully obtained ScienceON access token")
                return self.access_token

        except httpx.HTTPError as e:
            logger.error(f"HTTP error getting access token: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error getting access token: {e}")
            raise

    async def search_papers(self, query: str, cur_page: int = 1, row_count: int = 10) -> List[Dict]:
        """
        ScienceON에서 논문을 검색합니다.

        Args:
            query: 검색 쿼리
            cur_page: 페이지 번호
            row_count: 페이지당 결과 수

        Returns:
            검색된 논문 리스트
        """
        token = await self.get_access_token()

        url = f"{self.base_url}/openapicall.do"
        params = {
            "client_id": self.client_id,
            "token": token,
            "action": "search",
            "target": "ARTI",
            "searchQuery": f'{{"BI":"{query}"}}',
            "curPage": cur_page,
            "rowCount": row_count
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()

                # XML 응답 파싱
                root = ET.fromstring(response.content)
                papers = []

                for record in root.findall('.//record'):
                    paper_data = self._parse_paper_record(record)
                    papers.append(paper_data)

                logger.info(f"Search completed: {len(papers)} papers found for '{query}'")
                return papers

        except httpx.HTTPError as e:
            logger.error(f"HTTP error searching papers with query '{query}': {e}")
            raise
        except ET.ParseError as e:
            logger.error(f"XML parsing error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error searching papers with query '{query}': {e}")
            raise

    async def get_paper_details(self, cn: str) -> Dict:
        """
        특정 논문의 상세 정보를 가져옵니다.

        Args:
            cn: 논문 제어번호

        Returns:
            논문 상세 정보
        """
        token = await self.get_access_token()

        url = f"{self.base_url}/openapicall.do"
        params = {
            "client_id": self.client_id,
            "token": token,
            "version": "1.0",
            "action": "browse",
            "target": "ARTI",
            "cn": cn
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()

                # XML 응답 파싱
                root = ET.fromstring(response.content)
                record = root.find('.//record')

                if record is not None:
                    paper_data = self._parse_detailed_paper_record(record)
                    logger.info(f"Successfully retrieved details for paper {cn}")
                    return paper_data
                else:
                    raise ValueError(f"No paper found with CN {cn}")

        except httpx.HTTPError as e:
            logger.error(f"HTTP error getting paper details for {cn}: {e}")
            raise
        except ET.ParseError as e:
            logger.error(f"XML parsing error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error getting paper details for {cn}: {e}")
            raise

    def _parse_paper_record(self, record: ET.Element) -> Dict:
        """
        논문 검색 결과 XML 레코드를 파싱합니다.
        """
        def get_item_value(meta_code: str) -> str:
            item = record.find(f'.//item[@metaCode="{meta_code}"]')
            if item is not None and item.text:
                return item.text.strip()
            return ""

        abstract = clean_text(get_item_value('Abstract'))

        keywords_str = get_item_value('Keyword')
        keywords = [kw.strip() for kw in re.split(r'[,;\s]', keywords_str) if kw.strip()]

        paper_data = {
            'cn': get_item_value('CN'),
            'title': get_item_value('Title'),
            'abstract': abstract,
            'authors': get_item_value('Author'),
            'affiliation': get_item_value('Affiliation'),
            'journal': get_item_value('JournalName'),
            'publisher': get_item_value('Publisher'),
            'pub_year': get_item_value('Pubyear'),
            'keywords': keywords,
            'doi': get_item_value('DOI'),
            'content_url': get_item_value('ContentURL'),
            'fulltext_url': get_item_value('FulltextURL'),
            'page_info': get_item_value('PageInfo'),
            'issn': get_item_value('ISSN'),
            'vol_no1': get_item_value('VolNo1'),
            'vol_no2': get_item_value('VolNo2'),
            'degree': get_item_value('Degree'),
            # 의미적 검색을 위한 결합된 텍스트
            'combined_text': self._create_combined_text_from_paper(
                get_item_value('Title'),
                abstract,
                get_item_value('Keyword')
            )
        }

        return paper_data

    def _parse_detailed_paper_record(self, record: ET.Element) -> Dict:
        """
        논문 상세 정보 XML 레코드를 파싱합니다 (인용 정보 포함).
        """
        basic_info = self._parse_paper_record(record)

        # 인용 정보 추가 파싱
        citing_papers = []
        for citing_group in record.findall('.//item[@metaGroupCode="CitingDocumentInfo"]'):
            citing_info = self._parse_citation_info(citing_group, "Citing")
            if citing_info:
                citing_papers.append(citing_info)

        cited_papers = []
        for cited_group in record.findall('.//item[@metaGroupCode="CitedDocumentInfo"]'):
            cited_info = self._parse_citation_info(cited_group, "Cited")
            if cited_info:
                cited_papers.append(cited_info)

        similar_papers = []
        for similar_group in record.findall('.//item[@metaGroupCode="SimilarDocumentInfo"]'):
            similar_info = self._parse_citation_info(similar_group, "Similar")
            if similar_info:
                similar_papers.append(similar_info)

        # 상세 정보와 인용 정보 결합
        detailed_info = {
            **basic_info,
            'citation_info': {
                'citing_papers': citing_papers,
                'cited_papers': cited_papers,
                'similar_papers': similar_papers,
                'citation_count': len(citing_papers),
                'reference_count': len(cited_papers)
            }
        }

        return detailed_info

    def _parse_citation_info(self, group: ET.Element, prefix: str) -> Optional[Dict]:
        """
        인용 정보 그룹을 파싱합니다.
        """
        def get_value(suffix: str) -> str:
            item = group.find(f'.//item[@metaCode="{prefix}{suffix}"]')
            if item is not None and item.text:
                return item.text.strip()
            return ""

        title = get_value('Title')
        if not title:
            return None

        return {
            'title': title,
            'cn': get_value('Cn'),
            'authors': get_value('Author'),
            'pub_year': get_value('Pubyear'),
            'doi': get_value('DOI'),
            'content_url': get_value('ContentURL'),
            'journal': get_value('JournalName') if prefix == "Citing" else ""
        }

    def _create_combined_text_from_paper(self, title: str, abstract: str, keywords: str) -> str:
        """
        논문의 제목, 초록, 키워드를 결합한 텍스트를 생성합니다.
        """
        parts = []
        if title:
            parts.append(title)
        if abstract:
            parts.append(abstract)
        if keywords:
            parts.append(keywords)
        return ' '.join(parts)

    async def search_by_keywords(self, keywords: List[str], limit: int = 10) -> List[Dict]:
        """
        키워드 리스트로 논문을 검색합니다.
        각 키워드별로 검색 후 결과를 합칩니다.

        Args:
            keywords: 검색할 키워드 리스트
            limit: 반환할 최대 결과 수

        Returns:
            검색된 논문 리스트 (중복 제거됨)
        """
        all_results = []
        seen_cns = set()  # 중복 제거용 (CN 기준)

        # 각 키워드당 검색 개수 (전체 limit의 절반씩)
        per_keyword_limit = max(5, limit // len(keywords)) if keywords else 10

        # 각 키워드별로 검색
        for keyword in keywords[:5]:  # 최대 5개 키워드만 사용
            try:
                results = await self.search_papers(keyword.strip(), row_count=per_keyword_limit)

                # 중복 제거하며 추가
                for paper in results:
                    cn = paper.get('cn', '')
                    if cn and cn not in seen_cns:
                        seen_cns.add(cn)
                        all_results.append(paper)

                        # limit 도달하면 종료
                        if len(all_results) >= limit:
                            return all_results[:limit]

            except Exception as e:
                logger.warning(f"키워드 '{keyword}' 검색 실패: {e}")
                continue

        return all_results[:limit]