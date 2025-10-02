import asyncio
import json
import time
import os
from typing import Dict, List, Any
from config.settings import settings
from tools.research_tools import *
import logging

logger = logging.getLogger(__name__)

class KoreanResearchRecommendationAgent:
    def __init__(self):
        # 개발 모드 감지 (환경변수나 GPU 없음)
        self.dev_mode = (
            os.getenv("DEV_MODE", "false").lower() == "true" or
            not self._check_gpu_available() or
            not self._check_model_requirements()
        )

        if self.dev_mode:
            logger.info("🎭 개발 모드로 실행: Mock 모델 사용")
            from models.mock_model import MockSolarModel
            self.solar_model = MockSolarModel()
        else:
            logger.info("🚀 프로덕션 모드로 실행: 실제 SOLAR 모델 사용")
            from models.solar_model import SolarModel
            self.solar_model = SolarModel()

        self.max_candidates = 20
        self.final_recommendations = 5

    def _check_gpu_available(self) -> bool:
        """GPU 사용 가능 여부 확인"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _check_model_requirements(self) -> bool:
        """모델 로딩에 필요한 요구사항 확인"""
        try:
            # 최소 메모리 요구사항 확인 (예: 16GB)
            import psutil
            available_memory = psutil.virtual_memory().available / (1024**3)  # GB
            return available_memory > 16
        except ImportError:
            # psutil이 없으면 그냥 True 반환
            return True

    async def recommend(self, dataset_id: str) -> Dict[str, Any]:
        """
        메인 추천 함수

        Args:
            dataset_id: DataON 데이터셋 ID

        Returns:
            추천 결과 딕셔너리
        """
        start_time = time.time()

        try:
            logger.info(f"추천 프로세스 시작: 데이터셋 ID {dataset_id}")

            # 1단계: 소스 데이터셋 정보 조회
            source_data = await self._get_source_data(dataset_id)
            if 'error' in source_data:
                return {"error": f"소스 데이터셋 조회 실패: {source_data['error']}", "recommendations": []}

            # 2단계: 후보 수집
            candidates = await self._collect_candidates(source_data)
            logger.info(f"총 {len(candidates)}개 후보 수집 완료")

            # 3단계: 유사도 계산 및 순위 결정
            ranked_candidates = await self._rank_candidates(source_data, candidates)
            logger.info(f"상위 {len(ranked_candidates)}개 후보 순위 결정 완료")

            # 4단계: LLM을 사용한 최종 추천 생성
            final_recommendations = await self._generate_final_recommendations(
                source_data, ranked_candidates[:self.max_candidates]
            )

            processing_time = int((time.time() - start_time) * 1000)
            logger.info(f"추천 프로세스 완료: {processing_time}ms")

            return {
                "source_dataset": {
                    "id": dataset_id,
                    "title": source_data.get('title_ko', ''),
                    "description": source_data.get('description_ko', '')[:200] + "...",
                    "keywords": source_data.get('keywords', [])
                },
                "recommendations": final_recommendations,
                "processing_time_ms": processing_time,
                "candidates_analyzed": len(candidates),
                "model_info": self.solar_model.get_model_info()
            }

        except Exception as e:
            logger.error(f"추천 프로세스 실패: {e}")
            return {
                "error": f"추천 생성 중 오류 발생: {str(e)}",
                "recommendations": [],
                "processing_time_ms": int((time.time() - start_time) * 1000)
            }

    async def _get_source_data(self, dataset_id: str) -> Dict[str, Any]:
        """1단계: 소스 데이터셋 정보 조회"""
        try:
            source_data = await get_dataon_dataset_metadata(dataset_id)
            logger.info(f"소스 데이터셋 정보 조회 완료: {source_data.get('title_ko', '')}")
            return source_data
        except Exception as e:
            logger.error(f"소스 데이터셋 조회 실패: {e}")
            return {"error": str(e)}

    async def _collect_candidates(self, source_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """2단계: 후보 수집 (DataON + ScienceON)"""
        candidates = []

        try:
            # 키워드 추출
            keywords = source_data.get('keywords', [])
            if not keywords:
                # 제목과 설명에서 키워드 추출
                text = f"{source_data.get('title_ko', '')} {source_data.get('description_ko', '')}"
                keywords = extract_keywords_from_text(text)

            logger.info(f"검색 키워드: {keywords}")

            # 병렬로 후보 수집
            tasks = [
                self._search_similar_datasets(keywords),
                self._search_related_papers(keywords)
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # DataON 데이터셋 후보
            if not isinstance(results[0], Exception):
                for dataset in results[0]:
                    candidates.append({
                        'type': 'dataset',
                        'source': 'dataon',
                        'data': dataset,
                        'title': dataset.get('title_ko', ''),
                        'description': dataset.get('description_ko', ''),
                        'keywords': dataset.get('keywords', []),
                        'url': dataset.get('url', ''),
                        'combined_text': dataset.get('combined_text', '')
                    })

            # ScienceON 논문 후보
            if not isinstance(results[1], Exception):
                for paper in results[1]:
                    candidates.append({
                        'type': 'paper',
                        'source': 'scienceon',
                        'data': paper,
                        'title': paper.get('title', ''),
                        'description': paper.get('abstract', '')[:200] if paper.get('abstract') else '',
                        'keywords': paper.get('keywords', '').split(',') if paper.get('keywords') else [],
                        'url': paper.get('content_url', ''),
                        'combined_text': paper.get('combined_text', ''),
                        'cn': paper.get('cn', '')
                    })

            return candidates

        except Exception as e:
            logger.error(f"후보 수집 실패: {e}")
            return []

    async def _search_similar_datasets(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """DataON에서 유사한 데이터셋 검색"""
        try:
            return await search_similar_dataon_datasets(keywords, limit=15)
        except Exception as e:
            logger.error(f"DataON 검색 실패: {e}")
            return []

    async def _search_related_papers(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """ScienceON에서 관련 논문 검색"""
        try:
            papers = []
            # 키워드별로 검색하여 다양성 확보
            for keyword in keywords[:3]:  # 상위 3개 키워드만 사용
                keyword_papers = await search_scienceon_papers(keyword, limit=5)
                papers.extend(keyword_papers)

            # 중복 제거 (CN 기준)
            seen_cns = set()
            unique_papers = []
            for paper in papers:
                cn = paper.get('cn', '')
                if cn and cn not in seen_cns:
                    seen_cns.add(cn)
                    unique_papers.append(paper)

            return unique_papers[:15]  # 최대 15개

        except Exception as e:
            logger.error(f"ScienceON 검색 실패: {e}")
            return []

    async def _rank_candidates(self, source_data: Dict[str, Any], candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """3단계: 하이브리드 유사도 계산 및 후보 순위 결정"""
        try:
            source_text = source_data.get('combined_text', '')

            # 각 후보의 하이브리드 유사도 계산
            scored_candidates = []
            for candidate in candidates:
                try:
                    candidate_text = candidate.get('combined_text', '')

                    # 🚀 BM25 + 임베딩 하이브리드 유사도 계산
                    from tools.research_tools import calculate_hybrid_similarity
                    hybrid_result = calculate_hybrid_similarity(source_text, candidate_text)

                    similarity_score = hybrid_result['final_score']  # 하이브리드 최종 점수 사용

                    # 논문의 경우 상세 정보로 인용도 계산
                    citation_score = 0.0
                    if candidate['type'] == 'paper' and candidate.get('cn'):
                        paper_details = get_scienceon_paper_details(candidate['cn'])
                        if 'citation_info' in paper_details:
                            citation_score = calculate_citation_importance(paper_details['citation_info'])
                            candidate['citation_info'] = paper_details['citation_info']

                    # 복합 점수 계산 (하이브리드 유사도 + 인용도)
                    final_score = similarity_score * 0.7 + citation_score * 0.3

                    candidate.update({
                        'similarity_score': similarity_score,
                        'semantic_score': hybrid_result['semantic_score'],
                        'lexical_score': hybrid_result['lexical_score'],
                        'citation_score': citation_score,
                        'final_score': final_score,
                        'common_keywords': hybrid_result.get('common_keywords', [])
                    })

                    scored_candidates.append(candidate)

                except Exception as e:
                    logger.warning(f"후보 점수 계산 실패: {e}")
                    continue

            # 점수순으로 정렬
            ranked_candidates = sorted(scored_candidates, key=lambda x: x['final_score'], reverse=True)
            return ranked_candidates

        except Exception as e:
            logger.error(f"후보 순위 결정 실패: {e}")
            return candidates

    async def _generate_final_recommendations(self, source_data: Dict[str, Any], top_candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """4단계: SOLAR 모델을 사용한 최종 추천 생성"""
        try:
            # LLM용 컨텍스트 준비
            context = {
                'source_title': source_data.get('title_ko', ''),
                'source_description': source_data.get('description_ko', ''),
                'source_keywords': ', '.join(source_data.get('keywords', [])),
                'source_classification': source_data.get('classification_ko', ''),
                'candidates': top_candidates
            }

            # SOLAR 모델용 프롬프트 생성
            task_description = f"""
주어진 소스 데이터셋과 관련된 연구논문과 데이터셋을 {self.final_recommendations}개 추천해주세요.
각 추천에 대해 구체적이고 논리적인 근거를 제시하고, 추천 수준을 결정해주세요.
"""

            prompt = self.solar_model.create_korean_prompt(task_description, context)

            # LLM 호출
            response = await self.solar_model.generate(prompt)

            # JSON 응답 파싱
            recommendations = self._parse_llm_response(response, top_candidates)

            return recommendations

        except Exception as e:
            logger.error(f"최종 추천 생성 실패: {e}")
            # 폴백: 규칙 기반 추천
            return self._generate_fallback_recommendations(source_data, top_candidates)

    def _parse_llm_response(self, response: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """LLM 응답을 파싱하여 추천 결과 생성"""
        try:
            # JSON 부분 추출
            import re
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                parsed_response = json.loads(json_str)

                if 'recommendations' in parsed_response:
                    llm_recommendations = parsed_response['recommendations']

                    # LLM 응답과 후보 데이터 매칭
                    final_recommendations = []
                    for rec in llm_recommendations:
                        candidate_number = rec.get('candidate_number', 0)
                        if 1 <= candidate_number <= len(candidates):
                            candidate = candidates[candidate_number - 1]

                            # LLM 추천 정보와 후보 데이터 결합
                            final_rec = {
                                "type": candidate['type'],
                                "title": candidate['title'],
                                "description": candidate['description'][:200] + "..." if len(candidate.get('description', '')) > 200 else candidate.get('description', ''),
                                "score": rec.get('score', candidate.get('final_score', 0.5)),
                                "reason": rec.get('reason', '추천 이유 생성 실패'),  # LLM이 생성한 이유
                                "level": rec.get('level', '참고'),  # LLM이 결정한 레벨
                                "url": candidate['url']
                            }
                            final_recommendations.append(final_rec)

                    if final_recommendations:
                        logger.info(f"LLM 추천 생성 성공: {len(final_recommendations)}개")
                        return final_recommendations

            # JSON 파싱 실패시 폴백
            logger.warning("LLM 응답 파싱 실패, 폴백 모드로 전환")
            return self._generate_fallback_recommendations({}, candidates)

        except Exception as e:
            logger.error(f"LLM 응답 파싱 실패: {e}")
            return self._generate_fallback_recommendations({}, candidates)

    def _generate_fallback_recommendations(self, source_data: Dict[str, Any], candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        폴백: 간단한 점수 기반 추천 생성
        LLM 실패 시에만 사용 (추천 이유는 간략하게)
        """
        recommendations = []

        for candidate in candidates[:self.final_recommendations]:
            try:
                score = candidate.get('final_score', 0.5)

                # 간단한 레벨 결정
                if score >= 0.8:
                    level = "강추"
                elif score >= 0.65:
                    level = "추천"
                else:
                    level = "참고"

                # 간단한 추천 이유
                reason = f"유사도 점수 {score:.2f} (의미적 {candidate.get('semantic_score', 0.0):.2f}, 어휘적 {candidate.get('lexical_score', 0.0):.2f})"

                recommendation = {
                    "type": candidate['type'],
                    "title": candidate['title'],
                    "description": candidate['description'][:100] + "..." if len(candidate.get('description', '')) > 100 else candidate.get('description', ''),
                    "score": round(score, 2),
                    "reason": reason,
                    "level": level,
                    "url": candidate['url']
                }
                recommendations.append(recommendation)

            except Exception as e:
                logger.warning(f"폴백 추천 생성 실패: {e}")
                continue

        return recommendations