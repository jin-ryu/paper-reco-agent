import asyncio
import json
import time
import os
from typing import Dict, List, Any
from src.config.settings import settings
from src.tools.research_tools import *
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
            from src.models.mock_model import MockQwenModel
            self.llm_model = MockQwenModel()
        else:
            logger.info("🚀 프로덕션 모드로 실행: 실제 Qwen 모델 사용")
            from src.models.qwen_model import QwenModel
            self.llm_model = QwenModel()

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

            # 2단계: LLM으로 검색 쿼리 생성
            search_queries = await self._generate_search_queries(source_data)
            logger.info(f"검색 쿼리 생성 완료: 데이터셋({len(search_queries['dataset_queries'])}개), 논문({len(search_queries['paper_queries'])}개)")

            # 3단계: 후보 수집
            candidates = await self._collect_candidates_with_queries(search_queries)
            logger.info(f"총 {len(candidates)}개 후보 수집 완료")

            # 4단계: 유사도 계산 및 순위 결정
            ranked_candidates = await self._rank_candidates(source_data, candidates)
            logger.info(f"상위 {len(ranked_candidates)}개 후보 순위 결정 완료")

            # 5단계: LLM을 사용한 최종 추천 생성
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
                "model_info": self.llm_model.get_model_info()
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

    async def _generate_search_queries(self, source_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """2단계: LLM을 사용해서 최적의 검색 쿼리 생성"""
        try:
            # 제목과 설명 선택 (한글 우선, 없으면 영어)
            title = source_data.get('title_ko') or source_data.get('title_en', '')
            description = source_data.get('description_ko') or source_data.get('description_en', '')

            # API에서 가져온 원본 키워드
            original_keywords = source_data.get('keywords', [])
            keywords_str = ', '.join(original_keywords[:10]) if original_keywords else ''

            # 언어 설정 (dataset_main_lang_pc, dataset_sub_lang_pc 필드 참고)
            main_lang = source_data.get('dataset_main_lang_pc', 'Korean')
            sub_lang = source_data.get('dataset_sub_lang_pc', 'English')

            # 언어 매핑 (pc -> 실제 언어명)
            lang_map = {
                'KO': 'Korean',
                'EN': 'English',
                'JA': 'Japanese',
                'ZH': 'Chinese',
                'FR': 'French',
                'DE': 'German',
                'ES': 'Spanish',
                'RU': 'Russian'
            }
            main_lang = lang_map.get(main_lang, main_lang)
            sub_lang = lang_map.get(sub_lang, sub_lang)

            # 프롬프트 생성 (Qwen3-14B 최적화)
            prompt = f"""You are a research data search expert. Generate optimal search keywords to find related papers and datasets.

Input Dataset:
Title: {title}
Description: {description[:300]}...
Original Keywords (from API): {keywords_str}
Main Language: {main_lang}
Sub Language: {sub_lang}

Task:
1. Generate 3-5 NEW dataset search keywords (core topics, fields, data types)
2. Generate 3-5 NEW paper search keywords (research methods, theories, techniques)
3. Generate keywords in {main_lang} (main) and {sub_lang} (secondary)
4. DO NOT repeat the original keywords - only generate NEW ones

IMPORTANT: Output ONLY valid JSON. No thinking process, no explanations, no markdown code blocks.

Output this exact JSON structure:
{{"dataset_queries": ["new_kw1", "new_kw2", "new_kw3"], "paper_queries": ["paper_kw1", "paper_kw2", "paper_kw3"]}}"""

            # LLM 호출 (낮은 temperature로 일관된 응답 유도)
            response = await self.llm_model.generate(
                prompt,
                max_new_tokens=300,
                temperature=0.1
            )

            # JSON 파싱
            try:
                # <think> 태그 제거 (Qwen3 thinking 모드 출력)
                import re
                response_cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
                response_cleaned = response_cleaned.strip()

                # JSON 블록 추출 (```json ... ``` 또는 {...} 형식)
                json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_cleaned, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_match = re.search(r'\{.*\}', response_cleaned, re.DOTALL)
                    json_str = json_match.group(0) if json_match else response_cleaned

                queries = json.loads(json_str)

                # LLM이 생성한 새 키워드
                llm_dataset_queries = queries.get('dataset_queries', [])
                llm_paper_queries = queries.get('paper_queries', [])

                # 원본 키워드 + LLM 생성 키워드 병합 (중복 제거)
                def merge_and_dedupe(original: List[str], new_keywords: List[str], max_count: int = 10) -> List[str]:
                    """키워드 병합 및 중복 제거 (대소문자 무시)"""
                    result = []
                    seen = set()

                    # 원본 키워드 추가
                    for kw in original:
                        kw_clean = kw.strip()
                        kw_lower = kw_clean.lower()
                        if kw_clean and kw_lower not in seen:
                            result.append(kw_clean)
                            seen.add(kw_lower)

                    # 새 키워드 추가
                    for kw in new_keywords:
                        kw_clean = kw.strip()
                        kw_lower = kw_clean.lower()
                        if kw_clean and kw_lower not in seen:
                            result.append(kw_clean)
                            seen.add(kw_lower)

                    return result[:max_count]

                # 데이터셋/논문 쿼리 생성 (중복 제거됨)
                dataset_queries = merge_and_dedupe(original_keywords, llm_dataset_queries, 10)
                paper_queries = merge_and_dedupe(original_keywords, llm_paper_queries, 10)

                # 비어있으면 폴백
                if not dataset_queries or not paper_queries:
                    raise ValueError("Empty queries generated")

                logger.info(f"최종 키워드 - 데이터셋: {dataset_queries}")
                logger.info(f"최종 키워드 - 논문: {paper_queries}")
                logger.info(f"  (원본: {len(original_keywords)}개, LLM 추가: 데이터셋 {len(llm_dataset_queries)}개, 논문 {len(llm_paper_queries)}개)")
                return {
                    'dataset_queries': dataset_queries,
                    'paper_queries': paper_queries
                }

            except (json.JSONDecodeError, ValueError, AttributeError) as e:
                logger.warning(f"LLM 응답 파싱 실패, 폴백 사용: {e}")
                logger.warning(f"LLM 원본 응답:\n{response}")
                # 폴백: 기존 키워드 또는 추출
                fallback_keywords = source_data.get('keywords', [])
                if not fallback_keywords:
                    text = f"{title} {description}"
                    fallback_keywords = extract_keywords_from_text(text)

                return {
                    'dataset_queries': fallback_keywords[:5],
                    'paper_queries': fallback_keywords[:5]
                }

        except Exception as e:
            logger.error(f"검색 쿼리 생성 실패: {e}")
            # 폴백: 기존 키워드
            fallback_keywords = source_data.get('keywords', ['research', 'data'])[:5]
            return {
                'dataset_queries': fallback_keywords,
                'paper_queries': fallback_keywords
            }

    async def _collect_candidates_with_queries(self, search_queries: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """3단계: 생성된 쿼리로 후보 수집 (DataON + ScienceON)"""
        candidates = []

        try:
            # 병렬로 후보 수집
            tasks = [
                self._search_similar_datasets(search_queries['dataset_queries']),
                self._search_related_papers(search_queries['paper_queries'])
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

                    # 최종 점수는 하이브리드 유사도 그대로 사용
                    # (검색 API 응답에 이미 인용 정보 포함됨, 상세 조회 불필요)
                    final_score = similarity_score

                    candidate.update({
                        'similarity_score': similarity_score,
                        'semantic_score': hybrid_result['semantic_score'],
                        'lexical_score': hybrid_result['lexical_score'],
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
        """4단계: Qwen 모델을 사용한 최종 추천 생성"""
        try:
            # LLM용 컨텍스트 준비
            context = {
                'source_title': source_data.get('title_ko', ''),
                'source_description': source_data.get('description_ko', ''),
                'source_keywords': ', '.join(source_data.get('keywords', [])),
                'source_classification': source_data.get('classification_ko', ''),
                'candidates': top_candidates
            }

            # Qwen 모델용 프롬프트 생성
            task_description = f"""
주어진 소스 데이터셋과 관련된 연구논문과 데이터셋을 {self.final_recommendations}개 추천해주세요.
각 추천에 대해 구체적이고 논리적인 근거를 제시하고, 추천 수준을 결정해주세요.
"""

            prompt = self.llm_model.create_korean_prompt(task_description, context)

            # LLM 호출 (최대 2회 재시도)
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    logger.info(f"LLM 추천 생성 시도 {attempt + 1}/{max_retries}")
                    response = await self.llm_model.generate(prompt)

                    # JSON 응답 파싱
                    recommendations = self._parse_llm_response(response, top_candidates)

                    if recommendations:
                        return recommendations
                    else:
                        logger.warning(f"LLM 응답 파싱 실패 (시도 {attempt + 1}/{max_retries})")
                        if attempt < max_retries - 1:
                            # 프롬프트 수정하여 재시도
                            prompt = self._create_simplified_prompt(source_data, top_candidates)
                            logger.info("간소화된 프롬프트로 재시도")
                            continue
                        else:
                            logger.error("모든 재시도 실패, 추천 생성 불가")
                            return []

                except Exception as e:
                    logger.error(f"LLM 호출 실패 (시도 {attempt + 1}/{max_retries}): {e}")
                    if attempt == max_retries - 1:
                        return []

            return []

        except Exception as e:
            logger.error(f"최종 추천 생성 실패: {e}")
            return []

    def _parse_llm_response(self, response: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """LLM 응답을 파싱하여 추천 결과 생성"""
        try:
            import re

            # <think> 태그 제거 (Qwen3 thinking 모드 출력)
            # 빈 태그와 내용이 있는 태그 모두 제거
            response_cleaned = re.sub(r'<think>\s*</think>', '', response, flags=re.DOTALL)
            response_cleaned = re.sub(r'<think>.*?</think>', '', response_cleaned, flags=re.DOTALL)
            response_cleaned = response_cleaned.strip()

            # JSON 시작 부분까지 모든 텍스트 제거 (더 공격적)
            if '{' in response_cleaned:
                json_start_idx = response_cleaned.find('{')
                response_cleaned = response_cleaned[json_start_idx:]

            # 1. JSON 블록 추출 (```json ... ``` 또는 { ... })
            # 가장 바깥쪽 중괄호 쌍 찾기 (중첩 지원)
            json_match = re.search(r'```json\s*(\{.*\})\s*```', response_cleaned, re.DOTALL)
            if not json_match:
                # JSON 블록 마커 없이 찾기
                json_match = re.search(r'\{(?:[^{}]|(?:\{(?:[^{}]|\{[^{}]*\})*\}))*\}', response_cleaned, re.DOTALL)

            if json_match:
                json_str = json_match.group(1) if json_match.lastindex else json_match.group(0)

                # JSON 정리 (흔한 오류 수정)
                json_str = json_str.strip()

                # 주석 제거 (// ... 또는 ... // ...)
                json_str = re.sub(r'//.*?(?=\n|$)', '', json_str)
                json_str = re.sub(r',\s*\.\.\.', '', json_str)  # , ... 패턴 제거

                # 후행 쉼표 제거
                json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)

                # 단일 따옴표를 이중 따옴표로 (단, 문자열 내부 아닌 경우만)
                # json_str = json_str.replace("'", '"')  # 너무 공격적, 제거

                # 잘못된 배열 확장 표시 제거 (예: ["item1", ...])
                json_str = re.sub(r',\s*\.\.\.\s*\]', ']', json_str)

                logger.debug(f"정리된 JSON:\n{json_str[:500]}...")

                # 이쁘게 포맷팅해서 로깅
                try:
                    temp_parsed = json.loads(json_str)
                    logger.info(f"추출된 JSON:\n{json.dumps(temp_parsed, indent=2, ensure_ascii=False)}")
                except:
                    logger.info(f"추출된 JSON (raw):\n{json_str[:1000]}...")

                try:
                    parsed_response = json.loads(json_str)
                    logger.info(f"✅ JSON 파싱 성공")
                    logger.info(f"파싱된 타입: {type(parsed_response)}, 키: {parsed_response.keys() if isinstance(parsed_response, dict) else 'N/A'}")
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON 파싱 1차 실패: {e}, 수정 시도 중...")
                    logger.warning(f"실패한 JSON 앞부분:\n{json_str[:300]}")
                    # 더 공격적인 정리
                    json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)  # 제어 문자 제거
                    json_str = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', json_str)  # 잘못된 이스케이프 수정
                    # 마지막 시도: 유효하지 않은 문자 제거
                    json_str = re.sub(r'[^\x20-\x7E\u0080-\uFFFF{}[\]":,.\-+0-9]', '', json_str)
                    try:
                        parsed_response = json.loads(json_str)
                        logger.info(f"✅ JSON 파싱 성공 (2차 시도)")
                    except json.JSONDecodeError as e2:
                        logger.error(f"❌ JSON 파싱 2차 실패: {e2}")
                        logger.error(f"정리 후 JSON:\n{json_str[:300]}")
                        raise

                # parsed_response가 배열인 경우 처리
                if isinstance(parsed_response, list):
                    logger.warning(f"⚠️  응답이 배열입니다. 첫 번째 요소 사용")
                    if len(parsed_response) > 0:
                        parsed_response = parsed_response[0]

                if isinstance(parsed_response, dict) and 'recommendations' in parsed_response:
                    logger.info(f"recommendations 키 발견, {len(parsed_response['recommendations'])}개 항목")
                    llm_recommendations = parsed_response['recommendations']

                    # LLM 응답과 후보 데이터 매칭
                    final_recommendations = []
                    for idx, rec in enumerate(llm_recommendations[:self.final_recommendations]):
                        candidate_number = rec.get('candidate_number', 0)
                        logger.debug(f"처리 중: 순서={idx+1}, candidate_number={candidate_number}")

                        if 1 <= candidate_number <= len(candidates):
                            candidate = candidates[candidate_number - 1]

                            # LLM 추천 정보와 후보 데이터 결합
                            # rank는 추천 순서 (1부터 시작)
                            # title, type, score, url은 모두 candidates에서 가져옴
                            # reason, level만 LLM이 생성
                            final_rec = {
                                "rank": idx + 1,  # 추천 순서 (LLM이 반환한 순서)
                                "type": candidate['type'],
                                "title": candidate['title'],
                                "description": candidate['description'][:200] + "..." if len(candidate.get('description', '')) > 200 else candidate.get('description', ''),
                                "score": candidate.get('final_score', 0.5),  # E5 계산한 점수 사용
                                "reason": rec.get('reason', '추천 이유 생성 실패'),
                                "level": rec.get('level', '참고'),
                                "url": candidate['url']
                            }
                            final_recommendations.append(final_rec)
                            logger.debug(f"✅ 추천 항목 추가: {candidate['title'][:50]}")
                        else:
                            logger.warning(f"⚠️  잘못된 candidate_number: {candidate_number} (총 {len(candidates)}개)")

                    # 이미 순서대로 추가되었으므로 정렬 불필요
                    # final_recommendations.sort(key=lambda x: x['rank'])

                    if final_recommendations:
                        logger.info(f"✅ LLM 추천 생성 성공: {len(final_recommendations)}개")
                        return final_recommendations
                    else:
                        logger.warning(f"⚠️  매칭된 추천 항목 없음 (LLM 응답 {len(llm_recommendations)}개)")
                else:
                    logger.warning(f"⚠️  'recommendations' 키가 응답에 없음. 키 목록: {list(parsed_response.keys())}")

            # JSON 파싱 실패
            logger.warning(f"LLM 응답에서 JSON 추출 실패, 원문:\n{response[:500]}...")
            return []

        except Exception as e:
            logger.error(f"LLM 응답 파싱 실패: {e}")
            logger.error(f"LLM 응답:\n{response[:1000]}...")
            return []

    def _create_simplified_prompt(self, source_data: Dict[str, Any], candidates: List[Dict[str, Any]]) -> str:
        """
        간소화된 프롬프트 생성 (재시도용)
        더 명확하고 간단한 지시사항
        """
        # 상위 5개 후보만 사용
        top_5 = candidates[:5]

        candidates_text = ""
        for i, cand in enumerate(top_5, 1):
            candidates_text += f"\n[{i}] {cand.get('type', 'unknown')}: {cand.get('title', '')} (유사도: {cand.get('final_score', 0):.2f})\n"

        prompt = f"""Select 3-5 best recommendations and output as JSON ONLY. Do NOT use <think> tags.

Source Dataset: {source_data.get('title_ko') or source_data.get('title_en', '')}

Candidates:
{candidates_text}

Output this JSON structure exactly (start with '{{' character):
{{
  "recommendations": [
    {{"candidate_number": 1, "reason": "Very high similarity", "level": "강추"}},
    {{"candidate_number": 2, "reason": "High similarity", "level": "추천"}},
    {{"candidate_number": 3, "reason": "Related topic", "level": "참고"}}
  ]
}}

Rules:
- Only output: candidate_number (1-5), reason, level
- DO NOT output: rank, title, type, score
- Start your response with '{{' now:"""

        return prompt

    def _generate_fallback_recommendations(self, source_data: Dict[str, Any], candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        폴백: 간단한 점수 기반 추천 생성 (사용 안함 - 제거 예정)
        """
        logger.warning("폴백 함수 호출됨 - 이 함수는 더 이상 사용되지 않아야 함")
        recommendations = []

        for idx, candidate in enumerate(candidates[:self.final_recommendations], 1):
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
                    "rank": idx,
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