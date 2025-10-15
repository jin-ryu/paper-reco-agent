import json
import time
import os
from src.tools.research_tools import *
from src.models.prompts import create_search_queries_prompt, create_reranking_prompt
from src.utils.text_utils import clean_keywords
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
            logger.info("🚀 프로덕션 모드로 실행: 실제 LLM 모델 사용")
            from src.models.llm_model import LLMModel
            self.llm_model = LLMModel()

        # 후보 검색 설정
        self.search_per_keyword = 5  # 키워드당 검색할 개수
        self.max_paper_candidates = 10  # E5/BM25로 상위 10개 논문만 선별
        self.max_dataset_candidates = 10  # E5/BM25로 상위 10개 데이터셋만 선별

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

    async def recommend(
        self,
        dataset_id: str,
        num_paper_recommendations: int = 3,
        num_dataset_recommendations: int = 3
    ) -> Dict[str, Any]:
        """
        메인 추천 함수

        Args:
            dataset_id: DataON 데이터셋 ID
            num_paper_recommendations: 추천할 논문 개수 (기본값: 3)
            num_dataset_recommendations: 추천할 데이터셋 개수 (기본값: 3)

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
            candidates, search_result = await self._collect_candidates_with_queries(search_queries, dataset_id)
            logger.info(f"총 {len(candidates)}개 후보 수집 완료")

            # 4단계: 유사도 계산 및 순위 결정
            ranked_papers, ranked_datasets = await self._rank_candidates(source_data, candidates)
            logger.info(f"상위 {len(ranked_papers)}개 논문, {len(ranked_datasets)}개 데이터셋 순위 결정 완료")

            # 5단계: LLM을 사용한 최종 추천 생성
            paper_reco_task = self._get_llm_recommendations_for_type(
                source_data,
                ranked_papers[:self.max_paper_candidates],
                num_paper_recommendations,
                "paper"
            )
            dataset_reco_task = self._get_llm_recommendations_for_type(
                source_data,
                ranked_datasets[:self.max_dataset_candidates],
                num_dataset_recommendations,
                "dataset"
            )

            llm_results = await asyncio.gather(paper_reco_task, dataset_reco_task)
            
            paper_recommendations = llm_results[0]
            dataset_recommendations = llm_results[1]

            processing_time = int((time.time() - start_time) * 1000)
            logger.info(f"추천 프로세스 완료: {processing_time}ms")

            return {
                "source_dataset": {
                    "id": dataset_id,
                    "title": source_data.get('title', ''),
                    "description": source_data.get('description', '')[:200] + "...",
                    "keywords": source_data.get('keywords', [])
                },
                "search_result": search_result,
                "paper_recommendations": paper_recommendations,
                "dataset_recommendations": dataset_recommendations,
                "processing_time_ms": processing_time,
                "candidates_analyzed": len(candidates),
                "model_info": self.llm_model.get_model_info(),
                "embedding_model_info": self._get_embedding_model_info()
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
            logger.info(f"소스 데이터셋 정보 조회 완료: {source_data.get('title', '')}")
            return source_data
        except Exception as e:
            logger.error(f"소스 데이터셋 조회 실패: {e}")
            return {"error": str(e)}

    async def _generate_search_queries(self, source_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """2단계: LLM을 사용해서 최적의 검색 쿼리 생성 (재시도 로직 포함)"""
        max_retries = 2
        previous_error = None
        title = source_data.get('title', '')
        description = source_data.get('description', '')
        original_keywords = source_data.get('keywords', [])

        for attempt in range(max_retries):
            try:
                logger.info(f"LLM 검색 쿼리 생성 시도 {attempt + 1}/{max_retries}")

                # 프롬프트 생성 (에러 피드백 포함)
                prompt = create_search_queries_prompt(source_data, previous_error=previous_error)

                # 프롬프트 로깅 (디버깅용)
                logger.info(f'=' * 80)
                logger.info(f"LLM에게 키워드 추출 프롬프트:")
                logger.info(prompt)
                logger.info(f'=' * 80)

                # LLM 호출
                response = await self.llm_model.generate(
                    prompt,
                    max_new_tokens=300,
                    temperature=0.1
                )

                # JSON 파싱 및 검증
                import re
                response_cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
                json_match = re.search(r'```json\\s*(\\{.*?\\})\\s*```', response_cleaned, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_match = re.search(r'\\{.*\\}', response_cleaned, re.DOTALL)
                    json_str = json_match.group(0) if json_match else response_cleaned

                queries = json.loads(json_str)
                dataset_queries = clean_keywords(queries.get('dataset_queries', []))
                paper_queries = clean_keywords(queries.get('paper_queries', []))

                if not dataset_queries or not paper_queries:
                    raise ValueError("LLM이 생성한 쿼리가 비어 있습니다.")

                logger.info(f"✅ LLM 키워드 생성 완료")
                logger.info(f"   원본 키워드 ({len(original_keywords)}개): {original_keywords}")
                logger.info(f"   → 데이터셋 검색 ({len(dataset_queries)}개): {dataset_queries}")
                logger.info(f"   → 논문 검색 ({len(paper_queries)}개): {paper_queries}")
                return {
                    'dataset_queries': dataset_queries,
                    'paper_queries': paper_queries
                }

            except (json.JSONDecodeError, ValueError, AttributeError) as e:
                error_msg = f"LLM 응답 파싱 또는 검증 실패: {str(e)}"
                logger.warning(f"{error_msg} (시도 {attempt + 1}/{max_retries})")
                logger.warning(f"LLM 원본 응답:\n{response}")
                if attempt < max_retries - 1:
                    previous_error = error_msg
                    continue
                else:
                    logger.error("모든 재시도 실패, 폴백 로직을 사용합니다.")
                    break  # 루프를 빠져나가 폴백 로직 실행

            except Exception as e:
                error_msg = f"검색 쿼리 생성 중 예외 발생: {str(e)}"
                logger.error(f"{error_msg} (시도 {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    previous_error = error_msg
                    continue
                else:
                    logger.error("모든 재시도 실패, 폴백 로직을 사용합니다.")
                    break # 루프를 빠져나가 폴백 로직 실행

        # 폴백 로직
        logger.info("폴백 로직 실행: 기존 키워드 또는 텍스트에서 키워드를 추출합니다.")
        fallback_keywords = source_data.get('keywords', [])
        if not fallback_keywords:
            text = f"{title} {description}"
            fallback_keywords = extract_keywords_from_text(text)

        return {
            'dataset_queries': fallback_keywords,
            'paper_queries': fallback_keywords
        }

    async def _collect_candidates_with_queries(self, search_queries: Dict[str, List[str]], source_id: str) -> tuple:
        """3단계: 생성된 쿼리로 후보 수집 (DataON + ScienceON)"""
        candidates = []

        try:
            # 병렬로 후보 수집 (키워드별 상세 결과 포함)
            dataset_results = await self._search_similar_datasets_detailed(search_queries['dataset_queries'])
            paper_results = await self._search_related_papers_detailed(search_queries['paper_queries'])

            # search_result 구성
            search_result = {
                "paper_keywords": search_queries['paper_queries'],
                "dataset_keywords": search_queries['dataset_queries'],
                "paper_search_details": paper_results['details'],
                "dataset_search_details": dataset_results['details']
            }

            # DataON 데이터셋 후보
            for dataset in dataset_results['candidates']:
                candidates.append({
                    'type': 'dataset',
                    'source': 'dataon',
                    'data': dataset,
                    'title': dataset.get('title', ''),
                    'description': dataset.get('description', ''),
                    'keywords': clean_keywords(dataset.get('keywords', [])),  # 키워드 전처리
                    'url': dataset.get('url', ''),
                    'combined_text': dataset.get('combined_text', '')
                })

            # ScienceON 논문 후보
            for paper in paper_results['candidates']:
                candidates.append({
                    'type': 'paper',
                    'source': 'scienceon',
                    'data': paper,
                    'title': paper.get('title', ''),
                    'description': paper.get('abstract', ''),
                    'keywords': clean_keywords(paper.get('keywords', [])),  # 키워드 전처리
                    'url': paper.get('content_url', ''),
                    'combined_text': paper.get('combined_text', ''),
                    'cn': paper.get('cn', '')
                })

            # Filter out the source dataset itself
            filtered_candidates = [
                c for c in candidates
                if not (c['type'] == 'dataset' and c.get('data', {}).get('svc_id') == source_id)
            ]

            return filtered_candidates, search_result

        except Exception as e:
            logger.error(f"후보 수집 실패: {e}")
            return [], {"paper_keywords": [], "dataset_keywords": [], "paper_search_details": [], "dataset_search_details": []}

    async def _search_similar_datasets_detailed(self, keywords: List[str]) -> Dict[str, Any]:
        """DataON에서 유사한 데이터셋 검색 (키워드별 상세 정보 포함)"""
        try:
            all_datasets = []
            details = []
            seen_ids = set()

            # 키워드별로 검색하여 다양성 확보
            for keyword in keywords:
                keyword_datasets = await search_similar_dataon_datasets([keyword], limit=self.search_per_keyword)

                # 중복 제거 (svc_id 기준)
                for dataset in keyword_datasets:
                    svc_id = dataset.get('svc_id', '')
                    if svc_id and svc_id not in seen_ids:
                        seen_ids.add(svc_id)
                        all_datasets.append(dataset)

                details.append({
                    "keyword": keyword,
                    "count": len(keyword_datasets)
                })

            return {
                "candidates": all_datasets,  # 모든 후보 반환 (중복 제거만)
                "details": details
            }
        except Exception as e:
            logger.error(f"DataON 검색 실패: {e}")
            return {"candidates": [], "details": []}

    async def _search_related_papers_detailed(self, keywords: List[str]) -> Dict[str, Any]:
        """ScienceON에서 관련 논문 검색 (키워드별 상세 정보 포함)"""
        try:
            papers = []
            details = []

            # 키워드별로 검색하여 다양성 확보
            for keyword in keywords:
                keyword_papers = await search_scienceon_papers(keyword, limit=self.search_per_keyword)
                papers.extend(keyword_papers)
                details.append({
                    "keyword": keyword,
                    "count": len(keyword_papers)
                })

            # 중복 제거 (CN 기준)
            seen_cns = set()
            unique_papers = []
            for paper in papers:
                cn = paper.get('cn', '')
                if cn and cn not in seen_cns:
                    seen_cns.add(cn)
                    unique_papers.append(paper)

            return {
                "candidates": unique_papers,  # 모든 후보 반환 (중복 제거만)
                "details": details
            }

        except Exception as e:
            logger.error(f"ScienceON 검색 실패: {e}")
            return {"candidates": [], "details": []}

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

                    hybrid_result = calculate_hybrid_similarity(source_text, candidate_text, candidate['type'])

                    similarity_score = hybrid_result['hybrid_score']  # 하이브리드 점수 사용



                    # ID 추출 (프롬프트에서 사용하기 위해)
                    item_id = ''
                    if candidate['type'] == 'paper':
                        # 논문: cn 필드 또는 URL에서 추출
                        item_id = candidate.get('cn', '')
                        if not item_id and 'cn=' in candidate.get('url', ''):
                            item_id = candidate['url'].split('cn=')[1].split('&')[0]
                    else:
                        # 데이터셋: data의 svc_id 또는 URL에서 추출
                        item_id = candidate.get('data', {}).get('svc_id', '')
                        if not item_id and 'svcId=' in candidate.get('url', ''):
                            item_id = candidate['url'].split('svcId=')[1].split('&')[0]

                    candidate.update({
                        'id': item_id,  # ID 추가 (프롬프트에서 사용)
                        'similarity_score': similarity_score,
                        'semantic_score': hybrid_result['semantic_score'],
                        'lexical_score': hybrid_result['lexical_score'],
                        'hybrid_score': similarity_score,
                        'common_keywords': hybrid_result.get('common_keywords', [])
                    })

                    scored_candidates.append(candidate)

                except Exception as e:
                    logger.warning(f"후보 점수 계산 실패: {e}")
                    continue

            # 점수순으로 정렬
            ranked_candidates = sorted(scored_candidates, key=lambda x: x['hybrid_score'], reverse=True)

            # 논문과 데이터셋으로 분리
            ranked_papers = [c for c in ranked_candidates if c['type'] == 'paper']
            ranked_datasets = [c for c in ranked_candidates if c['type'] == 'dataset']

            return ranked_papers, ranked_datasets

        except Exception as e:
            logger.error(f"후보 순위 결정 실패: {e}")
            return candidates

    async def _get_llm_recommendations_for_type(self, source_data: Dict[str, Any], top_candidates: List[Dict[str, Any]], num_recommendations: int, candidate_type: str) -> List[Dict[str, Any]]:
        """특정 타입(논문/데이터셋)에 대한 LLM 추천 생성"""
        if not top_candidates:
            logger.info(f"후보 목록이 비어 있어 {candidate_type} 추천을 건너뜁니다.")
            return []
        try:
            # LLM용 컨텍스트 준비
            context = {
                'source_title': source_data.get('title', ''),
                'source_description': source_data.get('description', ''),
                'source_keywords': ', '.join(source_data.get('keywords', [])),
                'candidates': top_candidates,
                'candidate_type': candidate_type
            }



            # LLM 호출 (최대 2회 재시도, 에러 피드백 사용)
            max_retries = 2
            previous_error = None

            for attempt in range(max_retries):
                try:
                    logger.info(f"LLM {candidate_type} 추천 생성 시도 {attempt + 1}/{max_retries}")

                    prompt = create_reranking_prompt(
                        context,
                        num_recommendations,
                        previous_error=previous_error
                    )

                    # 프롬프트 로깅 (디버깅용)
                    logger.info(f'=' * 80)
                    logger.info(f"LLM에게 전송하는 {candidate_type} 프롬프트:")
                    logger.info(prompt)
                    logger.info(f'=' * 80)

                    response = await self.llm_model.generate(prompt)

                    # JSON 응답 파싱
                    recommendations = self._parse_llm_response(response, top_candidates, num_recommendations)

                    if recommendations:
                        return recommendations
                    else:
                        logger.warning(f"LLM {candidate_type} 응답 파싱 실패 (시도 {attempt + 1}/{max_retries})")
                        if attempt < max_retries - 1:
                            # 에러 메시지 생성하여 재시도
                            previous_error = "Failed to parse JSON response. Please ensure output is valid JSON starting with '{' character."
                            logger.info(f"에러 피드백과 함께 재시도: {previous_error}")
                            continue
                        else:
                            logger.error(f"모든 재시도 실패, {candidate_type} 추천 생성 불가")
                            return []

                except json.JSONDecodeError as e:
                    # JSON 파싱 에러를 명시적으로 캡처하여 에러 메시지 전달
                    error_msg = f"JSON parsing error: {str(e)}"
                    logger.error(f"LLM {candidate_type} 응답 파싱 에러 (시도 {attempt + 1}/{max_retries}): {error_msg}")
                    if attempt < max_retries - 1:
                        previous_error = error_msg
                        continue
                    else:
                        return []

                except Exception as e:
                    logger.error(f"LLM {candidate_type} 호출 실패 (시도 {attempt + 1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        previous_error = f"Generation error: {str(e)}"
                        continue
                    else:
                        return []

            return []

        except Exception as e:
            logger.error(f"최종 추천 생성 실패: {e}")
            return []

    def _parse_llm_response(self, response: str, candidates: List[Dict[str, Any]], num_recommendations: int) -> List[Dict[str, Any]]:
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
                    for rec in llm_recommendations[:num_recommendations]:
                        # LLM이 반환한 명시적 rank 값과 candidate_id 사용
                        rank = rec.get('rank', 0)
                        candidate_id = rec.get('candidate_id', '')
                        logger.debug(f"처리 중: rank={rank}, candidate_id={candidate_id}")

                        # candidate_id로 후보 찾기
                        candidate = None
                        for cand in candidates:
                            if cand.get('id', '') == candidate_id:
                                candidate = cand
                                break

                        if candidate:
                            # Platform 결정 (source 필드 활용)
                            platform = candidate.get('source', 'unknown')

                            # LLM 추천 정보와 후보 데이터 결합
                            # rank는 LLM이 명시적으로 반환한 값 사용
                            # title, type, score, url, id, platform, keywords는 모두 candidates에서 가져옴
                            # reason, level만 LLM이 생성
                            final_rec = {
                                "rank": rank,  # LLM이 명시적으로 반환한 rank 값 사용
                                "type": candidate['type'],
                                "id": candidate_id,  # LLM이 선택한 ID
                                "platform": platform,  # Platform 추가 (dataon/scienceon)
                                "title": candidate['title'],
                                "description": candidate['description'][:200] + "..." if len(candidate.get('description', '')) > 200 else candidate.get('description', ''),
                                "keywords": candidate.get('keywords', []),  # 키워드 추가
                                "score": candidate.get('hybrid_score', 0.5),  # E5 계산한 점수 사용
                                "reason": rec.get('reason', '추천 이유 생성 실패'),
                                "level": rec.get('level', '참고'),
                                "url": candidate['url']
                            }
                            final_recommendations.append(final_rec)
                            logger.debug(f"✅ 추천 항목 추가 (rank={rank}, id={candidate_id}): {candidate['title'][:50]}")
                        else:
                            logger.warning(f"⚠️  candidate_id '{candidate_id}'를 찾을 수 없음")

                    # rank 값으로 정렬 (LLM이 명시적으로 지정한 순위)
                    final_recommendations.sort(key=lambda x: x['rank'])

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

    def _get_embedding_model_info(self) -> Dict[str, Any]:
        """임베딩 모델 및 하이브리드 유사도 설정 정보 반환"""
        from src.config.settings import settings

        return {
            "embedding_model": settings.EMBEDDING_MODEL,
            "paper_hybrid_weights": {
                "alpha": settings.PAPER_HYBRID_ALPHA,
                "beta": settings.PAPER_HYBRID_BETA
            },
            "dataset_hybrid_weights": {
                "alpha": settings.DATASET_HYBRID_ALPHA,
                "beta": settings.DATASET_HYBRID_BETA
            }
        }
