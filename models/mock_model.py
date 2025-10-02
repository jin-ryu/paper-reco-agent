"""
Mock 모델 (개발/테스트용)
GPU 없이 빠르게 테스트할 수 있는 더미 모델
"""
import logging
import json
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class MockQwenModel:
    """
    개발 모드용 Mock Qwen 모델

    실제 LLM 대신 미리 정의된 응답을 반환
    """

    def __init__(self):
        self.model_name = "MockQwen (Dev Mode)"
        logger.info("🎭 Mock Qwen 모델 초기화 (개발 모드)")

    async def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        Mock 텍스트 생성 (미리 정의된 JSON 반환)
        """
        logger.info(f"Mock 생성: {len(prompt)} 글자 프롬프트")

        # 검색 쿼리 생성 요청인지 확인
        if "검색 키워드" in prompt or "search" in prompt.lower() or "query" in prompt.lower():
            mock_response = {
                "dataset_queries": ["climate change", "환경 데이터", "기후 변화"],
                "paper_queries": ["climate research", "환경 연구", "기후 과학"]
            }
        else:
            # 추천 생성
            mock_response = {
                "recommendations": [
                    {
                        "rank": 1,
                        "candidate_number": 1,
                        "title": "Mock Recommendation 1",
                        "type": "paper",
                        "score": 0.85,
                        "reason": "High semantic similarity with relevant keywords",
                        "level": "강추"
                    },
                    {
                        "rank": 2,
                        "candidate_number": 2,
                        "title": "Mock Recommendation 2",
                        "type": "dataset",
                        "score": 0.72,
                        "reason": "Related research field and methodology",
                        "level": "추천"
                    },
                    {
                        "rank": 3,
                        "candidate_number": 3,
                        "title": "Mock Recommendation 3",
                        "type": "paper",
                        "score": 0.65,
                        "reason": "Partially related topic",
                        "level": "참고"
                    }
                ]
            }

        return json.dumps(mock_response, ensure_ascii=False, indent=2)

    def create_korean_prompt(self, task_description: str, context: Dict[str, Any]) -> str:
        """Mock 프롬프트 생성 (실제로는 사용 안함)"""
        return f"Mock prompt for: {task_description}"

    def get_model_info(self) -> Dict[str, Any]:
        """Mock 모델 정보"""
        return {
            "model_name": self.model_name,
            "device": "cpu",
            "quantization": "none",
            "max_tokens": 512,
            "temperature": 0.1,
            "parameters": "Mock (0B)",
            "dev_mode": True
        }

    def cleanup(self):
        """Mock 리소스 정리 (아무것도 안함)"""
        logger.info("✅ Mock 모델 정리 완료")


# 하위 호환성을 위한 alias
MockSolarModel = MockQwenModel
