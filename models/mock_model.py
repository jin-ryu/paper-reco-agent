"""
Mock SOLAR 모델 (개발/테스트용)
GPU 없이 빠른 테스트를 위한 모의 모델
"""
import logging
import json
import random
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class MockSolarModel:
    """
    개발 모드용 Mock 모델
    실제 LLM 없이 빠르게 테스트 가능
    """

    def __init__(self):
        self.model_name = "mock-solar-10.7b"
        self.device = "mock"
        logger.info("🎭 Mock SOLAR 모델 로딩 (개발 모드)")

    async def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        Mock 텍스트 생성 (간단한 JSON 응답 반환)

        Args:
            prompt: 입력 프롬프트
            max_new_tokens: 무시됨
            temperature: 무시됨

        Returns:
            Mock JSON 응답
        """
        try:
            # 프롬프트에서 후보 수 추정
            candidate_count = prompt.count('[')

            # Mock 추천 생성
            recommendations = []
            for i in range(1, min(6, candidate_count + 1)):
                score = round(random.uniform(0.6, 0.95), 2)
                level = "강추" if score >= 0.8 else "추천" if score >= 0.7 else "참고"

                recommendations.append({
                    "candidate_number": i,
                    "title": f"Mock 후보 {i}",
                    "type": "paper" if i % 2 == 0 else "dataset",
                    "score": score,
                    "reason": f"공통 키워드로 높은 연관성; Mock 추천 {i}번",
                    "level": level
                })

            response = {"recommendations": recommendations}

            # JSON 형식으로 반환 (실제 LLM 응답처럼)
            return f"```json\n{json.dumps(response, ensure_ascii=False, indent=2)}\n```"

        except Exception as e:
            logger.error(f"Mock 생성 실패: {e}")
            return '```json\n{"recommendations": []}\n```'

    def create_korean_prompt(self, task_description: str, context: Dict[str, Any]) -> str:
        """
        프롬프트 생성 (Mock에서는 간단히 처리)

        Args:
            task_description: 작업 설명
            context: 컨텍스트

        Returns:
            Mock 프롬프트
        """
        return f"Mock prompt: {task_description}"

    def get_model_info(self) -> Dict[str, Any]:
        """Mock 모델 정보"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "quantization": "none",
            "max_tokens": 512,
            "temperature": 0.1,
            "parameters": "mock",
            "mode": "development"
        }

    def cleanup(self):
        """Mock cleanup (아무것도 안함)"""
        logger.info("🎭 Mock 모델 정리 완료")
