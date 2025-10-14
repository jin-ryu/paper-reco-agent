"""
Qwen3-14B 언어모델 래퍼
하이브리드 추천 시스템을 위한 LLM 인터페이스
"""
import logging
import torch
from typing import Optional, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.config.settings import settings

logger = logging.getLogger(__name__)


class QwenModel:
    """
    Qwen3-14B 모델 래퍼

    - 100+ 언어 지원 (영어, 한국어, 일본어, 중국어 등)
    - 14.8B 파라미터 (13.2B non-embedding)
    - 검색 쿼리 생성
    - 후보 분석 및 추천 생성
    - 추천 이유 작성
    """

    def __init__(self):
        self.model_name = settings.MODEL_NAME
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None

        self._load_model()

    def _load_model(self):
        """모델 로딩 (FP16)"""
        try:
            logger.info(f"🚀 Qwen 모델 로딩 시작: {self.model_name}")
            logger.info(f"   - 디바이스: {self.device}")

            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=settings.MODEL_CACHE_DIR,
                trust_remote_code=True
            )

            # 모델 로드 설정
            load_kwargs = {
                "cache_dir": settings.MODEL_CACHE_DIR,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True
            }

            # FP16 모드로 로드
            if self.device == "cuda":
                logger.info("   - FP16 모드 (~28GB VRAM)")
                load_kwargs["torch_dtype"] = torch.float16
            else:
                logger.info("   - CPU 모드 (느림)")

            # 모델 로드
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **load_kwargs
            )

            # 명시적으로 디바이스로 이동
            self.model = self.model.to(self.device)
            self.model.eval()

            logger.info(f"✅ Qwen 모델 로딩 완료")

        except Exception as e:
            logger.error(f"❌ 모델 로딩 실패: {e}")
            raise e

    async def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        텍스트 생성

        Args:
            prompt: 입력 프롬프트
            max_new_tokens: 최대 생성 토큰 수
            temperature: 샘플링 온도

        Returns:
            생성된 텍스트
        """
        try:
            if max_new_tokens is None:
                max_new_tokens = settings.MAX_TOKENS
            if temperature is None:
                temperature = settings.TEMPERATURE

            # Qwen3 채팅 형식으로 변환 (non-thinking 모드 강제)
            messages = [
                {"role": "system", "content": "You are a research recommendation assistant. You must output ONLY valid JSON format. Never use <think> tags or any explanations. Start your response directly with '{' character."},
                {"role": "user", "content": prompt}
            ]

            # 토크나이징 (채팅 템플릿 적용, thinking 모드 비활성화)
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False  # Qwen3 thinking 모드 완전히 비활성화
            )

            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=32768  # Qwen3는 32K 토큰 지원 (확장 시 128K)
            ).to(self.device)

            # 생성 (thinking mode 비활성화)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1  # 반복 방지
                )

            # 디코딩 (입력 프롬프트 제외)
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            return generated_text.strip()

        except Exception as e:
            logger.error(f"텍스트 생성 실패: {e}")
            return ""

    def create_korean_prompt(self, task_description: str, context: Dict[str, Any]) -> str:
        """
        한국어/다국어 작업을 위한 프롬프트 생성 (Qwen3 최적화)

        Args:
            task_description: 작업 설명
            context: 컨텍스트 정보

        Returns:
            프롬프트 문자열
        """
        source_title = context.get('source_title', '')
        source_description = context.get('source_description', '')[:500]
        source_keywords = context.get('source_keywords', '')
        source_classification = context.get('source_classification', '')

        candidates = context.get('candidates', [])

        # 후보 정보 포맷팅 (유사도 점수 포함)
        candidates_text = ""
        for i, candidate in enumerate(candidates[:15], 1):
            cand_type = candidate.get('type', 'unknown')
            title = candidate.get('title', '')
            desc = candidate.get('description', '')[:150]

            # 유사도 점수 상세 정보
            semantic_score = candidate.get('semantic_score', 0.0)
            lexical_score = candidate.get('lexical_score', 0.0)
            final_score = candidate.get('final_score', 0.0)
            common_keywords = candidate.get('common_keywords', [])[:5]

            keywords = ', '.join(candidate.get('keywords', []))[:150]

            candidates_text += f"\n[{i}] ({cand_type}) {title}\n"
            candidates_text += f"   Description: {desc}...\n"
            candidates_text += f"   Similarity: Semantic {semantic_score:.2f} | Lexical {lexical_score:.2f} | Final {final_score:.2f}\n"
            if common_keywords:
                candidates_text += f"   Common terms: {', '.join(common_keywords)}\n"
            if keywords:
                candidates_text += f"   Keywords: {keywords}\n"

        prompt = f"""# Task
You are a research data and paper recommendation expert.

**Important**: E5 embedding model has already calculated similarity scores. Your role:
1. Analyze similarity scores and select most relevant candidates
2. Write specific and logical reasons for each recommendation
3. Determine recommendation level (강추/추천/참고)

{task_description}

## Recommendation Level Criteria:
- 강추 (Strong): Final similarity ≥ 0.75, both semantic+lexical high
- 추천 (Recommend): Final similarity ≥ 0.60, semantic or lexical high
- 참고 (Reference): Final similarity ≥ 0.45, partial relevance

## Source Dataset:
Title: {source_title}
Description: {source_description}...
Keywords: {source_keywords}
Classification: {source_classification}

## Candidates (Filtered by E5 embedding):
{candidates_text}

## Output Format:
Output ONLY valid JSON. No explanations, comments, or examples.

{{
  "recommendations": [
    {{
      "candidate_number": 1,
      "reason": "High semantic similarity with common keywords",
      "level": "강추"
    }},
    {{
      "candidate_number": 2,
      "reason": "Related research field",
      "level": "추천"
    }}
  ]
}}

Rules:
- Output JSON only, start with '{{' character
- candidate_number: Number from candidate list above (1-15)
- reason: One concise sentence explaining why this is relevant
- level: "강추" (≥0.75) or "추천" (≥0.60) or "참고" (≥0.45)
- DO NOT include title, type, or score - only candidate_number, reason, level
"""
        return prompt

    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "dtype": "float16",
            "max_tokens": settings.MAX_TOKENS,
            "temperature": settings.TEMPERATURE,
            "parameters": "14.8B",
            "context_length": "32K (extendable to 128K)"
        }

    def cleanup(self):
        """리소스 정리"""
        try:
            if self.model:
                del self.model
            if self.tokenizer:
                del self.tokenizer

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("✅ 모델 리소스 정리 완료")
        except Exception as e:
            logger.error(f"리소스 정리 중 오류: {e}")
