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
            import gc

            # 1. Qwen 모델 삭제
            if hasattr(self, 'model') and self.model is not None:
                del self.model
                self.model = None

            # 2. 토크나이저 삭제
            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None

            # 3. Python 가비지 컬렉션 강제 실행
            gc.collect()

            # 4. PyTorch CUDA 캐시 비우기
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # 모든 CUDA 작업 완료 대기

            logger.info("✅ 모델 리소스 정리 완료")
        except Exception as e:
            logger.error(f"리소스 정리 중 오류: {e}")
