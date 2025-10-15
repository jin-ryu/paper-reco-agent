"""
범용 언어모델 래퍼
하이브리드 추천 시스템을 위한 LLM 인터페이스
지원 모델: Qwen3-14B, Gemma2-9B 등
"""
import logging
import torch
from typing import Optional, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.config.settings import settings

logger = logging.getLogger(__name__)


class LLMModel:
    """
    범용 언어모델 래퍼 (MODEL_NAME 환경변수로 모델 선택)

    지원 모델:
    - Qwen/Qwen3-14B (14.8B 파라미터, 32K context)
    - google/gemma-2-9b-it (9B 파라미터, 8K context)
    - 기타 Hugging Face의 causal LM 모델

    기능:
    - 검색 쿼리 생성
    - 후보 분석 및 추천 생성
    - 추천 이유 작성
    """

    def __init__(self):
        self.model_name = settings.MODEL_NAME
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None

        # 모델 종류 감지
        self.is_qwen = "qwen" in self.model_name.lower()
        self.is_gemma = "gemma" in self.model_name.lower()

        # 모델별 설정
        self._setup_model_config()

        self._load_model()

    def _setup_model_config(self):
        """모델별 설정"""
        if self.is_qwen:
            self.model_type = "Qwen"
            self.max_context_length = 32768
            self.supports_thinking = True
        elif self.is_gemma:
            self.model_type = "Gemma"
            self.max_context_length = 8192
            self.supports_thinking = False
        else:
            self.model_type = "Generic"
            self.max_context_length = 4096
            self.supports_thinking = False

    def _load_model(self):
        """모델 로딩 (FP16)"""
        try:
            logger.info(f"🚀 {self.model_type} 모델 로딩 시작: {self.model_name}")
            logger.info(f"   - 디바이스: {self.device}")

            # 토크나이저 로드
            tokenizer_kwargs = {
                "cache_dir": settings.MODEL_CACHE_DIR,
                "trust_remote_code": True
            }
            if settings.HF_TOKEN:
                tokenizer_kwargs["token"] = settings.HF_TOKEN

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                **tokenizer_kwargs
            )

            # 모델 로드 설정
            load_kwargs = {
                "cache_dir": settings.MODEL_CACHE_DIR,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True
            }
            if settings.HF_TOKEN:
                load_kwargs["token"] = settings.HF_TOKEN

            # FP16 모드로 로드
            if self.device == "cuda":
                logger.info("   - FP16 모드")
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

            logger.info(f"✅ {self.model_type} 모델 로딩 완료")

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

            # 채팅 형식으로 변환
            messages = [
                {"role": "system", "content": "You are a research recommendation assistant. You must output ONLY valid JSON format. Never use <think> tags or any explanations. Start your response directly with '{' character."},
                {"role": "user", "content": prompt}
            ]

            # 토크나이징 (모델별 처리)
            if self.is_qwen:
                # Qwen3: thinking 모드 비활성화
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False  # Qwen3 thinking 모드 완전히 비활성화
                )
            else:
                # Gemma 및 기타 모델: 일반 채팅 템플릿
                try:
                    text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                except Exception:
                    # 채팅 템플릿 지원 안하는 경우 직접 포맷팅
                    text = f"System: {messages[0]['content']}\n\nUser: {messages[1]['content']}\n\nAssistant:"

            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_context_length
            ).to(self.device)

            # 생성
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id if self.tokenizer.eos_token_id else self.tokenizer.pad_token_id,
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
        # 모델별 파라미터 수 추정
        if self.is_qwen and "14b" in self.model_name.lower():
            params = "14.8B"
        elif self.is_gemma and "9b" in self.model_name.lower():
            params = "9B"
        elif self.is_gemma and "2b" in self.model_name.lower():
            params = "2B"
        else:
            params = "Unknown"

        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "device": self.device,
            "dtype": "float16",
            "max_tokens": settings.MAX_TOKENS,
            "temperature": settings.TEMPERATURE,
            "parameters": params,
            "context_length": f"{self.max_context_length // 1024}K"
        }

    def cleanup(self):
        """리소스 정리"""
        try:
            import gc

            # 1. 모델 삭제
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
