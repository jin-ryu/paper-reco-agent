"""
SOLAR-10.7B 언어모델 래퍼
하이브리드 추천 시스템을 위한 LLM 인터페이스
"""
import logging
import torch
from typing import Optional, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
from config.settings import settings

# accelerate가 설치되어 있는지 확인
try:
    import accelerate
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False

# bitsandbytes가 설치되어 있는지 확인 (INT4/INT8 양자화용)
try:
    import bitsandbytes
    BITSANDBYTES_AVAILABLE = True
except Exception:
    BITSANDBYTES_AVAILABLE = False

logger = logging.getLogger(__name__)


class SolarModel:
    """
    SOLAR-10.7B Instruct 모델 래퍼

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
        """모델 로딩 (양자화 지원)"""
        try:
            logger.info(f"🚀 SOLAR 모델 로딩 시작: {self.model_name}")
            logger.info(f"   - 디바이스: {self.device}")
            logger.info(f"   - 양자화: {settings.QUANTIZATION}")

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

            # 양자화 설정 (메모리 절약)
            if settings.QUANTIZATION in ["int8", "int4"] and self.device == "cuda":
                # INT8/INT4는 bitsandbytes 필요
                if not BITSANDBYTES_AVAILABLE:
                    logger.warning("⚠️  bitsandbytes가 설치되지 않았거나 CUDA 문제 발생")
                    logger.warning(f"   {settings.QUANTIZATION} 양자화 불가 - FP16으로 대체")
                    logger.info("   - FP16 모드로 전환 (~21GB VRAM)")
                    load_kwargs["torch_dtype"] = torch.float16
                    if ACCELERATE_AVAILABLE:
                        load_kwargs["device_map"] = "auto"
                elif settings.QUANTIZATION == "int8":
                    logger.info("   - INT8 양자화 활성화 (~11GB VRAM)")
                    load_kwargs["load_in_8bit"] = True
                    load_kwargs["device_map"] = "auto"
                elif settings.QUANTIZATION == "int4":
                    logger.info("   - INT4 양자화 활성화 (~6GB VRAM)")
                    load_kwargs["load_in_4bit"] = True
                    load_kwargs["device_map"] = "auto"
            elif self.device == "cuda":
                # device_map 없이 단순 로드 (accelerate 이슈 회피)
                logger.info("   - FP16 모드 (~21GB VRAM, device_map 없음)")
                load_kwargs["torch_dtype"] = torch.float16
            else:
                logger.info("   - CPU 모드 (느림)")

            # 모델 로드
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **load_kwargs
            )

            # device_map 없을 때만 명시적으로 이동
            if "device_map" not in load_kwargs:
                self.model = self.model.to(self.device)

            self.model.eval()

            logger.info(f"✅ SOLAR 모델 로딩 완료")

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

            # 토크나이징
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.device)

            # 생성
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
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
        한국어 작업을 위한 프롬프트 생성 (SOLAR 최적화)

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
            candidates_text += f"   설명: {desc}...\n"
            candidates_text += f"   유사도: 의미적 {semantic_score:.2f} | 어휘적 {lexical_score:.2f} | 최종 {final_score:.2f}\n"
            if common_keywords:
                candidates_text += f"   공통 용어: {', '.join(common_keywords)}\n"
            if keywords:
                candidates_text += f"   키워드: {keywords}\n"

        prompt = f"""### Instruction:
당신은 연구 데이터와 논문을 추천하는 전문 에이전트입니다.

**중요**: 임베딩 모델(E5)이 이미 유사도를 계산했습니다. 당신의 역할은:
1. 유사도 점수를 분석하여 가장 관련성 높은 후보 선별
2. 각 추천에 대해 **구체적이고 논리적인 이유** 작성
3. 추천 레벨 결정 (강추/추천/참고)

{task_description}

### 추천 레벨 기준:
- **강추**: 최종 유사도 ≥ 0.75, 의미적+어휘적 모두 높음
- **추천**: 최종 유사도 ≥ 0.60, 의미적 또는 어휘적 높음
- **참고**: 최종 유사도 ≥ 0.45, 부분적 연관성

### Source Dataset:
제목: {source_title}
설명: {source_description}...
키워드: {source_keywords}
분류: {source_classification}

### Candidates (E5 임베딩으로 필터링됨):
{candidates_text}

### Response Format:
다음 JSON 형식으로만 응답하세요. **추가 설명 없이 JSON만 출력**하세요.

```json
{{
  "recommendations": [
    {{
      "candidate_number": 1,
      "title": "후보 제목 그대로",
      "type": "paper 또는 dataset",
      "score": 0.85,
      "reason": "의미적 유사도 0.85로 매우 높음; 공통 키워드 'A', 'B', 'C'로 주제 일치; 어휘적 매칭도 높아 핵심 용어 공유",
      "level": "강추"
    }}
  ]
}}
```

**중요**:
- reason은 2-3문장으로 구체적으로 작성
- 유사도 점수를 근거로 활용
- 공통 용어/키워드를 언급
"""
        return prompt

    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "quantization": settings.QUANTIZATION,
            "max_tokens": settings.MAX_TOKENS,
            "temperature": settings.TEMPERATURE,
            "parameters": "10.7B"
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
