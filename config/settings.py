import os
from dotenv import load_dotenv
from pydantic import BaseSettings
from typing import Optional

load_dotenv()

class Settings(BaseSettings):
    # API 키
    DATAON_SEARCH_KEY: str = os.getenv("DATAON_SEARCH_KEY", "")
    DATAON_META_KEY: str = os.getenv("DATAON_META_KEY", "")
    SCIENCEON_CLIENT_ID: str = os.getenv("SCIENCEON_CLIENT_ID", "")
    SCIENCEON_ACCOUNTS: str = os.getenv("SCIENCEON_ACCOUNTS", "")

    # 모델 설정
    MODEL_NAME: str = os.getenv("MODEL_NAME", "upstage/SOLAR-10.7B-Instruct-v1.0")
    MODEL_CACHE_DIR: str = os.getenv("MODEL_CACHE_DIR", "./models")

    # 양자화 설정 (메모리 최적화)
    # fp16: ~21GB (고사양), int8: ~11GB (중사양), int4: ~6GB (저사양)
    QUANTIZATION: str = os.getenv("QUANTIZATION", "int8")

    # GPU 메모리 사용률 (0.0 ~ 1.0)
    GPU_MEMORY_UTILIZATION: float = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.9"))

    # LLM 생성 설정
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "512"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.1"))

    # 개발 모드 (GPU 없을 때 Mock 모델 사용)
    DEV_MODE: bool = os.getenv("DEV_MODE", "false").lower() == "true"

    # Redis 설정 (선택사항 - 캐싱 성능 향상)
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    REDIS_PASSWORD: Optional[str] = os.getenv("REDIS_PASSWORD")
    USE_REDIS: bool = os.getenv("USE_REDIS", "false").lower() == "true"

    # 서버 설정
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

    # 로깅 설정
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "logs/app.log")

    # API URLs
    DATAON_BASE_URL: str = "https://dataon.kisti.re.kr"
    SCIENCEON_BASE_URL: str = "https://apigateway.kisti.re.kr"

    class Config:
        env_file = ".env"

settings = Settings()