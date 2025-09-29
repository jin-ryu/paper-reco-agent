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
    USE_VLLM: bool = os.getenv("USE_VLLM", "true").lower() == "true"
    QUANTIZATION: str = os.getenv("QUANTIZATION", "fp16")
    GPU_MEMORY_UTILIZATION: float = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.9"))
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "512"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.1"))

    # Redis 설정
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    REDIS_PASSWORD: Optional[str] = os.getenv("REDIS_PASSWORD")

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