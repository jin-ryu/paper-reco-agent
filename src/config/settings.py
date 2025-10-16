import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from typing import Optional

load_dotenv()

class Settings(BaseSettings):
    # API 키
    DATAON_SEARCH_KEY: str = os.getenv("DATAON_SEARCH_KEY", "")
    DATAON_META_KEY: str = os.getenv("DATAON_META_KEY", "")
    SCIENCEON_CLIENT_ID: str = os.getenv("SCIENCEON_CLIENT_ID", "")
    SCIENCEON_ACCOUNTS: str = os.getenv("SCIENCEON_ACCOUNTS", "")

    # 모델 설정
    MODEL_NAME: str = os.getenv("MODEL_NAME", "Qwen/Qwen3-14B")
    MODEL_CACHE_DIR: str = os.getenv("MODEL_CACHE_DIR", "/home/infidea/paper-reco-agent/model")
    HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN", None)

    # LLM 생성 설정
    MAX_TOKENS_RERANKING: int = int(os.getenv("MAX_TOKENS_RERANKING", "1024"))
    MAX_TOKENS_KEYWORD: int = int(os.getenv("MAX_TOKENS_KEYWORD", "300"))
    # 하위 호환성을 위한 MAX_TOKENS (MAX_TOKENS_RERANKING으로 폴백)
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", os.getenv("MAX_TOKENS_RERANKING", "1024")))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.0"))

    # 개발 모드 (GPU 없을 때 Mock 모델 사용)
    DEV_MODE: bool = os.getenv("DEV_MODE", "false").lower() == "true"

    # 임베딩 모델 설정
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")

    # 하이브리드 유사도 가중치
    PAPER_HYBRID_ALPHA: float = float(os.getenv("PAPER_HYBRID_ALPHA", "0.8"))
    PAPER_HYBRID_BETA: float = float(os.getenv("PAPER_HYBRID_BETA", "0.2"))
    DATASET_HYBRID_ALPHA: float = float(os.getenv("DATASET_HYBRID_ALPHA", "0.5"))
    DATASET_HYBRID_BETA: float = float(os.getenv("DATASET_HYBRID_BETA", "0.5"))

    # 서버 설정
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

    # 로깅 설정
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "logs/app.log")

    # API URLs
    DATAON_BASE_URL: str = os.getenv("DATAON_BASE_URL", "https://dataon.kisti.re.kr")
    SCIENCEON_BASE_URL: str = os.getenv("SCIENCEON_BASE_URL", "https://apigateway.kisti.re.kr")

    class Config:
        env_file = ".env"

settings = Settings()