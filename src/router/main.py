from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import logging
import os
from datetime import datetime

from src.agents.recommendation_agent import KoreanResearchRecommendationAgent
from src.config.settings import settings

# 로깅 설정
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(settings.LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# FastAPI 앱 초기화
app = FastAPI(
    title="Research Recommendation Agent",
    description="LLM 기반 다국어 연구 데이터/논문 추천 시스템",
    version="2.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 에이전트 인스턴스
agent = None

@app.on_event("startup")
async def startup_event():
    """서버 시작시 모델 로드"""
    global agent
    try:
        logger.info("🚀 서버 시작: 모델 로딩 중...")
        agent = KoreanResearchRecommendationAgent()
        logger.info("✅ 모델 로딩 완료")
    except Exception as e:
        logger.error(f"❌ 모델 로딩 실패: {e}")
        raise e

@app.on_event("shutdown")
async def shutdown_event():
    """서버 종료시 리소스 정리"""
    global agent
    if agent and agent.llm_model:
        agent.llm_model.cleanup()
        logger.info("🧹 리소스 정리 완료")

# 요청/응답 모델
class RecommendationRequest(BaseModel):
    dataset_id: str
    max_recommendations: Optional[int] = 5

class RecommendationItem(BaseModel):
    rank: int  # 추천 순위 (1=최고 추천)
    type: str  # "dataset" or "paper"
    title: str
    description: str
    score: float
    reason: str
    level: str  # "강추", "추천", "참고"
    url: str

class RecommendationResponse(BaseModel):
    source_dataset: dict
    recommendations: List[RecommendationItem]
    processing_time_ms: int
    candidates_analyzed: int
    model_info: dict

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    model_loaded: bool
    settings: dict

# API 엔드포인트
@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """
    데이터셋 ID를 받아서 관련 논문/데이터셋을 추천합니다.

    - **dataset_id**: DataON 데이터셋 ID
    - **max_recommendations**: 최대 추천 개수 (기본값: 5)
    """
    if not agent:
        raise HTTPException(status_code=503, detail="모델이 아직 로드되지 않았습니다")

    try:
        logger.info(f"추천 요청: 데이터셋 ID = {request.dataset_id}")

        # 에이전트 설정 업데이트
        if request.max_recommendations:
            agent.final_recommendations = min(request.max_recommendations, 10)  # 최대 10개 제한

        result = await agent.recommend(request.dataset_id)

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        logger.info(f"추천 완료: {len(result['recommendations'])}개 추천")

        return RecommendationResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"추천 처리 중 오류: {e}")
        raise HTTPException(status_code=500, detail=f"서버 내부 오류: {str(e)}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    서버 상태 확인
    """
    return HealthResponse(
        status="healthy" if agent else "loading",
        timestamp=datetime.now().isoformat(),
        model_loaded=agent is not None,
        settings={
            "model_name": settings.MODEL_NAME,
            "embedding_model": settings.EMBEDDING_MODEL,
            "quantization": settings.QUANTIZATION,
            "max_tokens": settings.MAX_TOKENS,
            "temperature": settings.TEMPERATURE,
            "dev_mode": settings.DEV_MODE
        }
    )

@app.get("/models/info")
async def get_model_info():
    """
    현재 로드된 모델 정보 조회
    """
    if not agent:
        raise HTTPException(status_code=503, detail="모델이 아직 로드되지 않았습니다")

    return agent.llm_model.get_model_info()

@app.get("/api/test/dataon/{dataset_id}")
async def test_dataon_api(dataset_id: str):
    """
    DataON API 연결 테스트
    """
    try:
        from src.tools.research_tools import get_dataon_dataset_metadata
        result = await get_dataon_dataset_metadata(dataset_id)
        return {"success": True, "data": result}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/test/scienceon")
async def test_scienceon_api():
    """
    ScienceON API 연결 테스트
    """
    try:
        from src.tools.research_tools import get_scienceon_access_token
        token = await get_scienceon_access_token()
        return {"success": True, "token_received": bool(token)}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/")
async def root():
    """
    루트 엔드포인트
    """
    return {
        "message": "Research Recommendation Agent",
        "version": "2.0.0",
        "model": settings.MODEL_NAME,
        "languages": "English, Japanese, Korean, Chinese, and 25+ more",
        "docs": "/docs",
        "health": "/health"
    }

# 예외 처리
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"전역 예외 발생: {exc}")
    return HTTPException(status_code=500, detail="서버 내부 오류가 발생했습니다")

if __name__ == "__main__":
    import uvicorn

    logger.info(f"서버 시작: {settings.HOST}:{settings.PORT}")
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
