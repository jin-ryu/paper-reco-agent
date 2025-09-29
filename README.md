# Korean Research Recommendation Agent

SOLAR-10.7B 기반 한국 연구 데이터 및 논문 추천 시스템

## 🎯 프로젝트 개요

이 프로젝트는 주어진 연구 데이터셋에 대해 의미적으로 관련성이 높은 연구논문과 데이터셋을 추천하는 AI 시스템입니다. 한국어 특화 소규모 언어모델(SOLAR-10.7B)을 활용하여 중저사양 하드웨어에서도 빠른 응답시간과 높은 추천 품질을 제공합니다.

### 주요 특징

- 🤖 **SOLAR-10.7B**: 한국어 특화 소규모 언어모델 활용
- 🔍 **다중 API 통합**: DataON + ScienceON API 연동
- 📊 **의미적 추천**: 한국어/영어 다국어 임베딩 기반 유사도 계산
- ⚡ **고성능**: vLLM 기반 빠른 추론, Redis 캐싱
- 🎓 **한국어 추천 이유**: 논리적이고 자연스러운 한국어 설명 생성
- 📈 **3단계 추천 레벨**: 강추/추천/참고로 차별화된 추천

## 🏗️ 시스템 아키텍처

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI       │    │  SOLAR-10.7B     │    │  Redis Cache    │
│   API Server    │◄──►│  Language Model  │    │  (Embeddings)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │
         ▼                        ▼
┌─────────────────┐    ┌──────────────────┐
│ Recommendation  │    │  Research Tools  │
│     Agent       │◄──►│  & Embeddings    │
└─────────────────┘    └──────────────────┘
         │                        │
         ▼                        ▼
┌─────────────────┐    ┌──────────────────┐
│   DataON API    │    │  ScienceON API   │
│   (Datasets)    │    │   (Papers)       │
└─────────────────┘    └──────────────────┘
```

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 프로젝트 클론
git clone <repository-url>
cd paper-reco-agent

# Python 가상환경 생성 (Python 3.8+ 필요)
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 환경 변수 설정

```bash
# .env 파일 생성
cp .env.example .env

# .env 파일 편집하여 API 키 설정
nano .env
```

필수 환경 변수:
```env
# API 키
DATAON_SEARCH_KEY=your_dataon_search_key
DATAON_META_KEY=your_dataon_meta_key
SCIENCEON_CLIENT_ID=your_scienceon_client_id
SCIENCEON_ACCOUNTS=your_scienceon_accounts

# 모델 설정
MODEL_NAME=upstage/SOLAR-10.7B-Instruct-v1.0
USE_VLLM=true
QUANTIZATION=fp16
```

### 3. Redis 설정 (선택사항)

캐싱 성능 향상을 위해 Redis 설치를 권장합니다:

```bash
# Docker로 Redis 실행
docker run -d -p 6379:6379 redis:latest

# 또는 시스템에 직접 설치
# macOS: brew install redis
# Ubuntu: sudo apt-get install redis-server
```

### 4. 서버 실행

```bash
# 개발 모드
python main.py

# 또는 uvicorn 직접 사용
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

서버가 시작되면 http://localhost:8000 에서 접근 가능합니다.

## 📖 API 사용법

### 추천 요청

```bash
curl -X POST "http://localhost:8000/recommend" \
     -H "Content-Type: application/json" \
     -d '{"dataset_id": "KISTI_DATA_12345", "max_recommendations": 5}'
```

### API 응답 예시

```json
{
  "source_dataset": {
    "id": "KISTI_DATA_12345",
    "title": "COVID-19 관련 한국인 생활패턴 데이터",
    "description": "코로나19 팬데믹 기간 중 한국인들의 생활패턴 변화를 조사한 데이터셋...",
    "keywords": ["COVID-19", "생활패턴", "한국인", "팬데믹"]
  },
  "recommendations": [
    {
      "type": "paper",
      "title": "COVID-19가 한국 가구의 소비패턴에 미친 영향",
      "description": "본 연구는 코로나19 팬데믹이 한국 가구의 소비 행태에 미친 영향을 분석...",
      "score": 0.89,
      "reason": "공통 키워드 'COVID-19', '한국인'으로 높은 연관성; 동일 연구 분야(사회과학) 소속; 최근 연구로 시의성 높음",
      "level": "강추",
      "url": "http://click.ndsl.kr/servlet/OpenAPIDetailView?..."
    }
  ],
  "processing_time_ms": 1847,
  "candidates_analyzed": 25,
  "model_info": {
    "model_name": "upstage/SOLAR-10.7B-Instruct-v1.0",
    "use_vllm": true,
    "quantization": "fp16"
  }
}
```

### 주요 엔드포인트

- `POST /recommend` - 추천 요청
- `GET /health` - 서버 상태 확인
- `GET /models/info` - 모델 정보 조회
- `GET /api/test/dataon/{dataset_id}` - DataON API 테스트
- `GET /api/test/scienceon` - ScienceON API 테스트
- `GET /docs` - Swagger API 문서

## ⚙️ 설정 옵션

### 모델 설정

다양한 모델과 설정을 지원합니다:

```env
# 권장 설정 (SOLAR-10.7B)
MODEL_NAME=upstage/SOLAR-10.7B-Instruct-v1.0
USE_VLLM=true
QUANTIZATION=fp16
GPU_MEMORY_UTILIZATION=0.9

# 메모리 절약 설정
QUANTIZATION=int8
GPU_MEMORY_UTILIZATION=0.7

# 다른 한국어 모델 옵션
MODEL_NAME=yanolja/EEVE-Korean-Instruct-10.8B-v1.0
MODEL_NAME=beomi/Llama-3-Open-Ko-8B-Instruct
```

### 성능 최적화

```env
# vLLM 사용 (권장)
USE_VLLM=true

# 응답 길이 제한 (토큰 절약)
MAX_TOKENS=512

# 창의성 조절 (일관성 중시)
TEMPERATURE=0.1
```

## 🧪 테스트

### API 연결 테스트

```bash
# DataON API 테스트
curl "http://localhost:8000/api/test/dataon/SAMPLE_ID"

# ScienceON API 테스트
curl "http://localhost:8000/api/test/scienceon"

# 서버 상태 확인
curl "http://localhost:8000/health"
```

### 단위 테스트 실행

```bash
# 전체 테스트
pytest

# 특정 테스트
pytest tests/test_dataon_api.py
pytest tests/test_recommendation_agent.py
```

## 📊 성능 벤치마크

### 하드웨어 요구사양

| 구성 요소 | 최소 사양 | 권장 사양 |
|-----------|-----------|-----------|
| GPU | RTX 3080 (10GB) | RTX 4090 (24GB) |
| RAM | 16GB | 32GB |
| 저장공간 | 50GB | 100GB |

### 성능 지표

- **응답 시간**: 1.5-2.5초 (5개 추천 기준)
- **메모리 사용량**: 18-22GB (SOLAR-10.7B FP16)
- **처리량**: 분당 30-50 요청
- **정확도**: nDCG@10 > 0.75 (내부 평가)

## 🔧 개발자 가이드

### 프로젝트 구조

```
paper-reco-agent/
├── agents/                     # 추천 에이전트
│   └── recommendation_agent.py
├── clients/                    # API 클라이언트
│   ├── dataon_client.py
│   └── scienceon_client.py
├── config/                     # 설정 파일
│   └── settings.py
├── models/                     # 언어모델 래퍼
│   └── solar_model.py
├── tools/                      # 유틸리티 도구
│   └── research_tools.py
├── tests/                      # 테스트 코드
├── logs/                       # 로그 파일
├── main.py                     # FastAPI 서버
├── requirements.txt            # 의존성
├── .env.example               # 환경변수 템플릿
└── README.md                  # 문서
```

### 새로운 모델 추가

1. `models/` 디렉토리에 새 모델 클래스 추가
2. `config/settings.py`에서 모델 설정 추가
3. `agents/recommendation_agent.py`에서 모델 통합

### 새로운 API 추가

1. `clients/` 디렉토리에 새 API 클라이언트 추가
2. `tools/research_tools.py`에 도구 함수 추가
3. `agents/recommendation_agent.py`에서 통합

## 🚨 문제 해결

### 일반적인 문제

**Q: CUDA 메모리 부족 오류**
```bash
# 해결 방법: 양자화 사용
export QUANTIZATION=int8
# 또는 GPU 메모리 사용량 조절
export GPU_MEMORY_UTILIZATION=0.7
```

**Q: 모델 로딩이 느림**
```bash
# 해결 방법: 모델 캐시 디렉토리 설정
export MODEL_CACHE_DIR=/fast/ssd/models
```

**Q: API 호출 실패**
```bash
# 해결 방법: API 키 확인
curl "http://localhost:8000/api/test/dataon/TEST_ID"
curl "http://localhost:8000/api/test/scienceon"
```

### 로그 확인

```bash
# 실시간 로그 모니터링
tail -f logs/app.log

# 에러 로그만 필터링
grep "ERROR" logs/app.log
```

## 📈 평가 지표

시스템은 다음 지표로 평가됩니다:

- **nDCG@10**: 상위 10개 추천의 정규화된 할인 누적 이득
- **MRR@10**: 상위 10개 추천의 평균 역순위
- **Recall@k**: k개 추천에서의 재현율
- **응답 시간**: 요청부터 응답까지의 처리 시간
- **추천 이유 품질**: 전문가 정성 평가

## 🤝 기여 방법

1. 이슈 등록 또는 기능 제안
2. Fork 후 feature 브랜치 생성
3. 코드 작성 및 테스트 추가
4. Pull Request 제출

## 📄 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.

## 🙏 감사의 말

- [Upstage](https://www.upstage.ai/) - SOLAR 모델 제공
- [KISTI](https://www.kisti.re.kr/) - DataON 및 ScienceON API
- [Hugging Face](https://huggingface.co/) - 모델 허브 및 Transformers
- [vLLM](https://github.com/vllm-project/vllm) - 고성능 추론 엔진

---

**📧 Contact**: 문의사항이 있으시면 이슈를 등록해주세요.

**🔗 Demo**: [http://localhost:8000/docs](http://localhost:8000/docs) (서버 실행 후 접근)