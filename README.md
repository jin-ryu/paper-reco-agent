# Research Recommendation Agent

**2025 DATA·AI 분석 경진대회 - 논문·데이터 추천 에이전트**

Qwen3-14B 기반 다국어 하이브리드 연구 데이터 및 논문 추천 시스템

---

## 목차

1. [프로젝트 개요](#프로젝트-개요)
2. [실행 환경 (HW/SW)](#실행-환경-hwsw)
3. [프로젝트 구조](#프로젝트-구조)
4. [데이터 및 모델](#데이터-및-모델)
5. [추론 수행 방법](#추론-수행-방법)
6. [시스템 아키텍처](#시스템-아키텍처)
7. [성능 평가](#성능-평가)
8. [저작권 및 라이선스](#저작권-및-라이선스)

---

## 프로젝트 개요

이 프로젝트는 **KISTI 2025 DATA·AI 분석 경진대회**를 위해 개발된 AI 추천 에이전트입니다. DataON에 등록된 연구 데이터셋을 입력받아 의미적으로 관련성이 높은 연구논문과 데이터셋을 추천합니다.

### 대회 요구사항 충족

- ✅ **소규모 언어모델**: Qwen3-14B (14.8B 파라미터)
- ✅ **고사양 H/W 지원**: FP16 정밀도로 ~28GB VRAM에서 동작
- ✅ **짧은 응답시간**: 하이브리드 RAG로 3-4초 이내 응답
- ✅ **낮은 실패율**: 임베딩 사전 필터링 + 재시도 로직으로 안정적 추론
- ✅ **네트워크 제약**: DataON/ScienceON API, LLM Endpoint만 사용
- ✅ **3-5건 추천**: LLM이 순위 결정, 강추/추천/참고 레벨 구분

### 주요 특징

- 🤖 **Qwen3-14B**: 다국어 고성능 언어모델 (Alibaba Cloud)
  - 100+ 언어 지원 (영어, 중국어, 한국어, 일본어 등)
  - 32K 토큰 컨텍스트 (확장 시 128K)
  - 뛰어난 reasoning 및 instruction following 성능

- 🔍 **LLM 기반 검색 쿼리 생성**: 의미적 이해를 통한 최적 검색어 추출
- 🌏 **하이브리드 RAG**: LLM 쿼리 생성 + E5 임베딩 + BM25 + LLM 재순위화
- ⚡ **최적화**: FP16 정밀도, query/passage 구분 인코딩, 재시도 로직

---

## 실행 환경 (HW/SW)

### 하드웨어(HW) 요구사항

#### 최소 사양
- **GPU**: NVIDIA RTX 3090 24GB 이상
- **RAM**: 32GB 이상
- **저장공간**: 50GB 이상 (모델 캐시 포함)
- **OS**: Linux (Ubuntu 20.04+) 또는 Windows 10/11

#### 권장 사양
- **GPU**: NVIDIA RTX 4090 24GB 이상 또는 H100 80GB
- **RAM**: 64GB 이상
- **저장공간**: 100GB 이상
- **OS**: Linux (Ubuntu 22.04)

#### 개발 및 테스트 환경
- **GPU**: NVIDIA H100 80GB HBM3
- **NVIDIA Driver**: 535.104.05
- **OS**: Linux Ubuntu 22.04
- **RAM**: 64GB+

### 소프트웨어(SW) 요구사항

#### 필수 소프트웨어

**1. Python**
- 버전: Python 3.10.18
- 설치:
  ```bash
  conda create -n paper-agent python=3.10
  conda activate paper-agent
  ```

**2. CUDA Toolkit**
- **CUDA 버전**: 12.8
- **cuDNN 버전**: 9.1.0.02 (91002)
- **PyTorch CUDA**: 12.8 (cu128)
- 설치 가이드: https://developer.nvidia.com/cuda-12-8-0-download-archive

**3. NVIDIA Driver**
- **버전**: 535.104.05 이상
- 설치 가이드: https://www.nvidia.com/download/index.aspx

**CUDA 확인**:
```bash
nvidia-smi
nvcc --version
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Version: {torch.version.cuda}')"
```

#### 주요 라이브러리 버전

```
PyTorch: 2.8.0+cu128
Transformers: 4.56.2
Sentence-Transformers: 5.1.1
FastAPI: 0.104.1
NumPy: 1.24.3
Pandas: 2.1.4
Scikit-learn: 1.3.2
```

**전체 의존성**: `requirements.txt` 참조 (pip freeze 출력)

---

## 프로젝트 구조

```
paper-reco-agent/
├── README.md                  # 이 파일 (프로젝트 전체 설명)
├── .env                       # 환경 변수 (API 키)
│
├── data/                      # 데이터 폴더
│   ├── test/                  # 테스트 데이터셋
│   │   └── testset_gemini_ver1.json  # 평가용 테스트 데이터
│   ├── inference_results/     # 추론 결과 저장 위치
│   └── evaluation_results/    # 평가 결과 저장 위치
│
├── model/                     # 학습된 모델 파일 (자동 다운로드)
│
├── src/                       # 소스코드 폴더
│   ├── agents/                # 추천 에이전트
│   │   └── recommendation_agent.py  # 메인 추천 로직
│   ├── clients/               # API 클라이언트
│   │   ├── dataon_client.py   # DataON API
│   │   └── scienceon_client.py  # ScienceON API
│   ├── config/                # 설정 파일
│   │   └── settings.py        # 환경 설정
│   ├── models/                # 언어모델 래퍼
│   │   ├── qwen_model.py      # Qwen3-14B 모델
│   │   └── mock_model.py      # 개발용 Mock 모델
│   ├── tools/                 # 유틸리티 도구
│   │   └── research_tools.py  # 검색, 임베딩, 유사도 계산
│   ├── evaluation/            # 평가 모듈
│   │   ├── __init__.py        # 평가 함수 export
│   │   └── metrics.py         # nDCG, MRR, Recall@k 등
│   └── main.py                # FastAPI 서버
│
├── notebooks/                 # Jupyter 노트북 (추론 실행)
│   └── inference.ipynb        # ⭐ 추론 + 평가 실행 노트북
│
├── scripts/                   # 실행 스크립트
│   ├── setup_environment.sh   # 환경 설정 스크립트
│   ├── run_inference.sh       # 추론 실행 스크립트
│   ├── test_evaluation.py     # 평가 모듈 테스트
│   └── run_server.sh          # FastAPI 서버 실행 스크립트
│
├── demo/                      # 데모 영상
├── figures/                   # 아키텍처 다이어그램 및 결과 그림
├── tests/                     # 단위 테스트
└── requirements.txt           # 의존성 목록 (상세 버전)
```

### 주요 파일 설명

- **notebooks/inference.ipynb**: 추론 + 평가 실행용 Jupyter 노트북 (필수)
- **src/agents/recommendation_agent.py**: 메인 추천 에이전트 로직
- **src/models/qwen_model.py**: Qwen3-14B 언어모델 래퍼
- **src/tools/research_tools.py**: 검색, 임베딩, 유사도 계산 함수
- **src/evaluation/metrics.py**: 평가 메트릭 (nDCG, MRR, Recall@k 등)
- **scripts/test_evaluation.py**: 평가 모듈 단위 테스트
- **requirements.txt**: 전체 의존성 목록 (pip freeze)

---

## 데이터 및 모델

### 사용 모델

#### 1. 언어모델: Qwen3-14B
- **출처**: [Qwen/Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B) (Alibaba Cloud)
- **파라미터**: 14.8B (13.2B non-embedding)
- **컨텍스트**: 32K 토큰 (확장 시 128K)
- **언어 지원**: 100+ 언어
- **라이선스**: Apache 2.0 (상업적 사용 가능)
- **정밀도**: FP16
- **용량**: ~28GB (FP16)
- **다운로드**: 자동 다운로드됨 (Hugging Face Hub)

#### 2. 임베딩 모델: Multilingual E5
- **출처**: [intfloat/multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large) (Microsoft)
- **차원**: 1024
- **특징**: 다국어 검색 최적화, Query/Passage 구분 인코딩
- **라이선스**: MIT (상업적 사용 가능)
- **용량**: ~2.2GB
- **다운로드**: 자동 다운로드됨

### 데이터

- **입력 데이터**: DataON 데이터셋 ID (예: `KISTI_DATA_12345`)
- **API 데이터**:
  - DataON API: 데이터셋 메타데이터 및 검색
  - ScienceON API: 논문 검색 및 메타데이터
- **출력 데이터**: JSON 형식 추천 결과 (`data/inference_results/`)

**데이터 저작권**: 경진대회 제공 API만 사용, 저작권 문제 없음

---

## 추론 수행 방법

### 방법 1: Jupyter Notebook (⭐ 권장)

**대회 심사자용 추론 실행 방법**:

```bash
# 1. 환경 활성화
conda activate paper-agent

# 2. Jupyter 실행
jupyter notebook notebooks/inference.ipynb

# 3. 노트북 셀 순차적으로 실행
#    - 데이터셋 ID만 변경하면 됨 (test_dataset_id 변수)
```

**노트북 실행 흐름**:
1. 환경 변수 로드 및 GPU 확인
2. 추천 에이전트 초기화 (모델 로드)
3. 데이터셋 ID 설정
4. 추론 실행 (`await agent.recommend(dataset_id)`)
5. 결과 확인 및 JSON 파일로 저장
6. **평가**: 테스트 데이터셋으로 성능 평가 (nDCG, MRR, Recall@k)

**주석**: 노트북 내 각 셀에 상세한 주석 포함

### 평가 실행

노트북의 평가 섹션(Section 10-15)에서 자동 평가를 실행할 수 있습니다:

```python
# 노트북 내에서 실행
# 1. 테스트 데이터 로드 (data/test/testset_gemini_ver1.json)
# 2. 배치 추론 실행
# 3. 평가 메트릭 계산 (nDCG@k, MRR@k, Recall@k, Precision@k)
# 4. 결과 저장 (data/evaluation_results/)
```

**평가 지표**:
- **nDCG@k**: Normalized Discounted Cumulative Gain (순위 품질)
- **MRR@k**: Mean Reciprocal Rank (첫 관련 아이템 순위)
- **Recall@k**: 재현율 (관련 아이템 찾은 비율)
- **Precision@k**: 정밀도 (추천 중 관련 아이템 비율)

**평가 결과 파일**:
- `data/evaluation_results/detailed_evaluation.json`: 상세 평가 결과
- `data/evaluation_results/evaluation_summary.csv`: 요약 (CSV)
- `data/evaluation_results/average_metrics.json`: 평균 메트릭
- `data/evaluation_results/failed_cases.json`: 실패 케이스

### 방법 2: 실행 스크립트

```bash
# 환경 설정 (최초 1회)
bash scripts/setup_environment.sh

# 추론 실행
bash scripts/run_inference.sh <dataset_id>

# 예시
bash scripts/run_inference.sh KISTI_DATA_12345
```

### 방법 3: FastAPI 서버

```bash
# 서버 시작
bash scripts/run_server.sh

# API 호출
curl -X POST "http://localhost:8000/recommend" \
     -H "Content-Type: application/json" \
     -d '{"dataset_id": "SAMPLE_ID", "max_recommendations": 5}'

# 또는 Swagger UI에서 테스트
# http://localhost:8000/docs
```

### 환경 설정

**1. 프로젝트 클론**
```bash
git clone <repository-url>
cd paper-reco-agent
```

**2. Conda 환경 생성**
```bash
conda create -n paper-agent python=3.10
conda activate paper-agent
```

**3. 의존성 설치**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**4. 환경 변수 설정**
```bash
# .env 파일에 API 키 입력
nano .env
```

`.env` 파일 내용:
```env
# DataON API 키
DATAON_SEARCH_KEY=your_key_here
DATAON_META_KEY=your_key_here

# ScienceON API 키
SCIENCEON_CLIENT_ID=your_client_id_here
SCIENCEON_ACCOUNTS=your_accounts_here

# 모델 설정
MODEL_NAME=Qwen/Qwen3-14B
EMBEDDING_MODEL=intfloat/multilingual-e5-large

# 개발 모드 (GPU 없을 때 Mock 모델 사용)
DEV_MODE=false
```

### 추론 결과 예시

```json
{
  "source_dataset": {
    "id": "KISTI_DATA_12345",
    "title": "COVID-19 관련 한국인 생활패턴 데이터",
    "keywords": ["COVID-19", "생활패턴", "한국인", "팬데믹"]
  },
  "recommendations": [
    {
      "rank": 1,
      "type": "paper",
      "title": "COVID-19가 한국 가구의 소비패턴에 미친 영향",
      "score": 0.892,
      "reason": "공통 키워드 'COVID-19', '한국인'으로 높은 연관성; 동일 연구 분야",
      "level": "강추",
      "url": "http://click.ndsl.kr/servlet/OpenAPIDetailView?..."
    },
    {
      "rank": 2,
      "type": "dataset",
      "title": "팬데믹 시기 생활 변화 설문조사 데이터",
      "score": 0.784,
      "reason": "관련 주제 및 데이터 유형; 시간적 연관성 높음",
      "level": "추천",
      "url": "https://dataon.kisti.re.kr/..."
    }
  ],
  "processing_time_ms": 4230,
  "candidates_analyzed": 30,
  "model_info": {
    "model_name": "Qwen/Qwen3-14B",
    "parameters": "14.8B",
    "dtype": "float16"
  }
}
```

---

## 시스템 아키텍처

### LLM 기반 하이브리드 RAG 파이프라인

```
입력: dataset_id
    │
    ▼
┌─────────────────────────────────────┐
│ 1. DataON API: 소스 데이터셋 조회    │
│    → 제목, 설명, 키워드 추출         │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 2. LLM 검색 쿼리 생성 (Qwen3-14B)   │
│    입력: 소스 제목 + 설명 + 키워드   │
│    출력:                             │
│    - 데이터셋 검색용 쿼리 3-5개      │
│    - 논문 검색용 쿼리 3-5개          │
│    (다국어 의미 이해 기반)           │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 3. API 후보 수집 (병렬)              │
│   - DataON: LLM 생성 쿼리로 검색     │
│     (15개 데이터셋)                  │
│   - ScienceON: LLM 생성 쿼리로 검색  │
│     (15개 논문)                      │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 4. 하이브리드 유사도 계산 및 필터링  │
│   - Semantic (E5): 70%              │
│     * query: 소스 데이터셋           │
│     * passage: 후보 문서             │
│     * 코사인 유사도 계산              │
│   - Lexical (BM25): 30%             │
│     * 어휘 매칭 점수                 │
│   - 최종 점수 = 의미적*0.7 + 어휘*0.3│
│   → 상위 15개 후보 선별              │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 5. LLM 최종 분석 (Qwen3-14B)        │
│   - 15개 후보 정밀 분석              │
│   - 유사도 점수 해석                 │
│   - 추천 순위 결정 (rank)            │
│   - 3-5개 최종 추천 선별             │
│   - 추천 이유 작성 (다국어)          │
│   - 추천 레벨 결정 (강추/추천/참고)  │
└─────────────────────────────────────┘
    │
    ▼
출력: 추천 3-5건 (순위, 제목, 설명, 점수, 이유, 레벨, URL)
```

### 핵심 개선점
- 🎯 **LLM이 검색 쿼리 생성**: 의미적 이해를 통해 최적의 검색어를 생성
- 🔄 **맞춤형 쿼리**: 데이터셋 검색과 논문 검색에 각각 최적화된 쿼리 사용
- ⚡ **빠른 필터링**: 검색 API 응답만으로 유사도 계산, 상세 조회 불필요
- 🎨 **하이브리드 스코어링**: E5 임베딩(의미) + BM25(어휘)로 정확도와 재현율 확보

---

## 성능 평가

### 시스템 성능 (고사양 기준)

| 메트릭 | FP16 (28GB) |
|--------|------------|
| **응답 시간** | 3-4초 |
| **메모리 사용** | ~28GB |
| **처리량** | 20-25 req/min |
| **실패율** | < 0.5% |

**응답 시간 분해**:
- LLM 쿼리 생성: 0.5-1초
- 후보 검색 (병렬): 0.5-1초
- 유사도 계산: 0.3-0.5초
- LLM 최종 분석: 1.5-2초
- **총합**: 3-4초 (FP16 기준)

### 추천 품질 평가 지표

시스템은 다음 지표로 평가됩니다:
- **nDCG@k**: Normalized Discounted Cumulative Gain (순위 품질, 0~1)
  - 1.0에 가까울수록 이상적인 순위
  - 상위 k개 추천의 정규화된 할인 누적 이득
- **MRR@k**: Mean Reciprocal Rank (첫 관련 아이템 순위, 0~1)
  - 첫 번째 관련 아이템의 역순위 (1/rank)
  - 1.0: 첫 번째 추천이 관련 있음
- **Recall@k**: 재현율 (0~1)
  - 전체 관련 아이템 중 상위 k개 추천에 포함된 비율
- **Precision@k**: 정밀도 (0~1)
  - 상위 k개 추천 중 실제 관련 있는 아이템의 비율

**평가 모듈**: `src/evaluation/metrics.py`에 구현됨

**평가 실행 방법**:
```bash
# 평가 모듈 단위 테스트
python scripts/test_evaluation.py

# 전체 평가 (Jupyter Notebook)
jupyter notebook notebooks/inference.ipynb
# → Section 10-15 실행
```

---

## 저작권 및 라이선스

### 오픈소스 라이브러리

본 프로젝트는 다음 오픈소스 라이브러리를 사용합니다:

- **Qwen3-14B**: Apache 2.0 License (Alibaba Cloud)
- **PyTorch**: BSD License
- **Transformers**: Apache 2.0 License (Hugging Face)
- **FastAPI**: MIT License
- **E5 Embeddings**: MIT License (Microsoft)

**모든 라이브러리는 상업적 사용이 가능한 라이선스입니다.**

### API 사용

- **DataON API**: KISTI 경진대회 제공 API 사용
- **ScienceON API**: KISTI 경진대회 제공 API 사용
- **저작권 문제 없음**: 경진대회 제공 데이터 및 API만 사용

## 참고 자료

- **Qwen3-14B**: https://huggingface.co/Qwen/Qwen3-14B
- **Multilingual E5**: https://huggingface.co/intfloat/multilingual-e5-large
- **DataON API**: https://dataon.gitbook.io/
- **ScienceON API**: https://scienceon.kisti.re.kr/apigateway/
- **대회 페이지**: https://aida.kisti.re.kr/competition/

