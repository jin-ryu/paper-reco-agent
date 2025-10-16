# Research Recommendation Agent

**2025 DATA·AI 분석 경진대회 - 논문·데이터 추천 에이전트**
 
LLM 기반 다국어 하이브리드 연구 데이터 및 논문 추천 시스템

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

- ✅ **소규모 언어모델**: Gemma-2-9B-IT (9B 파라미터) 또는 Qwen3-14B (14.8B 파라미터)
- ✅ **고사양 H/W 지원**: FP16 정밀도로 ~18GB (Gemma) 또는 ~28GB (Qwen) VRAM에서 동작
- ✅ **짧은 응답시간**: 하이브리드 RAG로 3-4초 이내 응답
- ✅ **낮은 실패율**: 임베딩 사전 필터링 + 재시도 로직 + description 길이 제한으로 안정적 추론
- ✅ **네트워크 제약**: DataON/ScienceON API, LLM Endpoint만 사용
- ✅ **3-5건 추천**: LLM이 순위 결정, 강추/추천/참고 레벨 구분

### 주요 특징

- 🤖 **멀티 모델 지원**:
  - **Gemma-2-9B-IT** (기본): Google의 9B 파라미터 모델, 8K 컨텍스트
  - **Qwen3-14B**: Alibaba Cloud의 14.8B 파라미터 모델, 32K 컨텍스트
  - 100+ 언어 지원 (영어, 중국어, 한국어, 일본어 등)
  - 뛰어난 reasoning 및 instruction following 성능

- 🔍 **LLM 기반 검색 쿼리 생성**: 의미적 이해를 통한 최적 검색어 추출
- 🌏 **하이브리드 RAG**: LLM 쿼리 생성 + E5 임베딩 + BM25 + LLM 재순위화
- ⚡ **최적화**:
  - FP16 정밀도, query/passage 구분 인코딩
  - 재시도 로직 (최대 2회)
  - Description 길이 제한 (1000자)으로 context overflow 방지
  - 키워드 전처리로 특수문자 제거 및 중복 제거

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
├── CONTEST.md                 # 대회 관련 추가 정보
├── .env                       # 환경 변수 (API 키, .env.example을 복사하여 사용)
│
├── data/                      # 데이터 폴더
│   └── test/                  # 테스트 데이터셋
│       └── testset_gemini_ver1.json  # 평가용 테스트 데이터
│
├── model/                     # 학습된 모델 파일 (자동 다운로드)
│
├── logs/                      # 로그 파일 저장 폴더
│
├── src/                       # 소스코드 폴더
│   ├── __init__.py
│   ├── agents/                # 추천 에이전트
│   │   ├── __init__.py
│   │   └── recommendation_agent.py  # 메인 추천 로직
│   ├── clients/               # API 클라이언트
│   │   ├── __init__.py
│   │   ├── dataon_client.py   # DataON API
│   │   └── scienceon_client.py  # ScienceON API
│   ├── config/                # 설정 파일
│   │   ├── __init__.py
│   │   └── settings.py        # 환경 설정
│   ├── evaluation/            # 평가 모듈
│   │   ├── __init__.py
│   │   └── metrics.py         # nDCG, MRR, Recall@k 등
│   ├── models/                # 언어모델 래퍼
│   │   ├── __init__.py
│   │   ├── llm_model.py       # 범용 LLM 모델 (Gemma/Qwen 지원)
│   │   ├── prompts.py         # LLM 프롬프트 템플릿
│   │   └── mock_model.py      # 개발용 Mock 모델
│   ├── router/                # FastAPI 라우터
│   │   ├── __init__.py
│   │   └── main.py            # FastAPI 서버
│   ├── tools/                 # 유틸리티 도구
│   │   ├── __init__.py
│   │   └── research_tools.py  # 검색, 임베딩, 유사도 계산
│   └── utils/                 # 공통 유틸리티
│       ├── __init__.py
│       └── text_utils.py      # 텍스트 정제 및 키워드 전처리
│
├── notebooks/                 # Jupyter 노트북
│   ├── inference.ipynb        # ⭐ 추론 실행 노트북
│   └── evaluation.ipynb       # ⭐ 평가 실행 노트북
│
├── scripts/                   # 실행 스크립트
│   └── setup_environment.sh   # 환경 설정 스크립트
│
├── demo/                      # 데모 영상
├── figures/                   # 결과 저장 폴더
│   ├── inference_results/     # 추론 결과 (타임스탬프별)
│   └── evaluation_results/    # 평가 결과 (타임스탬프별)
│
└── requirements.txt           # 의존성 목록 (상세 버전)
```

### 주요 파일 설명

- **notebooks/inference.ipynb**: 추론 실행용 Jupyter 노트북
- **notebooks/evaluation.ipynb**: 평가 실행용 Jupyter 노트북
- **src/router/main.py**: FastAPI 서버 엔드포인트
- **src/agents/recommendation_agent.py**: 메인 추천 에이전트 로직
- **src/models/llm_model.py**: 범용 LLM 모델 래퍼 (Gemma/Qwen 지원)
- **src/models/prompts.py**: LLM 프롬프트 템플릿 (쿼리 생성, 재순위화)
- **src/tools/research_tools.py**: 검색, 임베딩, 유사도 계산 함수
- **src/utils/text_utils.py**: 텍스트 정제 및 키워드 전처리 함수
- **src/evaluation/metrics.py**: 평가 메트릭 (nDCG, MRR, Recall@k 등)
- **scripts/setup_environment.sh**: Python 가상환경 및 의존성 설치 스크립트
- **requirements.txt**: 전체 의존성 목록 (pip freeze)

---

## 데이터 및 모델

### 사용 모델

#### 1. 언어모델 (선택 가능)

**A. Gemma-2-9B-IT (기본값)**
- **출처**: [google/gemma-2-9b-it](https://huggingface.co/google/gemma-2-9b-it) (Google)
- **파라미터**: 9B
- **컨텍스트**: 8K 토큰
- **언어 지원**: 다국어 (영어, 한국어, 중국어 등)
- **라이선스**: Gemma License (상업적 사용 가능)
- **정밀도**: FP16
- **용량**: ~18GB (FP16)
- **다운로드**: 자동 다운로드됨 (HuggingFace 토큰 필요)

**B. Qwen3-14B (대안)**
- **출처**: [Qwen/Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B) (Alibaba Cloud)
- **파라미터**: 14.8B (13.2B non-embedding)
- **컨텍스트**: 32K 토큰 (확장 시 128K)
- **언어 지원**: 100+ 언어
- **라이선스**: Apache 2.0 (상업적 사용 가능)
- **정밀도**: FP16
- **용량**: ~28GB (FP16)
- **다운로드**: 자동 다운로드됨 (Hugging Face Hub)

**모델 선택 방법**: `.env` 파일에서 `MODEL_NAME` 변수로 설정

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
- **출력 데이터**: JSON 형식 추천 결과 (`figures/inference_results/`)

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

**evaluation.ipynb 노트북**을 사용하여 평가를 실행할 수 있습니다:

```bash
# Jupyter 실행
jupyter notebook notebooks/evaluation.ipynb

# 노트북에서 모든 셀 순차 실행
```

**평가 프로세스**:
1.  **테스트 데이터셋 로드**: `data/test/testset_aug.json` 파일을 로드하여 평가에 사용할 13개의 테스트 케이스를 준비합니다.
2.  **추천 에이전트 초기화**: 설정된 LLM (Gemma/Qwen) 및 임베딩 모델을 로드하여 추천 에이전트를 준비합니다.
3.  **추천 생성 및 평가**: 각 테스트 케이스에 대해 다음을 수행합니다.
    *   에이전트를 통해 논문 및 데이터셋 추천을 생성합니다.
    *   생성된 추천 목록과 정답셋(`candidate_pool`)을 비교하여 평가 메트릭을 계산합니다.
4.  **결과 저장**: 모든 평가가 완료되면, 타임스탬프 기반의 결과 폴더(`figures/evaluation_results/<timestamp>/`)에 아래 파일들을 저장합니다.
5.  **리포트 생성**: 전체 결과를 종합하여 가독성 높은 요약 리포트를 생성하고 출력합니다.

**평가 지표**:
- **nDCG@k**: Normalized Discounted Cumulative Gain (순위 품질)
- **MRR@k**: Mean Reciprocal Rank (첫 관련 아이템 순위)
- **Recall@k**: 재현율 (관련 아이템 찾은 비율)
- **Precision@k**: 정밀도 (추천 중 관련 아이템 비율)

**평가 결과 파일**:
- `figures/evaluation_results/<timestamp>/EVALUATION_SUMMARY.txt`: ⭐ **(가장 먼저 확인)** 평가 설정, 카테고리별/종합 성능, 상위/하위 케이스 분석 등 핵심 결과를 요약한 리포트입니다.
- `figures/evaluation_results/<timestamp>/detailed_results.csv`: 각 테스트 케이스별 추천 결과 및 정답 여부를 포함한 상세 데이터 (CSV 형식).
- `figures/evaluation_results/<timestamp>/metrics.json`: 전체 테스트 케이스에 대한 평균 평가 지표(nDCG, MRR 등)를 저장한 파일.
- `figures/evaluation_results/<timestamp>/recommend_result.json`: 에이전트가 생성한 원본 추천 결과(JSON 형식) 전체를 저장한 파일.

### 방법 2: 자동 환경 설정 스크립트 (⭐ 권장)

프로젝트 환경을 자동으로 설정하는 스크립트를 제공합니다:

```bash
# 환경 설정 스크립트 실행 (최초 1회)
bash scripts/setup_environment.sh
```

**스크립트가 수행하는 작업**:
1. conda 설치 확인
2. `paper-agent` conda 환경 생성 (Python 3.10)
3. pip 업그레이드
4. requirements.txt에서 의존성 설치
5. CUDA 사용 가능 여부 확인
6. .env 템플릿 파일 생성 (없는 경우)

**환경 설정 후**:
```bash
# 1. .env 파일에 API 키 입력
nano .env

# 2. 환경 활성화
conda activate paper-agent

# 3. Jupyter 노트북 실행
jupyter notebook notebooks/inference.ipynb
```

### 방법 3: 수동 환경 설정

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

### 추론 결과 예시

```json
{
  "source_dataset": {
    "id": "KISTI_DATA_12345",
    "title": "COVID-19 관련 한국인 생활패턴 데이터",
    "description": "COVID-19 팬데믹 기간 동안 수집된...",
    "keywords": ["COVID-19", "생활패턴", "한국인", "팬데믹"]
  },
  "search_result": {
    "paper_keywords": ["COVID-19", "생활패턴", "한국인", "팬데믹"],
    "dataset_keywords": ["COVID-19", "데이터", "설문조사"],
    "paper_search_details": [
      {"keyword": "COVID-19", "count": 5},
      {"keyword": "생활패턴", "count": 5}
    ],
    "dataset_search_details": [
      {"keyword": "COVID-19", "count": 5},
      {"keyword": "데이터", "count": 5}
    ]
  },
  "paper_recommendations": [
    {
      "rank": 1,
      "type": "paper",
      "id": "DIKO0012345678",
      "platform": "scienceon",
      "title": "COVID-19가 한국 가구의 소비패턴에 미친 영향",
      "description": "COVID-19 팬데믹이 한국 가구의 소비 행태에 미친 영향을 분석...",
      "keywords": ["COVID-19", "소비패턴", "한국", "가구"],
      "score": 0.892,
      "reason": "공통 키워드 'COVID-19', '한국인'으로 높은 연관성; 동일 연구 분야",
      "level": "강추",
      "url": "http://click.ndsl.kr/servlet/OpenAPIDetailView?..."
    }
  ],
  "dataset_recommendations": [
    {
      "rank": 1,
      "type": "dataset",
      "id": "a27774ddf0c702847a996cee9d660ba4",
      "platform": "dataon",
      "title": "팬데믹 시기 생활 변화 설문조사 데이터",
      "description": "2020-2021년 팬데믹 기간 동안 수집된 생활패턴 변화 설문 데이터...",
      "keywords": ["팬데믹", "생활패턴", "설문조사"],
      "score": 0.784,
      "reason": "관련 주제 및 데이터 유형; 시간적 연관성 높음",
      "level": "추천",
      "url": "https://dataon.kisti.re.kr/..."
    }
  ],
  "processing_time_ms": 4230,
  "candidates_analyzed": 30,
  "model_info": {
    "model_name": "google/gemma-2-9b-it",
    "model_type": "Gemma",
    "device": "cuda",
    "dtype": "float16",
    "max_tokens": 512,
    "temperature": 0.1,
    "parameters": "9B",
    "context_length": "8K"
  },
  "embedding_model_info": {
    "embedding_model": "intfloat/multilingual-e5-large",
    "paper_hybrid_weights": {
      "alpha": 0.8,
      "beta": 0.2
    },
    "dataset_hybrid_weights": {
      "alpha": 0.6,
      "beta": 0.4
    }
  }
}
```

**추천 결과 필드 설명**:
- `source_dataset`: 입력 데이터셋 정보
- `search_result`: LLM이 생성한 검색 키워드 및 검색 결과 상세
  - `paper_keywords`: 논문 검색에 사용된 키워드
  - `dataset_keywords`: 데이터셋 검색에 사용된 키워드
  - `paper_search_details`: 키워드별 검색 결과 개수
  - `dataset_search_details`: 키워드별 검색 결과 개수
- `paper_recommendations`: 논문 추천 리스트
- `dataset_recommendations`: 데이터셋 추천 리스트
- 각 추천 항목:
  - `rank`: 추천 순위 (1부터 시작)
  - `type`: 타입 (`paper` 또는 `dataset`)
  - `id`: 고유 식별자 (논문: cn, 데이터셋: svc_id)
  - `platform`: 출처 플랫폼 (`scienceon` 또는 `dataon`)
  - `title`: 제목
  - `description`: 요약 설명 (최대 200자)
  - `keywords`: 키워드 리스트 (전처리됨)
  - `score`: 하이브리드 유사도 점수 (0.0~1.0)
  - `reason`: LLM이 생성한 추천 이유
  - `level`: 추천 수준 (`강추`, `추천`, `참고`)
  - `url`: 원문 링크
- `model_info`: LLM 모델 정보
- `embedding_model_info`: 임베딩 모델 및 하이브리드 가중치 정보

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
│ 2. LLM 검색 쿼리 생성               │
│    (Gemma-2-9B-IT / Qwen3-14B)     │
│    입력: 소스 제목 + 설명 + 키워드   │
│    출력:                             │
│    - 데이터셋 검색용 쿼리 3-5개      │
│    - 논문 검색용 쿼리 3-5개          │
│    - 키워드 전처리 (특수문자 제거)   │
│    (다국어 의미 이해 기반)           │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 3. API 후보 수집 (병렬, 키워드별)    │
│   ※ 설정 가능 (에이전트 __init__)    │
│   - search_per_keyword = 5          │
│     (키워드당 검색 개수)              │
│   - DataON: 키워드당 5개씩 검색      │
│     → 중복 제거 후 전체 수집         │
│   - ScienceON: 키워드당 5개씩 검색   │
│     → 중복 제거 후 전체 수집         │
│   - 키워드 전처리 적용               │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 4. 하이브리드 유사도 계산 및 필터링  │
│   ※ 설정 가능 (.env 파일)            │
│   - Semantic (E5):                  │
│     PAPER_HYBRID_ALPHA (기본 0.8)   │
│     DATASET_HYBRID_ALPHA (기본 0.6) │
│   - Lexical (BM25):                 │
│     PAPER_HYBRID_BETA (기본 0.2)    │
│     DATASET_HYBRID_BETA (기본 0.4)  │
│   - 계산 방식:                       │
│     * query: 소스 데이터셋           │
│     * passage: 후보 문서             │
│     * 코사인 유사도 (E5)             │
│     * BM25 어휘 매칭 점수            │
│   - 최종 점수로 정렬                 │
│   ※ 설정 가능 (에이전트 __init__)    │
│   → max_paper_candidates (기본 10)  │
│   → max_dataset_candidates (기본 10)│
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 5. LLM 최종 분석                    │
│    (Gemma-2-9B-IT / Qwen3-14B)     │
│   - 상위 10개 후보 정밀 분석         │
│   - Description 길이 제한 (1000자)   │
│   - 유사도 점수 해석                 │
│   - 추천 순위 결정 (rank)            │
│   - num_paper_recommendations개 선별│
│   - num_dataset_recommendations개 선별│
│   - 추천 이유 작성 (다국어)          │
│   - 추천 레벨 결정 (강추/추천/참고)  │
│   - 재시도 로직 (최대 2회)           │
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
  - 논문: E5 80% + BM25 20% (의미 중심, `.env`에서 조정 가능)
  - 데이터셋: E5 60% + BM25 40% (키워드 중심, `.env`에서 조정 가능)
- 🔧 **유연한 검색 설정** (에이전트 클래스에서 조정 가능):
  - `search_per_keyword`: 키워드당 검색 개수 (기본 5개)
  - `max_paper_candidates`: LLM에 보낼 논문 후보 개수 (기본 10개)
  - `max_dataset_candidates`: LLM에 보낼 데이터셋 후보 개수 (기본 10개)
  - 키워드별 검색 → 중복 제거 → 점수 계산 → 상위 N개 선별
- 🛡️ **안정성 개선**:
  - 키워드 전처리 (특수문자 제거, 중복 제거)
  - Description 길이 제한 (1000자)으로 context overflow 방지
  - LLM 재시도 로직 (최대 2회)

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

- **Gemma-2-9B-IT**: Gemma License (Google, 상업적 사용 가능)
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

- **Gemma-2-9B-IT**: https://huggingface.co/google/gemma-2-9b-it
- **Qwen3-14B**: https://huggingface.co/Qwen/Qwen3-14B
- **Multilingual E5**: https://huggingface.co/intfloat/multilingual-e5-large
- **DataON API**: https://dataon.gitbook.io/
- **ScienceON API**: https://scienceon.kisti.re.kr/apigateway/
- **대회 페이지**: https://aida.kisti.re.kr/competition/

