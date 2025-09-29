#!/bin/bash

# 개발 모드 (GPU 없이) 서버 실행 스크립트

set -e

echo "🎭 Korean Research Recommendation Agent - 개발 모드"
echo "GPU 없이 Mock 모델로 실행됩니다"

# 개발 모드 환경변수 설정
export DEV_MODE=true
export USE_VLLM=false
export MODEL_NAME=mock-solar
export REDIS_HOST=localhost

# 가상환경 확인
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  가상환경이 활성화되지 않았습니다."
    echo "다음 명령어로 가상환경을 활성화해주세요:"
    echo "python -m venv venv && source venv/bin/activate"
    exit 1
fi

# 개발용 의존성 설치
echo "📦 개발용 의존성 설치 중..."
pip install -r requirements-dev.txt

# 환경변수 파일 확인 및 생성
if [[ ! -f ".env" ]]; then
    echo "📝 개발용 .env 파일 생성 중..."
    cat > .env << EOL
# 개발 모드 설정
DEV_MODE=true
MODEL_NAME=mock-solar
USE_VLLM=false

# Mock API 키 (실제 키 없이도 테스트 가능)
DATAON_SEARCH_KEY=mock_search_key
DATAON_META_KEY=mock_meta_key
SCIENCEON_CLIENT_ID=mock_client_id
SCIENCEON_ACCOUNTS=mock_accounts

# 서버 설정
HOST=127.0.0.1
PORT=8000
DEBUG=true
LOG_LEVEL=INFO

# Redis (선택사항)
REDIS_HOST=localhost
REDIS_PORT=6379
EOL
    echo "✅ 개발용 .env 파일 생성 완료"
fi

# 로그 디렉토리 생성
mkdir -p logs

echo ""
echo "🌟 개발 서버 시작 중..."
echo "📖 API 문서: http://127.0.0.1:8000/docs"
echo "💚 상태 확인: http://127.0.0.1:8000/health"
echo "🧪 API 테스트: http://127.0.0.1:8000/api/test/dataon/SAMPLE_ID"
echo ""
echo "이 모드에서는 실제 AI 모델 대신 Mock 데이터를 사용합니다."
echo "API 구조와 로직 테스트용입니다."
echo ""
echo "서버를 중지하려면 Ctrl+C를 누르세요"

python main.py