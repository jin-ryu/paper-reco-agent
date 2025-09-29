#!/bin/bash

# Korean Research Recommendation Agent 서버 실행 스크립트

set -e

echo "🚀 Korean Research Recommendation Agent 서버 시작"

# 가상환경 활성화 확인
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  가상환경이 활성화되지 않았습니다."
    echo "다음 명령어로 가상환경을 활성화해주세요:"
    echo "source venv/bin/activate"
    exit 1
fi

# 환경변수 파일 확인
if [[ ! -f ".env" ]]; then
    echo "⚠️  .env 파일이 없습니다."
    echo "다음 명령어로 .env 파일을 생성해주세요:"
    echo "cp .env.example .env"
    echo "그 후 API 키를 설정해주세요."
    exit 1
fi

# 의존성 설치 확인
echo "📦 의존성 확인 중..."
pip install -r requirements.txt

# 로그 디렉토리 생성
mkdir -p logs

# Redis 연결 확인 (선택사항)
if command -v redis-cli &> /dev/null; then
    if redis-cli ping &> /dev/null; then
        echo "✅ Redis 연결 확인"
    else
        echo "⚠️  Redis 서버가 실행되지 않고 있습니다. 성능 향상을 위해 Redis를 실행하는 것을 권장합니다."
        echo "Docker: docker run -d -p 6379:6379 redis:latest"
    fi
fi

# GPU 메모리 확인
if command -v nvidia-smi &> /dev/null; then
    echo "🖥️  GPU 정보:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
fi

# 서버 시작
echo "🌟 서버 시작 중..."
echo "📖 API 문서: http://localhost:8000/docs"
echo "💚 상태 확인: http://localhost:8000/health"
echo ""
echo "서버를 중지하려면 Ctrl+C를 누르세요"

python main.py