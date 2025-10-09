#!/bin/bash
# FastAPI 서버 실행 스크립트
# 2025 DATA·AI 분석 경진대회 제출용

echo "======================================"
echo "Research Recommendation Agent"
echo "FastAPI 서버 실행"
echo "======================================"
echo ""

# 환경 변수 확인
if [ ! -f .env ]; then
    echo "❌ .env 파일이 없습니다."
    echo "   scripts/setup_environment.sh를 먼저 실행하세요."
    exit 1
fi

# 포트 설정 (기본값: 8000)
PORT=${1:-8000}

echo "서버 설정:"
echo "  - 호스트: 0.0.0.0"
echo "  - 포트: $PORT"
echo ""

# 서버 시작
echo "🚀 서버 시작 중..."
echo "   (중지하려면 Ctrl+C 누르세요)"
echo ""

python main.py

# 또는 uvicorn 직접 사용
# uvicorn main:app --host 0.0.0.0 --port $PORT --reload
