#!/bin/bash
# 환경 설정 스크립트
# 2025 DATA·AI 분석 경진대회 제출용

echo "======================================"
echo "Research Recommendation Agent"
echo "환경 설정 스크립트"
echo "======================================"
echo ""

# Python 버전 확인
echo "[1/5] Python 버전 확인..."
python --version
if [ $? -ne 0 ]; then
    echo "❌ Python이 설치되지 않았습니다. Python 3.10 이상을 설치하세요."
    exit 1
fi
echo "✅ Python 확인 완료"
echo ""

# CUDA 확인
echo "[2/5] CUDA 확인..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
    echo "✅ NVIDIA GPU 감지됨"
else
    echo "⚠️  NVIDIA GPU를 감지할 수 없습니다. CPU 모드로 실행됩니다."
fi
echo ""

# 가상환경 생성 (선택사항)
echo "[3/5] Python 가상환경 생성 (선택사항, 스킵하려면 Ctrl+C)..."
read -p "가상환경을 생성하시겠습니까? (y/n): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python -m venv venv
    source venv/bin/activate
    echo "✅ 가상환경 생성 및 활성화 완료"
else
    echo "⏭️  가상환경 생성 스킵"
fi
echo ""

# 의존성 설치
echo "[4/5] 의존성 패키지 설치..."
pip install --upgrade pip
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "❌ 의존성 설치 실패"
    exit 1
fi
echo "✅ 의존성 설치 완료"
echo ""

# 환경 변수 설정
echo "[5/5] 환경 변수 설정..."
if [ ! -f .env ]; then
    echo "❌ .env 파일이 없습니다."
    echo "   .env.example을 복사하여 .env를 생성하고 API 키를 입력하세요:"
    echo "   cp .env.example .env"
    echo "   nano .env"
    exit 1
else
    echo "✅ .env 파일 확인됨"
fi
echo ""

echo "======================================"
echo "✅ 환경 설정 완료!"
echo "======================================"
echo ""
echo "다음 단계:"
echo "1. .env 파일에서 API 키 확인/수정"
echo "2. python main.py 또는 uvicorn main:app --reload 실행"
echo "3. 또는 notebooks/inference.ipynb 노트북 실행"
echo ""
