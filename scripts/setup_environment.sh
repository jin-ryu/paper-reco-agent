#!/bin/bash

# ========================================
# Paper Recommendation Agent
# Environment Setup Script
# ========================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
ENV_NAME="paper-agent"
PYTHON_VERSION="3.10"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Paper Recommendation Agent${NC}"
echo -e "${BLUE}Environment Setup${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Step 1: Check if conda is installed
echo -e "${YELLOW}[1/6] Checking conda installation...${NC}"
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: conda is not installed.${NC}"
    echo -e "${RED}Please install Miniconda or Anaconda first.${NC}"
    echo -e "${RED}Download: https://docs.conda.io/en/latest/miniconda.html${NC}"
    exit 1
fi
echo -e "${GREEN}✓ conda found: $(conda --version)${NC}"
echo ""

# Step 2: Check if environment already exists
echo -e "${YELLOW}[2/6] Checking if environment '${ENV_NAME}' exists...${NC}"
if conda env list | grep -q "^${ENV_NAME} "; then
    echo -e "${YELLOW}Environment '${ENV_NAME}' already exists.${NC}"
    read -p "Do you want to remove and recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Removing existing environment...${NC}"
        conda env remove -n ${ENV_NAME} -y
        echo -e "${GREEN}✓ Environment removed${NC}"
    else
        echo -e "${YELLOW}Skipping environment creation. Using existing environment.${NC}"
        SKIP_ENV_CREATE=true
    fi
fi
echo ""

# Step 3: Create conda environment
if [ "$SKIP_ENV_CREATE" != true ]; then
    echo -e "${YELLOW}[3/6] Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}...${NC}"
    conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y
    echo -e "${GREEN}✓ Environment created${NC}"
else
    echo -e "${YELLOW}[3/6] Skipping environment creation${NC}"
fi
echo ""

# Step 4: Activate environment and install dependencies
echo -e "${YELLOW}[4/6] Installing Python dependencies...${NC}"
echo -e "${BLUE}This may take 5-10 minutes depending on your internet connection.${NC}"

# Get conda base path
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# Activate environment
conda activate ${ENV_NAME}

# Upgrade pip
echo -e "${BLUE}Upgrading pip...${NC}"
pip install --upgrade pip

# Install requirements
echo -e "${BLUE}Installing packages from requirements.txt...${NC}"
cd "${PROJECT_ROOT}"
pip install -r requirements.txt

echo -e "${GREEN}✓ Dependencies installed${NC}"
echo ""

# Step 5: Check CUDA availability
echo -e "${YELLOW}[5/6] Checking CUDA availability...${NC}"
python << EOF
import torch
import sys

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print("\033[0;32m✓ CUDA is properly configured\033[0m")
else:
    print("\033[1;33m⚠ CUDA not available. The model will run on CPU (very slow).\033[0m")
    print("\033[1;33m  Please install CUDA toolkit and compatible PyTorch.\033[0m")
EOF

echo ""

# Step 6: Setup .env file
echo -e "${YELLOW}[6/6] Checking .env configuration...${NC}"
if [ ! -f "${PROJECT_ROOT}/.env" ]; then
    echo -e "${YELLOW}.env file not found. Creating template...${NC}"
    cat > "${PROJECT_ROOT}/.env" << 'ENVFILE'
# ===== API 키 설정 =====
# DataON 및 ScienceON API 키를 설정하세요
DATAON_SEARCH_KEY=your_dataon_search_key_here
DATAON_META_KEY=your_dataon_meta_key_here
SCIENCEON_CLIENT_ID=your_scienceon_client_id_here
SCIENCEON_ACCOUNTS=your_scienceon_accounts_here

# ===== 모델 설정 =====
# 언어모델 선택 (다국어 성능 우수)
# MODEL_NAME=Qwen/Qwen3-14B  # 대안: Qwen3-14B (14.8B, 32K context)
MODEL_NAME=google/gemma-2-9b-it  # 기본: Gemma-2-9B-IT (9B, 8K context)
MODEL_CACHE_DIR=./model

# Hugging Face 토큰 (gated 모델 접근용, Gemma 필수)
HF_TOKEN=your_huggingface_token_here

# LLM 생성 파라미터
MAX_TOKENS=512
TEMPERATURE=0.1

# 개발 모드 (GPU 없을 때 Mock 모델 사용)
DEV_MODE=false

# ===== 임베딩 모델 설정 =====
# E5 모델: 다국어 지원, 한/영 논문 검색 최적화
EMBEDDING_MODEL=intfloat/multilingual-e5-large

# ===== 하이브리드 유사도 가중치 =====
# 논문 가중치
PAPER_HYBRID_ALPHA=0.8
PAPER_HYBRID_BETA=0.2
# 데이터셋 가중치
DATASET_HYBRID_ALPHA=0.6
DATASET_HYBRID_BETA=0.4

# ===== 서버 설정 =====
HOST=0.0.0.0
PORT=8000
DEBUG=false

# ===== 로깅 설정 =====
LOG_LEVEL=INFO
LOG_FILE=logs/app.log

# ===== API URLs =====
DATAON_BASE_URL=https://dataon.kisti.re.kr
SCIENCEON_BASE_URL=https://apigateway.kisti.re.kr
ENVFILE

    echo -e "${GREEN}✓ .env template created${NC}"
    echo -e "${YELLOW}⚠ Please edit .env file and add your API keys!${NC}"
else
    echo -e "${GREEN}✓ .env file already exists${NC}"
fi
echo ""

# Final message
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✓ Setup completed successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo -e "  1. Edit .env file and add your API keys:"
echo -e "     ${YELLOW}nano .env${NC}"
echo -e ""
echo -e "  2. Activate the environment:"
echo -e "     ${YELLOW}conda activate ${ENV_NAME}${NC}"
echo -e ""
echo -e "  3. Run inference notebook:"
echo -e "     ${YELLOW}jupyter notebook notebooks/inference.ipynb${NC}"
echo -e ""
echo -e "${BLUE}For more information, see README.md${NC}"
echo ""
