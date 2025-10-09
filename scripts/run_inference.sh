#!/bin/bash
# 추론 실행 스크립트
# 2025 DATA·AI 분석 경진대회 제출용

echo "======================================"
echo "Research Recommendation Agent"
echo "추론(Inference) 실행"
echo "======================================"
echo ""

# 데이터셋 ID 입력
if [ -z "$1" ]; then
    echo "사용법: ./scripts/run_inference.sh <dataset_id>"
    echo "예시: ./scripts/run_inference.sh KISTI_DATA_12345"
    exit 1
fi

DATASET_ID=$1
echo "데이터셋 ID: $DATASET_ID"
echo ""

# Python 스크립트 실행
echo "추론 시작..."
python -c "
import asyncio
import json
import sys
import os

# 프로젝트 루트 경로 추가
sys.path.insert(0, os.getcwd())

from src.agents.recommendation_agent import KoreanResearchRecommendationAgent

async def main():
    print('🚀 에이전트 초기화 중...')
    agent = KoreanResearchRecommendationAgent()

    print(f'🔍 추천 시작: {sys.argv[1]}')
    result = await agent.recommend(sys.argv[1])

    if 'error' in result:
        print(f'❌ 오류: {result[\"error\"]}')
        sys.exit(1)
    else:
        print(f'✅ 추천 완료: {len(result[\"recommendations\"])}개 추천')
        print(f'처리 시간: {result[\"processing_time_ms\"]}ms')

        # 결과 출력
        print('\\n' + '='*80)
        print('추천 결과')
        print('='*80)
        for rec in result['recommendations']:
            print(f\"\\n[{rec['rank']}위] {rec['level']} - {rec['type'].upper()}\")
            print(f\"제목: {rec['title']}\")
            print(f\"점수: {rec['score']:.3f}\")
            print(f\"이유: {rec['reason']}\")

        # JSON 파일로 저장
        os.makedirs('data/inference_results', exist_ok=True)
        output_file = f'data/inference_results/result_{sys.argv[1]}.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f'\\n✅ 결과 저장: {output_file}')

asyncio.run(main())
" $DATASET_ID

echo ""
echo "======================================"
echo "추론 완료"
echo "======================================"
