"""
추천 시스템 평가 메트릭

- nDCG@k: Normalized Discounted Cumulative Gain
- MRR@k: Mean Reciprocal Rank
- Recall@k: 재현율
- Precision@k: 정밀도
"""

import numpy as np
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


def calculate_dcg(relevance_scores: List[float], k: int = 10) -> float:
    """
    DCG (Discounted Cumulative Gain) 계산

    Args:
        relevance_scores: 순위별 관련성 점수 리스트
        k: 상위 k개까지만 고려

    Returns:
        DCG 점수
    """
    try:
        relevance_scores = relevance_scores[:k]

        if not relevance_scores:
            return 0.0

        # DCG = sum(rel_i / log2(i+1))
        dcg = 0.0
        for i, rel in enumerate(relevance_scores, 1):
            dcg += rel / np.log2(i + 1)

        return dcg
    except Exception as e:
        logger.error(f"DCG 계산 실패: {e}")
        return 0.0


def calculate_ndcg(relevance_scores: List[float], k: int = 10) -> float:
    """
    nDCG@k (Normalized Discounted Cumulative Gain) 계산

    추천 시스템의 순위 품질을 평가하는 지표
    - 1.0에 가까울수록 이상적인 순위
    - 0.0에 가까울수록 좋지 않은 순위

    Args:
        relevance_scores: 순위별 관련성 점수 리스트 (예: [3, 2, 3, 0, 1, 2])
        k: 상위 k개까지만 고려 (default: 10)

    Returns:
        nDCG@k 점수 (0.0 ~ 1.0)
    """
    try:
        if not relevance_scores:
            return 0.0

        # DCG 계산
        dcg = calculate_dcg(relevance_scores, k)

        # IDCG (Ideal DCG) 계산: 이상적인 순위 (내림차순 정렬)
        ideal_relevance = sorted(relevance_scores, reverse=True)
        idcg = calculate_dcg(ideal_relevance, k)

        # nDCG = DCG / IDCG
        if idcg == 0:
            return 0.0

        ndcg = dcg / idcg
        return ndcg

    except Exception as e:
        logger.error(f"nDCG 계산 실패: {e}")
        return 0.0


def calculate_mrr(relevance_scores: List[float], k: int = 10) -> float:
    """
    MRR@k (Mean Reciprocal Rank) 계산

    첫 번째 관련 있는 아이템의 순위의 역수
    - 1.0: 첫 번째 추천이 관련 있음
    - 0.5: 두 번째 추천이 관련 있음
    - 0.0: k개 안에 관련 있는 추천 없음

    Args:
        relevance_scores: 순위별 관련성 점수 리스트
        k: 상위 k개까지만 고려

    Returns:
        MRR@k 점수 (0.0 ~ 1.0)
    """
    try:
        relevance_scores = relevance_scores[:k]

        if not relevance_scores:
            return 0.0

        # 첫 번째 관련 있는 아이템의 순위 찾기 (relevance > 0)
        for i, rel in enumerate(relevance_scores, 1):
            if rel > 0:
                return 1.0 / i

        return 0.0

    except Exception as e:
        logger.error(f"MRR 계산 실패: {e}")
        return 0.0


def calculate_recall_at_k(
    recommended_ids: List[str],
    relevant_ids: List[str],
    k: int = 5
) -> float:
    """
    Recall@k 계산

    전체 관련 있는 아이템 중 상위 k개 추천에 포함된 비율

    Args:
        recommended_ids: 추천된 아이템 ID 리스트
        relevant_ids: 실제 관련 있는 아이템 ID 리스트
        k: 상위 k개까지만 고려

    Returns:
        Recall@k (0.0 ~ 1.0)
    """
    try:
        if not relevant_ids:
            return 0.0

        recommended_set = set(recommended_ids[:k])
        relevant_set = set(relevant_ids)

        # 교집합의 크기 / 전체 관련 아이템 수
        intersection = recommended_set.intersection(relevant_set)
        recall = len(intersection) / len(relevant_set)

        return recall

    except Exception as e:
        logger.error(f"Recall@k 계산 실패: {e}")
        return 0.0


def calculate_precision_at_k(
    recommended_ids: List[str],
    relevant_ids: List[str],
    k: int = 5
) -> float:
    """
    Precision@k 계산

    상위 k개 추천 중 실제 관련 있는 아이템의 비율

    Args:
        recommended_ids: 추천된 아이템 ID 리스트
        relevant_ids: 실제 관련 있는 아이템 ID 리스트
        k: 상위 k개까지만 고려

    Returns:
        Precision@k (0.0 ~ 1.0)
    """
    try:
        if not recommended_ids:
            return 0.0

        recommended_set = set(recommended_ids[:k])
        relevant_set = set(relevant_ids)

        # 교집합의 크기 / 추천된 아이템 수
        intersection = recommended_set.intersection(relevant_set)
        precision = len(intersection) / min(len(recommended_set), k)

        return precision

    except Exception as e:
        logger.error(f"Precision@k 계산 실패: {e}")
        return 0.0


def evaluate_recommendations(
    predictions: List[Dict[str, Any]],
    ground_truth: Dict[str, Any],
    k_values: List[int] = [3, 5, 10]
) -> Dict[str, Any]:
    """
    단일 데이터셋에 대한 추천 결과 평가

    Args:
        predictions: 모델이 생성한 추천 결과 리스트
            [{"type": "paper", "id": "...", "rank": 1, "score": 0.9}, ...]
        ground_truth: 정답 데이터 (candidate_pool 포함)
            {"papers": [...], "datasets": [...]}
        k_values: 평가할 k 값들

    Returns:
        평가 메트릭 딕셔너리
    """
    try:
        results = {}

        # 추천된 ID와 타입 추출
        predicted_papers = [p for p in predictions if p.get('type') == 'paper']
        predicted_datasets = [p for p in predictions if p.get('type') == 'dataset']

        predicted_paper_ids = [p.get('id', '') for p in predicted_papers]
        predicted_dataset_ids = [p.get('id', '') for p in predicted_datasets]

        # Ground truth에서 관련성 점수 매핑
        paper_relevance_map = {}
        dataset_relevance_map = {}

        if 'papers' in ground_truth:
            for paper in ground_truth['papers']:
                paper_id = paper.get('id', '')
                relevance = paper.get('relevance_score', 0)
                paper_relevance_map[paper_id] = relevance

        if 'datasets' in ground_truth:
            for dataset in ground_truth['datasets']:
                dataset_id = dataset.get('id', '')
                relevance = dataset.get('relevance_score', 0)
                dataset_relevance_map[dataset_id] = relevance

        # 관련 있는 아이템 필터링 (relevance_score > 0)
        relevant_paper_ids = [pid for pid, rel in paper_relevance_map.items() if rel > 0]
        relevant_dataset_ids = [did for did, rel in dataset_relevance_map.items() if rel > 0]

        # 각 k 값에 대해 평가
        for k in k_values:
            # 논문 평가
            paper_relevance_scores = [
                paper_relevance_map.get(pid, 0) for pid in predicted_paper_ids[:k]
            ]

            results[f'paper_ndcg@{k}'] = calculate_ndcg(paper_relevance_scores, k)
            results[f'paper_mrr@{k}'] = calculate_mrr(paper_relevance_scores, k)
            results[f'paper_recall@{k}'] = calculate_recall_at_k(
                predicted_paper_ids, relevant_paper_ids, k
            )
            results[f'paper_precision@{k}'] = calculate_precision_at_k(
                predicted_paper_ids, relevant_paper_ids, k
            )

            # 데이터셋 평가
            dataset_relevance_scores = [
                dataset_relevance_map.get(did, 0) for did in predicted_dataset_ids[:k]
            ]

            results[f'dataset_ndcg@{k}'] = calculate_ndcg(dataset_relevance_scores, k)
            results[f'dataset_mrr@{k}'] = calculate_mrr(dataset_relevance_scores, k)
            results[f'dataset_recall@{k}'] = calculate_recall_at_k(
                predicted_dataset_ids, relevant_dataset_ids, k
            )
            results[f'dataset_precision@{k}'] = calculate_precision_at_k(
                predicted_dataset_ids, relevant_dataset_ids, k
            )

            # 전체 평가 (논문 + 데이터셋)
            all_predicted_ids = predicted_paper_ids + predicted_dataset_ids
            all_relevant_ids = relevant_paper_ids + relevant_dataset_ids
            all_relevance_map = {**paper_relevance_map, **dataset_relevance_map}

            all_relevance_scores = [
                all_relevance_map.get(item_id, 0) for item_id in all_predicted_ids[:k]
            ]

            results[f'overall_ndcg@{k}'] = calculate_ndcg(all_relevance_scores, k)
            results[f'overall_mrr@{k}'] = calculate_mrr(all_relevance_scores, k)
            results[f'overall_recall@{k}'] = calculate_recall_at_k(
                all_predicted_ids, all_relevant_ids, k
            )
            results[f'overall_precision@{k}'] = calculate_precision_at_k(
                all_predicted_ids, all_relevant_ids, k
            )

        # 통계 정보 추가
        results['num_predicted_papers'] = len(predicted_papers)
        results['num_predicted_datasets'] = len(predicted_datasets)
        results['num_relevant_papers'] = len(relevant_paper_ids)
        results['num_relevant_datasets'] = len(relevant_dataset_ids)

        return results

    except Exception as e:
        logger.error(f"추천 평가 실패: {e}")
        return {}


def batch_evaluate(
    all_predictions: List[Dict[str, Any]],
    all_ground_truths: List[Dict[str, Any]],
    k_values: List[int] = [3, 5, 10]
) -> Dict[str, Any]:
    """
    여러 데이터셋에 대한 배치 평가

    Args:
        all_predictions: 모든 데이터셋의 추천 결과 리스트
        all_ground_truths: 모든 데이터셋의 정답 데이터 리스트
        k_values: 평가할 k 값들

    Returns:
        평균 메트릭 딕셔너리
    """
    try:
        if len(all_predictions) != len(all_ground_truths):
            raise ValueError("예측 결과와 정답 데이터의 개수가 다릅니다")

        all_results = []

        for pred, gt in zip(all_predictions, all_ground_truths):
            result = evaluate_recommendations(pred, gt, k_values)
            all_results.append(result)

        # 평균 계산
        avg_results = {}

        # 모든 메트릭 키 수집
        all_keys = set()
        for result in all_results:
            all_keys.update(result.keys())

        # 숫자 메트릭만 평균 계산
        numeric_keys = [k for k in all_keys if not k.startswith('num_')]

        for key in numeric_keys:
            values = [r.get(key, 0) for r in all_results]
            avg_results[f'avg_{key}'] = np.mean(values)
            avg_results[f'std_{key}'] = np.std(values)

        # 통계 정보
        avg_results['total_evaluated'] = len(all_results)
        avg_results['individual_results'] = all_results

        return avg_results

    except Exception as e:
        logger.error(f"배치 평가 실패: {e}")
        return {}


def format_evaluation_report(eval_results: Dict[str, Any], k_values: List[int] = [3, 5, 10]) -> str:
    """
    평가 결과를 읽기 쉬운 리포트 형식으로 포맷팅

    Args:
        eval_results: 평가 결과 딕셔너리
        k_values: 평가한 k 값들

    Returns:
        포맷팅된 리포트 문자열
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("추천 시스템 평가 결과")
    report_lines.append("=" * 80)
    report_lines.append("")

    for k in k_values:
        report_lines.append(f"📊 k={k} 평가 결과")
        report_lines.append("-" * 80)

        # Overall 메트릭
        report_lines.append(f"  전체 (Overall):")
        report_lines.append(f"    - nDCG@{k}:     {eval_results.get(f'overall_ndcg@{k}', 0):.4f}")
        report_lines.append(f"    - MRR@{k}:      {eval_results.get(f'overall_mrr@{k}', 0):.4f}")
        report_lines.append(f"    - Recall@{k}:   {eval_results.get(f'overall_recall@{k}', 0):.4f}")
        report_lines.append(f"    - Precision@{k}: {eval_results.get(f'overall_precision@{k}', 0):.4f}")
        report_lines.append("")

        # Paper 메트릭
        report_lines.append(f"  논문 (Papers):")
        report_lines.append(f"    - nDCG@{k}:     {eval_results.get(f'paper_ndcg@{k}', 0):.4f}")
        report_lines.append(f"    - MRR@{k}:      {eval_results.get(f'paper_mrr@{k}', 0):.4f}")
        report_lines.append(f"    - Recall@{k}:   {eval_results.get(f'paper_recall@{k}', 0):.4f}")
        report_lines.append(f"    - Precision@{k}: {eval_results.get(f'paper_precision@{k}', 0):.4f}")
        report_lines.append("")

        # Dataset 메트릭
        report_lines.append(f"  데이터셋 (Datasets):")
        report_lines.append(f"    - nDCG@{k}:     {eval_results.get(f'dataset_ndcg@{k}', 0):.4f}")
        report_lines.append(f"    - MRR@{k}:      {eval_results.get(f'dataset_mrr@{k}', 0):.4f}")
        report_lines.append(f"    - Recall@{k}:   {eval_results.get(f'dataset_recall@{k}', 0):.4f}")
        report_lines.append(f"    - Precision@{k}: {eval_results.get(f'dataset_precision@{k}', 0):.4f}")
        report_lines.append("")

    # 통계 정보
    report_lines.append("📈 추천 통계")
    report_lines.append("-" * 80)
    report_lines.append(f"  추천된 논문 수:     {eval_results.get('num_predicted_papers', 0)}")
    report_lines.append(f"  추천된 데이터셋 수: {eval_results.get('num_predicted_datasets', 0)}")
    report_lines.append(f"  관련 논문 수:       {eval_results.get('num_relevant_papers', 0)}")
    report_lines.append(f"  관련 데이터셋 수:   {eval_results.get('num_relevant_datasets', 0)}")
    report_lines.append("")
    report_lines.append("=" * 80)

    return "\n".join(report_lines)
