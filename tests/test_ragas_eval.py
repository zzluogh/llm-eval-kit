"""第 28 课 RAGAS 评测 — 单元测试。

测试覆盖:
  - 4 个指标函数的独立断言（边界值 + 典型值）
  - evaluate_sample 端到端
  - evaluate 批量聚合 + 报告生成
  - 10 组完整样本评测
"""
import math

import pytest
from llm_eval.ragas_eval import (
    _tokenize, _overlap_ratio, _cosine_similarity, _sentence_tokenize,
    faithfulness, answer_relevancy, context_recall, context_precision,
    evaluate_sample, evaluate, RAGSample, MetricScores,
)
from llm_eval.ragas_samples import SAMPLES


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

class TestTokenize:
    def test_basic(self):
        assert _tokenize("ROS2 is a framework!") == ["ros2", "is", "a", "framework"]

    def test_empty(self):
        assert _tokenize("") == []

    def test_numbers(self):
        assert _tokenize("version 2.5 has 3 parts") == ["version", "2", "5", "has", "3", "parts"]

    def test_special_chars(self):
        assert _tokenize("hello, world!!!") == ["hello", "world"]


class TestOverlapRatio:
    def test_full_overlap(self):
        assert _overlap_ratio(["a", "b"], ["a", "b", "c"]) == 1.0

    def test_partial_overlap(self):
        assert _overlap_ratio(["a", "b", "c"], ["a", "d", "e"]) == pytest.approx(1 / 3, abs=0.01)

    def test_no_overlap(self):
        assert _overlap_ratio(["x", "y"], ["a", "b"]) == 0.0

    def test_empty_numerator(self):
        assert _overlap_ratio([], ["a", "b"]) == 0.0


class TestCosineSimilarity:
    def test_identical(self):
        assert _cosine_similarity(["a", "b"], ["a", "b"]) == pytest.approx(1.0)

    def test_disjoint(self):
        assert _cosine_similarity(["a"], ["b"]) == 0.0

    def test_empty(self):
        assert _cosine_similarity([], ["a"]) == 0.0
        assert _cosine_similarity(["a"], []) == 0.0

    def test_partial(self):
        sim = _cosine_similarity(["a", "b", "c"], ["a", "b", "d"])
        assert 0.5 < sim < 1.0


class TestSentenceTokenize:
    def test_basic(self):
        assert _sentence_tokenize("Hello. World!") == ["Hello", "World"]

    def test_multiple_punctuation(self):
        result = _sentence_tokenize("A? B! C. D...")
        assert result == ["A", "B", "C", "D"]

    def test_no_punctuation(self):
        assert _sentence_tokenize("Hello World") == ["Hello World"]


# ---------------------------------------------------------------------------
# 4 核心指标
# ---------------------------------------------------------------------------

class TestFaithfulness:
    def test_perfect_match(self):
        score = faithfulness("ROS2 uses DDS.", ["ROS2 uses DDS for communication."])
        assert score == 1.0

    def test_complete_hallucination(self):
        score = faithfulness("The moon is made of cheese.", ["ROS2 is a robot OS."])
        assert score == 0.0

    def test_partial_support(self):
        """第一句全部命中(=1.0≥0.5,计1), 第二句0命中(0<0.5,不计) → 1/2 = 0.5"""
        score = faithfulness(
            "ROS2 is fast. It can fly.",
            ["ROS2 is fast and reliable."],
        )
        assert score == 0.5

    def test_empty_answer(self):
        assert faithfulness("", ["some context"]) == 0.0

    def test_empty_context(self):
        score = faithfulness("Some answer.", [])
        assert score == 0.0


class TestAnswerRelevancy:
    def test_high_relevancy(self):
        score = answer_relevancy("ROS2 uses DDS.", "What does ROS2 use?")
        assert score > 0.0

    def test_zero_overlap(self):
        score = answer_relevancy("Apples are fruit.", "What is ROS2?")
        assert score == 0.0

    def test_empty(self):
        assert answer_relevancy("", "Question?") == 0.0


class TestContextRecall:
    def test_full_recall(self):
        score = context_recall(
            ["ROS2 uses DDS middleware."],
            "ROS2 uses DDS.",
        )
        assert score == 1.0

    def test_zero_recall(self):
        score = context_recall(["It is sunny."], "ROS2 uses DDS.")
        assert score == 0.0

    def test_ground_truth_proper_subset(self):
        score = context_recall(
            ["ROS2 DDS communication nodes."],
            "ROS2 uses DDS for communication between nodes and sensors.",
        )
        assert 0.0 < score < 1.0


class TestContextPrecision:
    def test_all_relevant(self):
        score = context_precision(
            ["ROS2 is great.", "ROS2 uses DDS."],
            "What is ROS2?",
        )
        assert score == 1.0

    def test_half_relevant(self):
        score = context_precision(
            ["ROS2 uses DDS.", "The sky is blue."],
            "What is ROS2?",
        )
        assert score == 0.5

    def test_empty_contexts(self):
        assert context_precision([], "Question?") == 0.0


# ---------------------------------------------------------------------------
# 聚合函数
# ---------------------------------------------------------------------------

class TestEvaluateSample:
    def test_returns_metric_scores(self):
        sample = RAGSample(
            question="Q?", contexts=["C"], answer="A", ground_truth="GT",
        )
        scores = evaluate_sample(sample)
        assert isinstance(scores, MetricScores)
        assert 0.0 <= scores.faithfulness <= 1.0
        assert 0.0 <= scores.answer_relevancy <= 1.0
        assert 0.0 <= scores.context_recall <= 1.0
        assert 0.0 <= scores.context_precision <= 1.0


class TestEvaluate:
    def test_empty_list(self):
        result = evaluate([])
        assert len(result.per_sample) == 0
        assert result.avg.faithfulness == 0.0

    def test_single_sample(self):
        sample = RAGSample(
            question="What is X?",
            contexts=["X is a thing."],
            answer="X is a thing.",
            ground_truth="X is a thing.",
        )
        result = evaluate([sample])
        assert len(result.per_sample) == 1
        assert result.per_sample[0].faithfulness > 0.0

    def test_report_output(self):
        """验证 report() 不抛异常且包含关键行。"""
        sample = RAGSample(
            question="Q?", contexts=["C"], answer="A", ground_truth="G",
        )
        result = evaluate([sample])
        report = result.report()
        assert "Sample" in report
        assert "Faith" in report
        assert "AVG" in report


# ---------------------------------------------------------------------------
# 10 组完整样本
# ---------------------------------------------------------------------------

class TestFullDataset:
    def test_all_10_samples_return_valid_scores(self):
        result = evaluate(SAMPLES)
        assert len(result.per_sample) == 10
        for s in result.per_sample:
            assert 0.0 <= s.faithfulness <= 1.0
            assert 0.0 <= s.answer_relevancy <= 1.0
            assert 0.0 <= s.context_recall <= 1.0
            assert 0.0 <= s.context_precision <= 1.0

    def test_samples_have_variance(self):
        """验证不同样本之间得分不完全相同（数据多样性）。"""
        result = evaluate(SAMPLES)
        faith_scores = [s.faithfulness for s in result.per_sample]
        assert min(faith_scores) < max(faith_scores), (
            f"all faithfulness scores identical: {faith_scores[0]}"
        )

    def test_hallucination_scores_lower_than_perfect(self):
        """Q3（幻觉）faithfulness < Q1（完美）。"""
        s1 = evaluate_sample(SAMPLES[0])
        s3 = evaluate_sample(SAMPLES[2])
        assert s3.faithfulness < s1.faithfulness, (
            f"hallucination faithfulness {s3.faithfulness:.3f} !< "
            f"perfect faithfulness {s1.faithfulness:.3f}"
        )

    def test_noisy_context_lower_precision(self):
        """Q6（含1/3无关上下文）context_precision < 1.0（不完美）。"""
        s6 = evaluate_sample(SAMPLES[5])
        assert s6.context_precision < 1.0, (
            f"expected < 1.0 due to noisy context, got {s6.context_precision:.3f}"
        )

    def test_empty_context_recall_zero(self):
        """Q7（空上下文）context_recall = 0。"""
        s7 = evaluate_sample(SAMPLES[6])
        assert s7.context_recall == 0.0

    def test_avg_scores_in_range(self):
        result = evaluate(SAMPLES)
        assert 0.0 <= result.avg.faithfulness <= 1.0
        assert 0.0 <= result.avg.answer_relevancy <= 1.0
        assert 0.0 <= result.avg.context_recall <= 1.0
        assert 0.0 <= result.avg.context_precision <= 1.0
