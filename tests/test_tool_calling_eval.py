"""第 29 课 Function Calling 评测 — 单元测试。

测试覆盖:
  - 3 个工具函数独立断言
  - 5 项指标函数的独立断言（边界值 + 典型值）
  - evaluate_sample 端到端
  - evaluate 批量聚合 + 报告生成
  - 20 组完整样本评测
"""
import sys
from pathlib import Path

import pytest
from llm_eval.tool_calling_eval import (
    ToolDef, ToolCallSample, ToolCallScores, ToolCallEvalResult,
    _normalize_tool_name, _compare_params, _safe_f1,
    tool_name_accuracy, param_precision, param_recall, param_f1,
    exact_match_score, evaluate_sample, evaluate,
)
from llm_eval.tool_calling_samples import SAMPLES, PARAM_SYNONYMS


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

class TestNormalizeToolName:
    def test_lowercase(self):
        assert _normalize_tool_name("Get_Weather") == "get_weather"

    def test_strip(self):
        assert _normalize_tool_name("  set_timer  ") == "set_timer"

    def test_already_normalized(self):
        assert _normalize_tool_name("calculate") == "calculate"

    def test_mixed(self):
        assert _normalize_tool_name(" Search_Web ") == "search_web"


class TestCompareParams:
    def test_exact_match(self):
        correct, pred, exp = _compare_params(
            {"city": "Beijing"}, {"city": "Beijing"}
        )
        assert correct == 1
        assert pred == 1
        assert exp == 1

    def test_case_insensitive_string(self):
        correct, _, _ = _compare_params(
            {"city": "Beijing"}, {"city": "beijing"}
        )
        assert correct == 1

    def test_numeric_exact_match(self):
        correct, _, _ = _compare_params(
            {"duration_minutes": 15}, {"duration_minutes": 15}
        )
        assert correct == 1

    def test_numeric_mismatch(self):
        correct, _, _ = _compare_params(
            {"duration_minutes": 5}, {"duration_minutes": 10}
        )
        assert correct == 0

    def test_missing_param(self):
        correct, pred, exp = _compare_params(
            {"city": "Beijing", "units": "celsius"},
            {"city": "Beijing"},
        )
        assert correct == 1
        assert pred == 1
        assert exp == 2

    def test_extra_param(self):
        correct, pred, exp = _compare_params(
            {"city": "Beijing"},
            {"city": "Beijing", "units": "celsius"},
        )
        assert correct == 1
        assert pred == 2
        assert exp == 1

    def test_empty_both(self):
        correct, pred, exp = _compare_params({}, {})
        assert correct == 0
        assert pred == 0
        assert exp == 0

    def test_wrong_key_name(self):
        """键名不同即使值相同也不算匹配。"""
        correct, _, _ = _compare_params(
            {"query": "hello"}, {"search_term": "hello"}
        )
        assert correct == 0


class TestSafeF1:
    def test_perfect(self):
        assert _safe_f1(1.0, 1.0) == 1.0

    def test_zero_both(self):
        assert _safe_f1(0.0, 0.0) == 0.0

    def test_half(self):
        f1 = _safe_f1(0.5, 0.5)
        assert f1 == 0.5

    def test_precision_zero_recall_one(self):
        assert _safe_f1(0.0, 1.0) == 0.0

    def test_typical(self):
        f1 = _safe_f1(0.67, 0.5)
        expected = 2 * 0.67 * 0.5 / (0.67 + 0.5)
        assert f1 == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 5 项指标
# ---------------------------------------------------------------------------

class TestToolNameAccuracy:
    def test_exact_match(self):
        assert tool_name_accuracy("get_weather", "get_weather") == 1.0

    def test_case_insensitive(self):
        assert tool_name_accuracy("Get_Weather", "get_weather") == 1.0

    def test_mismatch(self):
        assert tool_name_accuracy("get_weather", "set_timer") == 0.0

    def test_both_empty(self):
        assert tool_name_accuracy("", "") == 1.0

    def test_one_empty(self):
        assert tool_name_accuracy("get_weather", "") == 0.0


class TestParamPrecision:
    def test_perfect(self):
        assert param_precision({"x": 1}, {"x": 1}) == 1.0

    def test_extra_param_lowers_precision(self):
        assert param_precision({"x": 1}, {"x": 1, "y": 2}) == 0.5

    def test_all_wrong(self):
        assert param_precision({"x": 1}, {"y": 2}) == 0.0

    def test_empty_predicted(self):
        assert param_precision({"x": 1}, {}) == 0.0

    def test_case_insensitive_value(self):
        assert param_precision({"city": "London"}, {"city": "london"}) == 1.0


class TestParamRecall:
    def test_perfect(self):
        assert param_recall({"x": 1}, {"x": 1}) == 1.0

    def test_missing_param_lowers_recall(self):
        assert param_recall({"x": 1, "y": 2}, {"x": 1}) == 0.5

    def test_all_missing(self):
        assert param_recall({"x": 1}, {}) == 0.0

    def test_extra_param_does_not_affect_recall(self):
        assert param_recall({"x": 1}, {"x": 1, "y": 2}) == 1.0

    def test_empty_expected(self):
        assert param_recall({}, {"x": 1}) == 0.0


class TestParamF1:
    def test_perfect(self):
        assert param_f1({"x": 1}, {"x": 1}) == 1.0

    def test_zero(self):
        assert param_f1({"x": 1}, {"y": 2}) == 0.0

    def test_precision_perfect_recall_half(self):
        f1 = param_f1({"x": 1, "y": 2}, {"x": 1, "y": 2, "z": 3})
        p = param_precision({"x": 1, "y": 2}, {"x": 1, "y": 2, "z": 3})
        r = param_recall({"x": 1, "y": 2}, {"x": 1, "y": 2, "z": 3})
        assert f1 == pytest.approx(_safe_f1(p, r))


class TestExactMatch:
    def test_full_match(self):
        assert exact_match_score("get_weather", "get_weather",
                                 {"city": "Beijing"}, {"city": "Beijing"}) == 1.0

    def test_tool_mismatch(self):
        assert exact_match_score("get_weather", "set_timer",
                                 {"city": "Beijing"}, {"city": "Beijing"}) == 0.0

    def test_param_mismatch(self):
        assert exact_match_score("get_weather", "get_weather",
                                 {"city": "Beijing"}, {"city": "Shanghai"}) == 0.0

    def test_extra_param(self):
        assert exact_match_score("get_weather", "get_weather",
                                 {"city": "Beijing"}, {"city": "Beijing", "units": "celsius"}) == 0.0

    def test_missing_param(self):
        assert exact_match_score("translate", "translate",
                                 {"text": "hi", "target_language": "French"},
                                 {"text": "hi"}) == 0.0

    def test_case_insensitive_full_match(self):
        assert exact_match_score("Get_Weather", "get_weather",
                                 {"city": "Tokyo"}, {"city": "tokyo"}) == 1.0


# ---------------------------------------------------------------------------
# 聚合函数
# ---------------------------------------------------------------------------

class TestEvaluateSample:
    def test_returns_tool_call_scores(self):
        sample = ToolCallSample(
            query="test", tools_available=[],
            expected_tool="t1", expected_params={"x": 1},
            predicted_tool="t1", predicted_params={"x": 1},
        )
        scores = evaluate_sample(sample)
        assert isinstance(scores, ToolCallScores)
        assert scores.tool_name_match == 1.0
        assert scores.param_precision == 1.0
        assert scores.param_recall == 1.0
        assert scores.param_f1 == 1.0
        assert scores.exact_match == 1.0

    def test_wrong_tool(self):
        sample = ToolCallSample(
            query="test", tools_available=[],
            expected_tool="get_weather", expected_params={"city": "Beijing"},
            predicted_tool="search", predicted_params={"query": "weather"},
        )
        scores = evaluate_sample(sample)
        assert scores.tool_name_match == 0.0
        assert scores.param_precision == 0.0
        assert scores.param_recall == 0.0
        assert scores.param_f1 == 0.0
        assert scores.exact_match == 0.0

    def test_empty_response(self):
        sample = ToolCallSample(
            query="test", tools_available=[],
            expected_tool="calculate", expected_params={"expression": "2+3"},
            predicted_tool="", predicted_params={},
        )
        scores = evaluate_sample(sample)
        assert scores.tool_name_match == 0.0
        assert scores.param_precision == 0.0
        assert scores.param_recall == 0.0
        assert scores.exact_match == 0.0


class TestEvaluate:
    def test_empty_list(self):
        result = evaluate([])
        assert len(result.per_sample) == 0
        assert result.avg.tool_name_match == 0.0

    def test_single_sample(self):
        sample = ToolCallSample(
            query="q", tools_available=[],
            expected_tool="t", expected_params={"k": "v"},
            predicted_tool="t", predicted_params={"k": "v"},
        )
        result = evaluate([sample])
        assert len(result.per_sample) == 1
        assert result.per_sample[0].exact_match == 1.0

    def test_report_output(self):
        sample = ToolCallSample(
            query="q", tools_available=[],
            expected_tool="t", expected_params={"k": "v"},
            predicted_tool="t", predicted_params={"k": "v"},
        )
        result = evaluate([sample])
        report = result.report()
        assert "Sample" in report
        assert "ToolOK" in report
        assert "pPrec" in report
        assert "pRecall" in report
        assert "pF1" in report
        assert "Exact" in report
        assert "AVG" in report

    def test_report_with_labels(self):
        sample = ToolCallSample(
            query="q", tools_available=[],
            expected_tool="t", expected_params={"k": "v"},
            predicted_tool="t", predicted_params={"k": "v"},
        )
        result = evaluate([sample, sample])
        report = result.report(labels=["Case1", "Case2"])
        assert "Case1" in report
        assert "Case2" in report

    def test_two_samples_averaging(self):
        s1 = ToolCallSample(
            query="q", tools_available=[],
            expected_tool="t", expected_params={"k": "v"},
            predicted_tool="t", predicted_params={"k": "v"},
        )
        s2 = ToolCallSample(
            query="q", tools_available=[],
            expected_tool="wrong", expected_params={},
            predicted_tool="right", predicted_params={},
        )
        result = evaluate([s1, s2])
        assert result.avg.tool_name_match == 0.5
        assert result.avg.param_precision == 1.0


# ---------------------------------------------------------------------------
# 同义词
# ---------------------------------------------------------------------------

class TestSynonyms:
    def test_peking_maps_to_beijing(self):
        correct, _, _ = _compare_params(
            {"city": "Beijing"}, {"city": "peking"},
            synonyms=PARAM_SYNONYMS,
        )
        assert correct == 1

    def test_without_synonyms_peking_fails(self):
        correct, _, _ = _compare_params(
            {"city": "Beijing"}, {"city": "peking"},
        )
        assert correct == 0

    def test_precision_with_synonyms(self):
        p = param_precision(
            {"city": "Beijing", "units": "celsius"},
            {"city": "peking", "units": "celsius"},
            synonyms=PARAM_SYNONYMS,
        )
        assert p == 1.0

    def test_recall_with_synonyms(self):
        r = param_recall(
            {"city": "Beijing"},
            {"city": "peking"},
            synonyms=PARAM_SYNONYMS,
        )
        assert r == 1.0

    def test_exact_match_with_synonyms(self):
        em = exact_match_score(
            "get_weather", "get_weather",
            {"city": "Beijing"}, {"city": "peking"},
            synonyms=PARAM_SYNONYMS,
        )
        assert em == 1.0

    def test_evaluate_sample_with_synonyms(self):
        s = ToolCallSample(
            query="Weather in Peking please",
            tools_available=[],
            expected_tool="get_weather",
            expected_params={"city": "Beijing"},
            predicted_tool="get_weather",
            predicted_params={"city": "peking"},
        )
        scores = evaluate_sample(s, synonyms=PARAM_SYNONYMS)
        assert scores.exact_match == 1.0

    def test_synonyms_does_not_affect_unrelated_keys(self):
        """同义词只影响映射到的 key，不影响其他 key。"""
        correct, _, _ = _compare_params(
            {"symbol": "AAPL"}, {"symbol": "aapl"},
            synonyms=PARAM_SYNONYMS,  # 没有 symbol 的映射
        )
        assert correct == 1  # 仍按普通字符串归一化匹配

    def test_language_synonyms(self):
        p = param_precision(
            {"text": "hello", "target_language": "fr"},
            {"text": "hello", "target_language": "french"},
            synonyms=PARAM_SYNONYMS,
        )
        assert p == 1.0


# ---------------------------------------------------------------------------
# 20 组完整样本
# ---------------------------------------------------------------------------

class TestFullDataset:
    def test_all_20_samples_return_valid_scores(self):
        result = evaluate(SAMPLES)
        assert len(result.per_sample) == 20
        for s in result.per_sample:
            assert 0.0 <= s.tool_name_match <= 1.0
            assert 0.0 <= s.param_precision <= 1.0
            assert 0.0 <= s.param_recall <= 1.0
            assert 0.0 <= s.param_f1 <= 1.0
            assert 0.0 <= s.exact_match <= 1.0

    def test_samples_have_variance(self):
        result = evaluate(SAMPLES)
        exact_scores = [s.exact_match for s in result.per_sample]
        assert min(exact_scores) < max(exact_scores), (
            f"all exact_match scores identical: {exact_scores[0]}"
        )

    def test_perfect_scores_on_q1_q2(self):
        """Q1 Q2 是完全匹配样本。"""
        s1 = evaluate_sample(SAMPLES[0])
        s2 = evaluate_sample(SAMPLES[1])
        assert s1.exact_match == 1.0
        assert s2.exact_match == 1.0

    def test_wrong_tool_q4_score_zero(self):
        """Q4 工具选错，tool_name_match = 0。"""
        s4 = evaluate_sample(SAMPLES[3])
        assert s4.tool_name_match == 0.0

    def test_empty_response_q9_zero(self):
        """Q9 空响应，各项指标为 0。"""
        s9 = evaluate_sample(SAMPLES[8])
        assert s9.tool_name_match == 0.0

    def test_case_insensitive_q10_match(self):
        """Q10 symbol 大小写不同但仍应匹配。"""
        s10 = evaluate_sample(SAMPLES[9])
        assert s10.param_precision == 1.0

    def test_case_insensitive_tool_name_q11_match(self):
        """Q11 工具名大小写不同但仍应匹配。"""
        s11 = evaluate_sample(SAMPLES[10])
        assert s11.tool_name_match == 1.0

    def test_no_param_predict_precision_zero(self):
        """Q14 选了正确工具但没给参数，precision=0、tool_name=1.0。"""
        s14 = evaluate_sample(SAMPLES[13])
        assert s14.tool_name_match == 1.0
        assert s14.param_precision == 0.0

    def test_both_empty_q15_scores(self):
        """Q15 问候不调工具，双方都空 → tool_name_match=1.0 exact=1.0。"""
        s15 = evaluate_sample(SAMPLES[14])
        assert s15.tool_name_match == 1.0
        assert s15.exact_match == 1.0

    def test_extra_optional_param_precision(self):
        """Q19 多给了 label（可选参数），precision 下降但 recall 不变。"""
        s19 = evaluate_sample(SAMPLES[18])
        assert s19.tool_name_match == 1.0
        assert s19.param_precision < 1.0
        assert s19.param_recall == 1.0

    def test_wrong_key_name_q20_zero(self):
        """Q20 参数键名不同，correct=0。"""
        s20 = evaluate_sample(SAMPLES[19])
        assert s20.param_precision == 0.0
        assert s20.param_recall == 0.0

    def test_avg_scores_in_range(self):
        result = evaluate(SAMPLES)
        assert 0.0 <= result.avg.tool_name_match <= 1.0
        assert 0.0 <= result.avg.param_precision <= 1.0
        assert 0.0 <= result.avg.param_recall <= 1.0
        assert 0.0 <= result.avg.param_f1 <= 1.0
        assert 0.0 <= result.avg.exact_match <= 1.0

    def test_perfect_case_all_ones(self):
        """完美样本 Q1: 全部 1.0。"""
        s1 = evaluate_sample(SAMPLES[0])
        assert s1.tool_name_match == 1.0
        assert s1.param_precision == 1.0
        assert s1.param_recall == 1.0
        assert s1.param_f1 == 1.0
        assert s1.exact_match == 1.0
