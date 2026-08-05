"""第 31 课-1 测试 — Pipeline Trace 排障框架验证。

覆盖:
  1. 工具函数 (4 个)
  2. 通用检测器 (5 个)
  3. 工具调用专用检测器 (4 个)
  4. PipelineTrace 数据结构
  5. 集成: trace_rag_pipeline / adapt_sample / evaluate_and_diagnose
  6. 边界值 / 空输入
"""
import sys
from pathlib import Path

import pytest

# ---- 框架层 ----
from llm_eval.pipeline_trace import (
    _tokenize, _overlap_ratio, _meaningful_tokens, _sentence_split,
    TraceStep, Failure, PipelineTrace,
    detect_hallucination, detect_empty_context, detect_irrelevant_context,
    detect_incomplete_answer, detect_slow_step,
    diagnose_all, trace_rag_pipeline,
)
# ---- 适配层 ----
from llm_eval.pipeline_trace_adapter import (
    detect_tool_mismatch, detect_param_value_error,
    detect_missing_params, detect_extra_params,
    adapt_sample, evaluate_and_diagnose, DiagnosisSummary,
)
# ---- 数据层 ----
from llm_eval.tool_calling_eval import ToolCallSample, ToolDef
from llm_eval.tool_calling_samples import SAMPLES, TOOLS
from llm_eval.model_adapters import rule_based_predict, error_prone_predict


# ============================================================================
# 1. 工具函数
# ============================================================================

class TestTokenize:
    def test_normal(self):
        assert _tokenize("Hello World!") == ["hello", "world"]

    def test_empty(self):
        assert _tokenize("") == []

    def test_numbers(self):
        assert _tokenize("ROS2 has 3 nodes.") == ["ros2", "has", "3", "nodes"]

    def test_special_chars(self):
        assert _tokenize("http://test.com/page?id=1") == ["http", "test", "com", "page", "id", "1"]


class TestMeaningfulTokens:
    def test_removes_stopwords(self):
        result = _meaningful_tokens("the cat is on the mat")
        assert "the" not in result
        assert "is" not in result
        assert "cat" in result

    def test_empty(self):
        assert _meaningful_tokens("") == set()

    def test_only_stopwords(self):
        assert _meaningful_tokens("the is a an") == set()


class TestSentenceSplit:
    def test_normal(self):
        result = _sentence_split("Hello. How are you? I am fine!")
        assert len(result) == 3

    def test_single(self):
        result = _sentence_split("Just one sentence")
        assert result == ["Just one sentence"]

    def test_empty(self):
        assert _sentence_split("") == []

    def test_trailing_period(self):
        result = _sentence_split("First. Second.")
        assert len(result) == 2


class TestOverlapRatio:
    def test_full_overlap(self):
        assert _overlap_ratio(["a", "b"], ["a", "b", "c"]) == pytest.approx(1.0)

    def test_half_overlap(self):
        assert _overlap_ratio(["a", "b", "c", "d"], ["a", "b"]) == pytest.approx(0.5)

    def test_no_overlap(self):
        assert _overlap_ratio(["x", "y"], ["a", "b"]) == pytest.approx(0.0)

    def test_empty_a(self):
        assert _overlap_ratio([], ["a", "b"]) == pytest.approx(0.0)


# ============================================================================
# 2. 通用检测器
# ============================================================================

class TestDetectHallucination:
    def test_clean_answer(self):
        is_hall, _ = detect_hallucination(
            "Nav2 uses costmaps for navigation.",
            ["Nav2 is a navigation framework that uses costmaps."],
        )
        assert not is_hall

    def test_hallucinated_answer(self):
        is_hall, _ = detect_hallucination(
            "Nav2 uses deep reinforcement learning to fly drones.",
            ["Nav2 is a navigation framework that uses costmaps."],
        )
        assert is_hall

    def test_partial_hallucination(self):
        is_hall, _ = detect_hallucination(
            "Nav2 uses costmaps. It can also cook dinner. And fly to the moon.",
            ["Nav2 is a navigation framework. It uses costmaps."],
        )
        assert is_hall

    def test_empty_answer(self):
        is_hall, _ = detect_hallucination("", ["Some context."])
        assert not is_hall

    def test_empty_context(self):
        is_hall, _ = detect_hallucination("Some answer.", [])
        assert is_hall  # empty context → no tokens → ratio < 0.5


class TestDetectEmptyContext:
    def test_empty_list(self):
        is_empty, _ = detect_empty_context([])
        assert is_empty

    def test_non_empty(self):
        is_empty, _ = detect_empty_context(["context"])
        assert not is_empty

    def test_whitespace_only(self):
        is_empty, _ = detect_empty_context(["   ", "\t\n"])
        assert is_empty

    def test_mixed(self):
        is_empty, _ = detect_empty_context(["   ", "real content"])
        assert not is_empty


class TestDetectIrrelevantContext:
    def test_relevant(self):
        is_irrel, _ = detect_irrelevant_context(
            ["Nav2 uses costmaps for navigation."],
            "How does Nav2 handle navigation?",
        )
        assert not is_irrel

    def test_irrelevant(self):
        is_irrel, _ = detect_irrelevant_context(
            ["The weather is sunny today."],
            "How does Nav2 handle navigation?",
        )
        assert is_irrel

    def test_empty_query_tokens(self):
        is_irrel, _ = detect_irrelevant_context(
            ["Some context."],
            "the is a",
        )
        assert not is_irrel  # no meaningful tokens → assume not irrelevant


class TestDetectIncompleteAnswer:
    def test_complete(self):
        is_inc, _ = detect_incomplete_answer(
            "Nav2 is a robust navigation framework for ROS2.",
            "Nav2 is a navigation framework for ROS2.",
        )
        assert not is_inc

    def test_incomplete(self):
        is_inc, _ = detect_incomplete_answer(
            "Yes.",
            "Nav2 is a navigation framework that provides planning, control, and behavior trees for ROS2 robots.",
        )
        assert is_inc

    def test_no_meaningful_tokens(self):
        is_inc, _ = detect_incomplete_answer(
            "a the is",
            "Nav2 navigation framework.",
        )
        assert is_inc

    def test_empty_ground_truth(self):
        is_inc, _ = detect_incomplete_answer(
            "Some answer.", ""
        )
        assert not is_inc


class TestDetectSlowStep:
    def test_fast(self):
        is_slow, _ = detect_slow_step(10.0, threshold_ms=500.0)
        assert not is_slow

    def test_slow(self):
        is_slow, _ = detect_slow_step(800.0, threshold_ms=500.0)
        assert is_slow

    def test_at_boundary(self):
        is_slow, _ = detect_slow_step(500.0, threshold_ms=500.0)
        assert not is_slow


# ============================================================================
# 3. 工具调用专用检测器
# ============================================================================

class TestDetectToolMismatch:
    def test_match(self):
        is_mm, _ = detect_tool_mismatch("get_weather", "get_weather")
        assert not is_mm

    def test_mismatch(self):
        is_mm, _ = detect_tool_mismatch("get_weather", "search")
        assert is_mm

    def test_no_call(self):
        is_mm, _ = detect_tool_mismatch("get_weather", "")
        assert is_mm

    def test_both_empty(self):
        is_mm, _ = detect_tool_mismatch("", "")
        assert not is_mm

    def test_case_insensitive(self):
        is_mm, _ = detect_tool_mismatch("Get_Weather", "get_weather")
        assert not is_mm


class TestDetectParamValueError:
    def test_all_match(self):
        is_err, _ = detect_param_value_error(
            {"city": "Beijing", "unit": "celsius"},
            {"city": "Beijing", "unit": "celsius"},
        )
        assert not is_err

    def test_value_mismatch(self):
        is_err, _ = detect_param_value_error(
            {"city": "Beijing"},
            {"city": "Shanghai"},
        )
        assert is_err

    def test_only_matching_keys_checked(self):
        is_err, _ = detect_param_value_error(
            {"city": "Beijing", "duration": "5"},
            {"city": "Beijing", "unknown": "x"},
        )
        assert not is_err  # shared key 'city' matches; 'duration' not in predicted

    def test_empty_both(self):
        is_err, _ = detect_param_value_error({}, {})
        assert not is_err


class TestDetectMissingParams:
    def test_none_missing(self):
        is_miss, _ = detect_missing_params(
            {"a": 1, "b": 2}, {"a": 1, "b": 2, "c": 3},
        )
        assert not is_miss

    def test_some_missing(self):
        is_miss, _ = detect_missing_params(
            {"a": 1, "b": 2, "c": 3}, {"a": 1},
        )
        assert is_miss

    def test_all_missing(self):
        is_miss, _ = detect_missing_params(
            {"a": 1, "b": 2}, {},
        )
        assert is_miss

    def test_empty_expected(self):
        is_miss, _ = detect_missing_params({}, {"a": 1})
        assert not is_miss


class TestDetectExtraParams:
    def test_no_extra(self):
        is_extra, _ = detect_extra_params(
            {"a": 1}, {"a": 1},
        )
        assert not is_extra

    def test_has_extra(self):
        is_extra, _ = detect_extra_params(
            {"a": 1}, {"a": 1, "b": 2},
        )
        assert is_extra

    def test_empty_expected(self):
        is_extra, _ = detect_extra_params({}, {"a": 1})
        assert is_extra

    def test_empty_both(self):
        is_extra, _ = detect_extra_params({}, {})
        assert not is_extra


# ============================================================================
# 4. 数据结构
# ============================================================================

class TestTraceStep:
    def test_creation(self):
        step = TraceStep("test", "in", "out", 12.5, "detail")
        assert step.step_name == "test"
        assert step.duration_ms == 12.5

    def test_default_detail(self):
        step = TraceStep("test", "in", "out", 5.0)
        assert step.detail == ""


class TestFailure:
    def test_creation(self):
        f = Failure("hallucination", "generation", "desc", "high")
        assert f.failure_type == "hallucination"
        assert f.severity == "high"

    def test_default_severity(self):
        f = Failure("test", "step", "desc")
        assert f.severity == "medium"


class TestPipelineTrace:
    def test_add_step_accumulates_duration(self):
        trace = PipelineTrace(query="test")
        trace.add_step(TraceStep("s1", "in", "out", 10.0))
        trace.add_step(TraceStep("s2", "in", "out", 20.0))
        assert trace.total_duration_ms == 30.0
        assert len(trace.steps) == 2

    def test_report_generates(self):
        trace = PipelineTrace(query="What is ROS2?")
        trace.add_step(TraceStep("retrieval", "query", "3 docs", 15.0))
        report = trace.report()
        assert "Pipeline Trace Report" in report
        assert "retrieval" in report

    def test_diagnose_empty(self):
        trace = PipelineTrace(query="test")
        diag = trace.diagnose()
        assert "No failures" in diag

    def test_diagnose_with_failures(self):
        trace = PipelineTrace(query="test")
        trace.failures = [Failure("test_fail", "step1", "Something wrong")]
        diag = trace.diagnose()
        assert "test_fail" in diag
        assert "Something wrong" in diag


# ============================================================================
# 5. 编排器
# ============================================================================

class TestDiagnoseAll:
    def test_all_healthy(self):
        failures = diagnose_all(
            query="What is Nav2?",
            contexts=["Nav2 is a navigation framework for ROS2."],
            answer="Nav2 is a navigation framework.",
            ground_truth="Nav2 is a navigation framework.",
        )
        assert len(failures) == 0

    def test_empty_context(self):
        failures = diagnose_all(
            query="What is Nav2?", contexts=[], answer="Nav2 is...", ground_truth="",
        )
        assert any(f.failure_type == "empty_context" for f in failures)

    def test_irrelevant_context(self):
        failures = diagnose_all(
            query="What is Nav2?",
            contexts=["The weather today is sunny."],
            answer="I don't know.", ground_truth="",
        )
        assert any(f.failure_type == "irrelevant_context" for f in failures)

    def test_hallucination(self):
        failures = diagnose_all(
            query="What is Nav2?",
            contexts=["Nav2 uses costmaps."],
            answer="Nav2 flies drones and cooks dinner.",
            ground_truth="",
        )
        assert any(f.failure_type == "hallucination" for f in failures)

    def test_incomplete_answer(self):
        failures = diagnose_all(
            query="What is Nav2?",
            contexts=["Nav2 is a framework."],
            answer="Yes.",
            ground_truth="Nav2 is a comprehensive navigation framework for ROS2 with planning, control, and behavior trees.",
        )
        assert any(f.failure_type == "incomplete_answer" for f in failures)


class TestTraceRagPipeline:
    def test_creates_steps(self):
        trace = trace_rag_pipeline(
            query="What is Nav2?",
            contexts=["Nav2 is a navigation framework."],
            answer="Nav2 is a navigation framework.",
            ground_truth="Nav2 is a navigation framework.",
        )
        step_names = [s.step_name for s in trace.steps]
        assert "retrieval" in step_names
        assert "generation" in step_names
        assert "answer_output" in step_names

    def test_healthy_pipeline(self):
        trace = trace_rag_pipeline(
            query="What is Nav2?",
            contexts=["Nav2 is a navigation framework for ROS2."],
            answer="Nav2 is a navigation framework.",
            ground_truth="Nav2 is a navigation framework.",
        )
        assert len(trace.failures) == 0

    def test_hallucinating_pipeline(self):
        trace = trace_rag_pipeline(
            query="What is Nav2?",
            contexts=["Nav2 uses costmaps for navigation."],
            answer="Nav2 flies drones and cooks dinner.", ground_truth="",
        )
        assert any(f.failure_type == "hallucination" for f in trace.failures)


# ============================================================================
# 6. 适配层集成
# ============================================================================

class TestAdaptSample:
    def test_creates_trace(self):
        sample = ToolCallSample(
            query="What is the weather in Beijing?",
            tools_available=TOOLS,
            expected_tool="get_weather",
            expected_params={"city": "Beijing"},
            predicted_tool="get_weather",
            predicted_params={"city": "Beijing"},
        )
        trace = adapt_sample(sample)
        assert trace.query == sample.query
        assert len(trace.steps) == 4
        step_names = [s.step_name for s in trace.steps]
        assert step_names == ["tool_scan", "tool_select", "param_extract", "final_call"]

    def test_perfect_match_no_failures(self):
        sample = ToolCallSample(
            query="Weather in Beijing",
            tools_available=TOOLS,
            expected_tool="get_weather",
            expected_params={"city": "Beijing"},
            predicted_tool="get_weather",
            predicted_params={"city": "Beijing"},
        )
        trace = adapt_sample(sample)
        tool_failures = [f for f in trace.failures
                         if f.failure_type in ("tool_mismatch", "param_value_error",
                                               "missing_params", "extra_params")]
        assert len(tool_failures) == 0

    def test_tool_mismatch_detected(self):
        sample = ToolCallSample(
            query="Weather in Beijing",
            tools_available=TOOLS,
            expected_tool="get_weather",
            expected_params={"city": "Beijing"},
            predicted_tool="search",
            predicted_params={"query": "Beijing"},
        )
        trace = adapt_sample(sample)
        assert any(f.failure_type == "tool_mismatch" for f in trace.failures)

    def test_extra_params_detected(self):
        sample = ToolCallSample(
            query="Weather in Beijing",
            tools_available=TOOLS,
            expected_tool="get_weather",
            expected_params={"city": "Beijing"},
            predicted_tool="get_weather",
            predicted_params={"city": "Beijing", "units": "metric"},
        )
        trace = adapt_sample(sample)
        assert any(f.failure_type == "extra_params" for f in trace.failures)

    def test_missing_params_detected(self):
        sample = ToolCallSample(
            query="Send email to alice@a.com subj hello body hi",
            tools_available=TOOLS,
            expected_tool="send_email",
            expected_params={"to": "alice@a.com", "subject": "hello", "body": "hi"},
            predicted_tool="send_email",
            predicted_params={"to": "alice@a.com"},
        )
        trace = adapt_sample(sample)
        assert any(f.failure_type == "missing_params" for f in trace.failures)

    def test_no_tool_expected(self):
        sample = ToolCallSample(
            query="Hello!",
            tools_available=TOOLS,
            expected_tool="",
            expected_params={},
            predicted_tool="",
            predicted_params={},
        )
        trace = adapt_sample(sample)
        tool_failures = [f for f in trace.failures
                         if f.failure_type in ("tool_mismatch", "param_value_error",
                                               "missing_params", "extra_params")]
        assert len(tool_failures) == 0


class TestEvaluateAndDiagnose:
    def test_with_samples(self):
        samples = [
            ToolCallSample(
                query="Weather in Beijing",
                tools_available=TOOLS,
                expected_tool="get_weather",
                expected_params={"city": "Beijing"},
                predicted_tool="get_weather",
                predicted_params={"city": "Beijing"},
            ),
            ToolCallSample(
                query="Weather in Shanghai",
                tools_available=TOOLS,
                expected_tool="get_weather",
                expected_params={"city": "Shanghai"},
                predicted_tool="search",  # wrong tool
                predicted_params={"query": "Shanghai"},
            ),
        ]
        traces, summary = evaluate_and_diagnose(samples, adapt_fn=adapt_sample)
        assert len(traces) == 2
        assert summary.total_samples == 2
        assert summary.healthy_count == 1
        assert "tool_mismatch" in summary.failure_breakdown

    def test_with_model_fn(self):
        sample = ToolCallSample(
            query="Weather in Beijing",
            tools_available=TOOLS,
            expected_tool="get_weather",
            expected_params={"city": "Beijing"},
            predicted_tool="",
            predicted_params={},
        )

        def fake_model(query, tools):
            return "get_weather", {"city": "Beijing"}

        traces, _ = evaluate_and_diagnose([sample], adapt_fn=adapt_sample)
        assert traces[0].query == "Weather in Beijing"


class TestDiagnosisSummary:
    def test_report_healthy(self):
        s = DiagnosisSummary(total_samples=10, healthy_count=10, failure_breakdown={})
        report = s.report()
        assert "All samples healthy" in report

    def test_report_with_failures(self):
        s = DiagnosisSummary(
            total_samples=10, healthy_count=5,
            failure_breakdown={"tool_mismatch": 3, "missing_params": 2},
        )
        report = s.report()
        assert "tool_mismatch" in report
        assert "30%" in report
