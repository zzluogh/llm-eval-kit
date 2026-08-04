"""流水线诊断适配层 —— 桥接 Function Calling 评测与 PipelineTrace 框架。

将 ToolCallSample (第29课) 转换为 PipelineTrace (第31课)，
自动诊断工具调用的 4 类常见失败：
  - tool_mismatch:   选错工具
  - param_value_error: 参数值错误
  - missing_params:   遗漏参数
  - extra_params:     幻觉多余参数

用法:
    from my_math.pipeline_trace_adapter import adapt_sample, evaluate_and_diagnose
    trace = adapt_sample(sample)
    print(trace.report())
    print(trace.diagnose())
"""
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Set, Any

from my_math.tool_calling_eval import ToolCallSample, ToolDef
from my_math.pipeline_trace import (           # noqa: E402
    PipelineTrace, TraceStep, Failure,         # 数据结构
    _tokenize, _meaningful_tokens,              # 工具函数
    detect_hallucination, detect_empty_context,  # 通用检测器
    detect_irrelevant_context, detect_incomplete_answer,
    diagnose_all, DiagnosisSummary,
)

from my_math.ragas_eval import RAGSample
from my_math.model_adapters import rule_based_predict, error_prone_predict

# ============================================================================
# 工具调用专用检测器（排障五步法"① 复现症状"→ 具体检测规则）
# ============================================================================

def detect_tool_mismatch(expected_tool: str, predicted_tool: str) -> Tuple[bool, str]:
    """检测工具选错：expected 和 predicted 不一致。"""
    e = expected_tool.strip().lower() if expected_tool else ""
    p = predicted_tool.strip().lower() if predicted_tool else ""
    if not e:
        return False, "No expected tool (expected is empty)."
    if not p:
        return True, f"Model did NOT call any tool. Expected: {e}."
    if e != p:
        return True, f"Tool mismatch: expected '{e}', but called '{p}'."
    return False, f"Tool match OK: '{e}' == '{p}'."


def detect_param_value_error(
    expected_params: Dict[str, object],
    predicted_params: Dict[str, object],
) -> Tuple[bool, str]:
    """检测参数值错误：key 相同但 value 不同。"""
    errors = []
    for key in expected_params:
        if key in predicted_params:
            e_val = str(expected_params[key]).strip().lower()
            p_val = str(predicted_params[key]).strip().lower()
            if e_val != p_val:
                errors.append(f"  {key}: expected '{e_val}' → got '{p_val}'")
    if errors:
        return True, "Parameter value mismatch:\n" + "\n".join(errors)
    return False, "All shared parameter values match OK."


def detect_missing_params(
    expected_params: Dict[str, object],
    predicted_params: Dict[str, object],
) -> Tuple[bool, str]:
    """检测遗漏参数：expected 中有但 predicted 中缺失的 key。"""
    expected_keys = set(expected_params.keys())
    predicted_keys = set(predicted_params.keys())
    missing = expected_keys - predicted_keys
    if missing:
        return True, f"Missing expected params: {sorted(missing)}."
    return False, "No missing params."


def detect_extra_params(
    expected_params: Dict[str, object],
    predicted_params: Dict[str, object],
) -> Tuple[bool, str]:
    """检测多余参数（幻觉）：predicted 中有但 expected 中没有的 key。"""
    expected_keys = set(expected_params.keys())
    predicted_keys = set(predicted_params.keys())
    extra = predicted_keys - expected_keys
    if extra:
        return True, f"Extra (hallucinated) params: {sorted(extra)}."
    return False, "No hallucinated params."


# ============================================================================
# ToolCallSample → PipelineTrace 适配器
# ============================================================================

def adapt_sample(
    sample: ToolCallSample,
    model_fn=None,
    step_delay_ms: float = 0.0,
) -> PipelineTrace:
    """将一条 Function Calling 评测样本转换为 PipelineTrace。

    映射关系:
        user query         → query
        available tools    → contexts (检索到的"候选操作")
        predicted call     → answer   (模型输出的"操作指令")
        expected call      → ground_truth (参考答案)

    Trace 步骤:
        1. tool_scan     — 扫描可用工具列表
        2. tool_select   — 选出最佳工具
        3. param_extract — 提取参数
        4. final_call    — 输出最终调用

    Args:
        sample:        第29课的 ToolCallSample
        step_delay_ms: 模拟每步耗时（0 = 用真实 time.sleep 测量）

    Returns:
        PipelineTrace 对象
    """
    # 如果有模型函数，先预测
    if model_fn:
        sample.predicted_tool, sample.predicted_params = model_fn(sample.query, sample.tools_available)

    trace = PipelineTrace(query=sample.query)

    # 将可用工具列表转成 contexts（检索到的候选操作）
    tool_descs = {}
    for t in sample.tools_available:
        if isinstance(t, ToolDef):
            tool_descs[t.name] = t.description
        elif isinstance(t, dict):
            tool_descs[t.get("name", "?")] = t.get("description", "")

    contexts = [f"{name}: {desc}" for name, desc in tool_descs.items()]

    def _ms(delay: float) -> float:
        if step_delay_ms > 0:
            time.sleep(step_delay_ms / 1000.0)
            return step_delay_ms
        t0 = time.perf_counter()
        time.sleep(0.002)
        return (time.perf_counter() - t0) * 1000

    # Step 1: 工具扫描
    d = _ms(step_delay_ms)
    trace.add_step(TraceStep(
        step_name="tool_scan",
        input_data=f"query ({len(sample.query.split())} words)",
        output_data=f"{len(tool_descs)} tools found",
        duration_ms=d,
        detail=f"Available: {sorted(tool_descs.keys())[:5]}",
    ))

    # Step 2: 工具选择
    d = _ms(step_delay_ms)
    trace.add_step(TraceStep(
        step_name="tool_select",
        input_data=f"{len(tool_descs)} candidate tools",
        output_data=f"selected: {sample.predicted_tool or '(none)'}",
        duration_ms=d,
    ))

    # Step 3: 参数提取
    d = _ms(step_delay_ms)
    pred_params_str = ", ".join(f"{k}={v}" for k, v in sample.predicted_params.items()) or "(none)"
    trace.add_step(TraceStep(
        step_name="param_extract",
        input_data=f"tools: {sample.predicted_tool or '(none)'}",
        output_data=f"params: {pred_params_str[:40]}",
        duration_ms=d,
    ))

    # Step 4: 最终调用
    d = _ms(step_delay_ms)
    call_str = f"{sample.predicted_tool}({pred_params_str})" if sample.predicted_tool else "(no call)"
    trace.add_step(TraceStep(
        step_name="final_call",
        input_data="extracted params",
        output_data=call_str[:60],
        duration_ms=d,
    ))

    # ================================================================
    # 诊断：工具调用专用检测器
    # ================================================================
    # 工具选错
    is_mismatch, desc = detect_tool_mismatch(
        sample.expected_tool or "", sample.predicted_tool or "")
    if is_mismatch:
        trace.failures.append(Failure("tool_mismatch", "tool_select", desc, "high"))

    # 参数值错
    is_val_err, desc = detect_param_value_error(
        sample.expected_params, sample.predicted_params)
    if is_val_err:
        trace.failures.append(Failure("param_value_error", "param_extract", desc, "medium"))

    # 遗漏参数
    is_missing, desc = detect_missing_params(
        sample.expected_params, sample.predicted_params)
    if is_missing:
        trace.failures.append(Failure("missing_params", "param_extract", desc, "medium"))

    # 多余参数
    is_extra, desc = detect_extra_params(
        sample.expected_params, sample.predicted_params)
    if is_extra:
        trace.failures.append(Failure("extra_params", "param_extract", desc, "low"))

    return trace

# ============================================================================
# RAGSample → PipelineTrace 适配器
# ============================================================================

def adapt_rag_sample(
    sample: "RAGSample",
    step_delay_ms: float = 0.0,
) -> PipelineTrace:
    """将一条 RAG 评测样本转换为 PipelineTrace。

    映射关系（几乎直通）:
        sample.question      → query
        sample.contexts      → contexts（直接是检索结果）
        sample.answer        → answer
        sample.ground_truth  → ground_truth

    Trace 步骤:
        1. retrieval      — 检索上下文
        2. context_merge  — 拼接上下文
        3. generation     — LLM 生成答案
        4. answer_output  — 答案输出

    Args:
        sample:        第28课的 RAGSample
        step_delay_ms: 模拟每步耗时

    Returns:
        PipelineTrace 对象
    """
    trace = PipelineTrace(query=sample.question)

    def _ms(delay: float) -> float:
        if step_delay_ms > 0:
            time.sleep(step_delay_ms / 1000.0)
            return step_delay_ms
        t0 = time.perf_counter()
        time.sleep(0.002)
        return (time.perf_counter() - t0) * 1000

    # Step 1: 检索
    d = _ms(step_delay_ms)
    retrieval_detail = (f"Retrieved {len(sample.contexts)} chunks, "
                        f"{sum(len(c.split()) for c in sample.contexts)} total words")
    trace.add_step(TraceStep(
        step_name="retrieval",
        input_data=f"query ({len(sample.question.split())} words)",
        output_data=f"{len(sample.contexts)} context(s)",
        duration_ms=d,
        detail=retrieval_detail,
    ))

    # Step 2: 上下文拼接
    d = _ms(step_delay_ms)
    ctx_text = " ".join(sample.contexts)
    trace.add_step(TraceStep(
        step_name="context_merge",
        input_data=f"{len(sample.contexts)} context(s)",
        output_data=f"{len(ctx_text.split())} words merged",
        duration_ms=d,
    ))

    # Step 3: 生成
    d = _ms(step_delay_ms)
    trace.add_step(TraceStep(
        step_name="generation",
        input_data=f"merged context ({len(ctx_text.split())} words)",
        output_data=f"answer ({len(sample.answer.split())} words)",
        duration_ms=d,
        detail=f"Generating from {len(ctx_text.split())}-word context",
    ))

    # Step 4: 回答输出
    d = _ms(step_delay_ms)
    trace.add_step(TraceStep(
        step_name="answer_output",
        input_data=f"raw answer ({len(sample.answer.split())} words)",
        output_data=f"final answer ({len(sample.answer.split())} words)",
        duration_ms=d,
    ))

    # 诊断：只调通用检测器，不加工具调用专用检测器
    step_durations = [(s.step_name, s.duration_ms) for s in trace.steps]
    trace.failures = diagnose_all(
        sample.question, sample.contexts,
        sample.answer, sample.ground_truth, step_durations,
    )

    return trace

# ============================================================================
# 批量适配
# ============================================================================

def evaluate_and_diagnose(
    samples: List[Any],          # ← ToolCallSample 或 RAGSample 都行
    adapt_fn,                    # ← adapt_rag_sample 或 adapt_sample
) -> Tuple[List[PipelineTrace], DiagnosisSummary]:
    """批量诊断：对每条样本生成 PipelineTrace 并汇总。

    Args:
        samples: ToolCallSample 列表
        model_fn: 可选，模型预测函数 f(query, tools) → (name, params)。
                  如果给 None，则使用样本中已有的 predicted_tool/params。

    Returns:
        (traces, summary) — 每条样本的 trace + 汇总统计
    """
    traces: List[PipelineTrace] = []
    failure_counts: Dict[str, int] = {}
    healthy = 0

    for s in samples:
        trace = adapt_fn(s)    # ← 唯一改动点
        traces.append(trace)

        if not trace.failures:
            healthy += 1
        for f in trace.failures:
            failure_counts[f.failure_type] = failure_counts.get(f.failure_type, 0) + 1

    summary = DiagnosisSummary(
        total_samples=len(samples),
        healthy_count=healthy,
        failure_breakdown=failure_counts,
    )
    return traces, summary
