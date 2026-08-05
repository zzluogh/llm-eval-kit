"""流水线追踪排障框架 —— LangSmith 本地模拟实现。

模拟 LangSmith 的核心能力：追踪 RAG / Agent 流水线每一步的执行，
自动诊断常见失败模式，帮助按排障五步法定位根因。

排障五步法：
  ① 复现症状  ② 最小隔离  ③ 定位根因  ④ 修复验证  ⑤ 归纳归档

概念对齐 LangSmith:
  LangSmith trace = 一串 runs (query → retriever → llm → output)
  本地 trace = 一串 TraceStep (retrieval → generation → answer)

用法:
    from llm_eval.pipeline_trace import trace_rag_pipeline
    trace = trace_rag_pipeline(
        query="What is Nav2?",
        contexts=["Nav2 is a navigation framework for ROS2."],
        answer="Nav2 is a tool for flying drones.",
        ground_truth="Nav2 is a navigation framework for ROS2."
    )
    print(trace.report())
    print(trace.diagnose())
"""
from __future__ import annotations
import time
import re
from dataclasses import dataclass, field
from typing import List, Tuple, Set, Optional


# ============================================================================
# 工具函数（停用词 + 分词 — 从 ragas_eval 复用概念）
# ============================================================================

_STOP_WORDS: Set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "i", "you", "he",
    "she", "it", "we", "they", "me", "him", "her", "us", "them", "my",
    "your", "his", "its", "our", "their", "this", "that", "these", "those",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
    "into", "about", "and", "or", "not", "but", "if", "so", "no", "what",
    "how", "when", "where", "which", "who", "whom", "why",
}


def _tokenize(text: str) -> List[str]:
    """分词：统一小写 + 按非字母数字切分 + 过滤空串。"""
    return [t.lower() for t in re.split(r"[^a-zA-Z0-9]+", text) if t]


def _overlap_ratio(tokens_a: List[str], tokens_b: List[str]) -> float:
    """计算 token 列表 a 中有多少比例同时出现在 b 中。"""
    set_b = set(tokens_b)
    if not tokens_a:
        return 0.0
    overlap = sum(1 for t in tokens_a if t in set_b)
    return overlap / len(tokens_a)


def _meaningful_tokens(text: str) -> Set[str]:
    """提取有意义的 token（去停用词）。"""
    return {t for t in _tokenize(text) if t not in _STOP_WORDS and len(t) > 1}


def _sentence_split(text: str) -> List[str]:
    """按句号/问号/感叹号拆分句子。"""
    parts = re.split(r"[.?!]+", text)
    return [p.strip() for p in parts if p.strip()]


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class TraceStep:
    """流水线中一个步骤的执行记录。

    Args:
        step_name:  步骤名（如 "retrieval", "generation", "answer"）
        input_data: 该步骤的输入（简短描述）
        output_data: 该步骤的输出（简短描述）
        duration_ms: 实际/模拟耗时（毫秒）
        detail: 额外细节（如检索到的前几词）
    """
    step_name: str
    input_data: str
    output_data: str
    duration_ms: float
    detail: str = ""


@dataclass
class Failure:
    """检测到的单条失败模式。

    Args:
        failure_type: 失败类型（如 "hallucination"）
        step_name:    发生在哪个步骤
        description:  可读的描述信息
        severity:     严重程度 (low / medium / high)
    """
    failure_type: str
    step_name: str
    description: str
    severity: str = "medium"


@dataclass
class PipelineTrace:
    """一次完整的流水线追踪记录。

    Args:
        query:        用户查询
        steps:        流水线各步骤的追踪记录
        failures:     自动诊断出的失败项
        total_duration_ms: 总耗时
    """
    query: str
    steps: List[TraceStep] = field(default_factory=list)
    failures: List[Failure] = field(default_factory=list)
    total_duration_ms: float = 0.0

    def add_step(self, step: TraceStep) -> None:
        """添加一个步骤并累加总耗时。"""
        self.steps.append(step)
        self.total_duration_ms += step.duration_ms

    def report(self) -> str:
        """生成追踪报告（ASCII 表格）。"""
        lines = [
            "=" * 72,
            f"  Pipeline Trace Report",
            f"  Query: {self.query[:60]}{'...' if len(self.query) > 60 else ''}",
            "=" * 72,
            f"  {'Step':<16} {'Duration':>9} {'Input':<20} {'Output':<20}",
            f"  {'-'*16} {'-'*9} {'-'*20} {'-'*20}",
        ]
        for s in self.steps:
            inp = s.input_data[:18] + ".." if len(s.input_data) > 18 else s.input_data
            out = s.output_data[:18] + ".." if len(s.output_data) > 18 else s.output_data
            lines.append(f"  {s.step_name:<16} {s.duration_ms:>7.1f}ms {inp:<20} {out:<20}")
        lines.append(f"  {'-'*16} {'-'*9} {'-'*20} {'-'*20}")
        lines.append(f"  {'TOTAL':<16} {self.total_duration_ms:>7.1f}ms")
        return "\n".join(lines)

    def diagnose(self) -> str:
        """生成诊断报告（列出所有检测到的失败项）。"""
        if not self.failures:
            return "[Diagnosis] No failures detected. Pipeline appears healthy.\n"
        lines = [
            f"\n[Diagnosis] {len(self.failures)} failure(s) detected:\n",
        ]
        for i, f in enumerate(self.failures, 1):
            lines.append(f"  {i}. [{f.severity.upper()}] {f.failure_type}")
            lines.append(f"     Step: {f.step_name}")
            lines.append(f"     {f.description}")
        return "\n".join(lines)


# ============================================================================
# 5 类失败检测器（排障五步法的"① 复现症状"映射为具体检测规则）
# ============================================================================

def detect_hallucination(answer: str, contexts: List[str]) -> Tuple[bool, str]:
    """检测幻觉：答案中的句子是否缺少上下文支撑。

    Returns:
        (is_hallucination, description)
    """
    ctx_text = " ".join(contexts)
    ctx_tokens = set(_tokenize(ctx_text))
    sentences = _sentence_split(answer)
    unsupported = []
    for sent in sentences:
        sent_tokens = _tokenize(sent)
        meaningful = [t for t in sent_tokens if t not in _STOP_WORDS and len(t) > 1]
        if not meaningful:
            continue
        overlap = sum(1 for t in meaningful if t in ctx_tokens)
        ratio = overlap / len(meaningful)
        if ratio < 0.5:
            unsupported.append(sent)

    if unsupported:
        yes = True
        msg = (f"Answer has {len(unsupported)}/{len(sentences)} unsupported sentences: "
               f"{unsupported[0][:60]}...")
    else:
        yes = False
        msg = f"All {len(sentences)} answer sentences are supported by contexts."
    return yes, msg


def detect_empty_context(contexts: List[str]) -> Tuple[bool, str]:
    """检测空检索：上下文列表为空或内容全为空白。"""
    if not contexts:
        return True, "Contexts list is empty — retriever returned nothing."
    if all(not c.strip() for c in contexts):
        return True, "All context strings are empty/whitespace."
    return False, f"Got {len(contexts)} non-empty context(s)."


def detect_irrelevant_context(contexts: List[str], query: str) -> Tuple[bool, str]:
    """检测无关检索：上下文与查询关键词语义零重叠。"""
    q_tokens = _meaningful_tokens(query)
    if not q_tokens:
        return False, "Query has no meaningful tokens to compare."
    ctx_text = " ".join(contexts)
    ctx_tokens = set(_tokenize(ctx_text))
    overlap = q_tokens & ctx_tokens
    if not overlap:
        return True, (f"Zero meaningful-token overlap between query"
                       f" and contexts. Query keywords: {sorted(q_tokens)[:5]}")
    return False, f"Query-context overlap tokens: {sorted(overlap)[:5]}"


def detect_incomplete_answer(answer: str, ground_truth: str) -> Tuple[bool, str]:
    """检测答案残缺：答案长度远短于参考答案，或只含 Yes/No。"""
    a_tokens = _meaningful_tokens(answer)
    gt_tokens = _meaningful_tokens(ground_truth)
    if not a_tokens:
        return True, "Answer has no meaningful tokens."
    if len(a_tokens) == 1 and len(gt_tokens) > 3:
        return True, f"Answer is only 1 meaningful word but ground_truth has {len(gt_tokens)}."
    if len(gt_tokens) > 0 and len(a_tokens) < len(gt_tokens) * 0.3:
        return True, (f"Answer ({len(a_tokens)} tokens) is <30% of "
                       f"ground_truth ({len(gt_tokens)} tokens).")
    return False, f"Answer ({len(a_tokens)} tokens) vs ground_truth ({len(gt_tokens)} tokens) — OK."


def detect_slow_step(duration_ms: float, threshold_ms: float = 500.0) -> Tuple[bool, str]:
    """检测耗时过长：超过阈值判定为性能瓶颈。

    Args:
        duration_ms: 步骤耗时
        threshold_ms: 阈值（默认 500ms，模拟本地快速规则时实际值更小）
    """
    if duration_ms > threshold_ms:
        return True, f"Step took {duration_ms:.1f}ms, exceeding threshold of {threshold_ms:.0f}ms."
    return False, f"Step duration {duration_ms:.1f}ms within threshold ({threshold_ms:.0f}ms)."


# ============================================================================
# 诊断编排器——组合所有检测器，映射排障五步法的"③ 定位根因"
# ============================================================================

def diagnose_all(
    query: str,
    contexts: List[str],
    answer: str,
    ground_truth: str = "",
    step_durations: Optional[List[Tuple[str, float]]] = None,
) -> List[Failure]:
    """对所有步骤运行检测器，返回检测到的失败项列表。

    Args:
        query: 用户查询
        contexts: 检索到的上下文
        answer: 模型回答
        ground_truth: 参考答案（可选，用于残缺检测）
        step_durations: [(step_name, duration_ms), ...] 用于慢步骤检测

    Returns:
        Failure 列表（空列表表示流水线健康）
    """
    failures: List[Failure] = []

    # 1. 空检索检测
    is_empty, desc = detect_empty_context(contexts)
    if is_empty:
        failures.append(Failure("empty_context", "retrieval", desc, "high"))

    # 2. 无关检索检测
    is_irrel, desc = detect_irrelevant_context(contexts, query)
    if is_irrel:
        failures.append(Failure("irrelevant_context", "retrieval", desc, "high"))

    # 3. 幻觉检测（需非空 context）
    if not is_empty:
        is_hall, desc = detect_hallucination(answer, contexts)
        if is_hall:
            failures.append(Failure("hallucination", "generation", desc, "medium"))

    # 4. 答案残缺检测（需 ground_truth）
    if ground_truth:
        is_inc, desc = detect_incomplete_answer(answer, ground_truth)
        if is_inc:
            failures.append(Failure("incomplete_answer", "generation", desc, "medium"))

    # 5. 慢步骤检测
    if step_durations:
        for step_name, dur in step_durations:
            is_slow, desc = detect_slow_step(dur)
            if is_slow:
                failures.append(Failure("slow_step", step_name, desc, "low"))

    return failures


# ============================================================================
# 流水线追踪主函数——模拟一段 RAG 流水线执行并诊断
# ============================================================================

def trace_rag_pipeline(
    query: str,
    contexts: List[str],
    answer: str,
    ground_truth: str = "",
) -> PipelineTrace:
    """追踪一条 RAG 流水线的完整执行，生成诊断报告。

    步骤: retrieval → context → generation → answer

    Args:
        query:        用户查询
        contexts:     检索返回的上下文
        answer:       模型生成的回答
        ground_truth: 参考答案（用于完整性诊断）

    Returns:
        PipelineTrace 对象，含 .report() 和 .diagnose() 方法
    """
    trace = PipelineTrace(query=query)

    # Step 1: 检索
    t0 = time.perf_counter()
    retrieval_detail = (f"Retrieved {len(contexts)} chunks, "
                        f"{sum(len(c.split()) for c in contexts)} total words")
    time.sleep(0.01)  # 模拟检索耗时
    t1 = time.perf_counter()
    trace.add_step(TraceStep(
        step_name="retrieval",
        input_data=f"query ({len(query.split())} words)",
        output_data=f"{len(contexts)} context(s)",
        duration_ms=(t1 - t0) * 1000,
        detail=retrieval_detail,
    ))

    # Step 2: 上下文拼接
    t0 = time.perf_counter()
    ctx_text = " ".join(contexts)
    time.sleep(0.005)  # 模拟拼接耗时
    t1 = time.perf_counter()
    trace.add_step(TraceStep(
        step_name="context_merge",
        input_data=f"{len(contexts)} context(s)",
        output_data=f"{len(ctx_text.split())} words merged",
        duration_ms=(t1 - t0) * 1000,
    ))

    # Step 3: 生成
    t0 = time.perf_counter()
    gen_detail = (
        f"Generating answer from {len(ctx_text.split())}-word context; "
        f"answer: {answer[:40]}..."
    )
    time.sleep(0.02)  # 模拟 LLM 生成耗时
    t1 = time.perf_counter()
    trace.add_step(TraceStep(
        step_name="generation",
        input_data=f"merged context ({len(ctx_text.split())} words)",
        output_data=f"answer ({len(answer.split())} words)",
        duration_ms=(t1 - t0) * 1000,
        detail=gen_detail,
    ))

    # Step 4: 回答输出
    t0 = time.perf_counter()
    time.sleep(0.002)  # 模拟后处理
    t1 = time.perf_counter()
    trace.add_step(TraceStep(
        step_name="answer_output",
        input_data=f"raw answer ({len(answer.split())} words)",
        output_data=f"final answer ({len(answer.split())} words)",
        duration_ms=(t1 - t0) * 1000,
    ))

    # 诊断
    step_durations = [(s.step_name, s.duration_ms) for s in trace.steps]
    trace.failures = diagnose_all(query, contexts, answer, ground_truth, step_durations)

    return trace

# ============================================================================
# 批量诊断汇总
# ============================================================================

@dataclass
class DiagnosisSummary:
    """批量诊断的汇总统计。"""
    total_samples: int
    healthy_count: int
    failure_breakdown: Dict[str, int]  # failure_type → count

    def report(self) -> str:
        lines = [
            "=" * 56,
            "  Diagnosis Summary (Total: {}/{})".format(
                self.healthy_count, self.total_samples),
            "=" * 56,
        ]
        if not self.failure_breakdown:
            lines.append("  All samples healthy. No failures detected.")
        else:
            total_failures = sum(self.failure_breakdown.values())
            lines.append(f"  {'Failure Type':<24} {'Count':>6} {'Rate':>8}")
            lines.append(f"  {'-'*24} {'-'*6} {'-'*8}")
            for ftype, count in sorted(self.failure_breakdown.items(),
                                       key=lambda x: -x[1]):
                rate = count / self.total_samples
                lines.append(f"  {ftype:<24} {count:>6} {rate:>7.0%}")
            lines.append(f"  {'-'*24} {'-'*6} {'-'*8}")
            lines.append(f"  {'Total failures':<24} {total_failures:>6}")
        return "\n".join(lines)