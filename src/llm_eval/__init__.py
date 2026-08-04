"""llm-eval-kit — 大模型评估工具包。

4 大评估模块 + 1 个诊断框架：
  - ragas_eval        RAGAS 本地实现：faithfulness / answer_relevancy / context_recall / context_precision
  - tool_calling_eval Function Calling 工具调用评测：5 项指标
  - needle_haystack   长上下文针海捞针测试
  - pipeline_trace    流水线追踪 + 自动诊断：5 类失败检测
  - model_adapters    多模型客户端封装
"""

from llm_eval.ragas_eval import (
    RAGSample, MetricScores, RAGEvalResult,
    faithfulness, answer_relevancy, context_recall, context_precision,
    evaluate_sample, evaluate,
)
from llm_eval.tool_calling_eval import (
    ToolDef, ToolCallSample, ToolCallScores, ToolCallEvalResult,
    tool_name_accuracy, param_precision, param_recall, param_f1, exact_match_score,
)
from llm_eval.pipeline_trace import (
    TraceStep, Failure, PipelineTrace,
    detect_hallucination, detect_empty_context, detect_irrelevant_context,
    detect_incomplete_answer, detect_slow_step, diagnose_all, trace_rag_pipeline,
)
