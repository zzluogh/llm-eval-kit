"""Function Calling / Tool Calling 准确度评测框架。(第29课, 第31课引用)

评估 LLM 工具调用的准确性：
  - tool_name_accuracy:    工具名是否选对
  - param_precision:       模型给对了多少参数值（不给多余参数）
  - param_recall:          期望参数中有多少被正确给出
  - param_f1:              precision 和 recall 的调和平均
  - exact_match:           工具名 + 所有参数完全正确

用法:
    from my_math.tool_calling_eval import evaluate, ToolCallSample, ToolDef
    samples = [ToolCallSample(query="...", tools_available=[...], ...)]
    result = evaluate(samples)
    print(result.report())
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class ToolDef:
    """工具定义。

    Args:
        name: 工具名（如 get_weather、set_timer）
        description: 工具功能描述
        parameters: 参数定义 dict, key=参数名, value=类型描述
    """
    name: str
    description: str
    parameters: Dict[str, str]


@dataclass
class ToolCallSample:
    """单条工具调用评测样本。

    Args:
        query: 用户自然语言查询
        tools_available: 可用工具列表
        expected_tool: 期望模型调用的工具名
        expected_params: 期望模型使用的参数值
        predicted_tool: 模型实际调用的工具名（模拟或真实模型输出）
        predicted_params: 模型实际使用的参数值
    """
    query: str
    tools_available: List[ToolDef]
    expected_tool: str
    expected_params: Dict[str, Any]
    predicted_tool: str = ""
    predicted_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCallScores:
    """单条样本的工具调用得分。

    Args:
        tool_name_match: 工具名是否匹配 (0.0 或 1.0)
        param_precision: 参数精确率 (正确参数数 / 预测参数总数)
        param_recall: 参数召回率 (正确参数数 / 期望参数总数)
        param_f1: precision 和 recall 的调和平均
        exact_match: 工具名 + 全部参数完全正确 (0.0 或 1.0)
    """
    tool_name_match: float = 0.0
    param_precision: float = 0.0
    param_recall: float = 0.0
    param_f1: float = 0.0
    exact_match: float = 0.0


@dataclass
class ToolCallEvalResult:
    """批量评测结果。

    Args:
        per_sample: 每条样本的 5 项得分
        avg: 所有样本的平均得分
    """
    per_sample: List[ToolCallScores] = field(default_factory=list)
    avg: ToolCallScores = field(default_factory=ToolCallScores)

    def report(self, labels: List[str] | None = None) -> str:
        """生成 ASCII 表格评测报告。

        Args:
            labels: 每条样本的标签列表（如 "Q1","Q2"...），默认用序号

        Returns:
            格式化的多行字符串报告
        """
        if labels is None:
            labels = [f"Q{i+1}" for i in range(len(self.per_sample))]
        header = f"{'Sample':<8} {'ToolOK':>7} {'pPrec':>7} {'pRecall':>6} {'pF1':>7} {'Exact':>7}"
        lines = [header, "-" * len(header)]
        for label, s in zip(labels, self.per_sample):
            lines.append(
                f"{label:<8} {s.tool_name_match:7.3f} {s.param_precision:7.3f} "
                f"{s.param_recall:6.3f} {s.param_f1:7.3f} {s.exact_match:7.3f}"
            )
        lines.append("-" * len(header))
        a = self.avg
        lines.append(
            f"{'AVG':<8} {a.tool_name_match:7.3f} {a.param_precision:7.3f} "
            f"{a.param_recall:6.3f} {a.param_f1:7.3f} {a.exact_match:7.3f}"
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _normalize_tool_name(name: str) -> str:
    """归一化工具名：小写 + 去首尾空白。

    Args:
        name: 原始工具名

    Returns:
        归一化后的工具名

    Examples:
        >>> _normalize_tool_name(" Get_Weather ")
        'get_weather'
    """
    return name.strip().lower()


def _compare_params(expected: Dict[str, Any], predicted: Dict[str, Any],
                     synonyms: Dict[str, Dict[str, str]] | None = None) -> tuple:
    """对比两套参数，返回匹配统计。

    字符串值按归一化比较（小写 + strip），数值直接 ==。
    支持同义词表归一化：先查 synonyms[key] 映射再比较。

    Args:
        expected: 期望参数 dict
        predicted: 模型预测参数 dict
        synonyms: {param_key: {synonym: canonical}}, 如 {"city": {"peking": "beijing"}}

    Returns:
        (correct_count, predicted_count, expected_count)

    Examples:
        >>> _compare_params({"city": "Beijing"}, {"city": "beijing", "units": "celsius"})
        (1, 2, 1)
    """
    correct = 0
    for key, exp_val in expected.items():
        pred_val = predicted.get(key)
        if pred_val is None:
            continue
        if isinstance(exp_val, str) and isinstance(pred_val, str):
            e = exp_val.strip().lower()
            p = pred_val.strip().lower()
            # 同义词归一化
            if synonyms and key in synonyms:
                syn_map = synonyms[key]
                e = syn_map.get(e, e)
                p = syn_map.get(p, p)
            if e == p:
                correct += 1
        else:
            if exp_val == pred_val:
                correct += 1
    return correct, len(predicted), len(expected)


def _safe_f1(precision: float, recall: float) -> float:
    """计算 F1 分数，处理分母为 0 的情况。

    Args:
        precision: 精确率
        recall: 召回率

    Returns:
        F1 分数, 0.0 如果 P+R==0
    """
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# 5 项评测指标
# ---------------------------------------------------------------------------

def tool_name_accuracy(expected_tool: str, predicted_tool: str) -> float:
    """工具名是否选对。

    Args:
        expected_tool: 期望的工具名
        predicted_tool: 模型预测的工具名

    Returns:
        1.0 匹配, 0.0 不匹配

    Examples:
        >>> tool_name_accuracy("get_weather", "Get_Weather")
        1.0
        >>> tool_name_accuracy("get_weather", "set_timer")
        0.0
    """
    return 1.0 if _normalize_tool_name(expected_tool) == _normalize_tool_name(predicted_tool) else 0.0


def param_precision(expected: Dict[str, Any], predicted: Dict[str, Any],
                     synonyms: Dict[str, Dict[str, str]] | None = None) -> float:
    """参数精确率：模型预测的参数中有多少是正确的。

    Args:
        expected: 期望参数
        predicted: 模型预测参数
        synonyms: 同义词表

    Returns:
        0.0 ~ 1.0。双方都空返回 1.0（无预测 = 无错误）。

    Examples:
        >>> param_precision({"city": "Beijing"}, {"city": "Beijing", "units": "celsius"})
        0.5  # 2 个预测, 1 个正确
    """
    correct, pred_count, exp_count = _compare_params(expected, predicted, synonyms)
    if pred_count == 0:
        return 1.0 if exp_count == 0 else 0.0
    return correct / pred_count


def param_recall(expected: Dict[str, Any], predicted: Dict[str, Any],
                  synonyms: Dict[str, Dict[str, str]] | None = None) -> float:
    """参数召回率：期望参数中有多少被模型正确给出。

    Args:
        expected: 期望参数
        predicted: 模型预测参数
        synonyms: 同义词表

    Returns:
        0.0 ~ 1.0。双方都空返回 1.0（无期望 = 无遗漏）。

    Examples:
        >>> param_recall({"city": "Beijing", "date": "2026-06-15"}, {"city": "Beijing"})
        0.5  # 2 个期望, 1 个命中
    """
    correct, pred_count, exp_count = _compare_params(expected, predicted, synonyms)
    if exp_count == 0:
        return 1.0 if pred_count == 0 else 0.0
    return correct / exp_count


def param_f1(expected: Dict[str, Any], predicted: Dict[str, Any],
             synonyms: Dict[str, Dict[str, str]] | None = None) -> float:
    """参数 F1：precision 和 recall 的调和平均。

    Args:
        expected: 期望参数
        predicted: 模型预测参数
        synonyms: 同义词表

    Returns:
        0.0 ~ 1.0
    """
    p = param_precision(expected, predicted, synonyms)
    r = param_recall(expected, predicted, synonyms)
    return _safe_f1(p, r)


def exact_match_score(expected_tool: str, predicted_tool: str,
                      expected_params: Dict[str, Any],
                      predicted_params: Dict[str, Any],
                      synonyms: Dict[str, Dict[str, str]] | None = None) -> float:
    """完全匹配：工具名正确 + 所有参数值完全一致。

    Args:
        expected_tool: 期望工具名
        predicted_tool: 预测工具名
        expected_params: 期望参数
        predicted_params: 预测参数
        synonyms: 同义词表

    Returns:
        1.0 完全匹配, 0.0 否则
    """
    if tool_name_accuracy(expected_tool, predicted_tool) != 1.0:
        return 0.0
    p = param_precision(expected_params, predicted_params, synonyms)
    r = param_recall(expected_params, predicted_params, synonyms)
    if p == 1.0 and r == 1.0:
        return 1.0
    return 0.0


# ---------------------------------------------------------------------------
# 聚合评估
# ---------------------------------------------------------------------------

def evaluate_sample(sample: ToolCallSample,
                    synonyms: Dict[str, Dict[str, str]] | None = None) -> ToolCallScores:
    """对单条样本计算 5 项工具调用指标。

    Args:
        sample: ToolCallSample 实例
        synonyms: 可选的同义词表 {param_key: {synonym: canonical}}

    Returns:
        ToolCallScores 包含 5 项指标得分
    """
    return ToolCallScores(
        tool_name_match=tool_name_accuracy(sample.expected_tool, sample.predicted_tool),
        param_precision=param_precision(sample.expected_params, sample.predicted_params, synonyms),
        param_recall=param_recall(sample.expected_params, sample.predicted_params, synonyms),
        param_f1=param_f1(sample.expected_params, sample.predicted_params, synonyms),
        exact_match=exact_match_score(
            sample.expected_tool, sample.predicted_tool,
            sample.expected_params, sample.predicted_params, synonyms,
        ),
    )


def evaluate(samples: List[ToolCallSample],
             synonyms: Dict[str, Dict[str, str]] | None = None) -> ToolCallEvalResult:
    """批量评测多条工具调用样本。

    Args:
        samples: ToolCallSample 列表
        synonyms: 可选的同义词表

    Returns:
        ToolCallEvalResult 含逐条得分和平均分
    """
    scores = [evaluate_sample(s, synonyms) for s in samples]
    n = len(scores) if scores else 1
    avg = ToolCallScores(
        tool_name_match=sum(s.tool_name_match for s in scores) / n,
        param_precision=sum(s.param_precision for s in scores) / n,
        param_recall=sum(s.param_recall for s in scores) / n,
        param_f1=sum(s.param_f1 for s in scores) / n,
        exact_match=sum(s.exact_match for s in scores) / n,
    )
    return ToolCallEvalResult(per_sample=scores, avg=avg)
