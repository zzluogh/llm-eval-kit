"""RAGAS 本地实现 — RAG 系统评测 4 核心指标。(第28课, 第31课引用)

无外部 NLP 模型依赖，基于 token 重叠 + 字符串匹配的轻量实现。
指标定义对齐 ragas 官方文档 (docs.ragas.io)：
  - faithfulness:      答案是否忠于检索到的上下文
  - answer_relevancy:  答案是否切题
  - context_recall:    检索结果覆盖了多少参考答案信息
  - context_precision: 检索结果中有多少与问题相关

用法:
    from llm_eval.ragas_eval import evaluate, RAGSample, RAGEvalResult
    samples = [RAGSample(question="...", contexts=["..."], answer="...", ground_truth="...")]
    result = evaluate(samples)
    print(result.report())
"""
from __future__ import annotations
import re
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Set

# 英文停用词 — 在 precision / relevancy 中过滤，避免 "is" "the" 等造成假阳性
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


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class RAGSample:
    """单条 RAG 评测样本。

    Args:
        question: 用户问题
        contexts: 检索系统返回的上下文列表（通常 1~5 条片段）
        answer: 模型基于 contexts 生成的回答
        ground_truth: 人工标注的参考答案 / 金标准
    """
    question: str
    contexts: List[str]
    answer: str
    ground_truth: str


@dataclass
class MetricScores:
    """单条样本的 4 指标得分。

    Args:
        faithfulness: 0.0~1.0，答案忠于上下文的比例
        answer_relevancy: 0.0~1.0，答案切题程度
        context_recall: 0.0~1.0，上下文覆盖参考答案的比例
        context_precision: 0.0~1.0，上下文中与问题相关的比例
    """
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_recall: float = 0.0
    context_precision: float = 0.0


@dataclass
class RAGEvalResult:
    """批量评测结果。

    Args:
        per_sample: 每条样本的 4 指标得分
        avg: 所有样本的平均指标
    """
    per_sample: List[MetricScores] = field(default_factory=list)
    avg: MetricScores = field(default_factory=MetricScores)

    def report(self, labels: List[str] | None = None) -> str:
        """生成 ASCII 表格评测报告。

        Args:
            labels: 每条样本的标签列表（如 "Q1","Q2"...），默认用序号

        Returns:
            格式化的多行字符串报告
        """
        if labels is None:
            labels = [f"Q{i+1}" for i in range(len(self.per_sample))]
        header = f"{'Sample':<8} {'Faith':>7} {'Relv':>7} {'cRecall':>7} {'cPrec':>7}"
        lines = [header, "-" * len(header)]
        for label, s in zip(labels, self.per_sample):
            lines.append(
                f"{label:<8} {s.faithfulness:7.3f} {s.answer_relevancy:7.3f} "
                f"{s.context_recall:7.3f} {s.context_precision:7.3f}"
            )
        lines.append("-" * len(header))
        a = self.avg
        lines.append(
            f"{'AVG':<8} {a.faithfulness:7.3f} {a.answer_relevancy:7.3f} "
            f"{a.context_recall:7.3f} {a.context_precision:7.3f}"
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """简易分词：转小写 → 按非字母数字切分 → 过滤空串。

    Args:
        text: 任意英文文本

    Returns:
        小写单词列表

    Examples:
        >>> _tokenize("ROS2 is a framework!")
        ['ros2', 'is', 'a', 'framework']
    """
    return [t for t in re.split(r'[^a-zA-Z0-9]+', text.lower()) if t]


def _overlap_ratio(tokens_a: List[str], tokens_b: List[str]) -> float:
    """计算 tokens_a 中有多大比例出现在 tokens_b 中。

    Args:
        tokens_a: 分子集合
        tokens_b: 分母参照集合

    Returns:
        0.0 ~ 1.0，tokens_a 在 tokens_b 中的命中比例

    Examples:
        >>> _overlap_ratio(['a','b','c'], ['a','c','d'])
        0.6666...   # 2/3
    """
    if not tokens_a:
        return 0.0
    set_b = set(tokens_b)
    hits = sum(1 for t in tokens_a if t in set_b)
    return hits / len(tokens_a)


def _cosine_similarity(tokens_a: List[str], tokens_b: List[str]) -> float:
    """两组 token 的余弦相似度（基于词频向量）。

    Args:
        tokens_a: 第一组 token
        tokens_b: 第二组 token

    Returns:
        0.0 ~ 1.0，余弦相似度
    """
    if not tokens_a or not tokens_b:
        return 0.0
    # 词频
    freq_a: Dict[str, int] = {}
    freq_b: Dict[str, int] = {}
    for t in tokens_a:
        freq_a[t] = freq_a.get(t, 0) + 1
    for t in tokens_b:
        freq_b[t] = freq_b.get(t, 0) + 1
    # 所有出现过的词
    all_keys = set(freq_a.keys()) | set(freq_b.keys())
    dot = sum(freq_a.get(k, 0) * freq_b.get(k, 0) for k in all_keys)
    mag_a = math.sqrt(sum(v * v for v in freq_a.values()))
    mag_b = math.sqrt(sum(v * v for v in freq_b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _sentence_tokenize(text: str) -> List[str]:
    """按句号/问号/感叹号切分句子，保留非空句。

    Args:
        text: 任意文本

    Returns:
        句子列表（已 strip 且非空）
    """
    raw = re.split(r'[.?!]+', text)
    return [s.strip() for s in raw if s.strip()]


# ---------------------------------------------------------------------------
# 4 核心指标
# ---------------------------------------------------------------------------

def faithfulness(answer: str, contexts: List[str]) -> float:
    """答案忠于上下文程度。

    逻辑：把 answer 拆成句子，逐句与 contexts 拼接文本做词重叠。
    如果某句 50% 以上的词在 context 中出现过 → 该句"有依据"。
    返回 "有依据句数 / 总句数"。

    对齐 ragas.metrics.faithfulness 的定义：
    > Measures how factually consistent the generated answer is with the given context.

    Args:
        answer: 模型生成的回答
        contexts: 检索返回的上下文列表

    Returns:
        0.0 ~ 1.0

    Examples:
        >>> faithfulness("ROS2 is a robot OS.", ["ROS2 is the next generation ROS."])
        0.5  # "ROS2 is" 有依据, "a robot OS" 无依据 → 视具体分词
    """
    sentences = _sentence_tokenize(answer)
    if not sentences:
        return 0.0
    ctx_text = " ".join(contexts)
    ctx_tokens = _tokenize(ctx_text)
    supported = 0
    for sent in sentences:
        sent_tokens = _tokenize(sent)
        if not sent_tokens:
            supported += 1  # 空句算有依据
            continue
        ratio = _overlap_ratio(sent_tokens, ctx_tokens)
        if ratio >= 0.5:
            supported += 1
    return supported / len(sentences)


def answer_relevancy(answer: str, question: str) -> float:
    """答案与问题的语义相关度。

    逻辑：用余弦相似度衡量 question 和 answer 的词频向量距离。
    对齐 ragas.metrics.answer_relevancy 的定义：
    > Assesses how pertinent the generated answer is to the given prompt.

    Args:
        answer: 模型生成的回答
        question: 用户问题

    Returns:
        0.0 ~ 1.0

    Examples:
        >>> answer_relevancy("ROS2 is a robot framework.", "What is ROS2?")
        > 0.5  # "ros2" 重合
    """
    return _cosine_similarity(_tokenize(question), _tokenize(answer))


def context_recall(contexts: List[str], ground_truth: str) -> float:
    """检索上下文对参考答案的覆盖率。

    逻辑：ground_truth 中有多少词汇能从 contexts 中找到。
    对齐 ragas.metrics.context_recall 的定义：
    > Measures the extent to which the retrieved context aligns with the
      ground truth answer.

    Args:
        contexts: 检索返回的上下文列表
        ground_truth: 参考答案

    Returns:
        0.0 ~ 1.0

    Examples:
        >>> context_recall(["ROS2 uses DDS middleware."], "ROS2 uses DDS for communication.")
        > 0.6  # "ros2 uses dds" 重合 4/6
    """
    #gt_tokens = _tokenize(ground_truth)
    gt_token = [t for t in _tokenize(ground_truth) if t not in _STOP_WORDS]
    #ctx_text = " ".join(contexts)
    #ctx_tokens = _tokenize(ctx_text)
    ctx_tokens = [t for t in _tokenize(" ".join(contexts)) if t not in _STOP_WORDS]
    return _overlap_ratio(gt_token, ctx_tokens)


def context_precision(contexts: List[str], question: str) -> float:
    """检索上下文的精准度（是否返回了无关内容）。

    逻辑：context 中有多少词与 question 相关（含扩展）。
    实际做法：取 contexts 中与 question 有有意义词重叠的句子比例。
    question 侧过滤停用词（is/the/what 等），context 侧不过滤。
    对齐 ragas.metrics.context_precision 的定义：
    > Evaluates whether all of the ground-truth relevant items present in
      the contexts are ranked higher or not.

    Args:
        contexts: 检索返回的上下文列表
        question: 用户问题

    Returns:
        0.0 ~ 1.0

    Examples:
        >>> context_precision(["ROS2 is great for robots.", "The sky is blue."],
        ...                   "What is ROS2?")
        0.5  # 第一条相关，第二条无关
    """
    if not contexts:
        return 0.0
    q_tokens_meaningful = {t for t in _tokenize(question) if t not in _STOP_WORDS}
    if not q_tokens_meaningful:
        return 0.0
    relevant = 0
    for ctx in contexts:
        ctx_tokens = set(_tokenize(ctx))
        if q_tokens_meaningful & ctx_tokens:
            relevant += 1
    return relevant / len(contexts)


# ---------------------------------------------------------------------------
# 聚合评估
# ---------------------------------------------------------------------------

def evaluate_sample(sample: RAGSample) -> MetricScores:
    """对单条样本计算 4 个指标。

    Args:
        sample: RAGSample 实例

    Returns:
        MetricScores 包含 4 个指标得分
    """
    return MetricScores(
        faithfulness=faithfulness(sample.answer, sample.contexts),
        answer_relevancy=answer_relevancy(sample.answer, sample.question),
        context_recall=context_recall(sample.contexts, sample.ground_truth),
        context_precision=context_precision(sample.contexts, sample.question),
    )


def evaluate(samples: List[RAGSample]) -> RAGEvalResult:
    """批量评测多条 RAG 样本。

    Args:
        samples: RAGSample 列表

    Returns:
        RAGEvalResult 含逐条得分和平均分
    """
    scores = [evaluate_sample(s) for s in samples]
    n = len(scores) if scores else 1
    avg = MetricScores(
        faithfulness=sum(s.faithfulness for s in scores) / n,
        answer_relevancy=sum(s.answer_relevancy for s in scores) / n,
        context_recall=sum(s.context_recall for s in scores) / n,
        context_precision=sum(s.context_precision for s in scores) / n,
    )
    return RAGEvalResult(per_sample=scores, avg=avg)
