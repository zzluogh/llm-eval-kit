"""针海捞针（Needle-in-a-Haystack）长上下文测试框架。

评估LLM在超长文档中检索关键信息的能力：
  - generate_hay / insert_needle: 构造测试文档
  - run_test_suite / compute_accuracy: 批量评测
  - mock_model_query: 本地模拟检索（无需GPU）

典型用法:
    >>> from llm_eval.needle_haystack import run_test_suite, mock_model_query, compute_accuracy
    >>> results = run_test_suite([100, 500], [0.0, 0.5, 1.0], mock_model_query)
    >>> compute_accuracy(results)
    1.0
"""

import random
import string
import math


def generate_hay(num_words):
    """生成随机小写单词组成的"草堆"文本。

    Args:
        num_words: 生成的单词数量

    Returns:
        空格分隔的随机小写字符串，共 num_words 个单词

    Examples:
        >>> hay = generate_hay(3)
        >>> len(hay.split())
        3
    """
    words = []
    for _ in range(num_words):
        word = ''.join(random.choices(string.ascii_lowercase, k=random.randint(3, 10)))
        words.append(word)
    return ' '.join(words)


def estimate_tokens(text):
    """估算文本的 token 数量。

    英文经验公式: 1 token ≈ 0.75 word, 向上取整保留安全余量。

    Args:
        text: 输入文本

    Returns:
        int, 估算的 token 数
    """
    words = text.split()
    return math.ceil(len(words) / 0.75)


def generate_needle():
    """生成一条标准的"针"信息，包含可识别的唯一 key。

    Returns:
        含 "NEEDLE-42" 的固定格式字符串
    """
    return "The secret key is NEEDLE-42"


def insert_needle(hay, needle, position):
    """在草堆文本的指定位置插入针信息。

    Args:
        hay: 原始草堆文本
        needle: 要插入的针信息
        position: 插入位置比例, 0.0=开头, 0.5=中间, 1.0=末尾

    Returns:
        插入 needle 后的完整文本

    Examples:
        >>> insert_needle("a b c d e", "NEEDLE", 0.0)
        'NEEDLE a b c d e'
        >>> insert_needle("a b c d e", "NEEDLE", 1.0)
        'a b c d e NEEDLE'
    """
    words = hay.split()
    idx = int(len(words) * position)
    #idx = round(len(words) * position)
    words.insert(idx, needle)
    return ' '.join(words)


def check_needle_found(model_output, needle):
    """检查模型输出中是否包含针信息。

    Args:
        model_output: 模型返回的文本
        needle: 预期的针信息

    Returns:
        True 如果 needle 完整出现在 model_output 中
    """
    return needle in model_output
    #return needle.lower() in model_output.lower()


def create_test_case(num_words, position, needle=None):
    """生成一个完整的测试用例（草堆 + 插针 + 元数据）。

    Args:
        num_words: 草堆的单词数
        position: 针插入位置比例 (0.0 ~ 1.0)
        needle: 自定义针信息, None 则使用默认针

    Returns:
        dict, 包含 doc / needle / position / num_words / estimated_tokens
    """
    if needle is None:
        needle = generate_needle()
    hay = generate_hay(num_words)
    doc = insert_needle(hay, needle, position)
    return {
        "num_words": num_words,
        "doc": doc,
        "needle": needle,
        "position": position,
        #"estimated_tokens": estimate_tokens(doc)
        "estimated_tokens": estimate_tokens(doc),
    }


def mock_model_query(doc, search_key="secret key"):
    """模拟模型检索: 在文档中搜索关键短语并返回上下文片段。

    用于本地验证框架逻辑，不依赖真实LLM。

    Args:
        doc: 待检索的文档
        search_key: 搜索的关键词, 默认 "secret key"

    Returns:
        若找到: 返回 search_key 所在位置的上下文片段 (前后各取若干字符)
        若未找到: 返回 "I couldn't find the secret key."
    """
    idx = doc.find(search_key)
    if idx == -1:
        return "I couldn't find the secret key."
    start = max(0, idx - 20)
    end = min(len(doc), idx + 80)
    return doc[start:end]

# 新增文件 my_ollama.py:
def ollama_query(doc):
    import requests
    resp = requests.post("http://localhost:11434/api/chat", json={
        "model": "qwen2.5:0.5b",
        "messages": [{"role":"user", "content": f"What is the secret key in this text: {doc}"}],
        "stream": False
    })
    return resp.json()["message"]["content"]


def run_test_suite(doc_lengths, positions, needle_finder, needle=None):
    """批量运行针海捞针测试套件。

    Args:
        doc_lengths: 文档长度列表, 如 [500, 1000, 2000, 4000]
        positions: 插入位置列表, 如 [0.0, 0.5, 0.9]
        needle_finder: 检索函数, 签名为 f(doc: str) -> str
        needle: 统一使用的针信息, None 则每次自动生成

    Returns:
        dict, key="W{len}_P{pos}" → bool, 表示该用例针是否被找到

    Examples:
        >>> def always(doc): return "The secret key is NEEDLE-42"
        >>> results = run_test_suite([100], [0.5], always)
        >>> results["W100_P0.5"]
        True
    """
    results = {}
    for length in doc_lengths:
        for pos in positions:
            case = create_test_case(length, pos, needle)
            output = needle_finder(case["doc"])
            found = check_needle_found(output, case["needle"])
            key = f"W{length}_P{pos:.1f}"
            results[key] = found
    return results


def compute_accuracy(results):
    """计算检索准确率。

    Args:
        results: run_test_suite 返回的 dict

    Returns:
        float, 0.0 ~ 1.0, 找到针的比例
    """
    if not results:
        return 0.0
    return sum(1 for v in results.values() if v) / len(results)
