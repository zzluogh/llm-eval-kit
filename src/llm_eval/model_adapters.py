"""Function Calling 模型适配器 — 统一接口 + 两套模拟模型。(第29课-2, 第31课引用)

统一接口:
    model_predict(query, tools) → (tool_name, params_dict)

内置模型:
    rule_based_predict:     关键词匹配, 高准确率 (~85%)
    error_prone_predict:    有意引入错误, 低准确率 (~50%)

用法:
    from llm_eval.model_adapters import rule_based_predict, error_prone_predict
    from llm_eval.tool_calling_eval import ToolDef
    tool, params = rule_based_predict("weather in Beijing?", [weather_tool])
"""
import re
import random
from typing import List, Dict, Tuple, Any
from llm_eval.tool_calling_eval import ToolDef


# ---------------------------------------------------------------------------
# 统一接口
# ---------------------------------------------------------------------------

def model_predict(query: str, tools: List[ToolDef]) -> Tuple[str, Dict[str, Any]]:
    """给定用户 query 和可用工具列表，返回 (tool_name, params)。

    Args:
        query: 用户自然语言查询
        tools: 可用工具列表

    Returns:
        (tool_name, params_dict)。不调工具时返回 ("", {})
    """
    return rule_based_predict(query, tools)


# ---------------------------------------------------------------------------
# 模型A: rule_based_predict — 基于关键词规则匹配
# ---------------------------------------------------------------------------

# 工具名关键词映射 (tool_name → list of trigger phrases)
_TOOL_KEYWORDS: Dict[str, List[str]] = {
    "get_weather":     ["weather", "temperature", "rain", "sunny", "hot", "cold", "天气"],
    "set_timer":       ["timer", "set a timer", "countdown", "minutes", "minute", "mins"],
    "send_email":      ["send email", "email", "mail to", "发送"],
    "calculate":       ["calculate", "compute", "what is", "2+", "sqrt", "divided by", "times", "plus", "minus", "="],
    "search":          ["search", "find", "look up", "lookup", "tutorials", "news", "query"],
    "translate":       ["translate", "translation", "to french", "to japanese", "to chinese"],
    "get_stock_price": ["stock", "price", "share", "股票", "TSLA", "AAPL", "symbol"],
    "play_music":      ["play", "song", "music", "volume", "beatles", "despacito"],
    "create_reminder": ["remind", "reminder", "remind me", "call mom", "at 2026"],
}

# 参数提取正则
_PARAM_EXTRACTORS = {
    "city":             r"(?:in|for|city of)\s+(\w+)",
    "duration_minutes": r"(\d+)\s*(?:minute|minutes|min|mins)",
    "to":              r"(?:\bto\s+|send\s+to\s+)([\w.@]+)",
    "subject":          r"(?:subject|about|regarding)\s+(.+?)(?:\s+(?:at|to|body|with)|\s*$)",
    "body":             r"(?:body|message|content)\s+(.+)",
    "expression":       r"(?:calculate|compute|what is)\s+(.+)",
    "query":            r"(?:search|find|look\s+up|lookup)\s+(.+?)(?:\s+(?:with|max|limit|in)|\s*$)",
    "text":             r"(?:remind me|translate)\s+(?:to\s+)?(.+?)(?:\s+(?:at|to|in|for)|\s*$)",
    "target_language":  r"(?:to|in)\s+(\w+(?:ese|ish|ch|an)?)",
    "symbol":           r"\b(TSLA|AAPL|GOOGL|MSFT|AMZN|META)\b",
    "song_name":        r"(?:play|song)\s+(.+?)(?:\s+(?:at|volume|by)|\s*$)",
    "datetime":         r"at\s+([\d\-]+\s+[\d:]+)",
    "max_results":      r"(?:max\s+)?(\d+)\s*(?:result|results|items)?",
    "volume":           r"(?:volume|vol)\s+(\d+)",
}

# 工具所需参数列表
_TOOL_PARAMS: Dict[str, List[str]] = {
    "get_weather":     ["city", "units"],
    "set_timer":       ["duration_minutes", "label"],
    "send_email":      ["to", "subject", "body"],
    "calculate":       ["expression"],
    "search":          ["query", "max_results"],
    "translate":       ["text", "target_language"],
    "get_stock_price": ["symbol"],
    "play_music":      ["song_name", "volume"],
    "create_reminder": ["text", "datetime"],
}


def rule_based_predict(query: str, tools: List[ToolDef]) -> Tuple[str, Dict[str, Any]]:
    """基于关键词规则提取工具调用。

    流程:
        1. 扫描 query 匹配 _TOOL_KEYWORDS → 选择匹配数最多的工具
        2. 用正则从 query 提取该工具所需的参数
        3. 返回 (tool_name, params), 无匹配返回 ("", {})

    Args:
        query: 用户查询
        tools: 可用工具列表

    Returns:
        (tool_name, params_dict)
    """
    if not tools:
        return "", {}

    available_names = {t.name for t in tools}

    # 1. 工具匹配: 统计各工具关键词命中数
    scores: Dict[str, int] = {}
    lower_q = query.lower()
    for name, keywords in _TOOL_KEYWORDS.items():
        if name not in available_names:
            continue
        score = sum(1 for kw in keywords if kw.lower() in lower_q)
        if score > 0:
            scores[name] = score

    if not scores:
        return "", {}

    best_tool = max(scores, key=scores.get)

    # 2. 参数提取
    params: Dict[str, Any] = {}
    for param_name in _TOOL_PARAMS.get(best_tool, []):
        if param_name in _PARAM_EXTRACTORS:
            m = re.search(_PARAM_EXTRACTORS[param_name], query, re.IGNORECASE)
            if m:
                value = m.group(1).strip()
                # 数值参数转换
                if param_name in ("duration_minutes", "max_results", "volume"):
                    try:
                        value = int(value)
                    except ValueError:
                        pass
                params[param_name] = value
            elif param_name in ("body",):
                params[param_name] = ""  # body 可选

    return best_tool, params


# ---------------------------------------------------------------------------
# 模型B: error_prone_predict — 有意引入错误的对比模型
# ---------------------------------------------------------------------------

_ERROR_TYPES = ["wrong_tool", "wrong_param_value", "missing_param", "extra_param"]

_WRONG_CITIES = ["Shanghai", "Tokyo", "Paris", "Sydney", "Moscow"]
_WRONG_LANGUAGES = ["German", "Korean", "Spanish", "Italian", "Arabic"]


def error_prone_predict(query: str, tools: List[ToolDef]) -> Tuple[str, Dict[str, Any]]:
    """在 rule_based 基础上随机引入错误，模拟不准确的模型。

    错误率: 约 40%, 包括:
      - 选错工具 (30% 时候选第二匹配)
      - 参数值随机替换 (城市/语言/数值)
      - 随机删除参数 (30%)
      - 随机增加多余参数 (20%)

    Args:
        query: 用户查询
        tools: 可用工具列表

    Returns:
        (tool_name, params_dict)
    """
    tool, params = rule_based_predict(query, tools)

    if not tool:
        return tool, params

    # 1. 选错工具 (35% => 选第二匹配; 无第二匹配时随机选一个)
    if random.random() < 0.35 and len(tools) >= 2:
        available_names = {t.name for t in tools}
        lower_q = query.lower()
        scores: Dict[str, int] = {}
        for name, keywords in _TOOL_KEYWORDS.items():
            if name not in available_names:
                continue
            score = sum(1 for kw in keywords if kw.lower() in lower_q)
            if score > 0:
                scores[name] = score
        # 尝试选第二高分，没有则随机
        sorted_tools = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_tools) >= 2 and sorted_tools[1][0] != tool:
            tool, params = rule_based_predict(query, [td for td in tools if td.name == sorted_tools[1][0]])
        else:
            # 随机选一个不等于当前工具的
            candidates = [t for t in tools if t.name != tool]
            if candidates:
                tool, params = rule_based_predict(query, candidates)

    # 2. 替换参数值 (30% => 随机替换)
    if params:
        for key in list(params.keys()):
            if random.random() < 0.30:
                if key == "city":
                    params[key] = random.choice(_WRONG_CITIES)
                elif key == "target_language":
                    params[key] = random.choice(_WRONG_LANGUAGES)
                elif key in ("duration_minutes", "max_results", "volume"):
                    if isinstance(params[key], int):
                        params[key] = params[key] + random.choice([-3, -1, 2, 5, 10])
                elif key in ("to",):
                    params[key] = params[key].replace("@", "_at_")

    # 3. 随机删除参数
    if params and random.random() < 0.3:
        remove_key = random.choice(list(params.keys()))
        del params[remove_key]

    # 4. 随机增加多余参数
    if random.random() < 0.2:
        extra_params = {
            "get_weather": ("units", "celsius"),
            "set_timer": ("label", "my timer"),
            "search": ("max_results", 5),
            "play_music": ("volume", 50),
            "send_email": ("priority", "high"),
        }
        if tool in extra_params:
            key, val = extra_params[tool]
            if key not in params:
                params[key] = val

    return tool, params
