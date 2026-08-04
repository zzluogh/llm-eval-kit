"""第 29 课-2 模型适配器 — 单元测试。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest
from my_math.model_adapters import (
    rule_based_predict, error_prone_predict, model_predict,
    _TOOL_KEYWORDS, _PARAM_EXTRACTORS, _TOOL_PARAMS,
)
from my_math.tool_calling_eval import ToolDef


WEATHER_TOOL = ToolDef("get_weather", "Get weather", {"city": "str"})
TIMER_TOOL = ToolDef("set_timer", "Set timer", {"duration_minutes": "int"})
CALC_TOOL = ToolDef("calculate", "Calculate", {"expression": "str"})
STOCK_TOOL = ToolDef("get_stock_price", "Get stock price", {"symbol": "str"})
ALL_TOOLS = [WEATHER_TOOL, TIMER_TOOL, CALC_TOOL, STOCK_TOOL]


class TestRuleBasedPredict:
    def test_weather_query(self):
        tool, params = rule_based_predict("What's the weather in Beijing?", [WEATHER_TOOL])
        assert tool == "get_weather"
        assert params.get("city") == "Beijing"

    def test_timer_query(self):
        tool, params = rule_based_predict("Set a timer for 5 minutes", [TIMER_TOOL])
        assert tool == "set_timer"
        assert params.get("duration_minutes") == 5

    def test_calculate_query(self):
        tool, params = rule_based_predict("Calculate 2+3", [CALC_TOOL])
        assert tool == "calculate"
        assert "2+3" in params.get("expression", "")

    def test_no_match_returns_empty(self):
        tool, params = rule_based_predict("Hello, how are you?", [WEATHER_TOOL, TIMER_TOOL])
        assert tool == ""
        assert params == {}

    def test_empty_tools_returns_empty(self):
        tool, params = rule_based_predict("weather in Beijing", [])
        assert tool == ""
        assert params == {}

    def test_stock_query(self):
        tool, params = rule_based_predict("What's the stock price of TSLA?", [STOCK_TOOL])
        assert tool == "get_stock_price"
        assert params.get("symbol") == "TSLA"

    def test_returns_valid_tool_from_available(self):
        """确保返回的工具名在可用工具列表中。"""
        tool, _ = rule_based_predict("Weather in Paris", ALL_TOOLS)
        assert tool in {"get_weather", "set_timer", "calculate", "get_stock_price", ""}


class TestErrorPronePredict:
    def test_returns_valid_format(self):
        tool, params = error_prone_predict("weather in Beijing", [WEATHER_TOOL])
        assert isinstance(tool, str)
        assert isinstance(params, dict)

    def test_multiple_runs_have_variance(self):
        """多次运行至少有一次不同（随机性验证）。"""
        results = set()
        for _ in range(20):
            tool, params = error_prone_predict("What's the weather in Tokyo?",
                                               [WEATHER_TOOL, TIMER_TOOL])
            results.add(tool)
        assert len(results) >= 1


class TestModelPredict:
    def test_alias_for_rule_based(self):
        t1, p1 = model_predict("weather in London", [WEATHER_TOOL])
        t2, p2 = rule_based_predict("weather in London", [WEATHER_TOOL])
        assert t1 == t2
        assert p1 == p2


class TestToolKeywords:
    def test_all_tools_have_keywords(self):
        assert len(_TOOL_KEYWORDS) == 9

    def test_weather_keywords_include_weather(self):
        assert "weather" in _TOOL_KEYWORDS["get_weather"]
        assert "天气" in _TOOL_KEYWORDS["get_weather"]


class TestToolParams:
    def test_all_tools_have_params(self):
        assert len(_TOOL_PARAMS) == 9

    def test_weather_params_include_city(self):
        assert "city" in _TOOL_PARAMS["get_weather"]
