"""Function Calling 评测 — 20 组测试样本。(第29课, 第31课引用)

覆盖 9 种场景：完美匹配 / 工具选错 / 参数缺少 / 参数多余 /
参数值错误 / 部分正确 / 大小写容忍 / 空响应 / 可选参数。
"""
from my_math.tool_calling_eval import ToolDef, ToolCallSample

# ---------------------------------------------------------------------------
# 工具定义（9 个行业常见工具）
# ---------------------------------------------------------------------------

TOOLS: list[ToolDef] = [
    ToolDef("get_weather",       "Get weather for a city",           {"city": "str", "units": "str"}),
    ToolDef("set_timer",         "Set a countdown timer",             {"duration_minutes": "int", "label": "str"}),
    ToolDef("send_email",        "Send an email",                     {"to": "str", "subject": "str", "body": "str"}),
    ToolDef("calculate",         "Evaluate a math expression",        {"expression": "str"}),
    ToolDef("search",            "Search the web",                    {"query": "str", "max_results": "int"}),
    ToolDef("translate",         "Translate text to a language",      {"text": "str", "target_language": "str"}),
    ToolDef("get_stock_price",   "Get real-time stock price",         {"symbol": "str"}),
    ToolDef("play_music",        "Play a song by name",               {"song_name": "str", "volume": "int"}),
    ToolDef("create_reminder",   "Create a reminder",                 {"text": "str", "datetime": "str"}),
]

# ---------------------------------------------------------------------------
# 20 组样本 — 覆盖 9 种场景
# ---------------------------------------------------------------------------

SAMPLES: list[ToolCallSample] = [
    # ── 场景 1: 完美匹配 ──
    ToolCallSample(
        query="What's the weather in Beijing?",
        tools_available=[TOOLS[0], TOOLS[4], TOOLS[7]],
        expected_tool="get_weather",
        expected_params={"city": "Beijing"},
        predicted_tool="get_weather",
        predicted_params={"city": "Beijing"},
    ),
    ToolCallSample(
        query="Set a timer for 15 minutes please",
        tools_available=[TOOLS[1], TOOLS[3], TOOLS[5]],
        expected_tool="set_timer",
        expected_params={"duration_minutes": 15},
        predicted_tool="set_timer",
        predicted_params={"duration_minutes": 15},
    ),
    ToolCallSample(
        query="Send an email to bob@example.com with subject Hello",
        tools_available=[TOOLS[2], TOOLS[6], TOOLS[8]],
        expected_tool="send_email",
        expected_params={"to": "bob@example.com", "subject": "Hello", "body": ""},
        predicted_tool="send_email",
        predicted_params={"to": "bob@example.com", "subject": "Hello", "body": ""},
    ),

    # ── 场景 2: 工具选错 ──
    ToolCallSample(
        query="What's the weather in Tokyo?",
        tools_available=[TOOLS[0], TOOLS[1], TOOLS[4]],
        expected_tool="get_weather",
        expected_params={"city": "Tokyo"},
        predicted_tool="search",  # 模型误选了 search
        predicted_params={"query": "weather in Tokyo"},
    ),

    # ── 场景 3: 正确工具 + 参数值错误 ──
    ToolCallSample(
        query="Set a timer for 5 minutes",
        tools_available=[TOOLS[1], TOOLS[3]],
        expected_tool="set_timer",
        expected_params={"duration_minutes": 5},
        predicted_tool="set_timer",
        predicted_params={"duration_minutes": 10},  # 模型听错了
    ),

    # ── 场景 4: 正确工具 + 缺少参数 ──
    ToolCallSample(
        query="Translate 'hello' to French",
        tools_available=[TOOLS[5], TOOLS[0], TOOLS[4]],
        expected_tool="translate",
        expected_params={"text": "hello", "target_language": "French"},
        predicted_tool="translate",
        predicted_params={"text": "hello"},  # 漏了 target_language
    ),

    # ── 场景 5: 正确工具 + 多余参数（幻觉） ──
    ToolCallSample(
        query="How much is Apple stock?",
        tools_available=[TOOLS[6], TOOLS[4], TOOLS[1]],
        expected_tool="get_stock_price",
        expected_params={"symbol": "AAPL"},
        predicted_tool="get_stock_price",
        predicted_params={"symbol": "AAPL", "currency": "USD"},  # 模型多给了 currency
    ),

    # ── 场景 6: 部分正确（关键参数对、次要参数错） ──
    ToolCallSample(
        query="Search for ROS2 tutorials",
        tools_available=[TOOLS[4], TOOLS[0], TOOLS[5]],
        expected_tool="search",
        expected_params={"query": "ROS2 tutorials", "max_results": 10},
        predicted_tool="search",
        predicted_params={"query": "ROS2 tutorials", "max_results": 5},  # query 对，max_results 错
    ),

    # ── 场景 7: 空响应（模型没调用任何工具） ──
    ToolCallSample(
        query="Calculate 2+3",
        tools_available=[TOOLS[3], TOOLS[1]],
        expected_tool="calculate",
        expected_params={"expression": "2+3"},
        predicted_tool="",  # 模型没返回工具调用
        predicted_params={},
    ),

    # ── 场景 8: 大小写容忍 ──
    ToolCallSample(
        query="What's the stock price of TSLA?",
        tools_available=[TOOLS[6], TOOLS[4]],
        expected_tool="get_stock_price",
        expected_params={"symbol": "TSLA"},
        predicted_tool="get_stock_price",
        predicted_params={"symbol": "tsla"},  # 模型返回小写, 应匹配
    ),

    # ── 场景 9: 工具名大小写容忍 ──
    ToolCallSample(
        query="Play some Beatles songs",
        tools_available=[TOOLS[7], TOOLS[4]],
        expected_tool="play_music",
        expected_params={"song_name": "Beatles"},
        predicted_tool="PLAY_MUSIC",  # 工具名大小写不同, 应匹配
        predicted_params={"song_name": "Beatles"},
    ),

    # ── 场景 10: 多参数完美匹配 ──
    ToolCallSample(
        query="Remind me to call mom at 2026-06-16 10:00",
        tools_available=[TOOLS[8], TOOLS[1], TOOLS[2]],
        expected_tool="create_reminder",
        expected_params={"text": "call mom", "datetime": "2026-06-16 10:00"},
        predicted_tool="create_reminder",
        predicted_params={"text": "call mom", "datetime": "2026-06-16 10:00"},
    ),

    # ── 场景 11: 选择无关工具 + 参数全错 ──
    ToolCallSample(
        query="What is 100 divided by 7?",
        tools_available=[TOOLS[3], TOOLS[6], TOOLS[8]],
        expected_tool="calculate",
        expected_params={"expression": "100/7"},
        predicted_tool="create_reminder",  # 完全选错
        predicted_params={"text": "100 divided by 7", "datetime": "today"},
    ),

    # ── 场景 12: 正确工具 + 参数空白 ──
    ToolCallSample(
        query="Get the latest news",
        tools_available=[TOOLS[4], TOOLS[0], TOOLS[7]],
        expected_tool="search",
        expected_params={"query": "latest news"},
        predicted_tool="search",
        predicted_params={},  # 选了正确工具但没给参数
    ),

    # ── 期望参数为空 — 简单问候不需要工具 ──
    ToolCallSample(
        query="Hello, how are you?",
        tools_available=[TOOLS[4], TOOLS[0]],
        expected_tool="",  # 不应调用任何工具
        expected_params={},
        predicted_tool="",
        predicted_params={},
    ),

    # ── 场景 13: 数值参数精确匹配 ──
    ToolCallSample(
        query="Play Despacito at volume 50",
        tools_available=[TOOLS[7], TOOLS[4], TOOLS[1]],
        expected_tool="play_music",
        expected_params={"song_name": "Despacito", "volume": 50},
        predicted_tool="play_music",
        predicted_params={"song_name": "Despacito", "volume": 50},
    ),

    # ── 场景 14: 工具选对但一半参数错 ──
    ToolCallSample(
        query="Send an email to alice@test.com about meeting at 3pm",
        tools_available=[TOOLS[2], TOOLS[6], TOOLS[8]],
        expected_tool="send_email",
        expected_params={"to": "alice@test.com", "subject": "meeting at 3pm", "body": ""},
        predicted_tool="send_email",
        predicted_params={"to": "alice@test.com", "subject": "lunch at noon", "body": ""},
    ),

    # ── 场景 15: 中英文混合 query 下的工具选择 ──
    ToolCallSample(
        query="帮我查一下北京天气",
        tools_available=[TOOLS[0], TOOLS[4], TOOLS[5]],
        expected_tool="get_weather",
        expected_params={"city": "Beijing"},
        predicted_tool="get_weather",
        predicted_params={"city": "Beijing"},
    ),

    # ── 场景 16: 正确工具 + 可选参数不要求 ──
    ToolCallSample(
        query="Timer for 3 minutes",
        tools_available=[TOOLS[1], TOOLS[3]],
        expected_tool="set_timer",
        expected_params={"duration_minutes": 3},
        predicted_tool="set_timer",
        predicted_params={"duration_minutes": 3, "label": "timer"},  # 模型多给了 label（可选参数）
    ),

    # ── 场景 17: 工具选对但参数键名不对（内容正确） ──
    ToolCallSample(
        query="Find me some Python tutorials",
        tools_available=[TOOLS[4], TOOLS[5], TOOLS[7]],
        expected_tool="search",
        expected_params={"query": "Python tutorials"},
        predicted_tool="search",
        predicted_params={"search_term": "Python tutorials"},  # 键名错了
    ),
]

# ---------------------------------------------------------------------------
# 同义词表 — 参数值归一化
# ---------------------------------------------------------------------------

PARAM_SYNONYMS: dict[str, dict[str, str]] = {
    "city": {
        "peking": "beijing",
        "nyc": "new york",
        "sf": "san francisco",
    },
    "target_language": {
        "chinese": "zh",
        "japanese": "ja",
        "french": "fr",
    },
    "units": {
        "c": "celsius",
        "f": "fahrenheit",
    },
}
