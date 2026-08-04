# llm-eval-kit

> 大模型评估工具包 — RAGAS / Function Calling / Pipeline Trace / Needle-in-Haystack。
> 纯 Python 实现，无 GPU 依赖，拿来即用。

---

## 为什么值得看

市面上评估 RAG 应用的工具很多，但大多要么依赖 GPU 跑嵌入模型（RAGAS 官方），要么需要对接 SaaS 平台（LangSmith）。这套工具包在 **不安装 GPU、不注册第三方服务** 的前提下，提供了实用的评估和诊断能力：

- 想快速知道你搭的 RAG 应用的检索和生成质量？→ `ragas_eval`
- 想知道模型调用工具时填的参数对不对？→ `tool_calling_eval`
- 分低了想定位是检索的问题还是生成的问题？→ `pipeline_trace`
- 想测长文档里模型能不能找到关键信息？→ `needle_haystack`

## 模块概览

```
llm-eval-kit/
├── ragas_eval          RAGAS 本地实现 — 4 核心指标 + 报告生成
├── tool_calling_eval   Function Calling — 5 项指标 + 同义词表
├── pipeline_trace      流水线追踪 — 5 类自动诊断（类 LangSmith）
├── needle_haystack     长上下文测试 — 针海捞针
└── model_adapters      多模型对比 — 规则模型 / 错误倾向模型
```

## 快速开始

```bash
git clone https://github.com/zzluogh/llm-eval-kit.git
cd llm-eval-kit
PYTHONPATH=src python3 -m pytest tests/ -v
```

## 模块详解

### ragas_eval — RAG 质量评估

**回答 4 个问题：**

| 指标 | 问题 | 怎么算 |
|------|------|--------|
| faithfulness | 答案有没有瞎编？ | 按句拆分 → 逐句检查是否有上下文依据 → 有依据句数/总句数 |
| answer_relevancy | 回答跑题没有？ | 问题和回答的余弦相似度 |
| context_recall | 该找的资料找全了吗？ | 标准答案关键词在检索结果中的命中率 |
| context_precision | 找回来的有多少是废话？ | 检索结果中与问题相关的条数比例 |

**使用示例：**

```python
from llm_eval.ragas_eval import evaluate, RAGSample

samples = [
    RAGSample(
        question="What is Nav2?",
        contexts=["Nav2 is a navigation framework for ROS2 robots."],
        answer="Nav2 is a navigation framework for robot operating systems.",
        ground_truth="Nav2 is the ROS2 navigation stack providing path planning and control.",
    )
]
result = evaluate(samples)
print(result.report())

# Output:
# Sample     Faith    Relv cRecall   cPrec
# Q1         1.000   0.320   0.750   0.500
# AVG        1.000   0.320   0.750   0.500
```

### tool_calling_eval — 工具调用评测

**5 项指标衡量模型填参数对不对：**

| 指标 | 问题 |
|------|------|
| tool_name_accuracy | 工具选对了吗？ |
| param_precision | 给的参数里几个是对的？ |
| param_recall | 该给的参数给了几个？ |
| param_f1 | 精确率和召回率的平衡 |
| exact_match | 一字不差？ |

**支持同义词表：** model 调 `get_weather(city="peking")` 而你期望 `city="beijing"` → 配置同义词映射自动判为正确。

### pipeline_trace — 流水线诊断

**不给你一个分数，告诉你分数为什么低。**

| 检测器 | 检测什么 |
|--------|---------|
| hallucination | 答案中是否有无依据的句子 |
| empty_context | 检索是否返回了空结果 |
| irrelevant_context | 检索结果是否与问题完全无关 |
| incomplete_answer | 回答是否太简陋 |
| slow_step | 哪个步骤耗时过长 |

### needle_haystack — 长上下文测试

模拟"在一本 10 万字书里找一句话"的场景。随机生成草堆文本 → 在特定位置插入"针" → 测试模型能否在不同上下文长度和位置找到它。

### model_adapters — 多模型对比

统一的模型接口，内置两种模拟适配器：
- `rule_based`：确定性回答，用于框架验证
- `error_prone`：带错误倾向，模拟真实模型的常见缺陷

## 技术栈

| 组件 | 选型 |
|------|------|
| 测试框架 | pytest 9.0 |
| 报告 | 内置 ASCII 表格 + Allure 可选 |
| 依赖 | 仅 Python 标准库（`re` `math` `dataclasses`） |
| CI/CD | GitHub Actions |

## 项目测试

```bash
PYTHONPATH=src python3 -m pytest tests/ -v
```

当前覆盖：200+ 测试用例，覆盖 4 模块的全部指标函数 + 边界 + 聚合逻辑。
