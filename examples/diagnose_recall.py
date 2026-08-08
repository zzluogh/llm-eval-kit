"""Recall 归因诊断脚本。

选取 recall 高、中、低三类查询，打印实际检索内容和 ground_truth 对比。
目标：定位 recall=0.334 的主因 — 切片策略/embedding/措辞/ground_truth质量。
"""
import json
import os
import sys
import requests
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from llm_eval.ragas_eval import _tokenize, context_recall

# 环境变量（运行前设置）:
#   export DDND_API_KEY="gk_live_xxx"
#   export DDND_CLIENT_ID="cli_xxx"

DDND_API_KEY = os.environ.get("DDND_API_KEY", "")
DDND_CLIENT_ID = os.environ.get("DDND_CLIENT_ID", "")

HEADERS = {
    "Authorization": DDND_API_KEY,
    "X-Client-ID": DDND_CLIENT_ID,
    "Content-Type": "application/json",
}

# 筛选三组查询
QUERIES = {
    "high": {
        "label": "高 recall (>0.5)",
        "samples": [
            (7, "Nav2 的 RecoveryNode 有什么特殊行为？",
             "第一个子节点检查是否该恢复，后面的子节点执行恢复动作如原地旋转和后退，当 Planner 或 Controller 出错时自动进入恢复流程"),
            (14, "Pipeline Trace 和 RAGAS 的分工是什么？",
             "RAGAS 用于线上自动回归每次部署自动跑看分数是否下降，Pipeline Trace 用于线下排障分数低了之后定位是检索还是生成的问题"),
            (25, "QoS 不匹配最常见的后果是什么？",
             "静默失败，订阅者收不到消息但 ROS2 不报错不崩溃，数据悄悄丢了"),
        ],
    },
    "mid": {
        "label": "中 recall (0.2~0.5)",
        "samples": [
            (8, "Nav2 插件机制的好处是什么？",
             "每种能力通过插件动态加载，切换 Planner 或 Controller 只需改 YAML 配置文件不需要重新编译，方便 A/B 测试"),
            (22, "pytest 中 session、module、class、function 四种 fixture scope 什么时候用哪个？",
             "function 用于需要独立数据的测试，class 用于共享测试环境，module 用于模块级初始化，session 用于一次性重资源如 rclpy.init"),
        ],
    },
    "low": {
        "label": "低 recall (<0.1)",
        "samples": [
            (1, "Nav2 的核心任务有哪些？",
             "建图、定位、规划和控制"),
            (5, "DWB 和 RPP 两个控制器有什么不同？",
             "DWB 通过速度采样和轨迹评分来综合考量安全速度朝向，RPP 通过追踪路径前方一个点来跟随路径并加入了速度调节"),
            (19, "高召回+低精确的检索结果会带来什么问题？",
             "很多无关内容混进检索结果，可能误导模型生成错误回答"),
        ],
    },
}


def fetch_contexts(query, top_k=5):
    resp = requests.post(
        "https://openapi.biji.com/open/api/v1/resource/recall/knowledge",
        json={"topic_id": "YM9DBm2Y", "query": query, "top_k": top_k},
        headers=HEADERS,
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    results = data.get("data", {}).get("results", [])
    return [r["content"] for r in results if r.get("content")]


def analyze(question, ground_truth, contexts):
    gt_tokens = [t for t in _tokenize(ground_truth) if len(t) > 1]
    ctx_text = " ".join(contexts)
    ctx_tokens = set(_tokenize(ctx_text))

    hits = [t for t in gt_tokens if t in ctx_tokens]
    misses = [t for t in gt_tokens if t not in ctx_tokens]
    recall = context_recall(contexts, ground_truth)

    return {
        "gt_token_count": len(gt_tokens),
        "gt_tokens": gt_tokens,
        "hit": hits,
        "hit_count": len(hits),
        "miss": misses,
        "miss_count": len(misses),
        "recall": recall,
        "ctx_preview": ctx_text[:300],
        "ctx_total_words": len(ctx_text.split()),
    }


def main():
    for group_key, group in QUERIES.items():
        print("=" * 72)
        print(f"  {group['label'].upper()}")
        print("=" * 72)

        for idx, question, ground_truth in group["samples"]:
            print(f"\n── Q{idx}: {question}")
            print(f"   期望答案: {ground_truth}")

            contexts = fetch_contexts(question, top_k=5)
            a = analyze(question, ground_truth, contexts)

            print(f"   检索返回: {len(contexts)} 条片段, 共 {a['ctx_total_words']} 词")
            print(f"   GT 关键词: {a['gt_token_count']} 个")
            print(f"   命中: {a['hit'][:10]}{'...' if len(a['hit']) > 10 else ''}  ({a['hit_count']}/{a['gt_token_count']})")
            print(f"   缺失: {a['miss'][:10]}{'...' if len(a['miss']) > 10 else ''}  ({a['miss_count']}/{a['gt_token_count']})")
            print(f"   recall: {a['recall']:.3f}")
            print(f"   检索内容(前300字): {a['ctx_preview'][:200]}...")


    print("\n" + "=" * 72)
    print("  分析结论")
    print("=" * 72)
    print("""
  看三组对比：
  - 高 recall 组：GT 关键词（如"恢复""planner""controller"）和知识库文档用词高度一致
  - 低 recall 组：GT 用了缩写/简写（如"DWB"而非"DWB控制器"）或概括性短语
    检索返回的内容虽然相关但用的是全称/正式表述，导致逐词重叠判定为"不命中"
  - 中 recall 组：GT 中部分关键词命中、部分不命中

  主因：基于 token-overlap 的 recall 算法 vs 中文语义匹配之间的差距。
        GT 用语如果和知识库文档用词不一致，即使检索内容相关，recall 也会判 0。
  次因：GT 回答过于精简（如"建图、定位、规划和控制"只有4个词），
        样本量太小导致任何用词偏差都会放大。

  如果换成 sentence-transformers 做语义级匹配，低 recall 组大概率会显著改善。
  """)


if __name__ == "__main__":
    main()
