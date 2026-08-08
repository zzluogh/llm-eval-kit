"""RAG 自动化评测脚本 —— 得到大脑检索 + DeepSeek 生成 + ragas_eval 打分。

数据流：
  test_queries.json → 得到大脑 recall API → contexts
                    → DeepSeek API → answer
                    → ragas_eval → 4指标基线

用法:
  PYTHONPATH=src python3 tests/run_rag_eval.py
"""
import json
import os
import sys
import time
import requests
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_eval.ragas_eval import evaluate, RAGSample

# 环境变量（运行前设置）:
#   export DDND_API_KEY="gk_live_xxx"
#   export DDND_CLIENT_ID="cli_xxx"
#   export DEEPSEEK_API_KEY="sk-xxx"

DDND_API_KEY = os.environ.get("DDND_API_KEY", "")
DDND_CLIENT_ID = os.environ.get("DDND_CLIENT_ID", "")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DDND_TOPIC_ID = "YM9DBm2Y"
DDND_RECALL_URL = "https://openapi.biji.com/open/api/v1/resource/recall/knowledge"
DEEPSEEK_CHAT_URL = "https://api.deepseek.com/chat/completions"


def recall_from_ddnd(query: str, top_k: int = 5):
    """调得到大脑检索接口，返回 contexts 列表。"""
    headers = {
        "Authorization": DDND_API_KEY,
        "X-Client-ID": DDND_CLIENT_ID,
        "Content-Type": "application/json",
    }
    body = {"topic_id": DDND_TOPIC_ID, "query": query, "top_k": top_k}
    resp = requests.post(DDND_RECALL_URL, json=body, headers=headers, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    results = data.get("data", {}).get("results", [])
    contexts = [r["content"] for r in results if r.get("content")]
    return contexts


def generate_answer(query: str, contexts: list[str]) -> str:
    """调 DeepSeek API 基于检索上下文生成回答。"""
    ctx_text = "\n\n".join(contexts)
    prompt = (
        f"根据以下参考资料回答用户问题。如果资料中没有相关信息，请如实说你不知道，不要编造。\n\n"
        f"参考资料：\n{ctx_text}\n\n"
        f"用户问题：{query}\n\n"
        f"回答："
    )
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 0.0,
    }
    resp = requests.post(DEEPSEEK_CHAT_URL, json=body, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def main():
    queries_path = Path(__file__).resolve().parents[1] / "data" / "test_queries.json"
    with open(queries_path) as f:
        queries = json.load(f)

    samples = []
    total = len(queries)

    print(f"开始评测：{total} 条查询")
    print(f"检索：得到大脑  topic_id={DDND_TOPIC_ID}")
    print(f"生成：DeepSeek  deepseek-chat")
    print("-" * 60)

    for i, q in enumerate(queries):
        question = q["input"]
        ground_truth = q["expected_output"]

        print(f"[{i+1}/{total}] {question[:50]}...")

        try:
            contexts = recall_from_ddnd(question, top_k=5)
        except Exception as e:
            print(f"  ❌ 检索失败: {e}")
            contexts = []

        try:
            answer = generate_answer(question, contexts) if contexts else "无法检索到相关信息。"
        except Exception as e:
            print(f"  ❌ 生成失败: {e}")
            answer = f"生成出错: {e}"

        sample = RAGSample(
            question=question,
            contexts=contexts,
            answer=answer,
            ground_truth=ground_truth,
        )
        samples.append(sample)
        print(f"  contexts={len(contexts)}条  answer={len(answer)}字")
        time.sleep(0.5)

    print("-" * 60)
    result = evaluate(samples)

    scores_per_row = result.per_sample
    n = len(scores_per_row)

    faithfulness_scores = [s.faithfulness for s in scores_per_row]
    relevancy_scores = [s.answer_relevancy for s in scores_per_row]
    recall_scores = [s.context_recall for s in scores_per_row]
    precision_scores = [s.context_precision for s in scores_per_row]

    hallucination_count = sum(1 for f in faithfulness_scores if f < 0.5)

    print("\n原始数据（逐条）:")
    print(result.report())

    print("\n" + "=" * 60)
    print(" 6.1 节基线数据")
    print("=" * 60)
    print(f"  faithfulness (平均):       {result.avg.faithfulness:.3f}")
    print(f"  answer_relevancy (平均):   {result.avg.answer_relevancy:.3f}")
    print(f"  context_recall (平均):     {result.avg.context_recall:.3f}")
    print(f"  context_precision (平均):  {result.avg.context_precision:.3f}")
    print(f"  幻觉率 (<0.5 faithfulness): {hallucination_count}/{n} = {hallucination_count/n:.0%}")
    print(f"  总样本数:                   {n}")


if __name__ == "__main__":
    main()
