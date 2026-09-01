"""
Demo: LLM-based rerank, as opposed to calculation-based rerank
(cosine similarity + keyword boost, BM25/EnsembleRetriever, etc.)

Instead of computing a numeric score (embedding similarity, substring match,
term frequency...), we hand the candidates to an LLM and ask IT to judge
relevance and return a ranked order. The "score" is now a model judgment,
not arithmetic.
"""

import json
import os

import requests
from openai import OpenAI
from qdrant_client import QdrantClient

openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
client = QdrantClient(url="http://localhost:6333")


def llm_rerank(query: str, candidates: list[dict], model: str = "gpt-4o-mini") -> list[dict]:
    """
    candidates: list of {"id": ..., "text": ...}
    Returns candidates reordered by LLM-judged relevance to `query`
    (most relevant first). No embeddings or score math involved here —
    the LLM reads the passages itself and decides the order.
    """
    numbered = "\n".join(f"[{i}] {c['text']}" for i, c in enumerate(candidates))

    prompt = f"""You are a search relevance judge. Rank the passages below from
MOST to LEAST relevant to the query.

Query: {query}

Passages:
{numbered}

Respond with ONLY a JSON array of the passage numbers in ranked order,
most relevant first. Example: [2, 0, 1]"""

    response = openai_client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )

    order = json.loads(response.choices[0].message.content)
    return [candidates[i] for i in order]


def embed(text: str) -> list[float]:
    response = requests.post(
        "http://localhost:11434/api/embed",
        json={"model": "mxbai-embed-large:v1", "input": text},
    )
    return response.json()["embeddings"][0]


def demo():
    query = input("Enter a prompt: ")

    # 1. cheap vector prefilter, same as usual (calculation-based)
    vector_results = client.query_points(
        collection_name="demo",
        query=embed(query),
        limit=5,
        with_payload=True,
    ).points

    print("\n--- vector-similarity order (cosine score) ---")
    for r in vector_results:
        print(f"{r.score:.4f} {r.payload['text']!r}")

    # 2. LLM rerank on top of that shortlist (judgment-based)
    candidates = [{"id": r.id, "text": r.payload["text"]} for r in vector_results]
    reranked = llm_rerank(query, candidates)

    print("\n--- llm_rerank order (model judgment) ---")
    for rank, c in enumerate(reranked, 1):
        print(f"{rank}. {c['text']!r}")


if __name__ == "__main__":
    demo()
