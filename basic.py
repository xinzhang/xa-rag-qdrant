import os

import requests
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

dummy_data = [
    "My name is Max",
    "Max likes to eat pizza",
    "Max likes to play basketball",
    "Max likes to play football",
    "My name is Manuel",
    "Manuel likes to play chess",
    "Manuel likes to play soccer",
    "Manuel likes to play tennis",
    "Manuel likes to play golf",
    "Manuel likes to play hockey",
    "Manuel likes to play volleyball",
]

client = QdrantClient(url="http://localhost:6333")
if not client.collection_exists(collection_name='demo'):
    client.create_collection(
        collection_name="demo",
        vectors_config=VectorParams(size=1024, distance=Distance.DOT),
    )

def ingest():
    for idx, text in enumerate(dummy_data):
        response = requests.post(
            "http://localhost:11434/api/embed",
            json={"model": "mxbai-embed-large:v1", "input": text},
        )
        data = response.json()
        embeddings = data["embeddings"][0]
        client.upsert(
            collection_name="demo",
            points=[PointStruct(id=idx, vector=embeddings, payload={"text": text})],
        )
        print(f"upserted [{idx}] {text!r}")


def query():
    prompt = input("Enter a prompt: ")
    adjusted_prompt = f"Represent this sentence for searching relevant passages: {prompt}"

    response = requests.post(
        "http://localhost:11434/api/embed",
        json={"model": "mxbai-embed-large:v1", "input": adjusted_prompt},
    )
    data = response.json()
    embeddings = data["embeddings"][0]

    results = client.query_points(
        collection_name="demo",
        query=embeddings,
        limit=5,
        with_payload=True,
    ).points

    for result in results:
        print(f"{result.score:.4f} {result.payload}")

    context = "\n".join(result.payload["text"] for result in results)
    answer(prompt, context)


def answer(question, context):
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Answer the question using only the provided context. "
                "If the context doesn't contain the answer, say you don't know.",
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ],
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    query()
