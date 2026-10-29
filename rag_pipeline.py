import os
import json
from pathlib import Path

from urllib.request import Request, urlopen

from pdf_processor import extract_text, PDF_PATH
from chunker import create_chunks
from vector_store import build_vector_store, search_chunks

NOT_FOUND = "Sorry, I could not find an answer to your question in the provided PDF."
MIN_RETRIEVAL_SCORE = 0.35
MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"


def load_local_env():
    "Load local configuration from .env file if it exists."
    env_file = Path(".env.local")

    if not env_file.exists():
        return

    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()

        if not line or line.startswith("#"):
            continue

        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_local_env()
API_KEY = os.getenv("OPENROUTER_API_KEY")


def answer_question(question, chunks, index, top_k=5):
    """Retrieve relevant PDF chunks and generate a grounded answer."""

    results = search_chunks(question, chunks, index, top_k=3)

    print("\n--- Retrieved Context ---")
    for i, item in enumerate(results, start=1):
        print(f"\nResult {i}")
        print(f"Page: {item['page']}")
        print(f"Similarity: {item['score']:.3f}")
        print(f"Text: {item['text']}")

    if not results or results[0]["score"] < MIN_RETRIEVAL_SCORE:
        return {"answer": NOT_FOUND, "pages": []}

    context = "\n\n".join(f"[Page {item['page']}]\n{item['text']}" for item in results)

    system_prompt = f"""
    You are a question-answering assistant for a single PDF.
    Rules:
    1. Answer only using facts explicitly supported by the context.
    2. Do not use outside knowledge or invent missing information.
    3. If the context does not contain enough evidence, reply exactly:
    "{NOT_FOUND}"
    4. Give a concise answer.
    5. Do not treat similarity scores as confidence or correctness scores.
    """

    user_prompt = f"""
    PDF context:
    {context}

    Question: {question}
Answer the question using only the information provided in the PDF context above. If the context does not contain enough evidence to answer the question, reply exactly: "{NOT_FOUND}".
    """

    request = Request(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        data=json.dumps(
            {
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.2,
                "max_tokens": 400,
            }
        ).encode("utf-8"),
        method="POST",
    )

    with urlopen(request, timeout=60) as response:
        data = json.loads(response.read().decode("utf-8"))

    answer = data["choices"][0]["message"]["content"].strip()

    pages = list(dict.fromkeys(item["page"] for item in results))

    return {"answer": answer, "pages": pages}


if __name__ == "__main__":
    if not API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Please set it in your environment or .env.local file."
        )

    pages = extract_text(PDF_PATH)
    chunks = create_chunks(pages)
    index, embeddings = build_vector_store(chunks)

    questions = [
        "What is supervised learning?",
        "What is the difference between supervised and unsupervised learning?",
        "Can you explain the concept of overfitting in machine learning?",
    ]

    for question in questions:
        result = answer_question(question, chunks, index)
        print(f"Question: {question}")
        print(f"Answer: {result['answer']}")
        print(f"Source Pages: {result['pages']}")
        print("-" * 50)
