import os
import json
from pathlib import Path

from urllib.request import Request, urlopen

from pdf_processor import extract_all_pdfs
from chunker import create_chunks
from vector_store import (
    build_vector_store,
    search_chunks,
    load_vector_store,
    save_vector_store,
)


# My fallback answer when the PDFs don't contain enough information
NOT_FOUND = "Sorry, I could not find an answer to your question in the provided PDF."

# Minimum similarity score required before sending chunks to the LLM
MIN_RETRIEVAL_SCORE = 0.35

# Free model I'm currently using through OpenRouter (configurable via env)
DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"


def load_local_env():
    # Read API key from .env or .env.local
    base_dir = Path(__file__).resolve().parent
    for env_name in (".env", ".env.local"):
        env_file = base_dir / env_name
        if not env_file.exists():
            continue

        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()

            # Skip blank lines and comments
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            os.environ.setdefault(
                key.strip(),
                value.strip().strip('"').strip("'")
            )


load_local_env()
API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)


def answer_question(question, chunks, index, top_k=5):
    # First, find the chunks that look most relevant to the question
    results = search_chunks(question, chunks, index, top_k=top_k)

    print("\n--- Retrieved Context ---")

    for i, item in enumerate(results, start=1):
        print(f"\nResult {i}")
        print(f"Source: {item['source']}")
        print(f"Page: {item['page']}")
        print(f"Similarity: {item['score']:.3f}")
        print(f"Text: {item['text']}")

    # No useful results? Don't ask the LLM to guess
    if not results or results[0]["score"] < MIN_RETRIEVAL_SCORE:
        return {"answer": NOT_FOUND, "sources": []}

    # Keep the filename and page number with every retrieved chunk
    context = "\n\n".join(
        f"[Source: {item['source']} | Page: {item['page']}]\n"
        f"{item['text']}"
        for item in results
    )

    # These rules tell the model to stick to the PDF content
    system_prompt = f"""
You are a question-answering assistant for information contained in PDFs.

Follow these rules:
1. Answer only using information supported by the provided context.
2. Do not use outside knowledge or make up missing information.
3. If the context does not contain enough evidence, reply exactly:
"{NOT_FOUND}"
4. Keep the answer concise.
5. Do not treat similarity scores as confidence or correctness scores.
"""

    user_prompt = f"""
PDF context:
{context}

Question: {question}

Answer using only the PDF context above.
If the context does not contain enough information, reply exactly:
"{NOT_FOUND}"
"""

    # Send the question and retrieved PDF text to OpenRouter
    request = Request(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        data=json.dumps({
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 1000,
        }).encode("utf-8"),
        method="POST",
    )

    with urlopen(request, timeout=60) as response:
        data = json.loads(response.read().decode("utf-8"))

    choices = data.get("choices", [])
    if not choices:
        return {"answer": NOT_FOUND, "sources": []}

    answer = choices[0].get("message", {}).get("content", "")
    if not answer:
        return {"answer": NOT_FOUND, "sources": []}

    answer = answer.strip()

    # If the model couldn't answer, don't show citations as if they support an answer
    if answer == NOT_FOUND:
        return {"answer": NOT_FOUND, "sources": []}

    # Keep each filename-page combination only once
    sources = []
    seen = set()

    for item in results:
        key = (item["source"], item["page"])

        if key not in seen:
            seen.add(key)
            sources.append({
                "source": item["source"],
                "page": item["page"]
            })

    return {
        "answer": answer,
        "sources": sources
    }


if __name__ == "__main__":
    # Make sure I have added my OpenRouter key before running the pipeline
    if not API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is missing. Add it to your .env or .env.local file."
        )

    # Load existing vector store if available, or build and save one
    index, chunks = load_vector_store()

    if index is None or chunks is None:
        pages = extract_all_pdfs()
        chunks = create_chunks(pages)

        print(f"\nTotal PDF pages: {len(pages)}")
        print(f"Total chunks: {len(chunks)}")

        # Convert the chunks into embeddings and create the searchable index
        index, embeddings = build_vector_store(chunks)
        save_vector_store(index, chunks)
        print("Vector store created and saved.")
    else:
        print("Existing vector store loaded successfully.")

    print(f"Total indexed chunks: {index.ntotal}")

    # A few questions to test whether retrieval and answering work properly
    questions = [
        "What is supervised learning?",
        "What is the difference between supervised and unsupervised learning?",
        "Can you explain the concept of overfitting in machine learning?",
    ]

    for question in questions:
        result = answer_question(question, chunks, index)

        print(f"\nQuestion: {question}")
        print(f"Answer: {result['answer']}")
        print("Sources:")

        if result["sources"]:
            for source in result["sources"]:
                print(f"- {source['source']} | Page {source['page']}")
        else:
            print("- No supporting source found.")

        print("-" * 60)