import os
from pathlib import path

import requests

from pdf_processor import extract_text,PDF_PATH
from chunker import create_chunks
from vector_store import build_vector_store,search_chunks

NOT_FOUND="Sorry, I could not find an answer to your question in the provided PDF."
MIN_RETRIEVAL_SCORE=0.35
MODEL="openrouter/free"

def load_local_env():
    "Load local configuration from .env file if it exists."
    env_file=path(".env.local")

    if not env_file.exists():
        return

    for line in env_file.read_text(encoding="utf-8").splitlines():
        line=line.strip()


        if not line or line.startswith("#"):
            continue

        key,value=line.split("=",1)
        os.environ.setdefault(
            key.strip(),
            value.strip().strip('"').strip("'")
        )

load_local_env()
API_KEY=os.getenv("OPENROUTER_API_KEY")

def answer_question(question,chunks,index,top_k=3):
    """Retrieve relevant PDF chunks and generate a grounded answer."""

    results=search_chunks(question,chunks,index,top_k=3)

    if not results or results[0]["score"]<MIN_RETRIEVAL_SCORE:
        return{
            "answer":NOT_FOUND,
            "pages":[]
        }

    context="\n\n".join(
        f"[Page {item['page']}]\n{item['text']}"
        for item in results
    )

    system_prompt= f"""
    You are a question-answering assistant for a single PDF.
    Rules:
    1. Answer only using facts explicitly supported by the context.
    2. Do not use outside knowledge or invent missing information.
    3. If the context does not contain enough evidence, reply exactly:
    "{NOT_FOUND}"
    4. Give a concise answer.
    5. Do not treat similarity scores as confidence or correctness scores.
    """

    user_prompt=f"""
    PDF context:
    {context}

    Question: {question}
Answer the question using only the information provided in the PDF context above. If the context does not contain enough evidence to answer the question, reply exactly: "{NOT_FOUND}".
    """

    response=requests.post