import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from pdf_processor import extract_text, PDF_PATH
from chunker import create_chunks

model=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def build_vector_store(chunks):
    texts=[chunk["text"] for chunk in chunks]

    embeddings=model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    index=faiss.IndexFlatIP(embeddings.shape[1])
    index.add(
        np.asarray(embeddings,dtype=np.float32)
    )
    return index,embeddings

def search_chunks(question,chunks,index,top_k=3):
    question_embedding=model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores,indices=index.search(
        np.asarray(question_embedding,dtype=np.float32),
        top_k
    )
    results=[]

    for score,idx in zip(scores[0],indices[0]):
        if idx!=1:
            results.append({
                "page":chunks[idx]["page"],
                "text":chunks[idx]["text"],
                "score":float(score)
            })

    return results

if __name__ == "__main__":
    pages = extract_text(PDF_PATH)
    chunks = create_chunks(pages)
    index, embeddings = build_vector_store(chunks)

    question = "What is supervised learning?"
    results = search_chunks(question, chunks, index)

    for i, result in enumerate(results, start=1):
        print(f"\n--- Result {i} ---")
        print(f"Page: {result['page']}")
        print(f"Similarity: {result['score']:.3f}")
        print(f"Text: {result['text']}")
