import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from pdf_processor import extract_all_pdfs
from chunker import create_chunks


# This model converts text into numbers so we can compare its meaning
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def build_vector_store(chunks):
    # Take the text from every chunk
    texts = [chunk["text"] for chunk in chunks]

    # Convert all chunks into embeddings
    # Normalization helps us compare their similarity using dot product
    embeddings = model.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    # FAISS expects embeddings in this format
    embeddings = np.asarray(embeddings, dtype=np.float32)

    # Create an index to search for similar chunks quickly
    # IndexFlatIP uses the inner product to compare embeddings
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index, embeddings


def search_chunks(question, chunks, index, top_k=5):
    # Convert the user's question into an embedding too
    question_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # Don't request more results than the index actually contains
    top_k = min(top_k, index.ntotal)

    # Find the top matching chunks for this question
    scores, indices = index.search(
        np.asarray(question_embedding, dtype=np.float32),
        top_k
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):
        # FAISS returns -1 when there is no valid result at that position
        if idx != -1:
            results.append({
                "source": chunks[idx]["source"],
                "page": chunks[idx]["page"],
                "text": chunks[idx]["text"],
                "score": float(score)
            })

    return results


if __name__ == "__main__":
    # Read all PDFs and split their text into smaller chunks
    pages = extract_all_pdfs()
    chunks = create_chunks(pages)

    print(f"Total PDF pages: {len(pages)}")
    print(f"Total chunks: {len(chunks)}")

    # Turn the chunks into embeddings and build the FAISS index
    index, embeddings = build_vector_store(chunks)

    print(f"Total indexed chunks: {index.ntotal}")
    print(f"Embedding dimension: {embeddings.shape[1]}")

    # Try a question to see which chunks the system finds
    question = "What is the difference between supervised and unsupervised learning?"
    results = search_chunks(question, chunks, index)

    print(f"\nQuestion: {question}")

    # Display the matching chunks with their scores and PDF locations
    for i, result in enumerate(results, start=1):
        print(f"\n--- Result {i} ---")
        print(f"Source: {result['source']}")
        print(f"Page: {result['page']}")
        print(f"Similarity: {result['score']:.3f}")
        print(f"Text: {result['text']}")