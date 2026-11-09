import faiss
import numpy as np
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer

from pdf_processor import extract_all_pdfs
from chunker import create_chunks


# This model converts text into numbers so we can compare its meaning
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# Save my FAISS index and PDF chunks so I can reuse them later
def save_vector_store(index, chunks, folder="vector_db"):
    folder = Path(folder)
    if not folder.is_absolute():
        folder = Path(__file__).resolve().parent / folder
    folder.mkdir(parents=True, exist_ok=True)

    # Save the FAISS index
    faiss.write_index(index, str(folder / "faiss.index"))

    # Save chunks with their PDF names and page numbers
    with open(folder / "chunks.pkl", "wb") as file:
        pickle.dump(chunks, file)


# Load the previously saved index and chunks
def load_vector_store(folder="vector_db"):
    folder = Path(folder)
    if not folder.is_absolute():
        folder = Path(__file__).resolve().parent / folder

    index_path = folder / "faiss.index"
    chunks_path = folder / "chunks.pkl"

    if not index_path.exists() or not chunks_path.exists():
        return None, None

    # Load the saved FAISS index
    index = faiss.read_index(str(index_path))

    # Load the saved PDF chunks
    with open(chunks_path, "rb") as file:
        chunks = pickle.load(file)

    return index, chunks


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

    # Retrieve extra candidates so we can remove duplicate pages
    # without ending up with fewer than top_k results
    candidate_k = min(top_k * 3, index.ntotal)

    # Find the most similar chunks using FAISS
    scores, indices = index.search(
        np.asarray(question_embedding, dtype=np.float32),
        candidate_k
    )

    results = []
    seen_pages = set()

    for score, idx in zip(scores[0], indices[0]):
        # FAISS returns -1 when there is no valid result at that position
        if idx == -1:
            continue

        # Get the original chunk using its FAISS index
        chunk = chunks[idx]

        # Identify each PDF page using its filename and page number
        page_key = (chunk["source"], chunk["page"])

        # Skip another chunk if this PDF page is already included
        if page_key in seen_pages:
            continue

        seen_pages.add(page_key)

        # Store the retrieved chunk along with its similarity score
        results.append({
            "source": chunk["source"],
            "page": chunk["page"],
            "text": chunk["text"],
            "score": float(score)
        })

        # Stop once we have collected the requested number of unique pages
        if len(results) == top_k:
            break

    return results



if __name__ == "__main__":
    # Check whether a saved vector store exists
    index, chunks = load_vector_store()

    if index is None or chunks is None:
        # Read all PDFs and split their text into smaller chunks
        pages = extract_all_pdfs()
        chunks = create_chunks(pages)

        print(f"Total PDF pages: {len(pages)}")
        print(f"Total chunks: {len(chunks)}")

        # Turn the chunks into embeddings and build the FAISS index
        index, embeddings = build_vector_store(chunks)
        save_vector_store(index, chunks)
        print("Vector store created and saved.")
    else:
        print("Existing vector store loaded successfully.")

    print(f"Total indexed chunks: {index.ntotal}")

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