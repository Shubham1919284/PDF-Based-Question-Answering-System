
from pdf_processor import extract_text, PDF_PATH


def create_chunks(pages, chunk_size=100, overlap=20):
    chunks = []

    for page in pages:
        words = page["text"].split()
        start = 0

        while start < len(words):
            end = start + chunk_size
            chunk_text = " ".join(words[start:end]).strip()

            if chunk_text:
                chunks.append({
                    "page": page["page"],
                    "text": chunk_text
                })

            start += chunk_size - overlap

    return chunks


if __name__ == "__main__":
    pages = extract_text(PDF_PATH)
    chunks = create_chunks(pages)

    print("Total chunks:", len(chunks))

    for i, chunk in enumerate(chunks[:3], start=1):
        print(f"\n--- Chunk {i} | Page {chunk['page']} ---")
        print(chunk["text"])
