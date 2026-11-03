from pdf_processor import extract_all_pdfs

def create_chunks(pages, chunk_size=100, overlap=20):
    # Split each page into smaller pieces so searching becomes easier
    chunks = []

    for page in pages:
        # Break the page text into individual words
        words = page["text"].split()
        start = 0

        while start < len(words):
            # Take 100 words at a time (by default)
            end = start + chunk_size
            chunk_text = " ".join(words[start:end]).strip()

            # Save the chunk along with its PDF name and page number
            if chunk_text:
                chunks.append({
                    "source": page["source"],
                    "page": page["page"],
                    "text": chunk_text
                })

            # Move forward, but keep 20 words from the previous chunk
            # This helps avoid losing context between chunks
            start += chunk_size - overlap

    return chunks


if __name__ == "__main__":
    # Read pages from all PDFs inside my data folder
    pages = extract_all_pdfs()

    # Split the extracted text into chunks
    chunks = create_chunks(pages)

    # Get the names of all PDFs and count their pages
    pdf_names = sorted({page["source"] for page in pages})
    page_counts = {
        source: sum(1 for page in pages if page["source"] == source)
        for source in pdf_names
    }

    print(f"PDF files: {', '.join(pdf_names)}")
    print(f"Total pages: {len(pages)}")
    print(f"Total chunks: {len(chunks)}")

    # Show how many pages and chunks came from each PDF
    for source in pdf_names:
        count = sum(1 for chunk in chunks if chunk["source"] == source)
        print(f"{source}: {page_counts[source]} pages, {count} chunks")

    # Print the first 3 chunks to check whether splitting worked properly
    for i, chunk in enumerate(chunks[:3], start=1):
        print(
            f"\n--- Chunk {i} | Source: {chunk['source']}, "
            f"Page {chunk['page']} ---"
        )
        print(chunk["text"])