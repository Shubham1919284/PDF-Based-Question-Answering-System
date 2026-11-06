import pymupdf
from pathlib import Path


# All my PDFs are stored in this folder
DATA_DIR = Path(__file__).resolve().parent / "data"
PDF_PATH = DATA_DIR / "sample.pdf"


def extract_text(pdf_path):
    # Read one PDF and keep the filename and page number with its text
    pdf_path = Path(pdf_path)
    pages = []

    with pymupdf.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            # Extract the text from the current page
            text = page.get_text("text")

            # Save the text along with its source and page number
            pages.append({
                "source": pdf_path.name,
                "page": page_number,
                "text": text
            })

    return pages


def extract_all_pdfs(data_dir=DATA_DIR):
    # Find every PDF inside my data folder
    data_dir = Path(data_dir)
    pdf_files = sorted(data_dir.glob("*.pdf"))

    # Stop here if the folder doesn't contain any PDFs
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {data_dir.resolve()}"
        )

    all_pages = []

    # Read each PDF and collect all its pages together
    for pdf_path in pdf_files:
        print(f"Reading: {pdf_path.name}...")
        all_pages.extend(extract_text(pdf_path))

    return all_pages


if __name__ == "__main__":
    # Extract text from all PDFs to check that everything works
    pages = extract_all_pdfs()

    # Get the names of the PDFs we loaded
    sources = sorted({page["source"] for page in pages})

    print(f"\nPDF files loaded: {len(sources)}")
    print(f"Total pages extracted: {len(pages)}")

    # Show how many pages were read from each PDF
    for source in sources:
        page_count = sum(
            1 for page in pages
            if page["source"] == source
        )
        print(f"{source}: {page_count} pages")