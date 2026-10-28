import pymupdf
from pathlib import Path

PDF_PATH = Path("data/sample.pdf")

def extract_text(pdf_path):
    pages=[]

    with pymupdf.open(pdf_path) as pdf:
        for page_number,page in enumerate(pdf,start=1):
            text=page.get_text("text")

            pages.append({"page":page_number, "text":text})

    return pages

if __name__ == "__main__":
    pages=extract_text(PDF_PATH)

    print(f"Total pages: {len(pages)}")

    for page in pages[:2]:
        print(f"\n--- Page {page['page']} ---\n")
        print(page["text"][:500])