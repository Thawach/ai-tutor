import pymupdf
from pathlib import Path


def load_pdf_pages(pdf_path: str) -> list[dict]:
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์: {pdf_path}")

    document = pymupdf.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document, start=1):
        text = page.get_text("text").strip()

        if text:
            pages.append(
                {
                    "source": pdf_path.name,
                    "page": page_number,
                    "text": text,
                }
            )

    document.close()

    return pages