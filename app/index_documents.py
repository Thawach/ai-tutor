import argparse

from pathlib import Path

from app.core.config import (
    ACTIVE_COURSE,
)

from app.courses.loader import (
    CourseProfileLoader,
)

from app.document_loader import (
    load_pdf_pages,
)

from app.text_splitter import (
    split_text,
)

from app.embedding import (
    create_embedding,
)

from app.vector_store import (
    add_document_chunk,
    get_collection,
    reset_collection,
)


def index_pdf(
    pdf_path: Path,
    chroma_path: str,
):

    print(
        f"\nกำลังประมวลผล: "
        f"{pdf_path.name}"
    )

    pages = load_pdf_pages(
        str(pdf_path)
    )

    total_chunks = 0

    for page in pages:

        chunks = split_text(
            page["text"]
        )

        for (
            chunk_index,
            chunk,
        ) in enumerate(chunks):

            chunk_id = (
                f"{pdf_path.stem}"
                f"_page_{page['page']}"
                f"_chunk_{chunk_index}"
            )

            embedding = create_embedding(
                chunk
            )

            add_document_chunk(
                chunk_id=chunk_id,
                text=chunk,
                embedding=embedding,
                source=page["source"],
                page=page["page"],
                chroma_path=chroma_path,
            )

            total_chunks += 1

            print(
                f"หน้า {page['page']} "
                f"chunk {chunk_index + 1}"
            )

    print(
        f"\nเสร็จแล้ว: "
        f"{total_chunks} chunks"
    )

    return total_chunks


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Index documents for the active course."
        )
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help=(
            "Reset the active course Chroma collection "
            "before indexing."
        ),
    )

    return parser.parse_args()


def main():

    args = parse_args()

    # -------------------------
    # Load active course
    # -------------------------

    profile = (
        CourseProfileLoader()
        .load(
            ACTIVE_COURSE
        )
    )

    document_folder = Path(
        profile.document_path
    )

    chroma_path = (
        profile.chroma_path
    )

    print(
        f"\nActive Course: "
        f"{profile.course_name}"
    )

    print(
        f"Document Path: "
        f"{document_folder}"
    )

    print(
        f"Chroma Path: "
        f"{chroma_path}"
    )

    # -------------------------
    # Validate document folder
    # -------------------------

    if not document_folder.exists():

        print(
            "\nไม่พบโฟลเดอร์เอกสาร: "
            f"{document_folder}"
        )

        return

    # -------------------------
    # Find PDF files
    # -------------------------

    pdf_files = sorted(
        document_folder.glob(
            "*.pdf"
        )
    )

    if not pdf_files:

        print(
            "\nไม่พบไฟล์ PDF ใน: "
            f"{document_folder}"
        )

        return

    # -------------------------
    # Inspect existing collection
    # -------------------------

    collection = get_collection(
        chroma_path=chroma_path,
    )

    existing_count = (
        collection.count()
    )

    print(
        f"Existing Chunks: "
        f"{existing_count}"
    )

    # -------------------------
    # Optional reset
    # -------------------------

    if args.reset:

        print(
            "\nReset requested."
        )

        print(
            "Resetting active course "
            "collection only..."
        )

        collection = reset_collection(
            chroma_path=chroma_path,
        )

        print(
            "Active course collection "
            "reset complete."
        )

    # -------------------------
    # Index PDFs
    # -------------------------

    total_indexed_chunks = 0

    for pdf_file in pdf_files:

        indexed_chunks = index_pdf(
            pdf_path=pdf_file,
            chroma_path=chroma_path,
        )

        total_indexed_chunks += (
            indexed_chunks
        )

    # -------------------------
    # Integrity check
    # -------------------------

    collection = get_collection(
        chroma_path=chroma_path,
    )

    final_count = (
        collection.count()
    )

    print(
        "\nIndex Summary"
    )

    print(
        f"- Indexed this run: "
        f"{total_indexed_chunks}"
    )

    print(
        f"- Collection count: "
        f"{final_count}"
    )

    if args.reset:

        if (
            final_count
            != total_indexed_chunks
        ):

            raise RuntimeError(
                "Index integrity check failed: "
                f"indexed "
                f"{total_indexed_chunks} chunks "
                f"but collection contains "
                f"{final_count} records."
            )

        print(
            "- Integrity check: PASS"
        )

if __name__ == "__main__":
    main()