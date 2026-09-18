from app.embedding import create_embedding
from app.vector_store import search_documents


def main():

    query = input(
        "ถามเกี่ยวกับเอกสาร: "
    )

    query_embedding = create_embedding(
        query
    )

    results = search_documents(
        query_embedding,
        n_results=3,
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    print("\nผลการค้นหา\n")

    for i, (document, metadata) in enumerate(
        zip(documents, metadatas),
        start=1,
    ):

        print("=" * 70)

        print(
            f"ผลลัพธ์ {i}"
        )

        print(
            f"ไฟล์: {metadata['source']}"
        )

        print(
            f"หน้า: {metadata['page']}"
        )

        print()

        print(document)

        print()


if __name__ == "__main__":
    main()