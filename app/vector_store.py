import chromadb

from app.config import CHROMA_PATH

from pathlib import Path

import chromadb


DEFAULT_COLLECTION_NAME = "course_documents"


# =========================================================
# CHROMA CLIENT
# =========================================================

def get_client(
    chroma_path: str | None = None,
):
    """
    Return a persistent Chroma client.

    If chroma_path is not provided, use the legacy
    CHROMA_PATH configuration for backward compatibility.
    """

    active_chroma_path = (
        chroma_path
        or CHROMA_PATH
    )

    return chromadb.PersistentClient(
        path=active_chroma_path
    )


# =========================================================
# COLLECTION
# =========================================================

def get_collection(
    chroma_path: str | None = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
):
    """
    Return the requested Chroma collection.
    """

    client = get_client(
        chroma_path=chroma_path,
    )

    return client.get_or_create_collection(
        name=collection_name
    )


def get_existing_collection(
    chroma_path: str | None = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
):
    """
    Return an existing Chroma collection for runtime reads.

    Runtime retrieval must never create a missing store
    or collection.
    """

    active_chroma_path = (
        chroma_path
        or CHROMA_PATH
    )

    store_path = Path(
        active_chroma_path
    )

    if not store_path.exists():

        raise FileNotFoundError(
            "Chroma store not found: "
            f"{store_path}"
        )

    if not store_path.is_dir():

        raise NotADirectoryError(
            "Chroma store path is not "
            "a directory: "
            f"{store_path}"
        )

    database_path = (
        store_path
        /
        "chroma.sqlite3"
    )

    if not database_path.is_file():

        raise FileNotFoundError(
            "Chroma database not found: "
            f"{database_path}"
        )

    client = chromadb.PersistentClient(
        path=str(store_path)
    )

    return client.get_collection(
        name=collection_name
    )

def validate_existing_store(
    chroma_path: str,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> int:
    """
    Validate that a runtime knowledge store exists,
    contains the required collection, and has indexed
    knowledge available.

    Returns the indexed chunk count.
    """

    if not chroma_path:

        raise ValueError(
            "chroma_path is required for "
            "runtime knowledge-store validation."
        )

    collection = get_existing_collection(
        chroma_path=chroma_path,
        collection_name=collection_name,
    )

    count = collection.count()

    if count <= 0:

        raise RuntimeError(
            "Chroma knowledge collection "
            "is empty: "
            f"{chroma_path}"
        )

    return count


# =========================================================
# RESET COLLECTION
# =========================================================

def reset_collection(
    chroma_path: str,
    collection_name: str = DEFAULT_COLLECTION_NAME,
):
    """
    Delete and recreate one Chroma collection inside
    the supplied course-specific Chroma path.

    Only the requested course store is affected.
    Other course stores are untouched.
    """

    if not chroma_path:
        raise ValueError(
            "chroma_path is required for reset_collection."
        )

    client = get_client(
        chroma_path=chroma_path,
    )

    # -------------------------
    # Delete existing collection
    # -------------------------

    try:

        client.delete_collection(
            name=collection_name,
        )

    except Exception:

        # The collection may not exist yet.
        pass

    # -------------------------
    # Recreate clean collection
    # -------------------------

    return client.get_or_create_collection(
        name=collection_name
    )


# =========================================================
# ADD DOCUMENT CHUNK
# =========================================================

def add_document_chunk(
    chunk_id: str,
    text: str,
    embedding: list[float],
    source: str,
    page: int,
    chroma_path: str | None = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
):

    collection = get_collection(
        chroma_path=chroma_path,
        collection_name=collection_name,
    )

    collection.add(
        ids=[
            chunk_id,
        ],
        documents=[
            text,
        ],
        embeddings=[
            embedding,
        ],
        metadatas=[
            {
                "source": source,
                "page": page,
            }
        ],
    )


# =========================================================
# SEARCH DOCUMENTS
# =========================================================

def search_documents(
    query_embedding: list[float],
    n_results: int = 3,
    chroma_path: str | None = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
):

    collection = get_existing_collection(
        chroma_path=chroma_path,
        collection_name=collection_name,
    )

    return collection.query(
        query_embeddings=[
            query_embedding,
        ],
        n_results=n_results,
    )