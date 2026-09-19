import chromadb
from unittest.mock import patch

from app.courses.loader import CourseProfileLoader
from app.services.knowledge_service import KnowledgeService


EXPECTED = {
    "electronics": {
        "records": 17,
        "source": "Lecture_Transistor.pdf",
        "other_course": "mathematics",
        "forbidden_source": "mathematics.pdf",
    },
    "mathematics": {
        "records": 124,
        "source": "mathematics.pdf",
        "other_course": "electronics",
        "forbidden_source": "Lecture_Transistor.pdf",
    },
}


def load_seed_vector(profile):
    client = chromadb.PersistentClient(
        path=profile.chroma_path
    )

    collection = client.get_collection(
        name="course_documents"
    )

    result = collection.get(
        limit=1,
        include=[
            "embeddings",
            "metadatas",
        ],
    )

    vector = list(result["embeddings"][0])

    source = (
        result["metadatas"][0]
        .get("source")
    )

    return (
        collection.count(),
        vector,
        source,
    )


def run_retrieval(
    *,
    profile,
    vector,
    query,
    expected_source,
    forbidden_source,
):
    service = KnowledgeService(
        course_profile=profile
    )

    startup_count = (
        service.validate_startup()
    )

    with patch(
        "app.services.knowledge_service.create_embedding",
        return_value=vector,
    ) as embedding_mock:

        result = service.retrieve(
            query=query,
            n_results=3,
        )

    if embedding_mock.call_count != 1:
        raise AssertionError(
            "Expected exactly one intercepted "
            "embedding call."
        )

    if not result.sources:
        raise AssertionError(
            f"No RAG sources returned for "
            f"{profile.course_id}."
        )

    returned_sources = {
        item.source
        for item in result.sources
    }

    if returned_sources != {
        expected_source
    }:
        raise AssertionError(
            f"{profile.course_id}: "
            f"unexpected sources "
            f"{sorted(returned_sources)}"
        )

    if forbidden_source in returned_sources:
        raise AssertionError(
            f"{profile.course_id}: "
            "cross-course source leakage detected."
        )

    if not result.context.strip():
        raise AssertionError(
            f"{profile.course_id}: "
            "empty RAG context."
        )

    if not result.citations:
        raise AssertionError(
            f"{profile.course_id}: "
            "missing citations."
        )

    citation_sources = {
        item.source
        for item in result.citations
    }

    if citation_sources != {
        expected_source
    }:
        raise AssertionError(
            f"{profile.course_id}: "
            "citation source mismatch."
        )

    return (
        startup_count,
        result,
    )


def main():
    print(
        "AI Tutor 16.22 "
        "Dual-Course Deployment Smoke"
    )
    print("-" * 56)

    loader = CourseProfileLoader()

    profiles = {
        course: loader.load(course)
        for course in EXPECTED
    }

    vectors = {}

    # -----------------------------------------------------
    # Store / vector integrity
    # -----------------------------------------------------

    for course, spec in EXPECTED.items():
        profile = profiles[course]

        (
            count,
            vector,
            source,
        ) = load_seed_vector(profile)

        if count != spec["records"]:
            raise AssertionError(
                f"{course}: "
                f"expected {spec['records']} records, "
                f"got {count}."
            )

        if len(vector) != 768:
            raise AssertionError(
                f"{course}: "
                f"expected 768-dimensional embedding, "
                f"got {len(vector)}."
            )

        if source != spec["source"]:
            raise AssertionError(
                f"{course}: "
                f"unexpected stored source "
                f"{source!r}."
            )

        vectors[course] = vector

        print(
            f"{course:<14} "
            f"STORE=PASS "
            f"RECORDS={count} "
            f"DIM={len(vector)}"
        )

    # -----------------------------------------------------
    # Normal course-local retrieval
    # -----------------------------------------------------

    for course, spec in EXPECTED.items():
        profile = profiles[course]

        startup_count, result = (
            run_retrieval(
                profile=profile,
                vector=vectors[course],
                query=(
                    f"deployment smoke "
                    f"{course}"
                ),
                expected_source=(
                    spec["source"]
                ),
                forbidden_source=(
                    spec["forbidden_source"]
                ),
            )
        )

        print(
            f"{course:<14} "
            f"RAG=PASS "
            f"STARTUP_RECORDS={startup_count} "
            f"RESULTS={len(result.sources)}"
        )

    # -----------------------------------------------------
    # Cross-course isolation
    #
    # Deliberately query each course store using an
    # embedding taken from the OTHER course.
    #
    # Because both stores use the same vector dimension,
    # this is a strong path-isolation test.
    # -----------------------------------------------------

    for course, spec in EXPECTED.items():
        profile = profiles[course]

        foreign_course = (
            spec["other_course"]
        )

        _, result = run_retrieval(
            profile=profile,
            vector=vectors[foreign_course],
            query=(
                "cross-course isolation smoke"
            ),
            expected_source=(
                spec["source"]
            ),
            forbidden_source=(
                spec["forbidden_source"]
            ),
        )

        returned_sources = sorted({
            item.source
            for item in result.sources
        })

        print(
            f"{course:<14} "
            f"ISOLATION=PASS "
            f"SOURCES={returned_sources}"
        )

    print("-" * 56)
    print(
        "NETWORK_EMBEDDING_CALLS=0 "
        "(embedding boundary intercepted)"
    )
    print(
        "DUAL_COURSE_RAG_SMOKE=PASS"
    )


if __name__ == "__main__":
    main()
