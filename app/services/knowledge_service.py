from app.embedding import create_embedding
from app.vector_store import (
    search_documents,
    validate_existing_store,
)

from app.courses.models import (
    CourseProfile,
)

from app.services.knowledge_models import (
    KnowledgeResult,
    KnowledgeSource,
    Citation,
)
from app.services.relevance_reranker import (
    RelevanceReranker,
)



class KnowledgeService:
    """
    Service สำหรับค้นความรู้จากฐานเอกสารของรายวิชา

    ทำหน้าที่:
    1. สร้าง embedding ของคำถาม
    2. ค้น ChromaDB
    3. แปลง raw Chroma result
       เป็น KnowledgeResult
    4. สร้าง context สำหรับ AI Tutor
    """
    MAX_CONTEXT_CHARS = 6000

    def __init__(
        self,
        course_profile: CourseProfile | None = None,
    ):
        self.course_profile = course_profile
        self.reranker = RelevanceReranker()

    def _get_chroma_path(
        self,
    ) -> str | None:

        if self.course_profile is None:
            return None

        return self.course_profile.chroma_path

    def validate_startup(
        self,
    ) -> int:

        if self.course_profile is None:

            raise ValueError(
                "CourseProfile is required "
                "for Tutor knowledge startup."
            )

        chroma_path = (
            self._get_chroma_path()
        )

        if not chroma_path:

            raise ValueError(
                "Active course does not define "
                "a chroma_path."
            )

        return validate_existing_store(
            chroma_path=chroma_path,
        )
        

    def retrieve(
        self,
        query: str,
        n_results: int = 3,
    ) -> KnowledgeResult:

        if not query.strip():
            return KnowledgeResult(
                query=query,
                context="",
                sources=[],
            )

        query_embedding = create_embedding(
            query
        )

        results = search_documents(
            query_embedding=query_embedding,
            n_results=n_results,
            chroma_path=self._get_chroma_path(),
        )

        sources = self._parse_results(
            results
        )

        sources = self._deduplicate_sources(
            sources
        )

        sources = self.reranker.rerank(
            query=query,
            sources=sources,
            max_results=3,
        )

        context = self._build_context(
            sources
        )

        citations = [
        Citation(
            source=item.source,
            page=item.page,
        )
        for item in sources
        ]

        return KnowledgeResult(
            query=query,
            context=context,
            sources=sources,
            citations=citations,
        )

    def _parse_results(
        self,
        results: dict,
    ) -> list[KnowledgeSource]:

        documents = results.get(
            "documents",
            [[]],
        )[0]

        metadatas = results.get(
            "metadatas",
            [[]],
        )[0]

        distances = results.get(
            "distances",
            [[]],
        )[0]

        sources = []

        for index, document in enumerate(
            documents
        ):

            metadata = (
                metadatas[index]
                if index < len(metadatas)
                else {}
            )

            distance = (
                distances[index]
                if index < len(distances)
                else None
            )

            sources.append(
                KnowledgeSource(
                    text=document,
                    source=metadata.get(
                        "source",
                        "unknown",
                    ),
                    page=metadata.get(
                        "page",
                        0,
                    ),
                    distance=distance,
                )
            )

        return sources

    def _build_context(
            self,
            sources: list[KnowledgeSource],
        ) -> str:

            if not sources:
                return ""

            sections = []
            total_length = 0

            for index, item in enumerate(
                sources,
                start=1,
            ):

                section = (
                    f"[Source {index}]\n"
                    f"Document: {item.source}\n"
                    f"Page: {item.page}\n"
                    f"{item.text}"
                )

                if (
                    total_length + len(section)
                    > self.MAX_CONTEXT_CHARS
                ):
                    break

                sections.append(section)
                total_length += len(section)

            return "\n\n".join(sections)

    def _deduplicate_sources(
        self,
        sources: list[KnowledgeSource],
    ) -> list[KnowledgeSource]:

        seen = set()
        unique_sources = []

        for item in sources:

            key = (
                item.source,
                item.page,
            )

            if key in seen:
                continue

            seen.add(key)
            unique_sources.append(item)

        return unique_sources