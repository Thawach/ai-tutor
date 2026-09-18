from app.services.knowledge_models import (
    KnowledgeSource,
)


class RelevanceReranker:
    """
    Lightweight relevance reranker.

    v1.1:
    - ใช้ vector distance เป็นหลัก
    - ใช้ keyword overlap เป็น secondary signal
    - ไม่ใช้ LLM เพื่อหลีกเลี่ยง latency เพิ่ม
    """

    KEYWORD_BONUS = 2.0

    def rerank(
        self,
        query: str,
        sources: list[KnowledgeSource],
        max_results: int = 3,
    ) -> list[KnowledgeSource]:

        if not sources:
            return []

        query_terms = self._extract_terms(
            query
        )

        ranked = []

        for source in sources:

            text_terms = self._extract_terms(
                source.text
            )

            overlap = len(
                query_terms.intersection(
                    text_terms
                )
            )

            distance = (
                source.distance
                if source.distance is not None
                else float("inf")
            )

            adjusted_distance = (
                distance
                - (
                    overlap
                    * self.KEYWORD_BONUS
                )
            )

            ranked.append(
                (
                    adjusted_distance,
                    distance,
                    overlap,
                    source,
                )
            )

        ranked.sort(
            key=lambda item: (
                item[0],
                item[1],
            )
        )

        return [
            item[3]
            for item in ranked[:max_results]
        ]

    def _extract_terms(
        self,
        text: str,
    ) -> set[str]:

        normalized = (
            text.lower()
            .replace(",", " ")
            .replace(".", " ")
            .replace("(", " ")
            .replace(")", " ")
            .replace(":", " ")
            .replace(";", " ")
            .replace("/", " ")
            .replace("-", " ")
        )

        return {
            token.strip()
            for token in normalized.split()
            if len(token.strip()) >= 2
        }