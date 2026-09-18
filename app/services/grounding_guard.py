from app.services.grounding_models import (
    GroundingStatus,
)

from app.services.knowledge_models import (
    KnowledgeResult,
)

from app.services.relevance_models import (
    RelevanceResult,
)


class GroundingGuard:
    """
    ตรวจว่า retrieved knowledge สามารถใช้เป็น
    grounded course knowledge ได้จริงหรือไม่
    """

    def evaluate(
        self,
        result: KnowledgeResult,
        relevance: RelevanceResult,
    ) -> GroundingStatus:

        if not result.has_context:
            return GroundingStatus(
                has_knowledge=False,
                reason="no_context",
            )

        if not result.sources:
            return GroundingStatus(
                has_knowledge=False,
                reason="no_sources",
            )

        if not relevance.is_relevant:
            return GroundingStatus(
                has_knowledge=False,
                reason="retrieved_context_not_relevant",
            )

        return GroundingStatus(
            has_knowledge=True,
            reason="knowledge_available",
        )