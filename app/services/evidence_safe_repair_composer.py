from app.services.repair_evidence_selection_models import (
    RepairEvidenceSelectionResult,
)


class EvidenceSafeRepairComposer:
    """
    Deterministically composes a factual repair body
    from verified source evidence only.

    No paraphrasing.
    No summarization.
    No inference.
    No LLM.
    """

    def compose(
        self,
        selection: RepairEvidenceSelectionResult,
    ) -> str | None:

        if not isinstance(
            selection,
            RepairEvidenceSelectionResult,
        ):
            raise TypeError(
                "selection must be "
                "RepairEvidenceSelectionResult."
            )

        if not selection.has_verified_evidence:
            return None

        evidence_blocks: list[str] = []

        for quote in selection.evidence_quotes:

            if not isinstance(quote, str):
                raise TypeError(
                    "Every evidence quote must be str."
                )

            if not quote.strip():
                raise ValueError(
                    "Evidence quote must not be empty."
                )

            # Preserve verified source text exactly.
            evidence_blocks.append(quote)

        if not evidence_blocks:
            return None

        return "\n\n".join(
            evidence_blocks
        )