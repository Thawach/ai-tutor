from dataclasses import dataclass
from typing import Literal


RepairEvidenceStatus = Literal[
    "verified",
    "no_evidence",
    "invalid",
]


@dataclass(frozen=True)
class RepairEvidenceSelectionResult:
    """
    Verified source evidence selected for factual
    response repair.

    status:
    - verified:
      one or more evidence excerpts were verified
      against the supplied course knowledge.

    - no_evidence:
      the selector found no suitable source evidence.

    - invalid:
      selector output could not be trusted or verified.
    """

    status: RepairEvidenceStatus

    evidence_quotes: tuple[str, ...]

    reason: str

    issues: tuple[str, ...] = ()

    @property
    def has_verified_evidence(
        self,
    ) -> bool:
        return (
            self.status == "verified"
            and bool(self.evidence_quotes)
        )