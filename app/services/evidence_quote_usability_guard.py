import re

from dataclasses import dataclass
from typing import Literal


EvidenceQuoteUsabilityStatus = Literal[
    "usable",
    "unusable",
]


@dataclass(frozen=True)
class EvidenceQuoteUsabilityResult:
    status: EvidenceQuoteUsabilityStatus
    reason: str
    issues: tuple[str, ...] = ()

    @property
    def is_usable(self) -> bool:
        return self.status == "usable"


class EvidenceQuoteUsabilityGuard:
    """
    Deterministic usability guard for exact verified
    evidence excerpts.

    This component does NOT:
    - verify source provenance;
    - rewrite evidence;
    - repair extraction;
    - infer missing content;
    - call an LLM.

    It only rejects high-confidence evidence defects
    that make an exact quote unsafe for direct
    user-facing composition.
    """

    _EXTRACTION_DAMAGE_PATTERNS = (
        re.compile(
            r"\bwith\s+the\s+voltage\s+and\s+as\s+shown\b",
            flags=re.IGNORECASE,
        ),
    )

    _UNFINISHED_ENDINGS = (
        re.compile(
            r"\bas\s*$",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"\bbecause\s*$",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"\btherefore\s*$",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"\bsuch\s+as\s*$",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"\bincluding\s*$",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r":\s*$",
            flags=re.IGNORECASE,
        ),
    )

    def evaluate(
        self,
        quote: str,
    ) -> EvidenceQuoteUsabilityResult:

        if not isinstance(
            quote,
            str,
        ):
            raise TypeError(
                "quote must be str."
            )

        normalized = self._normalize(
            quote
        )

        if not normalized:
            return EvidenceQuoteUsabilityResult(
                status="unusable",
                reason=(
                    "Evidence quote was empty."
                ),
                issues=(
                    "Empty evidence quote.",
                ),
            )

        for pattern in (
            self._EXTRACTION_DAMAGE_PATTERNS
        ):
            if pattern.search(
                normalized
            ):
                return EvidenceQuoteUsabilityResult(
                    status="unusable",
                    reason=(
                        "Evidence quote contains "
                        "a known extraction-damaged "
                        "text pattern."
                    ),
                    issues=(
                        (
                            "Extraction-damaged "
                            "evidence text."
                        ),
                    ),
                )

        for pattern in (
            self._UNFINISHED_ENDINGS
        ):
            if pattern.search(
                normalized
            ):
                return EvidenceQuoteUsabilityResult(
                    status="unusable",
                    reason=(
                        "Evidence quote ends with "
                        "an unfinished lead-in or "
                        "continuation marker."
                    ),
                    issues=(
                        (
                            "Incomplete evidence "
                            "lead-in."
                        ),
                    ),
                )

        return EvidenceQuoteUsabilityResult(
            status="usable",
            reason=(
                "Evidence quote passed "
                "deterministic usability checks."
            ),
            issues=(),
        )

    def _normalize(
        self,
        text: str,
    ) -> str:

        return " ".join(
            text
            .replace("\u00a0", " ")
            .replace("\u202f", " ")
            .split()
        )