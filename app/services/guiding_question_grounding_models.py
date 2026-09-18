from dataclasses import dataclass
from typing import Literal


GuidingQuestionStatus = Literal[
    "no_questions",
    "supported",
    "unsupported",
    "invalid",
]


@dataclass(frozen=True)
class GuidingQuestionGroundingResult:
    status: GuidingQuestionStatus
    reason: str
    issues: tuple[str, ...] = ()

    @property
    def is_safe(self) -> bool:
        return self.status in (
            "no_questions",
            "supported",
        )