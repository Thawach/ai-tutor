from dataclasses import dataclass
from typing import Literal


EvidenceSafeTranslationStatus = Literal[
    "translated",
    "not_required",
    "invalid",
]


@dataclass(frozen=True)
class EvidenceSafeTranslationResult:
    status: EvidenceSafeTranslationStatus
    translated_text: str
    source_text: str
    reason: str
    issues: tuple[str, ...] = ()

    @property
    def has_translation(self) -> bool:
        return (
            self.status == "translated"
            and bool(self.translated_text.strip())
        )