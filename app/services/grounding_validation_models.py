from dataclasses import dataclass, field

@dataclass
class GroundingValidationResult:
    """
    ผลการตรวจคำตอบของ AI Tutor
    เทียบกับ course knowledge ที่ retrieve มา
    """

    status: str

    confidence: float

    reason: str

    issues: list[str] = field(
        default_factory=list
    )

    @property
    def is_supported(self) -> bool:
        return self.status == "supported"