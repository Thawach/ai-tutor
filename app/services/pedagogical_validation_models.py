from dataclasses import dataclass, field


@dataclass
class PedagogicalValidationResult:
    """
    ผลการตรวจว่าคำตอบของ AI Tutor
    สอดคล้องกับ scaffolding strategy
    และ intervention ปัจจุบันหรือไม่
    """

    status: str

    confidence: float

    reason: str

    issues: list[str] = field(
        default_factory=list
    )

    @property
    def is_valid(self) -> bool:
        return self.status == "valid"