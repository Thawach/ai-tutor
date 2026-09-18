from dataclasses import dataclass


@dataclass
class RelevanceResult:
    """
    ผลการตรวจว่า retrieved course knowledge
    เกี่ยวข้องกับคำถามหรือไม่
    """

    is_relevant: bool
    confidence: float
    reason: str