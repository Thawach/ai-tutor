from dataclasses import dataclass
from typing import Optional


@dataclass
class EvaluationResult:
    """
    ผลการประเมินคำตอบของผู้เรียน
    """

    classification: str

    confidence: Optional[float] = None

    reason: Optional[str] = None

    misconception: Optional[str] = None