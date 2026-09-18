from dataclasses import dataclass


@dataclass
class MisconceptionRecord:
    """
    เก็บข้อมูลความเข้าใจคลาดเคลื่อนของผู้เรียน
    ในระดับ prototype

    ภายหลังจะสามารถ persist ลง PostgreSQL ได้
    """

    description: str

    occurrence_count: int = 1

    status: str = "active"