from dataclasses import dataclass


@dataclass(frozen=True)
class ScaffoldingIntervention:
    """
    นิยาม intervention พิเศษที่ใช้ร่วมกับ
    scaffolding strategy ปกติ
    """

    name: str
    description: str
    instruction: str


MISCONCEPTION_CORRECTION = ScaffoldingIntervention(
    name="misconception_correction",
    description=(
        "ใช้เมื่อผู้เรียนแสดงความเข้าใจคลาดเคลื่อน "
        "เพื่อแก้แนวคิดโดยไม่เฉลยทั้งหมดทันที"
    ),
    instruction="""
MISCONCEPTION INTERVENTION

The learner has demonstrated a misconception.

Important rules:

- Do NOT praise or confirm the incorrect idea.
- Clearly indicate that one part of the learner's reasoning
  needs to be reconsidered.
- Focus specifically on the identified misconception.
- Provide a corrective cue appropriate to the current
  scaffolding level.
- Do not overwhelm the learner with a full lecture.
- Do not introduce unrelated concepts.
- Encourage the learner to revise the incorrect idea.
- Ask exactly ONE main follow-up question.
"""
)


NO_SPECIAL_INTERVENTION = ScaffoldingIntervention(
    name="none",
    description="ไม่มี intervention พิเศษ",
    instruction="",
)


def get_intervention(
    evaluation: str | None,
) -> ScaffoldingIntervention:
    """
    เลือก intervention ตามผลการประเมิน
    """

    if evaluation == "misconception":
        return MISCONCEPTION_CORRECTION

    return NO_SPECIAL_INTERVENTION