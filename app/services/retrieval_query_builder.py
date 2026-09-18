class RetrievalQueryBuilder:
    """
    สร้าง query สำหรับค้นฐานความรู้

    v1:
    ใช้ original learning question โดยตรงก่อน

    ภายหลังสามารถเพิ่ม:
    - translation
    - query expansion
    - keyword extraction
    - topic context
    """

    def build(
        self,
        original_question: str,
        tutor_question: str | None = None,
        learner_response: str | None = None,
    ) -> str:

        parts = []

        if original_question:
            parts.append(original_question.strip())

        if tutor_question:
            parts.append(tutor_question.strip())

        return "\n".join(parts)