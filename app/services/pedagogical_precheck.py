from dataclasses import dataclass


@dataclass
class PedagogicalPrecheckResult:
    """
    ผลการตรวจเชิงโครงสร้างด้วย Python

    status:
    - valid
    - violation
    - uncertain
    """

    status: str
    reason: str
    issues: list[str]


class PedagogicalPrecheck:
    """
    ตรวจ pedagogical structure ที่ตัดสินได้แน่นอน
    โดยไม่เรียก LLM

    หากไม่แน่ใจ ให้คืน uncertain
    แล้วส่งต่อให้ PedagogicalResponseValidator
    """

    def evaluate(
        self,
        response: str,
        strategy_name: str,
    ) -> PedagogicalPrecheckResult:

        text = (
            response
            .replace("？", "?")
            .strip()
        )

        question_count = text.count("?")

        # -------------------------
        # Empty response
        # -------------------------

        if not text:
            return PedagogicalPrecheckResult(
                status="violation",
                reason="Tutor response is empty.",
                issues=[
                    "Empty tutor response.",
                ],
            )

        # -------------------------
        # Guiding question
        # -------------------------

        if strategy_name == "guiding_question":

            return self._check_guiding_question(
                text=text,
                question_count=question_count,
            )

        # -------------------------
        # Hint
        # -------------------------

        if strategy_name == "hint":

            return self._check_hint(
                text=text,
                question_count=question_count,
            )

        # -------------------------
        # Other strategies
        # -------------------------

        return PedagogicalPrecheckResult(
            status="uncertain",
            reason=(
                "This strategy requires semantic "
                "pedagogical evaluation."
            ),
            issues=[],
        )

    def _check_guiding_question(
        self,
        text: str,
        question_count: int,
    ) -> PedagogicalPrecheckResult:

        if question_count == 0:
            return PedagogicalPrecheckResult(
                status="violation",
                reason=(
                    "Guiding-question strategy requires "
                    "a question."
                ),
                issues=[
                    "No guiding question detected.",
                ],
            )

        if question_count > 1:
            return PedagogicalPrecheckResult(
                status="violation",
                reason=(
                    "Guiding-question strategy allows "
                    "only one main question."
                ),
                issues=[
                    "Multiple explicit questions detected.",
                ],
            )

        # ต้องจบด้วยคำถาม
        if not text.endswith("?"):
            return PedagogicalPrecheckResult(
                status="uncertain",
                reason=(
                    "One question was detected but the "
                    "response also contains text after it."
                ),
                issues=[],
            )

        # Level 1 ที่เป็นคำถามเดียวสั้น ๆ
        # สามารถยอมรับได้โดยไม่ต้องเรียก LLM
        return PedagogicalPrecheckResult(
            status="valid",
            reason=(
                "Exactly one explicit guiding question "
                "was detected."
            ),
            issues=[],
        )

    def _check_hint(
        self,
        text: str,
        question_count: int,
    ) -> PedagogicalPrecheckResult:

        if question_count == 0:
            return PedagogicalPrecheckResult(
                status="violation",
                reason=(
                    "Hint strategy requires a follow-up "
                    "question."
                ),
                issues=[
                    "No follow-up question detected.",
                ],
            )

        if question_count > 1:
            return PedagogicalPrecheckResult(
                status="violation",
                reason=(
                    "Hint strategy should contain one "
                    "main follow-up question."
                ),
                issues=[
                    "Multiple explicit questions detected.",
                ],
            )

        if not text.endswith("?"):
            return PedagogicalPrecheckResult(
                status="uncertain",
                reason=(
                    "A question exists but additional "
                    "text follows the question."
                ),
                issues=[],
            )

        # -------------------------
        # Detect clue before question
        # -------------------------

        question_index = text.rfind("?")

        before_question = (
            text[:question_index]
            .strip()
        )

        lines = [
            line.strip()
            for line in before_question.splitlines()
            if line.strip()
        ]

        # ถ้ามีหลายบรรทัด:
        # บรรทัดสุดท้ายคือ question text
        # บรรทัดก่อนหน้าคือ hint
        if len(lines) >= 2:

            hint_text = " ".join(
                lines[:-1]
            ).strip()

            if hint_text:
                return PedagogicalPrecheckResult(
                    status="valid",
                    reason=(
                        "A clue followed by one "
                        "follow-up question was detected."
                    ),
                    issues=[],
                )

        # -------------------------
        # Same-line Hint:
        # -------------------------

        lowered = text.lower()

        explicit_hint_markers = (
            "hint:",
            "hint -",
            "คำใบ้:",
            "คำใบ้ -",

            "ลองสังเกต",
            "ลองพิจารณา",
            "ลองนึกถึง",
            "ลองคิดถึง",
            "ลองคิดว่า",
            "ลองคิดดูว่า",
            "ลองดู",

            "think about",
            "consider",
            "notice",
            "focus on",
        )

        if any(
            marker in lowered
            for marker in explicit_hint_markers
        ):
            return PedagogicalPrecheckResult(
                status="valid",
                reason=(
                    "An explicit hint or attention-directing "
                    "cue followed by one question was detected."
                ),
                issues=[],
            )

        # -------------------------
        # Question only
        # -------------------------

        return PedagogicalPrecheckResult(
            status="uncertain",
            reason=(
                "One question was detected, but Python "
                "cannot reliably determine whether a "
                "meaningful hint precedes it."
            ),
            issues=[],
        )