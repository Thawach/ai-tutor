import re

from app.services.learner_turn_intent_models import (
    LearnerTurnIntentResult,
)


class LearnerTurnIntentGuard:
    """
    Deterministic learner-turn intent detector.

    This guard performs surface-level intent detection only.

    It does not:
    - call an LLM
    - evaluate learner correctness
    - modify learner input
    - perform retrieval
    - trigger scaffolding
    - change Tutor responses
    """

    INTENT_ANSWER = "answer"

    INTENT_FOLLOW_UP_QUESTION = (
        "follow_up_question"
    )

    INTENT_CLARIFICATION_QUESTION = (
        "clarification_question"
    )

    INTENT_TOPIC_CHANGE = (
        "topic_change"
    )

    INTENT_UNCERTAIN = "uncertain"

    # =========================================================
    # Explicit answer / don't-know signals
    # =========================================================

    DONT_KNOW_PATTERNS = (
        "ไม่รู้",
        "ไม่ทราบ",
        "จำไม่ได้",
        "ตอบไม่ได้",
        "นึกไม่ออก",
    )

    # =========================================================
    # Explicit topic-change signals
    # =========================================================

    TOPIC_CHANGE_PATTERNS = (
        "เปลี่ยนเรื่อง",
        "เปลี่ยนหัวข้อ",
        "ขอถามอีกเรื่อง",
        "ถามอีกเรื่อง",
        "ขอถามเรื่องใหม่",
        "หัวข้อใหม่",
        "อีกหัวข้อ",
        "ต่อไปขอถามเรื่อง",
    )

    # =========================================================
    # Clarification signals
    # =========================================================

    CLARIFICATION_PATTERNS = (
        "หมายถึงอะไร",
        "หมายความว่าอะไร",
        "แปลว่าอะไร",
        "ช่วยอธิบาย",
        "อธิบายอีกครั้ง",
        "อธิบายใหม่",
        "ขอคำอธิบาย",
        "ไม่เข้าใจคำว่า",
        "คำว่าอะไร",
    )

    # =========================================================
    # Thai question signals
    # =========================================================

    THAI_QUESTION_PATTERNS = (
        "อะไร",
        "อย่างไร",
        "ยังไง",
        "ทำไม",
        "เพราะอะไร",
        "เมื่อไร",
        "เมื่อไหร่",
        "ที่ไหน",
        "ตรงไหน",
        "ใคร",
        "เท่าไร",
        "เท่าไหร่",
        "กี่",
        "หรือไม่",
        "หรือเปล่า",
        "ไหม",
        "มั้ย",
        "หรือยัง",
    )

    # =========================================================
    # English question signals
    # =========================================================

    ENGLISH_QUESTION_PREFIXES = (
        "what ",
        "why ",
        "how ",
        "when ",
        "where ",
        "who ",
        "which ",
    )

    # =========================================================
    # Public API
    # =========================================================

    def evaluate(
        self,
        learner_text: str,
    ) -> LearnerTurnIntentResult:

        if not isinstance(
            learner_text,
            str,
        ):
            raise TypeError(
                "learner_text must be a string."
            )

        normalized = (
            self._normalize(
                learner_text
            )
        )

        # -----------------------------------------------------
        # Empty / low evidence
        # -----------------------------------------------------

        if not normalized:

            return LearnerTurnIntentResult(
                intent=self.INTENT_UNCERTAIN,
                reason=(
                    "Learner input is empty or "
                    "contains no usable text."
                ),
                confidence=0.0,
                signals=(),
                is_question=False,
            )

        # -----------------------------------------------------
        # Explicit don't-know response
        #
        # Important:
        # "ไม่รู้อะไรเลย" must remain an answer rather than
        # being misclassified merely because it contains "อะไร".
        # -----------------------------------------------------

        dont_know_signal = (
            self._find_pattern(
                normalized,
                self.DONT_KNOW_PATTERNS,
            )
        )

        if dont_know_signal is not None:

            return LearnerTurnIntentResult(
                intent=self.INTENT_ANSWER,
                reason=(
                    "Learner input contains an explicit "
                    "answer or don't-know response signal."
                ),
                confidence=0.95,
                signals=(
                    f"dont_know:{dont_know_signal}",
                ),
                is_question=False,
            )

        # -----------------------------------------------------
        # Explicit topic change
        #
        # Highest question-like routing priority because:
        #
        # "ขอถามอีกเรื่อง ตัวเก็บประจุคืออะไร"
        #
        # is primarily a new topic rather than a follow-up.
        # -----------------------------------------------------

        topic_signal = (
            self._find_pattern(
                normalized,
                self.TOPIC_CHANGE_PATTERNS,
            )
        )

        if topic_signal is not None:

            return LearnerTurnIntentResult(
                intent=self.INTENT_TOPIC_CHANGE,
                reason=(
                    "Learner input contains an explicit "
                    "topic-change signal."
                ),
                confidence=0.98,
                signals=(
                    f"topic_change:{topic_signal}",
                ),
                is_question=(
                    self._looks_like_question(
                        normalized
                    )
                ),
            )

        # -----------------------------------------------------
        # Clarification request
        # -----------------------------------------------------

        clarification_signal = (
            self._find_pattern(
                normalized,
                self.CLARIFICATION_PATTERNS,
            )
        )

        if clarification_signal is not None:

            return LearnerTurnIntentResult(
                intent=(
                    self.INTENT_CLARIFICATION_QUESTION
                ),
                reason=(
                    "Learner input explicitly requests "
                    "clarification or further explanation."
                ),
                confidence=0.95,
                signals=(
                    (
                        "clarification:"
                        f"{clarification_signal}"
                    ),
                ),
                is_question=True,
            )

        # -----------------------------------------------------
        # General question detection
        # -----------------------------------------------------

        question_signals = (
            self._collect_question_signals(
                normalized
            )
        )

        if question_signals:

            return LearnerTurnIntentResult(
                intent=(
                    self.INTENT_FOLLOW_UP_QUESTION
                ),
                reason=(
                    "Learner input contains deterministic "
                    "question-form signals."
                ),
                confidence=0.90,
                signals=tuple(
                    question_signals
                ),
                is_question=True,
            )

        # -----------------------------------------------------
        # Very short / low-evidence input
        # -----------------------------------------------------

        if len(normalized) <= 1:

            return LearnerTurnIntentResult(
                intent=self.INTENT_UNCERTAIN,
                reason=(
                    "Learner input is too short for "
                    "reliable deterministic intent "
                    "classification."
                ),
                confidence=0.25,
                signals=(
                    "low_evidence",
                ),
                is_question=False,
            )

        # -----------------------------------------------------
        # Statement-like fallback
        # -----------------------------------------------------

        return LearnerTurnIntentResult(
            intent=self.INTENT_ANSWER,
            reason=(
                "No deterministic question, clarification, "
                "or topic-change signal was detected."
            ),
            confidence=0.75,
            signals=(
                "statement_like",
            ),
            is_question=False,
        )

    # =========================================================
    # Internal helpers
    # =========================================================

    def _normalize(
        self,
        text: str,
    ) -> str:

        text = text.strip()

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.lower()

    def _find_pattern(
        self,
        text: str,
        patterns: tuple[str, ...],
    ) -> str | None:

        for pattern in patterns:

            if pattern in text:
                return pattern

        return None

    def _collect_question_signals(
        self,
        text: str,
    ) -> list[str]:

        signals: list[str] = []

        if (
            "?" in text
            or "？" in text
        ):

            signals.append(
                "question_mark"
            )

        for pattern in (
            self.THAI_QUESTION_PATTERNS
        ):

            if pattern in text:

                signals.append(
                    f"thai_question:{pattern}"
                )

        for prefix in (
            self.ENGLISH_QUESTION_PREFIXES
        ):

            if text.startswith(
                prefix
            ):

                signals.append(
                    (
                        "english_question:"
                        f"{prefix.strip()}"
                    )
                )

        return signals

    def _looks_like_question(
        self,
        text: str,
    ) -> bool:

        return bool(
            self._collect_question_signals(
                text
            )
        )