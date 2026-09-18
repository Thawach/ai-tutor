import re

from app.services.response_quality_models import (
    ResponseQualityResult,
)


class ResponseQualityGuard:
    """
    Deterministic final-response quality detector.

    Step 16.17.3A is telemetry-only.

    This guard does not:
    - call an LLM
    - repair responses
    - perform semantic grounding
    - validate scaffolding strategy compliance
    - validate language consistency
    """

    DEFAULT_MAX_CHARACTERS = 1200

    DEFAULT_MIN_CHARACTERS = 3

    DEFAULT_MAX_QUESTIONS = 2

    DEFAULT_DETECT_REPETITION = True

    def __init__(
        self,
        max_characters: int = DEFAULT_MAX_CHARACTERS,
        min_characters: int = DEFAULT_MIN_CHARACTERS,
        max_questions: int = DEFAULT_MAX_QUESTIONS,
        detect_repetition: bool = DEFAULT_DETECT_REPETITION,
    ):

        self.max_characters = (
            max_characters
        )

        self.min_characters = (
            min_characters
        )

        self.max_questions = (
            max_questions
        )

        self.detect_repetition = (
            detect_repetition
        )

    # =========================================================
    # Public API
    # =========================================================

    def evaluate(
        self,
        response: str,
    ) -> ResponseQualityResult:

        text = (
            response or ""
        ).strip()

        if not text:

            return ResponseQualityResult(
                status="uncertain",
                reason=(
                    "Tutor response is empty; "
                    "response quality cannot be "
                    "assessed reliably."
                ),
                issues=[
                    "Empty tutor response."
                ],
            )

        character_count = len(
            text
        )

        token_like_count = (
            self._count_token_like_units(
                text
            )
        )

        question_count = (
            self._count_questions(
                text
            )
        )

        sentence_count = (
            self._count_sentences(
                text
            )
        )

        too_long = (
            character_count
            > self.max_characters
        )

        too_short = (
            character_count
            < self.min_characters
        )

        too_many_questions = (
            question_count
            > self.max_questions
        )

        repetitive = (
            self.detect_repetition
            and
            self._detect_repetition(
                text
            )
        )

        issues = []

        if too_long:

            issues.append(
                "Tutor response exceeds the configured "
                "maximum response length."
            )

        if too_short:

            issues.append(
                "Tutor response is unusually short."
            )

        if too_many_questions:

            issues.append(
                "Tutor response contains more questions "
                "than the configured quality threshold."
            )

        if repetitive:

            issues.append(
                "Tutor response contains suspicious "
                "repetition."
            )

        if issues:

            return ResponseQualityResult(
                status="warning",
                reason=(
                    "One or more deterministic response "
                    "quality issues were detected."
                ),
                issues=issues,
                character_count=character_count,
                token_like_count=token_like_count,
                question_count=question_count,
                sentence_count=sentence_count,
                too_long=too_long,
                too_short=too_short,
                too_many_questions=(
                    too_many_questions
                ),
                repetitive=repetitive,
            )

        return ResponseQualityResult(
            status="acceptable",
            reason=(
                "No deterministic response-form "
                "quality issue was detected."
            ),
            issues=[],
            character_count=character_count,
            token_like_count=token_like_count,
            question_count=question_count,
            sentence_count=sentence_count,
            too_long=False,
            too_short=False,
            too_many_questions=False,
            repetitive=False,
        )

    # =========================================================
    # Counting helpers
    # =========================================================

    def _count_token_like_units(
        self,
        text: str,
    ) -> int:

        units = re.findall(
            r"\S+",
            text,
        )

        return len(
            units
        )

    def _count_questions(
        self,
        text: str,
    ) -> int:

        return text.count(
            "?"
        ) + text.count(
            "？"
        )

    def _count_sentences(
        self,
        text: str,
    ) -> int:

        parts = re.split(
            r"[.!?。！？]+",
            text,
        )

        return sum(
            1
            for part in parts
            if part.strip()
        )

    # =========================================================
    # Repetition detection
    # =========================================================

    def _detect_repetition(
        self,
        text: str,
    ) -> bool:

        normalized = (
            re.sub(
                r"\s+",
                " ",
                text.lower(),
            )
            .strip()
        )

        if not normalized:

            return False

        sentences = [
            sentence.strip()
            for sentence in re.split(
                r"[.!?。！？]+",
                normalized,
            )
            if sentence.strip()
        ]

        if len(sentences) >= 2:

            seen = set()

            for sentence in sentences:

                if sentence in seen:

                    return True

                seen.add(
                    sentence
                )

        words = re.findall(
            r"\S+",
            normalized,
        )

        if len(words) < 8:

            return False

        # Detect repeated adjacent 4-token sequences.
        window_size = 4

        windows = []

        for index in range(
            len(words)
            - window_size
            + 1
        ):

            windows.append(
                tuple(
                    words[
                        index:
                        index + window_size
                    ]
                )
            )

        for index in range(
            len(windows) - 1
        ):

            if (
                windows[index]
                ==
                windows[index + 1]
            ):

                return True

        return False