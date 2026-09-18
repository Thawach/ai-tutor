from app.services.pedagogical_precheck import (
    PedagogicalPrecheck,
    PedagogicalPrecheckResult,
)

from app.services.tutoring_response_mode_models import (
    TutoringResponseModeDecision,
)


class ResponseModePedagogicalPrecheck:
    """
    Deterministic pedagogical precheck that understands
    turn-specific tutoring response modes.

    Existing scaffolding behavior is delegated unchanged
    to PedagogicalPrecheck when the response mode preserves
    the normal scaffolding strategy.

    No LLM is used here.
    """

    _QUESTION_ONLY_PREFIXES = (
        # Thai
        "คุณคิดว่า",
        "คุณคิดอย่างไร",
        "ลองคิดว่า",
        "ลองพิจารณาว่า",
        "ลองอธิบายว่า",
        "ลองบอกว่า",

        # English
        "what ",
        "why ",
        "how ",
        "when ",
        "where ",
        "which ",
        "who ",
        "do you ",
        "does ",
        "did ",
        "can you ",
        "could you ",
        "would you ",
        "what do you think",
    )

    def __init__(
        self,
        base_precheck: PedagogicalPrecheck | None = None,
    ) -> None:

        self.base_precheck = (
            base_precheck
            if base_precheck is not None
            else PedagogicalPrecheck()
        )

    def evaluate(
        self,
        response: str,
        strategy_name: str,
        response_mode: TutoringResponseModeDecision,
    ) -> PedagogicalPrecheckResult:

        if not isinstance(
            response,
            str,
        ):
            raise TypeError(
                "response must be a string."
            )

        if not isinstance(
            strategy_name,
            str,
        ):
            raise TypeError(
                "strategy_name must be a string."
            )

        if not isinstance(
            response_mode,
            TutoringResponseModeDecision,
        ):
            raise TypeError(
                "response_mode must be "
                "TutoringResponseModeDecision."
            )

        # -------------------------------------------------
        # Existing scaffolding remains authoritative.
        # -------------------------------------------------

        if response_mode.preserve_scaffolding_strategy:

            return self.base_precheck.evaluate(
                response=response,
                strategy_name=strategy_name,
            )

        text = (
            response
            .replace("？", "?")
            .strip()
        )

        # -------------------------------------------------
        # Empty Tutor response
        # -------------------------------------------------

        if not text:

            return PedagogicalPrecheckResult(
                status="violation",
                reason=(
                    "Response-mode validation requires "
                    "a non-empty Tutor response."
                ),
                issues=[
                    "Tutor response is empty.",
                ],
            )

        question_count = (
            text.count("?")
        )

        # -------------------------------------------------
        # Maximum guiding-question limit
        # -------------------------------------------------

        max_questions = (
            response_mode.max_guiding_questions
        )

        if (
            max_questions is not None
            and question_count > max_questions
        ):

            return PedagogicalPrecheckResult(
                status="violation",
                reason=(
                    "Tutor response exceeds the "
                    "response-mode guiding-question "
                    "limit."
                ),
                issues=[
                    (
                        "Too many guiding questions "
                        f"for mode '{response_mode.mode}'."
                    ),
                ],
            )

        # -------------------------------------------------
        # Response modes that require an answer first
        # -------------------------------------------------

        if response_mode.direct_answer_required:

            # No question at all:
            # structurally this is a direct response.
            if question_count == 0:

                return PedagogicalPrecheckResult(
                    status="valid",
                    reason=(
                        "A direct response without a "
                        "guiding question satisfies the "
                        "current response mode."
                    ),
                    issues=[],
                )

            # Clear question-only response.
            if self._is_clear_question_only(
                text
            ):

                return PedagogicalPrecheckResult(
                    status="violation",
                    reason=(
                        "The current response mode "
                        "requires an answer or "
                        "clarification before any "
                        "guiding question."
                    ),
                    issues=[
                        (
                            "Question-only response "
                            "detected where a direct "
                            "answer is required."
                        ),
                    ],
                )

            # Clear explanation before the optional question.
            if self._has_clear_content_before_question(
                text
            ):

                return PedagogicalPrecheckResult(
                    status="valid",
                    reason=(
                        "A direct response followed by "
                        "an optional guiding question "
                        "was detected."
                    ),
                    issues=[],
                )

            # The structure is ambiguous.
            # Semantic validator will decide in 16.19.5B.
            return PedagogicalPrecheckResult(
                status="uncertain",
                reason=(
                    "A question is present, but the "
                    "deterministic precheck cannot "
                    "reliably determine whether a "
                    "direct answer precedes it."
                ),
                issues=[],
            )

        # -------------------------------------------------
        # Conservative fail-safe for future
        # non-preserving modes.
        # -------------------------------------------------

        return PedagogicalPrecheckResult(
            status="uncertain",
            reason=(
                "The non-preserving response mode "
                "does not define a deterministic "
                "precheck rule."
            ),
            issues=[],
        )

    def _is_clear_question_only(
        self,
        text: str,
    ) -> bool:

        normalized = (
            " ".join(
                text.split()
            )
            .strip()
            .lower()
        )

        if not normalized.endswith("?"):
            return False

        if normalized.count("?") != 1:
            return False

        return any(
            normalized.startswith(prefix)
            for prefix in self._QUESTION_ONLY_PREFIXES
        )

    def _has_clear_content_before_question(
        self,
        text: str,
    ) -> bool:

        question_index = (
            text.find("?")
        )

        if question_index <= 0:
            return False

        before_question = (
            text[:question_index]
            .strip()
        )

        # A sentence or paragraph boundary before the
        # question is strong deterministic evidence that
        # explanatory content precedes the question.
        return (
            "." in before_question
            or "!" in before_question
            or "\n" in before_question
        )