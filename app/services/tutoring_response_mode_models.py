from dataclasses import dataclass
from typing import Literal


TutoringResponseMode = Literal[
    "scaffolded",
    "answer_then_guide",
    "direct_clarification",
    "new_topic_scaffold",
]


GuidingQuestionPolicy = Literal[
    "required",
    "optional",
    "forbidden",
]


_VALID_RESPONSE_MODES = frozenset(
    {
        "scaffolded",
        "answer_then_guide",
        "direct_clarification",
        "new_topic_scaffold",
    }
)


_VALID_GUIDING_QUESTION_POLICIES = frozenset(
    {
        "required",
        "optional",
        "forbidden",
    }
)


@dataclass(frozen=True)
class TutoringResponseModeDecision:
    """
    Deterministic response-mode decision for one learner turn.

    This model contains no LLM logic and does not generate
    Tutor responses.

    mode:
    - scaffolded
        Use the normal scaffolding strategy.

    - answer_then_guide
        Answer the learner's question first, then optionally
        continue with a guiding question.

    - direct_clarification
        Give a direct, concise clarification. A Socratic
        guiding question must not be required.

    - new_topic_scaffold
        Start a fresh pedagogical sequence for a new topic.

    guiding_question_policy:
    - required
        The final Tutor response must contain a guiding question.

    - optional
        A guiding question may be used but is not mandatory.

    - forbidden
        No guiding question should be added.
    """

    mode: TutoringResponseMode

    reason: str

    direct_answer_required: bool

    guiding_question_policy: GuidingQuestionPolicy

    preserve_scaffolding_strategy: bool

    max_guiding_questions: int | None = None

    def __post_init__(self) -> None:

        if self.mode not in _VALID_RESPONSE_MODES:
            raise ValueError(
                "Unsupported tutoring response mode: "
                f"{self.mode}"
            )

        if not isinstance(
            self.reason,
            str,
        ):
            raise TypeError(
                "reason must be a string."
            )

        if not self.reason.strip():
            raise ValueError(
                "reason must not be empty."
            )

        if not isinstance(
            self.direct_answer_required,
            bool,
        ):
            raise TypeError(
                "direct_answer_required must be bool."
            )

        if (
            self.guiding_question_policy
            not in
            _VALID_GUIDING_QUESTION_POLICIES
        ):
            raise ValueError(
                "Unsupported guiding question policy: "
                f"{self.guiding_question_policy}"
            )

        if not isinstance(
            self.preserve_scaffolding_strategy,
            bool,
        ):
            raise TypeError(
                "preserve_scaffolding_strategy "
                "must be bool."
            )

        if (
            self.max_guiding_questions
            is not None
        ):

            if (
                isinstance(
                    self.max_guiding_questions,
                    bool,
                )
                or not isinstance(
                    self.max_guiding_questions,
                    int,
                )
            ):
                raise TypeError(
                    "max_guiding_questions must be "
                    "an integer or None."
                )

            if self.max_guiding_questions < 0:
                raise ValueError(
                    "max_guiding_questions cannot "
                    "be negative."
                )

        if (
            self.guiding_question_policy
            == "forbidden"
            and self.max_guiding_questions
            not in (
                None,
                0,
            )
        ):
            raise ValueError(
                "A forbidden guiding-question policy "
                "cannot allow guiding questions."
            )

        if (
            self.guiding_question_policy
            == "required"
            and self.max_guiding_questions
            == 0
        ):
            raise ValueError(
                "A required guiding-question policy "
                "cannot have a zero question limit."
            )