from dataclasses import dataclass

from app.courses.models import (
    CourseProfile,
)


@dataclass(frozen=True)
class ResolvedResponseStyle:
    """
    Fully resolved Tutor response style.

    This object contains effective values that are safe for
    downstream services to consume.

    The resolver does not mutate CourseProfile or
    ResponseStyleProfile.
    """

    primary_language: str

    allow_technical_english: bool

    technical_term_format: str

    tone: str

    explanation_depth: str

    question_style: str

    max_guiding_questions: int


class ResponseStyleResolver:
    """
    Resolve course-level response style configuration.

    Resolution rules:

    1. response_style.primary_language has precedence when
       explicitly configured with a non-empty value.

    2. If primary_language is empty, inherit course.language.

    3. Other response style values come from the typed
       ResponseStyleProfile.

    4. The resolver does not modify Tutor behavior directly.

    5. The resolver does not mutate the source CourseProfile.
    """

    def resolve(
        self,
        course_profile: CourseProfile,
    ) -> ResolvedResponseStyle:

        style = (
            course_profile.response_style
        )

        primary_language = (
            self._resolve_primary_language(
                course_profile=course_profile,
            )
        )

        return ResolvedResponseStyle(
            primary_language=(
                primary_language
            ),
            allow_technical_english=(
                style.allow_technical_english
            ),
            technical_term_format=(
                style.technical_term_format
            ),
            tone=(
                style.tone
            ),
            explanation_depth=(
                style.explanation_depth
            ),
            question_style=(
                style.question_style
            ),
            max_guiding_questions=(
                style.max_guiding_questions
            ),
        )

    # =========================================================
    # Resolution helpers
    # =========================================================

    def _resolve_primary_language(
        self,
        course_profile: CourseProfile,
    ) -> str:

        style_language = (
            course_profile
            .response_style
            .primary_language
            .strip()
            .lower()
        )

        course_language = (
            course_profile
            .language
            .strip()
            .lower()
        )

        if style_language:

            return (
                self._normalize_language_code(
                    style_language
                )
            )

        if course_language:

            return (
                self._normalize_language_code(
                    course_language
                )
            )

        # Fail-safe default for legacy or malformed in-memory
        # CourseProfile instances.
        return "th"

    def _normalize_language_code(
        self,
        language: str,
    ) -> str:

        value = (
            language
            .strip()
            .lower()
        )

        aliases = {
            "thai": "th",
            "th-th": "th",
            "english": "en",
            "en-us": "en",
            "en-gb": "en",
        }

        return aliases.get(
            value,
            value,
        )