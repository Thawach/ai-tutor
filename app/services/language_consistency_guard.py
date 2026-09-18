from dataclasses import dataclass

from app.courses.models import (
    CourseProfile,
)

from app.services.response_style_resolver import (
    ResponseStyleResolver,
)


@dataclass(frozen=True)
class LanguageConsistencyResult:
    """
    Result from deterministic language consistency detection.

    status:
    - consistent
    - mismatch
    - uncertain

    expected_language:
    - th
    - en
    - other configured language

    detected_language:
    - th
    - en
    - mixed
    - unknown
    """

    status: str
    expected_language: str
    detected_language: str
    reason: str

    thai_chars: int
    latin_chars: int


class LanguageConsistencyGuard:
    """
    Deterministic response-language detector.

    This service does NOT repair or rewrite Tutor responses.

    Its only responsibility is to detect whether the language
    of the final Tutor response is reasonably consistent with
    the active course language.

    Important design rule:

    Technical English terms inside Thai responses must not
    automatically cause a mismatch.

    Example:

        "อิมิตเตอร์ (Emitter) ทำหน้าที่อะไร?"

    should still be classified as Thai-dominant.
    """

    MIN_LANGUAGE_CHARS = 4

    STRONG_LANGUAGE_RATIO = 0.60

    SUPPORTED_LANGUAGE_CODES = {
        "th",
        "en",
    }

    def __init__(
        self,
        course_profile: CourseProfile,
    ):

        self.course_profile = (
            course_profile
        )

        self.response_style_resolver = (
            ResponseStyleResolver()
        )

        self.resolved_response_style = (
            self.response_style_resolver
            .resolve(
                course_profile
            )
        )

        self.expected_language = (
            self.resolved_response_style
            .primary_language
        )

    # =========================================================
    # Public API
    # =========================================================

    def evaluate(
        self,
        response: str,
    ) -> LanguageConsistencyResult:

        text = response.strip()

        # -----------------------------------------------------
        # Empty response
        # -----------------------------------------------------

        if not text:

            return LanguageConsistencyResult(
                status="uncertain",
                expected_language=(
                    self.expected_language
                ),
                detected_language="unknown",
                reason=(
                    "Tutor response is empty; "
                    "language cannot be determined."
                ),
                thai_chars=0,
                latin_chars=0,
            )

        # -----------------------------------------------------
        # Character counts
        # -----------------------------------------------------

        thai_chars = (
            self._count_thai_characters(
                text
            )
        )

        latin_chars = (
            self._count_latin_characters(
                text
            )
        )

        language_chars = (
            thai_chars
            +
            latin_chars
        )

        # -----------------------------------------------------
        # Too little language evidence
        # -----------------------------------------------------

        if (
            language_chars
            <
            self.MIN_LANGUAGE_CHARS
        ):

            return LanguageConsistencyResult(
                status="uncertain",
                expected_language=(
                    self.expected_language
                ),
                detected_language="unknown",
                reason=(
                    "There is not enough alphabetic "
                    "language evidence to determine "
                    "response language reliably."
                ),
                thai_chars=thai_chars,
                latin_chars=latin_chars,
            )

        # -----------------------------------------------------
        # Detect dominant language
        # -----------------------------------------------------

        detected_language = (
            self._detect_language(
                thai_chars=thai_chars,
                latin_chars=latin_chars,
            )
        )

        # -----------------------------------------------------
        # Unsupported configured language
        # -----------------------------------------------------

        if (
            self.expected_language
            not in
            self.SUPPORTED_LANGUAGE_CODES
        ):

            return LanguageConsistencyResult(
                status="uncertain",
                expected_language=(
                    self.expected_language
                ),
                detected_language=(
                    detected_language
                ),
                reason=(
                    "The active course language is not "
                    "currently supported by deterministic "
                    "language consistency detection."
                ),
                thai_chars=thai_chars,
                latin_chars=latin_chars,
            )

        # -----------------------------------------------------
        # Strong Thai response
        # -----------------------------------------------------

        if detected_language == "th":

            if (
                self.expected_language
                == "th"
            ):

                return LanguageConsistencyResult(
                    status="consistent",
                    expected_language="th",
                    detected_language="th",
                    reason=(
                        "Tutor response is predominantly "
                        "Thai and matches the active "
                        "course language."
                    ),
                    thai_chars=thai_chars,
                    latin_chars=latin_chars,
                )

            return LanguageConsistencyResult(
                status="mismatch",
                expected_language=(
                    self.expected_language
                ),
                detected_language="th",
                reason=(
                    "Tutor response is predominantly Thai "
                    "but the active course expects a "
                    "different language."
                ),
                thai_chars=thai_chars,
                latin_chars=latin_chars,
            )

        # -----------------------------------------------------
        # Strong English response
        # -----------------------------------------------------

        if detected_language == "en":

            if (
                self.expected_language
                == "en"
            ):

                return LanguageConsistencyResult(
                    status="consistent",
                    expected_language="en",
                    detected_language="en",
                    reason=(
                        "Tutor response is predominantly "
                        "English and matches the active "
                        "course language."
                    ),
                    thai_chars=thai_chars,
                    latin_chars=latin_chars,
                )

            return LanguageConsistencyResult(
                status="mismatch",
                expected_language=(
                    self.expected_language
                ),
                detected_language="en",
                reason=(
                    "Tutor response is predominantly "
                    "English but the active course "
                    "expects Thai."
                ),
                thai_chars=thai_chars,
                latin_chars=latin_chars,
            )

        # -----------------------------------------------------
        # Mixed Thai / English
        #
        # Mixed technical terminology should be allowed if the
        # expected language remains dominant.
        # -----------------------------------------------------

        if detected_language == "mixed":

            if (
                self.expected_language
                == "th"
                and
                thai_chars >= latin_chars
            ):

                return LanguageConsistencyResult(
                    status="consistent",
                    expected_language="th",
                    detected_language="mixed",
                    reason=(
                        "Tutor response contains mixed Thai "
                        "and English terminology, but Thai "
                        "remains dominant."
                    ),
                    thai_chars=thai_chars,
                    latin_chars=latin_chars,
                )

            if (
                self.expected_language
                == "en"
                and
                latin_chars >= thai_chars
            ):

                return LanguageConsistencyResult(
                    status="consistent",
                    expected_language="en",
                    detected_language="mixed",
                    reason=(
                        "Tutor response contains mixed "
                        "language terminology, but English "
                        "remains dominant."
                    ),
                    thai_chars=thai_chars,
                    latin_chars=latin_chars,
                )

            return LanguageConsistencyResult(
                status="uncertain",
                expected_language=(
                    self.expected_language
                ),
                detected_language="mixed",
                reason=(
                    "Tutor response contains substantial "
                    "Thai and English content and no safe "
                    "language mismatch decision can be made."
                ),
                thai_chars=thai_chars,
                latin_chars=latin_chars,
            )

        # -----------------------------------------------------
        # Unknown
        # -----------------------------------------------------

        return LanguageConsistencyResult(
            status="uncertain",
            expected_language=(
                self.expected_language
            ),
            detected_language="unknown",
            reason=(
                "Tutor response language could not be "
                "determined reliably."
            ),
            thai_chars=thai_chars,
            latin_chars=latin_chars,
        )

    # =========================================================
    # Language detection helpers
    # =========================================================

    def _detect_language(
        self,
        thai_chars: int,
        latin_chars: int,
    ) -> str:

        total = (
            thai_chars
            +
            latin_chars
        )

        if total <= 0:

            return "unknown"

        thai_ratio = (
            thai_chars
            /
            total
        )

        latin_ratio = (
            latin_chars
            /
            total
        )

        if (
            thai_ratio
            >=
            self.STRONG_LANGUAGE_RATIO
        ):

            return "th"

        if (
            latin_ratio
            >=
            self.STRONG_LANGUAGE_RATIO
        ):

            return "en"

        return "mixed"

    def _count_thai_characters(
        self,
        text: str,
    ) -> int:

        return sum(
            1
            for char in text
            if (
                "\u0E00"
                <= char
                <= "\u0E7F"
            )
        )

    def _count_latin_characters(
        self,
        text: str,
    ) -> int:

        return sum(
            1
            for char in text
            if (
                "A" <= char <= "Z"
                or
                "a" <= char <= "z"
            )
        )

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