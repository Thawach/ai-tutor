import re
from dataclasses import dataclass


@dataclass(frozen=True)
class GroundingClaimPrecheckResult:
    """
    Deterministic precheck result for tutor-response grounding.

    status:
    - no_claims
    - needs_validation
    """

    status: str
    reason: str


class GroundingClaimPrecheck:
    """
    Conservative deterministic fast-path for grounding validation.

    The purpose of this class is NOT to determine whether a factual
    claim is true or false.

    It only decides whether the response appears to contain a factual
    assertion that should be sent to the semantic grounding validator.

    Design principle:

    - Clear question with no embedded factual assertion
      -> no_claims

    - Instruction / encouragement with no factual assertion
      -> no_claims

    - Declarative factual statement
      -> needs_validation

    - Question containing an embedded factual assertion
      -> needs_validation

    - Ambiguous response
      -> needs_validation

    When uncertain, fail safely by returning needs_validation.
    """

    # =========================================================
    # Question / inquiry markers
    # =========================================================

    QUESTION_STARTERS = (
        # Thai
        "คุณคิดว่า",
        "คุณคิดอย่างไร",
        "คุณสังเกต",
        "คุณจะ",
        "คุณสามารถ",
        "เพราะอะไร",
        "ทำไม",
        "อย่างไร",
        "อะไร",
        "ข้อใด",
        "ส่วนใด",
        "ชั้นใด",
        "ตัวใด",
        "ลองคิดว่า",
        "ลองคิดดูว่า",
        "ลองพิจารณาว่า",
        "ลองสังเกตว่า",
        "ลองนึกว่า",
        "ลองนึกถึง",

        # English
        "what ",
        "why ",
        "how ",
        "which ",
        "where ",
        "when ",
        "do you ",
        "does ",
        "can you ",
        "could you ",
        "would you ",
        "what do you think",
        "think about ",
        "consider whether ",
    )

    # =========================================================
    # Non-factual instructional / pedagogical starters
    # =========================================================

    NON_FACTUAL_DIRECTIVE_STARTERS = (
        # Thai
        "ลองศึกษา",
        "ลองทบทวน",
        "ลองพิจารณา",
        "ลองสังเกต",
        "ลองนึกถึง",
        "ลองคิดถึง",
        "ลองเปรียบเทียบ",
        "ลองตรวจสอบ",
        "ลองอธิบาย",
        "ลองตอบ",
        "กรุณาศึกษา",
        "กรุณาทบทวน",

        # English
        "review ",
        "study ",
        "consider ",
        "think about ",
        "look at ",
        "focus on ",
        "try to ",
        "try reviewing ",
        "try considering ",
    )

    # =========================================================
    # Strong factual-assertion patterns
    #
    # These patterns intentionally focus on assertion forms
    # rather than every possible technical phrase.
    #
    # Course-specific embedded claims are additionally handled
    # by EmbeddedClaimTermGuard.
    # =========================================================

    FACTUAL_PATTERNS = (

        # -----------------------------------------------------
        # Thai: definitions / classifications
        # -----------------------------------------------------

        re.compile(
            r"\bคือ\b",
            re.IGNORECASE,
        ),

        re.compile(
            r"\bหมายถึง\b",
            re.IGNORECASE,
        ),

        re.compile(
            r"\bประกอบด้วย\b",
            re.IGNORECASE,
        ),

        re.compile(
            r"\bมีโครงสร้าง\b",
            re.IGNORECASE,
        ),

        re.compile(
            r"\bทำหน้าที่เป็น\b",
            re.IGNORECASE,
        ),

        re.compile(
            r"\bเกิดจาก\b",
            re.IGNORECASE,
        ),

        re.compile(
            r"\bส่งผลให้\b",
            re.IGNORECASE,
        ),

        re.compile(
            r"\bเท่ากับ\b",
            re.IGNORECASE,
        ),

        # -----------------------------------------------------
        # Thai + technical classification
        # -----------------------------------------------------

        re.compile(
            r"เป็น\s+"
            r"(?:n[\-\s]?type|p[\-\s]?type)",
            re.IGNORECASE,
        ),

        re.compile(
            r"คือ\s+"
            r"(?:n\s*[-–—]\s*p\s*[-–—]\s*n"
            r"|p\s*[-–—]\s*n\s*[-–—]\s*p)",
            re.IGNORECASE,
        ),

        # -----------------------------------------------------
        # English: factual assertions
        # -----------------------------------------------------

        re.compile(
            r"\bis\s+(?:an?\s+)?"
            r"(?:n[\-\s]?type|p[\-\s]?type)\b",
            re.IGNORECASE,
        ),

        re.compile(
            r"\bare\s+(?:n[\-\s]?type|p[\-\s]?type)\b",
            re.IGNORECASE,
        ),

        re.compile(
            r"\bconsists?\s+of\b",
            re.IGNORECASE,
        ),

        re.compile(
            r"\bis\s+composed\s+of\b",
            re.IGNORECASE,
        ),

        re.compile(
            r"\bhas\s+(?:an?\s+)?"
            r"(?:n\s*[-–—]\s*p\s*[-–—]\s*n"
            r"|p\s*[-–—]\s*n\s*[-–—]\s*p)"
            r"\s+structure\b",
            re.IGNORECASE,
        ),

        re.compile(
            r"\bmeans\b",
            re.IGNORECASE,
        ),

        re.compile(
            r"\bequals?\b",
            re.IGNORECASE,
        ),

        re.compile(
            r"\bis\s+caused\s+by\b",
            re.IGNORECASE,
        ),

        re.compile(
            r"\bresults?\s+in\b",
            re.IGNORECASE,
        ),
    )

    # =========================================================
    # Main evaluation
    # =========================================================

    def evaluate(
        self,
        response: str,
    ) -> GroundingClaimPrecheckResult:

        text = self._normalize_text(
            response
        )

        # -----------------------------------------------------
        # Empty response
        # -----------------------------------------------------

        if not text:

            return GroundingClaimPrecheckResult(
                status="needs_validation",
                reason=(
                    "Tutor response is empty and cannot be "
                    "safely classified as claim-free."
                ),
            )

        # -----------------------------------------------------
        # Strong factual assertion detection
        #
        # Do this BEFORE question/directive fast-paths because
        # factual claims can be embedded inside questions.
        #
        # Example:
        # "ลองสังเกตว่า NPN มีโครงสร้าง N-P-N แล้วชั้นกลางคืออะไร?"
        # -----------------------------------------------------

        factual_match = (
            self._find_factual_assertion(
                text
            )
        )

        if factual_match is not None:

            return GroundingClaimPrecheckResult(
                status="needs_validation",
                reason=(
                    "A possible factual assertion was detected "
                    "and requires semantic grounding validation: "
                    f"'{factual_match}'."
                ),
            )

        # -----------------------------------------------------
        # Pure question fast-path
        # -----------------------------------------------------

        if self._is_clear_question(
            text
        ):

            return GroundingClaimPrecheckResult(
                status="no_claims",
                reason=(
                    "A question without a detected factual "
                    "assertion was found."
                ),
            )

        # -----------------------------------------------------
        # Non-factual instructional / encouragement fast-path
        # -----------------------------------------------------

        if self._is_non_factual_directive(
            text
        ):

            return GroundingClaimPrecheckResult(
                status="no_claims",
                reason=(
                    "An instructional or reflective prompt "
                    "without a detected factual assertion "
                    "was found."
                ),
            )

        # -----------------------------------------------------
        # Conservative fallback
        # -----------------------------------------------------

        return GroundingClaimPrecheckResult(
            status="needs_validation",
            reason=(
                "The response is not clearly claim-free; "
                "semantic grounding validation is required."
            ),
        )

    # =========================================================
    # Helpers
    # =========================================================

    def _normalize_text(
        self,
        text: str,
    ) -> str:

        return (
            text
            .replace("？", "?")
            .replace("。", ".")
            .strip()
        )

    def _find_factual_assertion(
        self,
        text: str,
    ) -> str | None:

        for pattern in self.FACTUAL_PATTERNS:

            match = pattern.search(
                text
            )

            if match:

                return match.group(0)

        return None

    def _is_clear_question(
        self,
        text: str,
    ) -> bool:
        """
        Return True only when the whole response can be
        safely treated as a question-only response.

        A factual or explanatory statement followed by a
        guiding question must NOT use the no-claims fast-path.
        """

        if text.count("?") != 1:
            return False

        if not text.endswith("?"):
            return False

        lowered = text.lower()

        # -------------------------------------------------
        # Question starter at the beginning has precedence.
        #
        # Example:
        #
        # "What does the base do?"
        #
        # contains both "what " and "does ", but the latter
        # must not be mistaken for a second question clause.
        # -------------------------------------------------

        if any(
            lowered.startswith(
                starter.lower()
            )
            for starter in self.QUESTION_STARTERS
        ):
            return True

        # -------------------------------------------------
        # Mixed response guard
        #
        # Examples:
        #
        # "The base controls current.
        #  What happens next?"
        #
        # "เบสควบคุมกระแส
        #  คุณคิดว่าจะเกิดอะไรขึ้น?"
        #
        # These are not question-only responses.
        # -------------------------------------------------

        if self._has_content_before_question_clause(
            text
        ):
            return False

        # -------------------------------------------------
        # Conservative single-question fallback.
        #
        # Examples:
        #
        # "Base current?"
        # "แรงดันเบส?"
        #
        # No separate preceding response segment was found.
        # -------------------------------------------------

        return True

    def _has_content_before_question_clause(
        self,
        text: str,
    ) -> bool:
        """
        Detect a response that contains meaningful content
        before its final question clause.

        This guard is intentionally conservative.

        It does NOT decide whether the preceding content is
        factually correct. It only prevents a mixed
        explanation-plus-question response from being
        classified as claim-free.
        """

        if not text.endswith("?"):
            return False

        body = text[:-1].strip()

        if not body:
            return False

        # -------------------------------------------------
        # Explicit sentence / paragraph before question
        # -------------------------------------------------

        for boundary in (
            "\n",
            ".",
            "!",
        ):

            if boundary not in body:
                continue

            before, after = body.rsplit(
                boundary,
                1,
            )

            if (
                before.strip()
                and after.strip()
            ):
                return True

        # -------------------------------------------------
        # Question clause appears after leading content
        #
        # Examples:
        #
        # "The base controls current, what happens next?"
        #
        # "เบสควบคุมกระแส คุณคิดว่าจะเกิดอะไรขึ้น?"
        #
        # A topic prefix such as:
        #
        # "In an NPN transistor, what does the base do?"
        #
        # may also be routed to semantic validation.
        # That is intentional: false-negative grounding is
        # more costly than one conservative validator call.
        # -------------------------------------------------

        lowered = body.lower()

        for starter in self.QUESTION_STARTERS:

            marker = (
                starter
                .strip()
                .lower()
            )

            if not marker:
                continue

            index = lowered.find(
                marker
            )

            if index <= 0:
                continue

            prefix = body[
                :index
            ].strip()

            if not prefix:
                continue

            previous_character = (
                lowered[
                    index - 1
                ]
            )

            if (
                previous_character.isspace()
                or previous_character
                in ",;:—–-"
            ):
                return True

        return False

    def _is_non_factual_directive(
        self,
        text: str,
    ) -> bool:

        lowered = text.lower()

        return any(
            lowered.startswith(
                starter.lower()
            )
            for starter
            in self.NON_FACTUAL_DIRECTIVE_STARTERS
        )