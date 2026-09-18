import re
from dataclasses import dataclass

from app.courses.models import (
    CourseProfile,
)


@dataclass(frozen=True)
class DeterministicRepairResult:
    """
    Result of deterministic response repair.

    repaired:
    - True  = at least one deterministic replacement occurred
    - False = no deterministic replacement was possible
    """

    response: str
    repaired: bool
    replacements: list[tuple[str, str]]


class DeterministicResponseRepair:
    """
    Perform safe deterministic response repairs.

    Version 1 supports configured technical terminology
    replacement only.

    Domain-specific mappings come from CourseProfile.
    No LLM calls are performed.
    """

    def __init__(
        self,
        course_profile: CourseProfile | None = None,
    ):

        self.course_profile = (
            course_profile
        )

        if course_profile is None:

            self.technical_terms = {}

        else:

            self.technical_terms = dict(
                course_profile.technical_terms
            )

    def repair_terminology(
        self,
        response: str,
    ) -> DeterministicRepairResult:

        repaired_response = response

        replacements = []

        if not response.strip():

            return DeterministicRepairResult(
                response=response,
                repaired=False,
                replacements=[],
            )

        for (
            suspicious_term,
            preferred_term,
        ) in self.technical_terms.items():

            if not suspicious_term:
                continue

            if (
                suspicious_term.lower()
                not in repaired_response.lower()
            ):
                continue

            repaired_response = (
                self._replace_case_insensitive(
                    text=repaired_response,
                    old=suspicious_term,
                    new=preferred_term,
                )
            )

            repaired_response = self._normalize_spacing(
                repaired_response
            )

            replacements.append(
                (
                    suspicious_term,
                    preferred_term,
                )
            )

        return DeterministicRepairResult(
            response=repaired_response,
            repaired=bool(replacements),
            replacements=replacements,
        )

    def _normalize_spacing(
        self,
        text: str,
    ) -> str:
        """
        Normalize spacing introduced by deterministic
        terminology replacement.

        If a replacement ends with ')' and the next
        character immediately continues with Thai,
        English, or numeric text, insert one space.
        """

        return re.sub(
            r"\)(?=[\u0E00-\u0E7FA-Za-z0-9])",
            ") ",
            text,
        )

    def _replace_case_insensitive(
        self,
        text: str,
        old: str,
        new: str,
    ) -> str:
        """
        Replace all occurrences without changing
        unrelated text.

        This implementation avoids regex interpretation
        of technical terms.
        """

        lowered_text = text.lower()
        lowered_old = old.lower()

        result = []
        start = 0

        while True:

            index = lowered_text.find(
                lowered_old,
                start,
            )

            if index == -1:

                result.append(
                    text[start:]
                )

                break

            result.append(
                text[start:index]
            )

            result.append(
                new
            )

            start = (
                index + len(old)
            )

        return "".join(result)