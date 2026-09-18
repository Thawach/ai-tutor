import re

from dataclasses import dataclass
from typing import Literal


GroundingPrecisionGuardStatus = Literal[
    "clear",
    "precision_issue",
]


@dataclass(frozen=True)
class GroundingPrecisionGuardResult:
    """
    Deterministic check for high-risk factual precision
    that should not be trusted to semantic validation alone.
    """

    status: GroundingPrecisionGuardStatus
    issues: tuple[str, ...]


class GroundingPrecisionGuard:
    """
    Detects high-risk evidence strengthening such as:

    - unsupported numerical values
    - unsupported beta / current-gain terminology
    - unsupported majority qualifiers
    - unsupported proportionality claims

    This service does NOT determine general factual truth.
    It only checks whether high-risk precision introduced
    by the Tutor has corresponding evidence in the supplied
    course knowledge.

    No LLM is used.
    """

    _NUMERIC_VALUE_PATTERN = re.compile(
        r"(?<![\w.])"
        r"(\d+(?:[.,]\d+)?)"
        r"\s*"
        r"(mv|kv|v|ma|ka|a|"
        r"ohm|kohm|mohm|"
        r"hz|khz|mhz|ghz|"
        r"w|mw|kw|"
        r"%|โวลต์|แอมป์|เฮิรตซ์)"
        r"\b",
        re.IGNORECASE,
    )

    _BETA_PATTERN = re.compile(
        r"(?<!\w)(?:beta|β)(?!\w)",
        re.IGNORECASE,
    )

    _PROPORTIONAL_PATTERNS = (
        "proportional to",
        "directly proportional",
        "แปรผันตรง",
        "เป็นสัดส่วนกับ",
    )

    _MAJORITY_GROUPS = (
        (
            "most",
            "majority",
            "nearly all",
            "ส่วนใหญ่",
            "เกือบทั้งหมด",
        ),
    )

    def evaluate(
        self,
        response: str,
        knowledge_context: str,
    ) -> GroundingPrecisionGuardResult:

        if not isinstance(response, str):
            raise TypeError(
                "response must be str."
            )

        if not isinstance(
            knowledge_context,
            str,
        ):
            raise TypeError(
                "knowledge_context must be str."
            )

        factual_text = self._factual_text(
            response
        )

        response_norm = self._normalize(
            factual_text
        )

        knowledge_norm = self._normalize(
            knowledge_context
        )

        issues: list[str] = []

        self._check_numeric_values(
            response_norm=response_norm,
            knowledge_norm=knowledge_norm,
            issues=issues,
        )

        self._check_beta(
            response_norm=response_norm,
            knowledge_norm=knowledge_norm,
            issues=issues,
        )

        self._check_majority_qualifiers(
            response_norm=response_norm,
            knowledge_norm=knowledge_norm,
            issues=issues,
        )

        self._check_proportionality(
            response_norm=response_norm,
            knowledge_norm=knowledge_norm,
            issues=issues,
        )

        unique_issues = tuple(
            dict.fromkeys(issues)
        )

        return GroundingPrecisionGuardResult(
            status=(
                "precision_issue"
                if unique_issues
                else "clear"
            ),
            issues=unique_issues,
        )

    def _check_numeric_values(
        self,
        response_norm: str,
        knowledge_norm: str,
        issues: list[str],
    ) -> None:

        response_values = (
            self._extract_numeric_values(
                response_norm
            )
        )

        knowledge_values = set(
            self._extract_numeric_values(
                knowledge_norm
            )
        )

        for value in response_values:
            if value not in knowledge_values:
                issues.append(
                    (
                        "Unsupported precise numerical "
                        f"value: {value}."
                    )
                )

    def _check_beta(
        self,
        response_norm: str,
        knowledge_norm: str,
        issues: list[str],
    ) -> None:

        if (
            self._BETA_PATTERN.search(
                response_norm
            )
            and not self._BETA_PATTERN.search(
                knowledge_norm
            )
        ):
            issues.append(
                (
                    "Unsupported beta/current-gain "
                    "terminology."
                )
            )

    def _check_majority_qualifiers(
        self,
        response_norm: str,
        knowledge_norm: str,
        issues: list[str],
    ) -> None:

        for group in self._MAJORITY_GROUPS:

            response_has = any(
                term in response_norm
                for term in group
            )

            knowledge_has = any(
                term in knowledge_norm
                for term in group
            )

            if (
                response_has
                and not knowledge_has
            ):
                issues.append(
                    (
                        "Unsupported majority or "
                        "quantity qualifier."
                    )
                )

    def _check_proportionality(
        self,
        response_norm: str,
        knowledge_norm: str,
        issues: list[str],
    ) -> None:

        response_has = any(
            term in response_norm
            for term
            in self._PROPORTIONAL_PATTERNS
        )

        knowledge_has = any(
            term in knowledge_norm
            for term
            in self._PROPORTIONAL_PATTERNS
        )

        if (
            response_has
            and not knowledge_has
        ):
            issues.append(
                (
                    "Unsupported proportionality "
                    "relationship."
                )
            )

    def _extract_numeric_values(
        self,
        text: str,
    ) -> tuple[str, ...]:

        values = []

        for match in (
            self._NUMERIC_VALUE_PATTERN
            .finditer(text)
        ):
            number = (
                match.group(1)
                .replace(",", ".")
            )

            unit = (
                match.group(2)
                .lower()
            )

            values.append(
                f"{number} {unit}"
            )

        return tuple(values)

    def _factual_text(
        self,
        text: str,
    ) -> str:
        """
        Exclude clear question-only lines so that a numerical
        value appearing only inside a guiding question is not
        treated as a factual assertion.
        """

        factual_parts = []

        for line in text.splitlines():

            stripped = line.strip()

            if not stripped:
                continue

            if stripped.endswith("?"):
                continue

            factual_parts.append(
                stripped
            )

        return "\n".join(
            factual_parts
        )

    def _normalize(
        self,
        text: str,
    ) -> str:

        normalized = (
            text
            .replace("\u00a0", " ")
            .replace("\u202f", " ")
            .replace("β", "beta")
            .lower()
        )

        normalized = re.sub(
            r"\s+",
            " ",
            normalized,
        )

        return normalized.strip()