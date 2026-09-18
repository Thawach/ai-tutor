import json
import re

from app.ai import (
    chat_with_structured_ai,
)

from app.services.grounding_validation_models import (
    GroundingValidationResult,
)

from app.services.grounding_claim_precheck import (
    GroundingClaimPrecheck,
)


GROUNDING_VALIDATION_PROMPT = """
You are a strict grounding validator for a course-based
AI tutoring system.

Your task is to determine whether every FACTUAL CLAIM in
the AI Tutor response is supported by the supplied
COURSE KNOWLEDGE.

Evaluate factual grounding only.

Do NOT evaluate whether the response is pedagogically good.
Do NOT answer the learner's question yourself.


============================================================
CORE EVIDENCE PRINCIPLE
============================================================

A factual claim is SUPPORTED only when it is entailed by
the supplied COURSE KNOWLEDGE.

"Entailed" means that the claim is:

- explicitly stated in the course knowledge, or
- a conservative semantic paraphrase of information that
  is explicitly stated.

A claim is NOT supported merely because it is:

- plausible,
- scientifically familiar,
- normally true in the subject,
- compatible with the course knowledge,
- common textbook knowledge, or
- known from your own training data.

Use ONLY the supplied COURSE KNOWLEDGE as evidence.

Do not use outside knowledge even when you are confident
that the outside knowledge is correct.


============================================================
DO NOT STRENGTHEN THE SOURCE
============================================================

Do not convert a weaker source statement into a stronger
Tutor claim.

Examples of invalid strengthening include:

- "related to" does NOT automatically mean
  "proportional to"

- "associated with" does NOT automatically mean
  "causes"

- "depends on X" does NOT automatically mean
  "is controlled by Y"

- a qualitative relationship does NOT support an exact
  numerical value or threshold

- a stated difference does NOT justify additional
  comparative labels such as "higher", "lower",
  "heavily", or "lightly" unless the comparison is
  actually supported

- "some" or "a number of" does NOT justify "most",
  "nearly all", or another quantitative qualifier

- the existence of a component does NOT automatically
  establish an unstated function of that component

- a diagram or named structure does NOT automatically
  establish an unstated causal mechanism


============================================================
CLAIM-BY-CLAIM REQUIREMENT
============================================================

Evaluate EACH factual claim independently.

For every factual claim, ask:

1. What exactly does the Tutor assert?
2. What sentence, equation, figure description, or
   semantically equivalent statement in COURSE KNOWLEDGE
   supports that assertion?
3. Does accepting the Tutor claim require adding any
   technical fact, quantitative relationship, causal
   relationship, comparison, mechanism, or assumption that
   is not present in COURSE KNOWLEDGE?

If step 3 requires additional information, that portion of
the Tutor response is not fully supported.

============================================================
ATOMIC CLAIM EVIDENCE OUTPUT
============================================================

Decompose the AI Tutor response into atomic factual claims.

An atomic factual claim should contain only one independently
checkable factual assertion.

If one sentence contains multiple factual assertions, create
separate claim entries when necessary.

For EVERY atomic factual claim return:

- claim:
  a concise statement of the factual claim.

- response_quote:
  the shortest exact excerpt copied from the AI TUTOR RESPONSE
  that contains the claim.

- status:
  either supported or unsupported.

- evidence_quote:
  when supported, copy the shortest exact excerpt from
  COURSE KNOWLEDGE that supports the claim.

  The evidence_quote MUST be copied from COURSE KNOWLEDGE.
  Do not paraphrase it.
  Do not manufacture or reconstruct evidence.

  When no adequate supporting excerpt exists, classify the
  claim as unsupported and return an empty evidence_quote.

- issue:
  empty when the claim is supported.
  When unsupported, briefly state what part lacks evidence.

A weaker evidence statement cannot support a stronger claim.

In particular, an evidence quote containing only "related to"
does not support a Tutor claim asserting "controlled by",
"governed by", or "determined by".

Do not omit a factual claim merely because another claim in
the same sentence is supported.

If the Tutor response contains no factual claims, return an
empty claims array.

STRICT ATOMIC CLAIM OUTPUT CONTRACT:

Every item in "claims" MUST contain ALL of these fields:

- "claim"
- "response_quote"
- "status"
- "evidence_quote"
- "issue"

Never omit any field.

For a supported claim:
- status = "supported"
- evidence_quote MUST contain an exact excerpt from COURSE KNOWLEDGE
- issue MUST be exactly ""

For an unsupported claim:
- status = "unsupported"
- evidence_quote MUST be exactly ""
- issue MUST be a non-empty explanation of why the claim is unsupported.

Even when a claim is fully supported, the "issue" field is still required and MUST be present as an empty string.

============================================================
CLASSIFICATION
============================================================

supported:

Every factual claim is directly supported or is a
conservative paraphrase of the supplied course knowledge.

There are no added technical details, stronger
relationships, unsupported qualifiers, unsupported
numerical values, or contradictions.


partially_supported:

The central explanation is grounded in the course
knowledge, but one or more secondary factual claims,
technical details, qualifiers, relationships, mechanisms,
or translations are not adequately supported.

Use partially_supported when removing or weakening the
unsupported portion could make the response grounded.


unsupported:

Use this when:

- a central factual claim is not supported,
- an important technical claim is introduced without
  evidence,
- the response contradicts the course knowledge, or
- substantial rewriting would be required to make the
  response grounded.


============================================================
QUESTIONS
============================================================

A question by itself is not a factual assertion.

Do not treat the answer implied or requested by a
question as though the Tutor asserted that answer.

A question-only response with no embedded factual claim
may therefore be classified as supported.

If a question contains an embedded factual assertion,
validate only that embedded assertion.


============================================================
EVIDENCE DISCIPLINE
============================================================

- Judge only against the supplied COURSE KNOWLEDGE.
- Do not silently fill gaps using subject-matter knowledge.
- Do not reward a claim merely because it sounds correct.
- Do not infer evidence that is not actually present.
- Preserve distinctions between correlation, relationship,
  proportionality, dependence, control, and causation.
- Preserve distinctions between qualitative and
  quantitative statements.
- Preserve comparative qualifiers exactly.
- Do not infer an exact numerical value unless that value
  is explicitly supported by the course knowledge.
- Do not introduce a quantitative qualifier such as
  "most", "nearly all", "higher", or "lower" unless that
  qualifier is explicitly supported by the course knowledge.
- Pay attention to technical terminology and translation.
- Pay attention to invented facts and unsupported detail.

When evidence is ambiguous, prefer
partially_supported or unsupported over supported.

For partially_supported or unsupported results,
the issues list MUST identify the factual claim or claims
that lack sufficient evidence or contradict the source.

If all factual claims are supported, return an empty
issues list.

Return the result using the required structured format.

Confidence must be between 0.0 and 1.0.
"""


GROUNDING_VALIDATION_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": [
                "supported",
                "partially_supported",
                "unsupported",
            ],
        },
        "confidence": {
            "type": "number",
        },
        "reason": {
            "type": "string",
        },
        "issues": {
            "type": "array",
            "items": {
                "type": "string",
            },
        },
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {
                        "type": "string",
                    },
                    "response_quote": {
                        "type": "string",
                    },
                    "status": {
                        "type": "string",
                        "enum": [
                            "supported",
                            "unsupported",
                        ],
                    },
                    "evidence_quote": {
                        "type": "string",
                    },
                    "issue": {
                        "type": "string",
                    },
                },
                "required": [
                    "claim",
                    "response_quote",
                    "status",
                    "evidence_quote",
                    "issue",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "status",
        "confidence",
        "reason",
        "issues",
        "claims",
    ],
    "additionalProperties": False,
}

class ResponseGroundingValidator:
    """
    Validate factual claims in an AI Tutor response
    against retrieved course knowledge.
    """

    ALLOWED_STATUS = {
        "supported",
        "partially_supported",
        "unsupported",
    }

    ATOMIC_ALLOWED_STATUS = {
        "supported",
        "unsupported",
    }

    _CONTROL_RELATION_TERMS = (
        "controlled by",
        "controls",
        "control of",
        "controller",
        "governed by",
        "governs",
        "determined by",
        "determines",
        "regulated by",
        "regulates",
        "\u0e04\u0e27\u0e1a\u0e04\u0e38\u0e21",
        "\u0e16\u0e39\u0e01\u0e04\u0e27\u0e1a\u0e04\u0e38\u0e21\u0e42\u0e14\u0e22",
        "\u0e01\u0e33\u0e2b\u0e19\u0e14\u0e42\u0e14\u0e22",
    )

    def __init__(self) -> None:
        self._claim_precheck = (
            GroundingClaimPrecheck()
        )

    def validate(
        self,
        response: str,
        knowledge_context: str,
        task_name: str = "response_validator",
    ) -> GroundingValidationResult:

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

        if not knowledge_context.strip():
            return GroundingValidationResult(
                status="unsupported",
                confidence=1.0,
                reason=(
                    "No course knowledge was "
                    "available for validation."
                ),
                issues=[
                    "No course knowledge context."
                ],
            )

        # =====================================================
        # BUILD VALIDATION REQUEST
        # =====================================================

        messages = [
            {
                "role": "system",
                "content": (
                    GROUNDING_VALIDATION_PROMPT
                ),
            },
            {
                "role": "user",
                "content": f"""
COURSE KNOWLEDGE:

{knowledge_context}

AI TUTOR RESPONSE — START

{response}

AI TUTOR RESPONSE — END

Evaluate only factual claims explicitly present between
AI TUTOR RESPONSE — START
and
AI TUTOR RESPONSE — END.

EVIDENCE RULES:

- Validate ONLY factual claims explicitly stated in the
  AI TUTOR RESPONSE.
- Do not attribute claims to the tutor that are only
  present in COURSE KNOWLEDGE.
- Do not infer additional factual statements from a
  tutor question.
- A question asking the learner to determine a fact
  does not count as the tutor asserting that fact.
- The validation reason must describe only factual
  claims actually present in the AI TUTOR RESPONSE.
- If the tutor response contains a hint followed by
  a question, validate only the factual content of
  the hint itself.

Evaluate whether the AI Tutor response is grounded
in the course knowledge.
""",
            },
        ]

        # =====================================================
        # STRUCTURED SEMANTIC VALIDATION
        # =====================================================

        try:
            raw_result = (
                chat_with_structured_ai(
                    messages=messages,
                    schema_name=(
                        "response_grounding_validation"
                    ),
                    schema=(
                        GROUNDING_VALIDATION_SCHEMA
                    ),
                    task_name=task_name,
                )
            )

            data = json.loads(
                raw_result.strip()
            )

        except json.JSONDecodeError:
            return self._fail_closed(
                reason=(
                    "Grounding validator returned "
                    "invalid JSON."
                ),
                issue=(
                    "Invalid validator output."
                ),
            )

        except Exception as exc:

            provider_error = str(exc).strip()

            if len(provider_error) > 1200:
                provider_error = (
                    provider_error[:1200]
                    + "..."
                )

            return self._fail_closed(
                reason=(
                    "Grounding validation could not "
                    "be completed because the "
                    "validator service failed."
                ),
                issue=(
                    "Grounding validator service "
                    f"failure: {type(exc).__name__}: "
                    f"{provider_error}"
                ),
            )

        # =====================================================
        # ATOMIC CLAIM VALIDATION
        # =====================================================

        claims = data.get(
            "claims",
        )

        if not isinstance(
            claims,
            list,
        ):
            return self._fail_closed(
                reason=(
                    "Grounding validator returned "
                    "an invalid atomic claim "
                    "structure."
                ),
                issue=(
                    "Atomic claims were missing "
                    "or malformed."
                ),
            )

        # -----------------------------------------------------
        # Empty claim list is valid only when the
        # deterministic precheck also finds no factual claim.
        # -----------------------------------------------------

        if not claims:

            precheck = (
                self._claim_precheck.evaluate(
                    response=response,
                )
            )

            if (
                precheck.status
                == "no_claims"
            ):
                return GroundingValidationResult(
                    status="supported",
                    confidence=1.0,
                    reason=(
                        "The response contains no "
                        "factual claims requiring "
                        "grounding."
                    ),
                    issues=[],
                )

            return self._fail_closed(
                reason=(
                    "The semantic validator returned "
                    "no atomic claims for a response "
                    "that contains factual content."
                ),
                issue=(
                    "Atomic factual claim coverage "
                    "could not be verified."
                ),
            )

        # =====================================================
        # CONFIDENCE
        # =====================================================

        confidence = data.get(
            "confidence",
            0.0,
        )

        try:
            confidence = float(
                confidence
            )

        except (TypeError, ValueError):
            confidence = 0.0

        confidence = max(
            0.0,
            min(
                confidence,
                1.0,
            ),
        )

        supported_count = 0
        unsupported_count = 0

        final_issues: list[str] = []

        # =====================================================
        # VERIFY EACH ATOMIC CLAIM
        # =====================================================

        for index, item in enumerate(
            claims,
            start=1,
        ):

            if not isinstance(
                item,
                dict,
            ):
                return self._fail_closed(
                    reason=(
                        "Grounding validator returned "
                        "a malformed atomic claim."
                    ),
                    issue=(
                        f"Atomic claim {index} "
                        "was not an object."
                    ),
                )

            claim = item.get(
                "claim",
                "",
            )

            response_quote = item.get(
                "response_quote",
                "",
            )

            atomic_status = item.get(
                "status",
                "",
            )

            evidence_quote = item.get(
                "evidence_quote",
                "",
            )

            issue = item.get(
                "issue",
                "",
            )

            # -------------------------------------------------
            # Validate atomic field types
            # -------------------------------------------------

            if not all(
                isinstance(value, str)
                for value in (
                    claim,
                    response_quote,
                    atomic_status,
                    evidence_quote,
                    issue,
                )
            ):
                return self._fail_closed(
                    reason=(
                        "Grounding validator returned "
                        "invalid atomic claim fields."
                    ),
                    issue=(
                        f"Atomic claim {index} "
                        "contained non-string fields."
                    ),
                )

            claim = claim.strip()

            response_quote = (
                response_quote.strip()
            )

            atomic_status = (
                atomic_status.strip()
            )

            evidence_quote = (
                evidence_quote.strip()
            )

            issue = issue.strip()

            if (
                not claim
                or not response_quote
                or atomic_status
                not in self.ATOMIC_ALLOWED_STATUS
            ):
                return self._fail_closed(
                    reason=(
                        "Grounding validator returned "
                        "an incomplete atomic claim."
                    ),
                    issue=(
                        f"Atomic claim {index} "
                        "was incomplete."
                    ),
                )

            # -------------------------------------------------
            # Response quote must really exist in Tutor answer
            # -------------------------------------------------

            if not self._quote_exists(
                quote=response_quote,
                source=response,
            ):
                return self._fail_closed(
                    reason=(
                        "Grounding validator "
                        "attributed an unverifiable "
                        "claim to the Tutor response."
                    ),
                    issue=(
                        f"Atomic claim {index} used "
                        "a response quote not found "
                        "in the Tutor response."
                    ),
                )

            # -------------------------------------------------
            # Explicitly unsupported claim
            # -------------------------------------------------

            if (
                atomic_status
                == "unsupported"
            ):

                unsupported_count += 1

                final_issues.append(
                    issue
                    or (
                        "Unsupported atomic claim: "
                        f"{claim}"
                    )
                )

                continue

            # -------------------------------------------------
            # Supported claim must provide source evidence
            # -------------------------------------------------

            if not evidence_quote:

                unsupported_count += 1

                final_issues.append(
                    (
                        "Atomic claim marked "
                        "supported without course "
                        "evidence: "
                        f"{claim}"
                    )
                )

                continue

            # -------------------------------------------------
            # Evidence quote must really exist in knowledge
            # -------------------------------------------------

            if not self._quote_exists(
                quote=evidence_quote,
                source=knowledge_context,
            ):

                unsupported_count += 1

                final_issues.append(
                    (
                        "Validator evidence quote "
                        "was not found in course "
                        "knowledge for claim: "
                        f"{claim}"
                    )
                )

                continue

            # -------------------------------------------------
            # Relation-strengthening veto
            # -------------------------------------------------

            relation_issue = (
                self._control_relation_issue(
                    response_quote=(
                        response_quote
                    ),
                    evidence_quote=(
                        evidence_quote
                    ),
                )
            )

            if (
                relation_issue
                is not None
            ):

                unsupported_count += 1

                final_issues.append(
                    relation_issue
                )

                continue

            supported_count += 1

        # =====================================================
        # DETERMINISTIC AGGREGATION
        # =====================================================

        total_claims = (
            supported_count
            + unsupported_count
        )

        if unsupported_count == 0:

            status = "supported"

            reason = (
                f"All {total_claims} atomic "
                "factual claims were supported "
                "by verifiable course evidence."
            )

        elif supported_count == 0:

            status = "unsupported"

            reason = (
                f"None of the {total_claims} "
                "atomic factual claims were "
                "fully supported by verifiable "
                "course evidence."
            )

        else:

            status = (
                "partially_supported"
            )

            reason = (
                f"{unsupported_count} of "
                f"{total_claims} atomic factual "
                "claims were unsupported or "
                "lacked sufficient verifiable "
                "evidence."
            )

        final_issues = list(
            dict.fromkeys(
                final_issues
            )
        )

        return GroundingValidationResult(
            status=status,
            confidence=confidence,
            reason=reason,
            issues=final_issues,
        )

    # =========================================================
    # HELPERS
    # =========================================================

    def _fail_closed(
        self,
        reason: str,
        issue: str,
    ) -> GroundingValidationResult:

        return GroundingValidationResult(
            status="unsupported",
            confidence=0.0,
            reason=reason,
            issues=[
                issue
            ],
        )

    def _quote_exists(
        self,
        quote: str,
        source: str,
    ) -> bool:

        quote_norm = (
            self._normalize_for_match(
                quote
            )
        )

        source_norm = (
            self._normalize_for_match(
                source
            )
        )

        if not quote_norm:
            return False

        return (
            quote_norm
            in source_norm
        )

    def _control_relation_issue(
        self,
        response_quote: str,
        evidence_quote: str,
    ) -> str | None:
        """
        Prevent a control / determination relationship
        from being supported by evidence that describes
        different relation participants.

        Example rejected:

        response:
            emitter current controls collector current

        evidence:
            voltage controls current

        Merely sharing the verb "controls" is not enough.
        """

        response_norm = (
            self._normalize_for_match(
                response_quote
            )
        )

        evidence_norm = (
            self._normalize_for_match(
                evidence_quote
            )
        )

        response_has_control = any(
            term in response_norm
            for term
            in self._CONTROL_RELATION_TERMS
        )

        # No control relation in the Tutor claim:
        # this guard has nothing to evaluate.
        if not response_has_control:
            return None

        evidence_has_control = any(
            term in evidence_norm
            for term
            in self._CONTROL_RELATION_TERMS
        )

        # The Tutor asserts a stronger relation but
        # the evidence contains no equivalent relation.
        if not evidence_has_control:
            return (
                "Unsupported relation strengthening: "
                "the Tutor response asserts a control "
                "or determination relationship, but "
                "the cited course evidence does not."
            )

        response_relation = (
            self._extract_control_relation(
                response_quote
            )
        )

        evidence_relation = (
            self._extract_control_relation(
                evidence_quote
            )
        )

        # A control claim must have a verifiable
        # controller-target structure on both sides.
        if (
            response_relation is None
            or evidence_relation is None
        ):
            return (
                "Unsupported relation alignment: "
                "the controller and target could not "
                "be verified against the cited "
                "course evidence."
            )

        (
            response_controller,
            response_target,
        ) = response_relation

        (
            evidence_controller,
            evidence_target,
        ) = evidence_relation

        controller_supported = (
            self._relation_argument_supported(
                response_argument=(
                    response_controller
                ),
                evidence_argument=(
                    evidence_controller
                ),
            )
        )

        target_supported = (
            self._relation_argument_supported(
                response_argument=(
                    response_target
                ),
                evidence_argument=(
                    evidence_target
                ),
            )
        )

        if not (
            controller_supported
            and target_supported
        ):
            return (
                "Unsupported relation entity "
                "alignment: the Tutor response "
                "asserts a control relationship "
                "between entities that are not "
                "the controller-target pair in "
                "the cited course evidence."
            )

        return None

    def _extract_control_relation(
        self,
        text: str,
    ) -> tuple[str, str] | None:
        """
        Return:
            (controller, target)

        for simple active or passive control /
        determination relationships.
        """

        text_norm = (
            self._normalize_for_match(
                text
            )
        )

        # -------------------------------------------------
        # Thai passive relations
        # -------------------------------------------------

        thai_passive_markers = (
            "\u0e16\u0e39\u0e01\u0e04\u0e27\u0e1a\u0e04\u0e38\u0e21\u0e42\u0e14\u0e22",
            "\u0e01\u0e33\u0e2b\u0e19\u0e14\u0e42\u0e14\u0e22",
        )

        for marker in thai_passive_markers:

            if marker in text_norm:

                target, controller = (
                    text_norm.split(
                        marker,
                        1,
                    )
                )

                target = (
                    self._normalize_relation_argument(
                        target
                    )
                )

                controller = (
                    self._normalize_relation_argument(
                        controller
                    )
                )

                if (
                    controller
                    and target
                ):
                    return (
                        controller,
                        target,
                    )

        # -------------------------------------------------
        # English passive relation
        #
        # target is controlled by controller
        # -------------------------------------------------

        passive_match = re.search(
            (
                r"(?P<target>.+?)\s+"
                r"(?:is|are|was|were|be|being|been)\s+"
                r"(?:controlled|regulated|governed|determined)"
                r"\s+by\s+"
                r"(?P<controller>.+?)(?:[.!?]|$)"
            ),
            text_norm,
        )

        if passive_match:

            controller = (
                self._normalize_relation_argument(
                    passive_match.group(
                        "controller"
                    )
                )
            )

            target = (
                self._normalize_relation_argument(
                    passive_match.group(
                        "target"
                    )
                )
            )

            if (
                controller
                and target
            ):
                return (
                    controller,
                    target,
                )

        # -------------------------------------------------
        # English active relation
        #
        # controller controls target
        # -------------------------------------------------

        active_match = re.search(
            (
                r"(?P<controller>.+?)\s+"
                r"(?:controls?|regulates?|governs?|determines?)"
                r"\s+"
                r"(?P<target>.+?)(?:[.!?]|$)"
            ),
            text_norm,
        )

        if active_match:

            controller = (
                self._normalize_relation_argument(
                    active_match.group(
                        "controller"
                    )
                )
            )

            target = (
                self._normalize_relation_argument(
                    active_match.group(
                        "target"
                    )
                )
            )

            if (
                controller
                and target
            ):
                return (
                    controller,
                    target,
                )

        # -------------------------------------------------
        # Thai active relation
        # controller ควบคุม target
        # -------------------------------------------------

        thai_control = (
            "\u0e04\u0e27\u0e1a\u0e04\u0e38\u0e21"
        )

        if thai_control in text_norm:

            controller, target = (
                text_norm.split(
                    thai_control,
                    1,
                )
            )

            controller = (
                self._normalize_relation_argument(
                    controller
                )
            )

            target = (
                self._normalize_relation_argument(
                    target
                )
            )

            if (
                controller
                and target
            ):
                return (
                    controller,
                    target,
                )

        return None

    def _relation_argument_supported(
        self,
        response_argument: str,
        evidence_argument: str,
    ) -> bool:
        """
        A response may be equal to or weaker than
        the evidence, but must not introduce a more
        specific controller or target that does not
        appear in the evidence.

        Example:

        response:  collector current
        evidence:  the collector current
            -> True

        response:  collector current
        evidence:  current
            -> False
        """

        response_norm = (
            self._normalize_relation_argument(
                response_argument
            )
        )

        evidence_norm = (
            self._normalize_relation_argument(
                evidence_argument
            )
        )

        if (
            not response_norm
            or not evidence_norm
        ):
            return False

        response_wrapped = (
            f" {response_norm} "
        )

        evidence_wrapped = (
            f" {evidence_norm} "
        )

        return (
            response_wrapped
            in evidence_wrapped
        )

    def _normalize_relation_argument(
        self,
        text: str,
    ) -> str:

        normalized = (
            self._normalize_for_match(
                text
            )
        )

        normalized = re.sub(
            r"[^\w\u0e00-\u0e7f\-]+",
            " ",
            normalized,
            flags=re.UNICODE,
        )

        tokens = normalized.split()

        # Remove grammatical determiners that do not
        # identify the relation participant.
        while (
            tokens
            and tokens[0]
            in {
                "the",
                "a",
                "an",
            }
        ):
            tokens.pop(0)

        while (
            tokens
            and tokens[-1]
            in {
                "the",
                "a",
                "an",
            }
        ):
            tokens.pop()

        return " ".join(
            tokens
        ).strip()

    def _normalize_for_match(
        self,
        text: str,
    ) -> str:

        normalized = (
            text
            .replace("\u00a0", " ")
            .replace("\u202f", " ")
            .replace("\u2010", "-")
            .replace("\u2011", "-")
            .replace("\u2012", "-")
            .replace("\u2013", "-")
            .replace("\u2014", "-")
            .replace("\u2212", "-")
            .lower()
        )

        normalized = " ".join(
            normalized.split()
        )

        return normalized.strip()