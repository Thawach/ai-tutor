import json

from app.ai import (
    chat_with_structured_ai,
)

from app.services.repair_evidence_selection_models import (
    RepairEvidenceSelectionResult,
)

from app.services.evidence_quote_usability_guard import (
    EvidenceQuoteUsabilityGuard,
)


REPAIR_EVIDENCE_SELECTION_PROMPT = """
You are an evidence-selection component for a
course-grounded AI tutoring system.

Your task is NOT to answer the learner.

Your task is NOT to repair the Tutor response.

Your only task is to select source evidence from the
supplied COURSE KNOWLEDGE that could safely support a
new factual Tutor response.

STRICT EVIDENCE RULES:

- Use only COURSE KNOWLEDGE.
- Every evidence_quote must be copied verbatim from
  COURSE KNOWLEDGE.
- Do not paraphrase.
- Do not summarize.
- Do not rewrite.
- Do not combine text from separate source locations
  into one quote.
- Do not use outside knowledge.
- Do not infer facts that are not explicitly stated.
- Do not strengthen relationships.
- Do not convert "related to" into "controlled by",
  "determined by", "proportional to", or another
  stronger relation.
- Do not introduce exact values, quantities,
  comparisons, causal mechanisms, or qualifiers unless
  they are explicitly present in COURSE KNOWLEDGE.
- Select only evidence relevant to the learner's
  current request.
- Prefer the smallest SET of excerpts that together
  provide sufficient direct evidence for the
  learner's requested information.
- If COURSE KNOWLEDGE contains no suitable evidence,
  return an empty evidence_quotes list.

EVIDENCE SUFFICIENCY RULES:

- Do not select evidence merely because it mentions
  the same entity as the learner's question.

- The selected evidence set must address the TYPE of
  information requested by the learner.

- Distinguish among:
  * identity / definition
  * function / role
  * relationship
  * mechanism / process
  * condition / state
  * quantity / numerical value
  * comparison

- If the learner asks about a FUNCTION or ROLE,
  evidence that only identifies or names the entity
  is insufficient.

- If the learner asks about a RELATIONSHIP, select
  evidence that explicitly states the relationship
  and its relevant participants.

- If the answer requires a chain of statements,
  select multiple exact excerpts rather than one
  weak excerpt.

- Every selected evidence quote should be
  semantically complete enough to stand on its own.

- Do not select a truncated lead-in, sentence
  fragment, heading fragment, or clause whose
  meaning depends on omitted text.

- In particular, do not select an excerpt that ends
  with an unfinished connector such as "as",
  "because", "therefore", "and", or a colon when the
  content that completes the statement is absent
  from the selected quote.

- If a source sentence introduces an equation,
  figure, list, or continuation that is not included
  in the available text, prefer another complete
  source excerpt that supports the same point.

- Evidence provenance alone is not enough:
  a quote must be both exact and usable as a
  self-contained evidence unit.

EVIDENCE TEXT INTEGRITY RULES:

- Selected evidence must remain understandable after
  extraction from the source document.

- Do not select an excerpt when important variables,
  mathematical symbols, labels, or words appear to be
  missing from the extracted text.

- Do not select extraction-damaged sentences such as:
  "With the voltage and as shown, ..."

  This indicates that variables or symbols between
  "voltage" and "and" were lost during document
  extraction.

- Avoid excerpts whose meaning depends on omitted
  equations, figures, diagrams, variable labels, or
  nearby layout that is not present in the selected
  quote.

- A quote may exist exactly in COURSE KNOWLEDGE and
  still be unsuitable evidence if PDF extraction has
  made the sentence incomplete or semantically broken.

- When both damaged and clean evidence support the
  requested information, always prefer the clean
  self-contained textual statement.

- For example, prefer a complete statement such as:

  "Therefore, the collector current is related to the
  emitter current which is in turn a function of the
  B-E voltage."

  over an extraction-damaged biasing sentence whose
  voltage-variable labels are missing.  


- Prefer all directly useful evidence needed for a
  conservative answer over a single superficially
  relevant sentence.

- Do not treat lexical overlap as evidence
  sufficiency.

- If COURSE KNOWLEDGE contains only identity-level
  information for a function-level question, return
  an empty evidence_quotes list.

- A function or role question does NOT require the
  source to literally use the words "function",
  "role", "controls", or "regulates".

- If no direct function statement exists, a set of
  explicit mechanism, condition, or relationship
  statements may together provide sufficient
  evidence for a conservative operational answer.

- In that case, select all exact excerpts needed for
  the evidence chain.

- The downstream answer must preserve the individual
  source relationships. It must NOT compress the
  evidence chain into a stronger functional label
  such as "X controls Y" unless that controller-target
  relationship is explicitly stated in the source.

- Return no_evidence only when neither:
  (a) a direct role/function statement, nor
  (b) a sufficient explicit operational evidence chain
  exists in COURSE KNOWLEDGE.

EXAMPLE OF SUFFICIENT INDIRECT FUNCTION EVIDENCE:

If the learner asks what component X does, the course
knowledge may not literally say "X's function is ...".

However, if the source explicitly states:
1. the condition involving X,
2. the resulting physical or electrical relationship,
3. the resulting relationship to another quantity,

those exact excerpts may be selected together as an
operational evidence chain.

Do not invent a stronger summary relationship that is
absent from the excerpts.

SOURCE FIDELITY RULES:

- Never repair, reconstruct, restore, rewrite, clean up,
  or complete damaged COURSE KNOWLEDGE text.

- Never insert a missing variable, symbol, word,
  equation, label, punctuation mark, or connector
  into an evidence quote.

- Every evidence quote must be copied from COURSE
  KNOWLEDGE as it actually appears there.

- If an extraction-damaged source sentence would need
  editing before it becomes understandable, do NOT
  select a repaired version of that sentence.

- Instead:
  1. select another clean exact excerpt that supports
     the requested information; or
  2. omit that evidence item.

- If no clean exact evidence remains sufficient,
  return an empty evidence_quotes list.

- Evidence quality must never be improved by changing
  the source text. Selection may remove bad excerpts;
  it may not rewrite them.

EXAMPLE OF FORBIDDEN SOURCE RECONSTRUCTION:

COURSE KNOWLEDGE contains:

"With the voltage and as shown, the Base-Emitter
(B-E) junction is forward biased ..."

Do NOT return a reconstructed quote such as:

"With the VBE and VCB voltages as shown, the
Base-Emitter junction is forward biased ..."

even if that reconstruction appears technically
reasonable.

The reconstructed sentence is not an exact source
excerpt and must not be selected.  

EXAMPLE OF INSUFFICIENT EVIDENCE:

Learner asks:
"What does the Base do?"

Evidence says only:
"The three terminals are Base, Collector and Emitter."

This identifies the Base but does NOT establish its
function and must not be selected as sufficient
evidence by itself.  

EXAMPLE OF SUFFICIENT INDIRECT FUNCTION EVIDENCE:

If the learner asks what component X does, the course
knowledge may not literally say "X's function is ...".

However, if the source explicitly states:
1. the condition involving X,
2. the resulting physical or electrical relationship,
3. the resulting relationship to another quantity,

those exact excerpts may be selected together as an
operational evidence chain.

Do not invent a stronger summary relationship that is
absent from the excerpts.

EXAMPLE OF INCOMPLETE EVIDENCE:

Do NOT select:

"The current through the B-E junction is related
to the B-E voltage as"

when the equation or continuation introduced by
"as" is not included in the selected evidence.

Instead, select a complete source statement that
can stand independently, or omit that excerpt if
no complete version is available.

EXAMPLE OF EXTRACTION-DAMAGED EVIDENCE:

Do NOT select:

"With the voltage and as shown, the Base-Emitter
(B-E) junction is forward biased ..."

when the source extraction has omitted the voltage
variables that originally occurred between those
words.

Exact source matching alone does not make such an
excerpt a usable evidence unit.

IMPORTANT:

Python will independently verify that every selected
evidence_quote really exists in COURSE KNOWLEDGE.

Return only the required structured result.
"""


REPAIR_EVIDENCE_SELECTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "evidence_quotes": {
            "type": "array",
            "items": {
                "type": "string",
            },
            "maxItems": 6,
        },
        "reason": {
            "type": "string",
        },
    },
    "required": [
        "evidence_quotes",
        "reason",
    ],
}

_EVIDENCE_QUOTE_USABILITY_GUARD = (
    EvidenceQuoteUsabilityGuard()
)

class RepairEvidenceSelector:
    """
    LLM-assisted evidence selection with deterministic
    source-provenance verification.

    The LLM may propose evidence excerpts.

    Python owns the final decision about whether every
    proposed excerpt actually exists in the supplied
    course knowledge.
    """

    def select(
        self,
        learner_message: str,
        knowledge_context: str,
        validation_issues: list[str]
        | tuple[str, ...]
        | None = None,
    ) -> RepairEvidenceSelectionResult:

        if not isinstance(
            learner_message,
            str,
        ):
            raise TypeError(
                "learner_message must be str."
            )

        if not isinstance(
            knowledge_context,
            str,
        ):
            raise TypeError(
                "knowledge_context must be str."
            )

        if validation_issues is not None:
            if not isinstance(
                validation_issues,
                (list, tuple),
            ):
                raise TypeError(
                    "validation_issues must be "
                    "list[str], tuple[str, ...], "
                    "or None."
                )

            if not all(
                isinstance(item, str)
                for item in validation_issues
            ):
                raise TypeError(
                    "validation_issues must contain "
                    "only strings."
                )

        if not knowledge_context.strip():
            return (
                RepairEvidenceSelectionResult(
                    status="no_evidence",
                    evidence_quotes=(),
                    reason=(
                        "No course knowledge was "
                        "available for evidence "
                        "selection."
                    ),
                    issues=(
                        "No course knowledge context.",
                    ),
                )
            )

        issues_text = "\n".join(
            f"- {item}"
            for item in (
                validation_issues or ()
            )
        )

        if not issues_text:
            issues_text = (
                "- No prior grounding issue "
                "was supplied."
            )

        messages = [
            {
                "role": "system",
                "content": (
                    REPAIR_EVIDENCE_SELECTION_PROMPT
                ),
            },
            {
                "role": "user",
                "content": f"""
LEARNER MESSAGE:

{learner_message}


PRIOR GROUNDING ISSUES:

{issues_text}


COURSE KNOWLEDGE:

{knowledge_context}


Select the smallest sufficient set of exact source
excerpts that can directly support a conservative
factual response to the learner.

Match the evidence to the information type requested
by the learner. Do not select identity-only evidence
for a function or relationship question.
""",
            },
        ]

        try:
            raw_result = (
                chat_with_structured_ai(
                    messages=messages,
                    schema_name=(
                        "repair_evidence_selection"
                    ),
                    schema=(
                        REPAIR_EVIDENCE_SELECTION_SCHEMA
                    ),
                    task_name=(
                        "repair_evidence_selector"
                    ),
                )
            )

            data = json.loads(
                raw_result.strip()
            )

        except json.JSONDecodeError:
            return self._fail_closed(
                reason=(
                    "Repair evidence selector "
                    "returned invalid JSON."
                ),
                issue=(
                    "Invalid evidence selector output."
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
                    "Repair evidence selection could "
                    "not be completed because the "
                    "selector service failed."
                ),
                issue=(
                    "Repair evidence selector service "
                    f"failure: {type(exc).__name__}: "
                    f"{provider_error}"
                ),
            )

        evidence_quotes = data.get(
            "evidence_quotes",
        )

        reason = data.get(
            "reason",
            "",
        )

        if not isinstance(
            evidence_quotes,
            list,
        ):
            return self._fail_closed(
                reason=(
                    "Repair evidence selector returned "
                    "an invalid evidence structure."
                ),
                issue=(
                    "evidence_quotes was not a list."
                ),
            )

        if not isinstance(
            reason,
            str,
        ):
            return self._fail_closed(
                reason=(
                    "Repair evidence selector returned "
                    "an invalid reason."
                ),
                issue=(
                    "Evidence selection reason was "
                    "not a string."
                ),
            )

        verified_quotes: list[str] = []

        for index, quote in enumerate(
            evidence_quotes,
            start=1,
        ):

            if not isinstance(
                quote,
                str,
            ):
                return self._fail_closed(
                    reason=(
                        "Repair evidence selector "
                        "returned a malformed quote."
                    ),
                    issue=(
                        f"Evidence quote {index} "
                        "was not a string."
                    ),
                )

            quote = quote.strip()

            if not quote:
                return self._fail_closed(
                    reason=(
                        "Repair evidence selector "
                        "returned an empty quote."
                    ),
                    issue=(
                        f"Evidence quote {index} "
                        "was empty."
                    ),
                )

            if not self._quote_exists(
                quote=quote,
                source=knowledge_context,
            ):

                display_quote = quote.strip()

                if len(display_quote) > 300:
                    display_quote = (
                        display_quote[:300]
                        + "..."
                    )

                return self._fail_closed(
                    reason=(
                        "Repair evidence selector "
                        "proposed unverifiable source "
                        "evidence."
                    ),
                    issue=(
                        f"Evidence quote {index} "
                        "was not found in course "
                        "knowledge: "
                        f"{display_quote!r}"
                    ),
                )


            usability_result = (
                _EVIDENCE_QUOTE_USABILITY_GUARD
                .evaluate(
                    quote
                )
            )

            if not usability_result.is_usable:

                display_quote = quote.strip()

                if len(display_quote) > 300:
                    display_quote = (
                        display_quote[:300]
                        + "..."
                    )

                return self._fail_closed(
                    reason=(
                        "Repair evidence selector "
                        "proposed unusable source "
                        "evidence."
                    ),
                    issue=(
                        f"Evidence quote {index} "
                        "was exact but unusable as "
                        "a self-contained evidence "
                        "unit: "
                        f"{display_quote!r}. "
                        f"{usability_result.reason}"
                    ),
                )

            verified_quotes.append(
                quote
            )

            verified_quotes.append(
                quote
            )

        # Remove exact duplicates while preserving
        # source-selection order.
        verified_quotes = list(
            dict.fromkeys(
                verified_quotes
            )
        )

        if not verified_quotes:
            return (
                RepairEvidenceSelectionResult(
                    status="no_evidence",
                    evidence_quotes=(),
                    reason=(
                        reason.strip()
                        or (
                            "No suitable verified "
                            "course evidence was "
                            "selected."
                        )
                    ),
                    issues=(),
                )
            )

        return RepairEvidenceSelectionResult(
            status="verified",
            evidence_quotes=tuple(
                verified_quotes
            ),
            reason=(
                reason.strip()
                or (
                    "Verified course evidence "
                    "was selected."
                )
            ),
            issues=(),
        )

    def _fail_closed(
        self,
        reason: str,
        issue: str,
    ) -> RepairEvidenceSelectionResult:

        return RepairEvidenceSelectionResult(
            status="invalid",
            evidence_quotes=(),
            reason=reason,
            issues=(
                issue,
            ),
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