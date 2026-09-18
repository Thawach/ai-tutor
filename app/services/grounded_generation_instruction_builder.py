class GroundedGenerationInstructionBuilder:
    """
    Builds deterministic evidence-bounded generation
    instructions for Tutor responses.

    This builder:
    - does not call an LLM
    - does not inspect learner state
    - does not perform grounding validation
    - does not mutate course knowledge
    """

    def build(
        self,
        has_grounded_knowledge: bool,
    ) -> str:

        if not isinstance(
            has_grounded_knowledge,
            bool,
        ):
            raise TypeError(
                "has_grounded_knowledge must be bool."
            )

        if not has_grounded_knowledge:
            return ""

        return (
            "EVIDENCE-BOUNDED GENERATION:\n"
            "\n"
            "- Treat the supplied COURSE KNOWLEDGE as the "
            "complete factual boundary for this response.\n"
            "- Every factual or technical claim must be "
            "explicitly supported by the supplied course "
            "knowledge or be a conservative paraphrase of it.\n"
            "- Do not add facts from general model knowledge, "
            "even when those facts are normally correct.\n"
            "- Do not strengthen a relationship stated in the "
            "source. For example, 'related to' does not mean "
            "'proportional to', 'causes', or 'is controlled by'.\n"
            "- Do not add unsupported numerical values, "
            "thresholds, equations, gain factors, quantitative "
            "qualifiers, doping comparisons, or carrier "
            "proportions.\n"
            "- Do not introduce a technical mechanism that is "
            "not stated or conservatively implied by the "
            "course knowledge.\n"
            "- If a requested factual detail is not supported "
            "by the course knowledge, omit that detail or state "
            "that the course material does not provide it.\n"
        )