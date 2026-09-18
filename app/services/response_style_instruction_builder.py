from app.services.response_style_resolver import (
    ResolvedResponseStyle,
)


class ResponseStyleInstructionBuilder:
    """
    Build deterministic prompt instructions from a
    ResolvedResponseStyle.

    This service does not call an LLM.

    It converts typed response-style configuration into
    concise prompt instructions for downstream Tutor
    generation.

    Step 16.17.2D initially uses this service independently.
    Prompt integration should happen only after regression
    tests pass.
    """

    LANGUAGE_LABELS = {
        "th": "Thai",
        "en": "English",
    }

    TECHNICAL_TERM_FORMAT_INSTRUCTIONS = {
        "thai_with_english_parentheses": (
            "When appropriate, present Thai technical terms "
            "with the English term in parentheses."
        ),
        "english_only": (
            "Use English technical terminology."
        ),
        "thai_only": (
            "Prefer Thai technical terminology and avoid "
            "unnecessary English technical terms."
        ),
    }

    TONE_INSTRUCTIONS = {
        "supportive_academic": (
            "Use a supportive academic tone that is clear, "
            "respectful, and suitable for learning."
        ),
        "formal": (
            "Use a formal academic tone."
        ),
        "concise": (
            "Use a concise and direct instructional tone."
        ),
        "friendly": (
            "Use a friendly and approachable instructional tone."
        ),
    }

    EXPLANATION_DEPTH_INSTRUCTIONS = {
        "adaptive": (
            "Adapt explanation depth to the learner's apparent "
            "level and the current tutoring context."
        ),
        "brief": (
            "Keep explanations brief and focused."
        ),
        "detailed": (
            "Provide detailed explanations when explaining "
            "course concepts."
        ),
    }

    QUESTION_STYLE_INSTRUCTIONS = {
        "socratic": (
            "Use Socratic questioning when a guiding question "
            "is pedagogically appropriate."
        ),
        "direct": (
            "Prefer direct instructional questions."
        ),
        "reflective": (
            "Use reflective questions that encourage the learner "
            "to explain their reasoning."
        ),
    }

    def build(
        self,
        style: ResolvedResponseStyle,
    ) -> str:
        """
        Build deterministic response-style prompt instructions.

        The output is intentionally plain text so it can later
        be inserted into existing Tutor prompts without coupling
        this service to a specific prompt framework.
        """

        instructions = []

        # =====================================================
        # Primary language
        # =====================================================

        language_label = (
            self.LANGUAGE_LABELS.get(
                style.primary_language,
                style.primary_language,
            )
        )

        instructions.append(
            f"Respond primarily in {language_label}."
        )

        # =====================================================
        # Technical English policy
        # =====================================================

        if style.allow_technical_english:

            instructions.append(
                "English technical terms may be used when they "
                "improve clarity or preserve standard terminology."
            )

        else:

            instructions.append(
                "Avoid English technical terms unless they are "
                "strictly necessary."
            )

        # =====================================================
        # Technical term format
        # =====================================================

        technical_term_instruction = (
            self.TECHNICAL_TERM_FORMAT_INSTRUCTIONS.get(
                style.technical_term_format
            )
        )

        if technical_term_instruction:

            instructions.append(
                technical_term_instruction
            )

        else:

            instructions.append(
                "Follow the course-configured technical term "
                f"format: {style.technical_term_format}."
            )

        # =====================================================
        # Tone
        # =====================================================

        tone_instruction = (
            self.TONE_INSTRUCTIONS.get(
                style.tone
            )
        )

        if tone_instruction:

            instructions.append(
                tone_instruction
            )

        else:

            instructions.append(
                "Use the course-configured response tone: "
                f"{style.tone}."
            )

        # =====================================================
        # Explanation depth
        # =====================================================

        depth_instruction = (
            self.EXPLANATION_DEPTH_INSTRUCTIONS.get(
                style.explanation_depth
            )
        )

        if depth_instruction:

            instructions.append(
                depth_instruction
            )

        else:

            instructions.append(
                "Use the course-configured explanation depth: "
                f"{style.explanation_depth}."
            )

        # =====================================================
        # Question style
        # =====================================================

        question_instruction = (
            self.QUESTION_STYLE_INSTRUCTIONS.get(
                style.question_style
            )
        )

        if question_instruction:

            instructions.append(
                question_instruction
            )

        else:

            instructions.append(
                "Use the course-configured question style: "
                f"{style.question_style}."
            )

        # =====================================================
        # Maximum guiding questions
        # =====================================================

        if style.max_guiding_questions == 0:

            instructions.append(
                "Do not include a guiding question unless another "
                "higher-priority tutoring rule explicitly requires one."
            )

        elif style.max_guiding_questions == 1:

            instructions.append(
                "Use no more than one guiding question in a response."
            )

        else:

            instructions.append(
                "Use no more than "
                f"{style.max_guiding_questions} "
                "guiding questions in a response."
            )

        return "\n".join(
            f"- {instruction}"
            for instruction in instructions
        )