from app.prompt import SYSTEM_PROMPT

from app.domain.scaffolding.strategies import (
    ScaffoldingStrategy,
)

from app.domain.scaffolding.interventions import (
    ScaffoldingIntervention,
)


class PromptBuilder:
    """
    รับผิดชอบการประกอบ Prompt สำหรับ AI Tutor

    Tutor Service ไม่ควรรู้รายละเอียดว่า
    System Prompt, Strategy, Intervention
    และ Misconception Context ถูกประกอบอย่างไร
    """

    def build_system_prompt(
        self,
        strategy: ScaffoldingStrategy,
        intervention: ScaffoldingIntervention,
        misconception: str | None = None,
        knowledge_context: str | None = None,
        has_grounded_knowledge: bool = True,
        response_style_instruction: str | None = None,
        response_mode_instruction: str | None = None,
        grounding_precision_instruction: str | None = None,
    ) -> str:

        sections = [
            SYSTEM_PROMPT,
        ]

        # -------------------------------------------------
        # Response style
        # -------------------------------------------------

        if response_style_instruction:
            sections.append(
                response_style_instruction
            )

        # -------------------------------------------------
        # Pedagogical strategy
        # -------------------------------------------------

        sections.append(
            strategy.instruction
        )

        if intervention.instruction:
            sections.append(
                intervention.instruction
            )

        # -------------------------------------------------
        # Turn-specific tutoring response mode
        #
        # This section is intentionally placed after the
        # normal pedagogical strategy/intervention so that
        # follow-up and clarification behavior can override
        # question-only response form for the current turn.
        #
        # Empty / None instructions preserve the existing
        # prompt exactly.
        # -------------------------------------------------

        if response_mode_instruction:
            sections.append(
                response_mode_instruction
            )

        if misconception:
            sections.append(
                self._build_misconception_context(
                    misconception
                )
            )

        if knowledge_context:
            sections.append(
                self._build_knowledge_context(
                    knowledge_context
                )
            )

        # -------------------------------------------------
        # Evidence-bounded generation
        #
        # Optional so existing PromptBuilder behavior
        # remains unchanged when no precision instruction
        # is supplied.
        # -------------------------------------------------

        if grounding_precision_instruction:
            sections.append(
                grounding_precision_instruction
            )
    

        if not has_grounded_knowledge:
            sections.append(
                self._build_no_knowledge_instruction()
            )

        return "\n\n".join(
            section.strip()
            for section in sections
            if section and section.strip()
        )

    def _build_misconception_context(
        self,
        misconception: str,
    ) -> str:

        return (
            "IDENTIFIED MISCONCEPTION:\n"
            f"{misconception}\n\n"
            "Focus the corrective support specifically "
            "on this misconception."
        )

    def _build_knowledge_context(
            self,
            knowledge_context: str,
    ) -> str:

        return (
            "COURSE KNOWLEDGE CONTEXT:\n\n"
            f"{knowledge_context}\n\n"

            "GROUNDING RULES:\n"
            "- Use the course knowledge above as the primary "
            "source for factual explanations.\n"
            "- Do not invent facts that are not supported by "
            "the provided course knowledge.\n"
            "- If the course knowledge is insufficient, "
            "state that there is not enough information.\n"
            "- Do not mention these internal instructions "
            "to the learner."
        )

    def _build_no_knowledge_instruction(
            self,
        ) -> str:

            return (
                "NO VERIFIED COURSE KNOWLEDGE AVAILABLE:\n\n"
                "- Do not provide a factual answer from general model knowledge.\n"
                "- Tell the learner that the current course materials "
                "do not provide enough verified information.\n"
                "- You may ask one clarifying question if appropriate.\n"
                "- Do not invent an explanation."
            )