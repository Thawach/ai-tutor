from dataclasses import dataclass
from app.domain.learner.state import LearnerState
from app.domain.scaffolding.strategies import get_strategy


@dataclass
class ScaffoldingDecision:
    evaluation: str

    level_before: int
    level_after: int

    action: str
    reason: str

    strategy: str
    strategy_description: str


class ScaffoldingPolicy:
    """
    Adaptive scaffolding policy.

    ใช้ทั้งผลการประเมินล่าสุด
    และ performance streak ของผู้เรียน
    """

    CORRECT_STREAK_FOR_FADING = 2
    PARTIAL_STREAK_FOR_SUPPORT = 2

    def decide(
        self,
        state: LearnerState,
        evaluation: str,
    ) -> ScaffoldingDecision:

        level_before = state.scaffolding_level

        action = "keep"
        reason = "maintain_current_support"

        # -------------------------
        # Correct
        # -------------------------

        if evaluation == "correct":

            if (
                state.correct_streak
                >= self.CORRECT_STREAK_FOR_FADING
            ):

                state.decrease_support()

                action = "decrease"
                reason = "repeated_correct_responses"

                state.correct_streak = 0

            else:

                action = "keep"
                reason = "correct_response_monitoring"

        # -------------------------
        # Partial
        # -------------------------

        elif evaluation == "partial":

            if (
                state.partial_streak
                >= self.PARTIAL_STREAK_FOR_SUPPORT
            ):

                state.increase_support()

                action = "increase"
                reason = "repeated_partial_responses"

                state.partial_streak = 0

            else:

                action = "keep"
                reason = "partial_response"

        # -------------------------
        # Misconception
        # -------------------------

        elif evaluation == "misconception":

            state.increase_support()

            action = "increase"
            reason = "misconception_detected"

        # -------------------------
        # Don't know
        # -------------------------

        elif evaluation == "dont_know":

            state.increase_support()

            action = "increase"
            reason = "learner_needs_more_support"

        strategy = get_strategy(
            state.scaffolding_level
        )

        return ScaffoldingDecision(
            evaluation=evaluation,
            level_before=level_before,
            level_after=state.scaffolding_level,
            action=action,
            reason=reason,
            strategy=strategy.name,
            strategy_description=strategy.description,
        )