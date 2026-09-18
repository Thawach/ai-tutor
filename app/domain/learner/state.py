from app.domain.learner.misconception import (
    MisconceptionRecord,
)

class LearnerState:
    """
    เก็บสถานะการเรียนรู้ชั่วคราวของผู้เรียน

    ในเวอร์ชันนี้ยังเก็บในหน่วยความจำ
    ภายหลังจะ persist ใน PostgreSQL
    """

    def __init__(self):
        self.scaffolding_level = 1
        self.turn_count = 0
        self.last_evaluation = None

        # Performance history
        self.correct_streak = 0
        self.partial_streak = 0
        self.failure_streak = 0

        self.attempt_count = 0
        self.hint_count = 0

        self.misconceptions = []

    def increase_support(self):
        if self.scaffolding_level < 5:
            self.scaffolding_level += 1

    def decrease_support(self):
        if self.scaffolding_level > 1:
            self.scaffolding_level -= 1

    def next_turn(self):
        self.turn_count += 1

    def set_evaluation(self, evaluation: str):
        self.last_evaluation = evaluation
        self.attempt_count += 1

        self._update_streaks(evaluation)

    def _update_streaks(self, evaluation: str):

        if evaluation == "correct":

            self.correct_streak += 1

            self.partial_streak = 0
            self.failure_streak = 0

        elif evaluation == "partial":

            self.partial_streak += 1

            self.correct_streak = 0
            self.failure_streak = 0

        elif evaluation in (
            "misconception",
            "dont_know",
        ):

            self.failure_streak += 1

            self.correct_streak = 0
            self.partial_streak = 0

        else:

            self.correct_streak = 0
            self.partial_streak = 0
            self.failure_streak = 0

    def reset_streaks(self):

        self.correct_streak = 0
        self.partial_streak = 0
        self.failure_streak = 0

    def reset_learning_sequence(self):
        """
        Reset pedagogical state for a new learning topic
        while preserving session-level learner history.

        Preserved:
        - turn_count
        - misconceptions

        Reset:
        - scaffolding_level
        - last_evaluation
        - performance streaks
        - attempt_count
        - hint_count
        """

        self.scaffolding_level = 1
        self.last_evaluation = None

        self.reset_streaks()

        self.attempt_count = 0
        self.hint_count = 0

    def add_misconception(
            self,
            description: str,
        ):
            """
            เพิ่ม misconception ใหม่
            หรือเพิ่ม occurrence_count ถ้าเคยพบแล้ว
            """

            if not description:
                return

            normalized = description.strip().lower()

            for item in self.misconceptions:

                if item.description.strip().lower() == normalized:

                    item.occurrence_count += 1
                    item.status = "active"

                    return

            self.misconceptions.append(
                MisconceptionRecord(
                    description=description.strip()
                )
            )   

    def get_active_misconceptions(self):
        return [
            item
            for item in self.misconceptions
            if item.status == "active"
        ]

    def reset(self):

        self.scaffolding_level = 1
        self.turn_count = 0
        self.last_evaluation = None

        self.correct_streak = 0
        self.partial_streak = 0
        self.failure_streak = 0

        self.attempt_count = 0
        self.hint_count = 0

        self.misconceptions = []