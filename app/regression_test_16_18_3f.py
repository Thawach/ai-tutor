import inspect

from app.tutor import AITutor


class RegressionSuite16_18_3F:

    def __init__(self):
        self.passed = 0
        self.failed = 0

    def check(
        self,
        condition: bool,
        name: str,
        detail: str,
    ) -> None:

        if condition:
            self.passed += 1

            print(
                f"[PASS] {name}"
            )

            print(
                f"       {detail}"
            )

        else:
            self.failed += 1

            print(
                f"[FAIL] {name}"
            )

            print(
                f"       {detail}"
            )

    def test_t1_boundary_flag(
        self,
    ) -> None:

        source = inspect.getsource(
            AITutor.respond
        )

        self.check(
            (
                "new_learning_sequence_turn"
                in source
            ),
            "BOUNDARY-T1 Sequence flag",
            (
                "AITutor explicitly detects the "
                "first turn of a learning sequence."
            ),
        )

    def test_t2_flag_before_waiting_mutation(
        self,
    ) -> None:

        source = " ".join(
            inspect.getsource(
                AITutor.respond
            ).split()
        )

        flag_index = source.find(
            "new_learning_sequence_turn = "
            "( not self.waiting_for_response )"
        )

        waiting_index = source.find(
            "self.waiting_for_response = True"
        )

        self.check(
            (
                flag_index >= 0
                and
                waiting_index > flag_index
            ),
            "BOUNDARY-T2 Detection order",
            (
                "Sequence boundary is captured "
                "before waiting_for_response is "
                "mutated."
            ),
        )

    def test_t3_memory_guard(
        self,
    ) -> None:

        source = " ".join(
            inspect.getsource(
                AITutor.respond
            ).split()
        )

        self.check(
            (
                "retrieval_tutor_question = None"
                in source
                and
                "if not new_learning_sequence_turn"
                in source
                and
                (
                    "self.memory."
                    "get_last_assistant_message()"
                )
                in source
            ),
            "BOUNDARY-T3 Memory guard",
            (
                "Previous Tutor retrieval context "
                "is unavailable on a new sequence "
                "turn."
            ),
        )

    def test_t4_memory_not_reset(
        self,
    ) -> None:

        source = inspect.getsource(
            AITutor.respond
        )

        self.check(
            (
                "self.memory.reset()"
                not in source
            ),
            "BOUNDARY-T4 Memory preserved",
            (
                "Sequence isolation does not erase "
                "session conversation memory."
            ),
        )

    def test_t5_no_progress_dependency(
        self,
    ) -> None:

        source = inspect.getsource(
            AITutor.respond
        )

        condition = (
            (
                "learner_progress_snapshot_builder"
                not in source
            )
            and
            (
                "learner_progress_snapshot"
                not in source
            )
            and
            (
                "self.learner_progress_history."
                not in source
            )
            and
            (
                source.count(
                    "self._record_learner_progress_history("
                )
                == 2
            )
        )

        self.check(
            condition,
            "BOUNDARY-T5 Progress isolation",
            (
                "Retrieval boundary logic does not "
                "read learner-progress state for "
                "routing or retrieval decisions; "
                "only observation-only history "
                "recording is permitted."
            ),
        )
    def test_t6_single_turn_increment_call(
        self,
    ) -> None:

        source = inspect.getsource(
            AITutor.respond
        )

        next_turn_calls = source.count(
            "self.state.next_turn()"
        )

        self.check(
            next_turn_calls == 1,
            "BOUNDARY-T6 Single turn increment",
            (
                "AITutor.respond() increments "
                "session turn exactly once per "
                "learner message."
            ),
        )

    def run(self) -> None:

        print(
            "\nStep 16.18.3F "
            "Sequence Boundary Retrieval "
            "Regression Suite\n"
        )

        self.test_t1_boundary_flag()
        self.test_t2_flag_before_waiting_mutation()
        self.test_t3_memory_guard()
        self.test_t4_memory_not_reset()
        self.test_t5_no_progress_dependency()
        self.test_t6_single_turn_increment_call()

        print(
            "\nSUMMARY"
        )

        print(
            f"Passed: {self.passed}"
        )

        print(
            f"Failed: {self.failed}"
        )


def main() -> None:

    RegressionSuite16_18_3F().run()


if __name__ == "__main__":
    main()