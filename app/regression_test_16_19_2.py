import inspect

from types import SimpleNamespace

from app.tutor import AITutor


class RegressionSuite16_19_2:

    def __init__(self):

        self.passed = 0
        self.failed = 0

    # =========================================================
    # HELPERS
    # =========================================================

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

    # =========================================================
    # STRUCTURAL INTEGRATION
    # =========================================================

    def test_t1_two_return_paths(
        self,
    ) -> None:

        source = inspect.getsource(
            AITutor.respond
        )

        self.check(
            source.count(
                "return answer"
            ) == 2,
            "HISTORY-RUNTIME-T1 Return paths",
            (
                "AITutor.respond() retains exactly "
                "two completed-turn return paths."
            ),
        )

    def test_t2_two_gateway_calls(
        self,
    ) -> None:

        source = inspect.getsource(
            AITutor.respond
        )

        self.check(
            source.count(
                "self._record_learner_progress_history("
            ) == 2,
            "HISTORY-RUNTIME-T2 Gateway coverage",
            (
                "Both completed-turn return paths "
                "are covered by the single history "
                "recording gateway."
            ),
        )

    def test_t3_single_turn_increment(
        self,
    ) -> None:

        source = inspect.getsource(
            AITutor.respond
        )

        self.check(
            source.count(
                "self.state.next_turn()"
            ) == 1,
            "HISTORY-RUNTIME-T3 Single turn increment",
            (
                "History integration does not "
                "reintroduce duplicate session-turn "
                "increments."
            ),
        )

    def test_t4_gateway_before_returns(
        self,
    ) -> None:

        lines = (
            inspect.getsource(
                AITutor.respond
            )
            .splitlines()
        )

        return_indexes = [
            index
            for index, line
            in enumerate(lines)
            if (
                "return answer"
                in line
            )
        ]

        gateway_indexes = [
            index
            for index, line
            in enumerate(lines)
            if (
                "self._record_learner_progress_history("
                in line
            )
        ]

        condition = (
            len(return_indexes) == 2
            and
            len(gateway_indexes) == 2
            and
            gateway_indexes[0]
            < return_indexes[0]
            and
            gateway_indexes[1]
            < return_indexes[1]
        )

        self.check(
            condition,
            "HISTORY-RUNTIME-T4 Record before return",
            (
                "Progress history is recorded before "
                "each completed Tutor turn returns."
            ),
        )

    # =========================================================
    # GATEWAY SEMANTICS
    # =========================================================

    def test_t5_initial_record(
        self,
    ) -> None:

        tutor = AITutor()

        tutor.state.next_turn()

        record = (
            tutor._record_learner_progress_history(
                starts_new_sequence=True
            )
        )

        self.check(
            (
                tutor.learner_progress_history
                .count()
                == 1
                and
                record.history_index == 1
                and
                record.learning_sequence_id == 1
                and
                record.starts_new_sequence
                and
                record.snapshot.session_turn == 1
            ),
            "HISTORY-RUNTIME-T5 Initial record",
            (
                "The first completed turn creates "
                "the first record and first learning "
                "sequence."
            ),
        )

    def test_t6_evaluated_record(
        self,
    ) -> None:

        tutor = AITutor()

        tutor.state.next_turn()

        tutor._record_learner_progress_history(
            starts_new_sequence=True
        )

        tutor.state.next_turn()

        tutor.state.set_evaluation(
            "partial"
        )

        tutor.last_evaluation_result = (
            SimpleNamespace(
                classification="partial",
                confidence=0.8,
                reason="controlled",
                misconception=None,
            )
        )

        record = (
            tutor._record_learner_progress_history(
                starts_new_sequence=False
            )
        )

        snapshot = record.snapshot

        self.check(
            (
                record.learning_sequence_id == 1
                and
                snapshot.session_turn == 2
                and
                snapshot.sequence_last_evaluation
                == "partial"
                and
                snapshot.current_turn_evaluation
                == "partial"
                and
                snapshot.evaluation_performed
                and
                snapshot.attempt_count == 1
            ),
            "HISTORY-RUNTIME-T6 Evaluated turn",
            (
                "An evaluated learner answer is "
                "recorded in the active learning "
                "sequence with current-turn "
                "evaluation telemetry."
            ),
        )

    def test_t7_follow_up_same_sequence(
        self,
    ) -> None:

        tutor = AITutor()

        tutor.state.next_turn()

        tutor._record_learner_progress_history(
            starts_new_sequence=True
        )

        tutor.state.next_turn()

        tutor.state.set_evaluation(
            "partial"
        )

        tutor.last_evaluation_result = (
            SimpleNamespace(
                classification="partial",
                confidence=0.8,
                reason="controlled",
                misconception=None,
            )
        )

        tutor._record_learner_progress_history(
            starts_new_sequence=False
        )

        tutor.state.next_turn()

        intent = (
            tutor.learner_turn_intent_guard
            .evaluate(
                "เบสทำหน้าที่อะไร"
            )
        )

        routing = (
            tutor.learner_turn_routing_policy
            .decide(
                intent
            )
        )

        tutor.last_learner_turn_intent = (
            intent
        )

        tutor.last_learner_turn_routing = (
            routing
        )

        tutor.last_evaluation_result = None

        record = (
            tutor._record_learner_progress_history(
                starts_new_sequence=False
            )
        )

        snapshot = record.snapshot

        self.check(
            (
                record.learning_sequence_id == 1
                and
                not record.starts_new_sequence
                and
                snapshot.session_turn == 3
                and
                snapshot.learner_turn_intent
                == "follow_up_question"
                and
                snapshot.current_turn_evaluation
                is None
                and
                not snapshot.evaluation_performed
                and
                snapshot.sequence_last_evaluation
                == "partial"
            ),
            "HISTORY-RUNTIME-T7 Follow-up sequence",
            (
                "A follow-up question remains in "
                "the same learning sequence without "
                "being falsely recorded as an "
                "evaluated answer."
            ),
        )

    def test_t8_topic_change_new_sequence(
        self,
    ) -> None:

        tutor = AITutor()

        tutor.state.next_turn()

        tutor._record_learner_progress_history(
            starts_new_sequence=True
        )

        tutor.state.next_turn()

        intent = (
            tutor.learner_turn_intent_guard
            .evaluate(
                (
                    "ขอถามอีกเรื่อง "
                    "ตัวเก็บประจุทำงานอย่างไร"
                )
            )
        )

        routing = (
            tutor.learner_turn_routing_policy
            .decide(
                intent
            )
        )

        tutor.last_learner_turn_intent = (
            intent
        )

        tutor.last_learner_turn_routing = (
            routing
        )

        tutor.last_evaluation_result = None

        tutor.state.reset_learning_sequence()

        record = (
            tutor._record_learner_progress_history(
                starts_new_sequence=True
            )
        )

        snapshot = record.snapshot

        self.check(
            (
                record.learning_sequence_id == 2
                and
                record.starts_new_sequence
                and
                snapshot.session_turn == 2
                and
                snapshot.learner_turn_route
                == "topic_change"
                and
                snapshot.scaffolding_level == 1
                and
                snapshot.attempt_count == 0
                and
                snapshot.sequence_last_evaluation
                is None
            ),
            "HISTORY-RUNTIME-T8 Topic boundary",
            (
                "A topic change starts a new history "
                "sequence and records reset "
                "pedagogical state."
            ),
        )

    # =========================================================
    # SAFETY / OBSERVATION ONLY
    # =========================================================

    def test_t9_duplicate_rejected(
        self,
    ) -> None:

        tutor = AITutor()

        tutor.state.next_turn()

        tutor._record_learner_progress_history(
            starts_new_sequence=True
        )

        raised = False

        try:

            tutor._record_learner_progress_history(
                starts_new_sequence=False
            )

        except ValueError:

            raised = True

        self.check(
            raised,
            "HISTORY-RUNTIME-T9 Duplicate guard",
            (
                "Recording the same learner turn "
                "twice is rejected."
            ),
        )

    def test_t10_debug_does_not_record(
        self,
    ) -> None:

        tutor = AITutor()

        tutor.state.next_turn()

        tutor._record_learner_progress_history(
            starts_new_sequence=True
        )

        before = (
            tutor.learner_progress_history
            .count()
        )

        tutor.get_debug_info()
        tutor.get_debug_info()
        tutor.get_debug_info()

        after = (
            tutor.learner_progress_history
            .count()
        )

        self.check(
            before == after == 1,
            "HISTORY-RUNTIME-T10 Debug read-only",
            (
                "Reading debug telemetry never "
                "creates extra progress-history "
                "records."
            ),
        )

    def test_t11_topic_reset_preserves_history(
        self,
    ) -> None:

        tutor = AITutor()

        tutor.state.next_turn()

        tutor._record_learner_progress_history(
            starts_new_sequence=True
        )

        before = (
            tutor.learner_progress_history
            .count()
        )

        tutor.state.reset_learning_sequence()

        after = (
            tutor.learner_progress_history
            .count()
        )

        self.check(
            before == after == 1,
            "HISTORY-RUNTIME-T11 Sequence reset preserves",
            (
                "Resetting pedagogical sequence "
                "state does not erase session-level "
                "progress history."
            ),
        )

    def test_t12_gateway_no_ai_dependency(
        self,
    ) -> None:

        source = inspect.getsource(
            AITutor._record_learner_progress_history
        )

        forbidden = (
            "chat_with_ai",
            "evaluate_response",
            "relevance_gate",
            "knowledge_service",
            "prompt_builder",
            "response_repair",
        )

        condition = not any(
            item in source
            for item in forbidden
        )

        self.check(
            condition,
            "HISTORY-RUNTIME-T12 No AI dependency",
            (
                "Progress-history recording remains "
                "deterministic and observation-only."
            ),
        )

    def test_t13_session_reset_clears_history(
        self,
    ) -> None:

        tutor = AITutor()

        tutor.state.next_turn()

        tutor._record_learner_progress_history(
            starts_new_sequence=True
        )

        before_count = (
            tutor.learner_progress_history
            .count()
        )

        tutor.reset()

        after_count = (
            tutor.learner_progress_history
            .count()
        )

        after_sequence = (
            tutor.learner_progress_history
            .current_sequence_id()
        )

        self.check(
            (
                before_count == 1
                and
                after_count == 0
                and
                after_sequence == 0
            ),
            "HISTORY-RUNTIME-T13 Session reset",
            (
                "Explicit session reset clears "
                "learner progress history and "
                "returns history sequence identity "
                "to zero."
            ),
        )

    def test_t14_debug_history_defaults(
        self,
    ) -> None:

        tutor = AITutor()

        debug = tutor.get_debug_info()

        self.check(
            (
                debug[
                    "learner_progress_history_count"
                ]
                == 0
                and
                debug[
                    "learner_progress_history_sequence_count"
                ]
                == 0
                and
                debug[
                    "learner_progress_history_latest_record"
                ]
                is None
                and
                debug[
                    "learner_progress_history_latest_index"
                ]
                is None
                and
                debug[
                    "learner_progress_history_latest_sequence"
                ]
                is None
                and
                debug[
                    "learner_progress_history_latest_turn"
                ]
                is None
            ),
            "HISTORY-RUNTIME-T14 Debug defaults",
            (
                "Empty progress history exposes "
                "safe zero/None debug telemetry."
            ),
        )


    def test_t15_debug_latest_record(
        self,
    ) -> None:

        tutor = AITutor()

        tutor.state.next_turn()

        tutor._record_learner_progress_history(
            starts_new_sequence=True
        )

        debug = tutor.get_debug_info()

        latest = debug[
            "learner_progress_history_latest_record"
        ]

        self.check(
            (
                debug[
                    "learner_progress_history_count"
                ]
                == 1
                and
                debug[
                    "learner_progress_history_sequence_count"
                ]
                == 1
                and
                latest is not None
                and
                latest.history_index == 1
                and
                debug[
                    "learner_progress_history_latest_index"
                ]
                == 1
                and
                debug[
                    "learner_progress_history_latest_sequence"
                ]
                == 1
                and
                debug[
                    "learner_progress_history_latest_sequence_start"
                ]
                is True
                and
                debug[
                    "learner_progress_history_latest_turn"
                ]
                == 1
            ),
            "HISTORY-RUNTIME-T15 Debug latest",
            (
                "Debug telemetry exposes the latest "
                "typed progress-history record and "
                "aligned flat fields."
            ),
        )


    def test_t16_debug_sequence_count(
        self,
    ) -> None:

        tutor = AITutor()

        tutor.state.next_turn()

        tutor._record_learner_progress_history(
            starts_new_sequence=True
        )

        tutor.state.next_turn()

        tutor._record_learner_progress_history(
            starts_new_sequence=False
        )

        tutor.state.next_turn()

        tutor.state.reset_learning_sequence()

        tutor._record_learner_progress_history(
            starts_new_sequence=True
        )

        debug = tutor.get_debug_info()

        self.check(
            (
                debug[
                    "learner_progress_history_count"
                ]
                == 3
                and
                debug[
                    "learner_progress_history_sequence_count"
                ]
                == 2
                and
                debug[
                    "learner_progress_history_latest_sequence"
                ]
                == 2
                and
                debug[
                    "learner_progress_history_latest_sequence_start"
                ]
                is True
                and
                debug[
                    "learner_progress_history_latest_turn"
                ]
                == 3
            ),
            "HISTORY-RUNTIME-T16 Sequence telemetry",
            (
                "History telemetry distinguishes "
                "record count from learning-sequence "
                "count."
            ),
        )


    def test_t17_debug_history_read_only(
        self,
    ) -> None:

        tutor = AITutor()

        tutor.state.next_turn()

        tutor._record_learner_progress_history(
            starts_new_sequence=True
        )

        before = (
            tutor.learner_progress_history
            .count()
        )

        tutor.get_debug_info()
        tutor.get_debug_info()
        tutor.get_debug_info()

        after = (
            tutor.learner_progress_history
            .count()
        )

        self.check(
            before == after == 1,
            "HISTORY-RUNTIME-T17 Telemetry read-only",
            (
                "Repeated progress-history debug "
                "reads never create history records."
            ),
        )


    def test_t18_debug_history_no_ai_calls(
        self,
    ) -> None:

        tutor = AITutor()

        before = (
            tutor.get_debug_info()[
                "ai_usage"
            ][
                "calls"
            ]
        )

        tutor.get_debug_info()
        tutor.get_debug_info()
        tutor.get_debug_info()

        after = (
            tutor.get_debug_info()[
                "ai_usage"
            ][
                "calls"
            ]
        )

        self.check(
            before == after,
            "HISTORY-RUNTIME-T18 Telemetry no AI calls",
            (
                "Progress-history debug telemetry "
                "adds no LLM calls."
            ),
        )




    # =========================================================
    # RUN
    # =========================================================

    def run(self) -> None:

        print(
            "\nStep 16.19.2 Learning Progress "
            "History Regression Suite\n"
        )

        tests = [
            self.test_t1_two_return_paths,
            self.test_t2_two_gateway_calls,
            self.test_t3_single_turn_increment,
            self.test_t4_gateway_before_returns,
            self.test_t5_initial_record,
            self.test_t6_evaluated_record,
            self.test_t7_follow_up_same_sequence,
            self.test_t8_topic_change_new_sequence,
            self.test_t9_duplicate_rejected,
            self.test_t10_debug_does_not_record,
            self.test_t11_topic_reset_preserves_history,
            self.test_t12_gateway_no_ai_dependency,
            self.test_t13_session_reset_clears_history,
            self.test_t14_debug_history_defaults,
            self.test_t15_debug_latest_record,
            self.test_t16_debug_sequence_count,
            self.test_t17_debug_history_read_only,
            self.test_t18_debug_history_no_ai_calls,
        ]

        for test in tests:

            try:

                test()

            except Exception as error:

                self.failed += 1

                print(
                    f"[FAIL] {test.__name__}"
                )

                print(
                    f"       {error}"
                )

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

    RegressionSuite16_19_2().run()


if __name__ == "__main__":
    main()