from dataclasses import dataclass

from app.services.learner_turn_intent_guard import (
    LearnerTurnIntentGuard,
)
import inspect

from app.services.learner_turn_routing_policy import (
    LearnerTurnRoutingPolicy,
)

from app.tutor import AITutor
from app.state import TutorState
from app.state import TutorState

@dataclass
class TestResult:
    name: str
    passed: bool
    details: str


class RegressionSuite16_18:

    def __init__(
        self,
    ) -> None:

        self.results: list[
            TestResult
        ] = []

        self.guard = (
            LearnerTurnIntentGuard()
        )
        
        self.routing_policy = (
            LearnerTurnRoutingPolicy()
        )



    # =========================================================
    # Helpers
    # =========================================================

    def assert_equal(
        self,
        actual,
        expected,
        message: str,
    ) -> None:

        if actual != expected:

            raise AssertionError(
                f"{message}: "
                f"expected={expected!r}, "
                f"actual={actual!r}"
            )

    def assert_true(
        self,
        condition: bool,
        message: str,
    ) -> None:

        if not condition:

            raise AssertionError(
                message
            )

    def pass_test(
        self,
        name: str,
        details: str,
    ) -> None:

        self.results.append(
            TestResult(
                name=name,
                passed=True,
                details=details,
            )
        )

        print(
            f"[PASS] {name}"
        )

        print(
            f"       {details}"
        )

    def fail_test(
        self,
        name: str,
        details: str,
    ) -> None:

        self.results.append(
            TestResult(
                name=name,
                passed=False,
                details=details,
            )
        )

        print(
            f"[FAIL] {name}"
        )

        print(
            f"       {details}"
        )

    # =========================================================
    # 16.18.1 Intent Guard
    # =========================================================

    def test_intent_t1_answer(
        self,
    ) -> None:

        result = self.guard.evaluate(
            "NPN มี 3 ชั้น คือ n-p-n"
        )

        self.assert_equal(
            result.intent,
            "answer",
            "Statement intent",
        )

        self.assert_equal(
            result.is_question,
            False,
            "Statement question flag",
        )

        self.pass_test(
            "INTENT-T1 Answer",
            (
                "Statement-like learner input "
                "is classified as an answer."
            ),
        )

    def test_intent_t2_follow_up_question(
        self,
    ) -> None:

        result = self.guard.evaluate(
            "เบสทำหน้าที่อะไร"
        )

        self.assert_equal(
            result.intent,
            "follow_up_question",
            "Follow-up question intent",
        )

        self.assert_true(
            result.is_question,
            "Follow-up must be question-like",
        )

        self.pass_test(
            "INTENT-T2 Follow-up question",
            (
                "A learner follow-up question is "
                "not misclassified as an answer."
            ),
        )

    def test_intent_t3_question_mark(
        self,
    ) -> None:

        result = self.guard.evaluate(
            "อิมิตเตอร์อยู่ตรงไหน?"
        )

        self.assert_equal(
            result.intent,
            "follow_up_question",
            "Question-mark intent",
        )

        self.pass_test(
            "INTENT-T3 Question mark",
            (
                "Explicit question punctuation "
                "is detected deterministically."
            ),
        )

    def test_intent_t4_clarification(
        self,
    ) -> None:

        result = self.guard.evaluate(
            "คำว่าไบอัสหมายถึงอะไร"
        )

        self.assert_equal(
            result.intent,
            "clarification_question",
            "Clarification intent",
        )

        self.pass_test(
            "INTENT-T4 Clarification",
            (
                "Explicit clarification requests "
                "are classified separately."
            ),
        )

    def test_intent_t5_topic_change(
        self,
    ) -> None:

        result = self.guard.evaluate(
            (
                "ขอถามอีกเรื่อง "
                "ตัวเก็บประจุทำงานอย่างไร"
            )
        )

        self.assert_equal(
            result.intent,
            "topic_change",
            "Topic-change intent",
        )

        self.pass_test(
            "INTENT-T5 Topic change",
            (
                "Explicit topic-change signals "
                "take precedence over question form."
            ),
        )

    def test_intent_t6_dont_know_answer(
        self,
    ) -> None:

        result = self.guard.evaluate(
            "ไม่ทราบครับ"
        )

        self.assert_equal(
            result.intent,
            "answer",
            "Don't-know intent",
        )

        self.pass_test(
            "INTENT-T6 Don't know",
            (
                "Explicit don't-know input remains "
                "an evaluable learner answer."
            ),
        )

    def test_intent_t7_false_question_guard(
        self,
    ) -> None:

        result = self.guard.evaluate(
            "ไม่รู้อะไรเลยครับ"
        )

        self.assert_equal(
            result.intent,
            "answer",
            (
                "Don't-know text containing "
                "a question word"
            ),
        )

        self.pass_test(
            "INTENT-T7 False-question guard",
            (
                "Question vocabulary inside a "
                "don't-know response does not "
                "cause a false question intent."
            ),
        )

    def test_intent_t8_english_question(
        self,
    ) -> None:

        result = self.guard.evaluate(
            "What is the base region?"
        )

        self.assert_equal(
            result.intent,
            "follow_up_question",
            "English question intent",
        )

        self.pass_test(
            "INTENT-T8 English question",
            (
                "English WH-questions are detected "
                "without an LLM."
            ),
        )

    def test_intent_t9_english_statement(
        self,
    ) -> None:

        result = self.guard.evaluate(
            "Base is p-type"
        )

        self.assert_equal(
            result.intent,
            "answer",
            "English statement intent",
        )

        self.pass_test(
            "INTENT-T9 English statement",
            (
                "English technical statements "
                "remain learner answers."
            ),
        )

    def test_intent_t10_empty(
        self,
    ) -> None:

        result = self.guard.evaluate(
            "   "
        )

        self.assert_equal(
            result.intent,
            "uncertain",
            "Empty input intent",
        )

        self.pass_test(
            "INTENT-T10 Empty",
            (
                "Empty learner input is handled "
                "conservatively."
            ),
        )

    def test_intent_t11_invalid_type(
        self,
    ) -> None:

        raised = False

        try:

            self.guard.evaluate(
                None
            )

        except TypeError:

            raised = True

        self.assert_true(
            raised,
            (
                "Invalid learner input type "
                "must raise TypeError"
            ),
        )

        self.pass_test(
            "INTENT-T11 Invalid type",
            (
                "LearnerTurnIntentGuard rejects "
                "non-string input."
            ),
        )

    def test_intent_t12_deterministic(
        self,
    ) -> None:

        text = (
            "เบสทำหน้าที่อะไร"
        )

        first = self.guard.evaluate(
            text
        )

        second = self.guard.evaluate(
            text
        )

        self.assert_equal(
            first,
            second,
            "Deterministic intent result",
        )

        self.pass_test(
            "INTENT-T12 Deterministic",
            (
                "Identical learner input produces "
                "an identical intent result."
            ),
        )

    def test_intent_runtime_t13_debug_defaults(
        self,
    ) -> None:

        tutor = AITutor()

        debug = tutor.get_debug_info()

        self.assert_equal(
            debug.get(
                "learner_turn_intent"
            ),
            None,
            "Initial learner intent",
        )

        self.assert_equal(
            debug.get(
                "learner_turn_intent_confidence"
            ),
            0.0,
            "Initial learner intent confidence",
        )

        self.assert_equal(
            debug.get(
                "learner_turn_intent_is_question"
            ),
            False,
            "Initial learner question flag",
        )

        self.assert_equal(
            debug.get(
                "learner_turn_intent_signals"
            ),
            (),
            "Initial learner intent signals",
        )

        self.pass_test(
            "INTENT-RUNTIME-T13 Debug defaults",
            (
                "AITutor exposes safe default "
                "learner-turn intent telemetry."
            ),
        )

    def test_intent_runtime_t14_debug_telemetry(
        self,
    ) -> None:

        tutor = AITutor()

        result = self.guard.evaluate(
            "เบสทำหน้าที่อะไร"
        )

        tutor.last_learner_turn_intent = (
            result
        )

        debug = tutor.get_debug_info()

        self.assert_equal(
            debug.get(
                "learner_turn_intent"
            ),
            "follow_up_question",
            "Runtime learner intent",
        )

        self.assert_equal(
            debug.get(
                "learner_turn_intent_confidence"
            ),
            0.90,
            "Runtime learner intent confidence",
        )

        self.assert_equal(
            debug.get(
                "learner_turn_intent_is_question"
            ),
            True,
            "Runtime question flag",
        )

        self.assert_true(
            "thai_question:อะไร"
            in debug.get(
                "learner_turn_intent_signals"
            ),
            "Runtime intent signals",
        )

        self.pass_test(
            "INTENT-RUNTIME-T14 Debug telemetry",
            (
                "AITutor exposes deterministic "
                "learner-turn intent telemetry."
            ),
        )

    def test_intent_runtime_t15_reset_clears(
        self,
    ) -> None:

        tutor = AITutor()

        tutor.last_learner_turn_intent = (
            self.guard.evaluate(
                "เบสทำหน้าที่อะไร"
            )
        )

        self.assert_true(
            tutor.last_learner_turn_intent
            is not None,
            "Pre-reset learner intent",
        )

        tutor.reset()

        self.assert_equal(
            tutor.last_learner_turn_intent,
            None,
            "Post-reset learner intent",
        )

        self.assert_equal(
            tutor.get_debug_info().get(
                "learner_turn_intent"
            ),
            None,
            "Post-reset debug intent",
        )

        self.pass_test(
            "INTENT-RUNTIME-T15 Reset clears",
            (
                "Conversation reset clears "
                "learner-turn intent telemetry."
            ),
        )

    def test_intent_runtime_t16_pipeline_order(
        self,
    ) -> None:

        source = inspect.getsource(
            AITutor.respond
        )

        waiting_index = source.find(
            "if not self.waiting_for_response"
        )

        intent_index = source.find(
            "self.learner_turn_intent_guard"
        )

        evaluator_index = source.find(
            "evaluate_response("
        )

        self.assert_true(
            waiting_index >= 0,
            "Waiting-response branch exists",
        )

        self.assert_true(
            intent_index >= 0,
            "Intent guard integration exists",
        )

        self.assert_true(
            evaluator_index >= 0,
            "Evaluator integration exists",
        )

        self.assert_true(
            waiting_index
            < intent_index
            < evaluator_index,
            (
                "Learner-turn intent must be "
                "observed before evaluation"
            ),
        )

        self.pass_test(
            "INTENT-RUNTIME-T16 Pipeline order",
            (
                "Learner-turn intent is observed "
                "before learner-response evaluation."
            ),
        )

    def test_intent_runtime_t17_no_llm_calls(
        self,
    ) -> None:

        tutor = AITutor()

        before = (
            tutor.get_debug_info()[
                "ai_usage"
            ]["calls"]
        )

        tutor.last_learner_turn_intent = (
            self.guard.evaluate(
                "เบสทำหน้าที่อะไร"
            )
        )

        _ = tutor.get_debug_info()

        after = (
            tutor.get_debug_info()[
                "ai_usage"
            ]["calls"]
        )

        self.assert_equal(
            after,
            before,
            (
                "Learner-turn intent telemetry "
                "must not add AI calls"
            ),
        )

        self.pass_test(
            "INTENT-RUNTIME-T17 No LLM calls",
            (
                "Learner-turn intent telemetry "
                "remains fully deterministic."
            ),
        )

    def test_routing_t18_answer(
        self,
    ) -> None:

        intent = self.guard.evaluate(
            "NPN มี 3 ชั้น คือ n-p-n"
        )

        result = self.routing_policy.decide(
            intent
        )

        self.assert_equal(
            result.route,
            "evaluate_answer",
            "Answer route",
        )

        self.assert_equal(
            result.should_evaluate_response,
            True,
            "Answer evaluation routing",
        )

        self.assert_equal(
            result.replace_active_question,
            False,
            "Answer active-question routing",
        )

        self.pass_test(
            "ROUTING-T18 Answer",
            (
                "Normal learner answers remain "
                "routed to the Evaluator."
            ),
        )

    def test_routing_t19_follow_up(
        self,
    ) -> None:

        intent = self.guard.evaluate(
            "เบสทำหน้าที่อะไร"
        )

        result = self.routing_policy.decide(
            intent
        )

        self.assert_equal(
            result.route,
            "follow_up_question",
            "Follow-up route",
        )

        self.assert_equal(
            result.should_evaluate_response,
            False,
            "Follow-up evaluator bypass",
        )

        self.assert_equal(
            result.use_current_message_for_retrieval,
            True,
            "Follow-up retrieval routing",
        )

        self.assert_equal(
            result.replace_active_question,
            True,
            "Follow-up active question",
        )

        self.pass_test(
            "ROUTING-T19 Follow-up",
            (
                "Follow-up questions bypass learner "
                "answer evaluation."
            ),
        )

    def test_routing_t20_clarification(
        self,
    ) -> None:

        intent = self.guard.evaluate(
            "คำว่าไบอัสหมายถึงอะไร"
        )

        result = self.routing_policy.decide(
            intent
        )

        self.assert_equal(
            result.route,
            "clarification_question",
            "Clarification route",
        )

        self.assert_equal(
            result.should_evaluate_response,
            False,
            "Clarification evaluator bypass",
        )

        self.assert_equal(
            result.include_previous_tutor_context,
            True,
            "Clarification context routing",
        )

        self.assert_equal(
            result.replace_active_question,
            False,
            "Clarification sequence preservation",
        )

        self.pass_test(
            "ROUTING-T20 Clarification",
            (
                "Clarification requests bypass evaluation "
                "while retaining tutoring context."
            ),
        )

    def test_routing_t21_topic_change(
        self,
    ) -> None:

        intent = self.guard.evaluate(
            (
                "ขอถามอีกเรื่อง "
                "ตัวเก็บประจุทำงานอย่างไร"
            )
        )

        result = self.routing_policy.decide(
            intent
        )

        self.assert_equal(
            result.route,
            "topic_change",
            "Topic-change route",
        )

        self.assert_equal(
            result.should_evaluate_response,
            False,
            "Topic-change evaluator bypass",
        )

        self.assert_equal(
            result.include_previous_tutor_context,
            False,
            "Topic-change context isolation",
        )

        self.assert_equal(
            result.replace_active_question,
            True,
            "Topic-change active question",
        )

        self.pass_test(
            "ROUTING-T21 Topic change",
            (
                "Explicit topic changes isolate retrieval "
                "from the previous tutoring question."
            ),
        )

    def test_routing_t22_uncertain(
        self,
    ) -> None:

        intent = self.guard.evaluate(
            "ก"
        )

        result = self.routing_policy.decide(
            intent
        )

        self.assert_equal(
            result.route,
            "evaluate_uncertain",
            "Uncertain route",
        )

        self.assert_equal(
            result.should_evaluate_response,
            True,
            "Uncertain fail-safe routing",
        )

        self.pass_test(
            "ROUTING-T22 Uncertain",
            (
                "Uncertain learner intent preserves "
                "existing evaluation behavior."
            ),
        )

    def test_routing_t23_invalid_type(
        self,
    ) -> None:

        raised = False

        try:

            self.routing_policy.decide(
                "invalid"
            )

        except TypeError:

            raised = True

        self.assert_true(
            raised,
            (
                "Routing policy must reject "
                "invalid input types"
            ),
        )

        self.pass_test(
            "ROUTING-T23 Invalid type",
            (
                "LearnerTurnRoutingPolicy rejects "
                "non-intent input."
            ),
        )

    def test_routing_t24_deterministic(
        self,
    ) -> None:

        intent = self.guard.evaluate(
            "เบสทำหน้าที่อะไร"
        )

        first = self.routing_policy.decide(
            intent
        )

        second = self.routing_policy.decide(
            intent
        )

        self.assert_equal(
            first,
            second,
            "Deterministic routing result",
        )

        self.pass_test(
            "ROUTING-T24 Deterministic",
            (
                "Identical learner intent produces "
                "an identical routing decision."
            ),
        )

    def test_routing_runtime_t25_follow_up_bypass(
        self,
    ) -> None:

        intent = self.guard.evaluate(
            "เบสทำหน้าที่อะไร"
        )

        routing = (
            self.routing_policy.decide(
                intent
            )
        )

        self.assert_equal(
            intent.intent,
            "follow_up_question",
            "Follow-up intent",
        )

        self.assert_equal(
            routing.route,
            "follow_up_question",
            "Follow-up route",
        )

        self.assert_equal(
            routing.should_evaluate_response,
            False,
            "Follow-up evaluator bypass",
        )

        self.assert_equal(
            routing.use_current_message_for_retrieval,
            True,
            "Follow-up current retrieval",
        )

        self.pass_test(
            "ROUTING-RUNTIME-T25 Follow-up bypass",
            (
                "Follow-up learner questions bypass "
                "answer evaluation and use the current "
                "message for retrieval."
            ),
        )

    def test_routing_runtime_t26_evaluator_guarded(
        self,
    ) -> None:

        source = inspect.getsource(
            AITutor.respond
        )

        routing_index = source.find(
            "should_evaluate_response"
        )

        evaluator_index = source.find(
            "evaluate_response("
        )

        self.assert_true(
            routing_index >= 0,
            (
                "Routing evaluation condition "
                "must exist"
            ),
        )

        self.assert_true(
            evaluator_index >= 0,
            "Evaluator call must exist",
        )

        self.assert_true(
            routing_index
            < evaluator_index,
            (
                "Routing condition must be applied "
                "before evaluate_response()"
            ),
        )

        self.pass_test(
            "ROUTING-RUNTIME-T26 Evaluator guarded",
            (
                "AITutor guards learner-response "
                "evaluation with the routing decision."
            ),
        )

    def test_routing_runtime_t27_bypass_clears_stale_state(
        self,
    ) -> None:

        source = inspect.getsource(
            AITutor.respond
        )

        self.assert_true(
            "self.last_evaluation_result = None"
            in source,
            (
                "Bypass path must clear stale "
                "evaluation telemetry"
            ),
        )

        self.assert_true(
            "self.last_decision = None"
            in source,
            (
                "Bypass path must clear stale "
                "scaffolding decision telemetry"
            ),
        )

        self.pass_test(
            "ROUTING-RUNTIME-T27 Clear stale state",
            (
                "Question-like learner turns do not "
                "reuse evaluation or scaffolding "
                "decision telemetry from a prior turn."
            ),
        )

    def test_routing_runtime_t28_active_question_replacement(
        self,
    ) -> None:

        intent = self.guard.evaluate(
            "เบสทำหน้าที่อะไร"
        )

        routing = (
            self.routing_policy.decide(
                intent
            )
        )

        self.assert_equal(
            routing.replace_active_question,
            True,
            (
                "Follow-up active-question "
                "replacement"
            ),
        )

        source = inspect.getsource(
            AITutor.respond
        )

        replace_index = source.find(
            "replace_active_question"
        )

        original_index = source.find(
            "self.original_question = (",
            replace_index,
        )

        self.assert_true(
            replace_index >= 0,
            (
                "Active-question routing condition "
                "must exist"
            ),
        )

        self.assert_true(
            original_index > replace_index,
            (
                "Active learner question must be "
                "replaceable after routing"
            ),
        )

        self.pass_test(
            "ROUTING-RUNTIME-T28 Active question",
            (
                "Follow-up routing can promote the "
                "current learner question to the "
                "active tutoring question."
            ),
        )

    def test_routing_runtime_t29_topic_isolation(
        self,
    ) -> None:

        intent = self.guard.evaluate(
            (
                "ขอถามอีกเรื่อง "
                "ตัวเก็บประจุทำงานอย่างไร"
            )
        )

        routing = (
            self.routing_policy.decide(
                intent
            )
        )

        self.assert_equal(
            routing.route,
            "topic_change",
            "Topic-change route",
        )

        self.assert_equal(
            routing.use_current_message_for_retrieval,
            True,
            "Topic-change current retrieval",
        )

        self.assert_equal(
            routing.include_previous_tutor_context,
            False,
            "Topic-change context isolation",
        )

        self.assert_equal(
            routing.replace_active_question,
            True,
            "Topic-change active question",
        )

        self.pass_test(
            "ROUTING-RUNTIME-T29 Topic isolation",
            (
                "Topic changes use the new learner "
                "message without previous Tutor "
                "retrieval contamination."
            ),
        )

    def test_routing_runtime_t30_clarification_context(
        self,
    ) -> None:

        intent = self.guard.evaluate(
            "คำว่าไบอัสหมายถึงอะไร"
        )

        routing = (
            self.routing_policy.decide(
                intent
            )
        )

        self.assert_equal(
            routing.route,
            "clarification_question",
            "Clarification route",
        )

        self.assert_equal(
            routing.should_evaluate_response,
            False,
            "Clarification evaluator bypass",
        )

        self.assert_equal(
            routing.use_current_message_for_retrieval,
            True,
            "Clarification current retrieval",
        )

        self.assert_equal(
            routing.include_previous_tutor_context,
            True,
            "Clarification previous context",
        )

        self.assert_equal(
            routing.replace_active_question,
            False,
            "Clarification active question",
        )

        self.pass_test(
            "ROUTING-RUNTIME-T30 Clarification context",
            (
                "Clarification requests bypass "
                "evaluation while preserving local "
                "Tutor context."
            ),
        )

    def test_routing_runtime_t31_intervention_guard(
        self,
    ) -> None:

        source = inspect.getsource(
            AITutor.respond
        )

        compact_source = (
            " ".join(
                source.split()
            )
        )

        self.assert_true(
            "evaluation_performed_this_turn"
            in source,
            (
                "Per-turn evaluation flag "
                "must exist"
            ),
        )

        self.assert_true(
            (
                "get_intervention( "
                "intervention_evaluation )"
            )
            in compact_source,
            (
                "get_intervention() must use "
                "the current-turn evaluation "
                "instead of stale TutorState."
            ),
        )

        self.pass_test(
            "ROUTING-RUNTIME-T31 Intervention guard",
            (
                "Question-like turns do not reuse "
                "an intervention derived from a "
                "previous learner evaluation."
            ),
        )

    def test_routing_runtime_t32_no_llm_calls(
        self,
    ) -> None:

        tutor = AITutor()

        before = (
            tutor.get_debug_info()[
                "ai_usage"
            ]["calls"]
        )

        intent = self.guard.evaluate(
            "เบสทำหน้าที่อะไร"
        )

        routing = (
            self.routing_policy.decide(
                intent
            )
        )

        tutor.last_learner_turn_intent = (
            intent
        )

        tutor.last_learner_turn_routing = (
            routing
        )

        debug = tutor.get_debug_info()

        self.assert_equal(
            debug.get(
                "learner_turn_route"
            ),
            "follow_up_question",
            "Routing debug telemetry",
        )

        self.assert_equal(
            debug.get(
                "learner_turn_should_evaluate"
            ),
            False,
            "Routing evaluation telemetry",
        )

        after = (
            tutor.get_debug_info()[
                "ai_usage"
            ]["calls"]
        )

        self.assert_equal(
            after,
            before,
            (
                "Routing and routing telemetry "
                "must not add AI calls"
            ),
        )

        self.pass_test(
            "ROUTING-RUNTIME-T32 No LLM calls",
            (
                "Learner-turn routing and telemetry "
                "remain fully deterministic."
            ),
        )

    def test_routing_runtime_t33_evaluation_telemetry(
        self,
    ) -> None:

        tutor = AITutor()

        self.assert_equal(
            tutor.get_debug_info()["evaluation"],
            None,
            "Initial evaluation telemetry",
        )

        # Simulate stale accumulated state from an older turn.
        tutor.state.last_evaluation = "partial"

        # Current turn itself has no evaluation result.
        tutor.last_evaluation_result = None

        debug = tutor.get_debug_info()

        self.assert_equal(
            debug["evaluation"],
            None,
            (
                "Debug evaluation must represent "
                "the current turn, not stale "
                "LearnerState."
            ),
        )

        self.assert_equal(
            debug["evaluation_confidence"],
            None,
            "Current-turn evaluation confidence",
        )

        self.assert_equal(
            debug["evaluation_reason"],
            None,
            "Current-turn evaluation reason",
        )

        self.pass_test(
            "ROUTING-RUNTIME-T33 Per-turn evaluation",
            (
                "Evaluation telemetry does not leak "
                "classification from a previous "
                "learner turn."
            ),
        )

    def test_routing_runtime_t34_sequence_reset(
        self,
    ) -> None:

        state = TutorState()

        state.next_turn()
        state.next_turn()

        state.scaffolding_level = 4
        state.last_evaluation = "partial"

        state.correct_streak = 1
        state.partial_streak = 2
        state.failure_streak = 3

        state.attempt_count = 5
        state.hint_count = 2

        turn_before = state.turn_count

        state.reset_learning_sequence()

        self.assert_equal(
            state.scaffolding_level,
            1,
            "Scaffolding reset",
        )

        self.assert_equal(
            state.last_evaluation,
            None,
            "Evaluation reset",
        )

        self.assert_equal(
            state.correct_streak,
            0,
            "Correct streak reset",
        )

        self.assert_equal(
            state.partial_streak,
            0,
            "Partial streak reset",
        )

        self.assert_equal(
            state.failure_streak,
            0,
            "Failure streak reset",
        )

        self.assert_equal(
            state.attempt_count,
            0,
            "Attempt reset",
        )

        self.assert_equal(
            state.hint_count,
            0,
            "Hint reset",
        )

        self.assert_equal(
            state.turn_count,
            turn_before,
            "Session turn preserved",
        )

        self.pass_test(
            "ROUTING-RUNTIME-T34 Sequence reset",
            (
                "A topic sequence reset clears "
                "pedagogical state while preserving "
                "session turn history."
            ),
        )

    def test_routing_runtime_t35_history_preserved(
        self,
    ) -> None:

        state = TutorState()

        marker = object()

        state.misconceptions.append(
            marker
        )

        state.next_turn()

        state.reset_learning_sequence()

        self.assert_equal(
            state.turn_count,
            1,
            "Turn history preserved",
        )

        self.assert_equal(
            len(state.misconceptions),
            1,
            "Misconception history preserved",
        )

        self.assert_true(
            state.misconceptions[0]
            is marker,
            (
                "Topic reset must not clear "
                "session-level misconception history."
            ),
        )

        self.pass_test(
            "ROUTING-RUNTIME-T35 History preserved",
            (
                "Topic-sequence reset preserves "
                "session-level learner history."
            ),
        )

    def test_routing_runtime_t36_topic_reset_integration(
        self,
    ) -> None:

        source = inspect.getsource(
            AITutor.respond
        )

        compact_source = (
            " ".join(
                source.split()
            )
        )

        self.assert_true(
            (
                'learner_turn_routing.route '
                '== "topic_change"'
            )
            in compact_source,
            (
                "AITutor must explicitly detect "
                "topic-change routing."
            ),
        )

        self.assert_true(
            (
                "self.state."
                "reset_learning_sequence()"
            )
            in compact_source,
            (
                "Topic change must reset the "
                "active pedagogical sequence."
            ),
        )

        topic_index = compact_source.find(
            'learner_turn_routing.route '
            '== "topic_change"'
        )

        reset_index = compact_source.find(
            "self.state.reset_learning_sequence()"
        )

        self.assert_true(
            topic_index >= 0
            and
            reset_index > topic_index,
            (
                "Sequence reset must occur after "
                "the topic-change routing condition."
            ),
        )

        self.pass_test(
            "ROUTING-RUNTIME-T36 Topic reset integration",
            (
                "AITutor starts a fresh pedagogical "
                "sequence only for explicit "
                "topic changes."
            ),
        )



    # =========================================================
    # Runner
    # =========================================================

    def run(
        self,
    ) -> int:

        tests = [
            self.test_intent_t1_answer,
            self.test_intent_t2_follow_up_question,
            self.test_intent_t3_question_mark,
            self.test_intent_t4_clarification,
            self.test_intent_t5_topic_change,
            self.test_intent_t6_dont_know_answer,
            self.test_intent_t7_false_question_guard,
            self.test_intent_t8_english_question,
            self.test_intent_t9_english_statement,
            self.test_intent_t10_empty,
            self.test_intent_t11_invalid_type,
            self.test_intent_t12_deterministic,

            self.test_intent_runtime_t13_debug_defaults,
            self.test_intent_runtime_t14_debug_telemetry,
            self.test_intent_runtime_t15_reset_clears,
            self.test_intent_runtime_t16_pipeline_order,
            self.test_intent_runtime_t17_no_llm_calls,

            self.test_routing_t18_answer,
            self.test_routing_t19_follow_up,
            self.test_routing_t20_clarification,
            self.test_routing_t21_topic_change,
            self.test_routing_t22_uncertain,
            self.test_routing_t23_invalid_type,
            self.test_routing_t24_deterministic,

            self.test_routing_runtime_t25_follow_up_bypass,
            self.test_routing_runtime_t26_evaluator_guarded,
            self.test_routing_runtime_t27_bypass_clears_stale_state,
            self.test_routing_runtime_t28_active_question_replacement,
            self.test_routing_runtime_t29_topic_isolation,
            self.test_routing_runtime_t30_clarification_context,
            self.test_routing_runtime_t31_intervention_guard,
            self.test_routing_runtime_t32_no_llm_calls,
            self.test_routing_runtime_t33_evaluation_telemetry,
            self.test_routing_runtime_t34_sequence_reset,
            self.test_routing_runtime_t35_history_preserved,
            self.test_routing_runtime_t36_topic_reset_integration,
        ]

        print(
            "\nStep 16.18 Learner Turn Intent "
            "Regression Suite\n"
        )

        for test in tests:

            try:

                test()

            except Exception as exc:

                self.fail_test(
                    name=test.__name__,
                    details=str(exc),
                )

        self.print_summary()

        return (
            0
            if all(
                item.passed
                for item in self.results
            )
            else 1
        )

    def print_summary(
        self,
    ) -> None:

        passed = sum(
            item.passed
            for item in self.results
        )

        failed = sum(
            not item.passed
            for item in self.results
        )

        print(
            "\nSUMMARY"
        )

        print(
            f"Passed: {passed}"
        )

        print(
            f"Failed: {failed}"
        )


def main() -> None:

    suite = (
        RegressionSuite16_18()
    )

    raise SystemExit(
        suite.run()
    )


if __name__ == "__main__":
    main()