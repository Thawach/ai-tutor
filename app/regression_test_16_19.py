from dataclasses import FrozenInstanceError

from app.services.tutoring_response_mode_models import (
    TutoringResponseModeDecision,
)

import inspect

from app.services.learner_turn_intent_models import (
    LearnerTurnIntentResult,
)

from app.services.tutoring_response_strategy_policy import (
    TutoringResponseStrategyPolicy,
)
from app.tutor import AITutor

from app.services.tutoring_response_mode_instruction_builder import (
    TutoringResponseModeInstructionBuilder,
)

from app.prompts.builder import (
    PromptBuilder,
)

from app.domain.scaffolding.strategies import (
    get_strategy,
)

from app.domain.scaffolding.interventions import (
    get_intervention,
)

from app.services.response_mode_pedagogical_precheck import (
    ResponseModePedagogicalPrecheck,
)

from unittest.mock import Mock, patch

from app.services.response_mode_pedagogical_validator import (
    ResponseModePedagogicalValidator,
    RESPONSE_MODE_PEDAGOGICAL_VALIDATION_PROMPT,
)

from app.services.pedagogical_validation_models import (
    PedagogicalValidationResult,
)

from app.services.response_mode_pedagogical_precheck import (
    ResponseModePedagogicalPrecheck,
)

from app.services.response_mode_pedagogical_validator import (
    ResponseModePedagogicalValidator,
)

from app.services.response_mode_repair_service import (
    ResponseModeRepairService,
    RESPONSE_MODE_REPAIR_PROMPT,
)

from app.services.response_repair_service import (
    ResponseRepairService,
)

from app.services.grounding_validation_models import (
    GroundingValidationResult,
)

from app.services.response_mode_repair_service import (
    ResponseModeRepairService,
)

class Step1619RegressionSuite:

    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0

    def pass_test(
        self,
        name: str,
        description: str,
    ) -> None:

        self.passed += 1

        print(
            f"[PASS] {name}"
        )
        print(
            f"       {description}"
        )

    def fail_test(
        self,
        name: str,
        error: Exception | str,
    ) -> None:

        self.failed += 1

        print(
            f"[FAIL] {name}"
        )
        print(
            f"       {error}"
        )

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

    # =====================================================
    # 16.19.1 Typed Response Mode Model
    # =====================================================

    def test_mode_t1_scaffolded(
        self,
    ) -> None:

        decision = (
            TutoringResponseModeDecision(
                mode="scaffolded",
                reason=(
                    "Normal evaluated learner answer."
                ),
                direct_answer_required=False,
                guiding_question_policy="required",
                preserve_scaffolding_strategy=True,
                max_guiding_questions=1,
            )
        )

        self.assert_equal(
            decision.mode,
            "scaffolded",
            "Response mode",
        )

        self.assert_equal(
            decision.guiding_question_policy,
            "required",
            "Question policy",
        )

        self.assert_true(
            decision.preserve_scaffolding_strategy,
            (
                "Normal scaffolded mode must "
                "preserve the active strategy."
            ),
        )

        self.pass_test(
            "MODE-T1 Scaffolded",
            (
                "Normal tutoring behavior can be "
                "represented by the typed model."
            ),
        )

    def test_mode_t2_clarification(
        self,
    ) -> None:

        decision = (
            TutoringResponseModeDecision(
                mode="direct_clarification",
                reason=(
                    "Learner requested clarification."
                ),
                direct_answer_required=True,
                guiding_question_policy="optional",
                preserve_scaffolding_strategy=False,
                max_guiding_questions=1,
            )
        )

        self.assert_true(
            decision.direct_answer_required,
            (
                "Clarification mode must support "
                "a direct-answer requirement."
            ),
        )

        self.assert_equal(
            decision.guiding_question_policy,
            "optional",
            (
                "Clarification must be able to "
                "make guiding questions optional."
            ),
        )

        self.pass_test(
            "MODE-T2 Clarification",
            (
                "Direct clarification behavior is "
                "represented explicitly."
            ),
        )

    def test_mode_t3_follow_up(
        self,
    ) -> None:

        decision = (
            TutoringResponseModeDecision(
                mode="answer_then_guide",
                reason=(
                    "Learner asked a follow-up question."
                ),
                direct_answer_required=True,
                guiding_question_policy="optional",
                preserve_scaffolding_strategy=False,
                max_guiding_questions=1,
            )
        )

        self.assert_equal(
            decision.mode,
            "answer_then_guide",
            "Follow-up response mode",
        )

        self.assert_true(
            decision.direct_answer_required,
            (
                "Follow-up questions must support "
                "answer-first behavior."
            ),
        )

        self.pass_test(
            "MODE-T3 Follow-up",
            (
                "Answer-first follow-up behavior is "
                "represented explicitly."
            ),
        )

    def test_mode_t4_frozen(
        self,
    ) -> None:

        decision = (
            TutoringResponseModeDecision(
                mode="scaffolded",
                reason="Frozen model test.",
                direct_answer_required=False,
                guiding_question_policy="required",
                preserve_scaffolding_strategy=True,
                max_guiding_questions=1,
            )
        )

        raised = False

        try:
            decision.mode = (
                "direct_clarification"
            )

        except FrozenInstanceError:
            raised = True

        self.assert_true(
            raised,
            (
                "Response-mode decisions must "
                "be immutable."
            ),
        )

        self.pass_test(
            "MODE-T4 Frozen",
            (
                "Response-mode decisions cannot "
                "be mutated after creation."
            ),
        )

    def test_mode_t5_invalid_mode(
        self,
    ) -> None:

        raised = False

        try:
            TutoringResponseModeDecision(
                mode="unknown",
                reason="Invalid mode test.",
                direct_answer_required=False,
                guiding_question_policy="optional",
                preserve_scaffolding_strategy=False,
            )

        except ValueError:
            raised = True

        self.assert_true(
            raised,
            (
                "Unknown response modes must "
                "fail fast."
            ),
        )

        self.pass_test(
            "MODE-T5 Invalid mode",
            (
                "Unsupported response modes are "
                "rejected deterministically."
            ),
        )

    def test_mode_t6_invalid_question_policy(
        self,
    ) -> None:

        raised = False

        try:
            TutoringResponseModeDecision(
                mode="scaffolded",
                reason="Invalid policy test.",
                direct_answer_required=False,
                guiding_question_policy="sometimes",
                preserve_scaffolding_strategy=True,
            )

        except ValueError:
            raised = True

        self.assert_true(
            raised,
            (
                "Unknown question policies must "
                "fail fast."
            ),
        )

        self.pass_test(
            "MODE-T6 Invalid question policy",
            (
                "Unsupported guiding-question "
                "policies are rejected."
            ),
        )

    def test_mode_t7_deterministic(
        self,
    ) -> None:

        first = (
            TutoringResponseModeDecision(
                mode="direct_clarification",
                reason="Same deterministic input.",
                direct_answer_required=True,
                guiding_question_policy="optional",
                preserve_scaffolding_strategy=False,
                max_guiding_questions=1,
            )
        )

        second = (
            TutoringResponseModeDecision(
                mode="direct_clarification",
                reason="Same deterministic input.",
                direct_answer_required=True,
                guiding_question_policy="optional",
                preserve_scaffolding_strategy=False,
                max_guiding_questions=1,
            )
        )

        self.assert_equal(
            first,
            second,
            (
                "Identical typed inputs must "
                "produce identical decisions."
            ),
        )

        self.pass_test(
            "MODE-T7 Deterministic",
            (
                "Typed response-mode construction "
                "is deterministic and contains "
                "no LLM dependency."
            ),
        )

    # =====================================================
    # 16.19.2 Deterministic Response Strategy Policy
    # =====================================================

    def test_strategy_t8_initial(
        self,
    ) -> None:

        policy = (
            TutoringResponseStrategyPolicy()
        )

        decision = policy.decide(None)

        self.assert_equal(
            decision.mode,
            "new_topic_scaffold",
            "Initial response mode",
        )

        self.assert_true(
            decision.preserve_scaffolding_strategy,
            (
                "Initial turn must preserve normal "
                "scaffolding behavior."
            ),
        )

        self.pass_test(
            "STRATEGY-T8 Initial",
            (
                "Initial learner turn starts with "
                "normal new-topic scaffolding."
            ),
        )

    def test_strategy_t9_answer(
        self,
    ) -> None:

        policy = (
            TutoringResponseStrategyPolicy()
        )

        intent = LearnerTurnIntentResult(
            intent="answer",
            reason="Test answer.",
        )

        decision = policy.decide(
            intent
        )

        self.assert_equal(
            decision.mode,
            "scaffolded",
            "Answer mode",
        )

        self.assert_equal(
            decision.direct_answer_required,
            False,
            "Direct-answer requirement",
        )

        self.assert_true(
            decision.preserve_scaffolding_strategy,
            (
                "Normal learner answers must "
                "preserve scaffolding behavior."
            ),
        )

        self.pass_test(
            "STRATEGY-T9 Answer",
            (
                "Learner answers remain owned by "
                "the normal scaffolding strategy."
            ),
        )

    def test_strategy_t10_follow_up(
        self,
    ) -> None:

        policy = (
            TutoringResponseStrategyPolicy()
        )

        intent = LearnerTurnIntentResult(
            intent="follow_up_question",
            reason="Test follow-up.",
            is_question=True,
        )

        decision = policy.decide(
            intent
        )

        self.assert_equal(
            decision.mode,
            "answer_then_guide",
            "Follow-up response mode",
        )

        self.assert_true(
            decision.direct_answer_required,
            (
                "Follow-up question requires "
                "an answer-first response."
            ),
        )

        self.assert_equal(
            decision.guiding_question_policy,
            "optional",
            "Follow-up question policy",
        )

        self.assert_equal(
            decision.max_guiding_questions,
            1,
            "Follow-up question limit",
        )

        self.assert_equal(
            decision.preserve_scaffolding_strategy,
            False,
            (
                "Follow-up mode must be able to "
                "override normal question-only "
                "scaffolding."
            ),
        )

        self.pass_test(
            "STRATEGY-T10 Follow-up",
            (
                "Follow-up questions map to "
                "answer-first tutoring behavior."
            ),
        )

    def test_strategy_t11_clarification(
        self,
    ) -> None:

        policy = (
            TutoringResponseStrategyPolicy()
        )

        intent = LearnerTurnIntentResult(
            intent="clarification_question",
            reason="Test clarification.",
            is_question=True,
        )

        decision = policy.decide(
            intent
        )

        self.assert_equal(
            decision.mode,
            "direct_clarification",
            "Clarification mode",
        )

        self.assert_true(
            decision.direct_answer_required,
            (
                "Clarification must require "
                "a direct explanation."
            ),
        )

        self.assert_equal(
            decision.guiding_question_policy,
            "optional",
            "Clarification question policy",
        )

        self.assert_equal(
            decision.preserve_scaffolding_strategy,
            False,
            (
                "Clarification must not be forced "
                "through question-only scaffolding."
            ),
        )

        self.pass_test(
            "STRATEGY-T11 Clarification",
            (
                "Clarification requests map to "
                "direct explanation behavior."
            ),
        )

    def test_strategy_t12_topic_change(
        self,
    ) -> None:

        policy = (
            TutoringResponseStrategyPolicy()
        )

        intent = LearnerTurnIntentResult(
            intent="topic_change",
            reason="Test topic change.",
            is_question=True,
        )

        decision = policy.decide(
            intent
        )

        self.assert_equal(
            decision.mode,
            "new_topic_scaffold",
            "Topic-change mode",
        )

        self.assert_equal(
            decision.direct_answer_required,
            False,
            "Topic-change direct-answer policy",
        )

        self.assert_true(
            decision.preserve_scaffolding_strategy,
            (
                "A new topic should begin using "
                "the normal scaffolding strategy."
            ),
        )

        self.pass_test(
            "STRATEGY-T12 Topic change",
            (
                "Explicit topic changes start a "
                "fresh scaffolded learning sequence."
            ),
        )

    def test_strategy_t13_uncertain(
        self,
    ) -> None:

        policy = (
            TutoringResponseStrategyPolicy()
        )

        intent = LearnerTurnIntentResult(
            intent="uncertain",
            reason="Test uncertainty.",
        )

        decision = policy.decide(
            intent
        )

        self.assert_equal(
            decision.mode,
            "scaffolded",
            "Uncertain response mode",
        )

        self.assert_true(
            decision.preserve_scaffolding_strategy,
            (
                "Uncertain intent should preserve "
                "existing behavior conservatively."
            ),
        )

        self.pass_test(
            "STRATEGY-T13 Uncertain",
            (
                "Uncertain learner intent falls "
                "back to normal scaffolding."
            ),
        )

    def test_strategy_t14_unknown_fail_safe(
        self,
    ) -> None:

        policy = (
            TutoringResponseStrategyPolicy()
        )

        intent = LearnerTurnIntentResult(
            intent="future_intent",
            reason="Unknown future intent.",
        )

        decision = policy.decide(
            intent
        )

        self.assert_equal(
            decision.mode,
            "scaffolded",
            "Unknown-intent fallback",
        )

        self.assert_true(
            decision.preserve_scaffolding_strategy,
            (
                "Unknown intent must fail safely "
                "to existing behavior."
            ),
        )

        self.pass_test(
            "STRATEGY-T14 Unknown fail-safe",
            (
                "Unknown future learner intents "
                "fall back conservatively."
            ),
        )

    def test_strategy_t15_invalid_type(
        self,
    ) -> None:

        policy = (
            TutoringResponseStrategyPolicy()
        )

        raised = False

        try:
            policy.decide(
                "answer"
            )

        except TypeError:
            raised = True

        self.assert_true(
            raised,
            (
                "Response strategy policy must "
                "reject invalid input types."
            ),
        )

        self.pass_test(
            "STRATEGY-T15 Invalid type",
            (
                "Invalid policy inputs are rejected "
                "deterministically."
            ),
        )

    def test_strategy_t16_deterministic(
        self,
    ) -> None:

        policy = (
            TutoringResponseStrategyPolicy()
        )

        intent = LearnerTurnIntentResult(
            intent="clarification_question",
            reason="Same input.",
            confidence=0.95,
            is_question=True,
        )

        first = policy.decide(
            intent
        )

        second = policy.decide(
            intent
        )

        self.assert_equal(
            first,
            second,
            (
                "Identical learner intent must "
                "produce identical response mode."
            ),
        )

        self.pass_test(
            "STRATEGY-T16 Deterministic",
            (
                "Response strategy selection is "
                "fully deterministic."
            ),
        )

    def test_strategy_t17_no_llm(
        self,
    ) -> None:

        source = inspect.getsource(
            TutoringResponseStrategyPolicy
        )

        lowered = source.lower()

        self.assert_true(
            "chat_with_ai" not in lowered,
            (
                "Response strategy policy must "
                "not call the AI runtime."
            ),
        )

        self.assert_true(
            "groq" not in lowered,
            (
                "Response strategy policy must "
                "not depend on Groq."
            ),
        )

        self.assert_true(
            "openai" not in lowered,
            (
                "Response strategy policy must "
                "not depend on OpenAI."
            ),
        )

        self.pass_test(
            "STRATEGY-T17 No LLM",
            (
                "Intent-to-response-mode mapping "
                "contains no LLM dependency."
            ),
        )

    # =====================================================
    # 16.19.3 Runtime Observation Integration
    # =====================================================

    def test_runtime_t18_debug_defaults(
        self,
    ) -> None:

        tutor = AITutor()

        debug = tutor.get_debug_info()

        self.assert_equal(
            debug["tutoring_response_mode"],
            None,
            "Initial response-mode telemetry",
        )

        self.assert_equal(
            debug[
                "tutoring_direct_answer_required"
            ],
            None,
            "Initial direct-answer telemetry",
        )

        self.pass_test(
            "MODE-RUNTIME-T18 Debug defaults",
            (
                "Response-mode runtime telemetry "
                "starts with safe null values."
            ),
        )

    def test_runtime_t19_debug_projection(
        self,
    ) -> None:

        tutor = AITutor()

        decision = (
            TutoringResponseStrategyPolicy()
            .decide(
                LearnerTurnIntentResult(
                    intent="clarification_question",
                    reason="Test clarification.",
                    is_question=True,
                )
            )
        )

        tutor.last_tutoring_response_mode = (
            decision
        )

        debug = tutor.get_debug_info()

        self.assert_equal(
            debug["tutoring_response_mode"],
            "direct_clarification",
            "Runtime response mode",
        )

        self.assert_equal(
            debug[
                "tutoring_direct_answer_required"
            ],
            True,
            "Direct-answer telemetry",
        )

        self.assert_equal(
            debug[
                "tutoring_guiding_question_policy"
            ],
            "optional",
            "Question-policy telemetry",
        )

        self.assert_equal(
            debug[
                "tutoring_preserve_scaffolding_strategy"
            ],
            False,
            "Scaffolding-preservation telemetry",
        )

        self.pass_test(
            "MODE-RUNTIME-T19 Debug telemetry",
            (
                "AITutor exposes the complete typed "
                "response-mode decision."
            ),
        )

    def test_runtime_t20_policy_initialized(
        self,
    ) -> None:

        tutor = AITutor()

        self.assert_true(
            isinstance(
                tutor.tutoring_response_strategy_policy,
                TutoringResponseStrategyPolicy,
            ),
            (
                "AITutor must initialize the "
                "deterministic response strategy "
                "policy."
            ),
        )

        self.pass_test(
            "MODE-RUNTIME-T20 Policy initialized",
            (
                "AITutor owns the deterministic "
                "response strategy policy."
            ),
        )

    def test_runtime_t21_observation_in_pipeline(
        self,
    ) -> None:

        source = inspect.getsource(
            AITutor.respond
        )

        compact = "".join(
            source.split()
        )

        self.assert_true(
            (
                "self."
                "tutoring_response_strategy_policy."
                "decide("
            )
            in compact,
            (
                "AITutor.respond must observe the "
                "response strategy policy."
            ),
        )

        self.assert_true(
            (
                "self.last_tutoring_response_mode="
                "(tutoring_response_mode)"
            )
            in compact
            or
            (
                "self.last_tutoring_response_mode="
                "tutoring_response_mode"
            )
            in compact,
            (
                "AITutor must store current-turn "
                "response-mode telemetry."
            ),
        )

        self.pass_test(
            "MODE-RUNTIME-T21 Pipeline observation",
            (
                "Response-mode selection is observed "
                "during every Tutor turn."
            ),
        )

    def test_runtime_t22_reset_clears(
        self,
    ) -> None:

        tutor = AITutor()

        tutor.last_tutoring_response_mode = (
            TutoringResponseStrategyPolicy()
            .decide(
                LearnerTurnIntentResult(
                    intent="follow_up_question",
                    reason="Test.",
                    is_question=True,
                )
            )
        )

        tutor.reset()

        self.assert_equal(
            tutor.last_tutoring_response_mode,
            None,
            (
                "Conversation reset must clear "
                "response-mode telemetry."
            ),
        )

        self.pass_test(
            "MODE-RUNTIME-T22 Reset clears",
            (
                "Conversation reset clears the "
                "latest response-mode decision."
            ),
        )

    def test_runtime_t23_per_turn_clear(
        self,
    ) -> None:

        source = inspect.getsource(
            AITutor.respond
        )

        self.assert_true(
            (
                "self.last_tutoring_response_mode "
                "= None"
            )
            in source,
            (
                "Each Tutor turn must clear stale "
                "response-mode telemetry before "
                "making a new decision."
            ),
        )

        self.pass_test(
            "MODE-RUNTIME-T23 Per-turn clear",
            (
                "Response-mode telemetry cannot "
                "leak from a previous turn."
            ),
        )

    def test_runtime_t24_no_direct_branching(
        self,
    ) -> None:

        source = inspect.getsource(
            AITutor.respond
        )

        self.assert_true(
            "direct_answer_required"
            not in source,
            (
                "AITutor.respond must not contain "
                "ad-hoc branching on response-mode "
                "decision fields."
            ),
        )

        self.assert_true(
            "guiding_question_policy"
            not in source,
            (
                "AITutor.respond must delegate "
                "response-form instructions to "
                "the deterministic instruction "
                "builder."
            ),
        )

        self.pass_test(
            "MODE-RUNTIME-T24 No direct branching",
            (
                "AITutor integrates response mode "
                "through the instruction builder "
                "rather than ad-hoc branching."
            ),
        )

    # =====================================================
    # 16.19.4A Response Mode Instruction Builder
    # =====================================================

    def test_instruction_t25_follow_up(
        self,
    ) -> None:

        policy = TutoringResponseStrategyPolicy()
        builder = (
            TutoringResponseModeInstructionBuilder()
        )

        decision = policy.decide(
            LearnerTurnIntentResult(
                intent="follow_up_question",
                reason="Test follow-up.",
                is_question=True,
            )
        )

        instruction = builder.build(
            decision
        )

        self.assert_true(
            (
                "Answer the learner's current "
                "question directly"
            )
            in instruction,
            (
                "Follow-up instruction must require "
                "an answer-first response."
            ),
        )

        self.assert_true(
            (
                "Do not respond only with "
                "another question."
            )
            in instruction,
            (
                "Follow-up instruction must prevent "
                "question-only responses."
            ),
        )

        self.pass_test(
            "INSTRUCTION-T25 Follow-up",
            (
                "Follow-up mode produces deterministic "
                "answer-first prompt instructions."
            ),
        )

    def test_instruction_t26_clarification(
        self,
    ) -> None:

        policy = TutoringResponseStrategyPolicy()
        builder = (
            TutoringResponseModeInstructionBuilder()
        )

        decision = policy.decide(
            LearnerTurnIntentResult(
                intent="clarification_question",
                reason="Test clarification.",
                is_question=True,
            )
        )

        instruction = builder.build(
            decision
        )

        self.assert_true(
            (
                "Directly explain the term, concept, "
                "or point"
            )
            in instruction,
            (
                "Clarification instruction must "
                "require a direct explanation."
            ),
        )

        self.assert_true(
            (
                "Do not respond only with a "
                "Socratic or guiding question."
            )
            in instruction,
            (
                "Clarification instruction must "
                "prevent question-only responses."
            ),
        )

        self.pass_test(
            "INSTRUCTION-T26 Clarification",
            (
                "Clarification mode produces direct "
                "explanation instructions."
            ),
        )

    def test_instruction_t27_scaffolded_no_override(
        self,
    ) -> None:

        policy = TutoringResponseStrategyPolicy()
        builder = (
            TutoringResponseModeInstructionBuilder()
        )

        decision = policy.decide(
            LearnerTurnIntentResult(
                intent="answer",
                reason="Test answer.",
            )
        )

        self.assert_equal(
            builder.build(decision),
            "",
            (
                "Normal scaffolded behavior must "
                "not receive a response-mode "
                "prompt override."
            ),
        )

        self.pass_test(
            "INSTRUCTION-T27 Scaffolded unchanged",
            (
                "Normal learner answers preserve "
                "the existing prompt behavior."
            ),
        )

    def test_instruction_t28_new_topic_no_override(
        self,
    ) -> None:

        policy = TutoringResponseStrategyPolicy()
        builder = (
            TutoringResponseModeInstructionBuilder()
        )

        initial = policy.decide(None)

        topic_change = policy.decide(
            LearnerTurnIntentResult(
                intent="topic_change",
                reason="Test new topic.",
                is_question=True,
            )
        )

        self.assert_equal(
            builder.build(initial),
            "",
            "Initial-topic instruction",
        )

        self.assert_equal(
            builder.build(topic_change),
            "",
            "Topic-change instruction",
        )

        self.pass_test(
            "INSTRUCTION-T28 New topic unchanged",
            (
                "Initial and topic-change turns keep "
                "normal scaffolding prompt behavior."
            ),
        )

    def test_instruction_t29_invalid_type(
        self,
    ) -> None:

        builder = (
            TutoringResponseModeInstructionBuilder()
        )

        raised = False

        try:
            builder.build(
                "direct_clarification"
            )

        except TypeError:
            raised = True

        self.assert_true(
            raised,
            (
                "Instruction builder must reject "
                "invalid input types."
            ),
        )

        self.pass_test(
            "INSTRUCTION-T29 Invalid type",
            (
                "Invalid instruction-builder input "
                "fails deterministically."
            ),
        )

    def test_instruction_t30_deterministic(
        self,
    ) -> None:

        policy = TutoringResponseStrategyPolicy()
        builder = (
            TutoringResponseModeInstructionBuilder()
        )

        decision = policy.decide(
            LearnerTurnIntentResult(
                intent="clarification_question",
                reason="Same input.",
                confidence=0.95,
                is_question=True,
            )
        )

        first = builder.build(
            decision
        )

        second = builder.build(
            decision
        )

        self.assert_equal(
            first,
            second,
            (
                "Identical response-mode decisions "
                "must create identical instructions."
            ),
        )

        self.pass_test(
            "INSTRUCTION-T30 Deterministic",
            (
                "Response-mode prompt instructions "
                "are fully deterministic."
            ),
        )

    def test_instruction_t31_no_llm(
        self,
    ) -> None:

        source = inspect.getsource(
            TutoringResponseModeInstructionBuilder
        )

        lowered = source.lower()

        self.assert_true(
            "chat_with_ai" not in lowered,
            (
                "Instruction builder must not call "
                "the AI runtime."
            ),
        )

        self.assert_true(
            "groq" not in lowered,
            (
                "Instruction builder must not "
                "depend on Groq."
            ),
        )

        self.assert_true(
            "openai" not in lowered,
            (
                "Instruction builder must not "
                "depend on OpenAI."
            ),
        )

        self.pass_test(
            "INSTRUCTION-T31 No LLM",
            (
                "Response-mode instruction building "
                "contains no LLM dependency."
            ),
        )

    # =====================================================
    # 16.19.4B PromptBuilder Integration
    # =====================================================

    def test_prompt_t32_normal_unchanged(
        self,
    ) -> None:

        builder = PromptBuilder()

        strategy = get_strategy(1)
        intervention = get_intervention(None)

        before = builder.build_system_prompt(
            strategy=strategy,
            intervention=intervention,
            knowledge_context=(
                "Controlled course knowledge."
            ),
            has_grounded_knowledge=True,
            response_style_instruction=(
                "CONTROLLED RESPONSE STYLE"
            ),
        )

        after = builder.build_system_prompt(
            strategy=strategy,
            intervention=intervention,
            knowledge_context=(
                "Controlled course knowledge."
            ),
            has_grounded_knowledge=True,
            response_style_instruction=(
                "CONTROLLED RESPONSE STYLE"
            ),
            response_mode_instruction="",
        )

        self.assert_equal(
            after,
            before,
            (
                "An empty response-mode instruction "
                "must preserve the existing system "
                "prompt exactly."
            ),
        )

        self.pass_test(
            "PROMPT-T32 Normal unchanged",
            (
                "Normal scaffolding prompt behavior "
                "remains byte-for-byte unchanged."
            ),
        )

    def test_prompt_t33_follow_up_instruction(
        self,
    ) -> None:

        builder = PromptBuilder()

        strategy = get_strategy(1)
        intervention = get_intervention(None)

        mode_instruction = (
            "LEARNER RESPONSE MODE:\n"
            "FOLLOW-UP ANSWER-FIRST TEST"
        )

        prompt = builder.build_system_prompt(
            strategy=strategy,
            intervention=intervention,
            knowledge_context=(
                "Controlled course knowledge."
            ),
            response_mode_instruction=(
                mode_instruction
            ),
        )

        self.assert_true(
            mode_instruction in prompt,
            (
                "Follow-up response-mode instruction "
                "must appear in the final prompt."
            ),
        )

        self.assert_equal(
            prompt.count(
                "FOLLOW-UP ANSWER-FIRST TEST"
            ),
            1,
            (
                "Response-mode instruction must "
                "appear exactly once."
            ),
        )

        self.pass_test(
            "PROMPT-T33 Follow-up instruction",
            (
                "PromptBuilder includes the "
                "turn-specific response mode once."
            ),
        )

    def test_prompt_t34_clarification_instruction(
        self,
    ) -> None:

        builder = PromptBuilder()

        strategy = get_strategy(1)
        intervention = get_intervention(None)

        mode_instruction = (
            "LEARNER RESPONSE MODE:\n"
            "DIRECT CLARIFICATION TEST"
        )

        prompt = builder.build_system_prompt(
            strategy=strategy,
            intervention=intervention,
            knowledge_context=(
                "Controlled course knowledge."
            ),
            response_mode_instruction=(
                mode_instruction
            ),
        )

        self.assert_true(
            "DIRECT CLARIFICATION TEST"
            in prompt,
            (
                "Clarification response-mode "
                "instruction must be included."
            ),
        )

        self.pass_test(
            "PROMPT-T34 Clarification instruction",
            (
                "Direct clarification instructions "
                "can be integrated into the prompt."
            ),
        )

    def test_prompt_t35_order(
        self,
    ) -> None:

        builder = PromptBuilder()

        strategy = get_strategy(1)
        intervention = get_intervention(None)

        mode_marker = (
            "TURN-SPECIFIC-MODE-MARKER"
        )

        knowledge_marker = (
            "KNOWLEDGE-MARKER"
        )

        prompt = builder.build_system_prompt(
            strategy=strategy,
            intervention=intervention,
            knowledge_context=knowledge_marker,
            response_mode_instruction=mode_marker,
        )

        strategy_index = prompt.find(
            strategy.instruction.strip()
        )

        mode_index = prompt.find(
            mode_marker
        )

        knowledge_index = prompt.find(
            knowledge_marker
        )

        self.assert_true(
            strategy_index >= 0,
            "Strategy must exist in prompt.",
        )

        self.assert_true(
            mode_index > strategy_index,
            (
                "Turn-specific response mode must "
                "appear after the normal strategy."
            ),
        )

        self.assert_true(
            knowledge_index > mode_index,
            (
                "Course knowledge must remain after "
                "the response-mode instruction."
            ),
        )

        self.pass_test(
            "PROMPT-T35 Section order",
            (
                "Prompt order preserves strategy, "
                "turn override, then grounded "
                "knowledge."
            ),
        )

    def test_prompt_t36_style_preserved(
        self,
    ) -> None:

        builder = PromptBuilder()

        strategy = get_strategy(1)
        intervention = get_intervention(None)

        style_marker = (
            "CONTROLLED-STYLE-MARKER"
        )

        mode_marker = (
            "CONTROLLED-MODE-MARKER"
        )

        prompt = builder.build_system_prompt(
            strategy=strategy,
            intervention=intervention,
            knowledge_context=(
                "Controlled knowledge."
            ),
            response_style_instruction=(
                style_marker
            ),
            response_mode_instruction=(
                mode_marker
            ),
        )

        self.assert_true(
            style_marker in prompt,
            (
                "ResponseStyle instruction must "
                "remain present."
            ),
        )

        self.assert_true(
            mode_marker in prompt,
            (
                "Response-mode instruction must "
                "also be present."
            ),
        )

        self.pass_test(
            "PROMPT-T36 Style preserved",
            (
                "Response-mode integration does not "
                "remove ResponseStyle."
            ),
        )

    def test_prompt_t37_grounding_preserved(
        self,
    ) -> None:

        builder = PromptBuilder()

        strategy = get_strategy(1)
        intervention = get_intervention(None)

        knowledge = (
            "CONTROLLED COURSE KNOWLEDGE"
        )

        prompt = builder.build_system_prompt(
            strategy=strategy,
            intervention=intervention,
            knowledge_context=knowledge,
            response_mode_instruction=(
                "CONTROLLED RESPONSE MODE"
            ),
        )

        self.assert_true(
            knowledge in prompt,
            (
                "Course knowledge must remain "
                "present."
            ),
        )

        self.assert_true(
            "GROUNDING RULES:"
            in prompt,
            (
                "Grounding rules must remain "
                "present."
            ),
        )

        self.assert_true(
            (
                "Do not invent facts that are "
                "not supported"
            )
            in prompt,
            (
                "Grounding constraint must remain "
                "active."
            ),
        )

        self.pass_test(
            "PROMPT-T37 Grounding preserved",
            (
                "Response-mode integration preserves "
                "RAG knowledge and grounding rules."
            ),
        )

    def test_prompt_t38_deterministic(
        self,
    ) -> None:

        builder = PromptBuilder()

        strategy = get_strategy(1)
        intervention = get_intervention(None)

        kwargs = {
            "strategy": strategy,
            "intervention": intervention,
            "knowledge_context": (
                "Controlled deterministic knowledge."
            ),
            "has_grounded_knowledge": True,
            "response_style_instruction": (
                "Controlled style."
            ),
            "response_mode_instruction": (
                "Controlled response mode."
            ),
        }

        first = builder.build_system_prompt(
            **kwargs
        )

        second = builder.build_system_prompt(
            **kwargs
        )

        self.assert_equal(
            first,
            second,
            (
                "Identical prompt inputs must "
                "produce identical output."
            ),
        )

        self.pass_test(
            "PROMPT-T38 Deterministic",
            (
                "Response-mode PromptBuilder "
                "integration is deterministic."
            ),
        )

    # =====================================================
    # 16.19.4C Runtime Prompt Integration
    # =====================================================

    def test_runtime_prompt_t39_builder_initialized(
        self,
    ) -> None:

        tutor = AITutor()

        self.assert_true(
            isinstance(
                tutor.tutoring_response_mode_instruction_builder,
                TutoringResponseModeInstructionBuilder,
            ),
            (
                "AITutor must own the deterministic "
                "response-mode instruction builder."
            ),
        )

        self.pass_test(
            "PROMPT-RUNTIME-T39 Builder initialized",
            (
                "AITutor owns the deterministic "
                "response-mode instruction builder."
            ),
        )

    def test_runtime_prompt_t40_instruction_default(
        self,
    ) -> None:

        tutor = AITutor()

        self.assert_equal(
            tutor.last_tutoring_response_mode_instruction,
            None,
            "Initial response-mode instruction",
        )

        debug = tutor.get_debug_info()

        self.assert_equal(
            debug[
                "tutoring_response_mode_instruction_applied"
            ],
            False,
            "Initial prompt override telemetry",
        )

        self.pass_test(
            "PROMPT-RUNTIME-T40 Instruction default",
            (
                "Response-mode prompt override "
                "starts inactive."
            ),
        )

    def test_runtime_prompt_t41_instruction_pipeline(
        self,
    ) -> None:

        source = inspect.getsource(
            AITutor.respond
        )

        compact = "".join(
            source.split()
        )

        self.assert_true(
            (
                "self."
                "tutoring_response_mode_instruction_builder."
                "build(tutoring_response_mode)"
            )
            in compact,
            (
                "AITutor must convert the current "
                "response-mode decision into a "
                "prompt instruction."
            ),
        )

        self.assert_true(
            (
                "self."
                "last_tutoring_response_mode_instruction="
                "(tutoring_response_mode_instruction)"
            )
            in compact
            or
            (
                "self."
                "last_tutoring_response_mode_instruction="
                "tutoring_response_mode_instruction"
            )
            in compact,
            (
                "AITutor must retain current-turn "
                "response-mode instruction telemetry."
            ),
        )

        self.pass_test(
            "PROMPT-RUNTIME-T41 Instruction pipeline",
            (
                "Current response mode is converted "
                "to a deterministic prompt override."
            ),
        )

    def test_runtime_prompt_t42_promptbuilder_receives(
        self,
    ) -> None:

        source = inspect.getsource(
            AITutor.respond
        )

        compact = "".join(
            source.split()
        )

        self.assert_true(
            (
                "response_mode_instruction="
                "(tutoring_response_mode_instruction)"
            )
            in compact
            or
            (
                "response_mode_instruction="
                "tutoring_response_mode_instruction"
            )
            in compact,
            (
                "PromptBuilder must receive the "
                "current response-mode instruction."
            ),
        )

        self.pass_test(
            "PROMPT-RUNTIME-T42 PromptBuilder receives",
            (
                "AITutor passes the deterministic "
                "turn-specific instruction into "
                "PromptBuilder."
            ),
        )

    def test_runtime_prompt_t43_preserving_mode_empty(
        self,
    ) -> None:

        policy = TutoringResponseStrategyPolicy()
        builder = (
            TutoringResponseModeInstructionBuilder()
        )

        answer_decision = policy.decide(
            LearnerTurnIntentResult(
                intent="answer",
                reason="Test.",
            )
        )

        initial_decision = policy.decide(
            None
        )

        self.assert_equal(
            builder.build(answer_decision),
            "",
            "Answer prompt override",
        )

        self.assert_equal(
            builder.build(initial_decision),
            "",
            "Initial prompt override",
        )

        self.pass_test(
            "PROMPT-RUNTIME-T43 Preserve normal modes",
            (
                "Normal scaffolded modes still "
                "produce no prompt override."
            ),
        )

    def test_runtime_prompt_t44_reset_clears_instruction(
        self,
    ) -> None:

        tutor = AITutor()

        tutor.last_tutoring_response_mode_instruction = (
            "CONTROLLED OVERRIDE"
        )

        tutor.reset()

        self.assert_equal(
            tutor.last_tutoring_response_mode_instruction,
            None,
            (
                "Conversation reset must clear "
                "response-mode prompt instruction."
            ),
        )

        self.pass_test(
            "PROMPT-RUNTIME-T44 Reset clears",
            (
                "Conversation reset prevents stale "
                "response-mode instructions."
            ),
        )

    def test_runtime_prompt_t45_debug_projection(
        self,
    ) -> None:

        tutor = AITutor()

        tutor.last_tutoring_response_mode_instruction = (
            "CONTROLLED OVERRIDE"
        )

        debug = tutor.get_debug_info()

        self.assert_equal(
            debug[
                "tutoring_response_mode_instruction_applied"
            ],
            True,
            (
                "Debug telemetry must show an "
                "active response-mode prompt "
                "override."
            ),
        )

        self.pass_test(
            "PROMPT-RUNTIME-T45 Debug projection",
            (
                "Runtime telemetry exposes whether "
                "a turn-specific prompt override "
                "is active."
            ),
        )

    # =====================================================
    # 16.19.5A Response-Mode Pedagogical Precheck
    # =====================================================

    def test_pedagogy_t46_legacy_delegate(
        self,
    ) -> None:

        policy = TutoringResponseStrategyPolicy()
        precheck = (
            ResponseModePedagogicalPrecheck()
        )

        decision = policy.decide(None)

        result = precheck.evaluate(
            response=(
                "คุณคิดว่าโครงสร้างของ"
                "ทรานซิสเตอร์ NPN "
                "ประกอบด้วยส่วนใดบ้าง?"
            ),
            strategy_name="guiding_question",
            response_mode=decision,
        )

        self.assert_equal(
            result.status,
            "valid",
            (
                "Preserving response modes must "
                "retain legacy precheck behavior."
            ),
        )

        self.pass_test(
            "PEDAGOGY-T46 Legacy delegate",
            (
                "Normal scaffolding remains owned "
                "by the existing PedagogicalPrecheck."
            ),
        )

    def test_pedagogy_t47_direct_clarification(
        self,
    ) -> None:

        policy = TutoringResponseStrategyPolicy()
        precheck = (
            ResponseModePedagogicalPrecheck()
        )

        decision = policy.decide(
            LearnerTurnIntentResult(
                intent="clarification_question",
                reason="Test.",
                is_question=True,
            )
        )

        result = precheck.evaluate(
            response=(
                "เบส (Base) คือส่วนที่ผู้เรียน"
                "กำลังขอให้ชี้แจงเพิ่มเติม"
            ),
            strategy_name="guiding_question",
            response_mode=decision,
        )

        self.assert_equal(
            result.status,
            "valid",
            (
                "A direct clarification without a "
                "question must be structurally valid."
            ),
        )

        self.pass_test(
            "PEDAGOGY-T47 Direct clarification",
            (
                "Clarification mode permits a direct "
                "response without forcing a question."
            ),
        )

    def test_pedagogy_t48_follow_up_answer_then_question(
        self,
    ) -> None:

        policy = TutoringResponseStrategyPolicy()
        precheck = (
            ResponseModePedagogicalPrecheck()
        )

        decision = policy.decide(
            LearnerTurnIntentResult(
                intent="follow_up_question",
                reason="Test.",
                is_question=True,
            )
        )

        result = precheck.evaluate(
            response=(
                "คำอธิบายโดยตรงสำหรับคำถามของผู้เรียน.\n"
                "คุณคิดว่าประเด็นนี้สัมพันธ์กับอะไร?"
            ),
            strategy_name="guiding_question",
            response_mode=decision,
        )

        self.assert_equal(
            result.status,
            "valid",
            (
                "A direct answer followed by one "
                "question must be accepted."
            ),
        )

        self.pass_test(
            "PEDAGOGY-T48 Answer then guide",
            (
                "Follow-up mode permits an answer "
                "followed by one optional question."
            ),
        )

    def test_pedagogy_t49_question_only_violation(
        self,
    ) -> None:

        policy = TutoringResponseStrategyPolicy()
        precheck = (
            ResponseModePedagogicalPrecheck()
        )

        decision = policy.decide(
            LearnerTurnIntentResult(
                intent="clarification_question",
                reason="Test.",
                is_question=True,
            )
        )

        result = precheck.evaluate(
            response=(
                "คุณคิดว่าเบส (Base) "
                "หมายถึงอะไร?"
            ),
            strategy_name="guiding_question",
            response_mode=decision,
        )

        self.assert_equal(
            result.status,
            "violation",
            (
                "Question-only clarification must "
                "be rejected."
            ),
        )

        self.pass_test(
            "PEDAGOGY-T49 Question-only violation",
            (
                "Answer-required response modes reject "
                "question-only Tutor responses."
            ),
        )

    def test_pedagogy_t50_question_limit(
        self,
    ) -> None:

        policy = TutoringResponseStrategyPolicy()
        precheck = (
            ResponseModePedagogicalPrecheck()
        )

        decision = policy.decide(
            LearnerTurnIntentResult(
                intent="follow_up_question",
                reason="Test.",
                is_question=True,
            )
        )

        result = precheck.evaluate(
            response=(
                "คำอธิบายเบื้องต้น.\n"
                "ส่วนแรกคืออะไร? "
                "แล้วส่วนที่สองคืออะไร?"
            ),
            strategy_name="guiding_question",
            response_mode=decision,
        )

        self.assert_equal(
            result.status,
            "violation",
            (
                "Response mode must enforce the "
                "configured question limit."
            ),
        )

        self.pass_test(
            "PEDAGOGY-T50 Question limit",
            (
                "Response-mode precheck rejects "
                "excessive guiding questions."
            ),
        )

    def test_pedagogy_t51_empty_violation(
        self,
    ) -> None:

        policy = TutoringResponseStrategyPolicy()
        precheck = (
            ResponseModePedagogicalPrecheck()
        )

        decision = policy.decide(
            LearnerTurnIntentResult(
                intent="clarification_question",
                reason="Test.",
                is_question=True,
            )
        )

        result = precheck.evaluate(
            response="",
            strategy_name="guiding_question",
            response_mode=decision,
        )

        self.assert_equal(
            result.status,
            "violation",
            "Empty response status",
        )

        self.pass_test(
            "PEDAGOGY-T51 Empty response",
            (
                "Empty Tutor responses fail the "
                "response-mode pedagogical precheck."
            ),
        )

    def test_pedagogy_t52_invalid_type(
        self,
    ) -> None:

        precheck = (
            ResponseModePedagogicalPrecheck()
        )

        raised = False

        try:
            precheck.evaluate(
                response="test",
                strategy_name="guiding_question",
                response_mode="clarification",
            )

        except TypeError:
            raised = True

        self.assert_true(
            raised,
            (
                "Invalid response-mode input must "
                "fail fast."
            ),
        )

        self.pass_test(
            "PEDAGOGY-T52 Invalid type",
            (
                "Response-mode pedagogical precheck "
                "rejects invalid typed input."
            ),
        )

    def test_pedagogy_t53_no_llm(
        self,
    ) -> None:

        source = inspect.getsource(
            ResponseModePedagogicalPrecheck
        )

        lowered = source.lower()

        for forbidden in (
            "chat_with_ai",
            "groq",
            "openai",
            "litellm",
        ):
            self.assert_true(
                forbidden not in lowered,
                (
                    "Deterministic response-mode "
                    "precheck must not depend on "
                    f"{forbidden}."
                ),
            )

        self.pass_test(
            "PEDAGOGY-T53 No LLM",
            (
                "Response-mode pedagogical precheck "
                "is fully deterministic."
            ),
        )

    # =====================================================
    # 16.19.5B Response-Mode Semantic Validator
    # =====================================================

    def test_pedagogy_t54_legacy_delegation(
        self,
    ) -> None:

        base = Mock()

        base.validate.return_value = (
            PedagogicalValidationResult(
                status="valid",
                confidence=1.0,
                reason="Legacy validation.",
                issues=[],
            )
        )

        validator = (
            ResponseModePedagogicalValidator(
                base_validator=base
            )
        )

        mode = (
            TutoringResponseStrategyPolicy()
            .decide(None)
        )

        result = validator.validate(
            response="Legacy response?",
            scaffolding_level=1,
            strategy_name="guiding_question",
            strategy_instruction="Legacy strategy.",
            intervention_name="none",
            intervention_instruction="",
            response_mode=mode,
        )

        self.assert_equal(
            result.status,
            "valid",
            "Legacy validator result",
        )

        self.assert_equal(
            base.validate.call_count,
            1,
            (
                "Preserving mode must delegate "
                "exactly once."
            ),
        )

        self.pass_test(
            "PEDAGOGY-T54 Legacy semantic delegation",
            (
                "Preserving response modes retain "
                "the existing semantic validator."
            ),
        )

    def test_pedagogy_t55_mode_semantic_path(
        self,
    ) -> None:

        mode = (
            TutoringResponseStrategyPolicy()
            .decide(
                LearnerTurnIntentResult(
                    intent="clarification_question",
                    reason="Test.",
                    is_question=True,
                )
            )
        )

        validator = (
            ResponseModePedagogicalValidator()
        )

        mocked_json = (
            '{"status":"valid",'
            '"confidence":0.95,'
            '"reason":"Direct clarification was given.",'
            '"issues":[]}'
        )

        with patch(
            (
                "app.services."
                "response_mode_pedagogical_validator."
                "chat_with_structured_ai"
            ),
            return_value=mocked_json,
        ) as mocked_ai:

            result = validator.validate(
                response=(
                    "คำอธิบายโดยตรงสำหรับผู้เรียน"
                ),
                scaffolding_level=1,
                strategy_name="guiding_question",
                strategy_instruction=(
                    "Ask one guiding question."
                ),
                intervention_name="none",
                intervention_instruction="",
                response_mode=mode,
                response_mode_instruction=(
                    "Answer directly first."
                ),
            )

        self.assert_equal(
            result.status,
            "valid",
            "Response-mode semantic result",
        )

        self.assert_equal(
            mocked_ai.call_count,
            1,
            (
                "Non-preserving semantic path "
                "must make one validator call."
            ),
        )

        self.pass_test(
            "PEDAGOGY-T55 Response-mode semantic path",
            (
                "Non-preserving response modes use "
                "the response-mode-aware validator."
            ),
        )

    def test_pedagogy_t56_mode_context(
        self,
    ) -> None:

        mode = (
            TutoringResponseStrategyPolicy()
            .decide(
                LearnerTurnIntentResult(
                    intent="follow_up_question",
                    reason="Test.",
                    is_question=True,
                )
            )
        )

        validator = (
            ResponseModePedagogicalValidator()
        )

        with patch(
            (
                "app.services."
                "response_mode_pedagogical_validator."
                "chat_with_structured_ai"
            ),
            return_value=(
                '{"status":"valid",'
                '"confidence":1.0,'
                '"reason":"Valid.",'
                '"issues":[]}'
            ),
        ) as mocked_ai:

            validator.validate(
                response="Controlled response.",
                scaffolding_level=1,
                strategy_name="guiding_question",
                strategy_instruction="Strategy.",
                intervention_name="none",
                intervention_instruction="",
                response_mode=mode,
                response_mode_instruction=(
                    "CONTROLLED MODE INSTRUCTION"
                ),
            )

        kwargs = mocked_ai.call_args.kwargs

        messages = kwargs[
            "messages"
        ]

        user_content = messages[1][
            "content"
        ]

        self.assert_true(
            "answer_then_guide"
            in user_content,
            (
                "Validator context must contain "
                "the response mode."
            ),
        )

        self.assert_true(
            "DIRECT ANSWER REQUIRED:"
            in user_content,
            (
                "Validator must receive the "
                "direct-answer requirement."
            ),
        )

        self.assert_true(
            "CONTROLLED MODE INSTRUCTION"
            in user_content,
            (
                "Validator must receive the same "
                "turn-specific instruction used "
                "for Tutor generation."
            ),
        )

        self.pass_test(
            "PEDAGOGY-T56 Mode context",
            (
                "Semantic validator receives the "
                "complete response-mode context."
            ),
        )

    def test_pedagogy_t57_invalid_json_fail_closed(
        self,
    ) -> None:

        mode = (
            TutoringResponseStrategyPolicy()
            .decide(
                LearnerTurnIntentResult(
                    intent="clarification_question",
                    reason="Test.",
                    is_question=True,
                )
            )
        )

        validator = (
            ResponseModePedagogicalValidator()
        )

        with patch(
            (
                "app.services."
                "response_mode_pedagogical_validator."
                "chat_with_structured_ai"
            ),
            return_value="not-json",
        ):

            result = validator.validate(
                response="Controlled response.",
                scaffolding_level=1,
                strategy_name="guiding_question",
                strategy_instruction="Strategy.",
                intervention_name="none",
                intervention_instruction="",
                response_mode=mode,
            )

        self.assert_equal(
            result.status,
            "violation",
            (
                "Invalid semantic validator JSON "
                "must fail closed."
            ),
        )

        self.pass_test(
            "PEDAGOGY-T57 Invalid JSON",
            (
                "Invalid response-mode validator "
                "output fails closed."
            ),
        )

    def test_pedagogy_t58_invalid_status_fail_closed(
        self,
    ) -> None:

        validator = (
            ResponseModePedagogicalValidator()
        )

        result = validator._parse_result(
            (
                '{"status":"unknown",'
                '"confidence":0.9,'
                '"reason":"Test.",'
                '"issues":[]}'
            )
        )

        self.assert_equal(
            result.status,
            "violation",
            (
                "Unknown semantic status must "
                "fail closed."
            ),
        )

        self.pass_test(
            "PEDAGOGY-T58 Invalid status",
            (
                "Unknown validator status is "
                "normalized to violation."
            ),
        )

    def test_pedagogy_t59_confidence_clamp(
        self,
    ) -> None:

        validator = (
            ResponseModePedagogicalValidator()
        )

        high = validator._parse_result(
            (
                '{"status":"valid",'
                '"confidence":5,'
                '"reason":"Test.",'
                '"issues":[]}'
            )
        )

        low = validator._parse_result(
            (
                '{"status":"valid",'
                '"confidence":-2,'
                '"reason":"Test.",'
                '"issues":[]}'
            )
        )

        self.assert_equal(
            high.confidence,
            1.0,
            "High confidence clamp",
        )

        self.assert_equal(
            low.confidence,
            0.0,
            "Low confidence clamp",
        )

        self.pass_test(
            "PEDAGOGY-T59 Confidence clamp",
            (
                "Semantic confidence remains "
                "within the valid range."
            ),
        )

    def test_pedagogy_t60_invalid_type(
        self,
    ) -> None:

        validator = (
            ResponseModePedagogicalValidator()
        )

        raised = False

        try:
            validator.validate(
                response="Test.",
                scaffolding_level=1,
                strategy_name="guiding_question",
                strategy_instruction="Strategy.",
                intervention_name="none",
                intervention_instruction="",
                response_mode="clarification",
            )

        except TypeError:
            raised = True

        self.assert_true(
            raised,
            (
                "Semantic validator must reject "
                "invalid response-mode types."
            ),
        )

        self.pass_test(
            "PEDAGOGY-T60 Invalid type",
            (
                "Typed response-mode input is "
                "enforced."
            ),
        )

    def test_pedagogy_t61_preserving_no_new_llm(
        self,
    ) -> None:

        base = Mock()

        base.validate.return_value = (
            PedagogicalValidationResult(
                status="valid",
                confidence=1.0,
                reason="Legacy.",
                issues=[],
            )
        )

        validator = (
            ResponseModePedagogicalValidator(
                base_validator=base
            )
        )

        mode = (
            TutoringResponseStrategyPolicy()
            .decide(None)
        )

        with patch(
            (
                "app.services."
                "response_mode_pedagogical_validator."
                "chat_with_structured_ai"
            )
        ) as new_validator_ai:

            validator.validate(
                response="Legacy response?",
                scaffolding_level=1,
                strategy_name="guiding_question",
                strategy_instruction="Strategy.",
                intervention_name="none",
                intervention_instruction="",
                response_mode=mode,
            )

        self.assert_equal(
            new_validator_ai.call_count,
            0,
            (
                "Preserving modes must not add a "
                "new response-mode semantic call."
            ),
        )

        self.assert_equal(
            base.validate.call_count,
            1,
            (
                "Legacy semantic validation must "
                "remain responsible."
            ),
        )

        self.pass_test(
            "PEDAGOGY-T61 No added legacy LLM",
            (
                "Response-mode validator adds no "
                "new semantic call to preserving "
                "scaffolding modes."
            ),
        )


    # =====================================================
    # 16.19.5C-1 Runtime Pedagogical Routing
    # =====================================================

    def test_pedagogy_runtime_t62_adapters_initialized(
        self,
    ) -> None:

        tutor = AITutor()

        self.assert_true(
            isinstance(
                tutor.response_mode_pedagogical_precheck,
                ResponseModePedagogicalPrecheck,
            ),
            (
                "AITutor must own the response-mode "
                "pedagogical precheck adapter."
            ),
        )

        self.assert_true(
            isinstance(
                tutor.response_mode_pedagogical_validator,
                ResponseModePedagogicalValidator,
            ),
            (
                "AITutor must own the response-mode "
                "semantic validator adapter."
            ),
        )

        self.pass_test(
            "PEDAGOGY-RUNTIME-T62 Adapters initialized",
            (
                "AITutor owns both response-mode-aware "
                "pedagogical validation adapters."
            ),
        )

    def test_pedagogy_runtime_t63_precheck_routed(
        self,
    ) -> None:

        source = inspect.getsource(
            AITutor.respond
        )

        compact = "".join(
            source.split()
        )

        self.assert_true(
            (
                "self."
                "response_mode_pedagogical_precheck."
                "evaluate("
            )
            in compact,
            (
                "Runtime pedagogical precheck must "
                "use the response-mode adapter."
            ),
        )

        self.assert_true(
            (
                "response_mode="
                "(tutoring_response_mode)"
            )
            in compact
            or
            (
                "response_mode="
                "tutoring_response_mode"
            )
            in compact,
            (
                "Current response mode must be passed "
                "to pedagogical precheck."
            ),
        )

        self.pass_test(
            "PEDAGOGY-RUNTIME-T63 Precheck routed",
            (
                "Current Tutor response is checked "
                "against its response mode."
            ),
        )

    def test_pedagogy_runtime_t64_semantic_routed(
        self,
    ) -> None:

        source = inspect.getsource(
            AITutor.respond
        )

        compact = "".join(
            source.split()
        )

        self.assert_true(
            (
                "self."
                "response_mode_pedagogical_validator."
                "validate("
            )
            in compact,
            (
                "Uncertain pedagogical responses "
                "must use the response-mode-aware "
                "semantic validator."
            ),
        )

        self.assert_true(
            (
                "response_mode_instruction="
                "(tutoring_response_mode_instruction)"
            )
            in compact
            or
            (
                "response_mode_instruction="
                "tutoring_response_mode_instruction"
            )
            in compact,
            (
                "Semantic validation must receive "
                "the same response-mode instruction "
                "used for Tutor generation."
            ),
        )

        self.pass_test(
            "PEDAGOGY-RUNTIME-T64 Semantic routed",
            (
                "Semantic pedagogical validation "
                "uses the turn-specific mode context."
            ),
        )

    def test_pedagogy_runtime_t65_no_grounding_change(
        self,
    ) -> None:

        source = inspect.getsource(
            AITutor.respond
        )

        self.assert_true(
            (
                "self.response_grounding_validator"
                in source
            ),
            (
                "Response grounding validator must "
                "remain in the response pipeline."
            ),
        )

        self.assert_true(
            (
                "self.validation_escalation_policy"
                in source
            ),
            (
                "Validation escalation policy must "
                "remain in the response pipeline."
            ),
        )

        self.pass_test(
            "PEDAGOGY-RUNTIME-T65 Grounding preserved",
            (
                "Response-mode pedagogical routing "
                "does not replace grounding or "
                "escalation controls."
            ),
        )

    def test_pedagogy_runtime_t66_legacy_owned(
        self,
    ) -> None:

        tutor = AITutor()

        self.assert_true(
            (
                tutor.response_mode_pedagogical_precheck
                .base_precheck
                is tutor.pedagogical_precheck
            ),
            (
                "Response-mode precheck must wrap "
                "the existing legacy precheck."
            ),
        )

        self.assert_true(
            (
                tutor.response_mode_pedagogical_validator
                .base_validator
                is tutor.pedagogical_response_validator
            ),
            (
                "Response-mode semantic validator "
                "must wrap the existing legacy "
                "validator."
            ),
        )

        self.pass_test(
            "PEDAGOGY-RUNTIME-T66 Legacy ownership",
            (
                "Preserving modes remain delegated "
                "to the frozen pedagogical core."
            ),
        )

    def test_pedagogy_runtime_t67_no_direct_mode_branch(
        self,
    ) -> None:

        source = inspect.getsource(
            AITutor.respond
        )

        self.assert_true(
            (
                'if tutoring_response_mode.mode == '
                '"direct_clarification"'
            )
            not in source,
            (
                "AITutor must not hard-code "
                "clarification behavior."
            ),
        )

        self.assert_true(
            (
                'if tutoring_response_mode.mode == '
                '"answer_then_guide"'
            )
            not in source,
            (
                "AITutor must not hard-code "
                "follow-up behavior."
            ),
        )

        self.pass_test(
            "PEDAGOGY-RUNTIME-T67 Policy driven",
            (
                "Runtime pedagogical routing remains "
                "policy-driven rather than "
                "mode-name branching."
            ),
        )
    # =====================================================
    # 16.19.5C-2 Repair Re-validation Routing
    # =====================================================

    def test_pedagogy_runtime_t68_repair_precheck_routed(
        self,
    ) -> None:

        source = inspect.getsource(
            AITutor.respond
        )

        repair_start = source.find(
            "Repair pedagogical precheck"
        )

        accept_start = source.find(
            "ACCEPT REPAIRED ANSWER"
        )

        repair_section = source[
            repair_start:accept_start
        ]

        self.assert_true(
            (
                "self.response_mode_pedagogical_precheck"
                in repair_section
            ),
            (
                "Repaired responses must use the "
                "response-mode pedagogical precheck."
            ),
        )

        self.assert_true(
            "response_mode="
            in repair_section,
            (
                "Repair precheck must receive the "
                "current tutoring response mode."
            ),
        )

        self.pass_test(
            "PEDAGOGY-RUNTIME-T68 Repair precheck routed",
            (
                "Repaired Tutor responses use the "
                "same response-mode-aware precheck."
            ),
        )

    def test_pedagogy_runtime_t69_repair_semantic_routed(
        self,
    ) -> None:

        source = inspect.getsource(
            AITutor.respond
        )

        repair_start = source.find(
            "Repair pedagogical validation"
        )

        accept_start = source.find(
            "ACCEPT REPAIRED ANSWER"
        )

        repair_section = source[
            repair_start:accept_start
        ]

        self.assert_true(
            (
                "self.response_mode_pedagogical_validator"
                in repair_section
            ),
            (
                "Uncertain repaired responses must "
                "use the response-mode semantic "
                "validator."
            ),
        )

        self.assert_true(
            (
                "response_mode_instruction="
                in repair_section
            ),
            (
                "Repair semantic validation must "
                "receive the same response-mode "
                "instruction used for generation."
            ),
        )

        self.pass_test(
            "PEDAGOGY-RUNTIME-T69 Repair semantic routed",
            (
                "Repair semantic validation uses "
                "the current response-mode context."
            ),
        )

    def test_pedagogy_runtime_t70_no_direct_legacy_repair(
        self,
    ) -> None:

        source = inspect.getsource(
            AITutor.respond
        )

        repair_start = source.find(
            "Repair pedagogical precheck"
        )

        accept_start = source.find(
            "ACCEPT REPAIRED ANSWER"
        )

        repair_section = source[
            repair_start:accept_start
        ]

        self.assert_true(
            (
                "self.pedagogical_precheck."
                not in repair_section
            ),
            (
                "Repair path must not bypass the "
                "response-mode precheck adapter."
            ),
        )

        self.assert_true(
            (
                "self.pedagogical_response_validator."
                not in repair_section
            ),
            (
                "Repair path must not bypass the "
                "response-mode semantic adapter."
            ),
        )

        self.pass_test(
            "PEDAGOGY-RUNTIME-T70 No legacy bypass",
            (
                "Repair re-validation cannot bypass "
                "the response-mode adapters."
            ),
        )

    def test_pedagogy_runtime_t71_repair_grounding_preserved(
        self,
    ) -> None:

        source = inspect.getsource(
            AITutor.respond
        )

        repair_start = source.find(
            "VALIDATE REPAIRED RESPONSE"
        )

        final_start = source.find(
            "STEP 16.17.1"
        )

        repair_section = source[
            repair_start:final_start
        ]

        self.assert_true(
            (
                "self.grounding_claim_precheck"
                in repair_section
            ),
            (
                "Repair grounding precheck must "
                "remain active."
            ),
        )

        self.assert_true(
            (
                "self.embedded_claim_term_guard"
                in repair_section
            ),
            (
                "Repair terminology guard must "
                "remain active."
            ),
        )

        self.assert_true(
            (
                "self.response_grounding_validator"
                in repair_section
            ),
            (
                "Semantic grounding validation must "
                "remain active for repaired responses."
            ),
        )

        self.assert_true(
            (
                'repair_validation.status'
                in repair_section
            ),
            (
                "Repair acceptance must still depend "
                "on grounding validation."
            ),
        )

        self.pass_test(
            "PEDAGOGY-RUNTIME-T71 Repair grounding preserved",
            (
                "Response-mode re-validation does "
                "not weaken repaired-answer "
                "grounding controls."
            ),
        )

    def test_pedagogy_runtime_t72_repair_task_preserved(
        self,
    ) -> None:

        source = inspect.getsource(
            AITutor.respond
        )

        repair_start = source.find(
            "Repair pedagogical validation"
        )

        accept_start = source.find(
            "ACCEPT REPAIRED ANSWER"
        )

        repair_section = source[
            repair_start:accept_start
        ]

        self.assert_true(
            (
                '"pedagogical_validator_repair"'
                in repair_section
            ),
            (
                "Repair semantic validation should "
                "preserve the existing task name."
            ),
        )

        self.pass_test(
            "PEDAGOGY-RUNTIME-T72 Repair task preserved",
            (
                "Repair validator telemetry remains "
                "compatible with the frozen core."
            ),
        )
    # =====================================================
    # 16.19.5D-1 Response-Mode Repair Service
    # =====================================================

    def test_repair_t73_legacy_delegation(
        self,
    ) -> None:

        base = Mock()

        base.repair.return_value = (
            "LEGACY REPAIRED RESPONSE"
        )

        service = ResponseModeRepairService(
            base_repair_service=base
        )

        mode = (
            TutoringResponseStrategyPolicy()
            .decide(None)
        )

        validation = GroundingValidationResult(
            status="unsupported",
            confidence=1.0,
            reason="Controlled grounding issue.",
            issues=["Controlled issue."],
        )

        result = service.repair(
            original_response="Original.",
            knowledge_context="Knowledge.",
            validation=validation,
            learner_message="Learner.",
            scaffolding_level=1,
            strategy_name="guiding_question",
            strategy_instruction="Ask one question.",
            intervention_name="none",
            pedagogical_issues=[],
            response_mode=mode,
        )

        self.assert_equal(
            result,
            "LEGACY REPAIRED RESPONSE",
            "Legacy repaired response",
        )

        self.assert_equal(
            base.repair.call_count,
            1,
            (
                "Preserving response modes must "
                "delegate exactly once."
            ),
        )

        self.pass_test(
            "REPAIR-T73 Legacy delegation",
            (
                "Preserving response modes retain "
                "the frozen ResponseRepairService."
            ),
        )

    def test_repair_t74_follow_up_context(
        self,
    ) -> None:

        mode = (
            TutoringResponseStrategyPolicy()
            .decide(
                LearnerTurnIntentResult(
                    intent="follow_up_question",
                    reason="Test.",
                    is_question=True,
                )
            )
        )

        service = ResponseModeRepairService()

        validation = GroundingValidationResult(
            status="unsupported",
            confidence=1.0,
            reason="Controlled.",
            issues=[],
        )

        with patch(
            (
                "app.services."
                "response_mode_repair_service."
                "chat_with_ai"
            ),
            return_value="Controlled repaired response.",
        ) as mocked_ai:

            service.repair(
                original_response="Original.",
                knowledge_context="Knowledge.",
                validation=validation,
                learner_message="Follow-up?",
                scaffolding_level=1,
                strategy_name="guiding_question",
                strategy_instruction="Ask one question.",
                intervention_name="none",
                pedagogical_issues=[],
                response_mode=mode,
                response_mode_instruction=(
                    "CONTROLLED FOLLOW-UP MODE"
                ),
            )

        messages = mocked_ai.call_args.args[0]

        user_content = messages[1][
            "content"
        ]

        self.assert_true(
            "answer_then_guide"
            in user_content,
            (
                "Follow-up repair must receive "
                "answer_then_guide mode."
            ),
        )

        self.assert_true(
            "Direct answer required:\nTrue"
            in user_content,
            (
                "Follow-up repair must receive "
                "the direct-answer requirement."
            ),
        )

        self.assert_true(
            "CONTROLLED FOLLOW-UP MODE"
            in user_content,
            (
                "Repair must receive the same "
                "response-mode instruction."
            ),
        )

        self.pass_test(
            "REPAIR-T74 Follow-up context",
            (
                "Follow-up semantic repair receives "
                "answer-first response-mode context."
            ),
        )

    def test_repair_t75_clarification_context(
        self,
    ) -> None:

        mode = (
            TutoringResponseStrategyPolicy()
            .decide(
                LearnerTurnIntentResult(
                    intent="clarification_question",
                    reason="Test.",
                    is_question=True,
                )
            )
        )

        service = ResponseModeRepairService()

        validation = GroundingValidationResult(
            status="supported",
            confidence=1.0,
            reason="Grounding is acceptable.",
            issues=[],
        )

        with patch(
            (
                "app.services."
                "response_mode_repair_service."
                "chat_with_ai"
            ),
            return_value="Clarified response.",
        ) as mocked_ai:

            service.repair(
                original_response="Original.",
                knowledge_context="Knowledge.",
                validation=validation,
                learner_message="Clarify this.",
                scaffolding_level=1,
                strategy_name="guiding_question",
                strategy_instruction="Ask one question.",
                intervention_name="none",
                pedagogical_issues=[
                    "Question-only response.",
                ],
                response_mode=mode,
                response_mode_instruction=(
                    "CONTROLLED CLARIFICATION MODE"
                ),
            )

        messages = mocked_ai.call_args.args[0]

        user_content = messages[1][
            "content"
        ]

        self.assert_true(
            "direct_clarification"
            in user_content,
            (
                "Clarification repair must receive "
                "direct_clarification mode."
            ),
        )

        self.assert_true(
            "Guiding question policy:\noptional"
            in user_content,
            (
                "Clarification repair must know "
                "that a guiding question is optional."
            ),
        )

        self.assert_true(
            "Maximum guiding questions:\n1"
            in user_content,
            (
                "Clarification repair must receive "
                "the one-question maximum."
            ),
        )

        self.pass_test(
            "REPAIR-T75 Clarification context",
            (
                "Clarification semantic repair "
                "receives direct-explanation rules."
            ),
        )

    def test_repair_t76_no_legacy_question_conflict(
        self,
    ) -> None:

        self.assert_true(
            (
                "Return exactly ONE question"
                not in RESPONSE_MODE_REPAIR_PROMPT
            ),
            (
                "Response-mode repair prompt must "
                "not contain the legacy mandatory "
                "question rule."
            ),
        )

        self.assert_true(
            (
                "Do not provide the answer before "
                "the question"
                not in RESPONSE_MODE_REPAIR_PROMPT
            ),
            (
                "Response-mode repair prompt must "
                "not prohibit answer-first behavior."
            ),
        )

        self.assert_true(
            (
                "Do not return only another question."
                in RESPONSE_MODE_REPAIR_PROMPT
            ),
            (
                "Response-mode repair prompt must "
                "reject question-only behavior when "
                "a direct answer is required."
            ),
        )

        self.pass_test(
            "REPAIR-T76 No legacy conflict",
            (
                "Mode-aware repair prompt does not "
                "reintroduce legacy question-only "
                "constraints."
            ),
        )

    def test_repair_t77_invalid_mode_type(
        self,
    ) -> None:

        service = ResponseModeRepairService()

        validation = GroundingValidationResult(
            status="supported",
            confidence=1.0,
            reason="Test.",
            issues=[],
        )

        raised = False

        try:
            service.repair(
                original_response="Original.",
                knowledge_context="Knowledge.",
                validation=validation,
                learner_message="Learner.",
                scaffolding_level=1,
                strategy_name="guiding_question",
                strategy_instruction="Strategy.",
                intervention_name="none",
                pedagogical_issues=[],
                response_mode="clarification",
            )

        except TypeError:
            raised = True

        self.assert_true(
            raised,
            (
                "Repair service must reject invalid "
                "response-mode types."
            ),
        )

        self.pass_test(
            "REPAIR-T77 Invalid mode type",
            (
                "Response-mode repair requires the "
                "typed decision object."
            ),
        )

    def test_repair_t78_preserving_no_added_llm(
        self,
    ) -> None:

        base = Mock()

        base.repair.return_value = "Legacy."

        service = ResponseModeRepairService(
            base_repair_service=base
        )

        mode = (
            TutoringResponseStrategyPolicy()
            .decide(None)
        )

        validation = GroundingValidationResult(
            status="unsupported",
            confidence=1.0,
            reason="Test.",
            issues=[],
        )

        with patch(
            (
                "app.services."
                "response_mode_repair_service."
                "chat_with_ai"
            )
        ) as mode_ai:

            service.repair(
                original_response="Original.",
                knowledge_context="Knowledge.",
                validation=validation,
                learner_message="Learner.",
                scaffolding_level=1,
                strategy_name="guiding_question",
                strategy_instruction="Strategy.",
                intervention_name="none",
                pedagogical_issues=[],
                response_mode=mode,
            )

        self.assert_equal(
            mode_ai.call_count,
            0,
            (
                "Preserving modes must not add "
                "another repair LLM call."
            ),
        )

        self.assert_equal(
            base.repair.call_count,
            1,
            "Legacy repair call count",
        )

        self.pass_test(
            "REPAIR-T78 No added legacy LLM",
            (
                "Preserving modes add no new "
                "repair-model invocation."
            ),
        )

    def test_repair_t79_nonpreserving_one_llm(
        self,
    ) -> None:

        mode = (
            TutoringResponseStrategyPolicy()
            .decide(
                LearnerTurnIntentResult(
                    intent="clarification_question",
                    reason="Test.",
                    is_question=True,
                )
            )
        )

        validation = GroundingValidationResult(
            status="unsupported",
            confidence=1.0,
            reason="Test.",
            issues=[],
        )

        service = ResponseModeRepairService()

        with patch(
            (
                "app.services."
                "response_mode_repair_service."
                "chat_with_ai"
            ),
            return_value="Repaired.",
        ) as mocked_ai:

            service.repair(
                original_response="Original.",
                knowledge_context="Knowledge.",
                validation=validation,
                learner_message="Learner.",
                scaffolding_level=1,
                strategy_name="guiding_question",
                strategy_instruction="Strategy.",
                intervention_name="none",
                pedagogical_issues=[],
                response_mode=mode,
            )

        self.assert_equal(
            mocked_ai.call_count,
            1,
            (
                "Non-preserving semantic repair "
                "must use exactly one repair LLM call."
            ),
        )

        self.assert_equal(
            mocked_ai.call_args.kwargs[
                "task_name"
            ],
            "response_repair",
            (
                "Existing repair task telemetry "
                "must be preserved."
            ),
        )

        self.pass_test(
            "REPAIR-T79 One repair LLM",
            (
                "Mode-aware semantic repair uses "
                "exactly one response_repair call."
            ),
        )

    def test_repair_t80_context_preserved(
        self,
    ) -> None:

        mode = (
            TutoringResponseStrategyPolicy()
            .decide(
                LearnerTurnIntentResult(
                    intent="follow_up_question",
                    reason="Test.",
                    is_question=True,
                )
            )
        )

        validation = GroundingValidationResult(
            status="unsupported",
            confidence=0.8,
            reason="CONTROLLED GROUNDING REASON",
            issues=[
                "CONTROLLED GROUNDING ISSUE",
            ],
        )

        service = ResponseModeRepairService()

        with patch(
            (
                "app.services."
                "response_mode_repair_service."
                "chat_with_ai"
            ),
            return_value="Repaired.",
        ) as mocked_ai:

            service.repair(
                original_response=(
                    "CONTROLLED ORIGINAL RESPONSE"
                ),
                knowledge_context=(
                    "CONTROLLED COURSE KNOWLEDGE"
                ),
                validation=validation,
                learner_message=(
                    "CONTROLLED LEARNER MESSAGE"
                ),
                scaffolding_level=2,
                strategy_name="hint",
                strategy_instruction=(
                    "CONTROLLED STRATEGY"
                ),
                intervention_name=(
                    "CONTROLLED INTERVENTION"
                ),
                pedagogical_issues=[
                    "CONTROLLED PEDAGOGICAL ISSUE",
                ],
                response_mode=mode,
                response_mode_instruction=(
                    "CONTROLLED MODE INSTRUCTION"
                ),
            )

        messages = mocked_ai.call_args.args[0]

        content = messages[1][
            "content"
        ]

        for marker in (
            "CONTROLLED COURSE KNOWLEDGE",
            "CONTROLLED ORIGINAL RESPONSE",
            "CONTROLLED LEARNER MESSAGE",
            "CONTROLLED GROUNDING REASON",
            "CONTROLLED GROUNDING ISSUE",
            "CONTROLLED PEDAGOGICAL ISSUE",
            "CONTROLLED STRATEGY",
            "CONTROLLED INTERVENTION",
            "CONTROLLED MODE INSTRUCTION",
        ):
            self.assert_true(
                marker in content,
                (
                    "Repair context missing marker: "
                    f"{marker}"
                ),
            )

        self.pass_test(
            "REPAIR-T80 Context preserved",
            (
                "Mode-aware repair preserves "
                "grounding, pedagogical, learner, "
                "strategy, and course context."
            ),
        )
    # =====================================================
    # 16.19.5D-2 Runtime Repair Generation Routing
    # =====================================================

    def test_repair_runtime_t81_adapter_initialized(
        self,
    ) -> None:

        tutor = AITutor()

        self.assert_true(
            isinstance(
                tutor.response_mode_repair_service,
                ResponseModeRepairService,
            ),
            (
                "AITutor must own the response-mode "
                "repair adapter."
            ),
        )

        self.assert_true(
            (
                tutor.response_mode_repair_service
                .base_repair_service
                is tutor.response_repair_service
            ),
            (
                "Response-mode repair adapter must "
                "wrap the existing repair service."
            ),
        )

        self.pass_test(
            "REPAIR-RUNTIME-T81 Adapter initialized",
            (
                "AITutor owns the mode-aware repair "
                "adapter while preserving the legacy "
                "repair service."
            ),
        )

    def test_repair_runtime_t82_calls_routed(
        self,
    ) -> None:

        source = inspect.getsource(
            AITutor.respond
        )

        repair_start = source.find(
            "RESPONSE REPAIR"
        )

        validation_start = source.find(
            "VALIDATE REPAIRED RESPONSE"
        )

        repair_section = source[
            repair_start:validation_start
        ]

        repair_compact = "".join(
            repair_section.split()
        )

        # -------------------------------------------------
        # Both normal semantic repair paths must route
        # through the provider-failure-aware wrapper.
        # -------------------------------------------------

        self.assert_equal(
            repair_compact.count(
                "self._run_response_repair("
            ),
            2,
            (
                "Both normal semantic repair paths must "
                "route through the runtime repair wrapper."
            ),
        )

        # -------------------------------------------------
        # The wrapper itself must delegate to the frozen
        # response-mode-aware repair service exactly once.
        #
        # Compact the source because the production code
        # may format:
        #
        # self.response_mode_repair_service
        #     .repair(...)
        #
        # across multiple lines.
        # -------------------------------------------------

        helper_source = inspect.getsource(
            AITutor._run_response_repair
        )

        helper_compact = "".join(
            helper_source.split()
        )

        self.assert_equal(
            helper_compact.count(
                "self.response_mode_repair_service.repair("
            ),
            1,
            (
                "The runtime repair wrapper must delegate "
                "exactly once to the response-mode-aware "
                "repair service."
            ),
        )

        self.pass_test(
            "REPAIR-RUNTIME-T82 Repair calls routed",
            (
                "Both normal semantic repair paths route "
                "through the provider-failure-aware runtime "
                "wrapper, which delegates to the frozen "
                "response-mode-aware repair service."
            ),
        )

    def test_repair_runtime_t83_mode_passed(
        self,
    ) -> None:

        source = inspect.getsource(
            AITutor.respond
        )

        repair_start = source.find(
            "RESPONSE REPAIR"
        )

        validation_start = source.find(
            "VALIDATE REPAIRED RESPONSE"
        )

        repair_section = source[
            repair_start:validation_start
        ]

        compact = "".join(
            repair_section.split()
        )

        self.assert_true(
            compact.count(
                "response_mode=(tutoring_response_mode)"
            )
            >= 2,
            (
                "Both repair calls must receive "
                "the current response mode."
            ),
        )

        self.pass_test(
            "REPAIR-RUNTIME-T83 Mode passed",
            (
                "Semantic repair preserves the "
                "current turn response mode."
            ),
        )

    def test_repair_runtime_t84_instruction_passed(
        self,
    ) -> None:

        source = inspect.getsource(
            AITutor.respond
        )

        repair_start = source.find(
            "RESPONSE REPAIR"
        )

        validation_start = source.find(
            "VALIDATE REPAIRED RESPONSE"
        )

        repair_section = source[
            repair_start:validation_start
        ]

        compact = "".join(
            repair_section.split()
        )

        self.assert_true(
            compact.count(
                (
                    "response_mode_instruction="
                    "(tutoring_response_mode_instruction)"
                )
            )
            >= 2,
            (
                "Both repair calls must receive "
                "the response-mode instruction."
            ),
        )

        self.pass_test(
            "REPAIR-RUNTIME-T84 Instruction passed",
            (
                "Repair generation receives the "
                "same turn-specific instruction "
                "used for Tutor generation."
            ),
        )

    def test_repair_runtime_t85_no_legacy_bypass(
        self,
    ) -> None:

        source = inspect.getsource(
            AITutor.respond
        )

        repair_start = source.find(
            "RESPONSE REPAIR"
        )

        validation_start = source.find(
            "VALIDATE REPAIRED RESPONSE"
        )

        repair_section = source[
            repair_start:validation_start
        ]

        self.assert_true(
            (
                "self.response_repair_service"
                not in repair_section
            ),
            (
                "Runtime semantic repair must not "
                "bypass the response-mode adapter."
            ),
        )

        self.pass_test(
            "REPAIR-RUNTIME-T85 No legacy bypass",
            (
                "Runtime repair generation cannot "
                "call the legacy service directly."
            ),
        )

    def test_repair_runtime_t86_deterministic_path_preserved(
        self,
    ) -> None:

        source = inspect.getsource(
            AITutor.respond
        )

        repair_start = source.find(
            "RESPONSE REPAIR"
        )

        validation_start = source.find(
            "VALIDATE REPAIRED RESPONSE"
        )

        repair_section = source[
            repair_start:validation_start
        ]

        self.assert_true(
            (
                "self.deterministic_response_repair"
                in repair_section
            ),
            (
                "Deterministic terminology repair "
                "must remain in the pipeline."
            ),
        )

        self.assert_true(
            (
                "repair_terminology"
                in repair_section
            ),
            (
                "Configured deterministic term "
                "replacement must remain available."
            ),
        )

        self.pass_test(
            "REPAIR-RUNTIME-T86 Deterministic path preserved",
            (
                "Mode-aware LLM repair integration "
                "does not replace deterministic "
                "terminology repair."
            ),
        )

    def run(self) -> None:

        print(
            "\nStep 16.19 "
            "Intent-aware Tutoring Response "
            "Regression Suite\n"
        )

        tests = [
            self.test_mode_t1_scaffolded,
            self.test_mode_t2_clarification,
            self.test_mode_t3_follow_up,
            self.test_mode_t4_frozen,
            self.test_mode_t5_invalid_mode,
            (
                self
                .test_mode_t6_invalid_question_policy
            ),
            self.test_mode_t7_deterministic,

            self.test_strategy_t8_initial,
            self.test_strategy_t9_answer,
            self.test_strategy_t10_follow_up,
            self.test_strategy_t11_clarification,
            self.test_strategy_t12_topic_change,
            self.test_strategy_t13_uncertain,
            self.test_strategy_t14_unknown_fail_safe,
            self.test_strategy_t15_invalid_type,
            self.test_strategy_t16_deterministic,
            self.test_strategy_t17_no_llm,

            self.test_runtime_t18_debug_defaults,
            self.test_runtime_t19_debug_projection,
            self.test_runtime_t20_policy_initialized,
            self.test_runtime_t21_observation_in_pipeline,
            self.test_runtime_t22_reset_clears,
            self.test_runtime_t23_per_turn_clear,
            self.test_runtime_t24_no_direct_branching,

            self.test_instruction_t25_follow_up,
            self.test_instruction_t26_clarification,
            self.test_instruction_t27_scaffolded_no_override,
            self.test_instruction_t28_new_topic_no_override,
            self.test_instruction_t29_invalid_type,
            self.test_instruction_t30_deterministic,
            self.test_instruction_t31_no_llm,

            self.test_prompt_t32_normal_unchanged,
            self.test_prompt_t33_follow_up_instruction,
            self.test_prompt_t34_clarification_instruction,
            self.test_prompt_t35_order,
            self.test_prompt_t36_style_preserved,
            self.test_prompt_t37_grounding_preserved,
            self.test_prompt_t38_deterministic,

            self.test_runtime_prompt_t39_builder_initialized,
            self.test_runtime_prompt_t40_instruction_default,
            self.test_runtime_prompt_t41_instruction_pipeline,
            self.test_runtime_prompt_t42_promptbuilder_receives,
            self.test_runtime_prompt_t43_preserving_mode_empty,
            self.test_runtime_prompt_t44_reset_clears_instruction,
            self.test_runtime_prompt_t45_debug_projection,

            self.test_pedagogy_t46_legacy_delegate,
            self.test_pedagogy_t47_direct_clarification,
            self.test_pedagogy_t48_follow_up_answer_then_question,
            self.test_pedagogy_t49_question_only_violation,
            self.test_pedagogy_t50_question_limit,
            self.test_pedagogy_t51_empty_violation,
            self.test_pedagogy_t52_invalid_type,
            self.test_pedagogy_t53_no_llm,

            self.test_pedagogy_t54_legacy_delegation,
            self.test_pedagogy_t55_mode_semantic_path,
            self.test_pedagogy_t56_mode_context,
            self.test_pedagogy_t57_invalid_json_fail_closed,
            self.test_pedagogy_t58_invalid_status_fail_closed,
            self.test_pedagogy_t59_confidence_clamp,
            self.test_pedagogy_t60_invalid_type,
            self.test_pedagogy_t61_preserving_no_new_llm,

            self.test_pedagogy_runtime_t62_adapters_initialized,
            self.test_pedagogy_runtime_t63_precheck_routed,
            self.test_pedagogy_runtime_t64_semantic_routed,
            self.test_pedagogy_runtime_t65_no_grounding_change,
            self.test_pedagogy_runtime_t66_legacy_owned,
            self.test_pedagogy_runtime_t67_no_direct_mode_branch,

            self.test_pedagogy_runtime_t68_repair_precheck_routed,
            self.test_pedagogy_runtime_t69_repair_semantic_routed,
            self.test_pedagogy_runtime_t70_no_direct_legacy_repair,
            self.test_pedagogy_runtime_t71_repair_grounding_preserved,
            self.test_pedagogy_runtime_t72_repair_task_preserved,

            self.test_repair_t73_legacy_delegation,
            self.test_repair_t74_follow_up_context,
            self.test_repair_t75_clarification_context,
            self.test_repair_t76_no_legacy_question_conflict,
            self.test_repair_t77_invalid_mode_type,
            self.test_repair_t78_preserving_no_added_llm,
            self.test_repair_t79_nonpreserving_one_llm,
            self.test_repair_t80_context_preserved,

            self.test_repair_runtime_t81_adapter_initialized,
            self.test_repair_runtime_t82_calls_routed,
            self.test_repair_runtime_t83_mode_passed,
            self.test_repair_runtime_t84_instruction_passed,
            self.test_repair_runtime_t85_no_legacy_bypass,
            self.test_repair_runtime_t86_deterministic_path_preserved,
        ]

        for test in tests:

            try:
                test()

            except Exception as error:
                self.fail_test(
                    test.__name__,
                    error,
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

        if self.failed:
            raise SystemExit(1)


if __name__ == "__main__":
    Step1619RegressionSuite().run()