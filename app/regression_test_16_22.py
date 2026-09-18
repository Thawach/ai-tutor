import json
import os
import subprocess
import sys
from dataclasses import asdict
import tempfile
from pathlib import Path

from app.tutor import AITutor

from unittest.mock import (
    MagicMock,
    patch,
)

import app.infrastructure.ai.groq_provider as groq_provider
import app.infrastructure.ai.provider_factory as provider_factory

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import (
    MagicMock,
    patch,
)


from dataclasses import replace

from app.services.knowledge_service import (
    KnowledgeService,
)

from app.vector_store import (
    add_document_chunk,
    get_existing_collection,
)

from app.vector_store import (
    DEFAULT_COLLECTION_NAME,
    add_document_chunk,
)

from app.courses.loader import (
    CourseProfileLoader,
)

from app.regression_test_16_21 import (
    RegressionSuite16_21,
)

from app.services.grounding_validation_models import (
    GroundingValidationResult,
)

from app.services.repair_evidence_selection_models import (
    RepairEvidenceSelectionResult,
)

from app.services.guiding_question_grounding_models import (
    GuidingQuestionGroundingResult,
)

@dataclass
class RuntimeTestResult:
    name: str
    passed: bool
    details: str = ""


class ControlledProviderFailure(RuntimeError):
    """
    Synthetic provider/runtime failure used only by
    Step 16.22 deterministic regression tests.
    """

    pass


class RegressionSuite16_22:

    def __init__(self) -> None:

        self.results: list[RuntimeTestResult] = []

    # =====================================================
    # Assertion helpers
    # =====================================================

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

    def assert_false(
        self,
        condition: bool,
        message: str,
    ) -> None:

        if condition:

            raise AssertionError(
                message
            )

    # =====================================================
    # Result helpers
    # =====================================================

    def pass_test(
        self,
        name: str,
        details: str,
    ) -> None:

        self.results.append(
            RuntimeTestResult(
                name=name,
                passed=True,
                details=details,
            )
        )

    def fail_test(
        self,
        name: str,
        details: str,
    ) -> None:

        self.results.append(
            RuntimeTestResult(
                name=name,
                passed=False,
                details=details,
            )
        )

    # =====================================================
    # Controlled Tutor
    # =====================================================

    def build_grounded_tutor(
        self,
    ) -> AITutor:

        tutor = AITutor()

        knowledge_result = SimpleNamespace(
            query="controlled NPN query",
            context=(
                "The transistor has three terminals: "
                "Base, Collector, and Emitter. "
                "The BJT has two junctions. "
                "The Base-Emitter junction may be "
                "forward biased or reverse biased."
            ),
            sources=[],
            citations=[],
        )

        # ---------------------------------------------
        # No real retrieval.
        # ---------------------------------------------

        tutor.knowledge_service.retrieve = (
            lambda query, n_results=5:
            knowledge_result
        )

        # ---------------------------------------------
        # Force in-course relevance.
        #
        # The test must reach Tutor generation.
        # ---------------------------------------------

        tutor.relevance_gate.evaluate = (
            lambda query, knowledge_result:
            SimpleNamespace(
                is_relevant=True,
                confidence=1.0,
                reason=(
                    "Controlled Step 16.22 "
                    "course relevance."
                ),
            )
        )

        # ---------------------------------------------
        # Force grounded knowledge availability.
        # ---------------------------------------------

        tutor.grounding_guard.evaluate = (
            lambda result, relevance:
            SimpleNamespace(
                has_knowledge=True,
                reason=(
                    "Controlled Step 16.22 "
                    "course knowledge is available."
                ),
            )
        )

        return tutor

    # =====================================================
    # RUNTIME-T1
    #
    # Tutor generation provider failure must:
    #
    # - propagate the original exception
    # - close the incomplete learning sequence
    # - not write the failed turn to memory
    # - not fabricate a Tutor answer
    # =====================================================

    def test_runtime_t1_tutor_generation_failure(
        self,
    ) -> None:

        name = (
            "RUNTIME-T1 Tutor generation "
            "provider failure recovery"
        )

        try:

            tutor = (
                self.build_grounded_tutor()
            )

            learner_message = (
                "ทรานซิสเตอร์ NPN "
                "มีโครงสร้างอย่างไร"
            )

            # -----------------------------------------
            # Preconditions
            # -----------------------------------------

            self.assert_equal(
                tutor.original_question,
                None,
                (
                    "RUNTIME-T1 active question "
                    "must start empty"
                ),
            )

            self.assert_false(
                tutor.waiting_for_response,
                (
                    "RUNTIME-T1 waiting state "
                    "must start inactive"
                ),
            )

            self.assert_equal(
                len(
                    tutor.memory.get_messages()
                ),
                0,
                (
                    "RUNTIME-T1 conversation "
                    "memory must start empty"
                ),
            )

            # -----------------------------------------
            # Controlled provider failure
            # -----------------------------------------

            propagated_error = None

            try:

                with patch(
                    "app.tutor.chat_with_ai",
                    side_effect=(
                        ControlledProviderFailure(
                            "Controlled provider failure"
                        )
                    ),
                ):

                    tutor.respond(
                        learner_message
                    )

            except ControlledProviderFailure as error:

                propagated_error = error

            # -----------------------------------------
            # Exception contract
            # -----------------------------------------

            self.assert_true(
                propagated_error is not None,
                (
                    "RUNTIME-T1 provider exception "
                    "must propagate to the caller"
                ),
            )

            self.assert_equal(
                str(propagated_error),
                "Controlled provider failure",
                (
                    "RUNTIME-T1 must preserve the "
                    "original provider exception"
                ),
            )

            # -----------------------------------------
            # Learning-sequence recovery
            # -----------------------------------------

            self.assert_equal(
                tutor.original_question,
                None,
                (
                    "RUNTIME-T1 failed generation "
                    "must clear active question"
                ),
            )

            self.assert_false(
                tutor.waiting_for_response,
                (
                    "RUNTIME-T1 failed generation "
                    "must close waiting state"
                ),
            )

            # -----------------------------------------
            # Memory isolation
            # -----------------------------------------

            self.assert_equal(
                len(
                    tutor.memory.get_messages()
                ),
                0,
                (
                    "RUNTIME-T1 failed turn must "
                    "not enter ConversationMemory"
                ),
            )

            # -----------------------------------------
            # No fabricated generation
            # -----------------------------------------

            debug = (
                tutor.get_debug_info()
            )

            self.assert_equal(
                debug.get(
                    "generated_answer"
                ),
                None,
                (
                    "RUNTIME-T1 provider failure "
                    "must not create generated answer"
                ),
            )

            self.assert_equal(
                debug.get(
                    "validation_status"
                ),
                None,
                (
                    "RUNTIME-T1 grounding "
                    "validation must not run after "
                    "generation failure"
                ),
            )

            self.assert_equal(
                debug.get(
                    "escalation_action"
                ),
                None,
                (
                    "RUNTIME-T1 repair escalation "
                    "must not run after generation "
                    "failure"
                ),
            )

            self.assert_false(
                debug.get(
                    "response_repaired",
                    False,
                ),
                (
                    "RUNTIME-T1 generation failure "
                    "must not create repaired output"
                ),
            )

            self.pass_test(
                name=name,
                details=(
                    "Tutor generation provider failure "
                    "propagates while the incomplete "
                    "learning sequence is closed, "
                    "ConversationMemory remains clean, "
                    "and no fabricated response enters "
                    "validation or repair."
                ),
            )

        except Exception as error:

            self.fail_test(
                name=name,
                details=str(error),
            )

    # =====================================================
    # RUNTIME-T2
    #
    # Learner evaluator provider failure must:
    #
    # - propagate the original exception
    # - preserve the active learning sequence
    # - preserve prior valid ConversationMemory
    # - not write the failed learner answer
    # - not fabricate an evaluation
    # - not mutate pedagogical state
    #
    # session turn_count may advance because next_turn()
    # records the attempted Tutor turn.
    # =====================================================

    def test_runtime_t2_evaluator_failure(
        self,
    ) -> None:

        name = (
            "RUNTIME-T2 Learner evaluator "
            "provider failure recovery"
        )

        try:

            tutor = AITutor()

            original_question = (
                "ทรานซิสเตอร์ NPN "
                "มีโครงสร้างอย่างไร"
            )

            tutor_question = (
                "คุณคิดว่าในทรานซิสเตอร์ NPN "
                "มีส่วนประกอบหลักสามส่วนคืออะไร?"
            )

            learner_answer = (
                "มี Base Collector และ Emitter"
            )

            # -----------------------------------------
            # Seed a valid prior Tutor turn.
            # -----------------------------------------

            tutor.original_question = (
                original_question
            )

            tutor.waiting_for_response = True

            tutor.memory.add_user_message(
                original_question
            )

            tutor.memory.add_assistant_message(
                tutor_question
            )

            # -----------------------------------------
            # Snapshot pedagogical state before failure.
            # -----------------------------------------

            turn_before = (
                tutor.state.turn_count
            )

            scaffolding_before = (
                tutor.state.scaffolding_level
            )

            attempt_before = (
                tutor.state.attempt_count
            )

            correct_streak_before = (
                tutor.state.correct_streak
            )

            partial_streak_before = (
                tutor.state.partial_streak
            )

            failure_streak_before = (
                tutor.state.failure_streak
            )

            misconception_count_before = len(
                tutor.state.misconceptions
            )

            # -----------------------------------------
            # Controlled evaluator/provider failure.
            # -----------------------------------------

            propagated_error = None

            try:

                with patch(
                    "app.tutor.evaluate_response",
                    side_effect=(
                        ControlledProviderFailure(
                            "Controlled evaluator "
                            "provider failure"
                        )
                    ),
                ):

                    tutor.respond(
                        learner_answer
                    )

            except ControlledProviderFailure as error:

                propagated_error = error

            # -----------------------------------------
            # Exception contract
            # -----------------------------------------

            self.assert_true(
                propagated_error is not None,
                (
                    "RUNTIME-T2 evaluator provider "
                    "exception must propagate"
                ),
            )

            self.assert_equal(
                str(propagated_error),
                (
                    "Controlled evaluator "
                    "provider failure"
                ),
                (
                    "RUNTIME-T2 must preserve the "
                    "original evaluator exception"
                ),
            )

            # -----------------------------------------
            # Active sequence must survive.
            # -----------------------------------------

            self.assert_equal(
                tutor.original_question,
                original_question,
                (
                    "RUNTIME-T2 evaluator failure "
                    "must preserve active question"
                ),
            )

            self.assert_true(
                tutor.waiting_for_response,
                (
                    "RUNTIME-T2 evaluator failure "
                    "must preserve waiting state"
                ),
            )

            # -----------------------------------------
            # Prior memory must remain unchanged.
            # -----------------------------------------

            expected_memory = [
                {
                    "role": "user",
                    "content": original_question,
                },
                {
                    "role": "assistant",
                    "content": tutor_question,
                },
            ]

            self.assert_equal(
                tutor.memory.get_messages(),
                expected_memory,
                (
                    "RUNTIME-T2 failed learner answer "
                    "must not enter ConversationMemory"
                ),
            )

            # -----------------------------------------
            # No fabricated evaluation.
            # -----------------------------------------

            self.assert_equal(
                tutor.last_evaluation_result,
                None,
                (
                    "RUNTIME-T2 evaluator failure "
                    "must not create EvaluationResult"
                ),
            )

            self.assert_equal(
                tutor.last_decision,
                None,
                (
                    "RUNTIME-T2 evaluator failure "
                    "must not create scaffolding decision"
                ),
            )

            # -----------------------------------------
            # Pedagogical state must not mutate.
            # -----------------------------------------

            self.assert_equal(
                tutor.state.scaffolding_level,
                scaffolding_before,
                (
                    "RUNTIME-T2 evaluator failure "
                    "must preserve scaffolding level"
                ),
            )

            self.assert_equal(
                tutor.state.attempt_count,
                attempt_before,
                (
                    "RUNTIME-T2 evaluator failure "
                    "must not increment attempt count"
                ),
            )

            self.assert_equal(
                tutor.state.correct_streak,
                correct_streak_before,
                (
                    "RUNTIME-T2 evaluator failure "
                    "must preserve correct streak"
                ),
            )

            self.assert_equal(
                tutor.state.partial_streak,
                partial_streak_before,
                (
                    "RUNTIME-T2 evaluator failure "
                    "must preserve partial streak"
                ),
            )

            self.assert_equal(
                tutor.state.failure_streak,
                failure_streak_before,
                (
                    "RUNTIME-T2 evaluator failure "
                    "must preserve failure streak"
                ),
            )

            self.assert_equal(
                len(
                    tutor.state.misconceptions
                ),
                misconception_count_before,
                (
                    "RUNTIME-T2 evaluator failure "
                    "must not create misconception"
                ),
            )

            # -----------------------------------------
            # Session turn observation is allowed.
            # -----------------------------------------

            self.assert_equal(
                tutor.state.turn_count,
                turn_before + 1,
                (
                    "RUNTIME-T2 failed evaluator turn "
                    "remains a counted session turn"
                ),
            )

            # -----------------------------------------
            # Per-turn telemetry
            # -----------------------------------------

            debug = tutor.get_debug_info()

            self.assert_equal(
                debug.get(
                    "learner_turn_intent"
                ),
                "answer",
                (
                    "RUNTIME-T2 learner intent "
                    "should remain observable"
                ),
            )

            self.assert_equal(
                debug.get(
                    "learner_turn_route"
                ),
                "evaluate_answer",
                (
                    "RUNTIME-T2 evaluator route "
                    "should remain observable"
                ),
            )

            self.assert_equal(
                debug.get(
                    "generated_answer"
                ),
                None,
                (
                    "RUNTIME-T2 Tutor generation "
                    "must not run after evaluator failure"
                ),
            )

            self.assert_equal(
                debug.get(
                    "validation_status"
                ),
                None,
                (
                    "RUNTIME-T2 validation "
                    "must not run after evaluator failure"
                ),
            )

            self.pass_test(
                name=name,
                details=(
                    "Evaluator provider failure propagates "
                    "without corrupting the active learning "
                    "sequence, prior ConversationMemory, "
                    "evaluation state, scaffolding state, "
                    "streaks, attempts, or misconceptions; "
                    "the attempted session turn remains "
                    "observable."
                ),
            )

        except Exception as error:

            self.fail_test(
                name=name,
                details=str(error),
            )

    # =====================================================
    # RUNTIME-T3
    #
    # Repair provider failure must:
    #
    # - propagate the original provider exception
    # - never return the rejected unsafe Tutor answer
    # - preserve valid prior ConversationMemory
    # - preserve the active learning sequence
    # - not fabricate a repair candidate
    # - expose repair-provider failure telemetry
    # - not mutate pedagogical state
    # =====================================================

    def test_runtime_t3_repair_provider_failure(
        self,
    ) -> None:

        name = (
            "RUNTIME-T3 Repair provider "
            "failure recovery"
        )

        try:

            # Reuse the proven controlled E2E Tutor fixture
            # from Step 16.21.
            suite_16_21 = RegressionSuite16_21()

            tutor = (
                suite_16_21
                .build_grounded_tutor()
            )

            original_question = (
                "ทรานซิสเตอร์ NPN "
                "มีโครงสร้างอย่างไร"
            )

            prior_tutor_response = (
                "คุณคิดว่า Base, Collector และ "
                "Emitter เป็นส่วนสำคัญอย่างไร?"
            )

            learner_message = (
                "ช่วยอธิบายรอยต่อ Base-Emitter "
                "ให้ชัดเจนอีกครั้ง"
            )

            unsafe_initial = (
                "รอยต่อ Base-Emitter ของทรานซิสเตอร์ "
                "NPN ต้องใช้แรงดันประมาณ 0.7 V "
                "จึงจะทำงาน"
            )

            verified_quote = (
                "The Base-Emitter junction may be "
                "forward biased or reverse biased."
            )

            # ---------------------------------------------
            # Seed valid prior learning sequence
            # ---------------------------------------------

            tutor.original_question = (
                original_question
            )

            tutor.waiting_for_response = True

            tutor.memory.add_user_message(
                original_question
            )

            tutor.memory.add_assistant_message(
                prior_tutor_response
            )

            memory_before = list(
                tutor.memory.get_messages()
            )

            turn_before = (
                tutor.state.turn_count
            )

            scaffolding_before = (
                tutor.state.scaffolding_level
            )

            attempt_before = (
                tutor.state.attempt_count
            )

            correct_streak_before = (
                tutor.state.correct_streak
            )

            partial_streak_before = (
                tutor.state.partial_streak
            )

            failure_streak_before = (
                tutor.state.failure_streak
            )

            misconception_count_before = len(
                tutor.state.misconceptions
            )

            # ---------------------------------------------
            # Initial unsafe generation is rejected
            # ---------------------------------------------

            def controlled_grounding_validation(
                response: str,
                knowledge_context: str,
                task_name: str = "response_validator",
            ) -> GroundingValidationResult:

                if response == unsafe_initial:

                    return GroundingValidationResult(
                        status="unsupported",
                        confidence=1.0,
                        reason=(
                            "The initial response introduces "
                            "an exact 0.7 V requirement that "
                            "is absent from course knowledge."
                        ),
                        issues=[
                            (
                                "Unsupported exact numerical "
                                "B-E voltage requirement."
                            )
                        ],
                    )

                raise AssertionError(
                    "Unexpected response sent to "
                    "grounding validator: "
                    f"{response!r}"
                )

            tutor.response_grounding_validator.validate = (
                controlled_grounding_validation
            )

            # ---------------------------------------------
            # Verified evidence exists for repair
            # ---------------------------------------------

            def controlled_evidence_selection(
                learner_message: str,
                knowledge_context: str,
                validation_issues=None,
                **kwargs,
            ) -> RepairEvidenceSelectionResult:

                return RepairEvidenceSelectionResult(
                    status="verified",
                    evidence_quotes=(
                        verified_quote,
                    ),
                    reason=(
                        "Controlled exact course "
                        "evidence selected."
                    ),
                    issues=(),
                )

            tutor.repair_evidence_selector.select = (
                controlled_evidence_selection
            )

            # Keep semantic pedagogy deterministic.
            tutor.response_mode_pedagogical_validator.validate = (
                lambda *args, **kwargs:
                type(
                    "ControlledPedagogy",
                    (),
                    {
                        "status": "valid",
                        "confidence": 1.0,
                        "reason": (
                            "Controlled direct response "
                            "is pedagogically valid."
                        ),
                        "issues": [],
                    },
                )()
            )

            # ---------------------------------------------
            # Controlled repair-provider failure
            # ---------------------------------------------

            repair_calls = []

            def controlled_repair_failure(
                *args,
                **kwargs,
            ):

                repair_calls.append(
                    kwargs
                )

                raise ControlledProviderFailure(
                    "Controlled repair provider failure"
                )

            tutor.response_mode_repair_service.repair = (
                controlled_repair_failure
            )

            propagated_error = None
            returned_answer = None

            try:

                with patch(
                    "app.tutor.chat_with_ai",
                    return_value=unsafe_initial,
                ):

                    returned_answer = (
                        tutor.respond(
                            learner_message
                        )
                    )

            except ControlledProviderFailure as error:

                propagated_error = error

            # ---------------------------------------------
            # Runtime safety
            # ---------------------------------------------

            self.assert_true(
                propagated_error is not None,
                (
                    "RUNTIME-T3 repair provider "
                    "exception must propagate"
                ),
            )

            self.assert_equal(
                str(propagated_error),
                "Controlled repair provider failure",
                (
                    "RUNTIME-T3 must preserve the "
                    "original repair-provider exception"
                ),
            )

            self.assert_equal(
                returned_answer,
                None,
                (
                    "RUNTIME-T3 rejected unsafe answer "
                    "must never be returned"
                ),
            )

            self.assert_equal(
                len(repair_calls),
                1,
                (
                    "RUNTIME-T3 repair generation "
                    "must be attempted exactly once"
                ),
            )

            # ---------------------------------------------
            # State / memory isolation
            # ---------------------------------------------

            self.assert_equal(
                tutor.memory.get_messages(),
                memory_before,
                (
                    "RUNTIME-T3 failed repair turn "
                    "must not enter ConversationMemory"
                ),
            )

            self.assert_equal(
                tutor.original_question,
                original_question,
                (
                    "RUNTIME-T3 repair failure must "
                    "preserve the active question"
                ),
            )

            self.assert_true(
                tutor.waiting_for_response,
                (
                    "RUNTIME-T3 repair failure must "
                    "preserve waiting state"
                ),
            )

            self.assert_equal(
                tutor.state.turn_count,
                turn_before + 1,
                (
                    "RUNTIME-T3 failed repair remains "
                    "an observable attempted session turn"
                ),
            )

            self.assert_equal(
                tutor.state.scaffolding_level,
                scaffolding_before,
                (
                    "RUNTIME-T3 repair failure must "
                    "not change scaffolding level"
                ),
            )

            self.assert_equal(
                tutor.state.attempt_count,
                attempt_before,
                (
                    "RUNTIME-T3 repair failure must "
                    "not change attempt count"
                ),
            )

            self.assert_equal(
                tutor.state.correct_streak,
                correct_streak_before,
                (
                    "RUNTIME-T3 repair failure must "
                    "preserve correct streak"
                ),
            )

            self.assert_equal(
                tutor.state.partial_streak,
                partial_streak_before,
                (
                    "RUNTIME-T3 repair failure must "
                    "preserve partial streak"
                ),
            )

            self.assert_equal(
                tutor.state.failure_streak,
                failure_streak_before,
                (
                    "RUNTIME-T3 repair failure must "
                    "preserve failure streak"
                ),
            )

            self.assert_equal(
                len(
                    tutor.state.misconceptions
                ),
                misconception_count_before,
                (
                    "RUNTIME-T3 repair failure must "
                    "not create a misconception"
                ),
            )

            # ---------------------------------------------
            # Failure telemetry
            # ---------------------------------------------

            debug = tutor.get_debug_info()

            self.assert_equal(
                debug.get(
                    "generated_answer"
                ),
                unsafe_initial,
                (
                    "RUNTIME-T3 rejected initial "
                    "generation remains observable"
                ),
            )

            self.assert_equal(
                debug.get(
                    "repair_candidate"
                ),
                None,
                (
                    "RUNTIME-T3 provider failure "
                    "must not fabricate a repair candidate"
                ),
            )

            self.assert_equal(
                debug.get(
                    "repair_attempts"
                ),
                1,
                (
                    "RUNTIME-T3 must record exactly "
                    "one repair attempt"
                ),
            )

            self.assert_true(
                debug.get(
                    "repair_failed"
                ),
                (
                    "RUNTIME-T3 repair provider "
                    "failure must be observable"
                ),
            )

            self.assert_equal(
                debug.get(
                    "repair_failure_type"
                ),
                "repair_provider",
                (
                    "RUNTIME-T3 must classify the "
                    "runtime failure as repair_provider"
                ),
            )

            self.assert_equal(
                debug.get(
                    "repair_failure_reason"
                ),
                "Controlled repair provider failure",
                (
                    "RUNTIME-T3 must preserve the "
                    "repair-provider failure reason"
                ),
            )

            self.assert_equal(
                debug.get(
                    "repair_evidence_status"
                ),
                "verified",
                (
                    "RUNTIME-T3 must preserve the "
                    "verified repair-evidence telemetry"
                ),
            )

            self.pass_test(
                name=name,
                details=(
                    "Repair provider failure propagates "
                    "without exposing the rejected unsafe "
                    "answer, fabricating a repair candidate, "
                    "polluting ConversationMemory, or "
                    "mutating pedagogical state; provider "
                    "failure telemetry remains explicit."
                ),
            )

        except Exception as error:

            self.fail_test(
                name=name,
                details=str(error),
            )

    def test_runtime_t4_c3_provider_failure(
        self,
    ) -> None:

        name = (
            "RUNTIME-T4 C3 recovery provider "
            "failure boundary"
        )

        try:

            suite_16_21 = RegressionSuite16_21()

            tutor = (
                suite_16_21
                .build_grounded_tutor()
            )

            learner_message = (
                "แรงดันระหว่าง Base และ Emitter "
                "มีผลต่อกระแส Collector อย่างไร"
            )

            unsafe_initial = (
                "แรงดันระหว่าง Base และ Emitter "
                "ต้องมีอย่างน้อยกี่โวลต์ "
                "จึงจะทำให้กระแส Collector เพิ่มขึ้น?"
            )

            unsafe_normal_repair = (
                "แรงดันระหว่าง Base และ Emitter "
                "ต้องมากพอแค่ไหน "
                "จึงจะทำให้กระแส Collector เพิ่มขึ้น?"
            )

            verified_quote = (
                "Therefore, the collector current is related "
                "to the emitter current which is in turn a "
                "function of the B-E voltage."
            )

            full_knowledge_context = (
                "The transistor has three terminals: "
                "Base, Collector, and Emitter. "
                "The BJT has two junctions. "
                "The Base-Emitter junction may be "
                "forward biased or reverse biased. "
                + verified_quote
            )

            knowledge_result = SimpleNamespace(
                query="controlled B-E relation query",
                context=full_knowledge_context,
                sources=[],
                citations=[],
            )

            tutor.knowledge_service.retrieve = (
                lambda query, n_results=5:
                knowledge_result
            )

            # ---------------------------------------------
            # Statement-level grounding remains supported.
            # The unsafe content is in the guiding question.
            # ---------------------------------------------

            def controlled_grounding_validation(
                response: str,
                knowledge_context: str,
                task_name: str = "response_validator",
            ) -> GroundingValidationResult:

                if response in (
                    unsafe_initial,
                    unsafe_normal_repair,
                ):

                    return GroundingValidationResult(
                        status="supported",
                        confidence=1.0,
                        reason=(
                            "Controlled response contains "
                            "no unsupported factual assertion "
                            "at statement level."
                        ),
                        issues=[],
                    )

                raise AssertionError(
                    "Unexpected response sent to "
                    "grounding validator: "
                    f"{response!r}"
                )

            tutor.response_grounding_validator.validate = (
                controlled_grounding_validation
            )

            # ---------------------------------------------
            # Initial + normal repair guiding questions
            # remain knowledge-unsafe.
            # ---------------------------------------------

            def controlled_guiding_validation(
                response: str,
                knowledge_context: str,
            ) -> GuidingQuestionGroundingResult:

                if response == unsafe_initial:

                    return GuidingQuestionGroundingResult(
                        status="unsupported",
                        reason=(
                            "Course knowledge provides no "
                            "explicit minimum B-E voltage."
                        ),
                        issues=(
                            "Unsupported exact voltage "
                            "threshold requested.",
                        ),
                    )

                if response == unsafe_normal_repair:

                    return GuidingQuestionGroundingResult(
                        status="unsupported",
                        reason=(
                            "Course knowledge provides no "
                            "criterion for how much B-E "
                            "voltage is enough."
                        ),
                        issues=(
                            "Unsupported sufficient-threshold "
                            "question.",
                        ),
                    )

                raise AssertionError(
                    "Unexpected response sent to guiding-"
                    "question validator: "
                    f"{response!r}"
                )

            tutor.guiding_question_grounding_validator.validate = (
                controlled_guiding_validation
            )

            # ---------------------------------------------
            # Verified evidence required by C3.
            # ---------------------------------------------

            def controlled_evidence_selection(
                learner_message: str,
                knowledge_context: str,
                validation_issues=None,
                **kwargs,
            ) -> RepairEvidenceSelectionResult:

                return RepairEvidenceSelectionResult(
                    status="verified",
                    evidence_quotes=(
                        verified_quote,
                    ),
                    reason=(
                        "Controlled exact relation evidence."
                    ),
                    issues=(),
                )

            tutor.repair_evidence_selector.select = (
                controlled_evidence_selection
            )

            tutor.pedagogical_response_validator.validate = (
                lambda *args, **kwargs:
                SimpleNamespace(
                    status="valid",
                    confidence=1.0,
                    reason=(
                        "Controlled pedagogical result."
                    ),
                    issues=[],
                )
            )

            tutor.response_mode_pedagogical_validator.validate = (
                lambda *args, **kwargs:
                SimpleNamespace(
                    status="valid",
                    confidence=1.0,
                    reason=(
                        "Controlled response-mode "
                        "pedagogy."
                    ),
                    issues=[],
                )
            )

            # ---------------------------------------------
            # Call #1 = normal repair.
            # Call #2 = bounded C3 provider failure.
            # ---------------------------------------------

            repair_calls = []

            def controlled_repair(
                *args,
                **kwargs,
            ):

                repair_calls.append(
                    kwargs
                )

                if len(repair_calls) == 1:

                    return unsafe_normal_repair

                if len(repair_calls) == 2:

                    raise ControlledProviderFailure(
                        "Controlled C3 provider failure"
                    )

                raise AssertionError(
                    "C3 recovery exceeded its "
                    "single additional generation call."
                )

            tutor.response_mode_repair_service.repair = (
                controlled_repair
            )

            memory_before = list(
                tutor.memory.get_messages()
            )

            turn_before = (
                tutor.state.turn_count
            )

            scaffolding_before = (
                tutor.state.scaffolding_level
            )

            attempt_before = (
                tutor.state.attempt_count
            )

            propagated_error = None
            returned_answer = None

            try:

                with patch(
                    "app.tutor.chat_with_ai",
                    return_value=unsafe_initial,
                ):

                    returned_answer = (
                        tutor.respond(
                            learner_message
                        )
                    )

            except ControlledProviderFailure as error:

                propagated_error = error

            # ---------------------------------------------
            # Provider/runtime boundary
            # ---------------------------------------------

            self.assert_true(
                propagated_error is not None,
                (
                    "RUNTIME-T4 C3 provider "
                    "exception must propagate"
                ),
            )

            self.assert_equal(
                str(propagated_error),
                "Controlled C3 provider failure",
                (
                    "RUNTIME-T4 must preserve the "
                    "original C3 provider exception"
                ),
            )

            self.assert_equal(
                returned_answer,
                None,
                (
                    "RUNTIME-T4 must not return an "
                    "unsafe or fabricated answer"
                ),
            )

            # ---------------------------------------------
            # Frozen C3 invocation contract
            # ---------------------------------------------

            self.assert_equal(
                len(repair_calls),
                2,
                (
                    "RUNTIME-T4 must contain one normal "
                    "repair plus exactly one C3 generation"
                ),
            )

            self.assert_equal(
                repair_calls[0].get(
                    "original_response"
                ),
                unsafe_initial,
                (
                    "RUNTIME-T4 normal repair must "
                    "operate on the initial response"
                ),
            )

            self.assert_equal(
                repair_calls[1].get(
                    "original_response"
                ),
                unsafe_normal_repair,
                (
                    "RUNTIME-T4 C3 must operate on the "
                    "rejected normal repair candidate"
                ),
            )

            self.assert_equal(
                repair_calls[1].get(
                    "knowledge_context"
                ),
                tutor.last_repair_evidence_context,
                (
                    "RUNTIME-T4 C3 must remain bounded "
                    "to verified repair evidence"
                ),
            )

            self.assert_equal(
                repair_calls[1].get(
                    "strategy_name"
                ),
                "guiding_question",
                (
                    "RUNTIME-T4 must preserve the "
                    "guiding-question C3 strategy"
                ),
            )

            # ---------------------------------------------
            # State / memory recovery
            # ---------------------------------------------

            self.assert_equal(
                tutor.memory.get_messages(),
                memory_before,
                (
                    "RUNTIME-T4 failed C3 turn must not "
                    "enter ConversationMemory"
                ),
            )

            self.assert_equal(
                tutor.original_question,
                None,
                (
                    "RUNTIME-T4 failed initial tutoring "
                    "sequence must be closed"
                ),
            )

            self.assert_true(
                tutor.waiting_for_response is False,
                (
                    "RUNTIME-T4 must not leave Tutor "
                    "waiting for a response to an answer "
                    "that was never delivered"
                ),
            )

            self.assert_equal(
                tutor.state.turn_count,
                turn_before + 1,
                (
                    "RUNTIME-T4 attempted turn remains "
                    "observable at session level"
                ),
            )

            self.assert_equal(
                tutor.state.scaffolding_level,
                scaffolding_before,
                (
                    "RUNTIME-T4 must not contaminate "
                    "scaffolding state"
                ),
            )

            self.assert_equal(
                tutor.state.attempt_count,
                attempt_before,
                (
                    "RUNTIME-T4 must not contaminate "
                    "attempt state"
                ),
            )

            # ---------------------------------------------
            # Telemetry
            # ---------------------------------------------

            debug = tutor.get_debug_info()

            self.assert_equal(
                debug.get(
                    "generated_answer"
                ),
                unsafe_initial,
                (
                    "RUNTIME-T4 rejected initial "
                    "generation remains observable"
                ),
            )

            self.assert_equal(
                debug.get(
                    "repair_candidate"
                ),
                unsafe_normal_repair,
                (
                    "RUNTIME-T4 must preserve the "
                    "rejected normal repair candidate"
                ),
            )

            self.assert_equal(
                (
                    tutor
                    .last_evidence_safe_repair_candidate
                ),
                None,
                (
                    "RUNTIME-T4 failed C3 generation "
                    "must not fabricate a C3 candidate"
                ),
            )

            self.assert_true(
                debug.get(
                    "repair_failed"
                ),
                (
                    "RUNTIME-T4 C3 provider failure "
                    "must be observable"
                ),
            )

            self.assert_equal(
                debug.get(
                    "repair_failure_type"
                ),
                "repair_provider",
                (
                    "RUNTIME-T4 must classify C3 runtime "
                    "failure as repair_provider"
                ),
            )

            self.assert_equal(
                debug.get(
                    "repair_failure_reason"
                ),
                "Controlled C3 provider failure",
                (
                    "RUNTIME-T4 must preserve the C3 "
                    "provider failure reason"
                ),
            )

            self.assert_equal(
                debug.get(
                    "repair_evidence_status"
                ),
                "verified",
                (
                    "RUNTIME-T4 must preserve verified "
                    "evidence telemetry"
                ),
            )

            self.pass_test(
                name=name,
                details=(
                    "The single evidence-bounded C3 "
                    "generation may fail at the provider "
                    "boundary without exposing unsafe "
                    "output, retrying C3, polluting "
                    "ConversationMemory, or leaving an "
                    "undelivered tutoring sequence active; "
                    "failure telemetry remains explicit."
                ),
            )

        except Exception as error:

            self.fail_test(
                name=name,
                details=str(error),
            )


    def test_runtime_t5_evidence_selector_provider_failure(
        self,
    ) -> None:

        name = (
            "RUNTIME-T5 Evidence selector "
            "provider failure boundary"
        )

        try:

            suite_16_21 = RegressionSuite16_21()

            tutor = (
                suite_16_21
                .build_grounded_tutor()
            )

            learner_message = (
                "อธิบายการทำงานของรอยต่อ "
                "Base-Emitter ของทรานซิสเตอร์ NPN"
            )

            unsafe_initial = (
                "รอยต่อ Base-Emitter ของทรานซิสเตอร์ "
                "NPN ต้องใช้แรงดันประมาณ 0.7 V "
                "จึงจะทำงาน"
            )

            knowledge_context = (
                "The transistor has three terminals: "
                "Base, Collector, and Emitter. "
                "The Base-Emitter junction may be "
                "forward biased or reverse biased."
            )

            knowledge_result = SimpleNamespace(
                query="controlled B-E query",
                context=knowledge_context,
                sources=[],
                citations=[],
            )

            tutor.knowledge_service.retrieve = (
                lambda query, n_results=5:
                knowledge_result
            )

            # ---------------------------------------------
            # Force factual repair / evidence selection.
            # ---------------------------------------------

            def controlled_grounding_validation(
                response: str,
                knowledge_context: str,
                task_name: str = "response_validator",
            ) -> GroundingValidationResult:

                if response == unsafe_initial:

                    return GroundingValidationResult(
                        status="unsupported",
                        confidence=1.0,
                        reason=(
                            "Controlled unsupported exact "
                            "0.7 V requirement."
                        ),
                        issues=[
                            (
                                "Exact 0.7 V requirement is "
                                "not present in course knowledge."
                            )
                        ],
                    )

                raise AssertionError(
                    "Unexpected response sent to "
                    "grounding validator: "
                    f"{response!r}"
                )

            tutor.response_grounding_validator.validate = (
                controlled_grounding_validation
            )

            # ---------------------------------------------
            # Repair generation must never be reached.
            # ---------------------------------------------

            repair_calls = []

            def forbidden_repair(
                *args,
                **kwargs,
            ):

                repair_calls.append(
                    kwargs
                )

                raise AssertionError(
                    "Repair generation must not run after "
                    "evidence selector provider failure."
                )

            tutor.response_mode_repair_service.repair = (
                forbidden_repair
            )

            memory_before = list(
                tutor.memory.get_messages()
            )

            turn_before = (
                tutor.state.turn_count
            )

            scaffolding_before = (
                tutor.state.scaffolding_level
            )

            attempt_before = (
                tutor.state.attempt_count
            )

            correct_streak_before = (
                tutor.state.correct_streak
            )

            partial_streak_before = (
                tutor.state.partial_streak
            )

            failure_streak_before = (
                tutor.state.failure_streak
            )

            misconception_count_before = len(
                tutor.state.misconceptions
            )

            propagated_error = None
            returned_answer = None

            try:

                with patch(
                    "app.tutor.chat_with_ai",
                    return_value=unsafe_initial,
                ):

                    with patch(
                        (
                            "app.services."
                            "repair_evidence_selector."
                            "chat_with_structured_ai"
                        ),
                        side_effect=(
                            ControlledProviderFailure(
                                "Controlled evidence selector "
                                "provider failure"
                            )
                        ),
                    ) as selector_provider:

                        returned_answer = (
                            tutor.respond(
                                learner_message
                            )
                        )

            except Exception as error:

                propagated_error = error

            # ---------------------------------------------
            # Provider failure must be contained.
            # ---------------------------------------------

            self.assert_equal(
                propagated_error,
                None,
                (
                    "RUNTIME-T5 evidence selector "
                    "provider failure must be contained"
                ),
            )

            self.assert_true(
                returned_answer is not None,
                (
                    "RUNTIME-T5 must return the "
                    "deterministic grounding fallback"
                ),
            )

            self.assert_true(
                returned_answer != unsafe_initial,
                (
                    "RUNTIME-T5 unsafe initial answer "
                    "must never reach the learner"
                ),
            )

            self.assert_equal(
                selector_provider.call_count,
                1,
                (
                    "RUNTIME-T5 provider failure must "
                    "not trigger evidence reselection"
                ),
            )

            self.assert_equal(
                len(repair_calls),
                0,
                (
                    "RUNTIME-T5 repair generation must "
                    "remain blocked without verified evidence"
                ),
            )

            # ---------------------------------------------
            # State / memory isolation
            # ---------------------------------------------

            self.assert_equal(
                tutor.memory.get_messages(),
                memory_before,
                (
                    "RUNTIME-T5 failed factual turn must "
                    "not enter ConversationMemory"
                ),
            )

            self.assert_equal(
                tutor.original_question,
                None,
                (
                    "RUNTIME-T5 failed factual sequence "
                    "must be closed"
                ),
            )

            self.assert_true(
                tutor.waiting_for_response is False,
                (
                    "RUNTIME-T5 must not leave an "
                    "undelivered tutoring sequence active"
                ),
            )

            self.assert_equal(
                tutor.state.turn_count,
                turn_before + 1,
                (
                    "RUNTIME-T5 attempted turn remains "
                    "observable at session level"
                ),
            )

            self.assert_equal(
                tutor.state.scaffolding_level,
                scaffolding_before,
                (
                    "RUNTIME-T5 must preserve "
                    "scaffolding state"
                ),
            )

            self.assert_equal(
                tutor.state.attempt_count,
                attempt_before,
                (
                    "RUNTIME-T5 must preserve "
                    "attempt state"
                ),
            )

            self.assert_equal(
                tutor.state.correct_streak,
                correct_streak_before,
                (
                    "RUNTIME-T5 must preserve "
                    "correct streak"
                ),
            )

            self.assert_equal(
                tutor.state.partial_streak,
                partial_streak_before,
                (
                    "RUNTIME-T5 must preserve "
                    "partial streak"
                ),
            )

            self.assert_equal(
                tutor.state.failure_streak,
                failure_streak_before,
                (
                    "RUNTIME-T5 must preserve "
                    "failure streak"
                ),
            )

            self.assert_equal(
                len(
                    tutor.state.misconceptions
                ),
                misconception_count_before,
                (
                    "RUNTIME-T5 must not create "
                    "misconceptions"
                ),
            )

            # ---------------------------------------------
            # Evidence / failure telemetry
            # ---------------------------------------------

            debug = tutor.get_debug_info()

            self.assert_equal(
                debug.get(
                    "generated_answer"
                ),
                unsafe_initial,
                (
                    "RUNTIME-T5 rejected initial "
                    "generation remains observable"
                ),
            )

            self.assert_equal(
                debug.get(
                    "repair_evidence_status"
                ),
                "invalid",
                (
                    "RUNTIME-T5 provider failure must "
                    "produce invalid evidence status"
                ),
            )

            self.assert_equal(
                debug.get(
                    "repair_evidence_quotes"
                ),
                [],
                (
                    "RUNTIME-T5 failed evidence selector "
                    "must expose no evidence quotes"
                ),
            )

            self.assert_equal(
                debug.get(
                    "repair_evidence_selection_attempts"
                ),
                1,
                (
                    "RUNTIME-T5 provider failure must "
                    "stop after one selector attempt"
                ),
            )

            self.assert_true(
                debug.get(
                    "repair_evidence_retry_used"
                )
                is False,
                (
                    "RUNTIME-T5 provider failure must "
                    "not activate usability retry"
                ),
            )

            self.assert_equal(
                debug.get(
                    "repair_candidate"
                ),
                None,
                (
                    "RUNTIME-T5 must not fabricate "
                    "a repair candidate"
                ),
            )

            self.assert_true(
                debug.get(
                    "repair_failed"
                ),
                (
                    "RUNTIME-T5 failure must be "
                    "observable"
                ),
            )

            self.assert_equal(
                debug.get(
                    "repair_failure_type"
                ),
                "repair_evidence",
                (
                    "RUNTIME-T5 failure must be "
                    "classified as repair_evidence"
                ),
            )

            self.assert_equal(
                debug.get(
                    "repair_failure_reason"
                ),
                (
                    "Repair evidence selection could "
                    "not be completed because the "
                    "selector service failed."
                ),
                (
                    "RUNTIME-T5 must preserve the "
                    "selector fail-closed reason"
                ),
            )

            evidence_issues = (
                debug.get(
                    "repair_evidence_issues"
                )
                or []
            )

            self.assert_true(
                any(
                    "ControlledProviderFailure"
                    in str(issue)
                    for issue in evidence_issues
                ),
                (
                    "RUNTIME-T5 evidence telemetry must "
                    "retain provider failure provenance"
                ),
            )

            self.pass_test(
                name=name,
                details=(
                    "Evidence-selector structured-provider "
                    "failure is contained and converted to "
                    "invalid empty evidence; repair generation "
                    "is blocked, the unsafe initial answer "
                    "never reaches the learner, the factual "
                    "sequence closes cleanly, and no failed "
                    "turn enters ConversationMemory."
                ),
            )

        except Exception as error:

            self.fail_test(
                name=name,
                details=str(error),
            )



    def test_runtime_t6_grounding_validator_provider_failure(
        self,
    ) -> None:

        name = (
            "RUNTIME-T6 Grounding validator "
            "provider failure boundary"
        )

        try:

            suite_16_21 = RegressionSuite16_21()

            tutor = (
                suite_16_21
                .build_grounded_tutor()
            )

            learner_message = (
                "ทรานซิสเตอร์ NPN "
                "ประกอบด้วยขั้วอะไรบ้าง"
            )

            generated_answer = (
                "ทรานซิสเตอร์ NPN มีขั้ว "
                "Base, Collector และ Emitter"
            )

            knowledge_context = (
                "The transistor has three terminals: "
                "Base, Collector, and Emitter."
            )

            knowledge_result = SimpleNamespace(
                query="controlled NPN terminal query",
                context=knowledge_context,
                sources=[],
                citations=[],
            )

            tutor.knowledge_service.retrieve = (
                lambda query, n_results=5:
                knowledge_result
            )

            tutor.pedagogical_response_validator.validate = (
                lambda *args, **kwargs:
                SimpleNamespace(
                    status="valid",
                    confidence=1.0,
                    reason="Controlled pedagogy.",
                    issues=[],
                )
            )

            tutor.response_mode_pedagogical_validator.validate = (
                lambda *args, **kwargs:
                SimpleNamespace(
                    status="valid",
                    confidence=1.0,
                    reason="Controlled mode pedagogy.",
                    issues=[],
                )
            )

            evidence_calls = []

            def controlled_no_evidence(
                learner_message: str,
                knowledge_context: str,
                validation_issues=None,
                **kwargs,
            ) -> RepairEvidenceSelectionResult:

                evidence_calls.append(
                    {
                        "learner_message": learner_message,
                        "knowledge_context": knowledge_context,
                        "validation_issues": validation_issues,
                    }
                )

                return RepairEvidenceSelectionResult(
                    status="invalid",
                    evidence_quotes=(),
                    reason=(
                        "Controlled no verified "
                        "repair evidence."
                    ),
                    issues=(
                        "Controlled evidence block.",
                    ),
                )

            tutor.repair_evidence_selector.select = (
                controlled_no_evidence
            )

            repair_calls = []

            def forbidden_repair(
                *args,
                **kwargs,
            ):

                repair_calls.append(
                    kwargs
                )

                raise AssertionError(
                    "Repair generation must not run "
                    "without verified evidence."
                )

            tutor.response_mode_repair_service.repair = (
                forbidden_repair
            )

            memory_before = list(
                tutor.memory.get_messages()
            )

            turn_before = (
                tutor.state.turn_count
            )

            scaffolding_before = (
                tutor.state.scaffolding_level
            )

            attempt_before = (
                tutor.state.attempt_count
            )

            propagated_error = None
            returned_answer = None

            try:

                with patch(
                    "app.tutor.chat_with_ai",
                    return_value=generated_answer,
                ):

                    with patch(
                        (
                            "app.services."
                            "response_grounding_validator."
                            "chat_with_structured_ai"
                        ),
                        side_effect=(
                            ControlledProviderFailure(
                                "Controlled grounding "
                                "validator provider failure"
                            )
                        ),
                    ) as grounding_provider:

                        returned_answer = (
                            tutor.respond(
                                learner_message
                            )
                        )

            except Exception as error:

                propagated_error = error

            grounding = (
                tutor.last_grounding_validation
            )

            self.assert_equal(
                propagated_error,
                None,
                (
                    "RUNTIME-T6 grounding provider "
                    "failure must be contained"
                ),
            )

            self.assert_equal(
                grounding_provider.call_count,
                1,
                (
                    "RUNTIME-T6 grounding provider "
                    "must be called once"
                ),
            )

            self.assert_true(
                grounding is not None,
                (
                    "RUNTIME-T6 must retain a "
                    "fail-closed grounding result"
                ),
            )

            self.assert_equal(
                grounding.status,
                "unsupported",
                (
                    "RUNTIME-T6 provider failure must "
                    "fail closed as unsupported"
                ),
            )

            self.assert_true(
                any(
                    "ControlledProviderFailure"
                    in str(issue)
                    for issue in grounding.issues
                ),
                (
                    "RUNTIME-T6 grounding issues must "
                    "retain provider failure provenance"
                ),
            )

            self.assert_true(
                returned_answer is not None,
                (
                    "RUNTIME-T6 must return a safe "
                    "grounding fallback"
                ),
            )

            self.assert_true(
                returned_answer != generated_answer,
                (
                    "RUNTIME-T6 generated answer must "
                    "not reach the learner after "
                    "validator failure"
                ),
            )

            self.assert_equal(
                len(repair_calls),
                0,
                (
                    "RUNTIME-T6 repair generation must "
                    "remain blocked without evidence"
                ),
            )

            self.assert_equal(
                tutor.memory.get_messages(),
                memory_before,
                (
                    "RUNTIME-T6 failed factual turn "
                    "must not enter ConversationMemory"
                ),
            )

            self.assert_equal(
                tutor.original_question,
                None,
                (
                    "RUNTIME-T6 failed sequence "
                    "must be closed"
                ),
            )

            self.assert_true(
                tutor.waiting_for_response is False,
                (
                    "RUNTIME-T6 must not leave an "
                    "undelivered sequence active"
                ),
            )

            self.assert_equal(
                tutor.state.turn_count,
                turn_before + 1,
                (
                    "RUNTIME-T6 attempted turn remains "
                    "observable"
                ),
            )

            self.assert_equal(
                tutor.state.scaffolding_level,
                scaffolding_before,
                (
                    "RUNTIME-T6 must preserve "
                    "scaffolding state"
                ),
            )

            self.assert_equal(
                tutor.state.attempt_count,
                attempt_before,
                (
                    "RUNTIME-T6 must preserve "
                    "attempt state"
                ),
            )

            debug = tutor.get_debug_info()

            self.assert_equal(
                debug.get(
                    "validation_status"
                ),
                "unsupported",
                (
                    "RUNTIME-T6 debug grounding status "
                    "must expose fail-closed result"
                ),
            )

            self.assert_true(
                debug.get(
                    "repair_failed"
                ),
                (
                    "RUNTIME-T6 downstream repair "
                    "failure must be observable"
                ),
            )

            self.assert_equal(
                debug.get(
                    "repair_failure_type"
                ),
                "repair_evidence",
                (
                    "RUNTIME-T6 downstream block must "
                    "remain repair_evidence"
                ),
            )

            self.pass_test(
                name=name,
                details=(
                    "Grounding-validator structured-provider "
                    "failure is contained and converted to "
                    "unsupported; unsafe output is blocked, "
                    "repair cannot proceed without verified "
                    "evidence, and runtime state remains clean."
                ),
            )

        except Exception as error:

            self.fail_test(
                name=name,
                details=str(error),
            )       

    def test_runtime_t7_guiding_validator_provider_failure(
        self,
    ) -> None:

        name = (
            "RUNTIME-T7 Guiding-question validator "
            "provider failure boundary"
        )

        try:

            suite_16_21 = RegressionSuite16_21()

            tutor = (
                suite_16_21
                .build_grounded_tutor()
            )

            learner_message = (
                "ทรานซิสเตอร์ NPN มีขั้วอะไรบ้าง"
            )

            initial_question = (
                "คุณคิดว่าทรานซิสเตอร์ NPN "
                "มีขั้วอะไรบ้าง?"
            )

            safe_repair = (
                "คุณคิดว่า Base, Collector และ Emitter "
                "เป็นขั้วของทรานซิสเตอร์หรือไม่?"
            )

            knowledge_context = (
                "The transistor has three terminals: "
                "Base, Collector, and Emitter."
            )

            knowledge_result = SimpleNamespace(
                query="controlled transistor query",
                context=knowledge_context,
                sources=[],
                citations=[],
            )

            tutor.knowledge_service.retrieve = (
                lambda query, n_results=5:
                knowledge_result
            )

            def controlled_grounding_validation(
                response: str,
                knowledge_context: str,
                task_name: str = "response_validator",
            ) -> GroundingValidationResult:

                return GroundingValidationResult(
                    status="supported",
                    confidence=1.0,
                    reason=(
                        "Controlled factual grounding."
                    ),
                    issues=[],
                )

            tutor.response_grounding_validator.validate = (
                controlled_grounding_validation
            )

            tutor.pedagogical_response_validator.validate = (
                lambda *args, **kwargs:
                SimpleNamespace(
                    status="valid",
                    confidence=1.0,
                    reason="Controlled pedagogy.",
                    issues=[],
                )
            )

            tutor.response_mode_pedagogical_validator.validate = (
                lambda *args, **kwargs:
                SimpleNamespace(
                    status="valid",
                    confidence=1.0,
                    reason="Controlled mode pedagogy.",
                    issues=[],
                )
            )

            real_validate_guiding_questions = (
                tutor._validate_guiding_questions
            )

            guiding_helper_calls = []

            def controlled_validate_guiding_questions(
                response: str,
                knowledge_context: str,
            ) -> GuidingQuestionGroundingResult:

                guiding_helper_calls.append(
                    response
                )

                if len(
                    guiding_helper_calls
                ) == 1:

                    return (
                        real_validate_guiding_questions(
                            response=response,
                            knowledge_context=(
                                knowledge_context
                            ),
                        )
                    )

                if response == safe_repair:

                    return GuidingQuestionGroundingResult(
                        status="supported",
                        reason=(
                            "Controlled repaired guiding "
                            "question is supported."
                        ),
                        issues=(),
                    )

                raise AssertionError(
                    "Unexpected later guiding validation: "
                    f"{response!r}"
                )

            tutor._validate_guiding_questions = (
                controlled_validate_guiding_questions
            )

            repair_calls = []

            def controlled_repair(
                *args,
                **kwargs,
            ):

                repair_calls.append(
                    kwargs
                )

                if len(repair_calls) == 1:
                    return safe_repair

                raise AssertionError(
                    "RUNTIME-T7 must not exceed one "
                    "normal repair generation."
                )

            tutor.response_mode_repair_service.repair = (
                controlled_repair
            )

            memory_before = list(
                tutor.memory.get_messages()
            )

            turn_before = (
                tutor.state.turn_count
            )

            scaffolding_before = (
                tutor.state.scaffolding_level
            )

            attempt_before = (
                tutor.state.attempt_count
            )

            propagated_error = None
            returned_answer = None

            try:

                with patch(
                    "app.tutor.chat_with_ai",
                    return_value=initial_question,
                ):

                    with patch(
                        (
                            "app.services."
                            "guiding_question_grounding_validator."
                            "chat_with_structured_ai"
                        ),
                        side_effect=(
                            ControlledProviderFailure(
                                "Controlled guiding-question "
                                "validator provider failure"
                            )
                        ),
                    ) as guiding_provider:

                        returned_answer = (
                            tutor.respond(
                                learner_message
                            )
                        )

            except Exception as error:

                propagated_error = error

            initial_guiding = (
                tutor.last_guiding_question_grounding
            )

            repair_guiding = (
                tutor.last_repair_guiding_question_grounding
            )

            self.assert_equal(
                propagated_error,
                None,
                (
                    "RUNTIME-T7 guiding validator "
                    "failure must be contained"
                ),
            )

            self.assert_equal(
                guiding_provider.call_count,
                1,
                (
                    "RUNTIME-T7 provider failure "
                    "surface must be called once"
                ),
            )

            self.assert_true(
                initial_guiding is not None,
                (
                    "RUNTIME-T7 must retain initial "
                    "guiding validation telemetry"
                ),
            )

            self.assert_equal(
                initial_guiding.status,
                "invalid",
                (
                    "RUNTIME-T7 provider failure must "
                    "fail closed as invalid"
                ),
            )

            self.assert_true(
                any(
                    "ControlledProviderFailure"
                    in str(issue)
                    for issue in (
                        initial_guiding.issues
                    )
                ),
                (
                    "RUNTIME-T7 initial guiding issues "
                    "must retain provider provenance"
                ),
            )

            self.assert_true(
                returned_answer != initial_question,
                (
                    "RUNTIME-T7 invalid initial guiding "
                    "question must not reach learner"
                ),
            )

            self.assert_equal(
                returned_answer,
                safe_repair,
                (
                    "RUNTIME-T7 accepted safe repair "
                    "must become final answer"
                ),
            )

            self.assert_equal(
                len(repair_calls),
                1,
                (
                    "RUNTIME-T7 must perform exactly "
                    "one normal repair"
                ),
            )

            self.assert_true(
                repair_guiding is not None,
                (
                    "RUNTIME-T7 repaired guiding "
                    "question must be revalidated"
                ),
            )

            self.assert_equal(
                repair_guiding.status,
                "supported",
                (
                    "RUNTIME-T7 accepted repair must "
                    "pass guiding-question grounding"
                ),
            )

            debug = tutor.get_debug_info()

            self.assert_true(
                debug.get(
                    "response_repaired"
                ),
                (
                    "RUNTIME-T7 accepted repair must "
                    "be recorded"
                ),
            )

            self.assert_true(
                debug.get(
                    "repair_failed"
                )
                is False,
                (
                    "RUNTIME-T7 successful repair must "
                    "not remain marked failed"
                ),
            )

            memory_after = (
                tutor.memory.get_messages()
            )

            memory_text = " ".join(
                str(
                    item.get(
                        "content",
                        "",
                    )
                )
                for item in memory_after
                if isinstance(
                    item,
                    dict,
                )
            )

            self.assert_true(
                initial_question
                not in memory_text,
                (
                    "RUNTIME-T7 rejected initial "
                    "question must not enter memory"
                ),
            )

            self.assert_true(
                safe_repair
                in memory_text,
                (
                    "RUNTIME-T7 accepted repair must "
                    "be stored in memory"
                ),
            )

            self.assert_true(
                memory_after
                != memory_before,
                (
                    "RUNTIME-T7 successful Tutor turn "
                    "must update ConversationMemory"
                ),
            )

            self.assert_equal(
                tutor.original_question,
                learner_message,
                (
                    "RUNTIME-T7 successful repaired "
                    "question must keep active learner "
                    "sequence"
                ),
            )

            self.assert_true(
                tutor.waiting_for_response
                is True,
                (
                    "RUNTIME-T7 Tutor must wait for "
                    "learner response to delivered "
                    "safe repaired question"
                ),
            )

            self.assert_equal(
                tutor.state.turn_count,
                turn_before + 1,
                (
                    "RUNTIME-T7 turn count must "
                    "advance once"
                ),
            )

            self.assert_equal(
                tutor.state.scaffolding_level,
                scaffolding_before,
                (
                    "RUNTIME-T7 must preserve "
                    "scaffolding state"
                ),
            )

            self.assert_equal(
                tutor.state.attempt_count,
                attempt_before,
                (
                    "RUNTIME-T7 must preserve "
                    "attempt state"
                ),
            )

            self.pass_test(
                name=name,
                details=(
                    "Guiding-question validator provider "
                    "failure is contained and converted to "
                    "invalid; the rejected question never "
                    "reaches the learner, one safe repair "
                    "is accepted and stored, and the valid "
                    "learning sequence remains active."
                ),
            )

        except Exception as error:

            self.fail_test(
                name=name,
                details=str(error),
            )

    def test_runtime_t8_legacy_pedagogical_provider_failure(
        self,
    ) -> None:

        name = (
            "RUNTIME-T8 Legacy pedagogical "
            "provider failure boundary"
        )

        try:

            suite_16_21 = RegressionSuite16_21()

            tutor = (
                suite_16_21
                .build_grounded_tutor()
            )

            learner_message = (
                "ทรานซิสเตอร์ NPN มีขั้วอะไรบ้าง"
            )

            rejected_initial = (
                "ลองคิดดูอีกครั้ง"
            )

            safe_repair = (
                "คุณคิดว่าขั้วทั้งสามของ "
                "ทรานซิสเตอร์มีชื่อว่าอะไร?"
            )

            knowledge_context = (
                "The transistor has three terminals: "
                "Base, Collector, and Emitter. "
                "The Base-Emitter junction may be "
                "forward biased or reverse biased."
            )

            knowledge_result = SimpleNamespace(
                query="controlled T8 query",
                context=knowledge_context,
                sources=[],
                citations=[],
            )

            tutor.knowledge_service.retrieve = (
                lambda query, n_results=5:
                knowledge_result
            )

            tutor.response_grounding_validator.validate = (
                lambda *args, **kwargs:
                GroundingValidationResult(
                    status="supported",
                    confidence=1.0,
                    reason=(
                        "Controlled factual grounding."
                    ),
                    issues=[],
                )
            )

            tutor._validate_guiding_questions = (
                lambda response, knowledge_context:
                GuidingQuestionGroundingResult(
                    status="supported",
                    reason=(
                        "Controlled guiding-question "
                        "grounding."
                    ),
                    issues=(),
                )
            )

            tutor.response_mode_pedagogical_precheck.evaluate = (
                lambda *args, **kwargs:
                SimpleNamespace(
                    status="uncertain",
                    reason=(
                        "Controlled semantic-validation "
                        "path."
                    ),
                    issues=[],
                )
            )

            def forbidden_evidence_selection(
                *args,
                **kwargs,
            ):
                raise AssertionError(
                    "RUNTIME-T8 pedagogical-only "
                    "failure must not activate "
                    "repair evidence selection."
                )

            tutor.repair_evidence_selector.select = (
                forbidden_evidence_selection
            )

            repair_calls = []

            def controlled_repair(
                *args,
                **kwargs,
            ):

                repair_calls.append(
                    kwargs
                )

                if len(repair_calls) == 1:
                    return safe_repair

                raise AssertionError(
                    "RUNTIME-T8 must perform exactly "
                    "one repair generation."
                )

            tutor.response_mode_repair_service.repair = (
                controlled_repair
            )

            memory_before = list(
                tutor.memory.get_messages()
            )

            turn_before = (
                tutor.state.turn_count
            )

            scaffolding_before = (
                tutor.state.scaffolding_level
            )

            attempt_before = (
                tutor.state.attempt_count
            )

            correct_streak_before = (
                tutor.state.correct_streak
            )

            partial_streak_before = (
                tutor.state.partial_streak
            )

            failure_streak_before = (
                tutor.state.failure_streak
            )

            misconception_count_before = len(
                tutor.state.misconceptions
            )

            valid_repair_validation = (
                '{"status":"valid",'
                '"confidence":1.0,'
                '"reason":"Controlled repaired response '
                'is pedagogically valid.",'
                '"issues":[]}'
            )

            propagated_error = None
            returned_answer = None

            try:

                with patch(
                    "app.tutor.chat_with_ai",
                    return_value=rejected_initial,
                ):

                    with patch(
                        (
                            "app.services."
                            "pedagogical_response_validator."
                            "chat_with_structured_ai"
                        ),
                        side_effect=[
                            ControlledProviderFailure(
                                "Controlled legacy "
                                "pedagogical provider failure"
                            ),
                            valid_repair_validation,
                        ],
                    ) as pedagogical_provider:

                        returned_answer = (
                            tutor.respond(
                                learner_message
                            )
                        )

            except Exception as error:

                propagated_error = error

            initial_pedagogy = (
                tutor.last_pedagogical_validation
            )

            repair_pedagogy = (
                tutor.last_repair_pedagogical_validation
            )

            response_mode = (
                tutor.last_tutoring_response_mode
            )

            self.assert_equal(
                propagated_error,
                None,
                (
                    "RUNTIME-T8 pedagogical provider "
                    "failure must be contained"
                ),
            )

            self.assert_equal(
                pedagogical_provider.call_count,
                2,
                (
                    "RUNTIME-T8 must call the provider "
                    "once for rejected initial validation "
                    "and once for repair revalidation"
                ),
            )

            self.assert_true(
                response_mode is not None,
                (
                    "RUNTIME-T8 response mode must "
                    "remain observable"
                ),
            )

            self.assert_true(
                response_mode.preserve_scaffolding_strategy,
                (
                    "RUNTIME-T8 must exercise the "
                    "preserving legacy route"
                ),
            )

            self.assert_true(
                initial_pedagogy is not None,
                (
                    "RUNTIME-T8 initial pedagogical "
                    "result must be retained"
                ),
            )

            self.assert_equal(
                initial_pedagogy.status,
                "violation",
                (
                    "RUNTIME-T8 provider failure must "
                    "fail closed as violation"
                ),
            )

            self.assert_true(
                any(
                    "ControlledProviderFailure"
                    in str(issue)
                    for issue
                    in initial_pedagogy.issues
                ),
                (
                    "RUNTIME-T8 validation telemetry "
                    "must retain provider provenance"
                ),
            )

            self.assert_equal(
                len(repair_calls),
                1,
                (
                    "RUNTIME-T8 must perform exactly "
                    "one repair generation"
                ),
            )

            self.assert_equal(
                repair_calls[0].get(
                    "knowledge_context"
                ),
                knowledge_context,
                (
                    "RUNTIME-T8 pedagogical-only repair "
                    "must retain the full course context"
                ),
            )

            self.assert_true(
                repair_pedagogy is not None,
                (
                    "RUNTIME-T8 repaired response "
                    "must be pedagogically revalidated"
                ),
            )

            self.assert_equal(
                repair_pedagogy.status,
                "valid",
                (
                    "RUNTIME-T8 accepted repair must "
                    "pass pedagogical revalidation"
                ),
            )

            self.assert_equal(
                returned_answer,
                safe_repair,
                (
                    "RUNTIME-T8 final learner response "
                    "must be the accepted safe repair"
                ),
            )

            memory_after = (
                tutor.memory.get_messages()
            )

            memory_text = " ".join(
                str(
                    item.get(
                        "content",
                        "",
                    )
                )
                for item in memory_after
                if isinstance(
                    item,
                    dict,
                )
            )

            self.assert_true(
                rejected_initial
                not in memory_text,
                (
                    "RUNTIME-T8 rejected initial "
                    "response must not enter memory"
                ),
            )

            self.assert_true(
                safe_repair
                in memory_text,
                (
                    "RUNTIME-T8 accepted repair "
                    "must enter memory"
                ),
            )

            self.assert_true(
                memory_after != memory_before,
                (
                    "RUNTIME-T8 successful repaired "
                    "turn must update memory"
                ),
            )

            self.assert_equal(
                tutor.original_question,
                learner_message,
                (
                    "RUNTIME-T8 accepted repaired "
                    "Tutor question must retain the "
                    "active learning sequence"
                ),
            )

            self.assert_true(
                tutor.waiting_for_response is True,
                (
                    "RUNTIME-T8 Tutor must wait for "
                    "learner response after delivering "
                    "the repaired question"
                ),
            )

            self.assert_equal(
                tutor.state.turn_count,
                turn_before + 1,
                (
                    "RUNTIME-T8 turn count must "
                    "advance once"
                ),
            )

            self.assert_equal(
                tutor.state.scaffolding_level,
                scaffolding_before,
                (
                    "RUNTIME-T8 must preserve "
                    "scaffolding level"
                ),
            )

            self.assert_equal(
                tutor.state.attempt_count,
                attempt_before,
                (
                    "RUNTIME-T8 must preserve "
                    "attempt count"
                ),
            )

            self.assert_equal(
                tutor.state.correct_streak,
                correct_streak_before,
                (
                    "RUNTIME-T8 must preserve "
                    "correct streak"
                ),
            )

            self.assert_equal(
                tutor.state.partial_streak,
                partial_streak_before,
                (
                    "RUNTIME-T8 must preserve "
                    "partial streak"
                ),
            )

            self.assert_equal(
                tutor.state.failure_streak,
                failure_streak_before,
                (
                    "RUNTIME-T8 must preserve "
                    "failure streak"
                ),
            )

            self.assert_equal(
                len(tutor.state.misconceptions),
                misconception_count_before,
                (
                    "RUNTIME-T8 must not create "
                    "misconceptions"
                ),
            )

            debug = tutor.get_debug_info()

            self.assert_equal(
                debug.get(
                    "escalation_action"
                ),
                "llm_repair",
                (
                    "RUNTIME-T8 pedagogical violation "
                    "must use existing LLM repair policy"
                ),
            )

            self.assert_true(
                debug.get(
                    "response_repaired"
                ),
                (
                    "RUNTIME-T8 accepted repair "
                    "must be recorded"
                ),
            )

            self.assert_true(
                debug.get(
                    "repair_failed"
                )
                is False,
                (
                    "RUNTIME-T8 successful repair "
                    "must not remain failed"
                ),
            )

            self.pass_test(
                name=name,
                details=(
                    "Legacy pedagogical structured-provider "
                    "failure fails closed as a violation, "
                    "enters the existing pedagogical-only "
                    "repair path without evidence selection, "
                    "and stores only the accepted repair."
                ),
            )

        except Exception as error:

            self.fail_test(
                name=name,
                details=str(error),
            )

    def test_runtime_t9_response_mode_pedagogical_provider_failure(
        self,
    ) -> None:

        name = (
            "RUNTIME-T9 Response-mode pedagogical "
            "provider failure boundary"
        )

        try:

            suite_16_21 = RegressionSuite16_21()

            tutor = (
                suite_16_21
                .build_grounded_tutor()
            )

            original_question = (
                "แรงดันระหว่าง Base และ Emitter "
                "มีผลอย่างไร"
            )

            learner_message = (
                "คำว่าไบอัสหมายถึงอะไร"
            )

            rejected_initial = (
                "ไบอัสคือการกำหนดสภาวะ "
                "การทำงานของวงจร"
            )

            safe_repair = (
                "ในบริบทนี้ ไบอัสหมายถึง "
                "การกำหนดสภาวะของรอยต่อ "
                "Base-Emitter"
            )

            knowledge_context = (
                "The transistor has three terminals: "
                "Base, Collector, and Emitter. "
                "The Base-Emitter junction may be "
                "forward biased or reverse biased."
            )

            knowledge_result = SimpleNamespace(
                query="controlled T9 query",
                context=knowledge_context,
                sources=[],
                citations=[],
            )

            tutor.knowledge_service.retrieve = (
                lambda query, n_results=5:
                knowledge_result
            )

            tutor.original_question = (
                original_question
            )

            tutor.waiting_for_response = True

            tutor.response_grounding_validator.validate = (
                lambda *args, **kwargs:
                GroundingValidationResult(
                    status="supported",
                    confidence=1.0,
                    reason=(
                        "Controlled factual grounding."
                    ),
                    issues=[],
                )
            )

            tutor._validate_guiding_questions = (
                lambda response, knowledge_context:
                GuidingQuestionGroundingResult(
                    status="supported",
                    reason=(
                        "Controlled guiding-question "
                        "grounding."
                    ),
                    issues=(),
                )
            )

            tutor.response_mode_pedagogical_precheck.evaluate = (
                lambda *args, **kwargs:
                SimpleNamespace(
                    status="uncertain",
                    reason=(
                        "Controlled semantic-validation "
                        "path."
                    ),
                    issues=[],
                )
            )

            def forbidden_evidence_selection(
                *args,
                **kwargs,
            ):
                raise AssertionError(
                    "RUNTIME-T9 pedagogical-only "
                    "failure must not activate "
                    "repair evidence selection."
                )

            tutor.repair_evidence_selector.select = (
                forbidden_evidence_selection
            )

            repair_calls = []

            def controlled_repair(
                *args,
                **kwargs,
            ):

                repair_calls.append(
                    kwargs
                )

                if len(repair_calls) == 1:
                    return safe_repair

                raise AssertionError(
                    "RUNTIME-T9 must perform exactly "
                    "one repair generation."
                )

            tutor.response_mode_repair_service.repair = (
                controlled_repair
            )

            memory_before = list(
                tutor.memory.get_messages()
            )

            turn_before = (
                tutor.state.turn_count
            )

            scaffolding_before = (
                tutor.state.scaffolding_level
            )

            attempt_before = (
                tutor.state.attempt_count
            )

            correct_streak_before = (
                tutor.state.correct_streak
            )

            partial_streak_before = (
                tutor.state.partial_streak
            )

            failure_streak_before = (
                tutor.state.failure_streak
            )

            misconception_count_before = len(
                tutor.state.misconceptions
            )

            valid_repair_validation = (
                '{"status":"valid",'
                '"confidence":1.0,'
                '"reason":"Controlled repaired response '
                'is pedagogically valid.",'
                '"issues":[]}'
            )

            propagated_error = None
            returned_answer = None

            try:

                with patch(
                    "app.tutor.chat_with_ai",
                    return_value=rejected_initial,
                ):

                    with patch(
                        (
                            "app.services."
                            "response_mode_pedagogical_validator."
                            "chat_with_structured_ai"
                        ),
                        side_effect=[
                            ControlledProviderFailure(
                                "Controlled response-mode "
                                "pedagogical provider failure"
                            ),
                            valid_repair_validation,
                        ],
                    ) as pedagogical_provider:

                        returned_answer = (
                            tutor.respond(
                                learner_message
                            )
                        )

            except Exception as error:

                propagated_error = error

            initial_pedagogy = (
                tutor.last_pedagogical_validation
            )

            repair_pedagogy = (
                tutor.last_repair_pedagogical_validation
            )

            response_mode = (
                tutor.last_tutoring_response_mode
            )

            self.assert_equal(
                propagated_error,
                None,
                (
                    "RUNTIME-T9 response-mode "
                    "pedagogical provider failure "
                    "must be contained"
                ),
            )

            self.assert_equal(
                pedagogical_provider.call_count,
                2,
                (
                    "RUNTIME-T9 must call the provider "
                    "once for initial validation and "
                    "once for repair revalidation"
                ),
            )

            self.assert_true(
                tutor.last_learner_turn_intent
                is not None,
                (
                    "RUNTIME-T9 learner intent "
                    "must remain observable"
                ),
            )

            self.assert_equal(
                tutor.last_learner_turn_intent.intent,
                "clarification_question",
                (
                    "RUNTIME-T9 must exercise the "
                    "clarification route"
                ),
            )

            self.assert_true(
                response_mode is not None,
                (
                    "RUNTIME-T9 response mode "
                    "must remain observable"
                ),
            )

            self.assert_equal(
                response_mode.mode,
                "direct_clarification",
                (
                    "RUNTIME-T9 clarification must "
                    "use direct_clarification mode"
                ),
            )

            self.assert_true(
                response_mode.preserve_scaffolding_strategy
                is False,
                (
                    "RUNTIME-T9 must exercise the "
                    "non-preserving semantic route"
                ),
            )

            self.assert_true(
                initial_pedagogy is not None,
                (
                    "RUNTIME-T9 initial pedagogical "
                    "result must be retained"
                ),
            )

            self.assert_equal(
                initial_pedagogy.status,
                "violation",
                (
                    "RUNTIME-T9 provider failure must "
                    "fail closed as violation"
                ),
            )

            self.assert_true(
                any(
                    "ControlledProviderFailure"
                    in str(issue)
                    for issue
                    in initial_pedagogy.issues
                ),
                (
                    "RUNTIME-T9 validation telemetry "
                    "must retain provider provenance"
                ),
            )

            self.assert_equal(
                len(repair_calls),
                1,
                (
                    "RUNTIME-T9 must perform exactly "
                    "one repair generation"
                ),
            )

            self.assert_equal(
                repair_calls[0].get(
                    "knowledge_context"
                ),
                knowledge_context,
                (
                    "RUNTIME-T9 pedagogical-only repair "
                    "must retain full course context"
                ),
            )

            self.assert_true(
                repair_pedagogy is not None,
                (
                    "RUNTIME-T9 repaired response "
                    "must be pedagogically revalidated"
                ),
            )

            self.assert_equal(
                repair_pedagogy.status,
                "valid",
                (
                    "RUNTIME-T9 accepted repair must "
                    "pass response-mode pedagogical "
                    "revalidation"
                ),
            )

            self.assert_equal(
                returned_answer,
                safe_repair,
                (
                    "RUNTIME-T9 final learner response "
                    "must be accepted safe repair"
                ),
            )

            memory_after = (
                tutor.memory.get_messages()
            )

            memory_text = " ".join(
                str(
                    item.get(
                        "content",
                        "",
                    )
                )
                for item in memory_after
                if isinstance(
                    item,
                    dict,
                )
            )

            self.assert_true(
                rejected_initial
                not in memory_text,
                (
                    "RUNTIME-T9 rejected initial "
                    "response must not enter memory"
                ),
            )

            self.assert_true(
                safe_repair
                in memory_text,
                (
                    "RUNTIME-T9 accepted repair "
                    "must enter memory"
                ),
            )

            self.assert_true(
                memory_after != memory_before,
                (
                    "RUNTIME-T9 successful repaired "
                    "turn must update memory"
                ),
            )

            self.assert_equal(
                tutor.original_question,
                original_question,
                (
                    "RUNTIME-T9 clarification must "
                    "preserve the original active "
                    "learning question"
                ),
            )

            self.assert_true(
                tutor.waiting_for_response is True,
                (
                    "RUNTIME-T9 clarification must "
                    "preserve the active sequence"
                ),
            )

            self.assert_equal(
                tutor.state.turn_count,
                turn_before + 1,
                (
                    "RUNTIME-T9 turn count must "
                    "advance once"
                ),
            )

            self.assert_equal(
                tutor.state.scaffolding_level,
                scaffolding_before,
                (
                    "RUNTIME-T9 must preserve "
                    "scaffolding level"
                ),
            )

            self.assert_equal(
                tutor.state.attempt_count,
                attempt_before,
                (
                    "RUNTIME-T9 must preserve "
                    "attempt count"
                ),
            )

            self.assert_equal(
                tutor.state.correct_streak,
                correct_streak_before,
                (
                    "RUNTIME-T9 must preserve "
                    "correct streak"
                ),
            )

            self.assert_equal(
                tutor.state.partial_streak,
                partial_streak_before,
                (
                    "RUNTIME-T9 must preserve "
                    "partial streak"
                ),
            )

            self.assert_equal(
                tutor.state.failure_streak,
                failure_streak_before,
                (
                    "RUNTIME-T9 must preserve "
                    "failure streak"
                ),
            )

            self.assert_equal(
                len(tutor.state.misconceptions),
                misconception_count_before,
                (
                    "RUNTIME-T9 must not create "
                    "misconceptions"
                ),
            )

            debug = tutor.get_debug_info()

            self.assert_equal(
                debug.get(
                    "escalation_action"
                ),
                "llm_repair",
                (
                    "RUNTIME-T9 pedagogical violation "
                    "must use existing LLM repair policy"
                ),
            )

            self.assert_true(
                debug.get(
                    "response_repaired"
                ),
                (
                    "RUNTIME-T9 accepted repair "
                    "must be recorded"
                ),
            )

            self.assert_true(
                debug.get(
                    "repair_failed"
                )
                is False,
                (
                    "RUNTIME-T9 successful repair "
                    "must not remain failed"
                ),
            )

            self.pass_test(
                name=name,
                details=(
                    "Non-preserving response-mode "
                    "pedagogical provider failure fails "
                    "closed as a violation, uses exactly "
                    "one normal repair with full course "
                    "context, preserves the clarification "
                    "sequence, and stores only the "
                    "accepted repair."
                ),
            )

        except Exception as error:

            self.fail_test(
                name=name,
                details=str(error),
            )


    def test_runtime_t10_relevance_provider_failure_boundary(
        self,
    ) -> None:

        name = (
            "RUNTIME-T10 Relevance provider "
            "failure boundary"
        )

        try:

            class ControlledRelevanceProviderFailure(
                RuntimeError
            ):
                pass

            tutor = AITutor()

            learner_message = (
                "ทรานซิสเตอร์ NPN มีขั้วอะไรบ้าง"
            )

            knowledge_result = SimpleNamespace(
                query="controlled T10 relevance query",
                context=(
                    "The transistor has three terminals: "
                    "Base, Collector, and Emitter."
                ),
                sources=[],
                citations=[],
                has_context=True,
            )

            tutor.knowledge_service.retrieve = (
                lambda query, n_results=5:
                knowledge_result
            )

            memory_before = list(
                tutor.memory.get_messages()
            )

            turn_before = (
                tutor.state.turn_count
            )

            scaffolding_before = (
                tutor.state.scaffolding_level
            )

            attempt_before = (
                tutor.state.attempt_count
            )

            propagated_error = None
            returned_answer = None

            try:

                with patch(
                    (
                        "app.services.relevance_gate."
                        "chat_with_structured_ai"
                    ),
                    side_effect=(
                        ControlledRelevanceProviderFailure(
                            "Controlled relevance "
                            "provider failure"
                        )
                    ),
                ) as relevance_provider:

                    with patch(
                        "app.tutor.chat_with_ai",
                        return_value=(
                            "THIS RESPONSE MUST "
                            "NOT BE GENERATED"
                        ),
                    ) as tutor_generation:

                        returned_answer = (
                            tutor.respond(
                                learner_message
                            )
                        )

            except Exception as error:

                propagated_error = error

            memory_after = (
                tutor.memory.get_messages()
            )

            self.assert_equal(
                relevance_provider.call_count,
                1,
                (
                    "RUNTIME-T10 relevance provider "
                    "must be called exactly once"
                ),
            )

            self.assert_true(
                isinstance(
                    propagated_error,
                    ControlledRelevanceProviderFailure,
                ),
                (
                    "RUNTIME-T10 original relevance "
                    "provider failure must propagate"
                ),
            )

            self.assert_equal(
                returned_answer,
                None,
                (
                    "RUNTIME-T10 must not fabricate "
                    "a learner-facing response"
                ),
            )

            self.assert_equal(
                tutor.last_relevance_result,
                None,
                (
                    "RUNTIME-T10 must not fabricate "
                    "a relevant or irrelevant decision"
                ),
            )

            self.assert_equal(
                tutor.last_grounding_status,
                None,
                (
                    "RUNTIME-T10 grounding must not "
                    "run after unresolved relevance"
                ),
            )

            self.assert_equal(
                tutor_generation.call_count,
                0,
                (
                    "RUNTIME-T10 Tutor generation "
                    "must remain blocked"
                ),
            )

            self.assert_equal(
                tutor.last_generated_answer,
                None,
                (
                    "RUNTIME-T10 must not create "
                    "a generated Tutor answer"
                ),
            )

            self.assert_equal(
                memory_after,
                memory_before,
                (
                    "RUNTIME-T10 failed relevance turn "
                    "must not enter ConversationMemory"
                ),
            )

            self.assert_equal(
                tutor.original_question,
                None,
                (
                    "RUNTIME-T10 incomplete active "
                    "question must be cleared"
                ),
            )

            self.assert_true(
                tutor.waiting_for_response is False,
                (
                    "RUNTIME-T10 must not leave an "
                    "undelivered learning sequence active"
                ),
            )

            self.assert_equal(
                tutor.state.turn_count,
                turn_before + 1,
                (
                    "RUNTIME-T10 attempted runtime "
                    "turn must remain observable"
                ),
            )

            self.assert_equal(
                tutor.state.scaffolding_level,
                scaffolding_before,
                (
                    "RUNTIME-T10 must not alter "
                    "scaffolding level"
                ),
            )

            self.assert_equal(
                tutor.state.attempt_count,
                attempt_before,
                (
                    "RUNTIME-T10 must not create "
                    "a learner attempt"
                ),
            )

            self.pass_test(
                name=name,
                details=(
                    "Relevance structured-provider failure "
                    "propagates without fabricating a "
                    "relevance decision, grounding result, "
                    "out-of-course response, or Tutor "
                    "generation; ConversationMemory remains "
                    "clean and the incomplete learning "
                    "sequence is closed."
                ),
            )

        except Exception as error:

            self.fail_test(
                name=name,
                details=str(error),
            )


    def test_runtime_t11_knowledge_retrieval_failure_boundary(
        self,
    ) -> None:

        name = (
            "RUNTIME-T11 Knowledge retrieval "
            "failure boundary"
        )

        try:

            class ControlledRetrievalFailure(
                RuntimeError
            ):
                pass

            controlled_evaluation = (
                SimpleNamespace(
                    classification="partial",
                    confidence=0.95,
                    reason=(
                        "Controlled learner evaluation "
                        "before retrieval failure."
                    ),
                    misconception=None,
                )
            )

            prior_question = (
                "ทรานซิสเตอร์ NPN มีโครงสร้างอย่างไร"
            )

            prior_tutor_question = (
                "ทรานซิสเตอร์ NPN "
                "ประกอบด้วยขั้วใดบ้าง?"
            )

            def build_active_tutor(
                *,
                with_pedagogical_state=False,
            ):

                tutor = AITutor()

                tutor.original_question = (
                    prior_question
                )

                tutor.waiting_for_response = True

                tutor.memory.add_user_message(
                    prior_question
                )

                tutor.memory.add_assistant_message(
                    prior_tutor_question
                )

                if with_pedagogical_state:

                    tutor.state.set_evaluation(
                        "partial"
                    )

                return tutor

            def run_failure(
                tutor,
                learner_message,
            ):

                memory_before = [
                    dict(message)
                    for message
                    in tutor.memory.get_messages()
                ]

                turn_before = (
                    tutor.state.turn_count
                )

                propagated_error = None
                returned_answer = None

                with patch(
                    "app.tutor.evaluate_response",
                    return_value=controlled_evaluation,
                ) as evaluator_call:

                    with patch.object(
                        tutor.knowledge_service,
                        "retrieve",
                        side_effect=(
                            ControlledRetrievalFailure(
                                "Controlled knowledge "
                                "retrieval failure"
                            )
                        ),
                    ) as retrieval_call:

                        with patch.object(
                            tutor.relevance_gate,
                            "evaluate",
                        ) as relevance_call:

                            with patch(
                                "app.tutor.chat_with_ai",
                            ) as generation_call:

                                try:

                                    returned_answer = (
                                        tutor.respond(
                                            learner_message
                                        )
                                    )

                                except Exception as error:

                                    propagated_error = (
                                        error
                                    )

                memory_after = [
                    dict(message)
                    for message
                    in tutor.memory.get_messages()
                ]

                return SimpleNamespace(
                    propagated_error=(
                        propagated_error
                    ),
                    returned_answer=(
                        returned_answer
                    ),
                    evaluator_calls=(
                        evaluator_call.call_count
                    ),
                    retrieval_calls=(
                        retrieval_call.call_count
                    ),
                    relevance_calls=(
                        relevance_call.call_count
                    ),
                    generation_calls=(
                        generation_call.call_count
                    ),
                    memory_before=memory_before,
                    memory_after=memory_after,
                    turn_delta=(
                        tutor.state.turn_count
                        - turn_before
                    ),
                )

            # =====================================================
            # CASE 1
            # First/new learner question
            # =====================================================

            first_tutor = AITutor()

            first_result = run_failure(
                first_tutor,
                (
                    "ทรานซิสเตอร์ NPN "
                    "มีขั้วอะไรบ้าง"
                ),
            )

            self.assert_true(
                isinstance(
                    first_result.propagated_error,
                    ControlledRetrievalFailure,
                ),
                (
                    "RUNTIME-T11 first-turn retrieval "
                    "failure must propagate"
                ),
            )

            self.assert_equal(
                first_result.retrieval_calls,
                1,
                (
                    "RUNTIME-T11 first-turn retrieval "
                    "must be attempted exactly once"
                ),
            )

            self.assert_equal(
                first_result.relevance_calls,
                0,
                (
                    "RUNTIME-T11 relevance must not run "
                    "after failed first-turn retrieval"
                ),
            )

            self.assert_equal(
                first_result.generation_calls,
                0,
                (
                    "RUNTIME-T11 Tutor generation must "
                    "not run after failed retrieval"
                ),
            )

            self.assert_equal(
                first_tutor.last_knowledge_result,
                None,
                (
                    "RUNTIME-T11 must not fabricate "
                    "KnowledgeResult"
                ),
            )

            self.assert_equal(
                first_tutor.last_relevance_result,
                None,
                (
                    "RUNTIME-T11 must not fabricate "
                    "relevance telemetry"
                ),
            )

            self.assert_equal(
                first_tutor.last_grounding_status,
                None,
                (
                    "RUNTIME-T11 grounding must not run "
                    "after unresolved retrieval"
                ),
            )

            self.assert_equal(
                first_result.memory_after,
                first_result.memory_before,
                (
                    "RUNTIME-T11 failed first turn must "
                    "not enter ConversationMemory"
                ),
            )

            self.assert_equal(
                first_tutor.original_question,
                None,
                (
                    "RUNTIME-T11 failed new sequence "
                    "must be closed"
                ),
            )

            self.assert_true(
                first_tutor.waiting_for_response
                is False,
                (
                    "RUNTIME-T11 failed new sequence "
                    "must not remain active"
                ),
            )

            self.assert_equal(
                first_result.turn_delta,
                1,
                (
                    "RUNTIME-T11 failed retrieval turn "
                    "must remain observable"
                ),
            )

            # =====================================================
            # CASE 2
            # Learner answer after an active Tutor question
            # =====================================================

            answer_tutor = (
                build_active_tutor()
            )

            answer_result = run_failure(
                answer_tutor,
                (
                    "มี Base, Collector "
                    "และ Emitter"
                ),
            )

            self.assert_true(
                isinstance(
                    answer_result.propagated_error,
                    ControlledRetrievalFailure,
                ),
                (
                    "RUNTIME-T11 answer-turn retrieval "
                    "failure must propagate"
                ),
            )

            self.assert_equal(
                answer_result.evaluator_calls,
                1,
                (
                    "RUNTIME-T11 answer-like turn must "
                    "complete learner evaluation once"
                ),
            )

            self.assert_equal(
                answer_result.retrieval_calls,
                1,
                (
                    "RUNTIME-T11 answer retrieval "
                    "must be attempted once"
                ),
            )

            self.assert_equal(
                answer_result.relevance_calls,
                0,
                (
                    "RUNTIME-T11 answer-turn relevance "
                    "must remain blocked"
                ),
            )

            self.assert_equal(
                answer_result.generation_calls,
                0,
                (
                    "RUNTIME-T11 answer-turn Tutor "
                    "generation must remain blocked"
                ),
            )

            self.assert_equal(
                answer_tutor.original_question,
                prior_question,
                (
                    "RUNTIME-T11 answer retrieval failure "
                    "must preserve active question"
                ),
            )

            self.assert_true(
                answer_tutor.waiting_for_response
                is True,
                (
                    "RUNTIME-T11 answer retrieval failure "
                    "must preserve active sequence"
                ),
            )

            self.assert_equal(
                answer_tutor.state.last_evaluation,
                "partial",
                (
                    "RUNTIME-T11 completed learner "
                    "evaluation must remain committed"
                ),
            )

            self.assert_equal(
                answer_tutor.state.partial_streak,
                1,
                (
                    "RUNTIME-T11 completed partial "
                    "evaluation streak must be retained"
                ),
            )

            self.assert_equal(
                answer_tutor.state.attempt_count,
                1,
                (
                    "RUNTIME-T11 completed learner "
                    "attempt must remain committed"
                ),
            )

            self.assert_equal(
                answer_result.memory_after,
                answer_result.memory_before,
                (
                    "RUNTIME-T11 failed answer turn must "
                    "not enter ConversationMemory"
                ),
            )

            # =====================================================
            # CASE 3
            # Follow-up question
            # =====================================================

            follow_up_tutor = (
                build_active_tutor(
                    with_pedagogical_state=True,
                )
            )

            follow_up_result = run_failure(
                follow_up_tutor,
                "แล้ว Base คือขั้วอะไร?",
            )

            self.assert_equal(
                follow_up_result.evaluator_calls,
                0,
                (
                    "RUNTIME-T11 follow-up must not be "
                    "evaluated as a learner answer"
                ),
            )

            self.assert_equal(
                (
                    follow_up_tutor
                    .last_learner_turn_routing
                    .route
                ),
                "follow_up_question",
                (
                    "RUNTIME-T11 controlled follow-up "
                    "route"
                ),
            )

            self.assert_equal(
                follow_up_tutor.original_question,
                prior_question,
                (
                    "RUNTIME-T11 failed follow-up "
                    "must restore prior active question"
                ),
            )

            self.assert_true(
                follow_up_tutor.waiting_for_response
                is True,
                (
                    "RUNTIME-T11 failed follow-up must "
                    "preserve prior active sequence"
                ),
            )

            self.assert_equal(
                follow_up_tutor.state.last_evaluation,
                "partial",
                (
                    "RUNTIME-T11 failed follow-up must "
                    "preserve pedagogical state"
                ),
            )

            self.assert_equal(
                follow_up_tutor.state.attempt_count,
                1,
                (
                    "RUNTIME-T11 failed follow-up must "
                    "not create another learner attempt"
                ),
            )

            self.assert_equal(
                follow_up_result.memory_after,
                follow_up_result.memory_before,
                (
                    "RUNTIME-T11 failed follow-up must "
                    "not enter ConversationMemory"
                ),
            )

            # =====================================================
            # CASE 4
            # Clarification question
            # =====================================================

            clarification_tutor = (
                build_active_tutor(
                    with_pedagogical_state=True,
                )
            )

            clarification_result = run_failure(
                clarification_tutor,
                "คำว่า Base หมายถึงอะไร",
            )

            self.assert_equal(
                clarification_result.evaluator_calls,
                0,
                (
                    "RUNTIME-T11 clarification must not "
                    "be evaluated as learner answer"
                ),
            )

            self.assert_equal(
                (
                    clarification_tutor
                    .last_learner_turn_routing
                    .route
                ),
                "clarification_question",
                (
                    "RUNTIME-T11 controlled "
                    "clarification route"
                ),
            )

            self.assert_equal(
                clarification_tutor.original_question,
                prior_question,
                (
                    "RUNTIME-T11 failed clarification "
                    "must preserve active question"
                ),
            )

            self.assert_true(
                clarification_tutor.waiting_for_response
                is True,
                (
                    "RUNTIME-T11 failed clarification "
                    "must preserve active sequence"
                ),
            )

            self.assert_equal(
                clarification_tutor.state.last_evaluation,
                "partial",
                (
                    "RUNTIME-T11 failed clarification "
                    "must preserve pedagogical state"
                ),
            )

            self.assert_equal(
                clarification_tutor.state.attempt_count,
                1,
                (
                    "RUNTIME-T11 failed clarification "
                    "must not create another attempt"
                ),
            )

            self.assert_equal(
                clarification_result.memory_after,
                clarification_result.memory_before,
                (
                    "RUNTIME-T11 failed clarification "
                    "must not enter ConversationMemory"
                ),
            )

            # =====================================================
            # CASE 5
            # Explicit topic change
            # =====================================================

            topic_tutor = (
                build_active_tutor(
                    with_pedagogical_state=True,
                )
            )

            topic_result = run_failure(
                topic_tutor,
                (
                    "ขอถามอีกเรื่อง "
                    "ตัวเก็บประจุทำงานอย่างไร"
                ),
            )

            self.assert_equal(
                topic_result.evaluator_calls,
                0,
                (
                    "RUNTIME-T11 topic change must not "
                    "be evaluated as learner answer"
                ),
            )

            self.assert_equal(
                (
                    topic_tutor
                    .last_learner_turn_routing
                    .route
                ),
                "topic_change",
                (
                    "RUNTIME-T11 controlled "
                    "topic-change route"
                ),
            )

            self.assert_equal(
                topic_tutor.original_question,
                None,
                (
                    "RUNTIME-T11 failed topic-change "
                    "sequence must be closed"
                ),
            )

            self.assert_true(
                topic_tutor.waiting_for_response
                is False,
                (
                    "RUNTIME-T11 undelivered new topic "
                    "must not remain active"
                ),
            )

            self.assert_equal(
                topic_tutor.state.last_evaluation,
                None,
                (
                    "RUNTIME-T11 topic-change reset "
                    "must remain in effect"
                ),
            )

            self.assert_equal(
                topic_tutor.state.partial_streak,
                0,
                (
                    "RUNTIME-T11 topic-change sequence "
                    "must retain reset streak state"
                ),
            )

            self.assert_equal(
                topic_tutor.state.attempt_count,
                0,
                (
                    "RUNTIME-T11 topic-change sequence "
                    "must retain reset attempt state"
                ),
            )

            self.assert_equal(
                topic_result.memory_after,
                topic_result.memory_before,
                (
                    "RUNTIME-T11 failed topic change "
                    "must not enter ConversationMemory"
                ),
            )

            # =====================================================
            # Shared downstream safety
            # =====================================================

            for case_result in (
                answer_result,
                follow_up_result,
                clarification_result,
                topic_result,
            ):

                self.assert_true(
                    isinstance(
                        case_result.propagated_error,
                        ControlledRetrievalFailure,
                    ),
                    (
                        "RUNTIME-T11 original retrieval "
                        "failure must propagate"
                    ),
                )

                self.assert_equal(
                    case_result.retrieval_calls,
                    1,
                    (
                        "RUNTIME-T11 retrieval must not "
                        "retry automatically"
                    ),
                )

                self.assert_equal(
                    case_result.relevance_calls,
                    0,
                    (
                        "RUNTIME-T11 relevance must remain "
                        "blocked after retrieval failure"
                    ),
                )

                self.assert_equal(
                    case_result.generation_calls,
                    0,
                    (
                        "RUNTIME-T11 Tutor generation "
                        "must remain blocked after "
                        "retrieval failure"
                    ),
                )

                self.assert_equal(
                    case_result.turn_delta,
                    1,
                    (
                        "RUNTIME-T11 failed turn must "
                        "remain observable exactly once"
                    ),
                )

            self.pass_test(
                name=name,
                details=(
                    "Knowledge-retrieval runtime failure "
                    "propagates without fabricating course "
                    "knowledge or entering relevance, "
                    "grounding, or Tutor generation; new "
                    "and topic-change sequences close, "
                    "failed follow-ups restore the prior "
                    "active question, clarification preserves "
                    "the prior sequence, completed learner "
                    "evaluation remains committed, and no "
                    "failed turn enters ConversationMemory."
                ),
            )

        except Exception as error:

            self.fail_test(
                name=name,
                details=str(error),
            )
    def test_b1_t1_hard_session_recovery_integrity(
        self,
    ) -> None:

        name = (
            "B1-T1 Hard session recovery integrity"
        )

        try:

            class ControlledRetrievalFailure(
                RuntimeError
            ):
                pass

            class ControlledRepairFailure(
                RuntimeError
            ):
                pass

            # =====================================================
            # Canonical fresh-session state
            # =====================================================

            fresh = AITutor()
            tutor = AITutor()

            runtime_fields = sorted(
                field_name
                for field_name
                in fresh.__dict__
                if (
                    field_name.startswith("last_")
                    or
                    field_name
                    in {
                        "original_question",
                        "waiting_for_response",
                    }
                )
            )

            infrastructure_before = {
                "knowledge_service":
                    id(tutor.knowledge_service),

                "relevance_gate":
                    id(tutor.relevance_gate),

                "grounding_guard":
                    id(tutor.grounding_guard),

                "retrieval_query_builder":
                    id(
                        tutor.retrieval_query_builder
                    ),

                "response_repair_service":
                    id(
                        tutor.response_repair_service
                    ),

                "response_mode_repair_service":
                    id(
                        tutor
                        .response_mode_repair_service
                    ),
            }

            # =====================================================
            # Produce a real retrieval-failure residue.
            # =====================================================

            old_question = (
                "ทรานซิสเตอร์ NPN "
                "มีขั้วอะไรบ้าง"
            )

            with patch.object(
                tutor.knowledge_service,
                "retrieve",
                side_effect=(
                    ControlledRetrievalFailure(
                        "Controlled retrieval failure"
                    )
                ),
            ):

                retrieval_error = None

                try:

                    tutor.respond(
                        old_question
                    )

                except Exception as exc:

                    retrieval_error = exc

            self.assert_true(
                isinstance(
                    retrieval_error,
                    ControlledRetrievalFailure,
                ),
                (
                    "B1-T1 controlled retrieval "
                    "failure must propagate"
                ),
            )

            self.assert_equal(
                tutor.state.turn_count,
                1,
                (
                    "B1-T1 failed turn must remain "
                    "observable before hard reset"
                ),
            )

            self.assert_true(
                bool(
                    tutor.last_retrieval_query
                ),
                (
                    "B1-T1 retrieval failure must leave "
                    "observable retrieval telemetry "
                    "before reset"
                ),
            )

            # =====================================================
            # Also create explicit repair-failure telemetry.
            # =====================================================

            with patch.object(
                tutor.response_mode_repair_service,
                "repair",
                side_effect=(
                    ControlledRepairFailure(
                        "Controlled repair failure"
                    )
                ),
            ):

                repair_error = None

                try:

                    tutor._run_response_repair()

                except Exception as exc:

                    repair_error = exc

            self.assert_true(
                isinstance(
                    repair_error,
                    ControlledRepairFailure,
                ),
                (
                    "B1-T1 controlled repair failure "
                    "must propagate"
                ),
            )

            self.assert_true(
                tutor.last_repair_failed,
                (
                    "B1-T1 repair-provider failure "
                    "telemetry must exist before reset"
                ),
            )

            self.assert_equal(
                tutor.last_repair_failure_type,
                "repair_provider",
                (
                    "B1-T1 repair failure type must be "
                    "observable before reset"
                ),
            )

            # =====================================================
            # Dirty state/history explicitly as well.
            # =====================================================

            tutor.memory.add_user_message(
                "OLD SESSION USER"
            )

            tutor.memory.add_assistant_message(
                "OLD SESSION ASSISTANT"
            )

            tutor.state.scaffolding_level = 5
            tutor.state.last_evaluation = "partial"
            tutor.state.partial_streak = 4
            tutor.state.attempt_count = 7
            tutor.state.hint_count = 3

            tutor.state.add_misconception(
                "Controlled old misconception"
            )

            tutor.learner_progress_history._records.append(
                "__OLD_PROGRESS__"
            )

            tutor.learner_progress_history._current_sequence_id = (
                999
            )

            tutor.response_quality_history._records.append(
                "__OLD_QUALITY__"
            )

            # =====================================================
            # HARD SESSION RESET
            # =====================================================

            tutor.reset()

            # =====================================================
            # Conversation / learner state
            # =====================================================

            self.assert_equal(
                tutor.memory.get_messages(),
                [],
                (
                    "B1-T1 hard reset must clear "
                    "ConversationMemory"
                ),
            )

            self.assert_equal(
                tutor.original_question,
                None,
                (
                    "B1-T1 hard reset must clear "
                    "active question"
                ),
            )

            self.assert_true(
                tutor.waiting_for_response
                is False,
                (
                    "B1-T1 hard reset must close "
                    "active learning sequence"
                ),
            )

            self.assert_equal(
                tutor.state.scaffolding_level,
                1,
                (
                    "B1-T1 scaffolding level must "
                    "return to initial state"
                ),
            )

            self.assert_equal(
                tutor.state.turn_count,
                0,
                (
                    "B1-T1 session turn count must "
                    "return to zero"
                ),
            )

            self.assert_equal(
                tutor.state.last_evaluation,
                None,
                (
                    "B1-T1 evaluation state must clear"
                ),
            )

            self.assert_equal(
                tutor.state.correct_streak,
                0,
                "B1-T1 correct streak must clear",
            )

            self.assert_equal(
                tutor.state.partial_streak,
                0,
                "B1-T1 partial streak must clear",
            )

            self.assert_equal(
                tutor.state.failure_streak,
                0,
                "B1-T1 failure streak must clear",
            )

            self.assert_equal(
                tutor.state.attempt_count,
                0,
                "B1-T1 attempt count must clear",
            )

            self.assert_equal(
                tutor.state.hint_count,
                0,
                "B1-T1 hint count must clear",
            )

            self.assert_equal(
                tutor.state.misconceptions,
                [],
                (
                    "B1-T1 session misconceptions "
                    "must clear"
                ),
            )

            # =====================================================
            # Histories
            # =====================================================

            self.assert_equal(
                tutor.learner_progress_history._records,
                [],
                (
                    "B1-T1 learner-progress history "
                    "must clear"
                ),
            )

            self.assert_equal(
                (
                    tutor.learner_progress_history
                    ._current_sequence_id
                ),
                0,
                (
                    "B1-T1 learner-progress sequence "
                    "counter must reset"
                ),
            )

            self.assert_equal(
                tutor.response_quality_history._records,
                [],
                (
                    "B1-T1 response-quality history "
                    "must clear"
                ),
            )

            # =====================================================
            # Dynamic runtime telemetry parity.
            #
            # Every last_* field plus conversation-boundary fields
            # must match a newly constructed AITutor.
            # =====================================================

            runtime_mismatches = []

            for field_name in runtime_fields:

                actual = getattr(
                    tutor,
                    field_name,
                )

                expected = getattr(
                    fresh,
                    field_name,
                )

                if actual != expected:

                    runtime_mismatches.append(
                        (
                            field_name,
                            expected,
                            actual,
                        )
                    )

            self.assert_equal(
                runtime_mismatches,
                [],
                (
                    "B1-T1 all runtime telemetry must "
                    "match fresh-session defaults after "
                    "hard reset"
                ),
            )

            # =====================================================
            # Infrastructure identity must remain intact.
            # reset() clears session state; it must not rebuild
            # services/dependencies.
            # =====================================================

            infrastructure_after = {
                "knowledge_service":
                    id(tutor.knowledge_service),

                "relevance_gate":
                    id(tutor.relevance_gate),

                "grounding_guard":
                    id(tutor.grounding_guard),

                "retrieval_query_builder":
                    id(
                        tutor.retrieval_query_builder
                    ),

                "response_repair_service":
                    id(
                        tutor.response_repair_service
                    ),

                "response_mode_repair_service":
                    id(
                        tutor
                        .response_mode_repair_service
                    ),
            }

            self.assert_equal(
                infrastructure_after,
                infrastructure_before,
                (
                    "B1-T1 hard reset must preserve "
                    "infrastructure/service identity"
                ),
            )

            # =====================================================
            # Clean new session after reset.
            #
            # Verify that the old question cannot leak into the
            # next retrieval query.
            # =====================================================

            new_question = (
                "อินทิกรัลไม่จำกัดเขตคืออะไร"
            )

            observed_query = {
                "value": None,
            }

            knowledge_result = SimpleNamespace(
                query=new_question,
                context="",
                sources=[],
                citations=[],
            )

            def clean_retrieval(
                query,
                n_results=5,
            ):

                observed_query["value"] = query

                return knowledge_result

            with patch.object(
                tutor.knowledge_service,
                "retrieve",
                side_effect=clean_retrieval,
            ):

                with patch.object(
                    tutor.relevance_gate,
                    "evaluate",
                    return_value=SimpleNamespace(
                        is_relevant=False,
                        confidence=1.0,
                        reason=(
                            "Controlled new-session "
                            "out-of-course result."
                        ),
                    ),
                ):

                    with patch.object(
                        tutor.grounding_guard,
                        "evaluate",
                        return_value=SimpleNamespace(
                            has_knowledge=False,
                            reason=(
                                "Controlled no grounded "
                                "course knowledge."
                            ),
                        ),
                    ):

                        with patch(
                            "app.tutor.chat_with_ai",
                        ) as generation_call:

                            new_answer = (
                                tutor.respond(
                                    new_question
                                )
                            )

            self.assert_true(
                observed_query["value"]
                is not None,
                (
                    "B1-T1 new session must perform "
                    "fresh retrieval"
                ),
            )

            self.assert_true(
                new_question
                in observed_query["value"],
                (
                    "B1-T1 new retrieval query must "
                    "contain the new learner question"
                ),
            )

            self.assert_true(
                old_question
                not in observed_query["value"],
                (
                    "B1-T1 old-session question must "
                    "not leak into new retrieval"
                ),
            )

            self.assert_equal(
                generation_call.call_count,
                0,
                (
                    "B1-T1 controlled out-of-course "
                    "path must not invoke Tutor generation"
                ),
            )

            self.assert_true(
                isinstance(
                    new_answer,
                    str,
                )
                and
                bool(
                    new_answer.strip()
                ),
                (
                    "B1-T1 Tutor must remain usable "
                    "after hard session recovery"
                ),
            )

            self.pass_test(
                name=name,
                details=(
                    "Hard session reset clears memory, "
                    "learner state, histories, all runtime "
                    "telemetry, retrieval/repair failure "
                    "residue, preserves infrastructure "
                    "identity, and starts the next session "
                    "without old-context leakage."
                ),
            )

        except Exception as error:

            self.fail_test(
                name=name,
                details=str(error),
            )
    def _build_b2_active_tutor(
        self,
    ) -> AITutor:

        tutor = AITutor()

        original_question = (
            "ทรานซิสเตอร์ NPN "
            "มีโครงสร้างอย่างไร"
        )

        tutor.original_question = (
            original_question
        )

        tutor.waiting_for_response = True

        tutor.memory.add_user_message(
            original_question
        )

        tutor.memory.add_assistant_message(
            "ทรานซิสเตอร์ NPN "
            "ประกอบด้วยขั้วใดบ้าง?"
        )

        return tutor


    def _build_b2_evaluation(
        self,
    ):

        return SimpleNamespace(
            classification="partial",
            confidence=0.95,
            reason=(
                "Controlled successful "
                "learner evaluation."
            ),
            misconception=None,
        )


    def _build_b2_knowledge(
        self,
        query,
    ):

        return SimpleNamespace(
            query=query,
            context="",
            sources=[],
            citations=[],
        )


    def _build_b2_relevance(
        self,
    ):

        return SimpleNamespace(
            is_relevant=False,
            confidence=1.0,
            reason=(
                "Controlled out-of-course result."
            ),
        )


    def _build_b2_grounding(
        self,
    ):

        return SimpleNamespace(
            has_knowledge=False,
            reason=(
                "Controlled no grounded knowledge."
            ),
        )


    def test_b2_t1_exact_retry_reuses_evaluation(
        self,
    ) -> None:

        name = (
            "B2-T1 Exact retry reuses evaluation"
        )

        try:

            tutor = (
                self._build_b2_active_tutor()
            )

            learner_answer = (
                "มี Base, Collector "
                "และ Emitter"
            )

            retrieval_count = {
                "value": 0,
            }

            def retrieval(
                query,
                n_results=5,
            ):

                retrieval_count["value"] += 1

                if retrieval_count["value"] == 1:

                    raise RuntimeError(
                        "Controlled retrieval failure"
                    )

                return (
                    self._build_b2_knowledge(
                        query
                    )
                )

            with patch(
                "app.tutor.evaluate_response",
                return_value=(
                    self._build_b2_evaluation()
                ),
            ) as evaluator_call:

                with patch.object(
                    tutor.knowledge_service,
                    "retrieve",
                    side_effect=retrieval,
                ):

                    with patch.object(
                        tutor.relevance_gate,
                        "evaluate",
                        return_value=(
                            self._build_b2_relevance()
                        ),
                    ):

                        with patch.object(
                            tutor.grounding_guard,
                            "evaluate",
                            return_value=(
                                self._build_b2_grounding()
                            ),
                        ):

                            with patch(
                                "app.tutor.chat_with_ai",
                            ) as generation_call:

                                first_error = None

                                try:

                                    tutor.respond(
                                        learner_answer
                                    )

                                except Exception as exc:

                                    first_error = exc

                                self.assert_true(
                                    isinstance(
                                        first_error,
                                        RuntimeError,
                                    ),
                                    (
                                        "B2-T1 first retrieval "
                                        "failure must propagate"
                                    ),
                                )

                                self.assert_equal(
                                    tutor.state.attempt_count,
                                    1,
                                    (
                                        "B2-T1 evaluation must "
                                        "commit exactly once "
                                        "before retrieval failure"
                                    ),
                                )

                                self.assert_equal(
                                    (
                                        tutor
                                        ._pending_evaluated_retry_message
                                    ),
                                    learner_answer,
                                    (
                                        "B2-T1 failed evaluated "
                                        "attempt must arm the "
                                        "exact retry marker"
                                    ),
                                )

                                retry_answer = (
                                    tutor.respond(
                                        learner_answer
                                    )
                                )

            self.assert_equal(
                evaluator_call.call_count,
                1,
                (
                    "B2-T1 exact retry must reuse "
                    "the committed evaluation"
                ),
            )

            self.assert_equal(
                tutor.state.attempt_count,
                1,
                (
                    "B2-T1 exact retry must not "
                    "double-count learner attempts"
                ),
            )

            self.assert_equal(
                retrieval_count["value"],
                2,
                (
                    "B2-T1 downstream retrieval "
                    "must actually be retried"
                ),
            )

            self.assert_equal(
                (
                    tutor
                    ._pending_evaluated_retry_message
                ),
                None,
                (
                    "B2-T1 successful retry must "
                    "consume the retry marker"
                ),
            )

            self.assert_equal(
                generation_call.call_count,
                0,
                (
                    "B2-T1 controlled out-of-course "
                    "completion must not invoke Tutor "
                    "generation"
                ),
            )

            self.assert_true(
                isinstance(
                    retry_answer,
                    str,
                )
                and
                bool(
                    retry_answer.strip()
                ),
                (
                    "B2-T1 retry must complete with "
                    "a valid Tutor response"
                ),
            )

            self.pass_test(
                name=name,
                details=(
                    "An exact retry after post-evaluation "
                    "retrieval failure reuses the committed "
                    "evaluation, retries retrieval, and does "
                    "not double-count the learner attempt."
                ),
            )

        except Exception as error:

            self.fail_test(
                name=name,
                details=str(error),
            )

    def _build_b3_grounded_knowledge(
        self,
        query,
    ):

        return SimpleNamespace(
            query=query,
            context=(
                "NPN transistor course knowledge. "
                "The transistor material contains "
                "Emitter, Base, Collector and "
                "NPN structure information."
            ),
            sources=[],
            citations=[],
        )


    def _build_b3_relevance(
        self,
    ):

        return SimpleNamespace(
            is_relevant=True,
            confidence=1.0,
            reason=(
                "Controlled relevant result."
            ),
        )

    def _build_b3_grounded_knowledge(
        self,
        query,
    ):

        return SimpleNamespace(
            query=query,
            context=(
                "NPN transistor course knowledge. "
                "The transistor material contains "
                "Emitter, Base, Collector and "
                "NPN structure information."
            ),
            sources=[],
            citations=[],
        )


    def _build_b3_relevance(
        self,
    ):

        return SimpleNamespace(
            is_relevant=True,
            confidence=1.0,
            reason=(
                "Controlled relevant result."
            ),
        )


    def _build_b3_grounding(
        self,
    ):

        return SimpleNamespace(
            has_knowledge=True,
            reason=(
                "Controlled grounded knowledge."
            ),
        )


    def test_b2_t2_different_answer_is_new_attempt(
        self,
    ) -> None:

        name = (
            "B2-T2 Different answer is new attempt"
        )

        try:

            tutor = (
                self._build_b2_active_tutor()
            )

            first_answer = (
                "มี Base, Collector "
                "และ Emitter"
            )

            second_answer = (
                "Base ใช้ควบคุมการทำงาน "
                "ของทรานซิสเตอร์"
            )

            retrieval_count = {
                "value": 0,
            }

            def retrieval(
                query,
                n_results=5,
            ):

                retrieval_count["value"] += 1

                if retrieval_count["value"] == 1:

                    raise RuntimeError(
                        "Controlled retrieval failure"
                    )

                return (
                    self._build_b2_knowledge(
                        query
                    )
                )

            with patch(
                "app.tutor.evaluate_response",
                return_value=(
                    self._build_b2_evaluation()
                ),
            ) as evaluator_call:

                with patch.object(
                    tutor.knowledge_service,
                    "retrieve",
                    side_effect=retrieval,
                ):

                    with patch.object(
                        tutor.relevance_gate,
                        "evaluate",
                        return_value=(
                            self._build_b2_relevance()
                        ),
                    ):

                        with patch.object(
                            tutor.grounding_guard,
                            "evaluate",
                            return_value=(
                                self._build_b2_grounding()
                            ),
                        ):

                            with patch(
                                "app.tutor.chat_with_ai",
                            ):

                                try:

                                    tutor.respond(
                                        first_answer
                                    )

                                except RuntimeError:

                                    pass

                                self.assert_equal(
                                    (
                                        tutor
                                        ._pending_evaluated_retry_message
                                    ),
                                    first_answer,
                                    (
                                        "B2-T2 first failed "
                                        "evaluated attempt must "
                                        "arm retry state"
                                    ),
                                )

                                tutor.respond(
                                    second_answer
                                )

            self.assert_equal(
                evaluator_call.call_count,
                2,
                (
                    "B2-T2 a different learner "
                    "answer must be evaluated as "
                    "a new attempt"
                ),
            )

            self.assert_equal(
                tutor.state.attempt_count,
                2,
                (
                    "B2-T2 different answer must "
                    "increase attempt_count"
                ),
            )

            self.assert_equal(
                (
                    tutor
                    ._pending_evaluated_retry_message
                ),
                None,
                (
                    "B2-T2 old retry marker must "
                    "not survive a different answer"
                ),
            )

            self.pass_test(
                name=name,
                details=(
                    "Retry deduplication is exact-message "
                    "scoped; a different learner answer "
                    "remains a genuine new attempt."
                ),
            )

        except Exception as error:

            self.fail_test(
                name=name,
                details=str(error),
            )

    def test_b2_t3_repeated_failure_rearms_retry(
        self,
    ) -> None:

        name = (
            "B2-T3 Repeated failure re-arms retry"
        )

        try:

            tutor = (
                self._build_b2_active_tutor()
            )

            learner_answer = (
                "มี Base, Collector "
                "และ Emitter"
            )

            retrieval_count = {
                "value": 0,
            }

            def retrieval(
                query,
                n_results=5,
            ):

                retrieval_count["value"] += 1

                if retrieval_count["value"] <= 2:

                    raise RuntimeError(
                        "Controlled repeated "
                        "retrieval failure"
                    )

                return (
                    self._build_b2_knowledge(
                        query
                    )
                )

            with patch(
                "app.tutor.evaluate_response",
                return_value=(
                    self._build_b2_evaluation()
                ),
            ) as evaluator_call:

                with patch.object(
                    tutor.knowledge_service,
                    "retrieve",
                    side_effect=retrieval,
                ):

                    with patch.object(
                        tutor.relevance_gate,
                        "evaluate",
                        return_value=(
                            self._build_b2_relevance()
                        ),
                    ):

                        with patch.object(
                            tutor.grounding_guard,
                            "evaluate",
                            return_value=(
                                self._build_b2_grounding()
                            ),
                        ):

                            with patch(
                                "app.tutor.chat_with_ai",
                            ):

                                first_error = None

                                try:

                                    tutor.respond(
                                        learner_answer
                                    )

                                except Exception as exc:

                                    first_error = exc

                                self.assert_true(
                                    isinstance(
                                        first_error,
                                        RuntimeError,
                                    ),
                                    (
                                        "B2-T3 first retrieval "
                                        "failure must propagate"
                                    ),
                                )

                                self.assert_equal(
                                    (
                                        tutor
                                        ._pending_evaluated_retry_message
                                    ),
                                    learner_answer,
                                    (
                                        "B2-T3 first failure must "
                                        "arm retry state"
                                    ),
                                )

                                second_error = None

                                try:

                                    tutor.respond(
                                        learner_answer
                                    )

                                except Exception as exc:

                                    second_error = exc

                                self.assert_true(
                                    isinstance(
                                        second_error,
                                        RuntimeError,
                                    ),
                                    (
                                        "B2-T3 repeated retrieval "
                                        "failure must propagate"
                                    ),
                                )

                                self.assert_equal(
                                    (
                                        tutor
                                        ._pending_evaluated_retry_message
                                    ),
                                    learner_answer,
                                    (
                                        "B2-T3 second downstream "
                                        "failure must re-arm the "
                                        "same committed attempt"
                                    ),
                                )

                                tutor.respond(
                                    learner_answer
                                )

            self.assert_equal(
                evaluator_call.call_count,
                1,
                (
                    "B2-T3 repeated retries must "
                    "never re-evaluate the same "
                    "committed learner attempt"
                ),
            )

            self.assert_equal(
                tutor.state.attempt_count,
                1,
                (
                    "B2-T3 repeated downstream "
                    "failures must not inflate "
                    "attempt_count"
                ),
            )

            self.assert_equal(
                retrieval_count["value"],
                3,
                (
                    "B2-T3 retrieval must be attempted "
                    "once per submitted retry"
                ),
            )

            self.assert_equal(
                (
                    tutor
                    ._pending_evaluated_retry_message
                ),
                None,
                (
                    "B2-T3 marker must be consumed "
                    "after eventual completion"
                ),
            )

            self.pass_test(
                name=name,
                details=(
                    "Repeated downstream retrieval failures "
                    "re-arm the same evaluated-attempt marker "
                    "without re-running evaluation or inflating "
                    "learner progress state."
                ),
            )

        except Exception as error:

            self.fail_test(
                name=name,
                details=str(error),
            )

    def test_b2_t4_hard_reset_clears_retry_marker(
        self,
    ) -> None:

        name = (
            "B2-T4 Hard reset clears retry marker"
        )

        try:

            tutor = (
                self._build_b2_active_tutor()
            )

            learner_answer = (
                "มี Base, Collector "
                "และ Emitter"
            )

            with patch(
                "app.tutor.evaluate_response",
                return_value=(
                    self._build_b2_evaluation()
                ),
            ):

                with patch.object(
                    tutor.knowledge_service,
                    "retrieve",
                    side_effect=RuntimeError(
                        "Controlled retrieval failure"
                    ),
                ):

                    try:

                        tutor.respond(
                            learner_answer
                        )

                    except RuntimeError:

                        pass

            self.assert_equal(
                (
                    tutor
                    ._pending_evaluated_retry_message
                ),
                learner_answer,
                (
                    "B2-T4 setup must create an "
                    "actual pending retry marker"
                ),
            )

            tutor.reset()

            retry_fields = {
                "message":
                    tutor
                    ._pending_evaluated_retry_message,

                "original_question":
                    tutor
                    ._pending_evaluated_retry_original_question,

                "tutor_question":
                    tutor
                    ._pending_evaluated_retry_tutor_question,

                "evaluation":
                    tutor
                    ._pending_evaluated_retry_evaluation_result,

                "decision":
                    tutor
                    ._pending_evaluated_retry_decision,
            }

            self.assert_equal(
                retry_fields,
                {
                    "message": None,
                    "original_question": None,
                    "tutor_question": None,
                    "evaluation": None,
                    "decision": None,
                },
                (
                    "B2-T4 hard session reset must "
                    "clear all pending evaluated-retry "
                    "transaction state"
                ),
            )

            self.assert_equal(
                tutor.state.attempt_count,
                0,
                (
                    "B2-T4 hard reset must also "
                    "clear committed learner attempts"
                ),
            )

            self.assert_equal(
                tutor.memory.get_messages(),
                [],
                (
                    "B2-T4 hard reset must preserve "
                    "the existing hard-session boundary"
                ),
            )

            self.pass_test(
                name=name,
                details=(
                    "Hard session reset clears every "
                    "pending evaluated-retry field together "
                    "with learner state and conversation memory."
                ),
            )

        except Exception as error:

            self.fail_test(
                name=name,
                details=str(error),
            )

    def test_b3_t1_failed_turn_does_not_persist(
        self,
    ) -> None:

        name = (
            "B3-T1 Failed turn does not persist"
        )

        try:

            tutor = (
                self._build_b2_active_tutor()
            )

            learner_answer = (
                "มี Base, Collector "
                "และ Emitter"
            )

            memory_before = [
                dict(item)
                for item
                in tutor.memory.get_messages()
            ]

            history_before = len(
                tutor
                .learner_progress_history
                ._records
            )

            with patch(
                "app.tutor.evaluate_response",
                return_value=(
                    self._build_b2_evaluation()
                ),
            ) as evaluator_call:

                with patch.object(
                    tutor.knowledge_service,
                    "retrieve",
                    side_effect=RuntimeError(
                        "Controlled retrieval failure"
                    ),
                ) as retrieval_call:

                    with patch(
                        "app.tutor.chat_with_ai",
                    ) as generation_call:

                        error = None

                        try:

                            tutor.respond(
                                learner_answer
                            )

                        except Exception as exc:

                            error = exc

            self.assert_true(
                isinstance(
                    error,
                    RuntimeError,
                ),
                (
                    "B3-T1 retrieval failure "
                    "must propagate"
                ),
            )

            self.assert_equal(
                tutor.state.turn_count,
                1,
                (
                    "B3-T1 submitted learner turn "
                    "must remain observable"
                ),
            )

            self.assert_equal(
                tutor.state.attempt_count,
                1,
                (
                    "B3-T1 successful evaluation "
                    "must remain committed"
                ),
            )

            self.assert_equal(
                tutor.memory.get_messages(),
                memory_before,
                (
                    "B3-T1 failed turn must not "
                    "enter ConversationMemory"
                ),
            )

            self.assert_equal(
                len(
                    tutor
                    .learner_progress_history
                    ._records
                ),
                history_before,
                (
                    "B3-T1 failed turn must not "
                    "create completed-turn history"
                ),
            )

            self.assert_equal(
                evaluator_call.call_count,
                1,
                "B3-T1 evaluator call count",
            )

            self.assert_equal(
                retrieval_call.call_count,
                1,
                "B3-T1 retrieval call count",
            )

            self.assert_equal(
                generation_call.call_count,
                0,
                (
                    "B3-T1 Tutor generation must "
                    "not run after retrieval failure"
                ),
            )

            self.pass_test(
                name=name,
                details=(
                    "A learner evaluation may remain "
                    "committed after downstream retrieval "
                    "failure, while failed-turn conversation "
                    "memory and completed-turn progress history "
                    "remain uncommitted."
                ),
            )

        except Exception as error:

            self.fail_test(
                name=name,
                details=str(error),
            )

    def test_b3_t2_exact_retry_commits_once(
        self,
    ) -> None:

        name = (
            "B3-T2 Exact retry commits once"
        )

        try:

            tutor = (
                self._build_b2_active_tutor()
            )

            learner_answer = (
                "มี Base, Collector "
                "และ Emitter"
            )

            normal_response = (
                "คุณคิดว่าในทรานซิสเตอร์ NPN "
                "จะมีส่วนประกอบหลักอะไรบ้าง?"
            )

            memory_before = [
                dict(item)
                for item
                in tutor.memory.get_messages()
            ]

            history_before = len(
                tutor
                .learner_progress_history
                ._records
            )

            retrieval_count = {
                "value": 0,
            }

            def retrieval(
                query,
                n_results=5,
            ):

                retrieval_count["value"] += 1

                if retrieval_count["value"] == 1:

                    raise RuntimeError(
                        "Controlled retrieval failure"
                    )

                return (
                    self._build_b3_grounded_knowledge(
                        query
                    )
                )

            with patch(
                "app.tutor.evaluate_response",
                return_value=(
                    self._build_b2_evaluation()
                ),
            ) as evaluator_call:

                with patch.object(
                    tutor.knowledge_service,
                    "retrieve",
                    side_effect=retrieval,
                ):

                    with patch.object(
                        tutor.relevance_gate,
                        "evaluate",
                        return_value=(
                            self._build_b3_relevance()
                        ),
                    ):

                        with patch.object(
                            tutor.grounding_guard,
                            "evaluate",
                            return_value=(
                                self._build_b3_grounding()
                            ),
                        ):

                            with patch(
                                "app.tutor.chat_with_ai",
                                return_value=(
                                    normal_response
                                ),
                            ) as generation_call:

                                try:

                                    tutor.respond(
                                        learner_answer
                                    )

                                except RuntimeError:

                                    pass

                                self.assert_equal(
                                    tutor.memory.get_messages(),
                                    memory_before,
                                    (
                                        "B3-T2 failed first "
                                        "submission must not persist"
                                    ),
                                )

                                self.assert_equal(
                                    len(
                                        tutor
                                        .learner_progress_history
                                        ._records
                                    ),
                                    history_before,
                                    (
                                        "B3-T2 failed first "
                                        "submission must not "
                                        "create history"
                                    ),
                                )

                                answer = tutor.respond(
                                    learner_answer
                                )

            expected_memory = (
                memory_before
                +
                [
                    {
                        "role": "user",
                        "content": learner_answer,
                    },
                    {
                        "role": "assistant",
                        "content": answer,
                    },
                ]
            )

            self.assert_equal(
                evaluator_call.call_count,
                1,
                (
                    "B3-T2 exact retry must not "
                    "re-evaluate the attempt"
                ),
            )

            self.assert_equal(
                retrieval_count["value"],
                2,
                (
                    "B3-T2 retrieval must be "
                    "retried once"
                ),
            )

            self.assert_equal(
                generation_call.call_count,
                1,
                (
                    "B3-T2 only successful retry "
                    "may reach Tutor generation"
                ),
            )

            self.assert_equal(
                tutor.state.turn_count,
                2,
                (
                    "B3-T2 both submissions must "
                    "remain observable"
                ),
            )

            self.assert_equal(
                tutor.state.attempt_count,
                1,
                (
                    "B3-T2 one logical learner "
                    "attempt must be counted once"
                ),
            )

            self.assert_equal(
                tutor.memory.get_messages(),
                expected_memory,
                (
                    "B3-T2 completed retry must "
                    "persist exactly one user/"
                    "assistant pair"
                ),
            )

            self.assert_equal(
                len(
                    tutor
                    .learner_progress_history
                    ._records
                ),
                history_before + 1,
                (
                    "B3-T2 completed retry must "
                    "create exactly one history record"
                ),
            )

            self.pass_test(
                name=name,
                details=(
                    "A failed evaluated submission does "
                    "not persist, while its successful "
                    "exact retry commits exactly one "
                    "conversation pair and one completed-"
                    "turn progress record."
                ),
            )

        except Exception as error:

            self.fail_test(
                name=name,
                details=str(error),
            )

    def test_b3_t3_repeated_failures_commit_once(
        self,
    ) -> None:

        name = (
            "B3-T3 Repeated failures commit once"
        )

        try:

            tutor = (
                self._build_b2_active_tutor()
            )

            learner_answer = (
                "มี Base, Collector "
                "และ Emitter"
            )

            normal_response = (
                "คุณคิดว่าในทรานซิสเตอร์ NPN "
                "จะมีส่วนประกอบหลักอะไรบ้าง?"
            )

            memory_before = [
                dict(item)
                for item
                in tutor.memory.get_messages()
            ]

            history_before = len(
                tutor
                .learner_progress_history
                ._records
            )

            retrieval_count = {
                "value": 0,
            }

            def retrieval(
                query,
                n_results=5,
            ):

                retrieval_count["value"] += 1

                if retrieval_count["value"] <= 2:

                    raise RuntimeError(
                        "Controlled repeated "
                        "retrieval failure"
                    )

                return (
                    self._build_b3_grounded_knowledge(
                        query
                    )
                )

            with patch(
                "app.tutor.evaluate_response",
                return_value=(
                    self._build_b2_evaluation()
                ),
            ) as evaluator_call:

                with patch.object(
                    tutor.knowledge_service,
                    "retrieve",
                    side_effect=retrieval,
                ):

                    with patch.object(
                        tutor.relevance_gate,
                        "evaluate",
                        return_value=(
                            self._build_b3_relevance()
                        ),
                    ):

                        with patch.object(
                            tutor.grounding_guard,
                            "evaluate",
                            return_value=(
                                self._build_b3_grounding()
                            ),
                        ):

                            with patch(
                                "app.tutor.chat_with_ai",
                                return_value=(
                                    normal_response
                                ),
                            ) as generation_call:

                                for _ in range(2):

                                    try:

                                        tutor.respond(
                                            learner_answer
                                        )

                                    except RuntimeError:

                                        pass

                                self.assert_equal(
                                    tutor.memory.get_messages(),
                                    memory_before,
                                    (
                                        "B3-T3 repeated failed "
                                        "submissions must not "
                                        "enter memory"
                                    ),
                                )

                                self.assert_equal(
                                    len(
                                        tutor
                                        .learner_progress_history
                                        ._records
                                    ),
                                    history_before,
                                    (
                                        "B3-T3 repeated failed "
                                        "submissions must not "
                                        "create history"
                                    ),
                                )

                                answer = tutor.respond(
                                    learner_answer
                                )

            self.assert_equal(
                evaluator_call.call_count,
                1,
                (
                    "B3-T3 repeated retries must "
                    "reuse one evaluation"
                ),
            )

            self.assert_equal(
                retrieval_count["value"],
                3,
                (
                    "B3-T3 each submission must "
                    "retry downstream retrieval"
                ),
            )

            self.assert_equal(
                generation_call.call_count,
                1,
                (
                    "B3-T3 only eventual successful "
                    "submission reaches generation"
                ),
            )

            self.assert_equal(
                tutor.state.turn_count,
                3,
                (
                    "B3-T3 session turn count must "
                    "reflect all three submissions"
                ),
            )

            self.assert_equal(
                tutor.state.attempt_count,
                1,
                (
                    "B3-T3 one logical learner "
                    "attempt must remain one attempt"
                ),
            )

            self.assert_equal(
                len(
                    tutor.memory.get_messages()
                ),
                len(memory_before) + 2,
                (
                    "B3-T3 persistence must contain "
                    "only one completed user/"
                    "assistant pair"
                ),
            )

            self.assert_equal(
                tutor.memory.get_messages()[-2:],
                [
                    {
                        "role": "user",
                        "content": learner_answer,
                    },
                    {
                        "role": "assistant",
                        "content": answer,
                    },
                ],
                (
                    "B3-T3 completed pair must "
                    "correspond to eventual success"
                ),
            )

            self.assert_equal(
                len(
                    tutor
                    .learner_progress_history
                    ._records
                ),
                history_before + 1,
                (
                    "B3-T3 eventual completion must "
                    "produce exactly one history record"
                ),
            )

            self.pass_test(
                name=name,
                details=(
                    "Repeated downstream failures remain "
                    "non-persistent; the eventual successful "
                    "retry creates one completed conversation "
                    "pair and one progress-history record while "
                    "session turn_count reflects every submission."
                ),
            )

        except Exception as error:

            self.fail_test(
                name=name,
                details=str(error),
            )

    def test_b3_t4_out_of_course_commit_policy(
        self,
    ) -> None:

        name = (
            "B3-T4 Out-of-course commit policy"
        )

        try:

            tutor = AITutor()

            question = (
                "อินทิกรัลไม่จำกัดเขตคืออะไร"
            )

            knowledge_result = (
                SimpleNamespace(
                    query=question,
                    context="",
                    sources=[],
                    citations=[],
                )
            )

            tutor.knowledge_service.retrieve = (
                lambda query, n_results=5:
                knowledge_result
            )

            tutor.relevance_gate.evaluate = (
                lambda query, knowledge_result:
                SimpleNamespace(
                    is_relevant=False,
                    confidence=1.0,
                    reason=(
                        "Controlled out-of-course."
                    ),
                )
            )

            tutor.grounding_guard.evaluate = (
                lambda result, relevance:
                SimpleNamespace(
                    has_knowledge=False,
                    reason=(
                        "Controlled no grounded "
                        "course knowledge."
                    ),
                )
            )

            memory_before = [
                dict(item)
                for item
                in tutor.memory.get_messages()
            ]

            history_before = len(
                tutor
                .learner_progress_history
                ._records
            )

            with patch(
                "app.tutor.chat_with_ai",
            ) as generation_call:

                answer = tutor.respond(
                    question
                )

            self.assert_equal(
                tutor.state.turn_count,
                1,
                (
                    "B3-T4 completed out-of-course "
                    "submission must be observable"
                ),
            )

            self.assert_equal(
                tutor.memory.get_messages(),
                memory_before,
                (
                    "B3-T4 out-of-course fallback "
                    "must not enter ConversationMemory"
                ),
            )

            self.assert_equal(
                len(
                    tutor
                    .learner_progress_history
                    ._records
                ),
                history_before + 1,
                (
                    "B3-T4 completed out-of-course "
                    "turn must create one progress "
                    "history observation"
                ),
            )

            self.assert_equal(
                generation_call.call_count,
                0,
                (
                    "B3-T4 out-of-course path must "
                    "not invoke Tutor generation"
                ),
            )

            self.assert_true(
                tutor.original_question
                is None,
                (
                    "B3-T4 out-of-course path must "
                    "close the active sequence"
                ),
            )

            self.assert_true(
                tutor.waiting_for_response
                is False,
                (
                    "B3-T4 out-of-course path must "
                    "not leave the Tutor waiting"
                ),
            )

            self.assert_true(
                isinstance(
                    answer,
                    str,
                )
                and
                bool(
                    answer.strip()
                ),
                (
                    "B3-T4 fallback must remain "
                    "a valid completed Tutor response"
                ),
            )

            self.pass_test(
                name=name,
                details=(
                    "The out-of-course completed-turn "
                    "policy remains intentionally asymmetric: "
                    "ConversationMemory stays clean while one "
                    "progress-history observation records the "
                    "completed submission."
                ),
            )

        except Exception as error:

            self.fail_test(
                name=name,
                details=str(error),
            )

    def test_c1_t1_valid_cross_course_startup_identity(
        self,
    ) -> None:

        name = (
            "C1-T1 Valid cross-course startup identity"
        )

        try:

            observed = {}

            for course_id in (
                "electronics",
                "mathematics",
            ):

                with patch(
                    "app.tutor.ACTIVE_COURSE",
                    course_id,
                ):

                    tutor = AITutor()

                observed[course_id] = (
                    tutor.course_profile.course_id
                )

            self.assert_equal(
                observed["electronics"],
                "electronics",
                (
                    "C1-T1 electronics startup "
                    "identity mismatch"
                ),
            )

            self.assert_equal(
                observed["mathematics"],
                "mathematics",
                (
                    "C1-T1 mathematics startup "
                    "identity mismatch"
                ),
            )

            self.pass_test(
                name=name,
                details=(
                    "Each supported active course starts "
                    "with the matching canonical "
                    "CourseProfile identity."
                ),
            )

        except Exception as error:

            self.fail_test(
                name=name,
                details=str(error),
            )
    def test_c1_t2_unknown_course_fails_before_knowledge_service(
        self,
    ) -> None:

        name = (
            "C1-T2 Unknown course fails before "
            "KnowledgeService"
        )

        try:

            error = None

            with patch(
                "app.tutor.ACTIVE_COURSE",
                "__missing_course__",
            ):

                with patch(
                    "app.tutor.KnowledgeService",
                ) as knowledge_service:

                    try:

                        AITutor()

                    except Exception as exc:

                        error = exc

            self.assert_true(
                isinstance(
                    error,
                    FileNotFoundError,
                ),
                (
                    "C1-T2 unknown ACTIVE_COURSE "
                    "must raise FileNotFoundError"
                ),
            )

            self.assert_equal(
                knowledge_service.call_count,
                0,
                (
                    "C1-T2 KnowledgeService must "
                    "not be constructed after "
                    "course-profile startup failure"
                ),
            )

            self.pass_test(
                name=name,
                details=(
                    "An unknown ACTIVE_COURSE fails "
                    "at CourseProfile loading before "
                    "KnowledgeService construction, "
                    "with no silent course fallback."
                ),
            )

        except Exception as error:

            self.fail_test(
                name=name,
                details=str(error),
            )


    def test_c1_t3_invalid_course_profiles_fail_fast(
        self,
    ) -> None:

        name = (
            "C1-T3 Invalid course profiles fail fast"
        )

        try:

            # -------------------------------------------------
            # Malformed JSON
            # -------------------------------------------------

            malformed_root = Path(
                tempfile.mkdtemp()
            )

            malformed_dir = (
                malformed_root
                /
                "broken"
            )

            malformed_dir.mkdir()

            (
                malformed_dir
                /
                "course.json"
            ).write_text(
                "{ invalid json",
                encoding="utf-8",
            )

            malformed_loader = (
                CourseProfileLoader(
                    base_path=malformed_root
                )
            )

            malformed_error = None

            with patch(
                "app.tutor.ACTIVE_COURSE",
                "broken",
            ):

                with patch(
                    "app.tutor.CourseProfileLoader",
                    return_value=malformed_loader,
                ):

                    with patch(
                        "app.tutor.KnowledgeService",
                    ) as malformed_knowledge:

                        try:

                            AITutor()

                        except Exception as exc:

                            malformed_error = exc

            self.assert_true(
                isinstance(
                    malformed_error,
                    json.JSONDecodeError,
                ),
                (
                    "C1-T3 malformed course.json "
                    "must fail with JSONDecodeError"
                ),
            )

            self.assert_equal(
                malformed_knowledge.call_count,
                0,
                (
                    "C1-T3 malformed profile must "
                    "fail before KnowledgeService"
                ),
            )

            # -------------------------------------------------
            # Structurally invalid profile
            # -------------------------------------------------

            invalid_root = Path(
                tempfile.mkdtemp()
            )

            invalid_dir = (
                invalid_root
                /
                "invalid"
            )

            invalid_dir.mkdir()

            (
                invalid_dir
                /
                "course.json"
            ).write_text(
                json.dumps(
                    {
                        "course_id": "invalid",
                        "course_name":
                            "Invalid Course",

                        # document_path intentionally absent

                        "chroma_path":
                            "./data/chroma/invalid",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            invalid_loader = (
                CourseProfileLoader(
                    base_path=invalid_root
                )
            )

            invalid_error = None

            with patch(
                "app.tutor.ACTIVE_COURSE",
                "invalid",
            ):

                with patch(
                    "app.tutor.CourseProfileLoader",
                    return_value=invalid_loader,
                ):

                    with patch(
                        "app.tutor.KnowledgeService",
                    ) as invalid_knowledge:

                        try:

                            AITutor()

                        except Exception as exc:

                            invalid_error = exc

            self.assert_true(
                isinstance(
                    invalid_error,
                    ValueError,
                ),
                (
                    "C1-T3 invalid CourseProfile "
                    "must raise ValueError"
                ),
            )

            self.assert_equal(
                invalid_knowledge.call_count,
                0,
                (
                    "C1-T3 invalid CourseProfile "
                    "must fail before KnowledgeService"
                ),
            )

            self.pass_test(
                name=name,
                details=(
                    "Malformed JSON and invalid "
                    "CourseProfile configuration fail "
                    "during startup before downstream "
                    "knowledge infrastructure is created."
                ),
            )

        except Exception as error:

            self.fail_test(
                name=name,
                details=str(error),
            )

    def test_c1_t4_course_identity_mismatch_rejected(
        self,
    ) -> None:

        name = (
            "C1-T4 Course identity mismatch rejected"
        )

        try:

            root = Path(
                tempfile.mkdtemp()
            )

            course_dir = (
                root
                /
                "electronics"
            )

            course_dir.mkdir()

            (
                course_dir
                /
                "course.json"
            ).write_text(
                json.dumps(
                    {
                        "course_id":
                            "mathematics",

                        "course_name":
                            "Wrong Course Identity",

                        "document_path":
                            "./docs/electronics",

                        "chroma_path":
                            "./data/chroma/electronics",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            loader = CourseProfileLoader(
                base_path=root
            )

            profile = None
            error = None

            try:

                profile = loader.load(
                    "electronics"
                )

            except Exception as exc:

                error = exc

            self.assert_true(
                isinstance(
                    error,
                    ValueError,
                ),
                (
                    "C1-T4 requested/profile "
                    "course_id mismatch must "
                    "raise ValueError"
                ),
            )

            self.assert_true(
                profile is None,
                (
                    "C1-T4 mismatched profile "
                    "must not be returned"
                ),
            )

            self.assert_true(
                (
                    "does not match"
                    in str(error)
                ),
                (
                    "C1-T4 mismatch error must "
                    "identify the course identity "
                    "configuration problem"
                ),
            )

            self.pass_test(
                name=name,
                details=(
                    "CourseProfileLoader rejects a "
                    "course.json whose declared course_id "
                    "does not match the requested canonical "
                    "course, preventing silent cross-course "
                    "startup."
                ),
            )

        except Exception as error:

            self.fail_test(
                name=name,
                details=str(error),
            )      

    def test_c2_t1_real_course_stores_startup(
        self,
    ) -> None:

        name = (
            "C2-T1 Real course knowledge stores startup"
        )

        try:

            observed = {}

            for course_id in (
                "electronics",
                "mathematics",
            ):

                with patch(
                    "app.tutor.ACTIVE_COURSE",
                    course_id,
                ):

                    tutor = AITutor()

                observed[course_id] = {
                    "course_id":
                        tutor.course_profile.course_id,

                    "chroma_path":
                        tutor.course_profile.chroma_path,
                }

            self.assert_equal(
                observed["electronics"]["course_id"],
                "electronics",
                "C2-T1 electronics course identity",
            )

            self.assert_equal(
                observed["mathematics"]["course_id"],
                "mathematics",
                "C2-T1 mathematics course identity",
            )

            self.assert_true(
                Path(
                    observed["electronics"]["chroma_path"]
                ).is_dir(),
                "C2-T1 electronics Chroma store missing",
            )

            self.assert_true(
                Path(
                    observed["mathematics"]["chroma_path"]
                ).is_dir(),
                "C2-T1 mathematics Chroma store missing",
            )

            self.pass_test(
                name=name,
                details=(
                    "Both configured courses start only "
                    "with usable indexed knowledge stores."
                ),
            )

        except Exception as error:

            self.fail_test(
                name=name,
                details=str(error),
            )

    def test_c2_t2_missing_store_fails_without_creation(
        self,
    ) -> None:

        name = (
            "C2-T2 Missing knowledge store fails "
            "without creation"
        )

        try:

            with tempfile.TemporaryDirectory() as root:

                missing_store = (
                    Path(root)
                    /
                    "missing_chroma"
                )

                profile = replace(
                    CourseProfileLoader().load(
                        "electronics"
                    ),
                    chroma_path=str(
                        missing_store
                    ),
                )

                error = None
                tutor = None

                with patch(
                    "app.tutor.CourseProfileLoader",
                ) as loader_class:

                    loader_class.return_value.load.return_value = (
                        profile
                    )

                    try:

                        tutor = AITutor()

                    except Exception as exc:

                        error = exc

                self.assert_true(
                    isinstance(
                        error,
                        FileNotFoundError,
                    ),
                    (
                        "C2-T2 missing store must "
                        "raise FileNotFoundError"
                    ),
                )

                self.assert_true(
                    tutor is None,
                    (
                        "C2-T2 Tutor must not complete "
                        "startup with missing store"
                    ),
                )

                self.assert_true(
                    not missing_store.exists(),
                    (
                        "C2-T2 startup validation must "
                        "not create missing store"
                    ),
                )

            self.pass_test(
                name=name,
                details=(
                    "Missing course knowledge store "
                    "fails closed without filesystem mutation."
                ),
            )

        except Exception as error:

            self.fail_test(
                name=name,
                details=str(error),
            )

    def test_c2_t3_runtime_read_never_autocreates_store(
        self,
    ) -> None:

        name = (
            "C2-T3 Runtime read never autocreates store"
        )

        try:

            with tempfile.TemporaryDirectory() as root:

                missing_store = (
                    Path(root)
                    /
                    "runtime_missing"
                )

                profile = replace(
                    CourseProfileLoader().load(
                        "electronics"
                    ),
                    chroma_path=str(
                        missing_store
                    ),
                )

                service = KnowledgeService(
                    course_profile=profile
                )

                error = None

                try:

                    with patch(
                        (
                            "app.services."
                            "knowledge_service."
                            "create_embedding"
                        ),
                        return_value=[
                            0.0,
                            0.0,
                        ],
                    ):

                        service.retrieve(
                            "controlled query",
                            n_results=3,
                        )

                except Exception as exc:

                    error = exc

                self.assert_true(
                    isinstance(
                        error,
                        FileNotFoundError,
                    ),
                    (
                        "C2-T3 missing runtime store "
                        "must raise FileNotFoundError"
                    ),
                )

                self.assert_true(
                    not missing_store.exists(),
                    (
                        "C2-T3 runtime retrieval must "
                        "not create a Chroma store"
                    ),
                )

            self.pass_test(
                name=name,
                details=(
                    "Tutor retrieval is read-only and "
                    "cannot turn a missing store into "
                    "an empty Chroma database."
                ),
            )

        except Exception as error:

            self.fail_test(
                name=name,
                details=str(error),
            )

    def test_c2_t4_invalid_store_shapes_fail_fast(
        self,
    ) -> None:

        name = (
            "C2-T4 Invalid knowledge-store shapes fail fast"
        )

        try:

            with tempfile.TemporaryDirectory() as root:

                root_path = Path(root)

                # ---------------------------------------------
                # Empty directory
                # ---------------------------------------------

                empty_store = (
                    root_path
                    /
                    "empty_store"
                )

                empty_store.mkdir()

                empty_profile = replace(
                    CourseProfileLoader().load(
                        "electronics"
                    ),
                    chroma_path=str(
                        empty_store
                    ),
                )

                empty_error = None

                with patch(
                    "app.tutor.CourseProfileLoader",
                ) as loader_class:

                    loader_class.return_value.load.return_value = (
                        empty_profile
                    )

                    try:

                        AITutor()

                    except Exception as exc:

                        empty_error = exc

                self.assert_true(
                    isinstance(
                        empty_error,
                        FileNotFoundError,
                    ),
                    (
                        "C2-T4 empty Chroma directory "
                        "must fail startup"
                    ),
                )

                self.assert_equal(
                    list(
                        empty_store.iterdir()
                    ),
                    [],
                    (
                        "C2-T4 validation mutated "
                        "empty store directory"
                    ),
                )

                # ---------------------------------------------
                # Regular file
                # ---------------------------------------------

                file_store = (
                    root_path
                    /
                    "not_a_directory"
                )

                file_store.write_text(
                    "invalid knowledge store",
                    encoding="utf-8",
                )

                file_profile = replace(
                    CourseProfileLoader().load(
                        "electronics"
                    ),
                    chroma_path=str(
                        file_store
                    ),
                )

                file_error = None

                with patch(
                    "app.tutor.CourseProfileLoader",
                ) as loader_class:

                    loader_class.return_value.load.return_value = (
                        file_profile
                    )

                    try:

                        AITutor()

                    except Exception as exc:

                        file_error = exc

                self.assert_true(
                    isinstance(
                        file_error,
                        NotADirectoryError,
                    ),
                    (
                        "C2-T4 Chroma path pointing to "
                        "a file must raise "
                        "NotADirectoryError"
                    ),
                )

            self.pass_test(
                name=name,
                details=(
                    "Empty directories and regular-file "
                    "paths are rejected before Tutor startup."
                ),
            )

        except Exception as error:

            self.fail_test(
                name=name,
                details=str(error),
            )


    def test_c2_t5_document_path_not_runtime_dependency(
        self,
    ) -> None:

        name = (
            "C2-T5 Document path not runtime dependency"
        )

        try:

            with tempfile.TemporaryDirectory() as root:

                missing_docs = (
                    Path(root)
                    /
                    "missing_docs"
                )

                real_profile = (
                    CourseProfileLoader()
                    .load(
                        "electronics"
                    )
                )

                profile = replace(
                    real_profile,
                    document_path=str(
                        missing_docs
                    ),
                )

                tutor = None
                error = None

                with patch(
                    "app.tutor.CourseProfileLoader",
                ) as loader_class:

                    loader_class.return_value.load.return_value = (
                        profile
                    )

                    try:

                        tutor = AITutor()

                    except Exception as exc:

                        error = exc

                self.assert_true(
                    error is None,
                    (
                        "C2-T5 missing document_path "
                        "must not block runtime when "
                        "indexed store is valid"
                    ),
                )

                self.assert_true(
                    tutor is not None,
                    (
                        "C2-T5 Tutor failed despite "
                        "valid indexed knowledge store"
                    ),
                )

                self.assert_true(
                    not missing_docs.exists(),
                    (
                        "C2-T5 runtime must not create "
                        "the document source directory"
                    ),
                )

            self.pass_test(
                name=name,
                details=(
                    "Runtime availability depends on "
                    "the indexed Chroma store, not the "
                    "original document folder."
                ),
            )

        except Exception as error:

            self.fail_test(
                name=name,
                details=str(error),
            )
            
    def test_c2_t6_index_write_path_can_create_store(
        self,
    ) -> None:

        name = (
            "C2-T6 Index write path uses "
            "create-capable collection access"
        )

        try:

            fake_collection = MagicMock()

            with patch(
                "app.vector_store.get_collection",
                return_value=fake_collection,
            ) as get_collection_mock:

                with patch(
                    "app.vector_store.get_existing_collection",
                    side_effect=AssertionError(
                        "Index write path must not use "
                        "read-only collection access."
                    ),
                ) as get_existing_mock:

                    add_document_chunk(
                        chunk_id="c2-write-1",
                        text=(
                            "Controlled indexing "
                            "boundary document."
                        ),
                        embedding=[
                            0.1,
                            0.2,
                        ],
                        source="c2-test",
                        page=1,
                        chroma_path=(
                            "./controlled/"
                            "c2-write-store"
                        ),
                    )

            get_collection_mock.assert_called_once_with(
                chroma_path=(
                    "./controlled/"
                    "c2-write-store"
                ),
                collection_name=(
                    DEFAULT_COLLECTION_NAME
                ),
            )

            get_existing_mock.assert_not_called()

            fake_collection.add.assert_called_once_with(
                ids=[
                    "c2-write-1",
                ],
                documents=[
                    (
                        "Controlled indexing "
                        "boundary document."
                    ),
                ],
                embeddings=[
                    [
                        0.1,
                        0.2,
                    ],
                ],
                metadatas=[
                    {
                        "source":
                            "c2-test",

                        "page":
                            1,
                    },
                ],
            )

            self.pass_test(
                name=name,
                details=(
                    "Indexing uses the create-capable "
                    "write collection path, while the "
                    "read-only existing-collection path "
                    "is not used."
                ),
            )

        except Exception as error:

            self.fail_test(
                name=name,
                details=str(error),
            )

    def _run_c3_config_process(
        self,
        *,
        ai_provider: str,
        groq_api_key: str = "controlled-key",
        groq_model: str = "controlled-groq-model",
        ollama_model: str = "controlled-ollama-model",
    ) -> dict:

        env = os.environ.copy()

        env["AI_PROVIDER"] = ai_provider
        env["GROQ_API_KEY"] = groq_api_key
        env["GROQ_MODEL"] = groq_model
        env["OLLAMA_MODEL"] = ollama_model

        code = (
            "import json\n"
            "from app.core.config import "
            "AI_PROVIDER, ACTIVE_CHAT_MODEL\n"
            "print(json.dumps({"
            "'provider': AI_PROVIDER, "
            "'model': ACTIVE_CHAT_MODEL"
            "}))\n"
        )

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                code,
            ],
            env=env,
            capture_output=True,
            text=True,
        )

        data = None

        if result.returncode == 0:

            data = json.loads(
                result.stdout.strip()
            )

        return {
            "returncode":
                result.returncode,

            "data":
                data,

            "stdout":
                result.stdout,

            "stderr":
                result.stderr,
        }      



    def test_c3_t1_provider_normalization_and_model_alignment(
        self,
    ) -> None:

        name = (
            "C3-T1 Provider normalization and model alignment"
        )

        try:

            groq = self._run_c3_config_process(
                ai_provider="  GROQ  ",
            )

            ollama = self._run_c3_config_process(
                ai_provider="  Ollama  ",
            )

            self.assert_equal(
                groq["returncode"],
                0,
                "C3-T1 Groq config failed",
            )

            self.assert_equal(
                groq["data"],
                {
                    "provider": "groq",
                    "model": "controlled-groq-model",
                },
                (
                    "C3-T1 Groq provider/model "
                    "alignment mismatch"
                ),
            )

            self.assert_equal(
                ollama["returncode"],
                0,
                "C3-T1 Ollama config failed",
            )

            self.assert_equal(
                ollama["data"],
                {
                    "provider": "ollama",
                    "model": "controlled-ollama-model",
                },
                (
                    "C3-T1 Ollama provider/model "
                    "alignment mismatch"
                ),
            )

            self.pass_test(
                name=name,
                details=(
                    "Provider names normalize canonically "
                    "and select only their matching chat model."
                ),
            )

        except Exception as error:

            self.fail_test(
                name=name,
                details=str(error),
            )    


    def test_c3_t2_provider_factory_routes_exactly(
        self,
    ) -> None:

        name = (
            "C3-T2 Provider factory routes exactly"
        )

        try:

            for provider_name in (
                "groq",
                "ollama",
            ):

                with patch.object(
                    provider_factory,
                    "AI_PROVIDER",
                    provider_name,
                ):

                    with patch.object(
                        provider_factory,
                        "GroqProvider",
                    ) as groq_class:

                        with patch.object(
                            provider_factory,
                            "OllamaProvider",
                        ) as ollama_class:

                            result = (
                                provider_factory
                                .create_chat_provider()
                            )

                            if provider_name == "groq":

                                self.assert_true(
                                    result
                                    is groq_class.return_value,
                                    (
                                        "C3-T2 Groq provider "
                                        "was not returned"
                                    ),
                                )

                                self.assert_equal(
                                    groq_class.call_count,
                                    1,
                                    "C3-T2 Groq call count",
                                )

                                self.assert_equal(
                                    ollama_class.call_count,
                                    0,
                                    (
                                        "C3-T2 Groq selection "
                                        "touched Ollama"
                                    ),
                                )

                            else:

                                self.assert_true(
                                    result
                                    is ollama_class.return_value,
                                    (
                                        "C3-T2 Ollama provider "
                                        "was not returned"
                                    ),
                                )

                                self.assert_equal(
                                    ollama_class.call_count,
                                    1,
                                    "C3-T2 Ollama call count",
                                )

                                self.assert_equal(
                                    groq_class.call_count,
                                    0,
                                    (
                                        "C3-T2 Ollama selection "
                                        "touched Groq"
                                    ),
                                )

            self.pass_test(
                name=name,
                details=(
                    "Chat provider selection remains "
                    "strictly scoped to the configured provider."
                ),
            )

        except Exception as error:

            self.fail_test(
                name=name,
                details=str(error),
            )

    def test_c3_t3_missing_and_blank_groq_keys_fail_fast(
        self,
    ) -> None:

        name = (
            "C3-T3 Missing and blank Groq keys fail fast"
        )

        try:

            for api_key in (
                "",
                "   ",
            ):

                with patch.object(
                    groq_provider,
                    "GROQ_API_KEY",
                    api_key,
                ):

                    with patch.object(
                        groq_provider,
                        "Groq",
                    ) as groq_client:

                        error = None

                        try:

                            groq_provider.GroqProvider()

                        except Exception as exc:

                            error = exc

                        self.assert_true(
                            isinstance(
                                error,
                                ValueError,
                            ),
                            (
                                "C3-T3 invalid Groq key "
                                "must raise ValueError"
                            ),
                        )

                        self.assert_equal(
                            groq_client.call_count,
                            0,
                            (
                                "C3-T3 Groq client must not "
                                "be constructed with invalid key"
                            ),
                        )

            self.pass_test(
                name=name,
                details=(
                    "Missing and whitespace-only Groq "
                    "credentials are rejected before client creation."
                ),
            )

        except Exception as error:

            self.fail_test(
                name=name,
                details=str(error),
            )


    def test_c3_t4_unsupported_provider_fails_at_config(
        self,
    ) -> None:

        name = (
            "C3-T4 Unsupported provider fails at config"
        )

        try:

            result = self._run_c3_config_process(
                ai_provider="__unsupported__",
            )

            self.assert_true(
                result["returncode"] != 0,
                (
                    "C3-T4 unsupported provider "
                    "must fail configuration loading"
                ),
            )

            combined_output = (
                result["stdout"]
                +
                result["stderr"]
            )

            self.assert_true(
                (
                    "Unsupported AI_PROVIDER"
                    in combined_output
                ),
                (
                    "C3-T4 configuration failure "
                    "did not identify AI_PROVIDER"
                ),
            )

            self.assert_true(
                result["data"] is None,
                (
                    "C3-T4 unsupported provider "
                    "published an active model"
                ),
            )

            self.pass_test(
                name=name,
                details=(
                    "Unsupported AI_PROVIDER fails before "
                    "an Ollama or Groq model can be selected."
                ),
            )

        except Exception as error:

            self.fail_test(
                name=name,
                details=str(error),
            )

    def test_c3_t5_provider_construction_failure_never_falls_back(
        self,
    ) -> None:

        name = (
            "C3-T5 Provider construction failure never falls back"
        )

        try:

            original_error = RuntimeError(
                "controlled provider construction failure"
            )

            observed_error = None

            with patch.object(
                provider_factory,
                "AI_PROVIDER",
                "groq",
            ):

                with patch.object(
                    provider_factory,
                    "GroqProvider",
                    side_effect=original_error,
                ) as groq_class:

                    with patch.object(
                        provider_factory,
                        "OllamaProvider",
                    ) as ollama_class:

                        try:

                            (
                                provider_factory
                                .create_chat_provider()
                            )

                        except Exception as exc:

                            observed_error = exc

            self.assert_true(
                observed_error is original_error,
                (
                    "C3-T5 provider construction error "
                    "was replaced or swallowed"
                ),
            )

            self.assert_equal(
                groq_class.call_count,
                1,
                "C3-T5 Groq construction count",
            )

            self.assert_equal(
                ollama_class.call_count,
                0,
                (
                    "C3-T5 failure silently fell back "
                    "to Ollama"
                ),
            )

            self.pass_test(
                name=name,
                details=(
                    "Provider construction failures propagate "
                    "unchanged without cross-provider fallback."
                ),
            )

        except Exception as error:

            self.fail_test(
                name=name,
                details=str(error),
            )

    def test_c3_t6_embedding_provider_isolation(
        self,
    ) -> None:

        name = (
            "C3-T6 Embedding provider isolation"
        )

        try:

            with patch.object(
                provider_factory,
                "AI_PROVIDER",
                "groq",
            ):

                with patch.object(
                    provider_factory,
                    "GroqProvider",
                ) as groq_class:

                    with patch.object(
                        provider_factory,
                        "OllamaProvider",
                    ) as ollama_class:

                        result = (
                            provider_factory
                            .create_embedding_provider()
                        )

            self.assert_true(
                result
                is ollama_class.return_value,
                (
                    "C3-T6 embedding provider "
                    "must remain Ollama"
                ),
            )

            self.assert_equal(
                ollama_class.call_count,
                1,
                "C3-T6 Ollama construction count",
            )

            self.assert_equal(
                groq_class.call_count,
                0,
                (
                    "C3-T6 embedding creation "
                    "must not construct Groq"
                ),
            )

            self.pass_test(
                name=name,
                details=(
                    "Embedding infrastructure remains "
                    "independent of the selected chat provider."
                ),
            )

        except Exception as error:

            self.fail_test(
                name=name,
                details=str(error),
            )

    def _build_c4_tutor(
        self,
        course_id: str,
    ) -> AITutor:

        with patch(
            "app.tutor.ACTIVE_COURSE",
            course_id,
        ):

            return AITutor()


    def _c4_profile_signature(
        self,
        tutor: AITutor,
    ) -> dict:

        profile = tutor.course_profile

        return {
            "course_id":
                profile.course_id,

            "course_name":
                profile.course_name,

            "language":
                profile.language,

            "document_path":
                profile.document_path,

            "chroma_path":
                profile.chroma_path,

            "technical_terms":
                dict(
                    profile.technical_terms
                ),

            "embedded_claim_patterns":
                list(
                    profile.embedded_claim_patterns
                ),

            "response_style":
                asdict(
                    profile.response_style
                ),

            "response_quality":
                asdict(
                    profile.response_quality
                ),

            "metadata":
                dict(
                    profile.metadata
                ),
        }

    def test_c4_t1_cross_course_startup_order_isolated(
        self,
    ) -> None:

        name = (
            "C4-T1 Cross-course startup order isolated"
        )

        try:

            expected_e = (
                CourseProfileLoader()
                .load("electronics")
            )

            expected_m = (
                CourseProfileLoader()
                .load("mathematics")
            )

            expected_e_signature = {
                "course_id":
                    expected_e.course_id,

                "course_name":
                    expected_e.course_name,

                "language":
                    expected_e.language,

                "document_path":
                    expected_e.document_path,

                "chroma_path":
                    expected_e.chroma_path,

                "technical_terms":
                    dict(
                        expected_e.technical_terms
                    ),

                "embedded_claim_patterns":
                    list(
                        expected_e.embedded_claim_patterns
                    ),

                "response_style":
                    asdict(
                        expected_e.response_style
                    ),

                "response_quality":
                    asdict(
                        expected_e.response_quality
                    ),

                "metadata":
                    dict(
                        expected_e.metadata
                    ),
            }

            expected_m_signature = {
                "course_id":
                    expected_m.course_id,

                "course_name":
                    expected_m.course_name,

                "language":
                    expected_m.language,

                "document_path":
                    expected_m.document_path,

                "chroma_path":
                    expected_m.chroma_path,

                "technical_terms":
                    dict(
                        expected_m.technical_terms
                    ),

                "embedded_claim_patterns":
                    list(
                        expected_m.embedded_claim_patterns
                    ),

                "response_style":
                    asdict(
                        expected_m.response_style
                    ),

                "response_quality":
                    asdict(
                        expected_m.response_quality
                    ),

                "metadata":
                    dict(
                        expected_m.metadata
                    ),
            }

            # electronics -> mathematics -> electronics

            e1 = self._build_c4_tutor(
                "electronics"
            )

            m1 = self._build_c4_tutor(
                "mathematics"
            )

            e2 = self._build_c4_tutor(
                "electronics"
            )

            self.assert_equal(
                self._c4_profile_signature(e1),
                expected_e_signature,
                "C4-T1 first electronics profile",
            )

            self.assert_equal(
                self._c4_profile_signature(m1),
                expected_m_signature,
                "C4-T1 mathematics profile",
            )

            self.assert_equal(
                self._c4_profile_signature(e2),
                expected_e_signature,
                "C4-T1 second electronics profile",
            )

            # Reverse order as well.

            m2 = self._build_c4_tutor(
                "mathematics"
            )

            e3 = self._build_c4_tutor(
                "electronics"
            )

            m3 = self._build_c4_tutor(
                "mathematics"
            )

            self.assert_equal(
                self._c4_profile_signature(m2),
                expected_m_signature,
                "C4-T1 reverse first mathematics",
            )

            self.assert_equal(
                self._c4_profile_signature(e3),
                expected_e_signature,
                "C4-T1 reverse electronics",
            )

            self.assert_equal(
                self._c4_profile_signature(m3),
                expected_m_signature,
                "C4-T1 reverse second mathematics",
            )

            self.pass_test(
                name=name,
                details=(
                    "Tutor startup order does not change "
                    "the active course profile or nested "
                    "course configuration."
                ),
            )

        except Exception as error:

            self.fail_test(
                name=name,
                details=str(error),
            )

    def test_c4_t2_course_profile_instances_are_isolated(
        self,
    ) -> None:

        name = (
            "C4-T2 CourseProfile instances are isolated"
        )

        try:

            e1 = self._build_c4_tutor(
                "electronics"
            )

            m1 = self._build_c4_tutor(
                "mathematics"
            )

            e2 = self._build_c4_tutor(
                "electronics"
            )

            self.assert_true(
                e1.course_profile
                is not
                m1.course_profile,
                "C4-T2 cross-course profile shared",
            )

            self.assert_true(
                e1.course_profile
                is not
                e2.course_profile,
                "C4-T2 same-course profile shared",
            )

            self.assert_true(
                e1.course_profile.technical_terms
                is not
                e2.course_profile.technical_terms,
                (
                    "C4-T2 technical_terms "
                    "container shared"
                ),
            )

            self.assert_true(
                e1.course_profile.metadata
                is not
                e2.course_profile.metadata,
                (
                    "C4-T2 metadata container shared"
                ),
            )

            sentinel_term = (
                "__c4_profile_sentinel__"
            )

            sentinel_metadata = (
                "__c4_metadata_sentinel__"
            )

            e1.course_profile.technical_terms[
                sentinel_term
            ] = "controlled"

            e1.course_profile.metadata[
                sentinel_metadata
            ] = True

            self.assert_true(
                sentinel_term
                not in
                e2.course_profile.technical_terms,
                (
                    "C4-T2 nested technical-term "
                    "mutation leaked"
                ),
            )

            self.assert_true(
                sentinel_term
                not in
                m1.course_profile.technical_terms,
                (
                    "C4-T2 cross-course term "
                    "mutation leaked"
                ),
            )

            self.assert_true(
                sentinel_metadata
                not in
                e2.course_profile.metadata,
                (
                    "C4-T2 metadata mutation leaked"
                ),
            )

            self.assert_true(
                sentinel_metadata
                not in
                m1.course_profile.metadata,
                (
                    "C4-T2 cross-course metadata "
                    "mutation leaked"
                ),
            )

            self.pass_test(
                name=name,
                details=(
                    "Each Tutor owns an independent "
                    "CourseProfile and independent nested "
                    "course configuration containers."
                ),
            )

        except Exception as error:

            self.fail_test(
                name=name,
                details=str(error),
            )

    def test_c4_t3_course_bound_services_remain_isolated(
        self,
    ) -> None:

        name = (
            "C4-T3 Course-bound services remain isolated"
        )

        try:

            tutors = {
                "electronics":
                    self._build_c4_tutor(
                        "electronics"
                    ),

                "mathematics":
                    self._build_c4_tutor(
                        "mathematics"
                    ),
            }

            service_names = (
                "knowledge_service",
                "embedded_claim_term_guard",
                "language_consistency_guard",
                "deterministic_response_repair",
            )

            for course_id, tutor in tutors.items():

                for service_name in service_names:

                    service = getattr(
                        tutor,
                        service_name,
                    )

                    bound_profile = getattr(
                        service,
                        "course_profile",
                        None,
                    )

                    self.assert_true(
                        bound_profile
                        is
                        tutor.course_profile,
                        (
                            "C4-T3 "
                            f"{course_id}/"
                            f"{service_name} "
                            "not bound to Tutor profile"
                        ),
                    )

                    self.assert_equal(
                        bound_profile.course_id,
                        course_id,
                        (
                            "C4-T3 service course "
                            "identity mismatch"
                        ),
                    )

            electronics = tutors[
                "electronics"
            ]

            mathematics = tutors[
                "mathematics"
            ]

            self.assert_true(
                (
                    electronics
                    .knowledge_service
                    ._get_chroma_path()
                )
                !=
                (
                    mathematics
                    .knowledge_service
                    ._get_chroma_path()
                ),
                (
                    "C4-T3 course-bound knowledge "
                    "paths are not isolated"
                ),
            )

            self.pass_test(
                name=name,
                details=(
                    "Knowledge and validation/repair "
                    "services remain pinned to their "
                    "own Tutor CourseProfile."
                ),
            )

        except Exception as error:

            self.fail_test(
                name=name,
                details=str(error),
            )

    def test_c4_t4_runtime_state_is_not_shared(
        self,
    ) -> None:

        name = (
            "C4-T4 Runtime state is not shared"
        )

        try:

            electronics = (
                self._build_c4_tutor(
                    "electronics"
                )
            )

            mathematics = (
                self._build_c4_tutor(
                    "mathematics"
                )
            )

            self.assert_true(
                electronics.memory
                is not
                mathematics.memory,
                "C4-T4 ConversationMemory shared",
            )

            self.assert_true(
                electronics.state
                is not
                mathematics.state,
                "C4-T4 TutorState shared",
            )

            electronics.memory.messages.append(
                {
                    "role": "user",
                    "content": (
                        "__c4_memory_sentinel__"
                    ),
                }
            )

            electronics.original_question = (
                "__c4_question_sentinel__"
            )

            electronics.waiting_for_response = True
            electronics.state.turn_count = 99

            self.assert_equal(
                len(
                    mathematics.memory.messages
                ),
                0,
                (
                    "C4-T4 memory mutation "
                    "crossed Tutor instances"
                ),
            )

            self.assert_true(
                mathematics.original_question
                is None,
                (
                    "C4-T4 original_question "
                    "crossed Tutor instances"
                ),
            )

            self.assert_equal(
                mathematics.waiting_for_response,
                False,
                (
                    "C4-T4 waiting state "
                    "crossed Tutor instances"
                ),
            )

            self.assert_equal(
                mathematics.state.turn_count,
                0,
                (
                    "C4-T4 learner turn state "
                    "crossed Tutor instances"
                ),
            )

            self.pass_test(
                name=name,
                details=(
                    "Conversation, learner state, and "
                    "active-sequence state remain "
                    "instance-local across courses."
                ),
            )

        except Exception as error:

            self.fail_test(
                name=name,
                details=str(error),
            )

    def test_c4_t5_existing_instance_stays_course_pinned(
        self,
    ) -> None:

        name = (
            "C4-T5 Existing instance stays course pinned"
        )

        try:

            electronics = (
                self._build_c4_tutor(
                    "electronics"
                )
            )

            before_profile = (
                self._c4_profile_signature(
                    electronics
                )
            )

            before_path = (
                electronics
                .knowledge_service
                ._get_chroma_path()
            )

            mathematics = (
                self._build_c4_tutor(
                    "mathematics"
                )
            )

            after_profile = (
                self._c4_profile_signature(
                    electronics
                )
            )

            after_path = (
                electronics
                .knowledge_service
                ._get_chroma_path()
            )

            self.assert_equal(
                before_profile,
                after_profile,
                (
                    "C4-T5 existing electronics "
                    "profile changed after "
                    "mathematics startup"
                ),
            )

            self.assert_equal(
                before_path,
                after_path,
                (
                    "C4-T5 existing electronics "
                    "knowledge path changed"
                ),
            )

            self.assert_equal(
                electronics.course_profile.course_id,
                "electronics",
                (
                    "C4-T5 electronics Tutor "
                    "was rebound"
                ),
            )

            self.assert_equal(
                mathematics.course_profile.course_id,
                "mathematics",
                (
                    "C4-T5 mathematics Tutor "
                    "course mismatch"
                ),
            )

            self.pass_test(
                name=name,
                details=(
                    "Creating a Tutor for another course "
                    "cannot rewrite an already-running "
                    "Tutor's course binding."
                ),
            )

        except Exception as error:

            self.fail_test(
                name=name,
                details=str(error),
            )

    # =====================================================
    # Run
    # =====================================================

    def run(self) -> None:

        print()
        print(
            "Step 16.22 Runtime Reliability "
            "Regression Suite"
        )
        print()

        self.test_runtime_t1_tutor_generation_failure()
        self.test_runtime_t2_evaluator_failure()
        self.test_runtime_t3_repair_provider_failure()
        self.test_runtime_t4_c3_provider_failure()
        self.test_runtime_t5_evidence_selector_provider_failure()
        self.test_runtime_t6_grounding_validator_provider_failure()
        self.test_runtime_t7_guiding_validator_provider_failure()
        self.test_runtime_t8_legacy_pedagogical_provider_failure()
        self.test_runtime_t9_response_mode_pedagogical_provider_failure()
        self.test_runtime_t10_relevance_provider_failure_boundary()
        self.test_runtime_t11_knowledge_retrieval_failure_boundary()
        self.test_b1_t1_hard_session_recovery_integrity()
        self.test_b2_t1_exact_retry_reuses_evaluation()
        self.test_b2_t2_different_answer_is_new_attempt()
        self.test_b2_t3_repeated_failure_rearms_retry()
        self.test_b2_t4_hard_reset_clears_retry_marker()

        self.test_b3_t1_failed_turn_does_not_persist()
        self.test_b3_t2_exact_retry_commits_once()
        self.test_b3_t3_repeated_failures_commit_once()
        self.test_b3_t4_out_of_course_commit_policy()

        self.test_c1_t1_valid_cross_course_startup_identity()
        self.test_c1_t2_unknown_course_fails_before_knowledge_service()
        self.test_c1_t3_invalid_course_profiles_fail_fast()
        self.test_c1_t4_course_identity_mismatch_rejected()

        self.test_c2_t1_real_course_stores_startup()
        self.test_c2_t2_missing_store_fails_without_creation()
        self.test_c2_t3_runtime_read_never_autocreates_store()
        self.test_c2_t4_invalid_store_shapes_fail_fast()
        self.test_c2_t5_document_path_not_runtime_dependency()
        self.test_c2_t6_index_write_path_can_create_store()

        self.test_c3_t1_provider_normalization_and_model_alignment()
        self.test_c3_t2_provider_factory_routes_exactly()
        self.test_c3_t3_missing_and_blank_groq_keys_fail_fast()
        self.test_c3_t4_unsupported_provider_fails_at_config()
        self.test_c3_t5_provider_construction_failure_never_falls_back()
        self.test_c3_t6_embedding_provider_isolation()

        self.test_c4_t1_cross_course_startup_order_isolated()
        self.test_c4_t2_course_profile_instances_are_isolated()
        self.test_c4_t3_course_bound_services_remain_isolated()
        self.test_c4_t4_runtime_state_is_not_shared()
        self.test_c4_t5_existing_instance_stays_course_pinned()
        
        for result in self.results:

            status = (
                "PASS"
                if result.passed
                else "FAIL"
            )

            print(
                f"[{status}] "
                f"{result.name}"
            )

            if result.details:

                print(
                    f"       {result.details}"
                )

        passed = sum(
            result.passed
            for result in self.results
        )

        failed = (
            len(self.results)
            - passed
        )

        print()
        print("SUMMARY")
        print(
            f"Passed: {passed}"
        )
        print(
            f"Failed: {failed}"
        )

        if failed:

            raise SystemExit(1)


if __name__ == "__main__":

    RegressionSuite16_22().run()