from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

from app.tutor import AITutor

from app.services.grounding_validation_models import (
    GroundingValidationResult,
)

from app.services.guiding_question_grounding_models import (
    GuidingQuestionGroundingResult,
)

from app.services.repair_evidence_selection_models import (
    RepairEvidenceSelectionResult,
)
@dataclass
class E2ETestResult:
    name: str
    passed: bool
    details: str = ""


class RegressionSuite16_21:

    def __init__(self) -> None:

        self.results: list[E2ETestResult] = []

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

    # =====================================================
    # Result helpers
    # =====================================================

    def pass_test(
        self,
        name: str,
        details: str,
    ) -> None:

        self.results.append(
            E2ETestResult(
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
            E2ETestResult(
                name=name,
                passed=False,
                details=details,
            )
        )

    # =====================================================
    # Controlled grounded Tutor
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

        tutor.knowledge_service.retrieve = (
            lambda query, n_results=5:
            knowledge_result
        )

        tutor.relevance_gate.evaluate = (
            lambda query, knowledge_result:
            SimpleNamespace(
                is_relevant=True,
                confidence=1.0,
                reason=(
                    "Controlled E2E course relevance."
                ),
            )
        )
        

        tutor.grounding_guard.evaluate = (
            lambda result, relevance:
            SimpleNamespace(
                has_knowledge=True,
                reason=(
                    "Controlled E2E course knowledge "
                    "is available."
                ),
            )
        )

        return tutor

    # =====================================================
    # E2E-T1
    # Normal grounded guiding question
    # =====================================================

    def test_e2e_t1_normal_grounded_guiding_question(
        self,
    ) -> None:

        tutor = self.build_grounded_tutor()

        generated_response = (
            "คุณคิดว่าทรานซิสเตอร์ NPN "
            "มีขั้วหลักอะไรบ้าง?"
        )

        # The semantic guiding-question validator is an
        # LLM boundary owned by frozen Step 16.20.
        #
        # For deterministic E2E orchestration testing,
        # fix its result while keeping the complete
        # AITutor.respond() pipeline active.
        tutor.guiding_question_grounding_validator.validate = (
            lambda response, knowledge_context:
            GuidingQuestionGroundingResult(
                status="supported",
                reason=(
                    "Controlled E2E guiding question "
                    "is answerable from course knowledge."
                ),
                issues=(),
            )
        )

        with patch(
            "app.tutor.chat_with_ai",
            return_value=generated_response,
        ) as generation:

            answer = tutor.respond(
                "ทรานซิสเตอร์ NPN "
                "มีโครงสร้างอย่างไร"
            )

        self.assert_equal(
            generation.call_count,
            1,
            "Tutor generation call count",
        )

        self.assert_equal(
            tutor.last_generated_answer,
            generated_response,
            "Initial generated answer",
        )

        self.assert_equal(
            answer,
            generated_response,
            "Final Tutor answer",
        )

        self.assert_true(
            (
                tutor.last_guiding_question_grounding
                is not None
            ),
            (
                "Guiding-question grounding telemetry "
                "was not recorded."
            ),
        )

        self.assert_equal(
            (
                tutor.last_guiding_question_grounding
                .status
            ),
            "supported",
            "Guiding-question status",
        )

        self.assert_true(
            not tutor.last_repair_failed,
            (
                "Normal grounded guiding question "
                "unexpectedly entered failed repair."
            ),
        )

        self.pass_test(
            "E2E-T1 Normal grounded guiding question",
            (
                "A normal course question reaches "
                "generation, grounding, guiding-question "
                "validation, and final acceptance without "
                "unnecessary repair."
            ),
        )

    # =====================================================
    # E2E-T2
    # Out-of-course fail-safe
    # =====================================================

    def test_e2e_t2_out_of_course_fail_safe(
        self,
    ) -> None:

        tutor = self.build_grounded_tutor()

        tutor.relevance_gate.evaluate = (
            lambda query, knowledge_result:
            SimpleNamespace(
                is_relevant=False,
                confidence=1.0,
                reason=(
                    "Controlled E2E out-of-course query."
                ),
            )
        )

        tutor.grounding_guard.evaluate = (
            lambda result, relevance:
            SimpleNamespace(
                has_knowledge=False,
                reason=(
                    "Controlled E2E out-of-course query "
                    "has no grounded course knowledge."
                ),
            )
        )

        with patch(
            "app.tutor.chat_with_ai",
        ) as generation:

            answer = tutor.respond(
                "อินทิกรัลไม่จำกัดเขตคืออะไร"
            )

        self.assert_equal(
            generation.call_count,
            0,
            (
                "Out-of-course request must be blocked "
                "before Tutor generation"
            ),
        )

        self.assert_true(
            tutor.last_relevance_result is not None,
            "Relevance telemetry missing.",
        )

        self.assert_equal(
            tutor.last_relevance_result.is_relevant,
            False,
            "Out-of-course relevance result",
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
                "Out-of-course path did not return "
                "a safe fallback."
            ),
        )

        self.assert_true(
            tutor.last_generated_answer is None,
            (
                "Out-of-course request unexpectedly "
                "produced a Tutor LLM answer."
            ),
        )

        self.pass_test(
            "E2E-T2 Out-of-course fail-safe",
            (
                "An out-of-course question is blocked "
                "before Tutor generation and returns "
                "the deterministic course fallback."
            ),
        )

    # =====================================================
    # E2E-T3
    # Follow-up → answer_then_guide
    # =====================================================

    def test_e2e_t3_follow_up_answer_first(
        self,
    ) -> None:

        tutor = self.build_grounded_tutor()

        # Simulate an active tutoring sequence.
        tutor.original_question = (
            "ทรานซิสเตอร์ NPN "
            "มีโครงสร้างอย่างไร"
        )

        tutor.waiting_for_response = True

        follow_up_response = (
            "Base เป็นหนึ่งในสามขั้วหลักของ"
            "ทรานซิสเตอร์ NPN ได้แก่ "
            "Base, Collector และ Emitter"
        )

        # The follow-up answer contains a factual
        # statement, therefore semantic grounding
        # belongs to the frozen Step 16.20 boundary.
        #
        # Fix only that LLM boundary so this test can
        # observe E2E intent → routing → response mode
        # → final-answer behavior deterministically.
        def controlled_grounding_validation(
            response: str,
            knowledge_context: str,
            task_name: str = "response_validator",
        ) -> GroundingValidationResult:

            return GroundingValidationResult(
                status="supported",
                confidence=1.0,
                reason=(
                    "Controlled E2E factual response "
                    "is supported by course knowledge."
                ),
                issues=[],
            )

        tutor.response_grounding_validator.validate = (
            controlled_grounding_validation
        )

        with patch(
            "app.tutor.chat_with_ai",
            return_value=follow_up_response,
        ) as generation:

            answer = tutor.respond(
                "แล้ว Base คือขั้วอะไร?"
            )

        self.assert_equal(
            generation.call_count,
            1,
            "Follow-up Tutor generation call count",
        )

        self.assert_true(
            tutor.last_learner_turn_intent
            is not None,
            "Follow-up learner intent missing.",
        )

        self.assert_equal(
            tutor.last_learner_turn_intent.intent,
            "follow_up_question",
            "Follow-up learner intent",
        )

        self.assert_true(
            tutor.last_tutoring_response_mode
            is not None,
            "Follow-up response mode missing.",
        )

        self.assert_equal(
            tutor.last_tutoring_response_mode.mode,
            "answer_then_guide",
            "Follow-up response mode",
        )

        self.assert_true(
            (
                tutor.last_tutoring_response_mode
                .direct_answer_required
            ),
            (
                "Follow-up response mode must require "
                "a direct answer."
            ),
        )

        self.assert_equal(
            (
                tutor.last_tutoring_response_mode
                .guiding_question_policy
            ),
            "optional",
            "Follow-up guiding-question policy",
        )

        self.assert_equal(
            answer,
            follow_up_response,
            "Follow-up final answer",
        )

        self.assert_true(
            not tutor.last_repair_failed,
            (
                "Valid answer-first follow-up "
                "unexpectedly failed repair."
            ),
        )

        self.pass_test(
            "E2E-T3 Follow-up answer-first behavior",
            (
                "A learner follow-up bypasses normal "
                "answer evaluation, selects "
                "answer_then_guide mode, and returns "
                "the direct grounded response."
            ),
        )


    # =====================================================
    # E2E-T4
    # Clarification → direct_clarification
    # =====================================================

    def test_e2e_t4_clarification_direct_answer(
        self,
    ) -> None:

        tutor = self.build_grounded_tutor()

        # Simulate an active tutoring sequence.
        tutor.original_question = (
            "ทรานซิสเตอร์ NPN "
            "มีโครงสร้างอย่างไร"
        )

        tutor.waiting_for_response = True

        clarification_response = (
            "Base คือหนึ่งในสามขั้วหลักของ"
            "ทรานซิสเตอร์ NPN ร่วมกับ "
            "Collector และ Emitter"
        )

        # Controlled factual-grounding boundary.
        tutor.response_grounding_validator.validate = (
            lambda *args, **kwargs:
            GroundingValidationResult(
                status="supported",
                confidence=1.0,
                reason=(
                    "Controlled E2E clarification "
                    "is supported by course knowledge."
                ),
                issues=[],
            )
        )

        # Step 16.19 already owns semantic response-mode
        # validation. Keep this E2E test focused on
        # orchestration rather than provider variability.
        tutor.response_mode_pedagogical_validator.validate = (
            lambda *args, **kwargs:
            SimpleNamespace(
                status="valid",
                confidence=1.0,
                reason=(
                    "Controlled E2E clarification "
                    "satisfies direct-answer pedagogy."
                ),
                issues=[],
            )
        )

        with patch(
            "app.tutor.chat_with_ai",
            return_value=clarification_response,
        ) as generation:

            answer = tutor.respond(
                "ช่วยอธิบายคำว่า Base "
                "ให้ชัดเจนอีกครั้ง"
            )

        self.assert_equal(
            generation.call_count,
            1,
            "Clarification Tutor generation call count",
        )

        self.assert_true(
            tutor.last_learner_turn_intent
            is not None,
            "Clarification learner intent missing.",
        )

        self.assert_equal(
            tutor.last_learner_turn_intent.intent,
            "clarification_question",
            "Clarification learner intent",
        )

        self.assert_true(
            tutor.last_tutoring_response_mode
            is not None,
            "Clarification response mode missing.",
        )

        self.assert_equal(
            tutor.last_tutoring_response_mode.mode,
            "direct_clarification",
            "Clarification response mode",
        )

        self.assert_true(
            (
                tutor.last_tutoring_response_mode
                .direct_answer_required
            ),
            (
                "Clarification mode must require "
                "a direct answer."
            ),
        )

        self.assert_equal(
            (
                tutor.last_tutoring_response_mode
                .guiding_question_policy
            ),
            "optional",
            "Clarification guiding-question policy",
        )

        self.assert_equal(
            answer,
            clarification_response,
            "Clarification final answer",
        )

        self.assert_true(
            not tutor.last_repair_failed,
            (
                "Valid direct clarification "
                "unexpectedly failed repair."
            ),
        )

        self.pass_test(
            "E2E-T4 Clarification direct answer",
            (
                "An explicit clarification request "
                "selects direct_clarification mode "
                "and returns a direct grounded answer."
            ),
        )

    # =====================================================
    # E2E-T5
    # Topic change → fresh retrieval isolation
    # =====================================================

    def test_e2e_t5_topic_change_retrieval_isolation(
        self,
    ) -> None:

        tutor = self.build_grounded_tutor()

        # Previous active topic.
        tutor.original_question = (
            "ทรานซิสเตอร์ NPN "
            "มีโครงสร้างอย่างไร"
        )

        tutor.waiting_for_response = True

        retrieval_queries = []

        knowledge_result = SimpleNamespace(
            query="controlled diode query",
            context=(
                "A diode has two terminals and "
                "contains a p-n junction."
            ),
            sources=[],
            citations=[],
        )

        def controlled_retrieve(
            query,
            n_results=5,
        ):

            retrieval_queries.append(
                query
            )

            return knowledge_result

        tutor.knowledge_service.retrieve = (
            controlled_retrieve
        )

        topic_response = (
            "คุณคิดว่าไดโอดมีขั้วหลัก "
            "กี่ขั้ว?"
        )

        tutor.guiding_question_grounding_validator.validate = (
            lambda response, knowledge_context:
            GuidingQuestionGroundingResult(
                status="supported",
                reason=(
                    "Controlled E2E diode question "
                    "is answerable from course knowledge."
                ),
                issues=(),
            )
        )

        new_topic_message = (
            "ขอถามอีกเรื่อง "
            "ไดโอดมีโครงสร้างอย่างไร?"
        )

        with patch(
            "app.tutor.chat_with_ai",
            return_value=topic_response,
        ) as generation:

            answer = tutor.respond(
                new_topic_message
            )

        self.assert_equal(
            generation.call_count,
            1,
            "Topic-change Tutor generation call count",
        )

        self.assert_true(
            tutor.last_learner_turn_intent
            is not None,
            "Topic-change learner intent missing.",
        )

        self.assert_equal(
            tutor.last_learner_turn_intent.intent,
            "topic_change",
            "Topic-change learner intent",
        )

        self.assert_true(
            tutor.last_learner_turn_routing
            is not None,
            "Topic-change routing telemetry missing.",
        )

        self.assert_equal(
            tutor.last_learner_turn_routing.route,
            "topic_change",
            "Topic-change route",
        )

        self.assert_true(
            tutor.last_tutoring_response_mode
            is not None,
            "Topic-change response mode missing.",
        )

        self.assert_equal(
            tutor.last_tutoring_response_mode.mode,
            "new_topic_scaffold",
            "Topic-change response mode",
        )

        self.assert_equal(
            len(retrieval_queries),
            1,
            "Topic-change retrieval call count",
        )

        retrieval_query = (
            retrieval_queries[0]
        )

        self.assert_true(
            "ไดโอด" in retrieval_query,
            (
                "Topic-change retrieval query does "
                "not contain the new topic."
            ),
        )

        self.assert_true(
            "NPN" not in retrieval_query,
            (
                "Previous NPN topic leaked into "
                "the new retrieval query."
            ),
        )

        self.assert_equal(
            tutor.last_retrieval_query,
            retrieval_query,
            "Topic-change retrieval telemetry",
        )

        self.assert_equal(
            answer,
            topic_response,
            "Topic-change final answer",
        )

        self.assert_true(
            not tutor.last_repair_failed,
            (
                "Valid new-topic scaffold "
                "unexpectedly failed repair."
            ),
        )

        self.pass_test(
            "E2E-T5 Topic-change retrieval isolation",
            (
                "An explicit topic change starts a "
                "new-topic scaffold and retrieves from "
                "the new learner topic without leaking "
                "the previous NPN question."
            ),
        )

    # =====================================================
    # E2E-T6
    # Unsupported factual generation
    # → verified evidence
    # → repair
    # → safe final answer
    # =====================================================

    def test_e2e_t6_unsupported_factual_repair(
        self,
    ) -> None:

        tutor = self.build_grounded_tutor()

        # Keep the turn in a response mode where a direct
        # factual explanation is appropriate.
        tutor.original_question = (
            "ทรานซิสเตอร์ NPN "
            "มีโครงสร้างอย่างไร"
        )

        tutor.waiting_for_response = True

        learner_message = (
            "ช่วยอธิบายรอยต่อ Base-Emitter "
            "ให้ชัดเจนอีกครั้ง"
        )

        unsafe_initial = (
            "รอยต่อ Base-Emitter ของทรานซิสเตอร์ "
            "NPN ต้องใช้แรงดันประมาณ 0.7 V "
            "จึงจะทำงาน"
        )

        safe_repair = (
            "รอยต่อ Base-Emitter (B-E) "
            "สามารถอยู่ในสภาวะ forward biased "
            "หรือ reverse biased ได้"
        )

        verified_quote = (
            "The Base-Emitter junction may be "
            "forward biased or reverse biased."
        )

        validation_calls = []

        def controlled_grounding_validation(
            response: str,
            knowledge_context: str,
            task_name: str = "response_validator",
        ) -> GroundingValidationResult:

            validation_calls.append(
                {
                    "response": response,
                    "knowledge_context": (
                        knowledge_context
                    ),
                    "task_name": task_name,
                }
            )

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

            if response == safe_repair:

                return GroundingValidationResult(
                    status="supported",
                    confidence=1.0,
                    reason=(
                        "The repaired response is "
                        "supported by verified course "
                        "evidence."
                    ),
                    issues=[],
                )

            raise AssertionError(
                "Unexpected response sent to grounding "
                f"validator: {response!r}"
            )

        tutor.response_grounding_validator.validate = (
            controlled_grounding_validation
        )

        # -------------------------------------------------
        # Verified evidence boundary
        # -------------------------------------------------

        evidence_calls = []

        def controlled_evidence_selection(
            learner_message: str,
            knowledge_context: str,
            validation_issues=None,
        ) -> RepairEvidenceSelectionResult:

            evidence_calls.append(
                {
                    "learner_message": (
                        learner_message
                    ),
                    "knowledge_context": (
                        knowledge_context
                    ),
                    "validation_issues": (
                        validation_issues
                    ),
                }
            )

            return RepairEvidenceSelectionResult(
                status="verified",
                evidence_quotes=(
                    verified_quote,
                ),
                reason=(
                    "Controlled E2E exact course "
                    "evidence selected."
                ),
                issues=(),
            )

        tutor.repair_evidence_selector.select = (
            controlled_evidence_selection
        )

        # -------------------------------------------------
        # Repair generation boundary
        # -------------------------------------------------

        repair_calls = []

        def controlled_repair(
            *args,
            **kwargs,
        ):

            repair_calls.append(
                kwargs
            )

            return safe_repair

        tutor.response_mode_repair_service.repair = (
            controlled_repair
        )

        # -------------------------------------------------
        # Semantic pedagogy is already covered by 16.19.
        # Keep T6 focused on factual-repair orchestration.
        # -------------------------------------------------

        tutor.response_mode_pedagogical_validator.validate = (
            lambda *args, **kwargs:
            SimpleNamespace(
                status="valid",
                confidence=1.0,
                reason=(
                    "Controlled E2E direct response "
                    "is pedagogically valid."
                ),
                issues=[],
            )
        )

        with patch(
            "app.tutor.chat_with_ai",
            return_value=unsafe_initial,
        ) as generation:

            answer = tutor.respond(
                learner_message
            )

        # -------------------------------------------------
        # Initial generation
        # -------------------------------------------------

        self.assert_equal(
            generation.call_count,
            1,
            "Initial Tutor generation call count",
        )

        self.assert_equal(
            tutor.last_generated_answer,
            unsafe_initial,
            "Initial unsafe generated answer",
        )

        self.assert_true(
            tutor.last_grounding_validation
            is not None,
            "Initial grounding telemetry missing.",
        )

        self.assert_true(
            (
                tutor.last_grounding_validation.status
                != "supported"
            ),
            (
                "Unsupported initial factual answer "
                "was incorrectly accepted."
            ),
        )

        # -------------------------------------------------
        # Verified evidence preparation
        # -------------------------------------------------

        self.assert_equal(
            len(
                evidence_calls
            ),
            1,
            "Verified evidence selection call count",
        )

        self.assert_true(
            tutor.last_repair_evidence_selection
            is not None,
            (
                "Repair evidence selection telemetry "
                "missing."
            ),
        )

        self.assert_equal(
            (
                tutor
                .last_repair_evidence_selection
                .status
            ),
            "verified",
            "Repair evidence status",
        )

        self.assert_true(
            (
                tutor.last_repair_evidence_context
                is not None
            ),
            "Verified repair context missing.",
        )

        self.assert_true(
            (
                verified_quote
                in tutor.last_repair_evidence_context
            ),
            (
                "Verified exact evidence was not "
                "placed in the repair context."
            ),
        )

        # -------------------------------------------------
        # Repair generation
        # -------------------------------------------------

        self.assert_equal(
            len(
                repair_calls
            ),
            1,
            "Repair generation call count",
        )

        repair_call = (
            repair_calls[0]
        )

        self.assert_equal(
            repair_call.get(
                "original_response"
            ),
            unsafe_initial,
            "Repair original response",
        )

        self.assert_equal(
            repair_call.get(
                "knowledge_context"
            ),
            tutor.last_repair_evidence_context,
            (
                "Repair must use the verified "
                "evidence context."
            ),
        )

        # -------------------------------------------------
        # Repair revalidation
        # -------------------------------------------------

        self.assert_equal(
            tutor.last_repair_candidate,
            safe_repair,
            "Repair candidate telemetry",
        )

        self.assert_true(
            tutor.last_repair_validation
            is not None,
            "Repair grounding telemetry missing.",
        )

        self.assert_equal(
            tutor.last_repair_validation.status,
            "supported",
            "Repair grounding status",
        )

        # Initial + repaired response must both have
        # passed through semantic grounding validation.
        self.assert_equal(
            len(
                validation_calls
            ),
            2,
            "Grounding validation call count",
        )

        # -------------------------------------------------
        # Final acceptance
        # -------------------------------------------------

        self.assert_equal(
            answer,
            safe_repair,
            "Final repaired Tutor answer",
        )

        self.assert_true(
            tutor.last_response_repaired,
            (
                "Accepted repaired response was not "
                "marked as repaired."
            ),
        )

        self.assert_true(
            not tutor.last_repair_failed,
            (
                "Successful factual repair was "
                "incorrectly marked as failed."
            ),
        )

        self.assert_true(
            answer != unsafe_initial,
            (
                "Unsupported initial answer leaked "
                "to the learner."
            ),
        )

        self.pass_test(
            "E2E-T6 Unsupported factual repair",
            (
                "An unsupported factual generation "
                "is rejected, reduced to verified "
                "course evidence, repaired once, "
                "revalidated as supported, and only "
                "the safe repaired answer reaches "
                "the learner."
            ),
        )

    # =====================================================
    # E2E-T7
    # Unsafe guiding question
    # → unsafe normal repair
    # → verified evidence
    # → C3 safe recovery
    # → accepted final answer
    # =====================================================

    def test_e2e_t7_c3_safe_guiding_recovery(
        self,
    ) -> None:

        tutor = self.build_grounded_tutor()

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

        safe_c3_recovery = (
            "คุณคิดว่าแรงดันระหว่าง Base และ "
            "Emitter (B-E) มีความสัมพันธ์กับ "
            "กระแส Collector อย่างไร?"
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

        # -------------------------------------------------
        # Grounding validation
        #
        # All three candidates are factually bounded at
        # the statement level. The defect under test is
        # specifically the requested threshold inside the
        # guiding question.
        # -------------------------------------------------

        grounding_calls = []

        def controlled_grounding_validation(
            response: str,
            knowledge_context: str,
            task_name: str = "response_validator",
        ) -> GroundingValidationResult:

            grounding_calls.append(
                {
                    "response": response,
                    "knowledge_context": (
                        knowledge_context
                    ),
                    "task_name": task_name,
                }
            )

            return GroundingValidationResult(
                status="supported",
                confidence=1.0,
                reason=(
                    "Controlled E2E response contains "
                    "no unsupported factual assertion."
                ),
                issues=[],
            )

        tutor.response_grounding_validator.validate = (
            controlled_grounding_validation
        )

        # -------------------------------------------------
        # Guiding-question grounding
        # -------------------------------------------------

        guiding_calls = []

        def controlled_guiding_validation(
            response: str,
            knowledge_context: str,
        ) -> GuidingQuestionGroundingResult:

            guiding_calls.append(
                {
                    "response": response,
                    "knowledge_context": (
                        knowledge_context
                    ),
                }
            )

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

            if response == safe_c3_recovery:

                return GuidingQuestionGroundingResult(
                    status="supported",
                    reason=(
                        "The relationship requested by "
                        "the guiding question is directly "
                        "answerable from verified course "
                        "evidence."
                    ),
                    issues=(),
                )

            raise AssertionError(
                "Unexpected response sent to guiding-"
                "question validator: "
                f"{response!r}"
            )

        tutor.guiding_question_grounding_validator.validate = (
            controlled_guiding_validation
        )

        # -------------------------------------------------
        # Verified evidence selection
        # -------------------------------------------------

        evidence_calls = []

        def controlled_evidence_selection(
            learner_message: str,
            knowledge_context: str,
            validation_issues=None,
        ) -> RepairEvidenceSelectionResult:

            evidence_calls.append(
                {
                    "learner_message": (
                        learner_message
                    ),
                    "knowledge_context": (
                        knowledge_context
                    ),
                    "validation_issues": (
                        validation_issues
                    ),
                }
            )

            return RepairEvidenceSelectionResult(
                status="verified",
                evidence_quotes=(
                    verified_quote,
                ),
                reason=(
                    "Controlled E2E relation evidence "
                    "selected exactly from course "
                    "knowledge."
                ),
                issues=(),
            )

        tutor.repair_evidence_selector.select = (
            controlled_evidence_selection
        )

        # -------------------------------------------------
        # Repair generation
        #
        # Call #1 = normal repair, still unsafe.
        # Call #2 = C3 bounded recovery, safe.
        # -------------------------------------------------

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

                return safe_c3_recovery

            raise AssertionError(
                "C3 recovery attempted more than "
                "one additional repair generation."
            )

        tutor.response_mode_repair_service.repair = (
            controlled_repair
        )

        # -------------------------------------------------
        # Pedagogical semantic boundaries are already
        # covered by frozen Step 16.19.
        #
        # Keep this E2E test focused on the C3
        # orchestration and factual question boundary.
        # -------------------------------------------------

        tutor.pedagogical_response_validator.validate = (
            lambda *args, **kwargs:
            SimpleNamespace(
                status="valid",
                confidence=1.0,
                reason=(
                    "Controlled E2E guiding question "
                    "is pedagogically valid."
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
                    "Controlled E2E response-mode "
                    "pedagogy is valid."
                ),
                issues=[],
            )
        )

        # -------------------------------------------------
        # Execute one complete Tutor turn
        # -------------------------------------------------

        with patch(
            "app.tutor.chat_with_ai",
            return_value=unsafe_initial,
        ) as generation:

            answer = tutor.respond(
                learner_message
            )

        # =================================================
        # Initial generation
        # =================================================

        self.assert_equal(
            generation.call_count,
            1,
            "Initial Tutor generation call count",
        )

        self.assert_equal(
            tutor.last_generated_answer,
            unsafe_initial,
            "Initial unsafe guiding question",
        )

        self.assert_true(
            tutor.last_guiding_question_grounding
            is not None,
            (
                "Initial guiding-question validation "
                "telemetry missing."
            ),
        )

        self.assert_equal(
            (
                tutor
                .last_guiding_question_grounding
                .status
            ),
            "unsupported",
            "Initial guiding-question status",
        )

        # =================================================
        # Normal repair
        # =================================================

        self.assert_true(
            (
                tutor.last_repair_candidate
                == unsafe_normal_repair
            ),
            (
                "Normal repair candidate was not "
                "captured before C3 recovery."
            ),
        )

        self.assert_true(
            (
                tutor
                .last_repair_guiding_question_grounding
                is not None
            ),
            (
                "Normal repair guiding-question "
                "telemetry missing."
            ),
        )

        self.assert_equal(
            (
                tutor
                .last_repair_guiding_question_grounding
                .status
            ),
            "unsupported",
            (
                "Unsafe normal repair guiding-question "
                "status"
            ),
        )

        # =================================================
        # Verified evidence
        # =================================================

        self.assert_equal(
            len(
                evidence_calls
            ),
            1,
            "C3 verified evidence selection count",
        )

        self.assert_true(
            (
                tutor.last_repair_evidence_selection
                is not None
            ),
            (
                "C3 verified evidence selection "
                "telemetry missing."
            ),
        )

        self.assert_equal(
            (
                tutor
                .last_repair_evidence_selection
                .status
            ),
            "verified",
            "C3 evidence selection status",
        )

        self.assert_true(
            (
                tutor.last_repair_evidence_context
                is not None
            ),
            "C3 verified evidence context missing.",
        )

        self.assert_true(
            (
                verified_quote
                in tutor.last_repair_evidence_context
            ),
            (
                "Verified relation evidence missing "
                "from C3 context."
            ),
        )

        # =================================================
        # C3 repair invocation boundary
        # =================================================

        self.assert_equal(
            len(
                repair_calls
            ),
            2,
            (
                "Normal repair + C3 repair total "
                "call count"
            ),
        )

        normal_repair_call = (
            repair_calls[0]
        )

        c3_repair_call = (
            repair_calls[1]
        )

        self.assert_equal(
            normal_repair_call.get(
                "original_response"
            ),
            unsafe_initial,
            (
                "Normal repair must operate on "
                "the initial Tutor response."
            ),
        )

        self.assert_equal(
            c3_repair_call.get(
                "original_response"
            ),
            unsafe_normal_repair,
            (
                "C3 must operate on the rejected "
                "normal repair candidate."
            ),
        )

        self.assert_equal(
            c3_repair_call.get(
                "knowledge_context"
            ),
            tutor.last_repair_evidence_context,
            (
                "C3 must use only the verified "
                "repair evidence context."
            ),
        )

        self.assert_equal(
            c3_repair_call.get(
                "strategy_name"
            ),
            "guiding_question",
            "C3 strategy",
        )

        c3_validation = (
            c3_repair_call.get(
                "validation"
            )
        )

        self.assert_true(
            c3_validation is not None,
            "C3 validation input missing.",
        )

        self.assert_equal(
            c3_validation.status,
            "supported",
            (
                "C3 must start from a factually "
                "supported normal repair."
            ),
        )

        # =================================================
        # C3 revalidation
        # =================================================

        self.assert_true(
            (
                tutor
                .last_evidence_safe_guiding_question_grounding
                is not None
            ),
            (
                "C3 guiding-question revalidation "
                "telemetry missing."
            ),
        )

        self.assert_equal(
            (
                tutor
                .last_evidence_safe_guiding_question_grounding
                .status
            ),
            "supported",
            "C3 recovered guiding-question status",
        )

        # Make sure the safe C3 question was validated
        # against the verified evidence boundary.
        c3_guiding_calls = [
            item
            for item in guiding_calls
            if (
                item["response"]
                == safe_c3_recovery
            )
        ]

        self.assert_true(
            len(
                c3_guiding_calls
            )
            >= 1,
            (
                "C3 recovered guiding question was "
                "not validated."
            ),
        )

        self.assert_true(
            all(
                (
                    item["knowledge_context"]
                    ==
                    tutor.last_repair_evidence_context
                )
                for item in c3_guiding_calls
            ),
            (
                "Every validation of the C3 recovered "
                "guiding question must remain bounded "
                "to the verified repair evidence context."
            ),
        )

        # =================================================
        # Final acceptance
        # =================================================

        self.assert_equal(
            answer,
            safe_c3_recovery,
            "Final C3 recovered answer",
        )

        self.assert_true(
            tutor.last_evidence_safe_repair_used,
            (
                "Accepted C3 recovery was not marked "
                "as evidence-safe repair."
            ),
        )

        self.assert_true(
            tutor.last_response_repaired,
            (
                "Accepted C3 response was not marked "
                "as repaired."
            ),
        )

        self.assert_true(
            not tutor.last_repair_failed,
            (
                "Successful C3 recovery was "
                "incorrectly marked as failed."
            ),
        )

        # -------------------------------------------------
        # Unsafe-output leak checks
        # -------------------------------------------------

        self.assert_true(
            answer != unsafe_initial,
            (
                "Initial unsafe threshold question "
                "leaked to final output."
            ),
        )

        self.assert_true(
            answer != unsafe_normal_repair,
            (
                "Unsafe normal repair question "
                "leaked to final output."
            ),
        )

        self.pass_test(
            "E2E-T7 C3 safe guiding-question recovery",
            (
                "An unsafe initial guiding question "
                "and unsafe normal repair are rejected; "
                "verified evidence bounds one C3 "
                "regeneration, the recovered question "
                "passes revalidation, and only the safe "
                "C3 result reaches the learner."
            ),
        )

    # =====================================================
    # E2E-T8
    # Unsafe guiding question
    # → unsafe normal repair
    # → verified evidence
    # → unsafe C3 recovery
    # → reject
    # → deterministic fail-closed fallback
    # =====================================================

    def test_e2e_t8_c3_fail_closed(
        self,
    ) -> None:

        tutor = self.build_grounded_tutor()

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
            "จึงจะทำให้กระแส Collectorเพิ่มขึ้น?"
        )

        unsafe_c3_recovery = (
            "แรงดันระหว่าง Base และ Emitter "
            "(B-E) ต้องมีอย่างน้อยกี่โวลต์ "
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

        # -------------------------------------------------
        # Factual grounding
        #
        # The defect here is the unsupported factual
        # requirement inside each guiding question.
        # Statement-level grounding remains supported.
        # -------------------------------------------------

        grounding_calls = []

        def controlled_grounding_validation(
            response: str,
            knowledge_context: str,
            task_name: str = "response_validator",
        ) -> GroundingValidationResult:

            grounding_calls.append(
                {
                    "response": response,
                    "knowledge_context": (
                        knowledge_context
                    ),
                    "task_name": task_name,
                }
            )

            return GroundingValidationResult(
                status="supported",
                confidence=1.0,
                reason=(
                    "Controlled E2E response contains "
                    "no unsupported declarative claim."
                ),
                issues=[],
            )

        tutor.response_grounding_validator.validate = (
            controlled_grounding_validation
        )

        # -------------------------------------------------
        # Every guiding question remains unsafe.
        # -------------------------------------------------

        guiding_calls = []

        def controlled_guiding_validation(
            response: str,
            knowledge_context: str,
        ) -> GuidingQuestionGroundingResult:

            guiding_calls.append(
                {
                    "response": response,
                    "knowledge_context": (
                        knowledge_context
                    ),
                }
            )

            if response == unsafe_initial:

                return GuidingQuestionGroundingResult(
                    status="unsupported",
                    reason=(
                        "Course evidence contains no "
                        "explicit minimum B-E voltage."
                    ),
                    issues=(
                        "Unsupported exact threshold "
                        "question.",
                    ),
                )

            if response == unsafe_normal_repair:

                return GuidingQuestionGroundingResult(
                    status="unsupported",
                    reason=(
                        "Course evidence contains no "
                        "criterion for how much voltage "
                        "is sufficient."
                    ),
                    issues=(
                        "Unsupported sufficient-threshold "
                        "question.",
                    ),
                )

            if response == unsafe_c3_recovery:

                return GuidingQuestionGroundingResult(
                    status="unsupported",
                    reason=(
                        "Verified evidence still provides "
                        "no minimum B-E voltage."
                    ),
                    issues=(
                        "C3 still requests an unsupported "
                        "exact voltage threshold.",
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

        # -------------------------------------------------
        # Verified evidence selection
        # -------------------------------------------------

        evidence_calls = []

        def controlled_evidence_selection(
            learner_message: str,
            knowledge_context: str,
            validation_issues=None,
        ) -> RepairEvidenceSelectionResult:

            evidence_calls.append(
                {
                    "learner_message": learner_message,
                    "knowledge_context": (
                        knowledge_context
                    ),
                    "validation_issues": (
                        validation_issues
                    ),
                }
            )

            return RepairEvidenceSelectionResult(
                status="verified",
                evidence_quotes=(
                    verified_quote,
                ),
                reason=(
                    "Controlled E2E verified relation "
                    "evidence selected."
                ),
                issues=(),
            )

        tutor.repair_evidence_selector.select = (
            controlled_evidence_selection
        )

        # -------------------------------------------------
        # Repair generation
        #
        # #1 normal repair remains unsafe
        # #2 C3 remains unsafe
        # no third generation is allowed
        # -------------------------------------------------

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

                return unsafe_c3_recovery

            raise AssertionError(
                "Fail-closed C3 attempted more than "
                "one additional recovery generation."
            )

        tutor.response_mode_repair_service.repair = (
            controlled_repair
        )

        # -------------------------------------------------
        # Pedagogy itself is valid.
        #
        # Rejection must come from the knowledge-bounded
        # guiding-question gate, not from an unrelated
        # semantic pedagogy failure.
        # -------------------------------------------------

        tutor.pedagogical_response_validator.validate = (
            lambda *args, **kwargs:
            SimpleNamespace(
                status="valid",
                confidence=1.0,
                reason=(
                    "Controlled E2E question form is "
                    "pedagogically valid."
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
                    "Controlled E2E response-mode "
                    "pedagogy is valid."
                ),
                issues=[],
            )
        )

        # -------------------------------------------------
        # Execute one complete Tutor turn
        # -------------------------------------------------

        with patch(
            "app.tutor.chat_with_ai",
            return_value=unsafe_initial,
        ) as generation:

            answer = tutor.respond(
                learner_message
            )

        # =================================================
        # Initial candidate rejected
        # =================================================

        self.assert_equal(
            generation.call_count,
            1,
            "Initial Tutor generation call count",
        )

        self.assert_equal(
            tutor.last_generated_answer,
            unsafe_initial,
            "Initial unsafe guiding question",
        )

        self.assert_true(
            tutor.last_guiding_question_grounding
            is not None,
            (
                "Initial guiding-question telemetry "
                "missing."
            ),
        )

        self.assert_equal(
            tutor.last_guiding_question_grounding.status,
            "unsupported",
            "Initial guiding-question status",
        )

        # =================================================
        # Normal repair rejected
        # =================================================

        self.assert_equal(
            tutor.last_repair_candidate,
            unsafe_normal_repair,
            "Normal repair candidate",
        )

        self.assert_true(
            (
                tutor
                .last_repair_guiding_question_grounding
                is not None
            ),
            (
                "Normal repair guiding-question "
                "telemetry missing."
            ),
        )

        self.assert_equal(
            (
                tutor
                .last_repair_guiding_question_grounding
                .status
            ),
            "unsupported",
            "Normal repair guiding-question status",
        )

        # =================================================
        # Verified evidence boundary
        # =================================================

        self.assert_equal(
            len(
                evidence_calls
            ),
            1,
            "Verified evidence selection count",
        )

        self.assert_true(
            (
                tutor.last_repair_evidence_selection
                is not None
            ),
            (
                "Verified evidence telemetry missing."
            ),
        )

        self.assert_equal(
            (
                tutor
                .last_repair_evidence_selection
                .status
            ),
            "verified",
            "Verified evidence status",
        )

        self.assert_true(
            (
                verified_quote
                in tutor.last_repair_evidence_context
            ),
            (
                "Verified evidence missing from "
                "C3 repair context."
            ),
        )

        # =================================================
        # C3 bounded generation
        # =================================================

        self.assert_equal(
            len(
                repair_calls
            ),
            2,
            (
                "Normal repair + C3 repair total "
                "generation count"
            ),
        )

        self.assert_equal(
            (
                repair_calls[1]
                .get(
                    "original_response"
                )
            ),
            unsafe_normal_repair,
            (
                "C3 must operate on the rejected "
                "normal repair candidate."
            ),
        )

        self.assert_equal(
            (
                repair_calls[1]
                .get(
                    "knowledge_context"
                )
            ),
            tutor.last_repair_evidence_context,
            (
                "C3 must remain bounded to verified "
                "repair evidence."
            ),
        )

        # =================================================
        # Unsafe C3 rejected
        # =================================================

        c3_guiding_calls = [
            item
            for item in guiding_calls
            if (
                item["response"]
                == unsafe_c3_recovery
            )
        ]

        self.assert_true(
            len(
                c3_guiding_calls
            )
            >= 1,
            (
                "Unsafe C3 candidate was not checked "
                "by guiding-question grounding."
            ),
        )

        self.assert_true(
            all(
                (
                    item["knowledge_context"]
                    ==
                    tutor.last_repair_evidence_context
                )
                for item in c3_guiding_calls
            ),
            (
                "Unsafe C3 validation escaped the "
                "verified evidence boundary."
            ),
        )

        self.assert_true(
            (
                tutor
                .last_evidence_safe_guiding_question_grounding
                is not None
            ),
            (
                "C3 guiding-question rejection "
                "telemetry missing."
            ),
        )

        self.assert_equal(
            (
                tutor
                .last_evidence_safe_guiding_question_grounding
                .status
            ),
            "unsupported",
            "Unsafe C3 guiding-question status",
        )

        # =================================================
        # Final fail-closed behavior
        # =================================================

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
                "Fail-closed path did not produce "
                "a deterministic fallback."
            ),
        )

        self.assert_true(
            answer != unsafe_initial,
            (
                "Initial unsafe question leaked "
                "to final output."
            ),
        )

        self.assert_true(
            answer != unsafe_normal_repair,
            (
                "Unsafe normal repair leaked "
                "to final output."
            ),
        )

        self.assert_true(
            answer != unsafe_c3_recovery,
            (
                "Unsafe C3 candidate leaked "
                "to final output."
            ),
        )

        self.assert_true(
            tutor.last_repair_failed,
            (
                "Rejected unsafe C3 recovery was not "
                "marked as failed."
            ),
        )

        self.assert_equal(
            tutor.last_repair_failure_type,
            "pedagogical",
            "C3 failure type",
        )

        self.assert_true(
            not tutor.last_response_repaired,
            (
                "Failed C3 candidate was incorrectly "
                "marked as accepted repair."
            ),
        )

        self.assert_true(
            not tutor.last_evidence_safe_repair_used,
            (
                "Unsafe C3 candidate was incorrectly "
                "marked as evidence-safe accepted."
            ),
        )

        self.pass_test(
            "E2E-T8 C3 fail-closed",
            (
                "When the initial question, normal "
                "repair, and bounded C3 recovery all "
                "remain knowledge-unsafe, every unsafe "
                "candidate is rejected and only the "
                "deterministic fail-closed fallback "
                "reaches the learner."
            ),
        )
    # =====================================================
    # E2E-T9
    # Final-answer telemetry alignment
    # =====================================================

    def test_e2e_t9_final_answer_telemetry_alignment(
        self,
    ) -> None:

        tutor = self.build_grounded_tutor()

        tutor.original_question = (
            "ทรานซิสเตอร์ NPN "
            "มีโครงสร้างอย่างไร"
        )

        tutor.waiting_for_response = True

        learner_message = (
            "ช่วยอธิบายรอยต่อ Base-Emitter "
            "ให้ชัดเจนอีกครั้ง"
        )

        unsafe_initial = (
            "รอยต่อ Base-Emitter ของทรานซิสเตอร์ "
            "NPN ต้องใช้แรงดัน 0.7 V "
            "จึงจะทำงาน"
        )

        safe_repair = (
            "รอยต่อ Base-Emitter (B-E) "
            "สามารถอยู่ในสภาวะ forward biased "
            "หรือ reverse biased ได้"
        )

        verified_quote = (
            "The Base-Emitter junction may be "
            "forward biased or reverse biased."
        )

        # -------------------------------------------------
        # Initial grounding fails; repaired response passes.
        # -------------------------------------------------

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
                        "Exact 0.7 V requirement is "
                        "absent from course knowledge."
                    ),
                    issues=[
                        (
                            "Unsupported exact B-E "
                            "voltage requirement."
                        )
                    ],
                )

            if response == safe_repair:

                return GroundingValidationResult(
                    status="supported",
                    confidence=1.0,
                    reason=(
                        "Repaired response is supported "
                        "by course knowledge."
                    ),
                    issues=[],
                )

            raise AssertionError(
                "Unexpected grounding candidate: "
                f"{response!r}"
            )

        tutor.response_grounding_validator.validate = (
            controlled_grounding_validation
        )

        # -------------------------------------------------
        # Exact verified evidence
        # -------------------------------------------------

        tutor.repair_evidence_selector.select = (
            lambda learner_message,
            knowledge_context,
            validation_issues=None:
            RepairEvidenceSelectionResult(
                status="verified",
                evidence_quotes=(
                    verified_quote,
                ),
                reason=(
                    "Controlled E2E verified evidence."
                ),
                issues=(),
            )
        )

        # -------------------------------------------------
        # Successful factual repair
        # -------------------------------------------------

        tutor.response_mode_repair_service.repair = (
            lambda *args, **kwargs:
            safe_repair
        )

        tutor.response_mode_pedagogical_validator.validate = (
            lambda *args, **kwargs:
            SimpleNamespace(
                status="valid",
                confidence=1.0,
                reason=(
                    "Controlled E2E response is "
                    "pedagogically valid."
                ),
                issues=[],
            )
        )

        # -------------------------------------------------
        # Observe exactly what final-response telemetry sees.
        # Preserve the real deterministic guards.
        # -------------------------------------------------

        language_observed = []
        quality_observed = []

        original_language_evaluate = (
            tutor.language_consistency_guard.evaluate
        )

        original_quality_evaluate = (
            tutor.response_quality_guard.evaluate
        )

        def observing_language_evaluate(
            *args,
            **kwargs,
        ):

            if "response" in kwargs:

                observed_response = (
                    kwargs["response"]
                )

            else:

                observed_response = (
                    args[0]
                )

            language_observed.append(
                observed_response
            )

            return original_language_evaluate(
                *args,
                **kwargs,
            )

        def observing_quality_evaluate(
            *args,
            **kwargs,
        ):

            if args:

                observed_response = (
                    args[0]
                )

            else:

                observed_response = (
                    kwargs["response"]
                )

            quality_observed.append(
                observed_response
            )

            return original_quality_evaluate(
                *args,
                **kwargs,
            )

        tutor.language_consistency_guard.evaluate = (
            observing_language_evaluate
        )

        tutor.response_quality_guard.evaluate = (
            observing_quality_evaluate
        )

        # -------------------------------------------------
        # Execute complete turn
        # -------------------------------------------------

        with patch(
            "app.tutor.chat_with_ai",
            return_value=unsafe_initial,
        ):

            answer = tutor.respond(
                learner_message
            )

        # -------------------------------------------------
        # Repair actually happened
        # -------------------------------------------------

        self.assert_equal(
            answer,
            safe_repair,
            "Final repaired answer",
        )

        self.assert_true(
            tutor.last_response_repaired,
            (
                "T9 did not reach the successful "
                "repair path."
            ),
        )

        self.assert_true(
            not tutor.last_repair_failed,
            (
                "Successful T9 repair was marked "
                "as failed."
            ),
        )

        # -------------------------------------------------
        # Language telemetry must observe FINAL answer
        # -------------------------------------------------

        self.assert_true(
            len(
                language_observed
            )
            >= 1,
            (
                "Language consistency telemetry "
                "did not observe a response."
            ),
        )

        self.assert_equal(
            language_observed[-1],
            answer,
            (
                "Language consistency must observe "
                "the final learner-visible answer."
            ),
        )

        self.assert_true(
            (
                tutor.last_language_consistency
                is not None
            ),
            (
                "Final language telemetry missing."
            ),
        )

        # -------------------------------------------------
        # Quality telemetry must observe FINAL answer
        # -------------------------------------------------

        self.assert_true(
            len(
                quality_observed
            )
            >= 1,
            (
                "Response quality telemetry did not "
                "observe a response."
            ),
        )

        self.assert_equal(
            quality_observed[-1],
            answer,
            (
                "Response quality must observe the "
                "final learner-visible answer."
            ),
        )

        self.assert_true(
            (
                tutor.last_response_quality
                is not None
            ),
            (
                "Final response-quality telemetry "
                "missing."
            ),
        )

        # -------------------------------------------------
        # Initial unsafe candidate must not be the final
        # telemetry subject.
        # -------------------------------------------------

        self.assert_true(
            language_observed[-1]
            != unsafe_initial,
            (
                "Language telemetry remained attached "
                "to the unsafe initial generation."
            ),
        )

        self.assert_true(
            quality_observed[-1]
            != unsafe_initial,
            (
                "Quality telemetry remained attached "
                "to the unsafe initial generation."
            ),
        )

        self.pass_test(
            "E2E-T9 Final-answer telemetry alignment",
            (
                "After factual repair, language and "
                "response-quality telemetry observe "
                "the actual final learner-visible "
                "answer rather than the rejected "
                "initial generation."
            ),
        )
    # =====================================================
    # E2E-T10
    # Failed factual recovery
    # → fail closed
    # → no conversation-memory contamination
    # → tutoring sequence cleared
    # =====================================================

    def test_e2e_t10_failed_recovery_no_state_contamination(
        self,
    ) -> None:

        tutor = self.build_grounded_tutor()

        tutor.original_question = (
            "ทรานซิสเตอร์ NPN "
            "มีโครงสร้างอย่างไร"
        )

        tutor.waiting_for_response = True

        learner_message = (
            "ช่วยอธิบายแรงดัน Base-Emitter "
            "ให้ชัดเจนอีกครั้ง"
        )

        unsafe_initial = (
            "ทรานซิสเตอร์ NPN ต้องมีแรงดัน "
            "Base-Emitter เท่ากับ 0.7 V "
            "จึงจะทำงาน"
        )

        # -------------------------------------------------
        # Initial factual answer is unsupported.
        # -------------------------------------------------

        tutor.response_grounding_validator.validate = (
            lambda response,
            knowledge_context,
            task_name="response_validator":
            GroundingValidationResult(
                status="unsupported",
                confidence=1.0,
                reason=(
                    "Course knowledge contains no "
                    "exact 0.7 V requirement."
                ),
                issues=[
                    (
                        "Unsupported exact B-E "
                        "voltage requirement."
                    )
                ],
            )
        )

        # -------------------------------------------------
        # No verified evidence is available for the
        # requested exact value.
        # -------------------------------------------------

        evidence_calls = []

        def no_evidence_selection(
            learner_message: str,
            knowledge_context: str,
            validation_issues=None,
        ) -> RepairEvidenceSelectionResult:

            evidence_calls.append(
                {
                    "learner_message": (
                        learner_message
                    ),
                    "knowledge_context": (
                        knowledge_context
                    ),
                    "validation_issues": (
                        validation_issues
                    ),
                }
            )

            return RepairEvidenceSelectionResult(
                status="no_evidence",
                evidence_quotes=(),
                reason=(
                    "Course knowledge contains no "
                    "verified exact voltage evidence."
                ),
                issues=(),
            )

        tutor.repair_evidence_selector.select = (
            no_evidence_selection
        )

        # -------------------------------------------------
        # No repair generation should be possible without
        # a verified factual evidence context.
        # -------------------------------------------------

        repair_calls = []

        def forbidden_repair(
            *args,
            **kwargs,
        ):

            repair_calls.append(
                kwargs
            )

            raise AssertionError(
                "Repair generation must not run when "
                "verified factual evidence is absent."
            )

        tutor.response_mode_repair_service.repair = (
            forbidden_repair
        )

        tutor.response_mode_pedagogical_validator.validate = (
            lambda *args, **kwargs:
            SimpleNamespace(
                status="valid",
                confidence=1.0,
                reason=(
                    "Controlled E2E response form "
                    "is pedagogically valid."
                ),
                issues=[],
            )
        )

        # -------------------------------------------------
        # Observe ConversationMemory writes.
        # -------------------------------------------------

        memory_user_calls = []
        memory_assistant_calls = []

        tutor.memory.add_user_message = (
            lambda message:
            memory_user_calls.append(
                message
            )
        )

        tutor.memory.add_assistant_message = (
            lambda message:
            memory_assistant_calls.append(
                message
            )
        )

        # -------------------------------------------------
        # Execute complete turn
        # -------------------------------------------------

        with patch(
            "app.tutor.chat_with_ai",
            return_value=unsafe_initial,
        ):

            answer = tutor.respond(
                learner_message
            )

        # -------------------------------------------------
        # Evidence boundary
        # -------------------------------------------------

        self.assert_equal(
            len(
                evidence_calls
            ),
            1,
            "No-evidence selection call count",
        )

        self.assert_true(
            (
                tutor.last_repair_evidence_selection
                is not None
            ),
            (
                "No-evidence selection telemetry "
                "missing."
            ),
        )

        self.assert_equal(
            (
                tutor
                .last_repair_evidence_selection
                .status
            ),
            "no_evidence",
            "Failed recovery evidence status",
        )

        self.assert_true(
            (
                tutor.last_repair_evidence_context
                is None
            ),
            (
                "A factual repair context was created "
                "without verified evidence."
            ),
        )

        self.assert_equal(
            len(
                repair_calls
            ),
            0,
            (
                "Repair generation must be blocked "
                "without verified evidence."
            ),
        )

        # -------------------------------------------------
        # Final fail-closed result
        # -------------------------------------------------

        self.assert_true(
            tutor.last_repair_failed,
            (
                "Failed factual recovery was not "
                "marked as failed."
            ),
        )

        self.assert_true(
            not tutor.last_response_repaired,
            (
                "Failed recovery was incorrectly "
                "marked as successful repair."
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
                "Failed recovery did not return "
                "a deterministic fallback."
            ),
        )

        self.assert_true(
            answer != unsafe_initial,
            (
                "Unsafe initial generation leaked "
                "through the fail-closed path."
            ),
        )

        # -------------------------------------------------
        # ConversationMemory contamination guard
        # -------------------------------------------------

        self.assert_equal(
            len(
                memory_user_calls
            ),
            0,
            (
                "Failed recovery learner message "
                "was written to ConversationMemory."
            ),
        )

        self.assert_equal(
            len(
                memory_assistant_calls
            ),
            0,
            (
                "Fail-closed fallback was written "
                "to ConversationMemory."
            ),
        )

        # -------------------------------------------------
        # Tutoring sequence must be closed.
        # -------------------------------------------------

        self.assert_true(
            tutor.original_question is None,
            (
                "Failed recovery left a stale "
                "original question active."
            ),
        )

        self.assert_equal(
            tutor.waiting_for_response,
            False,
            (
                "Failed recovery left Tutor waiting "
                "for a response to a failed turn."
            ),
        )

        self.pass_test(
            "E2E-T10 Failed recovery state isolation",
            (
                "When factual recovery has no verified "
                "evidence, generation fails closed, "
                "the unsafe turn is not written to "
                "ConversationMemory, and the active "
                "tutoring sequence is cleared."
            ),
        )


    # =====================================================
    # 16.21B1 — MT-T1
    # Initial topic → learner answer → follow-up
    #
    # Follow-up must:
    # - bypass learner-answer evaluation
    # - become the active question
    # - retrieve from the current follow-up
    # - preserve session ConversationMemory
    # =====================================================

    def test_mt_t1_follow_up_retains_topic(
        self,
    ) -> None:

        tutor = self.build_grounded_tutor()

        initial_message = (
            "ทรานซิสเตอร์ NPN "
            "มีโครงสร้างอย่างไร"
        )

        learner_answer = (
            "มีสามขั้วคือ Base "
            "Collector และ Emitter"
        )

        follow_up_message = (
            "แล้ว Base คืออะไร?"
        )

        initial_response = (
            "คุณคิดว่าทรานซิสเตอร์ NPN "
            "มีขั้วหลักอะไรบ้าง?"
        )

        answer_turn_response = (
            "จากคำตอบของคุณ "
            "Base เป็นหนึ่งในขั้วหลักหรือไม่?"
        )

        follow_up_response = (
            "Base เป็นหนึ่งในสามขั้วหลักของ "
            "ทรานซิสเตอร์ NPN ได้แก่ Base, "
            "Collector และ Emitter"
        )

        retrieval_queries = []

        original_retrieve = (
            tutor.knowledge_service.retrieve
        )

        def controlled_retrieve(
            query,
            n_results=5,
        ):

            retrieval_queries.append(
                query
            )

            return original_retrieve(
                query,
                n_results,
            )

        tutor.knowledge_service.retrieve = (
            controlled_retrieve
        )

        # -------------------------------------------------
        # Semantic factual validation for the final
        # direct follow-up answer.
        # -------------------------------------------------

        tutor.response_grounding_validator.validate = (
            lambda *args, **kwargs:
            GroundingValidationResult(
                status="supported",
                confidence=1.0,
                reason=(
                    "Controlled MT factual response "
                    "is supported by course knowledge."
                ),
                issues=[],
            )
        )

        tutor.guiding_question_grounding_validator.validate = (
            lambda response, knowledge_context:
            GuidingQuestionGroundingResult(
                status="supported",
                reason=(
                    "Controlled MT guiding question "
                    "is answerable from course knowledge."
                ),
                issues=(),
            )
        )

        tutor.response_mode_pedagogical_validator.validate = (
            lambda *args, **kwargs:
            SimpleNamespace(
                status="valid",
                confidence=1.0,
                reason=(
                    "Controlled MT response mode "
                    "is pedagogically valid."
                ),
                issues=[],
            )
        )

        evaluation_result = SimpleNamespace(
            classification="partial",
            confidence=1.0,
            reason=(
                "Controlled MT learner answer."
            ),
            misconception=None,
        )

        generations = [
            initial_response,
            answer_turn_response,
            follow_up_response,
        ]

        generated_index = 0

        def controlled_generation(
            messages,
            task_name="tutor",
            **kwargs,
        ):

            nonlocal generated_index

            result = generations[
                generated_index
            ]

            generated_index += 1

            return result

        with patch(
            "app.tutor.evaluate_response",
            return_value=evaluation_result,
        ) as evaluator:

            with patch(
                "app.tutor.chat_with_ai",
                side_effect=controlled_generation,
            ) as generation:

                tutor.respond(
                    initial_message
                )

                tutor.respond(
                    learner_answer
                )

                answer = tutor.respond(
                    follow_up_message
                )

        # -------------------------------------------------
        # Three Tutor turns occurred.
        # -------------------------------------------------

        self.assert_equal(
            generation.call_count,
            3,
            "MT-T1 Tutor generation count",
        )

        # Only the answer-like middle turn may be
        # evaluated. The follow-up must bypass evaluator.
        self.assert_equal(
            evaluator.call_count,
            1,
            (
                "MT-T1 learner evaluation count"
            ),
        )

        self.assert_true(
            tutor.last_learner_turn_intent
            is not None,
            "MT-T1 final intent missing.",
        )

        self.assert_equal(
            tutor.last_learner_turn_intent.intent,
            "follow_up_question",
            "MT-T1 final learner intent",
        )

        self.assert_true(
            tutor.last_learner_turn_routing
            is not None,
            "MT-T1 routing missing.",
        )

        self.assert_equal(
            tutor.last_learner_turn_routing.route,
            "follow_up_question",
            "MT-T1 route",
        )

        self.assert_equal(
            tutor.original_question,
            follow_up_message,
            (
                "MT-T1 follow-up did not become "
                "the active learner question."
            ),
        )

        # -------------------------------------------------
        # Follow-up retrieval must be current-turn driven.
        # -------------------------------------------------

        self.assert_equal(
            len(
                retrieval_queries
            ),
            3,
            "MT-T1 retrieval count",
        )

        follow_up_query = (
            retrieval_queries[-1]
        )

        self.assert_true(
            "Base" in follow_up_query,
            (
                "MT-T1 follow-up retrieval does not "
                "contain the current Base question."
            ),
        )

        self.assert_true(
            (
                "ขั้วหลักอะไรบ้าง"
                not in follow_up_query
            ),
            (
                "MT-T1 previous Tutor question leaked "
                "into current follow-up retrieval."
            ),
        )

        # -------------------------------------------------
        # Session history must survive all three turns.
        # -------------------------------------------------

        messages = (
            tutor.memory.get_messages()
        )

        self.assert_equal(
            len(
                messages
            ),
            6,
            (
                "MT-T1 ConversationMemory message "
                "count"
            ),
        )

        self.assert_equal(
            messages[0]["content"],
            initial_message,
            "MT-T1 first learner memory",
        )

        self.assert_equal(
            messages[-1]["content"],
            follow_up_response,
            "MT-T1 latest Tutor memory",
        )

        self.assert_equal(
            answer,
            follow_up_response,
            "MT-T1 final answer",
        )

        self.pass_test(
            "MT-T1 Follow-up topic continuity",
            (
                "Across three turns, the learner "
                "answer is evaluated once, the "
                "follow-up bypasses evaluation and "
                "becomes the active question, current "
                "retrieval remains isolated, and "
                "session memory is preserved."
            ),
        )

    # =====================================================
    # 16.21B1 — MT-T2
    # Initial topic → learner answer → clarification
    #
    # Clarification must:
    # - bypass evaluator
    # - keep original active question
    # - clear stale evaluation telemetry
    # - preserve local conversation context
    # =====================================================

    def test_mt_t2_clarification_preserves_local_context(
        self,
    ) -> None:

        tutor = self.build_grounded_tutor()

        initial_message = (
            "ทรานซิสเตอร์ NPN "
            "มีโครงสร้างอย่างไร"
        )

        learner_answer = (
            "มี Base Collector และ Emitter"
        )

        clarification_message = (
            "ช่วยอธิบายคำว่า Base "
            "ให้ชัดเจนอีกครั้ง"
        )

        initial_response = (
            "คุณคิดว่าทรานซิสเตอร์ NPN "
            "มีขั้วหลักอะไรบ้าง?"
        )

        answer_turn_response = (
            "ในสามขั้วนี้ "
            "คุณคิดว่า Base คือขั้วใด?"
        )

        clarification_response = (
            "Base คือหนึ่งในสามขั้วหลักของ "
            "ทรานซิสเตอร์ NPN ร่วมกับ "
            "Collector และ Emitter"
        )

        tutor.response_grounding_validator.validate = (
            lambda *args, **kwargs:
            GroundingValidationResult(
                status="supported",
                confidence=1.0,
                reason=(
                    "Controlled MT clarification "
                    "is grounded."
                ),
                issues=[],
            )
        )

        tutor.guiding_question_grounding_validator.validate = (
            lambda response, knowledge_context:
            GuidingQuestionGroundingResult(
                status="supported",
                reason=(
                    "Controlled MT guiding question "
                    "is supported."
                ),
                issues=(),
            )
        )

        tutor.response_mode_pedagogical_validator.validate = (
            lambda *args, **kwargs:
            SimpleNamespace(
                status="valid",
                confidence=1.0,
                reason=(
                    "Controlled MT clarification "
                    "is pedagogically valid."
                ),
                issues=[],
            )
        )

        evaluation_result = SimpleNamespace(
            classification="partial",
            confidence=1.0,
            reason=(
                "Controlled MT partial answer."
            ),
            misconception=None,
        )

        generation_messages = []

        generations = [
            initial_response,
            answer_turn_response,
            clarification_response,
        ]

        generated_index = 0

        def controlled_generation(
            messages,
            task_name="tutor",
            **kwargs,
        ):

            nonlocal generated_index

            # Make a detached copy because this is
            # multi-turn evidence for the test.
            generation_messages.append(
                [
                    dict(item)
                    for item in messages
                ]
            )

            result = generations[
                generated_index
            ]

            generated_index += 1

            return result

        with patch(
            "app.tutor.evaluate_response",
            return_value=evaluation_result,
        ) as evaluator:

            with patch(
                "app.tutor.chat_with_ai",
                side_effect=controlled_generation,
            ):

                tutor.respond(
                    initial_message
                )

                tutor.respond(
                    learner_answer
                )

                # The second turn legitimately creates
                # evaluation telemetry.
                self.assert_true(
                    (
                        tutor.last_evaluation_result
                        is not None
                    ),
                    (
                        "MT-T2 setup did not create "
                        "learner evaluation state."
                    ),
                )

                answer = tutor.respond(
                    clarification_message
                )

        # -------------------------------------------------
        # Clarification must not invoke evaluator again.
        # -------------------------------------------------

        self.assert_equal(
            evaluator.call_count,
            1,
            (
                "MT-T2 evaluator must run only for "
                "the answer-like middle turn."
            ),
        )

        self.assert_equal(
            tutor.last_learner_turn_intent.intent,
            "clarification_question",
            "MT-T2 clarification intent",
        )

        self.assert_equal(
            tutor.last_learner_turn_routing.route,
            "clarification_question",
            "MT-T2 clarification route",
        )

        self.assert_equal(
            tutor.last_tutoring_response_mode.mode,
            "direct_clarification",
            "MT-T2 clarification response mode",
        )

        # -------------------------------------------------
        # Clarification stays attached to original topic.
        # -------------------------------------------------

        self.assert_equal(
            tutor.original_question,
            initial_message,
            (
                "MT-T2 clarification incorrectly "
                "replaced the active original question."
            ),
        )

        # -------------------------------------------------
        # Stale per-turn evaluation must not leak.
        # -------------------------------------------------

        self.assert_true(
            tutor.last_evaluation_result
            is None,
            (
                "MT-T2 clarification reused stale "
                "learner evaluation telemetry."
            ),
        )

        self.assert_true(
            tutor.last_decision is None,
            (
                "MT-T2 clarification reused stale "
                "scaffolding decision telemetry."
            ),
        )

        # -------------------------------------------------
        # But conversation context itself must remain.
        # The third generation should receive earlier
        # user + assistant messages from memory.
        # -------------------------------------------------

        clarification_generation_messages = (
            generation_messages[-1]
        )

        prior_contents = [
            item["content"]
            for item
            in clarification_generation_messages
            if item["role"] != "system"
        ]

        self.assert_true(
            initial_message
            in prior_contents,
            (
                "MT-T2 clarification lost the "
                "initial learner context."
            ),
        )

        self.assert_true(
            initial_response
            in prior_contents,
            (
                "MT-T2 clarification lost the "
                "previous Tutor context."
            ),
        )

        self.assert_true(
            learner_answer
            in prior_contents,
            (
                "MT-T2 clarification lost the "
                "learner's prior answer."
            ),
        )

        self.assert_equal(
            answer,
            clarification_response,
            "MT-T2 final clarification",
        )

        self.assert_equal(
            len(
                tutor.memory.get_messages()
            ),
            6,
            "MT-T2 session memory count",
        )

        self.pass_test(
            "MT-T2 Clarification context isolation",
            (
                "Clarification bypasses a second "
                "evaluation, clears stale evaluation "
                "and decision telemetry, keeps the "
                "original active topic, and still "
                "receives the local conversation "
                "history needed for explanation."
            ),
        )

    # =====================================================
    # 16.21B1 — MT-T3
    # Topic A → learner answer → explicit Topic B
    #
    # Topic change must:
    # - reset learning-sequence pedagogical state
    # - replace active question
    # - isolate Topic B retrieval
    # - preserve session ConversationMemory
    # =====================================================

    def test_mt_t3_topic_change_resets_sequence_preserves_history(
        self,
    ) -> None:

        tutor = self.build_grounded_tutor()

        topic_a_message = (
            "ทรานซิสเตอร์ NPN "
            "มีโครงสร้างอย่างไร"
        )

        learner_answer = (
            "มีสามขั้วคือ Base "
            "Collector และ Emitter"
        )

        topic_b_message = (
            "ขอถามอีกเรื่อง "
            "ไดโอดมีโครงสร้างอย่างไร?"
        )

        topic_a_response = (
            "คุณคิดว่าทรานซิสเตอร์ NPN "
            "มีขั้วหลักอะไรบ้าง?"
        )

        answer_turn_response = (
            "จากสามขั้วนี้ "
            "คุณคิดว่า Base อยู่ส่วนใด?"
        )

        topic_b_response = (
            "คุณคิดว่าไดโอดมีขั้วหลัก "
            "กี่ขั้ว?"
        )

        npn_context = (
            "The transistor has three terminals: "
            "Base, Collector, and Emitter."
        )

        diode_context = (
            "A diode has two terminals and "
            "contains a p-n junction."
        )

        retrieval_queries = []

        def controlled_retrieve(
            query,
            n_results=5,
        ):

            retrieval_queries.append(
                query
            )

            if "ไดโอด" in query:

                return SimpleNamespace(
                    query=query,
                    context=diode_context,
                    sources=[],
                    citations=[],
                )

            return SimpleNamespace(
                query=query,
                context=npn_context,
                sources=[],
                citations=[],
            )

        tutor.knowledge_service.retrieve = (
            controlled_retrieve
        )

        tutor.response_grounding_validator.validate = (
            lambda *args, **kwargs:
            GroundingValidationResult(
                status="supported",
                confidence=1.0,
                reason=(
                    "Controlled MT response "
                    "is grounded."
                ),
                issues=[],
            )
        )

        tutor.guiding_question_grounding_validator.validate = (
            lambda response, knowledge_context:
            GuidingQuestionGroundingResult(
                status="supported",
                reason=(
                    "Controlled MT guiding question "
                    "is supported by the current "
                    "topic knowledge."
                ),
                issues=(),
            )
        )

        evaluation_result = SimpleNamespace(
            classification="partial",
            confidence=1.0,
            reason=(
                "Controlled MT partial answer "
                "before topic change."
            ),
            misconception=None,
        )

        generations = [
            topic_a_response,
            answer_turn_response,
            topic_b_response,
        ]

        generated_index = 0

        def controlled_generation(
            messages,
            task_name="tutor",
            **kwargs,
        ):

            nonlocal generated_index

            result = generations[
                generated_index
            ]

            generated_index += 1

            return result

        with patch(
            "app.tutor.evaluate_response",
            return_value=evaluation_result,
        ) as evaluator:

            with patch(
                "app.tutor.chat_with_ai",
                side_effect=controlled_generation,
            ):

                tutor.respond(
                    topic_a_message
                )

                tutor.respond(
                    learner_answer
                )

                # Confirm Topic A genuinely created
                # learning-sequence evaluation state.
                self.assert_equal(
                    tutor.state.last_evaluation,
                    "partial",
                    (
                        "MT-T3 setup did not create "
                        "Topic A evaluation state."
                    ),
                )

                history_before_topic_change = [
                    dict(item)
                    for item
                    in tutor.memory.get_messages()
                ]

                answer = tutor.respond(
                    topic_b_message
                )

        # Only the answer-like Topic A turn should
        # have been evaluated.
        self.assert_equal(
            evaluator.call_count,
            1,
            "MT-T3 evaluator call count",
        )

        self.assert_equal(
            tutor.last_learner_turn_intent.intent,
            "topic_change",
            "MT-T3 topic-change intent",
        )

        self.assert_equal(
            tutor.last_learner_turn_routing.route,
            "topic_change",
            "MT-T3 topic-change route",
        )

        self.assert_equal(
            tutor.last_tutoring_response_mode.mode,
            "new_topic_scaffold",
            "MT-T3 new-topic response mode",
        )

        # -------------------------------------------------
        # Topic B becomes active.
        # -------------------------------------------------

        self.assert_equal(
            tutor.original_question,
            topic_b_message,
            (
                "MT-T3 Topic B did not replace "
                "the active learner question."
            ),
        )

        self.assert_true(
            tutor.waiting_for_response,
            (
                "MT-T3 Tutor is not waiting for "
                "the new Topic B sequence."
            ),
        )

        # -------------------------------------------------
        # Old pedagogical evaluation state must reset.
        # -------------------------------------------------

        self.assert_true(
            tutor.state.last_evaluation
            is None,
            (
                "MT-T3 Topic A evaluation leaked "
                "into the Topic B sequence."
            ),
        )

        self.assert_true(
            tutor.last_evaluation_result
            is None,
            (
                "MT-T3 Topic A evaluation telemetry "
                "leaked into Topic B."
            ),
        )

        self.assert_true(
            tutor.last_decision
            is None,
            (
                "MT-T3 Topic A scaffolding decision "
                "leaked into Topic B."
            ),
        )

        # -------------------------------------------------
        # Retrieval isolation
        # -------------------------------------------------

        topic_b_query = (
            retrieval_queries[-1]
        )

        self.assert_true(
            "ไดโอด" in topic_b_query,
            (
                "MT-T3 Topic B retrieval does not "
                "contain the diode topic."
            ),
        )

        self.assert_true(
            "NPN" not in topic_b_query,
            (
                "MT-T3 Topic A leaked into Topic B "
                "retrieval."
            ),
        )

        self.assert_true(
            (
                "Base อยู่ส่วนใด"
                not in topic_b_query
            ),
            (
                "MT-T3 previous Tutor question leaked "
                "into Topic B retrieval."
            ),
        )

        self.assert_equal(
            tutor.last_knowledge_result.context,
            diode_context,
            (
                "MT-T3 did not retrieve the "
                "Topic B knowledge context."
            ),
        )

        # -------------------------------------------------
        # Session history must NOT be erased by
        # reset_learning_sequence().
        # -------------------------------------------------

        history_after_topic_change = (
            tutor.memory.get_messages()
        )

        self.assert_equal(
            (
                history_after_topic_change[
                    :len(
                        history_before_topic_change
                    )
                ]
            ),
            history_before_topic_change,
            (
                "MT-T3 topic-sequence reset erased "
                "or modified prior session history."
            ),
        )

        self.assert_equal(
            len(
                history_after_topic_change
            ),
            6,
            (
                "MT-T3 session history should contain "
                "three complete learner/Tutor turns."
            ),
        )

        self.assert_equal(
            answer,
            topic_b_response,
            "MT-T3 final Topic B response",
        )

        self.pass_test(
            "MT-T3 Topic-change sequence boundary",
            (
                "After real Topic A evaluation, an "
                "explicit Topic B change resets the "
                "pedagogical sequence, replaces the "
                "active question, isolates retrieval "
                "to Topic B, and preserves prior "
                "session ConversationMemory."
            ),
        )

    # =====================================================
    # 16.21B2 — MT-T4
    # Successful turn → failed repair turn
    # → next learner question starts clean
    # =====================================================

    def test_mt_t4_failed_turn_next_turn_clean(
        self,
    ) -> None:

        tutor = self.build_grounded_tutor()

        first_question = (
            "ทรานซิสเตอร์ NPN "
            "มีโครงสร้างอย่างไร"
        )

        failed_turn_message = (
            "แล้วแรงดัน Base-Emitter "
            "ต้องเท่ากับเท่าไร"
        )

        next_question = (
            "Collector คือขั้วอะไร?"
        )

        first_response = (
            "คุณคิดว่าทรานซิสเตอร์ NPN "
            "มีขั้วหลักอะไรบ้าง?"
        )

        unsafe_failed_response = (
            "แรงดัน Base-Emitter "
            "ต้องเท่ากับ 0.7 V"
        )

        next_response = (
            "คุณคิดว่า Collector "
            "เป็นหนึ่งในสามขั้วหลักของ "
            "ทรานซิสเตอร์ NPN หรือไม่?"
        )

        retrieval_queries = []

        knowledge_result = SimpleNamespace(
            query="controlled NPN query",
            context=(
                "The transistor has three terminals: "
                "Base, Collector, and Emitter. "
                "The Base-Emitter junction may be "
                "forward biased or reverse biased."
            ),
            sources=[],
            citations=[],
        )

        def controlled_retrieve(
            query,
            n_results=5,
        ):

            retrieval_queries.append(
                query
            )

            return knowledge_result

        tutor.knowledge_service.retrieve = (
            controlled_retrieve
        )

        # -------------------------------------------------
        # Initial + next response supported.
        # Failed middle response unsupported.
        # -------------------------------------------------

        def grounding_validation(
            response,
            knowledge_context,
            task_name="response_validator",
        ):

            if response == unsafe_failed_response:

                return GroundingValidationResult(
                    status="unsupported",
                    confidence=1.0,
                    reason=(
                        "Exact 0.7 V value is absent "
                        "from course evidence."
                    ),
                    issues=[
                        (
                            "Unsupported exact voltage."
                        )
                    ],
                )

            return GroundingValidationResult(
                status="supported",
                confidence=1.0,
                reason=(
                    "Controlled MT response "
                    "is grounded."
                ),
                issues=[],
            )

        tutor.response_grounding_validator.validate = (
            grounding_validation
        )

        tutor.guiding_question_grounding_validator.validate = (
            lambda response, knowledge_context:
            GuidingQuestionGroundingResult(
                status="supported",
                reason=(
                    "Controlled guiding question "
                    "is supported."
                ),
                issues=(),
            )
        )

        # No verified exact evidence for failed turn.
        tutor.repair_evidence_selector.select = (
            lambda learner_message,
            knowledge_context,
            validation_issues=None:
            RepairEvidenceSelectionResult(
                status="no_evidence",
                evidence_quotes=(),
                reason=(
                    "No verified exact voltage "
                    "evidence exists."
                ),
                issues=(),
            )
        )

        generations = [
            first_response,
            unsafe_failed_response,
            next_response,
        ]

        generated_index = 0

        def controlled_generation(
            messages,
            task_name="tutor",
            **kwargs,
        ):

            nonlocal generated_index

            result = generations[
                generated_index
            ]

            generated_index += 1

            return result

        # Middle turn is interpreted as a follow-up
        # question, so evaluator must not own it.
        with patch(
            "app.tutor.chat_with_ai",
            side_effect=controlled_generation,
        ):

            tutor.respond(
                first_question
            )

            memory_after_first = [
                dict(item)
                for item
                in tutor.memory.get_messages()
            ]

            failed_answer = tutor.respond(
                failed_turn_message
            )

            self.assert_true(
                tutor.last_repair_failed,
                (
                    "MT-T4 middle turn did not "
                    "fail closed."
                ),
            )

            self.assert_true(
                tutor.original_question is None,
                (
                    "MT-T4 failed turn left an "
                    "active question."
                ),
            )

            self.assert_equal(
                tutor.waiting_for_response,
                False,
                (
                    "MT-T4 failed turn left Tutor "
                    "waiting for a response."
                ),
            )

            # Failed exchange must not enter memory.
            self.assert_equal(
                tutor.memory.get_messages(),
                memory_after_first,
                (
                    "MT-T4 failed exchange polluted "
                    "ConversationMemory."
                ),
            )

            answer = tutor.respond(
                next_question
            )

        # -------------------------------------------------
        # Next turn must be a new learning sequence.
        # -------------------------------------------------

        self.assert_equal(
            tutor.original_question,
            next_question,
            (
                "MT-T4 next question did not start "
                "a clean active sequence."
            ),
        )

        self.assert_true(
            tutor.waiting_for_response,
            (
                "MT-T4 Tutor is not waiting on the "
                "new clean sequence."
            ),
        )

        self.assert_equal(
            answer,
            next_response,
            "MT-T4 next clean answer",
        )

        self.assert_true(
            not tutor.last_repair_failed,
            (
                "MT-T4 repair-failed state leaked "
                "into the next turn."
            ),
        )

        # Current retrieval must be driven by new question.
        next_query = retrieval_queries[-1]

        self.assert_true(
            "Collector" in next_query,
            (
                "MT-T4 new retrieval does not use "
                "the new Collector question."
            ),
        )

        self.assert_true(
            "0.7" not in next_query,
            (
                "MT-T4 failed voltage turn leaked "
                "into subsequent retrieval."
            ),
        )

        # Memory now contains first successful exchange
        # + next successful exchange only.
        memory = tutor.memory.get_messages()

        self.assert_equal(
            len(memory),
            4,
            "MT-T4 final memory count",
        )

        self.assert_true(
            all(
                item["content"]
                != failed_turn_message
                for item in memory
            ),
            (
                "MT-T4 failed learner message "
                "appeared in memory."
            ),
        )

        self.assert_true(
            all(
                item["content"]
                != failed_answer
                for item in memory
            ),
            (
                "MT-T4 fail-closed fallback "
                "appeared in memory."
            ),
        )

        self.pass_test(
            "MT-T4 Failed turn recovery boundary",
            (
                "A failed factual turn closes its "
                "learning sequence without entering "
                "ConversationMemory, and the next "
                "learner question begins cleanly "
                "without retrieval contamination."
            ),
        )

    # =====================================================
    # 16.21B2 — MT-T5
    # Out-of-course turn
    # → no memory
    # → next in-course turn starts clean
    # =====================================================

    def test_mt_t5_out_of_course_next_in_course_clean(
        self,
    ) -> None:

        tutor = self.build_grounded_tutor()

        out_of_course = (
            "อินทิกรัลไม่จำกัดเขตคืออะไร"
        )

        in_course = (
            "ทรานซิสเตอร์ NPN "
            "มีขั้วอะไรบ้าง"
        )

        grounded_response = (
            "คุณคิดว่าทรานซิสเตอร์ NPN "
            "มีขั้วหลักอะไรบ้าง?"
        )

        retrieval_queries = []

        knowledge_result = SimpleNamespace(
            query="controlled query",
            context=(
                "The transistor has three terminals: "
                "Base, Collector, and Emitter."
            ),
            sources=[],
            citations=[],
        )

        def controlled_retrieve(
            query,
            n_results=5,
        ):

            retrieval_queries.append(
                query
            )

            return knowledge_result

        tutor.knowledge_service.retrieve = (
            controlled_retrieve
        )

        # Relevance depends on the current learner topic.
        tutor.relevance_gate.evaluate = (
            lambda query, knowledge_result:
            SimpleNamespace(
                is_relevant=(
                    "อินทิกรัล" not in query
                ),
                confidence=1.0,
                reason=(
                    "Controlled MT relevance."
                ),
            )
        )

        tutor.grounding_guard.evaluate = (
            lambda result, relevance:
            SimpleNamespace(
                has_knowledge=(
                    relevance.is_relevant
                ),
                reason=(
                    "Controlled MT grounding "
                    "availability."
                ),
            )
        )

        tutor.guiding_question_grounding_validator.validate = (
            lambda response, knowledge_context:
            GuidingQuestionGroundingResult(
                status="supported",
                reason=(
                    "Controlled grounded question."
                ),
                issues=(),
            )
        )

        with patch(
            "app.tutor.chat_with_ai",
            return_value=grounded_response,
        ) as generation:

            fallback = tutor.respond(
                out_of_course
            )

            # Out-of-course turn must not enter memory.
            self.assert_equal(
                len(
                    tutor.memory.get_messages()
                ),
                0,
                (
                    "MT-T5 out-of-course turn "
                    "polluted ConversationMemory."
                ),
            )

            self.assert_true(
                tutor.original_question is None,
                (
                    "MT-T5 out-of-course turn "
                    "left active question state."
                ),
            )

            self.assert_equal(
                tutor.waiting_for_response,
                False,
                (
                    "MT-T5 out-of-course turn "
                    "left Tutor waiting."
                ),
            )

            answer = tutor.respond(
                in_course
            )

        # Tutor generation only occurs for in-course turn.
        self.assert_equal(
            generation.call_count,
            1,
            (
                "MT-T5 Tutor generation count"
            ),
        )

        self.assert_true(
            isinstance(
                fallback,
                str,
            )
            and
            bool(
                fallback.strip()
            ),
            (
                "MT-T5 out-of-course fallback "
                "was empty."
            ),
        )

        self.assert_equal(
            answer,
            grounded_response,
            "MT-T5 in-course answer",
        )

        self.assert_equal(
            tutor.original_question,
            in_course,
            (
                "MT-T5 in-course turn did not "
                "start a new clean sequence."
            ),
        )

        self.assert_true(
            tutor.waiting_for_response,
            (
                "MT-T5 clean in-course sequence "
                "is not active."
            ),
        )

        # Last retrieval must contain only current topic.
        in_course_query = (
            retrieval_queries[-1]
        )

        self.assert_true(
            "NPN" in in_course_query,
            (
                "MT-T5 in-course retrieval does not "
                "contain current NPN topic."
            ),
        )

        self.assert_true(
            "อินทิกรัล" not in in_course_query,
            (
                "MT-T5 out-of-course topic leaked "
                "into next retrieval."
            ),
        )

        memory = tutor.memory.get_messages()

        self.assert_equal(
            len(memory),
            2,
            "MT-T5 memory count",
        )

        self.assert_equal(
            memory[0]["content"],
            in_course,
            (
                "MT-T5 first stored learner message "
                "must be the valid in-course turn."
            ),
        )

        self.assert_equal(
            memory[1]["content"],
            grounded_response,
            (
                "MT-T5 stored Tutor answer"
            ),
        )

        self.pass_test(
            "MT-T5 Out-of-course memory isolation",
            (
                "An out-of-course turn is excluded "
                "from ConversationMemory and clears "
                "its active sequence, so the next "
                "in-course question retrieves and "
                "starts independently."
            ),
        )

    # =====================================================
    # 16.21B2 — MT-T6
    # reset() = hard session reset
    # =====================================================

    def test_mt_t6_hard_session_reset(
        self,
    ) -> None:

        tutor = self.build_grounded_tutor()

        first_question = (
            "ทรานซิสเตอร์ NPN "
            "มีโครงสร้างอย่างไร"
        )

        first_response = (
            "คุณคิดว่าทรานซิสเตอร์ NPN "
            "มีขั้วหลักอะไรบ้าง?"
        )

        tutor.guiding_question_grounding_validator.validate = (
            lambda response, knowledge_context:
            GuidingQuestionGroundingResult(
                status="supported",
                reason=(
                    "Controlled MT reset setup."
                ),
                issues=(),
            )
        )

        with patch(
            "app.tutor.chat_with_ai",
            return_value=first_response,
        ):

            tutor.respond(
                first_question
            )

        # -------------------------------------------------
        # Prove state exists before reset.
        # -------------------------------------------------

        self.assert_true(
            len(
                tutor.memory.get_messages()
            )
            > 0,
            (
                "MT-T6 setup produced no "
                "ConversationMemory."
            ),
        )

        self.assert_true(
            tutor.original_question
            is not None,
            (
                "MT-T6 setup produced no "
                "active question."
            ),
        )

        self.assert_true(
            tutor.waiting_for_response,
            (
                "MT-T6 setup produced no "
                "active sequence."
            ),
        )

        self.assert_true(
            tutor.last_generated_answer
            is not None,
            (
                "MT-T6 setup produced no "
                "generation telemetry."
            ),
        )

        # -------------------------------------------------
        # Hard reset
        # -------------------------------------------------

        tutor.reset()

        # Conversation memory
        self.assert_equal(
            tutor.memory.get_messages(),
            [],
            (
                "MT-T6 reset did not clear "
                "ConversationMemory."
            ),
        )

        # Active tutoring sequence
        self.assert_true(
            tutor.original_question
            is None,
            (
                "MT-T6 reset did not clear "
                "original_question."
            ),
        )

        self.assert_equal(
            tutor.waiting_for_response,
            False,
            (
                "MT-T6 reset did not clear "
                "waiting_for_response."
            ),
        )

        # Current-turn telemetry
        self.assert_true(
            tutor.last_generated_answer
            is None,
            (
                "MT-T6 reset did not clear "
                "generated-answer telemetry."
            ),
        )

        self.assert_true(
            tutor.last_retrieval_query
            is None,
            (
                "MT-T6 reset did not clear "
                "retrieval telemetry."
            ),
        )

        self.assert_true(
            tutor.last_learner_turn_intent
            is None,
            (
                "MT-T6 reset did not clear "
                "learner-turn intent telemetry."
            ),
        )

        self.assert_true(
            tutor.last_tutoring_response_mode
            is None,
            (
                "MT-T6 reset did not clear "
                "response-mode telemetry."
            ),
        )

        self.assert_true(
            tutor.last_evaluation_result
            is None,
            (
                "MT-T6 reset did not clear "
                "evaluation telemetry."
            ),
        )

        self.assert_true(
            tutor.last_decision
            is None,
            (
                "MT-T6 reset did not clear "
                "scaffolding decision telemetry."
            ),
        )

        self.assert_true(
            not tutor.last_response_repaired,
            (
                "MT-T6 reset retained "
                "repair-success state."
            ),
        )

        self.assert_true(
            not tutor.last_repair_failed,
            (
                "MT-T6 reset retained "
                "repair-failure state."
            ),
        )

        # State-level session counter should restart.
        self.assert_equal(
            tutor.state.turn_count,
            0,
            "MT-T6 TutorState turn count",
        )

        self.pass_test(
            "MT-T6 Hard session reset",
            (
                "reset() clears ConversationMemory, "
                "active tutoring-sequence state, "
                "turn-level telemetry, repair state, "
                "and TutorState session counters."
            ),
        )
    # =====================================================
    # 16.21B2.1 — PORT-T1
    #
    # Same core Tutor behavior must survive a change
    # of CourseProfile.
    #
    # This test intentionally uses synthetic controlled
    # evidence. It tests Tutor orchestration portability,
    # not subject-matter correctness.
    # =====================================================

    def test_port_t1_cross_course_core_behavior(
        self,
    ) -> None:

        course_ids = (
            "electronics",
            "mathematics",
        )

        results = {}

        for course_id in course_ids:

            # ---------------------------------------------
            # Build a real AITutor under the selected
            # CourseProfile.
            #
            # AITutor imports ACTIVE_COURSE into tutor.py,
            # so patch the symbol owned by app.tutor.
            # ---------------------------------------------

            with patch(
                "app.tutor.ACTIVE_COURSE",
                course_id,
            ):

                tutor = AITutor()

            # Prove that the requested CourseProfile
            # was actually loaded.
            self.assert_equal(
                getattr(
                    tutor.course_profile,
                    "course_id",
                    None,
                ),
                course_id,
                (
                    "PORT-T1 loaded CourseProfile "
                    f"for {course_id}"
                ),
            )

            # ---------------------------------------------
            # Synthetic course-independent evidence
            # ---------------------------------------------

            knowledge_context = (
                "Concept Alpha contains Part One "
                "and Part Two. "
                "Part One is a component of "
                "Concept Alpha."
            )

            knowledge_result = SimpleNamespace(
                query="controlled Alpha query",
                context=knowledge_context,
                sources=[],
                citations=[],
            )

            retrieval_queries = []

            def controlled_retrieve(
                query,
                n_results=5,
            ):

                retrieval_queries.append(
                    query
                )

                return knowledge_result

            tutor.knowledge_service.retrieve = (
                controlled_retrieve
            )

            tutor.relevance_gate.evaluate = (
                lambda query, knowledge_result:
                SimpleNamespace(
                    is_relevant=True,
                    confidence=1.0,
                    reason=(
                        "Controlled portability "
                        "course relevance."
                    ),
                )
            )

            tutor.grounding_guard.evaluate = (
                lambda result, relevance:
                SimpleNamespace(
                    has_knowledge=True,
                    reason=(
                        "Controlled portability "
                        "knowledge is available."
                    ),
                )
            )

            # ---------------------------------------------
            # Controlled semantic boundaries.
            #
            # PORT-T1 is about orchestration surviving
            # CourseProfile changes. Grounding semantics
            # are already owned by frozen 16.20.
            # ---------------------------------------------

            tutor.response_grounding_validator.validate = (
                lambda *args, **kwargs:
                GroundingValidationResult(
                    status="supported",
                    confidence=1.0,
                    reason=(
                        "Controlled portability "
                        "response is grounded."
                    ),
                    issues=[],
                )
            )

            tutor.guiding_question_grounding_validator.validate = (
                lambda response, knowledge_context:
                GuidingQuestionGroundingResult(
                    status="supported",
                    reason=(
                        "Controlled portability "
                        "guiding question is supported."
                    ),
                    issues=(),
                )
            )

            tutor.pedagogical_response_validator.validate = (
                lambda *args, **kwargs:
                SimpleNamespace(
                    status="valid",
                    confidence=1.0,
                    reason=(
                        "Controlled portability "
                        "pedagogy is valid."
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
                        "Controlled portability "
                        "response mode is valid."
                    ),
                    issues=[],
                )
            )

            # ---------------------------------------------
            # Two-turn generic interaction
            # ---------------------------------------------

            initial_message = (
                "แนวคิด Alpha "
                "มีองค์ประกอบอะไรบ้าง?"
            )

            follow_up_message = (
                "แล้ว Part One คืออะไร?"
            )

            initial_response = (
                "คุณคิดว่าแนวคิด Alpha "
                "มีองค์ประกอบหลักกี่ส่วน?"
            )

            follow_up_response = (
                "Part One เป็นองค์ประกอบหนึ่ง "
                "ของแนวคิด Alpha"
            )

            generations = [
                initial_response,
                follow_up_response,
            ]

            generated_index = 0

            def controlled_generation(
                messages,
                task_name="tutor",
                **kwargs,
            ):

                nonlocal generated_index

                response = generations[
                    generated_index
                ]

                generated_index += 1

                return response

            # Evaluator must never own the second turn
            # because it is a follow-up question.
            with patch(
                "app.tutor.evaluate_response",
            ) as evaluator:

                with patch(
                    "app.tutor.chat_with_ai",
                    side_effect=controlled_generation,
                ) as generation:

                    first_answer = tutor.respond(
                        initial_message
                    )

                    second_answer = tutor.respond(
                        follow_up_message
                    )

            # ---------------------------------------------
            # Same behavioral contract for either course
            # ---------------------------------------------

            self.assert_equal(
                generation.call_count,
                2,
                (
                    "PORT-T1 Tutor generation count "
                    f"for {course_id}"
                ),
            )

            self.assert_equal(
                evaluator.call_count,
                0,
                (
                    "PORT-T1 follow-up must not be "
                    "evaluated as learner answer "
                    f"for {course_id}"
                ),
            )

            self.assert_equal(
                first_answer,
                initial_response,
                (
                    "PORT-T1 initial answer "
                    f"for {course_id}"
                ),
            )

            self.assert_equal(
                second_answer,
                follow_up_response,
                (
                    "PORT-T1 follow-up answer "
                    f"for {course_id}"
                ),
            )

            self.assert_true(
                tutor.last_learner_turn_intent
                is not None,
                (
                    "PORT-T1 final intent missing "
                    f"for {course_id}"
                ),
            )

            self.assert_equal(
                tutor.last_learner_turn_intent.intent,
                "follow_up_question",
                (
                    "PORT-T1 follow-up intent "
                    f"for {course_id}"
                ),
            )

            self.assert_equal(
                tutor.last_learner_turn_routing.route,
                "follow_up_question",
                (
                    "PORT-T1 follow-up route "
                    f"for {course_id}"
                ),
            )

            self.assert_equal(
                tutor.last_tutoring_response_mode.mode,
                "answer_then_guide",
                (
                    "PORT-T1 response mode "
                    f"for {course_id}"
                ),
            )

            self.assert_equal(
                tutor.original_question,
                follow_up_message,
                (
                    "PORT-T1 active question "
                    f"for {course_id}"
                ),
            )

            self.assert_true(
                not tutor.last_repair_failed,
                (
                    "PORT-T1 unexpected repair "
                    "failure for "
                    f"{course_id}"
                ),
            )

            self.assert_equal(
                len(
                    tutor.memory.get_messages()
                ),
                4,
                (
                    "PORT-T1 ConversationMemory "
                    f"for {course_id}"
                ),
            )

            self.assert_equal(
                len(
                    retrieval_queries
                ),
                2,
                (
                    "PORT-T1 retrieval count "
                    f"for {course_id}"
                ),
            )

            self.assert_true(
                "Part One"
                in retrieval_queries[-1],
                (
                    "PORT-T1 current follow-up "
                    "missing from retrieval for "
                    f"{course_id}"
                ),
            )

            # Snapshot only behavior that should be
            # invariant across courses.
            results[
                course_id
            ] = {
                "intent": (
                    tutor
                    .last_learner_turn_intent
                    .intent
                ),
                "route": (
                    tutor
                    .last_learner_turn_routing
                    .route
                ),
                "mode": (
                    tutor
                    .last_tutoring_response_mode
                    .mode
                ),
                "memory_count": len(
                    tutor.memory.get_messages()
                ),
                "repair_failed": (
                    tutor.last_repair_failed
                ),
            }

        # =================================================
        # Cross-course behavioral equivalence
        # =================================================

        self.assert_equal(
            results["electronics"],
            results["mathematics"],
            (
                "PORT-T1 core Tutor behavior changed "
                "when CourseProfile changed from "
                "electronics to mathematics."
            ),
        )

        self.pass_test(
            "PORT-T1 Cross-course core behavior",
            (
                "The same two-turn initial/follow-up "
                "Tutor behavior remains invariant "
                "under Electronics and Mathematics "
                "CourseProfiles while each Tutor "
                "loads its own course configuration."
            ),
        )

    # =====================================================
    # 16.21B2.2 — PORT-T2
    #
    # Cross-course multi-turn state-machine portability
    #
    # initial
    # → learner answer
    # → follow-up question
    # → explicit topic change
    #
    # The behavioral state transitions must remain
    # invariant under different CourseProfiles.
    # =====================================================

    def test_port_t2_cross_course_multi_turn_state_machine(
        self,
    ) -> None:

        course_ids = (
            "electronics",
            "mathematics",
        )

        results = {}

        for course_id in course_ids:

            # =================================================
            # Build Tutor under the selected CourseProfile
            # =================================================

            with patch(
                "app.tutor.ACTIVE_COURSE",
                course_id,
            ):

                tutor = AITutor()

            self.assert_equal(
                getattr(
                    tutor.course_profile,
                    "course_id",
                    None,
                ),
                course_id,
                (
                    "PORT-T2 CourseProfile "
                    f"for {course_id}"
                ),
            )

            # =================================================
            # Synthetic subject-independent knowledge
            # =================================================

            alpha_context = (
                "Concept Alpha contains Part One "
                "and Part Two. "
                "Part One is a component of "
                "Concept Alpha."
            )

            beta_context = (
                "Concept Beta contains Part Three "
                "and Part Four. "
                "Part Three is a component of "
                "Concept Beta."
            )

            retrieval_queries = []

            def controlled_retrieve(
                query,
                n_results=5,
            ):

                retrieval_queries.append(
                    query
                )

                if "Beta" in query:

                    return SimpleNamespace(
                        query=query,
                        context=beta_context,
                        sources=[],
                        citations=[],
                    )

                return SimpleNamespace(
                    query=query,
                    context=alpha_context,
                    sources=[],
                    citations=[],
                )

            tutor.knowledge_service.retrieve = (
                controlled_retrieve
            )

            tutor.relevance_gate.evaluate = (
                lambda query, knowledge_result:
                SimpleNamespace(
                    is_relevant=True,
                    confidence=1.0,
                    reason=(
                        "Controlled portability "
                        "relevance."
                    ),
                )
            )

            tutor.grounding_guard.evaluate = (
                lambda result, relevance:
                SimpleNamespace(
                    has_knowledge=True,
                    reason=(
                        "Controlled portability "
                        "knowledge available."
                    ),
                )
            )

            # =================================================
            # Isolate semantic layers already frozen
            # =================================================

            tutor.response_grounding_validator.validate = (
                lambda *args, **kwargs:
                GroundingValidationResult(
                    status="supported",
                    confidence=1.0,
                    reason=(
                        "Controlled portability "
                        "grounding."
                    ),
                    issues=[],
                )
            )

            tutor.guiding_question_grounding_validator.validate = (
                lambda response, knowledge_context:
                GuidingQuestionGroundingResult(
                    status="supported",
                    reason=(
                        "Controlled portability "
                        "guiding question."
                    ),
                    issues=(),
                )
            )

            tutor.pedagogical_response_validator.validate = (
                lambda *args, **kwargs:
                SimpleNamespace(
                    status="valid",
                    confidence=1.0,
                    reason=(
                        "Controlled portability "
                        "pedagogy."
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
                        "Controlled portability "
                        "response-mode pedagogy."
                    ),
                    issues=[],
                )
            )

            # =================================================
            # Four learner turns
            # =================================================

            initial_message = (
                "แนวคิด Alpha "
                "มีองค์ประกอบอะไรบ้าง?"
            )

            learner_answer = (
                "มี Part One และ Part Two"
            )

            follow_up_message = (
                "แล้ว Part One คืออะไร?"
            )

            topic_change_message = (
                "ขอถามอีกเรื่อง "
                "แนวคิด Beta มีองค์ประกอบอะไรบ้าง?"
            )

            initial_response = (
                "คุณคิดว่าแนวคิด Alpha "
                "มีองค์ประกอบหลักกี่ส่วน?"
            )

            answer_response = (
                "จากคำตอบของคุณ "
                "Part One เป็นองค์ประกอบของ "
                "แนวคิด Alpha หรือไม่?"
            )

            follow_up_response = (
                "Part One เป็นองค์ประกอบหนึ่ง "
                "ของแนวคิด Alpha"
            )

            topic_change_response = (
                "คุณคิดว่าแนวคิด Beta "
                "มีองค์ประกอบหลักกี่ส่วน?"
            )

            generations = [
                initial_response,
                answer_response,
                follow_up_response,
                topic_change_response,
            ]

            generated_index = 0

            def controlled_generation(
                messages,
                task_name="tutor",
                **kwargs,
            ):

                nonlocal generated_index

                response = generations[
                    generated_index
                ]

                generated_index += 1

                return response

            # =================================================
            # Learner-answer evaluation
            # =================================================

            evaluation_result = SimpleNamespace(
                classification="partial",
                confidence=1.0,
                reason=(
                    "Controlled portability "
                    "partial learner answer."
                ),
                misconception=None,
            )

            with patch(
                "app.tutor.evaluate_response",
                return_value=evaluation_result,
            ) as evaluator:

                with patch(
                    "app.tutor.chat_with_ai",
                    side_effect=controlled_generation,
                ) as generation:

                    # -----------------------------------------
                    # Turn 1 — initial
                    # -----------------------------------------

                    tutor.respond(
                        initial_message
                    )

                    self.assert_equal(
                        tutor.original_question,
                        initial_message,
                        (
                            "PORT-T2 initial active "
                            f"question for {course_id}"
                        ),
                    )

                    # -----------------------------------------
                    # Turn 2 — answer
                    # -----------------------------------------

                    tutor.respond(
                        learner_answer
                    )

                    self.assert_true(
                        tutor.last_evaluation_result
                        is not None,
                        (
                            "PORT-T2 learner answer "
                            "was not evaluated for "
                            f"{course_id}"
                        ),
                    )

                    self.assert_equal(
                        (
                            tutor
                            .last_evaluation_result
                            .classification
                        ),
                        "partial",
                        (
                            "PORT-T2 learner answer "
                            f"classification for {course_id}"
                        ),
                    )

                    self.assert_equal(
                        tutor.state.last_evaluation,
                        "partial",
                        (
                            "PORT-T2 state evaluation "
                            f"for {course_id}"
                        ),
                    )

                    # -----------------------------------------
                    # Turn 3 — follow-up
                    # -----------------------------------------

                    follow_up_answer = tutor.respond(
                        follow_up_message
                    )

                    self.assert_equal(
                        (
                            tutor
                            .last_learner_turn_intent
                            .intent
                        ),
                        "follow_up_question",
                        (
                            "PORT-T2 follow-up intent "
                            f"for {course_id}"
                        ),
                    )

                    self.assert_equal(
                        (
                            tutor
                            .last_learner_turn_routing
                            .route
                        ),
                        "follow_up_question",
                        (
                            "PORT-T2 follow-up route "
                            f"for {course_id}"
                        ),
                    )

                    self.assert_equal(
                        (
                            tutor
                            .last_tutoring_response_mode
                            .mode
                        ),
                        "answer_then_guide",
                        (
                            "PORT-T2 follow-up mode "
                            f"for {course_id}"
                        ),
                    )

                    self.assert_equal(
                        tutor.original_question,
                        follow_up_message,
                        (
                            "PORT-T2 follow-up did not "
                            "replace active question for "
                            f"{course_id}"
                        ),
                    )

                    self.assert_true(
                        tutor.last_evaluation_result
                        is None,
                        (
                            "PORT-T2 stale answer "
                            "evaluation telemetry leaked "
                            "into follow-up for "
                            f"{course_id}"
                        ),
                    )

                    self.assert_equal(
                        follow_up_answer,
                        follow_up_response,
                        (
                            "PORT-T2 follow-up answer "
                            f"for {course_id}"
                        ),
                    )

                    # -----------------------------------------
                    # Snapshot memory before topic change
                    # -----------------------------------------

                    memory_before_topic_change = [
                        dict(item)
                        for item
                        in tutor.memory.get_messages()
                    ]

                    # -----------------------------------------
                    # Turn 4 — explicit Topic Beta
                    # -----------------------------------------

                    topic_answer = tutor.respond(
                        topic_change_message
                    )

            # =================================================
            # Global invocation contract
            # =================================================

            self.assert_equal(
                generation.call_count,
                4,
                (
                    "PORT-T2 generation count "
                    f"for {course_id}"
                ),
            )

            # Only Turn 2 is answer-like.
            self.assert_equal(
                evaluator.call_count,
                1,
                (
                    "PORT-T2 evaluator count "
                    f"for {course_id}"
                ),
            )

            # =================================================
            # Topic-change routing
            # =================================================

            self.assert_equal(
                (
                    tutor
                    .last_learner_turn_intent
                    .intent
                ),
                "topic_change",
                (
                    "PORT-T2 topic-change intent "
                    f"for {course_id}"
                ),
            )

            self.assert_equal(
                (
                    tutor
                    .last_learner_turn_routing
                    .route
                ),
                "topic_change",
                (
                    "PORT-T2 topic-change route "
                    f"for {course_id}"
                ),
            )

            self.assert_equal(
                (
                    tutor
                    .last_tutoring_response_mode
                    .mode
                ),
                "new_topic_scaffold",
                (
                    "PORT-T2 topic-change mode "
                    f"for {course_id}"
                ),
            )

            # Topic change must become active question.
            self.assert_equal(
                tutor.original_question,
                topic_change_message,
                (
                    "PORT-T2 Topic Beta did not "
                    "become active question for "
                    f"{course_id}"
                ),
            )

            self.assert_true(
                tutor.waiting_for_response,
                (
                    "PORT-T2 Topic Beta sequence "
                    "is not active for "
                    f"{course_id}"
                ),
            )

            # =================================================
            # Learning-sequence state reset
            # =================================================

            self.assert_true(
                tutor.state.last_evaluation
                is None,
                (
                    "PORT-T2 Alpha evaluation state "
                    "leaked into Beta for "
                    f"{course_id}"
                ),
            )

            self.assert_true(
                tutor.last_evaluation_result
                is None,
                (
                    "PORT-T2 Alpha evaluation "
                    "telemetry leaked into Beta for "
                    f"{course_id}"
                ),
            )

            self.assert_true(
                tutor.last_decision
                is None,
                (
                    "PORT-T2 Alpha scaffolding "
                    "decision leaked into Beta for "
                    f"{course_id}"
                ),
            )

            # =================================================
            # Topic-B retrieval isolation
            # =================================================

            beta_query = (
                retrieval_queries[-1]
            )

            self.assert_true(
                "Beta" in beta_query,
                (
                    "PORT-T2 Beta topic missing "
                    "from retrieval for "
                    f"{course_id}"
                ),
            )

            self.assert_true(
                "Alpha" not in beta_query,
                (
                    "PORT-T2 Alpha leaked into "
                    "Beta retrieval for "
                    f"{course_id}"
                ),
            )

            self.assert_true(
                "Part One" not in beta_query,
                (
                    "PORT-T2 previous follow-up "
                    "leaked into Beta retrieval for "
                    f"{course_id}"
                ),
            )

            self.assert_equal(
                tutor.last_knowledge_result.context,
                beta_context,
                (
                    "PORT-T2 Beta knowledge "
                    f"context for {course_id}"
                ),
            )

            # =================================================
            # Session memory preservation
            # =================================================

            memory_after_topic_change = (
                tutor.memory.get_messages()
            )

            self.assert_equal(
                (
                    memory_after_topic_change[
                        :len(
                            memory_before_topic_change
                        )
                    ]
                ),
                memory_before_topic_change,
                (
                    "PORT-T2 topic-change reset "
                    "modified prior session memory "
                    f"for {course_id}"
                ),
            )

            # Four successful learner/Tutor exchanges.
            self.assert_equal(
                len(
                    memory_after_topic_change
                ),
                8,
                (
                    "PORT-T2 memory count "
                    f"for {course_id}"
                ),
            )

            self.assert_equal(
                topic_answer,
                topic_change_response,
                (
                    "PORT-T2 Topic Beta response "
                    f"for {course_id}"
                ),
            )

            self.assert_true(
                not tutor.last_repair_failed,
                (
                    "PORT-T2 unexpected repair "
                    "failure for "
                    f"{course_id}"
                ),
            )

            # =================================================
            # Course-invariant behavioral snapshot
            # =================================================

            results[
                course_id
            ] = {
                "final_intent": (
                    tutor
                    .last_learner_turn_intent
                    .intent
                ),
                "final_route": (
                    tutor
                    .last_learner_turn_routing
                    .route
                ),
                "final_mode": (
                    tutor
                    .last_tutoring_response_mode
                    .mode
                ),
                "state_evaluation": (
                    tutor.state.last_evaluation
                ),
                "last_evaluation": (
                    tutor.last_evaluation_result
                ),
                "decision": (
                    tutor.last_decision
                ),
                "memory_count": len(
                    memory_after_topic_change
                ),
                "retrieval_count": len(
                    retrieval_queries
                ),
                "repair_failed": (
                    tutor.last_repair_failed
                ),
            }

        # =====================================================
        # Electronics vs Mathematics must have identical
        # core behavioral outcome.
        # =====================================================

        self.assert_equal(
            results["electronics"],
            results["mathematics"],
            (
                "PORT-T2 multi-turn state-machine "
                "behavior changed across "
                "CourseProfiles."
            ),
        )

        self.pass_test(
            "PORT-T2 Cross-course multi-turn state machine",
            (
                "Initial tutoring, learner-answer "
                "evaluation, follow-up routing, "
                "active-question replacement, "
                "explicit topic change, pedagogical "
                "sequence reset, retrieval isolation, "
                "and ConversationMemory preservation "
                "remain behaviorally invariant across "
                "Electronics and Mathematics."
            ),
        )

    # =====================================================
    # 16.21B2.3 — PORT-T3
    #
    # Cross-course repair / fail-closed portability
    #
    # Scenario A:
    # unsupported factual generation
    # → verified evidence
    # → safe repair
    #
    # Scenario B:
    # unsupported factual generation
    # → no verified evidence
    # → fail closed
    #
    # Both behaviors must remain invariant across
    # Electronics and Mathematics CourseProfiles.
    # =====================================================

    def test_port_t3_cross_course_repair_fail_closed(
        self,
    ) -> None:

        course_ids = (
            "electronics",
            "mathematics",
        )

        safe_results = {}
        failed_results = {}

        # -------------------------------------------------
        # Synthetic factual boundary
        # -------------------------------------------------

        learner_message = (
            "ช่วยอธิบายความสัมพันธ์ของ "
            "Value กับ Input"
        )

        unsafe_response = (
            "Value ต้องมีค่าเท่ากับ 42 "
            "จึงจะสัมพันธ์กับ Input"
        )

        safe_repair = (
            "คุณคิดว่า Value "
            "มีความสัมพันธ์กับ Input อย่างไร?"
        )

        verified_quote = (
            "Value is related to Input."
        )

        full_context = (
            "Value is related to Input. "
            "No exact required value is specified."
        )

        for course_id in course_ids:

            # =================================================
            # Scenario A
            # Verified evidence → successful safe repair
            # =================================================

            with patch(
                "app.tutor.ACTIVE_COURSE",
                course_id,
            ):

                tutor = AITutor()

            self.assert_equal(
                getattr(
                    tutor.course_profile,
                    "course_id",
                    None,
                ),
                course_id,
                (
                    "PORT-T3 safe-path CourseProfile "
                    f"for {course_id}"
                ),
            )

            knowledge_result = SimpleNamespace(
                query="controlled relation query",
                context=full_context,
                sources=[],
                citations=[],
            )

            tutor.knowledge_service.retrieve = (
                lambda query, n_results=5:
                knowledge_result
            )

            tutor.relevance_gate.evaluate = (
                lambda query, knowledge_result:
                SimpleNamespace(
                    is_relevant=True,
                    confidence=1.0,
                    reason=(
                        "Controlled portability "
                        "relevance."
                    ),
                )
            )

            tutor.grounding_guard.evaluate = (
                lambda result, relevance:
                SimpleNamespace(
                    has_knowledge=True,
                    reason=(
                        "Controlled portability "
                        "knowledge available."
                    ),
                )
            )

            grounding_calls = []

            def safe_path_grounding(
                response,
                knowledge_context,
                task_name="response_validator",
            ):

                grounding_calls.append(
                    {
                        "response": response,
                        "knowledge_context": (
                            knowledge_context
                        ),
                        "task_name": task_name,
                    }
                )

                if response == unsafe_response:

                    return GroundingValidationResult(
                        status="unsupported",
                        confidence=1.0,
                        reason=(
                            "Exact value 42 is absent "
                            "from the evidence."
                        ),
                        issues=[
                            (
                                "Unsupported exact "
                                "required value."
                            )
                        ],
                    )

                if response == safe_repair:

                    return GroundingValidationResult(
                        status="supported",
                        confidence=1.0,
                        reason=(
                            "The repaired relation is "
                            "explicitly supported."
                        ),
                        issues=[],
                    )

                raise AssertionError(
                    "Unexpected PORT-T3 grounding "
                    f"candidate: {response!r}"
                )

            tutor.response_grounding_validator.validate = (
                safe_path_grounding
            )

            guiding_question_calls = []

            def controlled_guiding_question_validation(
                response,
                knowledge_context,
            ):

                guiding_question_calls.append(
                    {
                        "response": response,
                        "knowledge_context": (
                            knowledge_context
                        ),
                    }
                )

                return GuidingQuestionGroundingResult(
                    status="supported",
                    reason=(
                        "The recovered relation question "
                        "is answerable from verified evidence."
                    ),
                    issues=(),
                )

            tutor.guiding_question_grounding_validator.validate = (
                controlled_guiding_question_validation
            )

            tutor.pedagogical_response_validator.validate = (
                lambda *args, **kwargs:
                SimpleNamespace(
                    status="valid",
                    confidence=1.0,
                    reason=(
                        "Controlled portability "
                        "pedagogy."
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
                        "Controlled portability "
                        "response-mode pedagogy."
                    ),
                    issues=[],
                )
            )

            evidence_calls = []

            def verified_evidence_selection(
                learner_message,
                knowledge_context,
                validation_issues=None,
            ):

                evidence_calls.append(
                    {
                        "learner_message": (
                            learner_message
                        ),
                        "knowledge_context": (
                            knowledge_context
                        ),
                    }
                )

                return RepairEvidenceSelectionResult(
                    status="verified",
                    evidence_quotes=(
                        verified_quote,
                    ),
                    reason=(
                        "Exact verified portability "
                        "evidence selected."
                    ),
                    issues=(),
                )

            tutor.repair_evidence_selector.select = (
                verified_evidence_selection
            )

            repair_calls = []

            def successful_repair(
                *args,
                **kwargs,
            ):

                repair_calls.append(
                    kwargs
                )

                return safe_repair

            tutor.response_mode_repair_service.repair = (
                successful_repair
            )

            with patch(
                "app.tutor.chat_with_ai",
                return_value=unsafe_response,
            ) as generation:

                answer = tutor.respond(
                    learner_message
                )

            # ---------------------------------------------
            # Initial generation + verified repair
            # ---------------------------------------------

            self.assert_equal(
                generation.call_count,
                1,
                (
                    "PORT-T3 safe-path initial "
                    f"generation for {course_id}"
                ),
            )

            self.assert_equal(
                len(
                    evidence_calls
                ),
                1,
                (
                    "PORT-T3 evidence-selection "
                    f"count for {course_id}"
                ),
            )

            self.assert_equal(
                len(
                    repair_calls
                ),
                1,
                (
                    "PORT-T3 repair-generation "
                    f"count for {course_id}"
                ),
            )

            self.assert_equal(
                answer,
                safe_repair,
                (
                    "PORT-T3 safe final answer "
                    f"for {course_id}"
                ),
            )

            self.assert_true(
                tutor.last_response_repaired,
                (
                    "PORT-T3 safe path was not "
                    "marked repaired for "
                    f"{course_id}"
                ),
            )

            self.assert_true(
                not tutor.last_repair_failed,
                (
                    "PORT-T3 safe repair was marked "
                    "failed for "
                    f"{course_id}"
                ),
            )

            self.assert_equal(
                (
                    tutor
                    .last_repair_evidence_selection
                    .status
                ),
                "verified",
                (
                    "PORT-T3 verified evidence "
                    f"status for {course_id}"
                ),
            )

            self.assert_true(
                verified_quote
                in tutor.last_repair_evidence_context,
                (
                    "PORT-T3 verified evidence "
                    "missing from repair context for "
                    f"{course_id}"
                ),
            )

            self.assert_equal(
                (
                    repair_calls[0]
                    .get(
                        "knowledge_context"
                    )
                ),
                tutor.last_repair_evidence_context,
                (
                    "PORT-T3 repair generation "
                    "escaped verified evidence for "
                    f"{course_id}"
                ),
            )

            # Pure guiding-question repair does not need
            # factual semantic grounding validation.
            #
            # It must instead pass the knowledge-bounded
            # guiding-question validator against the exact
            # verified repair evidence context.
            # ---------------------------------------------

            repair_guiding_calls = [
                item
                for item in guiding_question_calls
                if item["response"] == safe_repair
            ]

            self.assert_true(
                len(
                    repair_guiding_calls
                )
                >= 1,
                (
                    "PORT-T3 repaired guiding question "
                    "was not revalidated for "
                    f"{course_id}"
                ),
            )

            self.assert_true(
                all(
                    (
                        item["knowledge_context"]
                        ==
                        tutor.last_repair_evidence_context
                    )
                    for item
                    in repair_guiding_calls
                ),
                (
                    "PORT-T3 repaired guiding-question "
                    "validation escaped verified evidence "
                    "for "
                    f"{course_id}"
                ),
            )
            safe_results[
                course_id
            ] = {
                "final_answer": answer,
                "response_repaired": (
                    tutor.last_response_repaired
                ),
                "repair_failed": (
                    tutor.last_repair_failed
                ),
                "evidence_status": (
                    tutor
                    .last_repair_evidence_selection
                    .status
                ),
                "repair_calls": len(
                    repair_calls
                ),
            }

            self.assert_true(
                all(
                    item["response"] != safe_repair
                    for item in grounding_calls
                ),
                (
                    "PORT-T3 pure guiding-question repair "
                    "unexpectedly entered factual semantic "
                    "grounding validation for "
                    f"{course_id}"
                ),
            )

            # =================================================
            # Scenario B
            # No evidence → fail closed
            # Fresh Tutor instance required.
            # =================================================

            with patch(
                "app.tutor.ACTIVE_COURSE",
                course_id,
            ):

                failed_tutor = AITutor()

            self.assert_equal(
                getattr(
                    failed_tutor.course_profile,
                    "course_id",
                    None,
                ),
                course_id,
                (
                    "PORT-T3 fail-path CourseProfile "
                    f"for {course_id}"
                ),
            )

            failed_tutor.knowledge_service.retrieve = (
                lambda query, n_results=5:
                knowledge_result
            )

            failed_tutor.relevance_gate.evaluate = (
                lambda query, knowledge_result:
                SimpleNamespace(
                    is_relevant=True,
                    confidence=1.0,
                    reason=(
                        "Controlled portability "
                        "relevance."
                    ),
                )
            )

            failed_tutor.grounding_guard.evaluate = (
                lambda result, relevance:
                SimpleNamespace(
                    has_knowledge=True,
                    reason=(
                        "Controlled portability "
                        "knowledge available."
                    ),
                )
            )

            failed_tutor.response_grounding_validator.validate = (
                lambda response,
                knowledge_context,
                task_name="response_validator":
                GroundingValidationResult(
                    status="unsupported",
                    confidence=1.0,
                    reason=(
                        "Requested exact value is "
                        "not supported."
                    ),
                    issues=[
                        (
                            "Unsupported exact "
                            "required value."
                        )
                    ],
                )
            )

            failed_tutor.pedagogical_response_validator.validate = (
                lambda *args, **kwargs:
                SimpleNamespace(
                    status="valid",
                    confidence=1.0,
                    reason=(
                        "Controlled portability "
                        "pedagogy."
                    ),
                    issues=[],
                )
            )

            failed_tutor.response_mode_pedagogical_validator.validate = (
                lambda *args, **kwargs:
                SimpleNamespace(
                    status="valid",
                    confidence=1.0,
                    reason=(
                        "Controlled portability "
                        "response-mode pedagogy."
                    ),
                    issues=[],
                )
            )

            no_evidence_calls = []

            def no_evidence_selection(
                learner_message,
                knowledge_context,
                validation_issues=None,
            ):

                no_evidence_calls.append(
                    learner_message
                )

                return RepairEvidenceSelectionResult(
                    status="no_evidence",
                    evidence_quotes=(),
                    reason=(
                        "No verified evidence supports "
                        "the requested exact value."
                    ),
                    issues=(),
                )

            failed_tutor.repair_evidence_selector.select = (
                no_evidence_selection
            )

            forbidden_repair_calls = []

            def forbidden_repair(
                *args,
                **kwargs,
            ):

                forbidden_repair_calls.append(
                    kwargs
                )

                raise AssertionError(
                    "PORT-T3 repair generation "
                    "must not run without verified "
                    "evidence."
                )

            failed_tutor.response_mode_repair_service.repair = (
                forbidden_repair
            )

            with patch(
                "app.tutor.chat_with_ai",
                return_value=unsafe_response,
            ) as failed_generation:

                fallback = failed_tutor.respond(
                    learner_message
                )

            # ---------------------------------------------
            # Fail-closed contract
            # ---------------------------------------------

            self.assert_equal(
                failed_generation.call_count,
                1,
                (
                    "PORT-T3 fail-path initial "
                    f"generation for {course_id}"
                ),
            )

            self.assert_equal(
                len(
                    no_evidence_calls
                ),
                1,
                (
                    "PORT-T3 no-evidence selection "
                    f"count for {course_id}"
                ),
            )

            self.assert_equal(
                len(
                    forbidden_repair_calls
                ),
                0,
                (
                    "PORT-T3 repair ran without "
                    "verified evidence for "
                    f"{course_id}"
                ),
            )

            self.assert_equal(
                (
                    failed_tutor
                    .last_repair_evidence_selection
                    .status
                ),
                "no_evidence",
                (
                    "PORT-T3 no-evidence status "
                    f"for {course_id}"
                ),
            )

            self.assert_true(
                failed_tutor.last_repair_failed,
                (
                    "PORT-T3 fail path was not "
                    "marked failed for "
                    f"{course_id}"
                ),
            )

            self.assert_true(
                not failed_tutor.last_response_repaired,
                (
                    "PORT-T3 fail path was "
                    "incorrectly marked repaired for "
                    f"{course_id}"
                ),
            )

            self.assert_true(
                isinstance(
                    fallback,
                    str,
                )
                and
                bool(
                    fallback.strip()
                ),
                (
                    "PORT-T3 fail-closed fallback "
                    "is empty for "
                    f"{course_id}"
                ),
            )

            self.assert_true(
                fallback != unsafe_response,
                (
                    "PORT-T3 unsafe response leaked "
                    "through fail-closed path for "
                    f"{course_id}"
                ),
            )

            # Failed exchange cannot enter memory.
            self.assert_equal(
                failed_tutor.memory.get_messages(),
                [],
                (
                    "PORT-T3 failed exchange "
                    "polluted memory for "
                    f"{course_id}"
                ),
            )

            self.assert_true(
                failed_tutor.original_question
                is None,
                (
                    "PORT-T3 failed sequence left "
                    "active question for "
                    f"{course_id}"
                ),
            )

            self.assert_equal(
                failed_tutor.waiting_for_response,
                False,
                (
                    "PORT-T3 failed sequence left "
                    "Tutor waiting for "
                    f"{course_id}"
                ),
            )

            failed_results[
                course_id
            ] = {
                "repair_failed": (
                    failed_tutor.last_repair_failed
                ),
                "response_repaired": (
                    failed_tutor.last_response_repaired
                ),
                "evidence_status": (
                    failed_tutor
                    .last_repair_evidence_selection
                    .status
                ),
                "memory_count": len(
                    failed_tutor
                    .memory
                    .get_messages()
                ),
                "active_question": (
                    failed_tutor.original_question
                ),
                "waiting": (
                    failed_tutor
                    .waiting_for_response
                ),
            }

        # =====================================================
        # Cross-course equivalence
        # =====================================================

        self.assert_equal(
            safe_results["electronics"],
            safe_results["mathematics"],
            (
                "PORT-T3 successful repair behavior "
                "changed across CourseProfiles."
            ),
        )

        self.assert_equal(
            failed_results["electronics"],
            failed_results["mathematics"],
            (
                "PORT-T3 fail-closed behavior "
                "changed across CourseProfiles."
            ),
        )

        self.pass_test(
            "PORT-T3 Cross-course repair and fail-closed",
            (
                "Verified-evidence factual repair "
                "and no-evidence fail-closed behavior "
                "remain invariant across Electronics "
                "and Mathematics CourseProfiles."
            ),
        )

    # =====================================================
    # 16.21B3 — MT-T7
    #
    # Long multi-turn cumulative state-drift test
    #
    # Turn 1  Initial Topic Alpha
    # Turn 2  Learner answer
    # Turn 3  Follow-up
    # Turn 4  Clarification
    # Turn 5  Explicit Topic Beta change
    # Turn 6  Learner answer on Topic Beta
    #
    # This validates cumulative state transitions on one
    # real AITutor instance.
    # =====================================================

    def test_mt_t7_long_multi_turn_state_drift(
        self,
    ) -> None:

        tutor = AITutor()

        # =================================================
        # Synthetic course-independent knowledge
        # =================================================

        alpha_context = (
            "Concept Alpha contains Part One "
            "and Part Two. "
            "Part One is a component of "
            "Concept Alpha."
        )

        beta_context = (
            "Concept Beta contains Part Three "
            "and Part Four. "
            "Part Three is a component of "
            "Concept Beta."
        )

        retrieval_queries = []

        def controlled_retrieve(
            query,
            n_results=5,
        ):

            retrieval_queries.append(
                query
            )

            if "Beta" in query:

                return SimpleNamespace(
                    query=query,
                    context=beta_context,
                    sources=[],
                    citations=[],
                )

            return SimpleNamespace(
                query=query,
                context=alpha_context,
                sources=[],
                citations=[],
            )

        tutor.knowledge_service.retrieve = (
            controlled_retrieve
        )

        tutor.relevance_gate.evaluate = (
            lambda query, knowledge_result:
            SimpleNamespace(
                is_relevant=True,
                confidence=1.0,
                reason=(
                    "Controlled long-sequence "
                    "relevance."
                ),
            )
        )

        tutor.grounding_guard.evaluate = (
            lambda result, relevance:
            SimpleNamespace(
                has_knowledge=True,
                reason=(
                    "Controlled long-sequence "
                    "knowledge is available."
                ),
            )
        )

        # =================================================
        # Frozen semantic layers are not under test here.
        # Keep the real AITutor orchestration/state machine.
        # =================================================

        tutor.response_grounding_validator.validate = (
            lambda *args, **kwargs:
            GroundingValidationResult(
                status="supported",
                confidence=1.0,
                reason=(
                    "Controlled long-sequence "
                    "response is grounded."
                ),
                issues=[],
            )
        )

        tutor.guiding_question_grounding_validator.validate = (
            lambda response, knowledge_context:
            GuidingQuestionGroundingResult(
                status="supported",
                reason=(
                    "Controlled long-sequence "
                    "guiding question is supported."
                ),
                issues=(),
            )
        )

        tutor.pedagogical_response_validator.validate = (
            lambda *args, **kwargs:
            SimpleNamespace(
                status="valid",
                confidence=1.0,
                reason=(
                    "Controlled long-sequence "
                    "pedagogy is valid."
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
                    "Controlled long-sequence "
                    "response mode is valid."
                ),
                issues=[],
            )
        )

        # =================================================
        # Learner messages
        # =================================================

        alpha_question = (
            "แนวคิด Alpha "
            "มีองค์ประกอบอะไรบ้าง?"
        )

        alpha_answer = (
            "มี Part One และ Part Two"
        )

        follow_up_question = (
            "แล้ว Part One คืออะไร?"
        )

        clarification = (
            "ช่วยอธิบาย Part One "
            "ให้ชัดเจนอีกครั้ง"
        )

        beta_question = (
            "ขอถามอีกเรื่อง "
            "แนวคิด Beta มีองค์ประกอบอะไรบ้าง?"
        )

        beta_answer = (
            "มี Part Three และ Part Four"
        )

        # =================================================
        # Controlled Tutor responses
        # =================================================

        alpha_initial_response = (
            "คุณคิดว่าแนวคิด Alpha "
            "มีองค์ประกอบหลักกี่ส่วน?"
        )

        alpha_answer_response = (
            "จากคำตอบของคุณ "
            "คุณคิดว่า Part One "
            "เป็นองค์ประกอบของ "
            "แนวคิด Alpha หรือไม่?"
        )

        follow_up_response = (
            "Part One เป็นองค์ประกอบหนึ่ง "
            "ของแนวคิด Alpha"
        )

        clarification_response = (
            "Part One เป็นองค์ประกอบหนึ่ง "
            "ของแนวคิด Alpha"
        )

        beta_initial_response = (
            "คุณคิดว่าแนวคิด Beta "
            "มีองค์ประกอบหลักกี่ส่วน?"
        )

        beta_answer_response = (
            "จากคำตอบของคุณ "
            "คุณคิดว่า Part Three "
            "เป็นองค์ประกอบของ "
            "แนวคิด Beta หรือไม่?"
        )

        generations = [
            alpha_initial_response,
            alpha_answer_response,
            follow_up_response,
            clarification_response,
            beta_initial_response,
            beta_answer_response,
        ]

        generated_index = 0

        def controlled_generation(
            messages,
            task_name="tutor",
            **kwargs,
        ):

            nonlocal generated_index

            if generated_index >= len(
                generations
            ):
                raise AssertionError(
                    "MT-T7 unexpected extra Tutor "
                    "generation call."
                )

            response = generations[
                generated_index
            ]

            generated_index += 1

            return response

        # =================================================
        # Controlled learner evaluation
        #
        # Only Turn 2 and Turn 6 should be evaluated.
        # =================================================

        evaluation_result = SimpleNamespace(
            classification="partial",
            confidence=1.0,
            reason=(
                "Controlled long-sequence "
                "partial learner answer."
            ),
            misconception=None,
        )

        with patch(
            "app.tutor.evaluate_response",
            return_value=evaluation_result,
        ) as evaluator:

            with patch(
                "app.tutor.chat_with_ai",
                side_effect=controlled_generation,
            ) as generation:

                # =========================================
                # TURN 1 — Initial Topic Alpha
                # =========================================

                turn_1 = tutor.respond(
                    alpha_question
                )

                self.assert_equal(
                    turn_1,
                    alpha_initial_response,
                    "MT-T7 Turn 1 response",
                )

                self.assert_equal(
                    tutor.original_question,
                    alpha_question,
                    (
                        "MT-T7 Turn 1 active "
                        "question"
                    ),
                )

                self.assert_true(
                    tutor.waiting_for_response,
                    (
                        "MT-T7 Turn 1 sequence "
                        "not active."
                    ),
                )

                # =========================================
                # TURN 2 — Learner answer Alpha
                # =========================================

                turn_2 = tutor.respond(
                    alpha_answer
                )

                self.assert_equal(
                    turn_2,
                    alpha_answer_response,
                    "MT-T7 Turn 2 response",
                )

                self.assert_true(
                    tutor.last_evaluation_result
                    is not None,
                    (
                        "MT-T7 Alpha learner answer "
                        "was not evaluated."
                    ),
                )

                self.assert_equal(
                    (
                        tutor
                        .last_evaluation_result
                        .classification
                    ),
                    "partial",
                    (
                        "MT-T7 Alpha evaluation"
                    ),
                )

                self.assert_equal(
                    tutor.state.last_evaluation,
                    "partial",
                    (
                        "MT-T7 Alpha state "
                        "evaluation"
                    ),
                )

                self.assert_true(
                    tutor.last_decision
                    is not None,
                    (
                        "MT-T7 Alpha learner answer "
                        "did not create a scaffolding "
                        "decision."
                    ),
                )

                # =========================================
                # TURN 3 — Follow-up Alpha
                # =========================================

                turn_3 = tutor.respond(
                    follow_up_question
                )

                self.assert_equal(
                    turn_3,
                    follow_up_response,
                    "MT-T7 Turn 3 response",
                )

                self.assert_equal(
                    (
                        tutor
                        .last_learner_turn_intent
                        .intent
                    ),
                    "follow_up_question",
                    (
                        "MT-T7 follow-up intent"
                    ),
                )

                self.assert_equal(
                    (
                        tutor
                        .last_learner_turn_routing
                        .route
                    ),
                    "follow_up_question",
                    (
                        "MT-T7 follow-up route"
                    ),
                )

                self.assert_equal(
                    (
                        tutor
                        .last_tutoring_response_mode
                        .mode
                    ),
                    "answer_then_guide",
                    (
                        "MT-T7 follow-up "
                        "response mode"
                    ),
                )

                # Follow-up becomes active question.
                self.assert_equal(
                    tutor.original_question,
                    follow_up_question,
                    (
                        "MT-T7 follow-up did not "
                        "replace active question."
                    ),
                )

                # Stale learner evaluation must be gone.
                self.assert_true(
                    tutor.last_evaluation_result
                    is None,
                    (
                        "MT-T7 stale Alpha "
                        "evaluation telemetry leaked "
                        "into follow-up."
                    ),
                )

                self.assert_true(
                    tutor.last_decision
                    is None,
                    (
                        "MT-T7 stale Alpha decision "
                        "leaked into follow-up."
                    ),
                )

                # =========================================
                # TURN 4 — Clarification Alpha
                # =========================================

                turn_4 = tutor.respond(
                    clarification
                )

                self.assert_equal(
                    turn_4,
                    clarification_response,
                    "MT-T7 Turn 4 response",
                )

                self.assert_equal(
                    (
                        tutor
                        .last_learner_turn_intent
                        .intent
                    ),
                    "clarification_question",
                    (
                        "MT-T7 clarification intent"
                    ),
                )

                self.assert_equal(
                    (
                        tutor
                        .last_learner_turn_routing
                        .route
                    ),
                    "clarification_question",
                    (
                        "MT-T7 clarification route"
                    ),
                )

                self.assert_equal(
                    (
                        tutor
                        .last_tutoring_response_mode
                        .mode
                    ),
                    "direct_clarification",
                    (
                        "MT-T7 clarification "
                        "response mode"
                    ),
                )

                # Clarification must remain attached to
                # current Alpha follow-up question.
                self.assert_equal(
                    tutor.original_question,
                    follow_up_question,
                    (
                        "MT-T7 clarification "
                        "incorrectly replaced the "
                        "active question."
                    ),
                )

                self.assert_true(
                    tutor.last_evaluation_result
                    is None,
                    (
                        "MT-T7 clarification reused "
                        "stale learner evaluation."
                    ),
                )

                # Snapshot full Alpha history.
                memory_before_beta = [
                    dict(item)
                    for item
                    in tutor.memory.get_messages()
                ]

                self.assert_equal(
                    len(
                        memory_before_beta
                    ),
                    8,
                    (
                        "MT-T7 Alpha session memory "
                        "count before topic change"
                    ),
                )

                # =========================================
                # TURN 5 — Explicit Topic Beta
                # =========================================

                turn_5 = tutor.respond(
                    beta_question
                )

                self.assert_equal(
                    turn_5,
                    beta_initial_response,
                    "MT-T7 Turn 5 response",
                )

                self.assert_equal(
                    (
                        tutor
                        .last_learner_turn_intent
                        .intent
                    ),
                    "topic_change",
                    (
                        "MT-T7 topic-change intent"
                    ),
                )

                self.assert_equal(
                    (
                        tutor
                        .last_learner_turn_routing
                        .route
                    ),
                    "topic_change",
                    (
                        "MT-T7 topic-change route"
                    ),
                )

                self.assert_equal(
                    (
                        tutor
                        .last_tutoring_response_mode
                        .mode
                    ),
                    "new_topic_scaffold",
                    (
                        "MT-T7 new-topic mode"
                    ),
                )

                self.assert_equal(
                    tutor.original_question,
                    beta_question,
                    (
                        "MT-T7 Beta topic did not "
                        "become active question."
                    ),
                )

                # Pedagogical state from Alpha must reset.
                self.assert_true(
                    tutor.state.last_evaluation
                    is None,
                    (
                        "MT-T7 Alpha evaluation "
                        "state leaked into Beta."
                    ),
                )

                self.assert_true(
                    tutor.last_evaluation_result
                    is None,
                    (
                        "MT-T7 Alpha evaluation "
                        "telemetry leaked into Beta."
                    ),
                )

                self.assert_true(
                    tutor.last_decision
                    is None,
                    (
                        "MT-T7 Alpha scaffolding "
                        "decision leaked into Beta."
                    ),
                )

                # =========================================
                # TURN 6 — Learner answer Beta
                # =========================================

                turn_6 = tutor.respond(
                    beta_answer
                )

                self.assert_equal(
                    turn_6,
                    beta_answer_response,
                    "MT-T7 Turn 6 response",
                )

        # =================================================
        # Global generation/evaluation counts
        # =================================================

        self.assert_equal(
            generation.call_count,
            6,
            (
                "MT-T7 Tutor generation count"
            ),
        )

        # Only:
        # - Turn 2 Alpha answer
        # - Turn 6 Beta answer
        self.assert_equal(
            evaluator.call_count,
            2,
            (
                "MT-T7 learner evaluation count"
            ),
        )

        # =================================================
        # Verify second evaluation belongs to Beta,
        # not stale Alpha state.
        # =================================================

        second_evaluation = (
            evaluator.call_args_list[1]
        )

        second_evaluation_kwargs = (
            second_evaluation.kwargs
        )

        self.assert_equal(
            (
                second_evaluation_kwargs
                .get(
                    "original_question"
                )
            ),
            beta_question,
            (
                "MT-T7 Beta answer was evaluated "
                "against a stale Alpha question."
            ),
        )

        self.assert_equal(
            (
                second_evaluation_kwargs
                .get(
                    "learner_response"
                )
            ),
            beta_answer,
            (
                "MT-T7 wrong learner response "
                "reached Beta evaluation."
            ),
        )

        # =================================================
        # Beta evaluation now owns current state.
        # =================================================

        self.assert_true(
            tutor.last_evaluation_result
            is not None,
            (
                "MT-T7 final Beta evaluation "
                "missing."
            ),
        )

        self.assert_equal(
            (
                tutor
                .last_evaluation_result
                .classification
            ),
            "partial",
            (
                "MT-T7 final Beta evaluation"
            ),
        )

        self.assert_equal(
            tutor.state.last_evaluation,
            "partial",
            (
                "MT-T7 final TutorState "
                "evaluation"
            ),
        )

        self.assert_true(
            tutor.last_decision
            is not None,
            (
                "MT-T7 Beta answer did not create "
                "a fresh scaffolding decision."
            ),
        )

        # =================================================
        # Retrieval sequence
        # =================================================

        self.assert_equal(
            len(
                retrieval_queries
            ),
            6,
            (
                "MT-T7 retrieval count"
            ),
        )

        # Topic-change retrieval must contain only Beta.
        beta_topic_query = (
            retrieval_queries[4]
        )

        self.assert_true(
            "Beta" in beta_topic_query,
            (
                "MT-T7 Beta topic missing from "
                "topic-change retrieval."
            ),
        )

        self.assert_true(
            "Alpha" not in beta_topic_query,
            (
                "MT-T7 Alpha leaked into Beta "
                "topic-change retrieval."
            ),
        )

        self.assert_true(
            "Part One"
            not in beta_topic_query,
            (
                "MT-T7 Alpha follow-up leaked "
                "into Beta topic-change retrieval."
            ),
        )

        # The following learner answer must also remain
        # inside Beta retrieval context.
        beta_answer_query = (
            retrieval_queries[5]
        )

        self.assert_true(
            "Beta" in beta_answer_query,
            (
                "MT-T7 Beta answer retrieval lost "
                "the active Beta topic."
            ),
        )

        self.assert_true(
            "Alpha" not in beta_answer_query,
            (
                "MT-T7 Alpha leaked back into "
                "Beta answer retrieval."
            ),
        )

        self.assert_equal(
            tutor.last_knowledge_result.context,
            beta_context,
            (
                "MT-T7 final knowledge context "
                "is not Beta."
            ),
        )

        # =================================================
        # Session memory preservation
        #
        # reset_learning_sequence() must not erase Alpha.
        # =================================================

        final_memory = (
            tutor.memory.get_messages()
        )

        self.assert_equal(
            (
                final_memory[
                    :len(
                        memory_before_beta
                    )
                ]
            ),
            memory_before_beta,
            (
                "MT-T7 topic change modified "
                "previous Alpha session history."
            ),
        )

        # Six successful exchanges:
        # 6 learner + 6 Tutor messages
        self.assert_equal(
            len(
                final_memory
            ),
            12,
            (
                "MT-T7 final ConversationMemory "
                "count"
            ),
        )

        self.assert_equal(
            final_memory[0]["content"],
            alpha_question,
            (
                "MT-T7 first session message"
            ),
        )

        self.assert_equal(
            final_memory[-2]["content"],
            beta_answer,
            (
                "MT-T7 final learner memory"
            ),
        )

        self.assert_equal(
            final_memory[-1]["content"],
            beta_answer_response,
            (
                "MT-T7 final Tutor memory"
            ),
        )

        # =================================================
        # Final active sequence must belong to Beta
        # =================================================

        self.assert_equal(
            tutor.original_question,
            beta_question,
            (
                "MT-T7 final active question "
                "does not belong to Beta."
            ),
        )

        self.assert_true(
            tutor.waiting_for_response,
            (
                "MT-T7 Beta sequence unexpectedly "
                "closed."
            ),
        )

        self.assert_true(
            not tutor.last_repair_failed,
            (
                "MT-T7 accumulated a stale "
                "repair-failure state."
            ),
        )

        self.pass_test(
            "MT-T7 Long multi-turn state drift",
            (
                "Across six consecutive Tutor turns, "
                "answer evaluation, follow-up routing, "
                "clarification, active-question "
                "retention, explicit topic change, "
                "learning-sequence reset, Beta "
                "re-evaluation, retrieval isolation, "
                "and session-memory preservation "
                "remain internally consistent without "
                "cumulative state drift."
            ),
        )
    # =====================================================
    # Runner
    # =====================================================

    def run(
        self,
    ) -> int:

        print()
        print(
            "Step 16.21 End-to-End Tutor "
            "Behavior Regression Suite"
        )
        print()

        tests = [
            (
                self
                .test_e2e_t1_normal_grounded_guiding_question
            ),
            (
                self
                .test_e2e_t2_out_of_course_fail_safe
            ),
            (
                self
                .test_e2e_t3_follow_up_answer_first
            ),
            (
                self
                .test_e2e_t4_clarification_direct_answer
            ),
            (
                self
                .test_e2e_t5_topic_change_retrieval_isolation
            ),
            (
                self
                .test_e2e_t6_unsupported_factual_repair
            ),
            (
                self
                .test_e2e_t7_c3_safe_guiding_recovery
            ),
            (
                self
                .test_e2e_t8_c3_fail_closed
            ),
            (
                self
                .test_e2e_t9_final_answer_telemetry_alignment
            ),
            (
                self
                .test_e2e_t10_failed_recovery_no_state_contamination
            ),
            (
                self
                .test_mt_t1_follow_up_retains_topic
            ),
            (
                self
                .test_mt_t2_clarification_preserves_local_context
            ),
            (
                self
                .test_mt_t3_topic_change_resets_sequence_preserves_history
            ),
            (
                self
                .test_mt_t4_failed_turn_next_turn_clean
            ),
            (
                self
                .test_mt_t5_out_of_course_next_in_course_clean
            ),
            (
                self
                .test_mt_t6_hard_session_reset
            ),
            (
                self
                .test_port_t1_cross_course_core_behavior
            ),
            (
                self
                .test_port_t2_cross_course_multi_turn_state_machine
            ),
            (
                self
                .test_port_t3_cross_course_repair_fail_closed
            ),
            (
                self
                .test_mt_t7_long_multi_turn_state_drift
            ),
        ]

        for test in tests:

            try:

                test()

            except Exception as exc:

                self.fail_test(
                    name=test.__name__,
                    details=str(exc),
                )

        print()

        for item in self.results:

            status = (
                "PASS"
                if item.passed
                else "FAIL"
            )

            print(
                f"[{status}] {item.name}"
            )

            if item.details:

                print(
                    f"       {item.details}"
                )

        passed = sum(
            item.passed
            for item in self.results
        )

        failed = (
            len(
                self.results
            )
            -
            passed
        )

        print()
        print("SUMMARY")
        print(
            f"Passed: {passed}"
        )
        print(
            f"Failed: {failed}"
        )

        return (
            0
            if failed == 0
            else 1
        )


def main() -> int:

    return (
        RegressionSuite16_21()
        .run()
    )


if __name__ == "__main__":

    raise SystemExit(
        main()
    )