from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

from app.tutor import AITutor

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
class TestResult:
    name: str
    passed: bool
    details: str = ""


class RegressionSuite16_16:

    def __init__(self):

        self.results: list[TestResult] = []



    # =========================================================
    # Controlled tutor environment
    # =========================================================

    def build_controlled_tutor(
        self,
    ) -> AITutor:

        tutor = AITutor()

        # -----------------------------------------------------
        # Controlled course knowledge result
        # -----------------------------------------------------

        knowledge_result = SimpleNamespace(
            query="controlled query",
            context=(
                "NPN transistor course knowledge. "
                "The transistor material contains "
                "Emitter, Base, Collector and "
                "NPN structure information."
            ),
            sources=[],
            citations=[],
        )

        tutor.knowledge_service.retrieve = (
            lambda query, n_results=5:
            knowledge_result
        )

        # -----------------------------------------------------
        # Retrieval is relevant
        # -----------------------------------------------------

        tutor.relevance_gate.evaluate = (
            lambda query, knowledge_result:
            SimpleNamespace(
                is_relevant=True,
                confidence=1.0,
                reason="controlled relevant result",
            )
        )

        # -----------------------------------------------------
        # Grounded knowledge is available
        # -----------------------------------------------------

        tutor.grounding_guard.evaluate = (
            lambda result, relevance:
            SimpleNamespace(
                has_knowledge=True,
                reason="controlled knowledge available",
            )
        )

        return tutor

    # =========================================================
    # Result helpers
    # =========================================================

    def pass_test(
        self,
        name: str,
        details: str = "",
    ) -> None:

        self.results.append(
            TestResult(
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
            TestResult(
                name=name,
                passed=False,
                details=details,
            )
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

    # =========================================================
    # Suite runner
    # =========================================================

    def run(self) -> int:

        print(
            "\nStep 16.16 Regression Suite\n"
        )

        tests = [
            self.test_t1_normal_grounded,
            self.test_t2_out_of_course_guard,
            self.test_t3_deterministic_terminology_repair,
            self.test_t4_pedagogical_only_repair,
            self.test_t5_factual_grounding_repair,
            self.test_t6_grounding_fail_closed,
            self.test_t7_pedagogical_fail_closed,
            self.test_t8_repair_attempt_limit,
        ]

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

    def print_summary(self) -> None:

        print()

        for item in self.results:

            status = (
                "PASS"
                if item.passed
                else "FAIL"
            )

            print(
                f"[{status}] "
                f"{item.name}"
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
            len(self.results)
            - passed
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

    # =========================================================
    # Tests
    # =========================================================

    def test_t1_normal_grounded(
        self,
    ) -> None:

        test_name = (
            "T1 Normal grounded response"
        )

        tutor = (
            self.build_controlled_tutor()
        )

        normal_response = (
            "คุณคิดว่าในทรานซิสเตอร์ NPN "
            "จะมีส่วนประกอบหลักอะไรบ้าง?"
        )

        with patch(
            "app.tutor.chat_with_ai",
            return_value=normal_response,
        ):

            answer = tutor.respond(
                "ทรานซิสเตอร์ NPN "
                "มีโครงสร้างอย่างไร"
            )

        debug = tutor.get_debug_info()

        self.assert_equal(
            answer,
            normal_response,
            "Normal answer changed unexpectedly",
        )

        self.assert_equal(
            debug[
                "grounding_claim_precheck_status"
            ],
            "no_claims",
            "Grounding precheck",
        )

        self.assert_equal(
            debug[
                "validation_status"
            ],
            "supported",
            "Grounding validation",
        )

        self.assert_equal(
            debug[
                "pedagogical_status"
            ],
            "valid",
            "Pedagogical validation",
        )

        self.assert_equal(
            debug[
                "escalation_action"
            ],
            "accept",
            "Escalation action",
        )

        self.assert_equal(
            debug[
                "repair_attempts"
            ],
            0,
            "Repair attempts",
        )

        self.assert_equal(
            debug[
                "repair_failed"
            ],
            False,
            "Repair failure state",
        )

        self.pass_test(
            test_name,
            (
                "normal response accepted "
                "without repair"
            ),
        )

    def test_t2_out_of_course_guard(
    self,
    ) -> None:

        test_name = (
            "T2 Out-of-course guard"
        )

        tutor = AITutor()

        knowledge_result = SimpleNamespace(
            query="controlled query",
            context="",
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
                is_relevant=False,
                confidence=1.0,
                reason="controlled out-of-course",
            )
        )

        tutor.grounding_guard.evaluate = (
            lambda result, relevance:
            SimpleNamespace(
                has_knowledge=False,
                reason=(
                    "retrieved_context_not_relevant"
                ),
            )
        )

        tutor_called = {
            "value": False,
        }

        def forbidden_tutor_call(
            *args,
            **kwargs,
        ):

            tutor_called["value"] = True

            raise AssertionError(
                "Tutor LLM must not be called "
                "for out-of-course input."
            )

        with patch(
            "app.tutor.chat_with_ai",
            side_effect=forbidden_tutor_call,
        ):

            answer = tutor.respond(
                "กฎของเลขยกกำลังคืออะไร"
            )

        debug = tutor.get_debug_info()

        self.assert_equal(
            tutor_called["value"],
            False,
            "Tutor LLM call state",
        )

        self.assert_equal(
            debug[
                "grounding_has_knowledge"
            ],
            False,
            "Grounding availability",
        )

        self.assert_equal(
            tutor.waiting_for_response,
            False,
            "waiting_for_response",
        )

        self.assert_equal(
            tutor.original_question,
            None,
            "original_question",
        )

        self.assert_true(
            bool(answer.strip()),
            "Out-of-course fallback is empty",
        )

        self.pass_test(
            test_name,
            (
                "out-of-course request blocked "
                "before tutor generation"
            ),
        )

    def test_t3_deterministic_terminology_repair(
    self,
    ) -> None:

        test_name = (
            "T3 Deterministic terminology repair"
        )

        tutor = (
            self.build_controlled_tutor()
        )

        # ---------------------------------------------------------
        # Step 16.20 compatibility isolation
        #
        # T3 owns deterministic terminology repair only.
        # Guiding-question grounding is a later validation layer
        # and must not convert this controlled terminology-only
        # case into a combined pedagogical/GQ failure.
        # ---------------------------------------------------------

        tutor.guiding_question_grounding_validator.validate = (
            lambda response, knowledge_context:
            GuidingQuestionGroundingResult(
                status="supported",
                reason=(
                    "Controlled Step 16.16 T3 harness "
                    "treats guiding-question grounding as "
                    "already validated by the later "
                    "Step 16.20 layer."
                ),
                issues=(),
            )
        )

        bad_term_response = (
            "คุณคิดว่าอิเล็กเตอร์"
            "ทำหน้าที่อะไร?"
        )

        with patch(
            "app.tutor.chat_with_ai",
            return_value=bad_term_response,
        ):

            answer = tutor.respond(
                "ทรานซิสเตอร์ NPN "
                "มีโครงสร้างอย่างไร"
            )

        debug = tutor.get_debug_info()

        self.assert_equal(
            debug[
                "embedded_claim_term_guard_status"
            ],
            "terminology_issue",
            "Terminology guard",
        )

        self.assert_equal(
            debug[
                "escalation_action"
            ],
            "deterministic_repair",
            "Escalation action",
        )

        self.assert_equal(
            debug[
                "repair_mode"
            ],
            "deterministic",
            "Repair mode",
        )

        self.assert_equal(
            debug[
                "repair_attempts"
            ],
            1,
            "Repair attempts",
        )

        self.assert_equal(
            debug[
                "repair_status"
            ],
            "supported",
            "Post-repair grounding",
        )

        self.assert_equal(
            debug[
                "repair_pedagogical_status"
            ],
            "valid",
            "Post-repair pedagogy",
        )

        self.assert_true(
            "อิมิตเตอร์ (Emitter)"
            in answer,
            "Terminology was not repaired",
        )

        self.assert_true(
            "อิเล็กเตอร์"
            not in answer,
            "Incorrect terminology remains",
        )

        self.pass_test(
            test_name,
            (
                "terminology repaired by "
                "deterministic path"
            ),
        )

    def test_t4_pedagogical_only_repair(
    self,
    ) -> None:

        test_name = (
            "T4 Pedagogical-only repair"
        )

        tutor = (
            self.build_controlled_tutor()
        )

        initial_response = (
            "ลองศึกษาทรานซิสเตอร์ NPN "
            "เพิ่มเติม"
        )

        repaired_response = (
            "คุณคิดว่าโครงสร้างของ"
            "ทรานซิสเตอร์ NPN "
            "ประกอบด้วยส่วนใดบ้าง?"
        )

        repair_calls = {
            "count": 0,
        }

        def controlled_repair(
            **kwargs,
        ):

            repair_calls["count"] += 1

            return repaired_response

        tutor.response_repair_service.repair = (
            controlled_repair
        )

        with patch(
            "app.tutor.chat_with_ai",
            return_value=initial_response,
        ):

            answer = tutor.respond(
                "ทรานซิสเตอร์ NPN "
                "มีโครงสร้างอย่างไร"
            )

        debug = tutor.get_debug_info()

        self.assert_equal(
            debug[
                "grounding_claim_precheck_status"
            ],
            "no_claims",
            "Initial grounding precheck",
        )

        self.assert_equal(
            debug[
                "validation_status"
            ],
            "supported",
            "Initial grounding validation",
        )

        self.assert_equal(
            debug[
                "pedagogical_status"
            ],
            "violation",
            "Initial pedagogy",
        )

        self.assert_equal(
            debug[
                "escalation_action"
            ],
            "llm_repair",
            "Escalation action",
        )

        self.assert_equal(
            repair_calls["count"],
            1,
            "Repair service call count",
        )

        self.assert_equal(
            debug[
                "repair_attempts"
            ],
            1,
            "Repair attempts",
        )

        self.assert_equal(
            debug[
                "repair_status"
            ],
            "supported",
            "Post-repair grounding",
        )

        self.assert_equal(
            debug[
                "repair_pedagogical_status"
            ],
            "valid",
            "Post-repair pedagogy",
        )

        self.assert_equal(
            answer,
            repaired_response,
            "Accepted repaired response",
        )

        self.pass_test(
            test_name,
            (
                "pedagogical failure repaired "
                "with one LLM repair attempt"
            ),
        )

    def test_t5_factual_grounding_repair(
    self,
    ) -> None:

        test_name = (
            "T5 Factual grounding repair"
        )

        tutor = (
            self.build_controlled_tutor()
        )

        initial_response = (
            "ทรานซิสเตอร์ NPN "
            "มีโครงสร้าง X-X-X"
        )

        repaired_response = (
            "คุณคิดว่าโครงสร้างของ"
            "ทรานซิสเตอร์ NPN "
            "ประกอบด้วยส่วนใดบ้าง?"
        )

        grounding_validator_calls = {
            "count": 0,
        }

        repair_calls = {
            "count": 0,
        }

        evidence_selector_calls = {
            "count": 0,
        }

        verified_evidence = (
            RepairEvidenceSelectionResult(
                status="verified",
                evidence_quotes=(
                    "Emitter, Base, Collector",
                ),
                reason=(
                    "Controlled verified course evidence."
                ),
                issues=(),
            )
        )

        def controlled_grounding_validation(
            response,
            knowledge_context,
            task_name=None,
        ):

            grounding_validator_calls[
                "count"
            ] += 1

            return GroundingValidationResult(
                status="unsupported",
                confidence=1.0,
                reason=(
                    "Controlled unsupported "
                    "factual claim."
                ),
                issues=[
                    "X-X-X is unsupported.",
                ],
            )

        def controlled_evidence_selection(
            learner_message,
            knowledge_context,
            validation_issues=None,
        ):

            evidence_selector_calls[
                "count"
            ] += 1

            return verified_evidence

        def controlled_repair(
            **kwargs,
        ):

            repair_calls["count"] += 1

            return repaired_response

        tutor.response_grounding_validator.validate = (
            controlled_grounding_validation
        )

        tutor.repair_evidence_selector.select = (
            controlled_evidence_selection
        )

        tutor.response_repair_service.repair = (
            controlled_repair
        )

        with patch(
            "app.tutor.chat_with_ai",
            return_value=initial_response,
        ):

            answer = tutor.respond(
                "ทรานซิสเตอร์ NPN "
                "มีโครงสร้างอย่างไร"
            )

        debug = tutor.get_debug_info()

        self.assert_equal(
            debug[
                "grounding_claim_precheck_status"
            ],
            "needs_validation",
            "Initial grounding precheck",
        )

        self.assert_equal(
            grounding_validator_calls[
                "count"
            ],
            1,
            "Initial grounding validator calls",
        )

        self.assert_equal(
            debug[
                "validation_status"
            ],
            "unsupported",
            "Initial grounding validation",
        )

        self.assert_equal(
            debug[
                "escalation_action"
            ],
            "llm_repair",
            "Escalation action",
        )

        self.assert_equal(
            evidence_selector_calls[
                "count"
            ],
            1,
            "Repair evidence selector call count",
        )

        self.assert_equal(
            debug[
                "repair_evidence_status"
            ],
            "verified",
            "Repair evidence status",
        )       

        self.assert_equal(
            repair_calls["count"],
            1,
            "Repair service call count",
        )

        self.assert_equal(
            debug[
                "repair_status"
            ],
            "supported",
            "Post-repair grounding",
        )

        self.assert_equal(
            debug[
                "repair_pedagogical_status"
            ],
            "valid",
            "Post-repair pedagogy",
        )

        self.assert_equal(
            answer,
            repaired_response,
            "Accepted repaired response",
        )

        self.pass_test(
            test_name,
            (
                "unsupported factual claim "
                "was semantically validated "
                "and repaired"
            ),
        )

    def test_t6_grounding_fail_closed(
    self,
    ) -> None:

        test_name = (
            "T6 Grounding fail-closed"
        )

        tutor = (
            self.build_controlled_tutor()
        )

        bad_response = (
            "ทรานซิสเตอร์ NPN "
            "มีโครงสร้าง X-X-X"
        )

        def always_unsupported(
            response,
            knowledge_context,
            task_name=None,
        ):

            return GroundingValidationResult(
                status="unsupported",
                confidence=1.0,
                reason=(
                    "Controlled grounding failure."
                ),
                issues=[
                    "Controlled unsupported claim.",
                ],
            )

        verified_evidence = (
            RepairEvidenceSelectionResult(
                status="verified",
                evidence_quotes=(
                    "Emitter, Base, Collector",
                ),
                reason=(
                    "Controlled verified evidence "
                    "for grounding fail-closed test."
                ),
                issues=(),
            )
        )

        def controlled_evidence_selection(
            learner_message,
            knowledge_context,
            validation_issues=None,
        ):

            return verified_evidence

        tutor.response_grounding_validator.validate = (
            always_unsupported
        )

        tutor.repair_evidence_selector.select = (
            controlled_evidence_selection
        )

        tutor.response_repair_service.repair = (
            lambda **kwargs:
            bad_response
        )
                
        tutor.response_grounding_validator.validate = (
            always_unsupported
        )

        tutor.response_repair_service.repair = (
            lambda **kwargs:
            bad_response
        )

        with patch(
            "app.tutor.chat_with_ai",
            return_value=bad_response,
        ):

            answer = tutor.respond(
                "ทรานซิสเตอร์ NPN "
                "มีโครงสร้างอย่างไร"
            )

        debug = tutor.get_debug_info()

        self.assert_equal(
            debug[
                "repair_failed"
            ],
            True,
            "Repair failure",
        )

        self.assert_equal(
            debug[
                "repair_failure_type"
            ],
            "grounding",
            "Repair failure type",
        )

        self.assert_equal(
            tutor.waiting_for_response,
            False,
            "waiting_for_response",
        )

        self.assert_equal(
            tutor.original_question,
            None,
            "original_question",
        )

        self.assert_true(
            "X-X-X" not in answer,
            (
                "Unsupported repaired response "
                "escaped fail-closed gate"
            ),
        )

        self.pass_test(
            test_name,
            (
                "unsupported repaired response "
                "was blocked"
            ),
        )

    def test_t7_pedagogical_fail_closed(
    self,
    ) -> None:

        test_name = (
            "T7 Pedagogical fail-closed"
        )

        tutor = (
            self.build_controlled_tutor()
        )

        invalid_response = (
            "ลองศึกษาทรานซิสเตอร์ NPN "
            "เพิ่มเติม"
        )

        tutor.response_repair_service.repair = (
            lambda **kwargs:
            invalid_response
        )

        with patch(
            "app.tutor.chat_with_ai",
            return_value=invalid_response,
        ):

            answer = tutor.respond(
                "ทรานซิสเตอร์ NPN "
                "มีโครงสร้างอย่างไร"
            )

        debug = tutor.get_debug_info()

        self.assert_equal(
            debug[
                "repair_failed"
            ],
            True,
            "Repair failure",
        )

        self.assert_equal(
            debug[
                "repair_failure_type"
            ],
            "pedagogical",
            "Repair failure type",
        )

        self.assert_equal(
            tutor.waiting_for_response,
            False,
            "waiting_for_response",
        )

        self.assert_equal(
            tutor.original_question,
            None,
            "original_question",
        )

        self.assert_true(
            answer != invalid_response,
            (
                "Pedagogically invalid "
                "repaired response escaped"
            ),
        )

        self.pass_test(
            test_name,
            (
                "pedagogically invalid repair "
                "was blocked"
            ),
        )

    def test_t8_repair_attempt_limit(
    self,
    ) -> None:

        test_name = (
            "T8 Repair attempt limit"
        )

        tutor = (
            self.build_controlled_tutor()
        )

        invalid_response = (
            "ลองศึกษาทรานซิสเตอร์ NPN "
            "เพิ่มเติม"
        )

        repair_called = {
            "value": False,
        }

        def forbidden_repair(
            **kwargs,
        ):

            repair_called["value"] = True

            raise AssertionError(
                "Repair service must not be called "
                "after repair limit is reached."
            )

        tutor.response_repair_service.repair = (
            forbidden_repair
        )

        # -----------------------------------------------------
        # Simulate an already consumed repair quota.
        #
        # respond() resets attempts at the beginning,
        # so inject the limit after generation by using
        # a controlled escalation-policy wrapper.
        # -----------------------------------------------------

        original_decide = (
            tutor.validation_escalation_policy
            .decide
        )

        def controlled_decide(
            *args,
            **kwargs,
        ):

            decision = original_decide(
                *args,
                **kwargs,
            )

            tutor.last_repair_attempts = (
                tutor.MAX_REPAIR_ATTEMPTS
            )

            return decision

        tutor.validation_escalation_policy.decide = (
            controlled_decide
        )

        with patch(
            "app.tutor.chat_with_ai",
            return_value=invalid_response,
        ):

            answer = tutor.respond(
                "ทรานซิสเตอร์ NPN "
                "มีโครงสร้างอย่างไร"
            )

        debug = tutor.get_debug_info()

        self.assert_equal(
            debug[
                "repair_limit_reached"
            ],
            True,
            "Repair limit state",
        )

        self.assert_equal(
            debug[
                "repair_failed"
            ],
            True,
            "Repair failure state",
        )

        self.assert_equal(
            debug[
                "repair_failure_type"
            ],
            "repair_limit",
            "Repair failure type",
        )

        self.assert_equal(
            repair_called["value"],
            False,
            "Repair service call state",
        )

        self.assert_equal(
            debug[
                "repair_attempts"
            ],
            1,
            "Repair attempts",
        )

        self.assert_true(
            bool(answer.strip()),
            "Repair-limit fallback is empty",
        )

        self.pass_test(
            test_name,
            (
                "repair blocked after "
                "maximum attempt limit"
            ),
        )

def main():

    suite = RegressionSuite16_16()

    exit_code = suite.run()

    raise SystemExit(
        exit_code
    )


if __name__ == "__main__":

    main()