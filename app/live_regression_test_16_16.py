from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

from app.tutor import AITutor

from app.courses.loader import (
    CourseProfileLoader,
)

from app.document_loader import (
    load_pdf_pages,
)

from app.services.knowledge_service import (
    KnowledgeService,
)

@dataclass
class LiveTestResult:
    name: str
    passed: bool
    details: str = ""


class LiveRegressionSuite16_16:
    """
    Live integration regression suite for Step 16.16.

    Unlike regression_test_16_16.py, this suite uses:

    - real CourseProfile
    - real ChromaDB
    - real embeddings
    - real RAG retrieval
    - real relevance gate
    - real Tutor LLM
    - real validation pipeline

    No Tutor/validator/repair response is mocked.

    The only patch used here changes ACTIVE_COURSE before
    constructing AITutor so multiple courses can be tested
    in one Python process.
    """

    ELECTRONICS_QUERY = (
        "ทรานซิสเตอร์ NPN มีโครงสร้างอย่างไร"
    )

    MATHEMATICS_QUERY = (
        "กฎของเลขยกกำลัง "
        "(Laws of indices) คืออะไร"
    )

    # Hard safety ceiling.
    #
    # The normal target is usually lower than this,
    # but a live LLM response may legitimately require
    # semantic validation or repair.
    MAX_GROUNDED_CALLS = 7

    # Out-of-course should normally use only relevance_gate.
    MAX_OUT_OF_COURSE_CALLS = 2

    def __init__(self):

        self.results: list[LiveTestResult] = []

        # Store debug snapshots so later tests can audit
        # telemetry without repeating unnecessary LLM calls.
        self.snapshots = {}

    # =========================================================
    # Assertion helpers
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

    def assert_less_equal(
        self,
        actual,
        expected,
        message: str,
    ) -> None:

        if actual > expected:

            raise AssertionError(
                f"{message}: "
                f"expected <= {expected!r}, "
                f"actual={actual!r}"
            )

    # =========================================================
    # Result helpers
    # =========================================================

    def pass_test(
        self,
        name: str,
        details: str = "",
    ) -> None:

        self.results.append(
            LiveTestResult(
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
            LiveTestResult(
                name=name,
                passed=False,
                details=details,
            )
        )

    # =========================================================
    # Course helpers
    # =========================================================

    def build_tutor(
        self,
        course_id: str,
    ) -> AITutor:
        """
        Build a real AITutor for the requested course.

        AITutor imports ACTIVE_COURSE into app.tutor,
        therefore patch that module-level value only
        during construction.

        All resulting services remain real.
        """

        with patch(
            "app.tutor.ACTIVE_COURSE",
            course_id,
        ):

            tutor = AITutor()

        self.assert_equal(
            tutor.course_profile.course_id,
            course_id,
            "AITutor active course",
        )

        return tutor

    def load_profile(
        self,
        course_id: str,
    ):

        return (
            CourseProfileLoader()
            .load(
                course_id
            )
        )

    def get_allowed_pdf_pages(
        self,
        course_id: str,
    ) -> dict[str, int]:

        profile = self.load_profile(
            course_id
        )

        folder = Path(
            profile.document_path
        )

        result = {}

        for pdf_path in sorted(
            folder.glob("*.pdf")
        ):

            pages = load_pdf_pages(
                str(pdf_path)
            )

            if not pages:
                continue

            result[
                pdf_path.name
            ] = max(
                item["page"]
                for item in pages
            )

        return result

    def assert_sources_belong_to_course(
        self,
        course_id: str,
        sources: list[dict],
    ) -> None:

        allowed = (
            self.get_allowed_pdf_pages(
                course_id
            )
        )

        self.assert_true(
            bool(allowed),
            (
                f"No PDF documents found "
                f"for course {course_id!r}"
            ),
        )

        self.assert_true(
            bool(sources),
            (
                f"No RAG sources returned "
                f"for course {course_id!r}"
            ),
        )

        for source in sources:

            source_name = source.get(
                "source"
            )

            page = source.get(
                "page"
            )

            self.assert_true(
                source_name in allowed,
                (
                    "Cross-course or unknown "
                    f"source detected: {source_name!r}"
                ),
            )

            self.assert_true(
                isinstance(page, int)
                and
                1 <= page <= allowed[source_name],
                (
                    "Invalid citation page: "
                    f"{source_name!r} page={page!r}"
                ),
            )

    # =========================================================
    # Live tests
    # =========================================================

    def test_l1_electronics_grounded(
        self,
    ) -> None:

        name = (
            "L1 Electronics grounded question"
        )

        tutor = self.build_tutor(
            "electronics"
        )

        answer = tutor.respond(
            self.ELECTRONICS_QUERY
        )

        debug = tutor.get_debug_info()

        self.assert_equal(
            debug.get(
                "relevance_is_relevant"
            ),
            True,
            "Electronics relevance",
        )

        self.assert_equal(
            debug.get(
                "grounding_has_knowledge"
            ),
            True,
            "Electronics grounding",
        )

        self.assert_true(
            debug.get(
                "generated_answer"
            )
            is not None,
            "Tutor generation is missing",
        )

        self.assert_equal(
            debug.get(
                "repair_failed"
            ),
            False,
            "Electronics repair failure",
        )

        self.assert_true(
            bool(answer.strip()),
            "Electronics answer is empty",
        )

        self.assert_sources_belong_to_course(
            "electronics",
            debug.get(
                "knowledge_sources",
                [],
            ),
        )

        calls = (
            debug["ai_usage"]["calls"]
        )

        self.assert_less_equal(
            calls,
            self.MAX_GROUNDED_CALLS,
            "Electronics LLM call budget",
        )

        self.snapshots[
            "electronics_grounded"
        ] = debug

        self.pass_test(
            name,
            (
                f"relevant=True, "
                f"grounded=True, "
                f"calls={calls}, "
                f"action="
                f"{debug.get('escalation_action')}"
            ),
        )

    def test_l2_electronics_rejects_math(
        self,
    ) -> None:

        name = (
            "L2 Electronics rejects mathematics"
        )

        tutor = self.build_tutor(
            "electronics"
        )

        answer = tutor.respond(
            self.MATHEMATICS_QUERY
        )

        debug = tutor.get_debug_info()

        self.assert_equal(
            debug.get(
                "relevance_is_relevant"
            ),
            False,
            "Out-of-course relevance",
        )

        self.assert_equal(
            debug.get(
                "grounding_has_knowledge"
            ),
            False,
            "Out-of-course grounding",
        )

        self.assert_equal(
            debug.get(
                "generated_answer"
            ),
            None,
            "Tutor should not generate answer",
        )

        self.assert_equal(
            tutor.waiting_for_response,
            False,
            "Out-of-course waiting state",
        )

        self.assert_equal(
            tutor.original_question,
            None,
            "Out-of-course original question",
        )

        self.assert_true(
            bool(answer.strip()),
            "Out-of-course fallback is empty",
        )

        calls = (
            debug["ai_usage"]["calls"]
        )

        self.assert_less_equal(
            calls,
            self.MAX_OUT_OF_COURSE_CALLS,
            "Out-of-course LLM call budget",
        )

        self.snapshots[
            "electronics_out_of_course"
        ] = debug

        self.pass_test(
            name,
            (
                f"blocked before tutor generation, "
                f"calls={calls}"
            ),
        )

    def test_l3_mathematics_grounded(
        self,
    ) -> None:

        name = (
            "L3 Mathematics grounded question"
        )

        tutor = self.build_tutor(
            "mathematics"
        )

        answer = tutor.respond(
            self.MATHEMATICS_QUERY
        )

        debug = tutor.get_debug_info()

        self.assert_equal(
            debug.get(
                "relevance_is_relevant"
            ),
            True,
            "Mathematics relevance",
        )

        self.assert_equal(
            debug.get(
                "grounding_has_knowledge"
            ),
            True,
            "Mathematics grounding",
        )

        self.assert_true(
            debug.get(
                "generated_answer"
            )
            is not None,
            "Mathematics tutor generation missing",
        )

        self.assert_equal(
            debug.get(
                "repair_failed"
            ),
            False,
            "Mathematics repair failure",
        )

        self.assert_true(
            bool(answer.strip()),
            "Mathematics answer is empty",
        )

        self.assert_sources_belong_to_course(
            "mathematics",
            debug.get(
                "knowledge_sources",
                [],
            ),
        )

        calls = (
            debug["ai_usage"]["calls"]
        )

        self.assert_less_equal(
            calls,
            self.MAX_GROUNDED_CALLS,
            "Mathematics LLM call budget",
        )

        self.snapshots[
            "mathematics_grounded"
        ] = debug

        self.pass_test(
            name,
            (
                f"relevant=True, "
                f"grounded=True, "
                f"calls={calls}, "
                f"action="
                f"{debug.get('escalation_action')}"
            ),
        )

    def test_l4_mathematics_rejects_electronics(
        self,
    ) -> None:

        name = (
            "L4 Mathematics rejects electronics"
        )

        tutor = self.build_tutor(
            "mathematics"
        )

        answer = tutor.respond(
            self.ELECTRONICS_QUERY
        )

        debug = tutor.get_debug_info()

        self.assert_equal(
            debug.get(
                "relevance_is_relevant"
            ),
            False,
            "Mathematics out-of-course relevance",
        )

        self.assert_equal(
            debug.get(
                "grounding_has_knowledge"
            ),
            False,
            "Mathematics out-of-course grounding",
        )

        self.assert_equal(
            debug.get(
                "generated_answer"
            ),
            None,
            "Tutor should not generate answer",
        )

        self.assert_equal(
            tutor.waiting_for_response,
            False,
            "Out-of-course waiting state",
        )

        self.assert_equal(
            tutor.original_question,
            None,
            "Out-of-course original question",
        )

        self.assert_true(
            bool(answer.strip()),
            "Out-of-course fallback is empty",
        )

        calls = (
            debug["ai_usage"]["calls"]
        )

        self.assert_less_equal(
            calls,
            self.MAX_OUT_OF_COURSE_CALLS,
            "Out-of-course LLM call budget",
        )

        self.snapshots[
            "mathematics_out_of_course"
        ] = debug

        self.pass_test(
            name,
            (
                f"blocked before tutor generation, "
                f"calls={calls}"
            ),
        )

    def test_l5_chroma_course_isolation(
        self,
    ) -> None:

        name = (
            "L5 Multi-course Chroma isolation"
        )

        electronics = (
            self.load_profile(
                "electronics"
            )
        )

        mathematics = (
            self.load_profile(
                "mathematics"
            )
        )

        electronics_result = (
            KnowledgeService(
                course_profile=electronics
            )
            .retrieve(
                self.ELECTRONICS_QUERY,
                n_results=5,
            )
        )

        mathematics_result = (
            KnowledgeService(
                course_profile=mathematics
            )
            .retrieve(
                self.MATHEMATICS_QUERY,
                n_results=5,
            )
        )

        electronics_sources = [
            {
                "source": item.source,
                "page": item.page,
            }
            for item
            in electronics_result.sources
        ]

        mathematics_sources = [
            {
                "source": item.source,
                "page": item.page,
            }
            for item
            in mathematics_result.sources
        ]

        self.assert_sources_belong_to_course(
            "electronics",
            electronics_sources,
        )

        self.assert_sources_belong_to_course(
            "mathematics",
            mathematics_sources,
        )

        electronics_names = {
            item["source"]
            for item in electronics_sources
        }

        mathematics_names = {
            item["source"]
            for item in mathematics_sources
        }

        self.assert_true(
            electronics_names.isdisjoint(
                mathematics_names
            ),
            (
                "Course Chroma sources overlap: "
                f"{electronics_names & mathematics_names}"
            ),
        )

        self.pass_test(
            name,
            (
                "electronics and mathematics "
                "retrieval stores are isolated"
            ),
        )

    def test_l6_citation_integrity(
        self,
    ) -> None:

        name = (
            "L6 Citation and source integrity"
        )

        test_cases = (
            (
                "electronics",
                self.ELECTRONICS_QUERY,
            ),
            (
                "mathematics",
                self.MATHEMATICS_QUERY,
            ),
        )

        for (
            course_id,
            query,
        ) in test_cases:

            profile = self.load_profile(
                course_id
            )

            result = (
                KnowledgeService(
                    course_profile=profile
                )
                .retrieve(
                    query=query,
                    n_results=5,
                )
            )

            source_pairs = [
                (
                    item.source,
                    item.page,
                )
                for item
                in result.sources
            ]

            citation_pairs = [
                (
                    item.source,
                    item.page,
                )
                for item
                in result.citations
            ]

            self.assert_equal(
                citation_pairs,
                source_pairs,
                (
                    f"Citation/source mismatch "
                    f"for {course_id}"
                ),
            )

            source_dicts = [
                {
                    "source": item.source,
                    "page": item.page,
                }
                for item
                in result.sources
            ]

            self.assert_sources_belong_to_course(
                course_id,
                source_dicts,
            )

        self.pass_test(
            name,
            (
                "citations match retrieved "
                "document/page metadata"
            ),
        )

    def test_l7_telemetry_schema(
        self,
    ) -> None:

        name = (
            "L7 Repair and validation telemetry"
        )

        debug = self.snapshots.get(
            "electronics_grounded"
        )

        self.assert_true(
            debug is not None,
            (
                "Electronics grounded snapshot "
                "is unavailable"
            ),
        )

        required_keys = (
            "grounding_claim_precheck_status",
            "validation_status",
            "pedagogical_status",
            "escalation_action",
            "repair_mode",
            "repair_attempts",
            "max_repair_attempts",
            "repair_limit_reached",
            "repair_failed",
            "repair_failure_type",
            "repair_failure_reason",
            "ai_usage",
        )

        missing = [
            key
            for key in required_keys
            if key not in debug
        ]

        self.assert_equal(
            missing,
            [],
            "Missing telemetry keys",
        )

        self.assert_equal(
            debug.get(
                "max_repair_attempts"
            ),
            1,
            "Repair-attempt configuration",
        )

        self.assert_true(
            debug.get(
                "repair_attempts",
                0,
            )
            <=
            debug.get(
                "max_repair_attempts",
                1,
            ),
            (
                "Repair attempt count "
                "exceeded configured limit"
            ),
        )

        self.pass_test(
            name,
            (
                "validation/repair telemetry "
                "schema is complete"
            ),
        )

    def test_l8_ai_usage_budget(
        self,
    ) -> None:

        name = (
            "L8 Live AI usage budget"
        )

        required = (
            "electronics_grounded",
            "electronics_out_of_course",
            "mathematics_grounded",
            "mathematics_out_of_course",
        )

        missing = [
            key
            for key in required
            if key not in self.snapshots
        ]

        self.assert_equal(
            missing,
            [],
            "Missing usage snapshots",
        )

        total_calls = 0
        details = []

        for key in required:

            debug = self.snapshots[key]

            usage = debug.get(
                "ai_usage",
                {}
            )

            calls = usage.get(
                "calls",
                0,
            )

            records = usage.get(
                "records",
                [],
            )

            self.assert_equal(
                len(records),
                calls,
                (
                    f"AI usage record count "
                    f"for {key}"
                ),
            )

            self.assert_true(
                usage.get(
                    "total_tokens",
                    0,
                )
                >= 0,
                "Negative token usage detected",
            )

            self.assert_true(
                usage.get(
                    "estimated_cost_usd",
                    0.0,
                )
                >= 0.0,
                "Negative estimated cost detected",
            )

            if (
                "out_of_course"
                in key
            ):

                self.assert_less_equal(
                    calls,
                    self.MAX_OUT_OF_COURSE_CALLS,
                    (
                        f"Out-of-course budget "
                        f"for {key}"
                    ),
                )

            else:

                self.assert_less_equal(
                    calls,
                    self.MAX_GROUNDED_CALLS,
                    (
                        f"Grounded-response budget "
                        f"for {key}"
                    ),
                )

            total_calls += calls

            details.append(
                f"{key}={calls}"
            )

        self.pass_test(
            name,
            (
                ", ".join(details)
                +
                f", total={total_calls}"
            ),
        )

    # =========================================================
    # Runner
    # =========================================================

    def run(self) -> int:

        print(
            "\nStep 16.16 Live Integration Regression\n"
        )

        tests = [
            self.test_l1_electronics_grounded,
            self.test_l2_electronics_rejects_math,
            self.test_l3_mathematics_grounded,
            self.test_l4_mathematics_rejects_electronics,
            self.test_l5_chroma_course_isolation,
            self.test_l6_citation_integrity,
            self.test_l7_telemetry_schema,
            self.test_l8_ai_usage_budget,
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
                result.passed
                for result in self.results
            )
            else 1
        )

    def print_summary(self) -> None:

        print()

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

        print(
            "\nSUMMARY"
        )

        print(
            f"Passed: {passed}"
        )

        print(
            f"Failed: {failed}"
        )


def main():

    suite = (
        LiveRegressionSuite16_16()
    )

    exit_code = suite.run()

    raise SystemExit(
        exit_code
    )


if __name__ == "__main__":

    main()