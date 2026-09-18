from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path
from app.domain.scaffolding.strategies import (
    get_strategy,
)

from app.domain.scaffolding.interventions import (
    get_intervention,
)

from app.courses.loader import CourseProfileLoader

from app.courses.models import (
    CourseProfile,
    ResponseStyleProfile,
)

from app.services.language_consistency_guard import (
    LanguageConsistencyGuard,
)

from app.services.response_style_resolver import (
    ResponseStyleResolver,
    ResolvedResponseStyle,
)

from app.services.response_style_instruction_builder import (
    ResponseStyleInstructionBuilder,
)

from app.services.response_quality_guard import (
    ResponseQualityGuard,
)
from app.services.response_quality_models import (
    ResponseQualityResult,
)

from app.services.response_quality_policy import (
    ResponseQualityPolicy,
)
from app.services.response_quality_history_models import (
    ResponseQualityTurnRecord,
)

from app.services.response_quality_analytics import (
    ResponseQualityAnalytics,
)

from app.services.response_quality_history import (
    ResponseQualityHistory,
)

from app.services.response_quality_history_models import (
    ResponseQualityTurnRecord,
)

from app.tutor import AITutor


@dataclass
class TestResult:
    name: str
    passed: bool
    details: str = ""


class RegressionSuite16_17:

    def __init__(self):

        self.results: list[TestResult] = []

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
    # Guard helpers
    # =========================================================

    def build_guard(
        self,
        course_id: str,
    ) -> LanguageConsistencyGuard:

        profile = (
            CourseProfileLoader()
            .load(
                course_id
            )
        )

        return LanguageConsistencyGuard(
            course_profile=profile
        )

    def build_style_resolver(
        self,
    ) -> ResponseStyleResolver:

        return ResponseStyleResolver()

    def build_style_instruction_builder(
        self,
    ) -> ResponseStyleInstructionBuilder:

        return ResponseStyleInstructionBuilder()
    
    # =========================================================
    # Controlled AITutor helper
    # =========================================================

    def build_controlled_tutor(
        self,
    ) -> AITutor:

        tutor = AITutor()

        knowledge_result = SimpleNamespace(
            query="controlled query",
            context=(
                "Controlled course knowledge "
                "for NPN transistor structure."
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
                    "controlled relevant result"
                ),
            )
        )

        tutor.grounding_guard.evaluate = (
            lambda result, relevance:
            SimpleNamespace(
                has_knowledge=True,
                reason=(
                    "controlled knowledge available"
                ),
            )
        )

        from app.services.guiding_question_grounding_models import (
            GuidingQuestionGroundingResult,
        )

        # ---------------------------------------------------------
        # Step 16.17 compatibility isolation.
        #
        # Guiding-question grounding belongs to Step 16.20.
        # These Step 16.17 tests are verifying deterministic
        # language/style/quality observation, so the later
        # semantic guiding-question validator must not introduce
        # unrelated provider calls or rewrite the controlled
        # final response.
        # ---------------------------------------------------------

        tutor.guiding_question_grounding_validator.validate = (
            lambda response, knowledge_context:
            GuidingQuestionGroundingResult(
                status="supported",
                reason=(
                    "Controlled Step 16.17 test harness "
                    "treats guiding-question grounding as "
                    "already validated by the later "
                    "Step 16.20 layer."
                ),
                issues=(),
            )
        )


        return tutor


    def build_prompt_integrated_tutor(
        self,
    ) -> AITutor:

        tutor = AITutor()

        knowledge_result = SimpleNamespace(
            query="controlled query",
            context=(
                "Controlled course knowledge "
                "for NPN transistor structure."
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
                reason="controlled relevant result",
            )
        )

        tutor.grounding_guard.evaluate = (
            lambda result, relevance:
            SimpleNamespace(
                has_knowledge=True,
                reason="controlled knowledge available",
            )
        )

        from app.services.guiding_question_grounding_models import (
            GuidingQuestionGroundingResult,
        )

        tutor.guiding_question_grounding_validator.validate = (
            lambda response, knowledge_context:
            GuidingQuestionGroundingResult(
                status="supported",
                reason=(
                    "Controlled Step 16.17 test harness "
                    "treats guiding-question grounding as "
                    "already validated by the later "
                    "Step 16.20 layer."
                ),
                issues=(),
            )
        )


        return tutor

    # =========================================================
    # Tests
    # =========================================================

    def test_lng_t1_thai_consistent(
        self,
    ) -> None:

        name = (
            "LNG-T1 Thai response consistent"
        )

        guard = self.build_guard(
            "electronics"
        )

        result = guard.evaluate(
            "คุณคิดว่าในทรานซิสเตอร์ NPN "
            "จะมีส่วนประกอบหลักอะไรบ้าง?"
        )

        self.assert_equal(
            result.status,
            "consistent",
            "Language status",
        )

        self.assert_equal(
            result.expected_language,
            "th",
            "Expected language",
        )

        self.assert_equal(
            result.detected_language,
            "th",
            "Detected language",
        )

        self.pass_test(
            name,
            "Thai response correctly detected.",
        )

    def test_lng_t2_english_mismatch(
        self,
    ) -> None:

        name = (
            "LNG-T2 English response mismatch"
        )

        guard = self.build_guard(
            "electronics"
        )

        result = guard.evaluate(
            "What part of the NPN transistor "
            "is the emitter?"
        )

        self.assert_equal(
            result.status,
            "mismatch",
            "Language status",
        )

        self.assert_equal(
            result.expected_language,
            "th",
            "Expected language",
        )

        self.assert_equal(
            result.detected_language,
            "en",
            "Detected language",
        )

        self.pass_test(
            name,
            "English response rejected for Thai course.",
        )

    def test_lng_t3_thai_with_technical_english(
        self,
    ) -> None:

        name = (
            "LNG-T3 Thai with technical English"
        )

        guard = self.build_guard(
            "electronics"
        )

        result = guard.evaluate(
            "คุณคิดว่าอิมิตเตอร์ "
            "(Emitter) ทำหน้าที่อะไร?"
        )

        self.assert_equal(
            result.status,
            "consistent",
            "Language status",
        )

        self.assert_equal(
            result.expected_language,
            "th",
            "Expected language",
        )

        self.pass_test(
            name,
            (
                "Technical English terms do not "
                "cause false mismatch."
            ),
        )

    def test_lng_t4_mixed_uncertain(
        self,
    ) -> None:

        name = (
            "LNG-T4 Mixed-heavy response uncertain"
        )

        guard = self.build_guard(
            "electronics"
        )

        result = guard.evaluate(
            "ลองพิจารณา Base-Emitter "
            "junction ก่อน"
        )

        self.assert_equal(
            result.status,
            "uncertain",
            "Language status",
        )

        self.assert_equal(
            result.detected_language,
            "mixed",
            "Detected language",
        )

        self.pass_test(
            name,
            "Mixed-heavy response remains conservative.",
        )

    def test_lng_t5_short_uncertain(
        self,
    ) -> None:

        name = (
            "LNG-T5 Short response uncertain"
        )

        guard = self.build_guard(
            "electronics"
        )

        result = guard.evaluate(
            "NPN?"
        )

        self.assert_equal(
            result.status,
            "uncertain",
            "Language status",
        )

        self.assert_equal(
            result.detected_language,
            "unknown",
            "Detected language",
        )

        self.pass_test(
            name,
            "Low-evidence response remains uncertain.",
        )

    def test_lng_t6_empty_uncertain(
        self,
    ) -> None:

        name = (
            "LNG-T6 Empty response uncertain"
        )

        guard = self.build_guard(
            "electronics"
        )

        result = guard.evaluate(
            ""
        )

        self.assert_equal(
            result.status,
            "uncertain",
            "Language status",
        )

        self.assert_equal(
            result.detected_language,
            "unknown",
            "Detected language",
        )

        self.pass_test(
            name,
            "Empty response remains uncertain.",
        )

    def test_lng_t7_mathematics_thai(
        self,
    ) -> None:

        name = (
            "LNG-T7 Mathematics Thai consistent"
        )

        guard = self.build_guard(
            "mathematics"
        )

        result = guard.evaluate(
            "คุณคิดว่ากฎของเลขยกกำลัง"
            "ช่วยในการจัดรูปนิพจน์อย่างไร?"
        )

        self.assert_equal(
            result.status,
            "consistent",
            "Language status",
        )

        self.assert_equal(
            result.expected_language,
            "th",
            "Expected language",
        )

        self.assert_equal(
            result.detected_language,
            "th",
            "Detected language",
        )

        self.pass_test(
            name,
            (
                "Language guard remains "
                "course-agnostic."
            ),
        )

    def test_lng_t8_tutor_telemetry(
        self,
    ) -> None:

        name = (
            "LNG-T8 AITutor final-answer telemetry"
        )

        tutor = (
            self.build_controlled_tutor()
        )

        response = (
            "คุณคิดว่าในทรานซิสเตอร์ NPN "
            "จะมีส่วนประกอบหลักอะไรบ้าง?"
        )

        with patch(
            "app.tutor.chat_with_ai",
            return_value=response,
        ):

            answer = tutor.respond(
                "ทรานซิสเตอร์ NPN "
                "มีโครงสร้างอย่างไร"
            )

        debug = tutor.get_debug_info()

        self.assert_equal(
            answer,
            response,
            "Final answer",
        )

        self.assert_equal(
            debug.get(
                "language_consistency_status"
            ),
            "consistent",
            "Language telemetry status",
        )

        self.assert_equal(
            debug.get(
                "language_expected"
            ),
            "th",
            "Language telemetry expected",
        )

        self.assert_equal(
            debug.get(
                "language_detected"
            ),
            "th",
            "Language telemetry detected",
        )

        self.assert_true(
            debug.get(
                "language_thai_chars",
                0,
            )
            > 0,
            "Thai character telemetry missing",
        )

        self.pass_test(
            name,
            (
                "AITutor stores final-answer "
                "language telemetry."
            ),
        )

    def test_lng_t9_no_extra_llm_calls(
        self,
    ) -> None:

        name = (
            "LNG-T9 Detection adds no LLM calls"
        )

        tutor = (
            self.build_controlled_tutor()
        )

        response = (
            "คุณคิดว่าในทรานซิสเตอร์ NPN "
            "จะมีส่วนประกอบหลักอะไรบ้าง?"
        )

        with patch(
            "app.tutor.chat_with_ai",
            return_value=response,
        ):

            tutor.respond(
                "ทรานซิสเตอร์ NPN "
                "มีโครงสร้างอย่างไร"
            )

        debug = tutor.get_debug_info()

        self.assert_equal(
            debug.get(
                "language_consistency_status"
            ),
            "consistent",
            "Language status",
        )

        # In the controlled environment, the only
        # patched Tutor generation call is not recorded
        # by the real provider usage collector.
        #
        # Therefore language detection must not create
        # any additional AI usage records.
        self.assert_equal(
            debug[
                "ai_usage"
            ][
                "calls"
            ],
            0,
            (
                "Language consistency detection "
                "must not create LLM calls"
            ),
        )

        self.pass_test(
            name,
            (
                "Language detection is fully "
                "deterministic."
            ),
        )

    def test_lng_t10_mismatch_telemetry_only(
        self,
    ) -> None:

        name = (
            "LNG-T10 Mismatch is telemetry-only"
        )

        tutor = (
            self.build_controlled_tutor()
        )

        english_response = (
            "What part of the NPN transistor "
            "do you think is the emitter?"
        )

        with patch(
            "app.tutor.chat_with_ai",
            return_value=english_response,
        ):

            answer = tutor.respond(
                "ทรานซิสเตอร์ NPN "
                "มีโครงสร้างอย่างไร"
            )

        debug = tutor.get_debug_info()

        self.assert_equal(
            debug.get(
                "language_consistency_status"
            ),
            "mismatch",
            "Language mismatch status",
        )

        self.assert_equal(
            debug.get(
                "language_expected"
            ),
            "th",
            "Expected language",
        )

        self.assert_equal(
            debug.get(
                "language_detected"
            ),
            "en",
            "Detected language",
        )

        self.assert_equal(
            answer,
            english_response,
            (
                "Step 16.17.1 must not rewrite "
                "a language mismatch"
            ),
        )

        self.assert_equal(
            debug.get(
                "repair_attempts"
            ),
            0,
            (
                "Language mismatch must not enter "
                "Step 16.16 repair pipeline"
            ),
        )

        self.pass_test(
            name,
            (
                "Mismatch detected without "
                "changing response behavior."
            ),
        )


    def test_lng_t11_resolved_style_language_precedence(
        self,
    ) -> None:

        name = (
            "LNG-T11 Resolved style language precedence"
        )

        profile = CourseProfile(
            course_id="language_precedence",
            course_name="Language Precedence",
            language="th",
            document_path="./docs/test",
            chroma_path="./data/chroma/test",
            response_style=ResponseStyleProfile(
                primary_language="en"
            ),
        )

        guard = LanguageConsistencyGuard(
            profile
        )

        result = guard.evaluate(
            "What part of the transistor "
            "is the emitter?"
        )

        self.assert_equal(
            guard.expected_language,
            "en",
            (
                "Language guard did not use "
                "resolved response language"
            ),
        )

        self.assert_equal(
            result.status,
            "consistent",
            "English response status",
        )

        self.assert_equal(
            result.expected_language,
            "en",
            "Expected language",
        )

        self.assert_equal(
            result.detected_language,
            "en",
            "Detected language",
        )

        self.pass_test(
            name,
            (
                "Language guard uses resolved "
                "response_style language."
            ),
        )


    def test_lng_t12_course_language_does_not_override_style(
        self,
    ) -> None:

        name = (
            "LNG-T12 Course language does not override style"
        )

        profile = CourseProfile(
            course_id="language_mismatch",
            course_name="Language Mismatch",
            language="th",
            document_path="./docs/test",
            chroma_path="./data/chroma/test",
            response_style=ResponseStyleProfile(
                primary_language="en"
            ),
        )

        guard = LanguageConsistencyGuard(
            profile
        )

        result = guard.evaluate(
            "คุณคิดว่าอิมิตเตอร์ทำหน้าที่อะไร?"
        )

        self.assert_equal(
            guard.expected_language,
            "en",
            "Resolved expected language",
        )

        self.assert_equal(
            result.status,
            "mismatch",
            (
                "Thai response should be mismatch "
                "when resolved language is English"
            ),
        )

        self.assert_equal(
            result.expected_language,
            "en",
            "Expected language telemetry",
        )

        self.assert_equal(
            result.detected_language,
            "th",
            "Detected language",
        )

        self.pass_test(
            name,
            (
                "Course language does not override "
                "explicit response style."
            ),
        )

    def test_style_t1_electronics_typed_resolution(
        self,
    ) -> None:

        name = (
            "STYLE-T1 Electronics typed resolution"
        )

        profile = (
            CourseProfileLoader()
            .load(
                "electronics"
            )
        )

        result = (
            self.build_style_resolver()
            .resolve(
                profile
            )
        )

        self.assert_true(
            isinstance(
                result,
                ResolvedResponseStyle,
            ),
            (
                "Resolver did not return "
                "ResolvedResponseStyle"
            ),
        )

        self.assert_equal(
            result.primary_language,
            "th",
            "Resolved primary language",
        )

        self.assert_equal(
            result.allow_technical_english,
            True,
            "Technical English policy",
        )

        self.assert_equal(
            result.technical_term_format,
            "thai_with_english_parentheses",
            "Technical term format",
        )

        self.assert_equal(
            result.tone,
            "supportive_academic",
            "Tone",
        )

        self.assert_equal(
            result.explanation_depth,
            "adaptive",
            "Explanation depth",
        )

        self.assert_equal(
            result.question_style,
            "socratic",
            "Question style",
        )

        self.assert_equal(
            result.max_guiding_questions,
            1,
            "Maximum guiding questions",
        )

        self.pass_test(
            name,
            (
                "Electronics response style "
                "resolved to typed configuration."
            ),
        )

    def test_style_t2_mathematics_typed_resolution(
        self,
    ) -> None:

        name = (
            "STYLE-T2 Mathematics typed resolution"
        )

        profile = (
            CourseProfileLoader()
            .load(
                "mathematics"
            )
        )

        result = (
            self.build_style_resolver()
            .resolve(
                profile
            )
        )

        self.assert_true(
            isinstance(
                result,
                ResolvedResponseStyle,
            ),
            (
                "Mathematics resolver result "
                "is not typed"
            ),
        )

        self.assert_equal(
            result.primary_language,
            "th",
            "Resolved language",
        )

        self.assert_equal(
            result.tone,
            "supportive_academic",
            "Resolved tone",
        )

        self.assert_equal(
            result.question_style,
            "socratic",
            "Resolved question style",
        )

        self.assert_equal(
            result.max_guiding_questions,
            1,
            "Resolved maximum questions",
        )

        self.pass_test(
            name,
            (
                "Mathematics response style "
                "resolved correctly."
            ),
        )


    def test_style_t3_style_language_precedence(
        self,
    ) -> None:

        name = (
            "STYLE-T3 Style language precedence"
        )

        profile = CourseProfile(
            course_id="style_precedence",
            course_name="Style Precedence",
            language="th",
            document_path="./docs/test",
            chroma_path="./data/chroma/test",
            response_style=ResponseStyleProfile(
                primary_language="en"
            ),
        )

        result = (
            self.build_style_resolver()
            .resolve(
                profile
            )
        )

        self.assert_equal(
            profile.language,
            "th",
            "Course language",
        )

        self.assert_equal(
            profile.response_style.primary_language,
            "en",
            "Configured response language",
        )

        self.assert_equal(
            result.primary_language,
            "en",
            (
                "Response style language must "
                "override course language"
            ),
        )

        self.pass_test(
            name,
            (
                "response_style.primary_language "
                "takes precedence."
            ),
        )


    def test_style_t4_course_language_fallback(
        self,
    ) -> None:

        name = (
            "STYLE-T4 Course language fallback"
        )

        profile = CourseProfile(
            course_id="style_fallback",
            course_name="Style Fallback",
            language="en",
            document_path="./docs/test",
            chroma_path="./data/chroma/test",
            response_style=ResponseStyleProfile(
                primary_language=""
            ),
        )

        result = (
            self.build_style_resolver()
            .resolve(
                profile
            )
        )

        self.assert_equal(
            result.primary_language,
            "en",
            (
                "Empty response language did "
                "not inherit course language"
            ),
        )

        self.pass_test(
            name,
            (
                "Empty style language inherits "
                "course language."
            ),
        )

    def test_style_t5_thai_alias_normalization(
        self,
    ) -> None:

        name = (
            "STYLE-T5 Thai alias normalization"
        )

        profile = CourseProfile(
            course_id="thai_alias",
            course_name="Thai Alias",
            language="en",
            document_path="./docs/test",
            chroma_path="./data/chroma/test",
            response_style=ResponseStyleProfile(
                primary_language="Thai"
            ),
        )

        result = (
            self.build_style_resolver()
            .resolve(
                profile
            )
        )

        self.assert_equal(
            result.primary_language,
            "th",
            "Thai alias normalization",
        )

        self.pass_test(
            name,
            "Thai alias normalized to th.",
        )

    def test_style_t6_english_alias_normalization(
        self,
    ) -> None:

        name = (
            "STYLE-T6 English alias normalization"
        )

        profile = CourseProfile(
            course_id="english_alias",
            course_name="English Alias",
            language="th",
            document_path="./docs/test",
            chroma_path="./data/chroma/test",
            response_style=ResponseStyleProfile(
                primary_language="en-US"
            ),
        )

        result = (
            self.build_style_resolver()
            .resolve(
                profile
            )
        )

        self.assert_equal(
            result.primary_language,
            "en",
            "English alias normalization",
        )

        self.pass_test(
            name,
            "en-US alias normalized to en.",
        )

    def test_style_t7_default_fields_preserved(
        self,
    ) -> None:

        name = (
            "STYLE-T7 Default fields preserved"
        )

        profile = CourseProfile(
            course_id="partial_style",
            course_name="Partial Style",
            language="th",
            document_path="./docs/test",
            chroma_path="./data/chroma/test",
            response_style=ResponseStyleProfile(
                primary_language="th",
                tone="formal",
            ),
        )

        result = (
            self.build_style_resolver()
            .resolve(
                profile
            )
        )

        self.assert_equal(
            result.tone,
            "formal",
            "Configured tone",
        )

        self.assert_equal(
            result.question_style,
            "socratic",
            "Default question style",
        )

        self.assert_equal(
            result.explanation_depth,
            "adaptive",
            "Default explanation depth",
        )

        self.assert_equal(
            result.allow_technical_english,
            True,
            "Default technical-English policy",
        )

        self.assert_equal(
            result.max_guiding_questions,
            1,
            "Default maximum guiding questions",
        )

        self.pass_test(
            name,
            (
                "Explicit style value overrides "
                "while defaults remain intact."
            ),
        )

    def test_style_t8_source_profile_immutable(
        self,
    ) -> None:

        name = (
            "STYLE-T8 Source profile remains unchanged"
        )

        profile = (
            CourseProfileLoader()
            .load(
                "electronics"
            )
        )

        before_profile = profile
        before_style = profile.response_style

        result = (
            self.build_style_resolver()
            .resolve(
                profile
            )
        )

        self.assert_equal(
            profile,
            before_profile,
            (
                "CourseProfile changed during "
                "style resolution"
            ),
        )

        self.assert_equal(
            profile.response_style,
            before_style,
            (
                "ResponseStyleProfile changed "
                "during resolution"
            ),
        )

        self.assert_true(
            result is not profile.response_style,
            (
                "Resolved style should be a "
                "separate value object"
            ),
        )

        self.pass_test(
            name,
            (
                "Resolver does not mutate "
                "source configuration."
            ),
        )

    def test_style_inst_t1_electronics_instructions(
        self,
    ) -> None:

        name = (
            "STYLE-INST-T1 Electronics instructions"
        )

        profile = (
            CourseProfileLoader()
            .load(
                "electronics"
            )
        )

        style = (
            self.build_style_resolver()
            .resolve(
                profile
            )
        )

        instruction = (
            self.build_style_instruction_builder()
            .build(
                style
            )
        )

        self.assert_true(
            "Respond primarily in Thai."
            in instruction,
            "Primary-language instruction missing",
        )

        self.assert_true(
            "English technical terms may be used"
            in instruction,
            "Technical-English instruction missing",
        )

        self.assert_true(
            "English term in parentheses"
            in instruction,
            "Technical-term format instruction missing",
        )

        self.assert_true(
            "supportive academic tone"
            in instruction,
            "Tone instruction missing",
        )

        self.assert_true(
            "Adapt explanation depth"
            in instruction,
            "Explanation-depth instruction missing",
        )

        self.assert_true(
            "Socratic questioning"
            in instruction,
            "Question-style instruction missing",
        )

        self.assert_true(
            "no more than one guiding question"
            in instruction,
            "Guiding-question limit missing",
        )

        self.pass_test(
            name,
            (
                "Resolved Electronics style converted "
                "to deterministic prompt instructions."
            ),
        )


    def test_style_inst_t2_english_language(
        self,
    ) -> None:

        name = (
            "STYLE-INST-T2 English language"
        )

        profile = CourseProfile(
            course_id="english_style",
            course_name="English Style",
            language="th",
            document_path="./docs/test",
            chroma_path="./data/chroma/test",
            response_style=ResponseStyleProfile(
                primary_language="en"
            ),
        )

        style = (
            self.build_style_resolver()
            .resolve(
                profile
            )
        )

        instruction = (
            self.build_style_instruction_builder()
            .build(
                style
            )
        )

        self.assert_true(
            "Respond primarily in English."
            in instruction,
            "English instruction missing",
        )

        self.pass_test(
            name,
            (
                "Resolved English response language "
                "converted correctly."
            ),
        )


    def test_style_inst_t3_technical_english_disabled(
        self,
    ) -> None:

        name = (
            "STYLE-INST-T3 Technical English disabled"
        )

        profile = CourseProfile(
            course_id="no_technical_english",
            course_name="No Technical English",
            language="th",
            document_path="./docs/test",
            chroma_path="./data/chroma/test",
            response_style=ResponseStyleProfile(
                allow_technical_english=False
            ),
        )

        style = (
            self.build_style_resolver()
            .resolve(
                profile
            )
        )

        instruction = (
            self.build_style_instruction_builder()
            .build(
                style
            )
        )

        self.assert_true(
            (
                "Avoid English technical terms "
                "unless they are strictly necessary."
            )
            in instruction,
            (
                "Technical-English disabled "
                "instruction missing"
            ),
        )

        self.pass_test(
            name,
            (
                "Technical-English restriction "
                "converted correctly."
            ),
        )

    def test_style_inst_t4_zero_guiding_questions(
        self,
    ) -> None:

        name = (
            "STYLE-INST-T4 Zero guiding questions"
        )

        profile = CourseProfile(
            course_id="zero_questions",
            course_name="Zero Questions",
            language="th",
            document_path="./docs/test",
            chroma_path="./data/chroma/test",
            response_style=ResponseStyleProfile(
                max_guiding_questions=0
            ),
        )

        style = (
            self.build_style_resolver()
            .resolve(
                profile
            )
        )

        instruction = (
            self.build_style_instruction_builder()
            .build(
                style
            )
        )

        self.assert_true(
            "Do not include a guiding question"
            in instruction,
            (
                "Zero-guiding-question "
                "instruction missing"
            ),
        )

        self.pass_test(
            name,
            (
                "Zero guiding-question policy "
                "converted correctly."
            ),
        )

    def test_style_inst_t5_multiple_guiding_questions(
        self,
    ) -> None:

        name = (
            "STYLE-INST-T5 Multiple guiding questions"
        )

        profile = CourseProfile(
            course_id="multiple_questions",
            course_name="Multiple Questions",
            language="th",
            document_path="./docs/test",
            chroma_path="./data/chroma/test",
            response_style=ResponseStyleProfile(
                max_guiding_questions=3
            ),
        )

        style = (
            self.build_style_resolver()
            .resolve(
                profile
            )
        )

        instruction = (
            self.build_style_instruction_builder()
            .build(
                style
            )
        )

        self.assert_true(
            (
                "Use no more than 3 "
                "guiding questions in a response."
            )
            in instruction,
            (
                "Multiple guiding-question "
                "instruction missing"
            ),
        )

        self.pass_test(
            name,
            (
                "Configured guiding-question limit "
                "converted correctly."
            ),
        )

    def test_style_inst_t6_deterministic_output(
        self,
    ) -> None:

        name = (
            "STYLE-INST-T6 Deterministic output"
        )

        profile = (
            CourseProfileLoader()
            .load(
                "electronics"
            )
        )

        style = (
            self.build_style_resolver()
            .resolve(
                profile
            )
        )

        builder = (
            self.build_style_instruction_builder()
        )

        first = builder.build(
            style
        )

        second = builder.build(
            style
        )

        self.assert_equal(
            first,
            second,
            (
                "Instruction builder produced "
                "non-deterministic output"
            ),
        )

        self.pass_test(
            name,
            (
                "Response-style instructions "
                "are deterministic."
            ),
        )

    def test_style_prompt_t1_instruction_in_system_prompt(
        self,
    ) -> None:

        name = (
            "STYLE-PROMPT-T1 Style instruction in system prompt"
        )

        tutor = (
            self.build_prompt_integrated_tutor()
        )

        response = (
            "คุณคิดว่าโครงสร้างของทรานซิสเตอร์ "
            "NPN ประกอบด้วยส่วนใดบ้าง?"
        )

        with patch(
            "app.tutor.chat_with_ai",
            return_value=response,
        ):

            tutor.respond(
                "ทรานซิสเตอร์ NPN "
                "มีโครงสร้างอย่างไร"
            )

        prompt = (
            tutor.get_last_system_prompt()
        )

        self.assert_true(
            "Respond primarily in Thai."
            in prompt,
            (
                "Response-style language instruction "
                "missing from final system prompt"
            ),
        )

        self.pass_test(
            name,
            (
                "Response-style instruction appears "
                "in final system prompt."
            ),
        )


    def test_style_prompt_t2_technical_english_policy(
        self,
    ) -> None:

        name = (
            "STYLE-PROMPT-T2 Technical English policy"
        )

        tutor = (
            self.build_prompt_integrated_tutor()
        )

        with patch(
            "app.tutor.chat_with_ai",
            return_value=(
                "คุณคิดว่าอิมิตเตอร์ "
                "(Emitter) ทำหน้าที่อะไร?"
            ),
        ):

            tutor.respond(
                "ทรานซิสเตอร์ NPN "
                "มีโครงสร้างอย่างไร"
            )

        prompt = (
            tutor.get_last_system_prompt()
        )

        self.assert_true(
            "English technical terms may be used"
            in prompt,
            (
                "Technical-English policy missing "
                "from system prompt"
            ),
        )

        self.assert_true(
            "English term in parentheses"
            in prompt,
            (
                "Technical-term format instruction "
                "missing from system prompt"
            ),
        )

        self.pass_test(
            name,
            (
                "Technical-English response-style "
                "policy is present."
            ),
        )

    def test_style_prompt_t3_tone_and_depth(
        self,
    ) -> None:

        name = (
            "STYLE-PROMPT-T3 Tone and explanation depth"
        )

        tutor = (
            self.build_prompt_integrated_tutor()
        )

        with patch(
            "app.tutor.chat_with_ai",
            return_value=(
                "คุณคิดว่าโครงสร้างของ NPN "
                "ประกอบด้วยส่วนใดบ้าง?"
            ),
        ):

            tutor.respond(
                "ทรานซิสเตอร์ NPN "
                "มีโครงสร้างอย่างไร"
            )

        prompt = (
            tutor.get_last_system_prompt()
        )

        self.assert_true(
            "supportive academic tone"
            in prompt,
            "Tone instruction missing",
        )

        self.assert_true(
            "Adapt explanation depth"
            in prompt,
            (
                "Explanation-depth instruction missing"
            ),
        )

        self.pass_test(
            name,
            (
                "Tone and explanation-depth "
                "instructions are integrated."
            ),
        )

    def test_style_prompt_t4_question_policy(
        self,
    ) -> None:

        name = (
            "STYLE-PROMPT-T4 Question-style policy"
        )

        tutor = (
            self.build_prompt_integrated_tutor()
        )

        with patch(
            "app.tutor.chat_with_ai",
            return_value=(
                "คุณคิดว่าโครงสร้างของ NPN "
                "ประกอบด้วยส่วนใดบ้าง?"
            ),
        ):

            tutor.respond(
                "ทรานซิสเตอร์ NPN "
                "มีโครงสร้างอย่างไร"
            )

        prompt = (
            tutor.get_last_system_prompt()
        )

        self.assert_true(
            "Socratic questioning"
            in prompt,
            (
                "Socratic question-style "
                "instruction missing"
            ),
        )

        self.assert_true(
            "no more than one guiding question"
            in prompt,
            (
                "Guiding-question limit missing "
                "from final prompt"
            ),
        )

        self.pass_test(
            name,
            (
                "Question style and guiding-question "
                "limit are integrated."
            ),
        )      

    def test_style_prompt_t5_legacy_language_rules_absent(
        self,
    ) -> None:

        name = (
            "STYLE-PROMPT-T5 Legacy language rules absent"
        )

        tutor = (
            self.build_prompt_integrated_tutor()
        )

        with patch(
            "app.tutor.chat_with_ai",
            return_value=(
                "คุณคิดว่าโครงสร้างของ NPN "
                "ประกอบด้วยส่วนใดบ้าง?"
            ),
        ):

            tutor.respond(
                "ทรานซิสเตอร์ NPN "
                "มีโครงสร้างอย่างไร"
            )

        prompt = (
            tutor.get_last_system_prompt()
        )

        self.assert_true(
            "LANGUAGE RULES:"
            not in prompt,
            (
                "Legacy knowledge-context LANGUAGE RULES "
                "must not return"
            ),
        )

        self.assert_true(
            (
                "The language of the course knowledge "
                "does NOT determine the response language."
            )
            not in prompt,
            (
                "Legacy knowledge-language ownership "
                "still exists"
            ),
        )

        self.pass_test(
            name,
            (
                "Legacy knowledge-context language "
                "rules remain removed."
            ),
        )

    def test_style_prompt_t6_deterministic_prompt(
        self,
    ) -> None:

        name = (
            "STYLE-PROMPT-T6 Deterministic prompt integration"
        )

        tutor = (
            self.build_prompt_integrated_tutor()
        )

        strategy = get_strategy(
            tutor.state.scaffolding_level
        )

        intervention = get_intervention(
            tutor.state.last_evaluation
        )

        first = (
            tutor.prompt_builder
            .build_system_prompt(
                strategy=strategy,
                intervention=intervention,
                misconception=None,
                knowledge_context=(
                    "Controlled knowledge."
                ),
                has_grounded_knowledge=True,
                response_style_instruction=(
                    tutor.response_style_instruction
                ),
            )
        )

        second = (
            tutor.prompt_builder
            .build_system_prompt(
                strategy=strategy,
                intervention=intervention,
                misconception=None,
                knowledge_context=(
                    "Controlled knowledge."
                ),
                has_grounded_knowledge=True,
                response_style_instruction=(
                    tutor.response_style_instruction
                ),
            )
        )

        self.assert_equal(
            first,
            second,
            (
                "Prompt integration is "
                "not deterministic"
            ),
        )

        self.pass_test(
            name,
            (
                "Same style and context produce "
                "the same system prompt."
            ),
        )    

    def test_style_prompt_t7_knowledge_context_preserved(
        self,
    ) -> None:

        name = (
            "STYLE-PROMPT-T7 Knowledge context preserved"
        )

        tutor = (
            self.build_prompt_integrated_tutor()
        )

        controlled_context = (
            "CONTROLLED_UNIQUE_KNOWLEDGE_TOKEN_16172D3"
        )

        knowledge_result = SimpleNamespace(
            query="controlled query",
            context=controlled_context,
            sources=[],
            citations=[],
        )

        tutor.knowledge_service.retrieve = (
            lambda query, n_results=5:
            knowledge_result
        )

        with patch(
            "app.tutor.chat_with_ai",
            return_value=(
                "คุณคิดว่าโครงสร้างของ NPN "
                "ประกอบด้วยส่วนใดบ้าง?"
            ),
        ):

            tutor.respond(
                "ทรานซิสเตอร์ NPN "
                "มีโครงสร้างอย่างไร"
            )

        prompt = (
            tutor.get_last_system_prompt()
        )

        self.assert_true(
            controlled_context
            in prompt,
            (
                "Response-style integration removed "
                "or modified knowledge context"
            ),
        )

        self.assert_true(
            "GROUNDING RULES:"
            in prompt,
            (
                "Grounding rules disappeared after "
                "style integration"
            ),
        )

        self.pass_test(
            name,
            (
                "Style integration preserves RAG "
                "knowledge and grounding sections."
            ),
        )


    def test_style_prompt_t8_no_extra_llm_calls(
        self,
    ) -> None:

        name = (
            "STYLE-PROMPT-T8 Prompt style adds no LLM calls"
        )

        tutor = (
            self.build_prompt_integrated_tutor()
        )

        with patch(
            "app.tutor.chat_with_ai",
            return_value=(
                "คุณคิดว่าโครงสร้างของ NPN "
                "ประกอบด้วยส่วนใดบ้าง?"
            ),
        ):

            tutor.respond(
                "ทรานซิสเตอร์ NPN "
                "มีโครงสร้างอย่างไร"
            )

        debug = (
            tutor.get_debug_info()
        )

        self.assert_equal(
            debug[
                "ai_usage"
            ][
                "calls"
            ],
            0,
            (
                "Deterministic response-style "
                "prompt integration must not "
                "create additional AI calls"
            ),
        )

        self.pass_test(
            name,
            (
                "Response-style prompt composition "
                "is fully deterministic."
            ),
        )

    def test_style_prompt_t9_base_language_rule_absent(
        self,
    ) -> None:

        name = (
            "STYLE-PROMPT-T9 Base language rule absent"
        )

        tutor = (
            self.build_prompt_integrated_tutor()
        )

        with patch(
            "app.tutor.chat_with_ai",
            return_value=(
                "คุณคิดว่าโครงสร้างของ NPN "
                "ประกอบด้วยส่วนใดบ้าง?"
            ),
        ):

            tutor.respond(
                "ทรานซิสเตอร์ NPN "
                "มีโครงสร้างอย่างไร"
            )

        prompt = (
            tutor.get_last_system_prompt()
        )

        self.assert_true(
            (
                "Respond in the same language "
                "as the learner."
            )
            not in prompt,
            (
                "Base SYSTEM_PROMPT still owns "
                "response language"
            ),
        )

        self.pass_test(
            name,
            (
                "Base system prompt no longer "
                "owns response language."
            ),
        )


    def test_style_prompt_t10_strategy_language_rule_absent(
        self,
    ) -> None:

        name = (
            "STYLE-PROMPT-T10 Strategy language rule absent"
        )

        tutor = (
            self.build_prompt_integrated_tutor()
        )

        with patch(
            "app.tutor.chat_with_ai",
            return_value=(
                "คุณคิดว่าโครงสร้างของ NPN "
                "ประกอบด้วยส่วนใดบ้าง?"
            ),
        ):

            tutor.respond(
                "ทรานซิสเตอร์ NPN "
                "มีโครงสร้างอย่างไร"
            )

        prompt = (
            tutor.get_last_system_prompt()
        )

        self.assert_true(
            (
                "Respond entirely in the same "
                "language as the learner."
            )
            not in prompt,
            (
                "Scaffolding strategy still owns "
                "response language"
            ),
        )

        self.assert_true(
            "Respond primarily in Thai."
            in prompt,
            (
                "Resolved response-style language "
                "instruction is missing"
            ),
        )

        self.pass_test(
            name,
            (
                "ResponseStyle remains the single "
                "prompt language owner."
            ),
        )

    def test_style_runtime_t1_debug_schema(
        self,
    ) -> None:

        name = (
            "STYLE-RUNTIME-T1 Debug telemetry schema"
        )

        tutor = AITutor()

        debug = tutor.get_debug_info()

        self.assert_equal(
            debug.get("response_style_language"),
            "th",
            "Resolved runtime language",
        )

        self.assert_equal(
            debug.get("response_style_tone"),
            "supportive_academic",
            "Resolved runtime tone",
        )

        self.assert_equal(
            debug.get("response_style_depth"),
            "adaptive",
            "Resolved runtime explanation depth",
        )

        self.assert_equal(
            debug.get("response_style_question_style"),
            "socratic",
            "Resolved runtime question style",
        )

        self.assert_equal(
            debug.get(
                "response_style_max_guiding_questions"
            ),
            1,
            "Resolved runtime guiding-question limit",
        )

        self.pass_test(
            name,
            (
                "Resolved response-style telemetry "
                "is exposed through debug info."
            ),
        )


    def test_style_runtime_t2_term_policy_telemetry(
        self,
    ) -> None:

        name = (
            "STYLE-RUNTIME-T2 Technical term telemetry"
        )

        tutor = AITutor()

        debug = tutor.get_debug_info()

        self.assert_equal(
            debug.get(
                "response_style_technical_english"
            ),
            True,
            "Technical-English telemetry",
        )

        self.assert_equal(
            debug.get(
                "response_style_term_format"
            ),
            "thai_with_english_parentheses",
            "Technical-term format telemetry",
        )

        self.pass_test(
            name,
            (
                "Technical terminology style "
                "is visible at runtime."
            ),
        )


    def test_style_runtime_t3_no_extra_llm_calls(
        self,
    ) -> None:

        name = (
            "STYLE-RUNTIME-T3 Telemetry adds no LLM calls"
        )

        tutor = (
            self.build_prompt_integrated_tutor()
        )

        with patch(
            "app.tutor.chat_with_ai",
            return_value=(
                "คุณคิดว่าโครงสร้างของ NPN "
                "ประกอบด้วยส่วนใดบ้าง?"
            ),
        ):

            tutor.respond(
                "ทรานซิสเตอร์ NPN "
                "มีโครงสร้างอย่างไร"
            )

        debug = tutor.get_debug_info()

        self.assert_equal(
            debug["ai_usage"]["calls"],
            0,
            (
                "Response-style runtime telemetry "
                "must not create AI calls"
            ),
        )

        self.assert_equal(
            debug.get(
                "response_style_language"
            ),
            "th",
            "Runtime response-style language missing",
        )

        self.assert_equal(
            debug.get(
                "response_style_tone"
            ),
            "supportive_academic",
            "Runtime response-style tone missing",
        )

        self.pass_test(
            name,
            (
                "Runtime response-style telemetry "
                "remains fully deterministic."
            ),
        )

     # =========================================================
    # RESPONSE QUALITY GUARD TESTS
    # Step 16.17.3B
    # =========================================================

    def test_quality_t1_normal_response_acceptable(
        self,
    ) -> None:

        guard = ResponseQualityGuard()

        result = guard.evaluate(
            "คุณคิดว่าโครงสร้างของทรานซิสเตอร์ "
            "NPN ประกอบด้วยส่วนใดบ้าง?"
        )

        assert isinstance(
            result,
            ResponseQualityResult,
        )

        assert result.status == "acceptable"
        assert result.issues == []
        assert result.too_long is False
        assert result.too_short is False
        assert result.too_many_questions is False
        assert result.repetitive is False

        self.pass_test(
            "QUALITY-T1 Normal response acceptable",
            (
                "Normal tutoring response passes "
                "deterministic quality checks."
            ),
        )        

    def test_quality_t2_multiple_questions_warning(
        self,
    ) -> None:

        guard = ResponseQualityGuard()

        result = guard.evaluate(
            "ส่วนแรกคืออะไร? "
            "ส่วนที่สองคืออะไร? "
            "ส่วนที่สามคืออะไร? "
            "คุณคิดว่าเพราะอะไร?"
        )

        assert result.status == "warning"
        assert result.too_many_questions is True
        assert result.question_count > 1
        assert any(
            "more questions"
            in issue.lower()
            for issue in result.issues
        )

        self.pass_test(
            "QUALITY-T2 Multiple questions warning",
            (
                "Multiple guiding questions are detected "
                "as a quality warning."
            ),
        )        

    def test_quality_t3_long_response_warning(
        self,
    ) -> None:

        guard = ResponseQualityGuard(
            max_characters=100,
        )

        result = guard.evaluate(
            "คำอธิบายเกี่ยวกับทรานซิสเตอร์ " * 20
        )

        assert result.status == "warning"
        assert result.too_long is True
        assert result.character_count > 100

        assert any(
            "maximum response length"
            in issue.lower()
            for issue in result.issues
        )
        self.pass_test(
            "QUALITY-T3 Long response warning",
            (
                "Excessively long response is detected "
                "deterministically."
            ),
        )

    def test_quality_t4_repetition_warning(
        self,
    ) -> None:

        guard = ResponseQualityGuard()

        result = guard.evaluate(
            "ลองพิจารณาโครงสร้างก่อน. "
            "ลองพิจารณาโครงสร้างก่อน."
        )

        assert result.status == "warning"
        assert result.repetitive is True

        assert any(
            "repetition"
            in issue.lower()
            for issue in result.issues
        )

        self.pass_test(
            "QUALITY-T4 Repetition warning",
            (
                "Repeated response content is detected "
                "as a quality warning."
            ),
        )

    def test_quality_t5_empty_response_uncertain(
        self,
    ) -> None:

        guard = ResponseQualityGuard()

        result = guard.evaluate("")

        assert result.status == "uncertain"
        assert result.character_count == 0
        assert result.token_like_count == 0
        assert result.question_count == 0
        assert result.sentence_count == 0

        assert any(
            "empty"
            in issue.lower()
            for issue in result.issues
        )

        self.pass_test(
            "QUALITY-T5 Empty response uncertain",
            (
                "Empty response remains conservative "
                "and is classified as uncertain."
            ),
        )

    def test_quality_t6_short_technical_response_safe(
        self,
    ) -> None:

        guard = ResponseQualityGuard()

        result = guard.evaluate("NPN?")

        assert result.status == "acceptable"
        assert result.too_short is False
        assert result.too_many_questions is False
        assert result.repetitive is False
        assert result.question_count == 1

        self.pass_test(
            "QUALITY-T6 Short technical response safe",
            (
                "Short technical response is not "
                "incorrectly flagged as low quality."
            ),
        )

    def test_quality_t7_result_metrics(
        self,
    ) -> None:

        guard = ResponseQualityGuard()

        result = guard.evaluate(
            "ลองพิจารณาโครงสร้างของ "
            "NPN ก่อน แล้วชั้นตรงกลางคืออะไร?"
        )

        assert isinstance(
            result.character_count,
            int,
        )
        assert isinstance(
            result.token_like_count,
            int,
        )
        assert isinstance(
            result.question_count,
            int,
        )
        assert isinstance(
            result.sentence_count,
            int,
        )

        assert result.character_count > 0
        assert result.token_like_count > 0
        assert result.question_count == 1
        assert result.sentence_count >= 1

        self.pass_test(
            "QUALITY-T7 Result metrics",
            (
                "Response quality result exposes "
                "deterministic quality metrics."
            ),
        )


    def test_quality_t8_deterministic_output(
        self,
    ) -> None:

        guard = ResponseQualityGuard()

        response = (
            "คุณคิดว่า Base "
            "มีบทบาทอย่างไรในทรานซิสเตอร์ NPN?"
        )

        first = guard.evaluate(response)
        second = guard.evaluate(response)

        assert first == second

        self.pass_test(
            "QUALITY-T8 Deterministic output",
            (
                "Same response produces the same "
                "quality evaluation."
            ),
        )


    def test_quality_t9_no_llm_dependency(
        self,
    ) -> None:

        source_path = (
            Path(__file__).parent
            / "services"
            / "response_quality_guard.py"
        )

        source = source_path.read_text(
            encoding="utf-8"
        )

        forbidden_dependencies = (
            "chat_with_ai",
            "openai",
            "groq",
            "litellm",
        )

        source_lower = source.lower()

        for dependency in forbidden_dependencies:
            assert dependency not in source_lower

        self.pass_test(
            "QUALITY-T9 No LLM dependency",
            (
                "ResponseQualityGuard has no "
                "LLM dependency."
            ),
        )

    def test_quality_t10_warning_is_detection_only(
        self,
    ) -> None:

        guard = ResponseQualityGuard()

        original_response = (
            "ส่วนแรกคืออะไร? "
            "ส่วนที่สองคืออะไร? "
            "ส่วนที่สามคืออะไร?"
        )

        result = guard.evaluate(
            original_response
        )

        assert result.status == "warning"

        # The quality guard is detection-only.
        # It must not rewrite or repair the response.
        assert not hasattr(
            result,
            "repaired_response",
        )
        assert not hasattr(
            result,
            "replacement_response",
        )       
        self.pass_test(
            "QUALITY-T10 Warning is detection-only",
            (
                "Quality warning does not rewrite "
                "or repair the tutor response."
            ),
        )


    def test_quality_runtime_t1_aitutor_telemetry(
        self,
    ) -> None:

        tutor = (
            self.build_prompt_integrated_tutor()
        )

        with patch(
            "app.tutor.chat_with_ai",
            return_value=(
                "คุณคิดว่าโครงสร้างของ NPN "
                "ประกอบด้วยส่วนใดบ้าง?"
            ),
        ):

            tutor.respond(
                "ทรานซิสเตอร์ NPN "
                "มีโครงสร้างอย่างไร"
            )

        debug = tutor.get_debug_info()

        self.assert_equal(
            debug.get(
                "response_quality_status"
            ),
            "acceptable",
            "Response quality status",
        )

        self.assert_equal(
            debug.get(
                "response_quality_question_count"
            ),
            1,
            "Response quality question count",
        )

        self.pass_test(
            "QUALITY-RUNTIME-T1 AITutor telemetry",
            (
                "AITutor exposes final-response "
                "quality telemetry."
            ),
        )

    def test_quality_runtime_t2_detection_only(
        self,
    ) -> None:

        tutor = (
            self.build_prompt_integrated_tutor()
        )

        original = (
            "ลองพิจารณาโครงสร้างของทรานซิสเตอร์ NPN "
            + ("ในภาพประกอบและข้อมูลที่กำหนด " * 80)
            + "แล้วอธิบายสิ่งที่สังเกตได้?"
        )

        before = original

        result = (
            tutor.response_quality_guard
            .evaluate(original)
        )

        after = original

        self.assert_equal(
            result.status,
            "warning",
            "Quality warning telemetry",
        )

        self.assert_equal(
            result.too_long,
            True,
            "Long response warning",
        )

        self.assert_equal(
            after,
            before,
            (
                "ResponseQualityGuard must "
                "not mutate response text"
            ),
        )

        self.pass_test(
            "QUALITY-RUNTIME-T2 Detection-only",
            (
                "Quality warning does not mutate "
                "the evaluated Tutor response."
            ),
        )

    def test_quality_runtime_t3_no_extra_llm_calls(
        self,
    ) -> None:

        tutor = (
            self.build_prompt_integrated_tutor()
        )

        with patch(
            "app.tutor.chat_with_ai",
            return_value=(
                "คุณคิดว่าโครงสร้างของ NPN "
                "ประกอบด้วยส่วนใดบ้าง?"
            ),
        ):

            tutor.respond(
                "ทรานซิสเตอร์ NPN "
                "มีโครงสร้างอย่างไร"
            )

        debug = tutor.get_debug_info()

        self.assert_equal(
            debug["ai_usage"]["calls"],
            0,
            (
                "Response quality telemetry "
                "must not add AI calls"
            ),
        )

        self.pass_test(
            "QUALITY-RUNTIME-T3 No extra LLM calls",
            (
                "Response quality observation "
                "is fully deterministic."
            ),
        )

    def test_quality_runtime_t4_final_answer_observed(
        self,
    ) -> None:

        tutor = (
            self.build_prompt_integrated_tutor()
        )

        final_answer = (
            "คุณคิดว่าอิมิตเตอร์ "
            "(Emitter) ทำหน้าที่อะไร?"
        )

        with patch(
            "app.tutor.chat_with_ai",
            return_value=final_answer,
        ):

            answer = tutor.respond(
                "ทรานซิสเตอร์ NPN "
                "มีโครงสร้างอย่างไร"
            )

        debug = tutor.get_debug_info()

        self.assert_equal(
            debug.get(
                "response_quality_character_count"
            ),
            len(answer.strip()),
            (
                "Quality guard did not observe "
                "the final answer"
            ),
        )

        self.pass_test(
            "QUALITY-RUNTIME-T4 Final answer observed",
            (
                "Response quality telemetry is "
                "calculated from final Tutor answer."
            ),
        )

    def test_quality_config_t1_guard_defaults(
        self,
    ) -> None:

        guard = ResponseQualityGuard()

        self.assert_equal(
            guard.max_characters,
            1200,
            "Default max characters",
        )

        self.assert_equal(
            guard.min_characters,
            3,
            "Default min characters",
        )

        self.assert_equal(
            guard.max_questions,
            2,
            "Default max questions",
        )

        self.assert_equal(
            guard.detect_repetition,
            True,
            "Default repetition detection",
        )

        self.pass_test(
            "QUALITY-CONFIG-T1 Guard defaults",
            (
                "Legacy ResponseQualityGuard "
                "defaults remain backward-compatible."
            ),
        )

    def test_quality_config_t2_course_profile_applied(
        self,
    ) -> None:

        tutor = (
            self.build_prompt_integrated_tutor()
        )

        profile = (
            tutor.course_profile
            .response_quality
        )

        guard = (
            tutor.response_quality_guard
        )

        self.assert_equal(
            guard.max_characters,
            profile.max_characters,
            "Configured max characters",
        )

        self.assert_equal(
            guard.min_characters,
            profile.min_characters,
            "Configured min characters",
        )

        self.assert_equal(
            guard.max_questions,
            profile.max_questions,
            "Configured max questions",
        )

        self.assert_equal(
            guard.detect_repetition,
            profile.detect_repetition,
            "Configured repetition detection",
        )

        self.pass_test(
            "QUALITY-CONFIG-T2 Course profile applied",
            (
                "AITutor ResponseQualityGuard uses "
                "typed course quality configuration."
            ),
        )

    def test_quality_config_t3_custom_length_threshold(
        self,
    ) -> None:

        guard = ResponseQualityGuard(
            max_characters=10,
        )

        result = guard.evaluate(
            "12345678901"
        )

        self.assert_equal(
            result.too_long,
            True,
            "Custom maximum length",
        )

        self.assert_equal(
            result.status,
            "warning",
            "Custom maximum length status",
        )

        self.pass_test(
            "QUALITY-CONFIG-T3 Custom length threshold",
            (
                "Configured maximum response length "
                "is enforced deterministically."
            ),
        )

    def test_quality_config_t4_custom_question_threshold(
        self,
    ) -> None:

        guard = ResponseQualityGuard(
            max_questions=1,
        )

        result = guard.evaluate(
            "คำถามแรกคืออะไร? "
            "คำถามที่สองคืออะไร?"
        )

        self.assert_equal(
            result.question_count,
            2,
            "Question count",
        )

        self.assert_equal(
            result.too_many_questions,
            True,
            "Custom question threshold",
        )

        self.pass_test(
            "QUALITY-CONFIG-T4 Custom question threshold",
            (
                "Configured question threshold "
                "is applied deterministically."
            ),
        )

    def test_quality_config_t5_repetition_disabled(
        self,
    ) -> None:

        guard = ResponseQualityGuard(
            detect_repetition=False,
        )

        result = guard.evaluate(
            "ลองพิจารณาวงจรนี้อย่างละเอียด. "
            "ลองพิจารณาวงจรนี้อย่างละเอียด."
        )

        self.assert_equal(
            result.repetitive,
            False,
            "Disabled repetition detection",
        )

        self.assert_true(
            not any(
                "repetition"
                in issue.lower()
                for issue in result.issues
            ),
            (
                "Disabled repetition detection "
                "must not create repetition issue"
            ),
        )

        self.pass_test(
            "QUALITY-CONFIG-T5 Repetition disabled",
            (
                "Course configuration can disable "
                "deterministic repetition detection."
            ),
        )


    def test_quality_config_runtime_t1_debug_schema(
        self,
    ) -> None:

        tutor = (
            self.build_prompt_integrated_tutor()
        )

        debug = tutor.get_debug_info()

        self.assert_equal(
            debug.get(
                "response_quality_config_max_characters"
            ),
            tutor.response_quality_guard
            .max_characters,
            "Runtime max characters telemetry",
        )

        self.assert_equal(
            debug.get(
                "response_quality_config_min_characters"
            ),
            tutor.response_quality_guard
            .min_characters,
            "Runtime min characters telemetry",
        )

        self.assert_equal(
            debug.get(
                "response_quality_config_max_questions"
            ),
            tutor.response_quality_guard
            .max_questions,
            "Runtime max questions telemetry",
        )

        self.assert_equal(
            debug.get(
                "response_quality_config_detect_repetition"
            ),
            tutor.response_quality_guard
            .detect_repetition,
            "Runtime repetition telemetry",
        )

        self.pass_test(
            "QUALITY-CONFIG-RUNTIME-T1 Debug schema",
            (
                "Active response-quality thresholds "
                "are exposed through debug telemetry."
            ),
        )

    def test_quality_config_runtime_t2_active_guard_values(
        self,
    ) -> None:

        tutor = (
            self.build_prompt_integrated_tutor()
        )

        tutor.response_quality_guard = (
            ResponseQualityGuard(
                max_characters=777,
                min_characters=5,
                max_questions=1,
                detect_repetition=False,
            )
        )

        debug = tutor.get_debug_info()

        self.assert_equal(
            debug.get(
                "response_quality_config_max_characters"
            ),
            777,
            "Active runtime max characters",
        )

        self.assert_equal(
            debug.get(
                "response_quality_config_min_characters"
            ),
            5,
            "Active runtime min characters",
        )

        self.assert_equal(
            debug.get(
                "response_quality_config_max_questions"
            ),
            1,
            "Active runtime max questions",
        )

        self.assert_equal(
            debug.get(
                "response_quality_config_detect_repetition"
            ),
            False,
            "Active runtime repetition setting",
        )

        self.pass_test(
            "QUALITY-CONFIG-RUNTIME-T2 Active guard values",
            (
                "Debug telemetry reflects the "
                "active ResponseQualityGuard configuration."
            ),
        )

    def test_quality_config_runtime_t3_no_llm_calls(
        self,
    ) -> None:

        tutor = (
            self.build_prompt_integrated_tutor()
        )

        before = (
            tutor.get_debug_info()[
                "ai_usage"
            ]["calls"]
        )

        tutor.get_debug_info()

        after = (
            tutor.get_debug_info()[
                "ai_usage"
            ]["calls"]
        )

        self.assert_equal(
            after,
            before,
            (
                "Quality configuration telemetry "
                "must not add AI calls"
            ),
        )

        self.pass_test(
            "QUALITY-CONFIG-RUNTIME-T3 No LLM calls",
            (
                "Runtime quality configuration "
                "telemetry is fully deterministic."
            ),
        )

    def test_quality_policy_runtime_t1_telemetry(
        self,
    ) -> None:

        tutor = (
            self.build_prompt_integrated_tutor()
        )

        with patch(
            "app.tutor.chat_with_ai",
            return_value=(
                "คุณคิดว่าโครงสร้างของ NPN "
                "ประกอบด้วยส่วนใดบ้าง?"
            ),
        ):

            tutor.respond(
                "ทรานซิสเตอร์ NPN "
                "มีโครงสร้างอย่างไร"
            )

        debug = tutor.get_debug_info()

        self.assert_equal(
            debug.get(
                "response_quality_policy_status"
            ),
            "acceptable",
            "Quality policy runtime status",
        )

        self.assert_equal(
            debug.get(
                "response_quality_policy_source_status"
            ),
            "acceptable",
            "Quality policy source status",
        )

        self.assert_equal(
            debug.get(
                "response_quality_policy_requires_attention"
            ),
            False,
            "Quality policy attention flag",
        )

        self.pass_test(
            "QUALITY-POLICY-RUNTIME-T1 Telemetry",
            (
                "AITutor exposes deterministic "
                "response-quality policy telemetry."
            ),
        )

    def test_quality_policy_runtime_t2_source_alignment(
        self,
    ) -> None:

        tutor = (
            self.build_prompt_integrated_tutor()
        )

        with patch(
            "app.tutor.chat_with_ai",
            return_value=(
                "คุณคิดว่า NPN "
                "ประกอบด้วยส่วนใดบ้าง?"
            ),
        ):

            tutor.respond(
                "โครงสร้าง NPN"
            )

        debug = tutor.get_debug_info()

        self.assert_equal(
            debug.get(
                "response_quality_policy_source_status"
            ),
            debug.get(
                "response_quality_status"
            ),
            (
                "Policy source must match "
                "final quality result"
            ),
        )

        self.pass_test(
            "QUALITY-POLICY-RUNTIME-T2 Source alignment",
            (
                "ResponseQualityPolicy receives "
                "the final ResponseQualityResult."
            ),
        )       
    def test_quality_policy_runtime_t3_detection_only(
        self,
    ) -> None:

        tutor = (
            self.build_prompt_integrated_tutor()
        )

        quality = (
            tutor.response_quality_guard
            .evaluate(
                "คำถามหนึ่ง? คำถามสอง? คำถามสาม?"
            )
        )

        policy = (
            tutor.response_quality_policy
            .evaluate(
                quality
            )
        )

        self.assert_true(
            not hasattr(
                policy,
                "repaired_response",
            ),
            (
                "Quality policy must not expose "
                "a repaired response"
            ),
        )

        self.assert_true(
            not hasattr(
                policy,
                "replacement_response",
            ),
            (
                "Quality policy must not expose "
                "a replacement response"
            ),
        )

        self.pass_test(
            "QUALITY-POLICY-RUNTIME-T3 Detection-only",
            (
                "Response quality policy remains "
                "observational only."
            ),
        )

    def test_quality_policy_runtime_t4_no_llm_calls(
        self,
    ) -> None:

        tutor = (
            self.build_prompt_integrated_tutor()
        )

        before = (
            tutor.get_debug_info()[
                "ai_usage"
            ]["calls"]
        )

        quality = (
            tutor.response_quality_guard
            .evaluate(
                "NPN?"
            )
        )

        tutor.response_quality_policy.evaluate(
            quality
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
                "Response quality policy "
                "must not add AI calls"
            ),
        )

        self.pass_test(
            "QUALITY-POLICY-RUNTIME-T4 No LLM calls",
            (
                "ResponseQualityPolicy is fully "
                "deterministic at runtime."
            ),
        )

    def test_quality_policy_matrix_t1_acceptable(
        self,
    ) -> None:

        policy = ResponseQualityPolicy()

        quality = ResponseQualityResult(
            status="acceptable",
            reason="No quality issue.",
            issues=[],
            character_count=100,
            token_like_count=10,
            question_count=1,
            sentence_count=1,
            too_long=False,
            too_short=False,
            too_many_questions=False,
            repetitive=False,
        )

        result = policy.evaluate(
            quality
        )

        self.assert_equal(
            result.status,
            "acceptable",
            "Acceptable quality policy status",
        )

        self.assert_equal(
            result.source_status,
            "acceptable",
            "Acceptable source status",
        )

        self.assert_equal(
            result.issue_count,
            0,
            "Acceptable issue count",
        )

        self.assert_equal(
            result.requires_attention,
            False,
            "Acceptable attention flag",
        )

        self.pass_test(
            "QUALITY-POLICY-MATRIX-T1 Acceptable",
            (
                "Acceptable quality result maps "
                "to acceptable policy status."
            ),
        )

    def test_quality_policy_matrix_t2_too_long_advisory(
        self,
    ) -> None:

        policy = ResponseQualityPolicy()

        quality = ResponseQualityResult(
            status="warning",
            reason="Response too long.",
            issues=[
                "Tutor response exceeds the configured "
                "maximum response length."
            ],
            character_count=1500,
            token_like_count=100,
            question_count=1,
            sentence_count=10,
            too_long=True,
            too_short=False,
            too_many_questions=False,
            repetitive=False,
        )

        result = policy.evaluate(
            quality
        )

        self.assert_equal(
            result.status,
            "advisory",
            "Too-long advisory status",
        )

        self.assert_equal(
            result.issue_count,
            1,
            "Too-long issue count",
        )

        self.assert_equal(
            result.requires_attention,
            False,
            "Too-long attention flag",
        )

        self.pass_test(
            "QUALITY-POLICY-MATRIX-T2 Too long",
            (
                "Single too-long issue maps "
                "to advisory policy status."
            ),
        )

    def test_quality_policy_matrix_t3_too_short_advisory(
        self,
    ) -> None:

        policy = ResponseQualityPolicy()

        quality = ResponseQualityResult(
            status="warning",
            reason="Response too short.",
            issues=[
                "Tutor response is unusually short."
            ],
            character_count=1,
            token_like_count=1,
            question_count=0,
            sentence_count=1,
            too_long=False,
            too_short=True,
            too_many_questions=False,
            repetitive=False,
        )

        result = policy.evaluate(
            quality
        )

        self.assert_equal(
            result.status,
            "advisory",
            "Too-short advisory status",
        )

        self.assert_equal(
            result.requires_attention,
            False,
            "Too-short attention flag",
        )

        self.pass_test(
            "QUALITY-POLICY-MATRIX-T3 Too short",
            (
                "Single too-short issue maps "
                "to advisory policy status."
            ),
        )

    def test_quality_policy_matrix_t4_questions_advisory(
        self,
    ) -> None:

        policy = ResponseQualityPolicy()

        quality = ResponseQualityResult(
            status="warning",
            reason="Too many questions.",
            issues=[
                "Tutor response contains more questions "
                "than the configured quality threshold."
            ],
            character_count=100,
            token_like_count=10,
            question_count=3,
            sentence_count=3,
            too_long=False,
            too_short=False,
            too_many_questions=True,
            repetitive=False,
        )

        result = policy.evaluate(
            quality
        )

        self.assert_equal(
            result.status,
            "advisory",
            "Question advisory status",
        )

        self.assert_equal(
            result.requires_attention,
            False,
            "Question advisory attention flag",
        )

        self.pass_test(
            "QUALITY-POLICY-MATRIX-T4 Questions",
            (
                "Single excessive-question issue maps "
                "to advisory policy status."
            ),
        )

    def test_quality_policy_matrix_t5_repetition_advisory(
        self,
    ) -> None:

        policy = ResponseQualityPolicy()

        quality = ResponseQualityResult(
            status="warning",
            reason="Repeated content.",
            issues=[
                "Tutor response contains repetition."
            ],
            character_count=200,
            token_like_count=20,
            question_count=0,
            sentence_count=4,
            too_long=False,
            too_short=False,
            too_many_questions=False,
            repetitive=True,
        )

        result = policy.evaluate(
            quality
        )

        self.assert_equal(
            result.status,
            "advisory",
            "Repetition advisory status",
        )

        self.assert_equal(
            result.issue_count,
            1,
            "Repetition issue count",
        )

        self.assert_equal(
            result.requires_attention,
            False,
            "Repetition attention flag",
        )

        self.pass_test(
            "QUALITY-POLICY-MATRIX-T5 Repetition",
            (
                "Single repetition issue maps "
                "to advisory policy status."
            ),
        )

    def test_quality_policy_matrix_t6_multiple_attention(
        self,
    ) -> None:

        policy = ResponseQualityPolicy()

        quality = ResponseQualityResult(
            status="warning",
            reason="Multiple quality issues.",
            issues=[
                "Tutor response exceeds the configured "
                "maximum response length.",
                "Tutor response contains repetition.",
            ],
            character_count=1500,
            token_like_count=100,
            question_count=0,
            sentence_count=20,
            too_long=True,
            too_short=False,
            too_many_questions=False,
            repetitive=True,
        )

        result = policy.evaluate(
            quality
        )

        self.assert_equal(
            result.status,
            "attention",
            "Multiple-issue policy status",
        )

        self.assert_equal(
            result.issue_count,
            2,
            "Multiple-issue count",
        )

        self.assert_equal(
            result.requires_attention,
            True,
            "Multiple-issue attention flag",
        )

        self.pass_test(
            "QUALITY-POLICY-MATRIX-T6 Multiple issues",
            (
                "Multiple quality issues map "
                "to attention policy status."
            ),
        )

    def test_quality_policy_matrix_t7_uncertain_attention(
        self,
    ) -> None:

        policy = ResponseQualityPolicy()

        quality = ResponseQualityResult(
            status="uncertain",
            reason="Empty tutor response.",
            issues=[
                "Empty tutor response."
            ],
            character_count=0,
            token_like_count=0,
            question_count=0,
            sentence_count=0,
            too_long=False,
            too_short=False,
            too_many_questions=False,
            repetitive=False,
        )

        result = policy.evaluate(
            quality
        )

        self.assert_equal(
            result.status,
            "attention",
            "Uncertain policy status",
        )

        self.assert_equal(
            result.source_status,
            "uncertain",
            "Uncertain source status",
        )

        self.assert_equal(
            result.requires_attention,
            True,
            "Uncertain attention flag",
        )

        self.pass_test(
            "QUALITY-POLICY-MATRIX-T7 Uncertain",
            (
                "Uncertain quality result maps "
                "conservatively to attention."
            ),
        )

    def test_quality_policy_matrix_t8_unknown_fail_safe(
        self,
    ) -> None:

        policy = ResponseQualityPolicy()

        quality = ResponseQualityResult(
            status="unexpected_status",
            reason="Unknown status.",
            issues=[],
            character_count=100,
            token_like_count=10,
            question_count=0,
            sentence_count=1,
            too_long=False,
            too_short=False,
            too_many_questions=False,
            repetitive=False,
        )

        result = policy.evaluate(
            quality
        )

        self.assert_equal(
            result.status,
            "attention",
            "Unknown fail-safe status",
        )

        self.assert_equal(
            result.source_status,
            "unexpected_status",
            "Unknown source status preserved",
        )

        self.assert_equal(
            result.requires_attention,
            True,
            "Unknown fail-safe attention",
        )

        self.pass_test(
            "QUALITY-POLICY-MATRIX-T8 Unknown fail-safe",
            (
                "Unknown quality status fails "
                "safely to attention."
            ),
        )

    def test_quality_policy_matrix_t9_invalid_type(
        self,
    ) -> None:

        policy = ResponseQualityPolicy()

        raised = False

        try:

            policy.evaluate(
                "not-a-quality-result"
            )

        except TypeError:

            raised = True

        self.assert_equal(
            raised,
            True,
            "Invalid policy input type",
        )

        self.pass_test(
            "QUALITY-POLICY-MATRIX-T9 Invalid type",
            (
                "ResponseQualityPolicy rejects "
                "invalid input types."
            ),
        )

    def test_quality_policy_matrix_t10_deterministic(
        self,
    ) -> None:

        policy = ResponseQualityPolicy()

        quality = ResponseQualityResult(
            status="warning",
            reason="Too many questions.",
            issues=[
                "Tutor response contains more questions "
                "than the configured quality threshold."
            ],
            character_count=100,
            token_like_count=10,
            question_count=3,
            sentence_count=3,
            too_long=False,
            too_short=False,
            too_many_questions=True,
            repetitive=False,
        )

        first = policy.evaluate(
            quality
        )

        second = policy.evaluate(
            quality
        )

        self.assert_equal(
            first,
            second,
            "Deterministic policy result",
        )

        self.pass_test(
            "QUALITY-POLICY-MATRIX-T10 Deterministic",
            (
                "Identical quality results produce "
                "identical policy results."
            ),
        )


    def test_quality_history_runtime_t1_starts_empty(
        self,
    ) -> None:

        tutor = (
            self.build_prompt_integrated_tutor()
        )

        history = (
            tutor.response_quality_history
        )

        self.assert_equal(
            history.count,
            0,
            "Initial quality history count",
        )

        self.assert_equal(
            history.latest,
            None,
            "Initial latest quality record",
        )

        self.assert_equal(
            history.records,
            (),
            "Initial quality history records",
        )

        self.pass_test(
            "QUALITY-HISTORY-RUNTIME-T1 Starts empty",
            (
                "AITutor response-quality history "
                "starts empty."
            ),
        )

    def test_quality_history_runtime_t2_records_turn(
        self,
    ) -> None:

        tutor = (
            self.build_prompt_integrated_tutor()
        )

        with patch(
            "app.tutor.chat_with_ai",
            return_value=(
                "คุณคิดว่าโครงสร้างของ NPN "
                "ประกอบด้วยส่วนใดบ้าง?"
            ),
        ):

            answer = tutor.respond(
                "ทรานซิสเตอร์ NPN "
                "มีโครงสร้างอย่างไร"
            )

        history = (
            tutor.response_quality_history
        )

        record = history.latest

        self.assert_equal(
            history.count,
            1,
            "Quality history count",
        )

        self.assert_true(
            record is not None,
            "Latest quality history record",
        )

        self.assert_equal(
            record.turn_number,
            tutor.state.turn_count,
            "Recorded turn number",
        )

        self.assert_equal(
            record.character_count,
            len(answer.strip()),
            (
                "History must describe "
                "the final Tutor answer"
            ),
        )

        self.assert_equal(
            record.quality_status,
            tutor.last_response_quality.status,
            "Recorded quality status",
        )

        self.assert_equal(
            record.policy_status,
            tutor.last_response_quality_policy.status,
            "Recorded policy status",
        )

        self.pass_test(
            "QUALITY-HISTORY-RUNTIME-T2 Records turn",
            (
                "A completed Tutor turn creates "
                "one final-response quality record."
            ),
        )

    def test_quality_history_runtime_t3_threshold_snapshot(
        self,
    ) -> None:

        tutor = (
            self.build_prompt_integrated_tutor()
        )

        with patch(
            "app.tutor.chat_with_ai",
            return_value=(
                "คุณคิดว่า NPN "
                "ประกอบด้วยส่วนใดบ้าง?"
            ),
        ):

            tutor.respond(
                "โครงสร้างของ NPN"
            )

        record = (
            tutor.response_quality_history
            .latest
        )

        guard = (
            tutor.response_quality_guard
        )

        self.assert_equal(
            record.max_characters,
            guard.max_characters,
            "History max characters snapshot",
        )

        self.assert_equal(
            record.min_characters,
            guard.min_characters,
            "History min characters snapshot",
        )

        self.assert_equal(
            record.max_questions,
            guard.max_questions,
            "History max questions snapshot",
        )

        self.assert_equal(
            record.detect_repetition,
            guard.detect_repetition,
            "History repetition setting snapshot",
        )

        self.pass_test(
            "QUALITY-HISTORY-RUNTIME-T3 Threshold snapshot",
            (
                "Each quality-history record preserves "
                "the active quality thresholds."
            ),
        )

    def test_quality_history_runtime_t4_reset_clears(
        self,
    ) -> None:

        tutor = (
            self.build_prompt_integrated_tutor()
        )

        record = ResponseQualityTurnRecord(
            turn_number=1,
            quality_status="acceptable",
            policy_status="acceptable",
            issue_count=0,
            requires_attention=False,
            character_count=50,
            token_like_count=5,
            question_count=1,
            sentence_count=1,
            too_long=False,
            too_short=False,
            too_many_questions=False,
            repetitive=False,
        )

        tutor.response_quality_history.add(
            record
        )

        self.assert_equal(
            tutor.response_quality_history.count,
            1,
            "Pre-reset quality history count",
        )

        tutor.reset()

        self.assert_equal(
            tutor.response_quality_history.count,
            0,
            "Post-reset quality history count",
        )

        self.assert_equal(
            tutor.response_quality_history.latest,
            None,
            "Post-reset latest record",
        )

        self.assert_equal(
            tutor.state.turn_count,
            0,
            "Post-reset turn count",
        )

        self.pass_test(
            "QUALITY-HISTORY-RUNTIME-T4 Reset clears",
            (
                "Conversation reset clears "
                "response-quality history."
            ),
        )
    def test_quality_history_runtime_t5_no_llm_calls(
        self,
    ) -> None:

        tutor = (
            self.build_prompt_integrated_tutor()
        )

        before = (
            tutor.get_debug_info()[
                "ai_usage"
            ]["calls"]
        )

        record = ResponseQualityTurnRecord(
            turn_number=1,
            quality_status="acceptable",
            policy_status="acceptable",
            issue_count=0,
            requires_attention=False,
            character_count=50,
            token_like_count=5,
            question_count=1,
            sentence_count=1,
            too_long=False,
            too_short=False,
            too_many_questions=False,
            repetitive=False,
        )

        tutor.response_quality_history.add(
            record
        )

        _ = (
            tutor.response_quality_history
            .records
        )

        _ = (
            tutor.response_quality_history
            .latest
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
                "Response-quality history "
                "must not add AI calls"
            ),
        )

        self.pass_test(
            "QUALITY-HISTORY-RUNTIME-T5 No LLM calls",
            (
                "Turn-level quality history "
                "is fully deterministic."
            ),
        )

    def test_quality_analytics_t1_empty_history(
        self,
    ) -> None:

        history = (
            ResponseQualityHistory()
        )

        result = (
            ResponseQualityAnalytics()
            .summarize(
                history
            )
        )

        self.assert_equal(
            result.total_turns,
            0,
            "Empty analytics total turns",
        )

        self.assert_equal(
            result.attention_rate,
            0.0,
            "Empty analytics attention rate",
        )

        self.assert_equal(
            result.latest_policy_status,
            None,
            "Empty analytics latest policy",
        )

        self.pass_test(
            "QUALITY-ANALYTICS-T1 Empty history",
            (
                "Empty quality history produces "
                "safe zero-value analytics."
            ),
        )

    def test_quality_analytics_t2_summary_matrix(
        self,
    ) -> None:

        history = (
            ResponseQualityHistory()
        )

        records = (
            ResponseQualityTurnRecord(
                turn_number=1,
                quality_status="acceptable",
                policy_status="acceptable",
                issue_count=0,
                requires_attention=False,
                character_count=100,
                token_like_count=10,
                question_count=1,
                sentence_count=2,
                too_long=False,
                too_short=False,
                too_many_questions=False,
                repetitive=False,
            ),
            ResponseQualityTurnRecord(
                turn_number=2,
                quality_status="warning",
                policy_status="advisory",
                issue_count=1,
                requires_attention=False,
                character_count=200,
                token_like_count=20,
                question_count=2,
                sentence_count=4,
                too_long=True,
                too_short=False,
                too_many_questions=False,
                repetitive=False,
            ),
            ResponseQualityTurnRecord(
                turn_number=3,
                quality_status="warning",
                policy_status="attention",
                issue_count=2,
                requires_attention=True,
                character_count=300,
                token_like_count=30,
                question_count=3,
                sentence_count=6,
                too_long=True,
                too_short=False,
                too_many_questions=True,
                repetitive=False,
            ),
        )

        for record in records:
            history.add(
                record
            )

        result = (
            ResponseQualityAnalytics()
            .summarize(
                history
            )
        )

        self.assert_equal(
            result.total_turns,
            3,
            "Analytics total turns",
        )

        self.assert_equal(
            result.acceptable_turns,
            1,
            "Analytics acceptable turns",
        )

        self.assert_equal(
            result.advisory_turns,
            1,
            "Analytics advisory turns",
        )

        self.assert_equal(
            result.attention_turns,
            1,
            "Analytics attention turns",
        )

        self.assert_equal(
            result.issue_turns,
            2,
            "Analytics issue turns",
        )

        self.assert_equal(
            result.total_issues,
            3,
            "Analytics total issues",
        )

        self.assert_equal(
            result.average_character_count,
            200.0,
            "Analytics average characters",
        )

        self.assert_equal(
            result.average_question_count,
            2.0,
            "Analytics average questions",
        )

        self.pass_test(
            "QUALITY-ANALYTICS-T2 Summary matrix",
            (
                "Session quality analytics "
                "aggregate history correctly."
            ),
        )

    def test_quality_analytics_t3_runtime_debug(
        self,
    ) -> None:

        tutor = (
            self.build_prompt_integrated_tutor()
        )

        with patch(
            "app.tutor.chat_with_ai",
            return_value=(
                "คุณคิดว่า NPN "
                "ประกอบด้วยส่วนใดบ้าง?"
            ),
        ):

            tutor.respond(
                "โครงสร้างของ NPN"
            )

        debug = tutor.get_debug_info()

        self.assert_equal(
            debug.get(
                "response_quality_analytics_total_turns"
            ),
            tutor.response_quality_history.count,
            "Runtime analytics turn count",
        )

        self.assert_equal(
            debug.get(
                "response_quality_analytics_latest_policy_status"
            ),
            tutor.response_quality_history
            .latest.policy_status,
            "Runtime analytics latest policy",
        )

        self.pass_test(
            "QUALITY-ANALYTICS-T3 Runtime debug",
            (
                "AITutor exposes session-level "
                "quality analytics through debug info."
            ),
        )

    def test_quality_analytics_t4_reset(
        self,
    ) -> None:

        tutor = (
            self.build_prompt_integrated_tutor()
        )

        record = ResponseQualityTurnRecord(
            turn_number=1,
            quality_status="warning",
            policy_status="attention",
            issue_count=2,
            requires_attention=True,
            character_count=100,
            token_like_count=10,
            question_count=3,
            sentence_count=3,
            too_long=False,
            too_short=False,
            too_many_questions=True,
            repetitive=True,
        )

        tutor.response_quality_history.add(
            record
        )

        before = tutor.get_debug_info()

        self.assert_equal(
            before.get(
                "response_quality_analytics_total_turns"
            ),
            1,
            "Pre-reset analytics turns",
        )

        tutor.reset()

        after = tutor.get_debug_info()

        self.assert_equal(
            after.get(
                "response_quality_analytics_total_turns"
            ),
            0,
            "Post-reset analytics turns",
        )

        self.assert_equal(
            after.get(
                "response_quality_analytics_attention_rate"
            ),
            0.0,
            "Post-reset attention rate",
        )

        self.pass_test(
            "QUALITY-ANALYTICS-T4 Reset",
            (
                "Quality analytics reset naturally "
                "when session history is cleared."
            ),
        )
    def test_quality_analytics_t5_read_only(
        self,
    ) -> None:

        history = (
            ResponseQualityHistory()
        )

        record = ResponseQualityTurnRecord(
            turn_number=1,
            quality_status="acceptable",
            policy_status="acceptable",
            issue_count=0,
            requires_attention=False,
            character_count=100,
            token_like_count=10,
            question_count=1,
            sentence_count=1,
            too_long=False,
            too_short=False,
            too_many_questions=False,
            repetitive=False,
        )

        history.add(
            record
        )

        before = history.records

        ResponseQualityAnalytics().summarize(
            history
        )

        after = history.records

        self.assert_equal(
            after,
            before,
            (
                "Analytics must not mutate "
                "quality history"
            ),
        )

        self.pass_test(
            "QUALITY-ANALYTICS-T5 Read-only",
            (
                "Analytics summarize history "
                "without mutating it."
            ),
        )

    def test_quality_analytics_t6_deterministic(
        self,
    ) -> None:

        history = (
            ResponseQualityHistory()
        )

        history.add(
            ResponseQualityTurnRecord(
                turn_number=1,
                quality_status="acceptable",
                policy_status="acceptable",
                issue_count=0,
                requires_attention=False,
                character_count=100,
                token_like_count=10,
                question_count=1,
                sentence_count=1,
                too_long=False,
                too_short=False,
                too_many_questions=False,
                repetitive=False,
            )
        )

        analytics = (
            ResponseQualityAnalytics()
        )

        first = analytics.summarize(
            history
        )

        second = analytics.summarize(
            history
        )

        self.assert_equal(
            first,
            second,
            "Deterministic analytics result",
        )

        self.pass_test(
            "QUALITY-ANALYTICS-T6 Deterministic",
            (
                "Identical quality history produces "
                "identical analytics."
            ),
        )

    def test_quality_analytics_t7_no_llm_calls(
        self,
    ) -> None:

        tutor = (
            self.build_prompt_integrated_tutor()
        )

        before = (
            tutor.get_debug_info()[
                "ai_usage"
            ]["calls"]
        )

        tutor.response_quality_analytics.summarize(
            tutor.response_quality_history
        )

        tutor.get_debug_info()

        after = (
            tutor.get_debug_info()[
                "ai_usage"
            ]["calls"]
        )

        self.assert_equal(
            after,
            before,
            (
                "Response-quality analytics "
                "must not add AI calls"
            ),
        )

        self.pass_test(
            "QUALITY-ANALYTICS-T7 No LLM calls",
            (
                "Session-level quality analytics "
                "is fully deterministic."
            ),
        )




    # =========================================================
    # Runner
    # =========================================================

    def run(self) -> int:

        print(
            "\nStep 16.17 Response Quality "
            "Regression Suite\n"
        )

        tests = [
            # -------------------------------------------------
            # Language consistency
            # -------------------------------------------------
            self.test_lng_t1_thai_consistent,
            self.test_lng_t2_english_mismatch,
            self.test_lng_t3_thai_with_technical_english,
            self.test_lng_t4_mixed_uncertain,
            self.test_lng_t5_short_uncertain,
            self.test_lng_t6_empty_uncertain,
            self.test_lng_t7_mathematics_thai,
            self.test_lng_t8_tutor_telemetry,
            self.test_lng_t9_no_extra_llm_calls,
            self.test_lng_t10_mismatch_telemetry_only,
            self.test_lng_t11_resolved_style_language_precedence,
            self.test_lng_t12_course_language_does_not_override_style,

            # -------------------------------------------------
            # Response style resolution
            # -------------------------------------------------
            self.test_style_t1_electronics_typed_resolution,
            self.test_style_t2_mathematics_typed_resolution,
            self.test_style_t3_style_language_precedence,
            self.test_style_t4_course_language_fallback,
            self.test_style_t5_thai_alias_normalization,
            self.test_style_t6_english_alias_normalization,
            self.test_style_t7_default_fields_preserved,
            self.test_style_t8_source_profile_immutable,

            # -------------------------------------------------
            # Response style prompt instructions
            # -------------------------------------------------
            self.test_style_inst_t1_electronics_instructions,
            self.test_style_inst_t2_english_language,
            self.test_style_inst_t3_technical_english_disabled,
            self.test_style_inst_t4_zero_guiding_questions,
            self.test_style_inst_t5_multiple_guiding_questions,
            self.test_style_inst_t6_deterministic_output,

            # -------------------------------------------------
            # Response style prompt integration
            # -------------------------------------------------
            self.test_style_prompt_t1_instruction_in_system_prompt,
            self.test_style_prompt_t2_technical_english_policy,
            self.test_style_prompt_t3_tone_and_depth,
            self.test_style_prompt_t4_question_policy,
            self.test_style_prompt_t5_legacy_language_rules_absent,
            self.test_style_prompt_t6_deterministic_prompt,
            self.test_style_prompt_t7_knowledge_context_preserved,
            self.test_style_prompt_t8_no_extra_llm_calls,
            self.test_style_prompt_t9_base_language_rule_absent,
            self.test_style_prompt_t10_strategy_language_rule_absent,

            # -------------------------------------------------
            # Response style runtime telemetry
            # -------------------------------------------------
            self.test_style_runtime_t1_debug_schema,
            self.test_style_runtime_t2_term_policy_telemetry,
            self.test_style_runtime_t3_no_extra_llm_calls,

            # -------------------------------------------------
            # Response Quality Guard
            # -------------------------------------------------
            self.test_quality_t1_normal_response_acceptable,
            self.test_quality_t2_multiple_questions_warning,
            self.test_quality_t3_long_response_warning,
            self.test_quality_t4_repetition_warning,
            self.test_quality_t5_empty_response_uncertain,
            self.test_quality_t6_short_technical_response_safe,
            self.test_quality_t7_result_metrics,
            self.test_quality_t8_deterministic_output,
            self.test_quality_t9_no_llm_dependency,
            self.test_quality_t10_warning_is_detection_only,

            self.test_quality_runtime_t1_aitutor_telemetry,
            self.test_quality_runtime_t2_detection_only,
            self.test_quality_runtime_t3_no_extra_llm_calls,
            self.test_quality_runtime_t4_final_answer_observed,

            self.test_quality_config_t1_guard_defaults,
            self.test_quality_config_t2_course_profile_applied,
            self.test_quality_config_t3_custom_length_threshold,
            self.test_quality_config_t4_custom_question_threshold,
            self.test_quality_config_t5_repetition_disabled,

            self.test_quality_config_runtime_t1_debug_schema,
            self.test_quality_config_runtime_t2_active_guard_values,
            self.test_quality_config_runtime_t3_no_llm_calls,

            self.test_quality_policy_runtime_t1_telemetry,
            self.test_quality_policy_runtime_t2_source_alignment,
            self.test_quality_policy_runtime_t3_detection_only,
            self.test_quality_policy_runtime_t4_no_llm_calls,

            self.test_quality_policy_matrix_t1_acceptable,
            self.test_quality_policy_matrix_t2_too_long_advisory,
            self.test_quality_policy_matrix_t3_too_short_advisory,
            self.test_quality_policy_matrix_t4_questions_advisory,
            self.test_quality_policy_matrix_t5_repetition_advisory,
            self.test_quality_policy_matrix_t6_multiple_attention,
            self.test_quality_policy_matrix_t7_uncertain_attention,
            self.test_quality_policy_matrix_t8_unknown_fail_safe,
            self.test_quality_policy_matrix_t9_invalid_type,
            self.test_quality_policy_matrix_t10_deterministic,  

            self.test_quality_history_runtime_t1_starts_empty,
            self.test_quality_history_runtime_t2_records_turn,
            self.test_quality_history_runtime_t3_threshold_snapshot,
            self.test_quality_history_runtime_t4_reset_clears,
            self.test_quality_history_runtime_t5_no_llm_calls,     

            self.test_quality_analytics_t1_empty_history,
            self.test_quality_analytics_t2_summary_matrix,
            self.test_quality_analytics_t3_runtime_debug,
            self.test_quality_analytics_t4_reset,
            self.test_quality_analytics_t5_read_only,
            self.test_quality_analytics_t6_deterministic,
            self.test_quality_analytics_t7_no_llm_calls,    
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

    def print_summary(
        self,
    ) -> None:

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
            -
            passed
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
        RegressionSuite16_17()
    )

    exit_code = (
        suite.run()
    )

    raise SystemExit(
        exit_code
    )


if __name__ == "__main__":

    main()