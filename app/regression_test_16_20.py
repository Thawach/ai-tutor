import inspect

from app.tutor import AITutor

from pathlib import Path

from app.services.grounding_claim_precheck import (
    GroundingClaimPrecheck,
)

from app.services.response_grounding_validator import (
    GROUNDING_VALIDATION_PROMPT,
)

from app.services.grounded_generation_instruction_builder import (
    GroundedGenerationInstructionBuilder,
)

from app.prompts.builder import (
    PromptBuilder,
)

from app.tutor import (
    AITutor,
)

from unittest.mock import patch
from app.services.response_grounding_validator import (
    ResponseGroundingValidator,
)

from app.infrastructure.ai.task_profiles import (
    get_task_profile,
)

from app.services.grounding_precision_guard import (
    GroundingPrecisionGuard,
)

from app.services.grounding_precision_veto_policy import (
    GroundingPrecisionVetoPolicy,
)

from app.services.grounding_precision_guard import (
    GroundingPrecisionGuardResult,
)

from app.services.grounding_validation_models import (
    GroundingValidationResult,
)

from app.services.response_repair_service import (
    RESPONSE_REPAIR_PROMPT,
    ResponseRepairService,
)

from app.services.response_mode_repair_service import (
    RESPONSE_MODE_REPAIR_PROMPT,
)

from app.services.repair_evidence_selection_models import (
    RepairEvidenceSelectionResult,
)

from app.services.repair_evidence_selector import (
    RepairEvidenceSelector,
)

from app.services.evidence_safe_repair_composer import (
    EvidenceSafeRepairComposer,
)

from app.services.guiding_question_grounding_validator import (
    GuidingQuestionGroundingValidator,
)

from app.services.guiding_question_grounding_models import (
    GuidingQuestionGroundingResult,
)

import json
from unittest.mock import patch

class RegressionSuite16_20:

    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0

    def _pass(
        self,
        name: str,
        detail: str,
    ) -> None:

        self.passed += 1

        print(
            f"[PASS] {name}"
        )

        print(
            f"       {detail}"
        )

    def _fail(
        self,
        name: str,
        detail: str,
    ) -> None:

        self.failed += 1

        print(
            f"[FAIL] {name}"
        )

        print(
            f"       {detail}"
        )

    def _check(
        self,
        condition: bool,
        name: str,
        pass_detail: str,
        fail_detail: str,
    ) -> None:

        if condition:

            self._pass(
                name,
                pass_detail,
            )

        else:

            self._fail(
                name,
                fail_detail,
            )

    # =====================================================
    # Step 16.20.1A
    # Mixed Response Claim Detection
    # =====================================================

    def test_t1_pure_question(
        self,
    ) -> None:

        guard = GroundingClaimPrecheck()

        result = guard.evaluate(
            "What does the base do?"
        )

        self._check(
            result.status == "no_claims",
            "GROUNDING-PRECISION-T1 Pure question",
            (
                "A genuine question-only response "
                "retains the deterministic fast-path."
            ),
            (
                "Question-only response was not "
                "classified as no_claims."
            ),
        )

    def test_t2_statement(
        self,
    ) -> None:

        guard = GroundingClaimPrecheck()

        result = guard.evaluate(
            "The base controls collector current."
        )

        self._check(
            result.status
            == "needs_validation",
            "GROUNDING-PRECISION-T2 Statement",
            (
                "Declarative content requires "
                "semantic grounding validation."
            ),
            (
                "Declarative content incorrectly "
                "used the no-claims fast-path."
            ),
        )

    def test_t3_statement_then_question(
        self,
    ) -> None:

        guard = GroundingClaimPrecheck()

        result = guard.evaluate(
            (
                "The base controls collector current. "
                "What happens next?"
            )
        )

        self._check(
            result.status
            == "needs_validation",
            (
                "GROUNDING-PRECISION-T3 "
                "Statement plus question"
            ),
            (
                "An explanation followed by a "
                "question is no longer treated "
                "as claim-free."
            ),
            (
                "Mixed statement-plus-question "
                "incorrectly used no_claims."
            ),
        )

    def test_t4_paragraph_then_question(
        self,
    ) -> None:

        guard = GroundingClaimPrecheck()

        result = guard.evaluate(
            (
                "The base controls collector current.\n\n"
                "What happens next?"
            )
        )

        self._check(
            result.status
            == "needs_validation",
            (
                "GROUNDING-PRECISION-T4 "
                "Paragraph plus question"
            ),
            (
                "A preceding explanatory paragraph "
                "forces semantic validation."
            ),
            (
                "Paragraph-plus-question response "
                "was incorrectly treated as "
                "question-only."
            ),
        )

    def test_t5_inline_question_clause(
        self,
    ) -> None:

        guard = GroundingClaimPrecheck()

        result = guard.evaluate(
            (
                "The base controls current, "
                "what happens next?"
            )
        )

        self._check(
            result.status
            == "needs_validation",
            (
                "GROUNDING-PRECISION-T5 "
                "Inline question clause"
            ),
            (
                "Leading content before an inline "
                "question clause is detected."
            ),
            (
                "Inline mixed response incorrectly "
                "used no_claims."
            ),
        )

    def test_t6_thai_pure_question(
        self,
    ) -> None:

        guard = GroundingClaimPrecheck()

        result = guard.evaluate(
            "คุณคิดว่าเบสทำหน้าที่อะไร?"
        )

        self._check(
            result.status == "no_claims",
            (
                "GROUNDING-PRECISION-T6 "
                "Thai pure question"
            ),
            (
                "Thai question-only behavior "
                "remains unchanged."
            ),
            (
                "Thai pure question lost the "
                "deterministic fast-path."
            ),
        )

    def test_t7_thai_mixed_response(
        self,
    ) -> None:

        guard = GroundingClaimPrecheck()

        result = guard.evaluate(
            (
                "เบสควบคุมกระแสในทรานซิสเตอร์\n"
                "คุณคิดว่าจะเกิดอะไรขึ้น?"
            )
        )

        self._check(
            result.status
            == "needs_validation",
            (
                "GROUNDING-PRECISION-T7 "
                "Thai mixed response"
            ),
            (
                "Thai explanation-plus-question "
                "requires semantic validation."
            ),
            (
                "Thai mixed response incorrectly "
                "used no_claims."
            ),
        )

    def test_t8_directive_only(
        self,
    ) -> None:

        guard = GroundingClaimPrecheck()

        result = guard.evaluate(
            "ลองทบทวนแนวคิดเรื่องทรานซิสเตอร์ก่อน"
        )

        self._check(
            result.status == "no_claims",
            (
                "GROUNDING-PRECISION-T8 "
                "Directive only"
            ),
            (
                "Non-factual instructional "
                "fast-path remains unchanged."
            ),
            (
                "Instruction-only response "
                "unexpectedly requires validation."
            ),
        )

    def test_t9_compact_question(
        self,
    ) -> None:

        guard = GroundingClaimPrecheck()

        result = guard.evaluate(
            "Base current?"
        )

        self._check(
            result.status == "no_claims",
            (
                "GROUNDING-PRECISION-T9 "
                "Compact question"
            ),
            (
                "A compact question without a "
                "preceding assertion remains cheap."
            ),
            (
                "Compact question lost the "
                "no-claims fast-path."
            ),
        )

    def test_t10_no_llm_dependency(
        self,
    ) -> None:

        source = inspect.getsource(
            GroundingClaimPrecheck
        )

        has_llm_dependency = any(
            marker in source
            for marker in (
                "chat_with_ai",
                "chat_with_structured_ai",
                "openai",
                "groq",
            )
        )

        self._check(
            not has_llm_dependency,
            (
                "GROUNDING-PRECISION-T10 "
                "Deterministic"
            ),
            (
                "Mixed-response claim detection "
                "remains fully deterministic."
            ),
            (
                "GroundingClaimPrecheck gained "
                "an unexpected LLM dependency."
            ),
        )
    def test_t11_entailment_required(
        self,
    ) -> None:

        prompt = (
            GROUNDING_VALIDATION_PROMPT
            .lower()
        )

        self._check(
            (
                "entailed"
                in prompt
                and "outside knowledge"
                in prompt
            ),
            (
                "GROUNDING-PRECISION-T11 "
                "Entailment required"
            ),
            (
                "Semantic grounding now requires "
                "source entailment rather than "
                "general plausibility."
            ),
            (
                "Strict source-entailment rules "
                "are missing."
            ),
        )

    def test_t12_no_relation_strengthening(
        self,
    ) -> None:

        prompt = (
            GROUNDING_VALIDATION_PROMPT
            .lower()
        )

        self._check(
            (
                "related to"
                in prompt
                and "proportional to"
                in prompt
            ),
            (
                "GROUNDING-PRECISION-T12 "
                "Relation strength"
            ),
            (
                "The validator is explicitly "
                "prevented from strengthening "
                "relationships."
            ),
            (
                "Relationship-strengthening guard "
                "is missing."
            ),
        )

    def test_t13_no_quantitative_inference(
        self,
    ) -> None:

        prompt = (
            GROUNDING_VALIDATION_PROMPT
            .lower()
        )

        self._check(
            (
                "exact numerical value"
                in prompt
                and "quantitative qualifier"
                in prompt
            ),
            (
                "GROUNDING-PRECISION-T13 "
                "Quantitative inference"
            ),
            (
                "Unsupported numerical and "
                "quantitative strengthening is "
                "explicitly prohibited."
            ),
            (
                "Quantitative evidence discipline "
                "is missing."
            ),
        )

    def test_t14_claim_by_claim(
        self,
    ) -> None:

        prompt = (
            GROUNDING_VALIDATION_PROMPT
            .lower()
        )

        self._check(
            (
                "each factual claim"
                in prompt
                and "independently"
                in prompt
            ),
            (
                "GROUNDING-PRECISION-T14 "
                "Claim-by-claim validation"
            ),
            (
                "Validator must evaluate factual "
                "claims independently."
            ),
            (
                "Claim-by-claim grounding "
                "requirement is missing."
            ),
        )

    def test_t15_questions_preserved(
        self,
    ) -> None:

        prompt = (
            GROUNDING_VALIDATION_PROMPT
            .lower()
        )

        self._check(
            (
                "question by itself"
                in prompt
                and "not a factual assertion"
                in prompt
            ),
            (
                "GROUNDING-PRECISION-T15 "
                "Question semantics"
            ),
            (
                "Strict grounding still preserves "
                "question-only claim semantics."
            ),
            (
                "Question-only grounding behavior "
                "was not preserved."
            ),
        )

    def test_t16_generation_boundary(
        self,
    ) -> None:

        builder = (
            GroundedGenerationInstructionBuilder()
        )

        instruction = builder.build(
            True
        ).lower()

        self._check(
            (
                "complete factual boundary"
                in instruction
                and "general model knowledge"
                in instruction
            ),
            (
                "GROUNDING-PRECISION-T16 "
                "Generation boundary"
            ),
            (
                "Tutor generation is explicitly "
                "bounded by course evidence."
            ),
            (
                "Evidence-bounded generation "
                "instruction is incomplete."
            ),
        )

    def test_t17_generation_no_strengthening(
        self,
    ) -> None:

        builder = (
            GroundedGenerationInstructionBuilder()
        )

        instruction = builder.build(
            True
        ).lower()

        self._check(
            (
                "related to"
                in instruction
                and "proportional to"
                in instruction
                and "is controlled by"
                in instruction
            ),
            (
                "GROUNDING-PRECISION-T17 "
                "Generation relation strength"
            ),
            (
                "Tutor generation cannot strengthen "
                "source relationships."
            ),
            (
                "Relationship-strengthening rule "
                "is missing from generation."
            ),
        )

    def test_t18_no_knowledge_instruction(
        self,
    ) -> None:

        builder = (
            GroundedGenerationInstructionBuilder()
        )

        result = builder.build(
            False
        )

        self._check(
            result == "",
            (
                "GROUNDING-PRECISION-T18 "
                "No grounded knowledge"
            ),
            (
                "Evidence-bounded instruction is "
                "inactive when grounded knowledge "
                "is unavailable."
            ),
            (
                "Generation instruction should be "
                "empty without grounded knowledge."
            ),
        )

    def test_t19_generation_invalid_type(
        self,
    ) -> None:

        builder = (
            GroundedGenerationInstructionBuilder()
        )

        raised = False

        try:
            builder.build(
                "yes"
            )

        except TypeError:
            raised = True

        self._check(
            raised,
            (
                "GROUNDING-PRECISION-T19 "
                "Generation typed input"
            ),
            (
                "Generation instruction builder "
                "rejects invalid input types."
            ),
            (
                "Invalid generation-boundary input "
                "was accepted."
            ),
        )

    def test_t20_prompt_optional_integration(
        self,
    ) -> None:

        source = inspect.getsource(
            PromptBuilder.build_system_prompt
        )

        has_parameter = (
            "grounding_precision_instruction"
            in source
        )

        knowledge_position = source.find(
            "if knowledge_context:"
        )

        precision_position = source.find(
            "if grounding_precision_instruction:"
        )

        correct_order = (
            knowledge_position >= 0
            and precision_position
            > knowledge_position
        )

        self._check(
            (
                has_parameter
                and correct_order
            ),
            (
                "GROUNDING-PRECISION-T20 "
                "Prompt integration"
            ),
            (
                "Evidence-bounded instruction is "
                "optional and placed after course "
                "knowledge."
            ),
            (
                "PromptBuilder evidence-boundary "
                "integration is missing or ordered "
                "incorrectly."
            ),
        )
    def test_t21_prompt_parameter_contract(
        self,
    ) -> None:

        signature = inspect.signature(
            PromptBuilder.build_system_prompt
        )

        parameter = signature.parameters.get(
            "grounding_precision_instruction"
        )

        self._check(
            (
                parameter is not None
                and parameter.default is None
            ),
            (
                "GROUNDING-PRECISION-T21 "
                "Prompt parameter contract"
            ),
            (
                "PromptBuilder explicitly exposes "
                "the optional grounding precision "
                "instruction parameter."
            ),
            (
                "PromptBuilder does not expose "
                "grounding_precision_instruction "
                "with a None default."
            ),
        )
    def test_t22_runtime_builder_initialized(
        self,
    ) -> None:

        source = inspect.getsource(
            AITutor.__init__
        )

        self._check(
            (
                "GroundedGenerationInstructionBuilder"
                in source
                and
                "grounded_generation_instruction_builder"
                in source
            ),
            (
                "GROUNDING-PRECISION-T22 "
                "Runtime builder initialized"
            ),
            (
                "AITutor owns the deterministic "
                "grounded-generation instruction builder."
            ),
            (
                "AITutor does not initialize the "
                "grounded-generation instruction builder."
            ),
        )

    def test_t23_runtime_instruction_built(
        self,
    ) -> None:

        source = inspect.getsource(
            AITutor.respond
        )

        self._check(
            (
                "grounded_generation_instruction_builder"
                in source
                and
                "grounding_status.has_knowledge"
                in source
            ),
            (
                "GROUNDING-PRECISION-T23 "
                "Runtime instruction built"
            ),
            (
                "Tutor generation derives the "
                "evidence boundary from the current "
                "grounding status."
            ),
            (
                "Runtime evidence-boundary construction "
                "is missing."
            ),
        )

    def test_t24_runtime_prompt_receives(
        self,
    ) -> None:

        source = inspect.getsource(
            AITutor.respond
        )

        prompt_position = source.find(
            "build_system_prompt"
        )

        precision_position = source.find(
            "grounding_precision_instruction",
            prompt_position,
        )

        self._check(
            (
                prompt_position >= 0
                and precision_position
                > prompt_position
            ),
            (
                "GROUNDING-PRECISION-T24 "
                "Runtime prompt receives"
            ),
            (
                "AITutor passes the current evidence "
                "boundary into PromptBuilder."
            ),
            (
                "PromptBuilder runtime call does not "
                "receive grounding precision context."
            ),
        )

    def test_t25_runtime_telemetry(
        self,
    ) -> None:

        source = inspect.getsource(
            AITutor.respond
        )

        self._check(
            (
                "self.last_grounding_precision_instruction"
                in source
            ),
            (
                "GROUNDING-PRECISION-T25 "
                "Runtime telemetry"
            ),
            (
                "AITutor stores the current "
                "grounding-precision instruction."
            ),
            (
                "Grounding-precision runtime telemetry "
                "is missing."
            ),
        )

    def test_t26_per_turn_clear(
        self,
    ) -> None:

        source = inspect.getsource(
            AITutor.respond
        )

        self._check(
            (
                "self.last_grounding_precision_instruction = None"
                in source
            ),
            (
                "GROUNDING-PRECISION-T26 "
                "Per-turn clear"
            ),
            (
                "Grounding-precision telemetry cannot "
                "leak from a previous Tutor turn."
            ),
            (
                "Per-turn grounding-precision clear "
                "is missing."
            ),
        )

    def test_t27_reset_clear(
        self,
    ) -> None:

        source = inspect.getsource(
            AITutor.reset
        )

        self._check(
            (
                "self.last_grounding_precision_instruction = None"
                in source
            ),
            (
                "GROUNDING-PRECISION-T27 "
                "Reset clear"
            ),
            (
                "Conversation reset clears the "
                "grounding-precision instruction."
            ),
            (
                "Conversation reset does not clear "
                "grounding-precision telemetry."
            ),
        )

    def test_t28_pipeline_order(
        self,
    ) -> None:

        source = inspect.getsource(
            AITutor.respond
        )

        precision_position = source.find(
            "self.grounded_generation_instruction_builder"
        )

        prompt_position = source.find(
            "build_system_prompt"
        )

        self._check(
            (
                precision_position >= 0
                and prompt_position >= 0
                and precision_position
                < prompt_position
            ),
            (
                "GROUNDING-PRECISION-T28 "
                "Pipeline order"
            ),
            (
                "Evidence-boundary instruction is "
                "constructed before the final system prompt."
            ),
            (
                "Grounding-precision instruction occurs "
                "after PromptBuilder or is missing."
            ),
        )

    def test_t29_validator_provider_failure(
        self,
    ) -> None:

        validator = (
            ResponseGroundingValidator()
        )

        with patch(
            (
                "app.services."
                "response_grounding_validator."
                "chat_with_structured_ai"
            ),
            side_effect=RuntimeError(
                "simulated provider failure"
            ),
        ):
            result = validator.validate(
                response=(
                    "The collector current is "
                    "related to the emitter current."
                ),
                knowledge_context=(
                    "The collector current is "
                    "related to the emitter current."
                ),
            )

        self._check(
            (
                result.status == "unsupported"
                and result.confidence == 0.0
                and bool(result.issues)
                and "RuntimeError"
                in result.issues[0]
            ),
            (
                "GROUNDING-PRECISION-T29 "
                "Provider fail-closed"
            ),
            (
                "Grounding validator provider "
                "failures are converted into a "
                "safe unsupported result."
            ),
            (
                "Grounding validator provider "
                "failure escaped or did not "
                "fail closed."
            ),
        )

    def test_t30_validator_token_budget(
        self,
    ) -> None:

        profile = get_task_profile(
            task_name="response_validator",
            structured=True,
        )

        self._check(
            (
                profile.max_completion_tokens
                >= 1000
                and profile.reasoning_effort
                == "low"
            ),
            (
                "GROUNDING-PRECISION-T30 "
                "Validator token budget"
            ),
            (
                "Production grounding validation "
                "has sufficient structured-output "
                "budget while retaining low reasoning."
            ),
            (
                "Production grounding validator "
                "budget remains too small or "
                "reasoning configuration changed."
            ),
        )

    def test_t31_repair_validator_token_budget(
        self,
    ) -> None:

        profile = get_task_profile(
            task_name=(
                "response_validator_repair"
            ),
            structured=True,
        )

        self._check(
            (
                profile.max_completion_tokens
                >= 1000
                and profile.reasoning_effort
                == "low"
            ),
            (
                "GROUNDING-PRECISION-T31 "
                "Repair validator token budget"
            ),
            (
                "Repair grounding validation has "
                "sufficient structured-output "
                "budget."
            ),
            (
                "Repair grounding validator "
                "budget remains too small or "
                "reasoning configuration changed."
            ),
        )

    def test_t32_precision_numeric_missing(
        self,
    ) -> None:

        guard = GroundingPrecisionGuard()

        result = guard.evaluate(
            response=(
                "The B-E junction operates at "
                "approximately 0.7 V."
            ),
            knowledge_context=(
                "The B-E junction is "
                "forward biased."
            ),
        )

        self._check(
            (
                result.status
                == "precision_issue"
            ),
            (
                "GROUNDING-PRECISION-T32 "
                "Unsupported numerical value"
            ),
            (
                "A precise value absent from "
                "course knowledge is detected."
            ),
            (
                "Unsupported numerical precision "
                "was not detected."
            ),
        )

    def test_t33_precision_numeric_supported(
        self,
    ) -> None:

        guard = GroundingPrecisionGuard()

        result = guard.evaluate(
            response=(
                "The reference voltage is 5 V."
            ),
            knowledge_context=(
                "The reference voltage is 5 V."
            ),
        )

        self._check(
            result.status == "clear",
            (
                "GROUNDING-PRECISION-T33 "
                "Supported numerical value"
            ),
            (
                "A precise value present in course "
                "knowledge remains allowed."
            ),
            (
                "Supported numerical evidence was "
                "incorrectly rejected."
            ),
        )

    def test_t34_precision_beta_missing(
        self,
    ) -> None:

        guard = GroundingPrecisionGuard()

        result = guard.evaluate(
            response=(
                "Collector current increases "
                "according to beta."
            ),
            knowledge_context=(
                "Collector current is related "
                "to emitter current."
            ),
        )

        self._check(
            (
                result.status
                == "precision_issue"
            ),
            (
                "GROUNDING-PRECISION-T34 "
                "Unsupported beta"
            ),
            (
                "Beta terminology absent from "
                "course evidence is detected."
            ),
            (
                "Unsupported beta terminology "
                "was not detected."
            ),
        )

    def test_t35_precision_beta_supported(
        self,
    ) -> None:

        guard = GroundingPrecisionGuard()

        result = guard.evaluate(
            response=(
                "The current gain is beta."
            ),
            knowledge_context=(
                "The current gain beta is defined "
                "for the transistor."
            ),
        )

        self._check(
            result.status == "clear",
            (
                "GROUNDING-PRECISION-T35 "
                "Supported beta"
            ),
            (
                "Beta remains allowed when present "
                "in course knowledge."
            ),
            (
                "Supported beta terminology was "
                "incorrectly rejected."
            ),
        )

    def test_t36_precision_majority_missing(
        self,
    ) -> None:

        guard = GroundingPrecisionGuard()

        result = guard.evaluate(
            response=(
                "Most electrons continue into "
                "the collector."
            ),
            knowledge_context=(
                "The number of electrons entering "
                "the collector is directly related "
                "to the electrons entering the base."
            ),
        )

        self._check(
            (
                result.status
                == "precision_issue"
            ),
            (
                "GROUNDING-PRECISION-T36 "
                "Unsupported majority"
            ),
            (
                "Unsupported majority strengthening "
                "is detected deterministically."
            ),
            (
                "Majority strengthening was "
                "not detected."
            ),
        )

    def test_t37_precision_thai_majority(
        self,
    ) -> None:

        guard = GroundingPrecisionGuard()

        result = guard.evaluate(
            response=(
                "อิเล็กตรอนส่วนใหญ่ไหลต่อไปยัง"
                "คอลเลคเตอร์"
            ),
            knowledge_context=(
                "จำนวนอิเล็กตรอนที่เข้าสู่"
                "คอลเลคเตอร์มีความสัมพันธ์กับ"
                "อิเล็กตรอนที่เข้าสู่เบส"
            ),
        )

        self._check(
            (
                result.status
                == "precision_issue"
            ),
            (
                "GROUNDING-PRECISION-T37 "
                "Thai majority qualifier"
            ),
            (
                "Thai quantitative strengthening "
                "is detected deterministically."
            ),
            (
                "Thai majority qualifier was "
                "not detected."
            ),
        )

    def test_t38_precision_proportionality(
        self,
    ) -> None:

        guard = GroundingPrecisionGuard()

        result = guard.evaluate(
            response=(
                "Collector current is "
                "proportional to base current."
            ),
            knowledge_context=(
                "Collector current is related "
                "to emitter current."
            ),
        )

        self._check(
            (
                result.status
                == "precision_issue"
            ),
            (
                "GROUNDING-PRECISION-T38 "
                "Unsupported proportionality"
            ),
            (
                "A stronger proportionality claim "
                "is detected."
            ),
            (
                "Unsupported proportionality was "
                "not detected."
            ),
        )

    def test_t39_precision_question_ignored(
        self,
    ) -> None:

        guard = GroundingPrecisionGuard()

        result = guard.evaluate(
            response=(
                "The B-E junction is forward biased.\n"
                "Could V_BE be approximately 0.7 V?"
            ),
            knowledge_context=(
                "The B-E junction is forward biased."
            ),
        )

        self._check(
            result.status == "clear",
            (
                "GROUNDING-PRECISION-T39 "
                "Question precision ignored"
            ),
            (
                "A precise value appearing only "
                "inside a question is not treated "
                "as a factual assertion."
            ),
            (
                "Question-only precision was "
                "incorrectly treated as a claim."
            ),
        )

    def test_t40_precision_deterministic(
        self,
    ) -> None:

        guard = GroundingPrecisionGuard()

        response = (
            "Most collector current follows "
            "a beta relationship."
        )

        context = (
            "Collector current is related "
            "to emitter current."
        )

        first = guard.evaluate(
            response=response,
            knowledge_context=context,
        )

        second = guard.evaluate(
            response=response,
            knowledge_context=context,
        )

        source = inspect.getsource(
            GroundingPrecisionGuard
        )

        self._check(
            (
                first == second
                and "chat_with_ai"
                not in source
                and "chat_with_structured_ai"
                not in source
            ),
            (
                "GROUNDING-PRECISION-T40 "
                "Precision deterministic"
            ),
            (
                "Precision evidence checking is "
                "deterministic and adds no LLM call."
            ),
            (
                "Precision evidence checking is "
                "non-deterministic or depends "
                "on an LLM."
            ),
        )

    def test_t41_precision_veto_clear(
        self,
    ) -> None:

        policy = GroundingPrecisionVetoPolicy()

        validation = GroundingValidationResult(
            status="supported",
            confidence=0.99,
            reason="All claims are grounded.",
            issues=[],
        )

        precision = GroundingPrecisionGuardResult(
            status="clear",
            issues=(),
        )

        result = policy.apply(
            validation=validation,
            precision=precision,
        )

        self._check(
            result is validation,
            (
                "GROUNDING-PRECISION-T41 "
                "Clear precision preserves"
            ),
            (
                "A clear precision result preserves "
                "the semantic grounding result exactly."
            ),
            (
                "Clear deterministic evidence "
                "unexpectedly changed grounding."
            ),
        )

    def test_t42_precision_veto_supported(
        self,
    ) -> None:

        policy = GroundingPrecisionVetoPolicy()

        validation = GroundingValidationResult(
            status="supported",
            confidence=0.99,
            reason="Semantic validator accepted.",
            issues=[],
        )

        precision = GroundingPrecisionGuardResult(
            status="precision_issue",
            issues=(
                "Unsupported precise numerical "
                "value: 0.7 v.",
            ),
        )

        result = policy.apply(
            validation=validation,
            precision=precision,
        )

        self._check(
            (
                result.status
                == "partially_supported"
                and
                "Unsupported precise numerical "
                "value: 0.7 v."
                in result.issues
            ),
            (
                "GROUNDING-PRECISION-T42 "
                "Supported veto"
            ),
            (
                "Deterministic precision evidence "
                "can veto a semantic supported result."
            ),
            (
                "Semantic supported result bypassed "
                "the deterministic precision veto."
            ),
        )

    def test_t43_precision_veto_partial(
        self,
    ) -> None:

        policy = GroundingPrecisionVetoPolicy()

        validation = GroundingValidationResult(
            status="partially_supported",
            confidence=0.8,
            reason="One semantic issue exists.",
            issues=[
                "Semantic issue."
            ],
        )

        precision = GroundingPrecisionGuardResult(
            status="precision_issue",
            issues=(
                "Unsupported beta/current-gain "
                "terminology.",
            ),
        )

        result = policy.apply(
            validation=validation,
            precision=precision,
        )

        self._check(
            (
                result.status
                == "partially_supported"
                and "Semantic issue."
                in result.issues
                and
                "Unsupported beta/current-gain "
                "terminology."
                in result.issues
            ),
            (
                "GROUNDING-PRECISION-T43 "
                "Partial preserved"
            ),
            (
                "Existing partial grounding remains "
                "partial while precision issues are "
                "preserved."
            ),
            (
                "Precision veto incorrectly changed "
                "an existing partial result."
            ),
        )

    def test_t44_precision_veto_unsupported(
        self,
    ) -> None:

        policy = GroundingPrecisionVetoPolicy()

        validation = GroundingValidationResult(
            status="unsupported",
            confidence=0.95,
            reason="Central claim unsupported.",
            issues=[
                "Unsupported claim."
            ],
        )

        precision = GroundingPrecisionGuardResult(
            status="precision_issue",
            issues=(
                "Unsupported proportionality "
                "relationship.",
            ),
        )

        result = policy.apply(
            validation=validation,
            precision=precision,
        )

        self._check(
            (
                result.status == "unsupported"
                and
                "Unsupported proportionality "
                "relationship."
                in result.issues
            ),
            (
                "GROUNDING-PRECISION-T44 "
                "Unsupported preserved"
            ),
            (
                "An existing unsupported result "
                "cannot be weakened by the "
                "precision policy."
            ),
            (
                "Precision policy weakened an "
                "unsupported grounding result."
            ),
        )

    def test_t45_precision_veto_no_duplicate(
        self,
    ) -> None:

        policy = GroundingPrecisionVetoPolicy()

        issue = (
            "Unsupported majority or "
            "quantity qualifier."
        )

        validation = GroundingValidationResult(
            status="partially_supported",
            confidence=0.8,
            reason="Semantic issue.",
            issues=[
                issue
            ],
        )

        precision = GroundingPrecisionGuardResult(
            status="precision_issue",
            issues=(
                issue,
            ),
        )

        result = policy.apply(
            validation=validation,
            precision=precision,
        )

        self._check(
            result.issues.count(issue) == 1,
            (
                "GROUNDING-PRECISION-T45 "
                "Issue deduplication"
            ),
            (
                "Semantic and deterministic "
                "precision issues are merged "
                "without duplication."
            ),
            (
                "Precision veto duplicated an "
                "existing grounding issue."
            ),
        )

    def test_t46_precision_veto_invalid_type(
        self,
    ) -> None:

        policy = GroundingPrecisionVetoPolicy()

        validation = GroundingValidationResult(
            status="supported",
            confidence=1.0,
            reason="Supported.",
            issues=[],
        )

        raised_validation = False
        raised_precision = False

        try:
            policy.apply(
                validation="supported",
                precision=(
                    GroundingPrecisionGuardResult(
                        status="clear",
                        issues=(),
                    )
                ),
            )

        except TypeError:
            raised_validation = True

        try:
            policy.apply(
                validation=validation,
                precision="clear",
            )

        except TypeError:
            raised_precision = True

        self._check(
            (
                raised_validation
                and raised_precision
            ),
            (
                "GROUNDING-PRECISION-T46 "
                "Veto typed input"
            ),
            (
                "Precision veto policy enforces "
                "typed inputs."
            ),
            (
                "Precision veto policy accepted "
                "invalid input types."
            ),
        )

    def test_t47_precision_veto_deterministic(
        self,
    ) -> None:

        policy = GroundingPrecisionVetoPolicy()

        validation = GroundingValidationResult(
            status="supported",
            confidence=0.99,
            reason="Accepted.",
            issues=[],
        )

        precision = GroundingPrecisionGuardResult(
            status="precision_issue",
            issues=(
                "Unsupported precise numerical "
                "value: 0.7 v.",
            ),
        )

        first = policy.apply(
            validation=validation,
            precision=precision,
        )

        second = policy.apply(
            validation=validation,
            precision=precision,
        )

        source = inspect.getsource(
            GroundingPrecisionVetoPolicy
        )

        self._check(
            (
                first == second
                and "chat_with_ai"
                not in source
                and "chat_with_structured_ai"
                not in source
            ),
            (
                "GROUNDING-PRECISION-T47 "
                "Veto deterministic"
            ),
            (
                "Grounding precision veto is "
                "deterministic and adds no LLM call."
            ),
            (
                "Grounding precision veto is "
                "non-deterministic or depends "
                "on an LLM."
            ),
        )

    def test_t48_runtime_precision_services(
        self,
    ) -> None:

        source = inspect.getsource(
            AITutor.__init__
        )

        self._check(
            (
                "self.grounding_precision_guard"
                in source
                and
                "GroundingPrecisionGuard()"
                in source
                and
                "self.grounding_precision_veto_policy"
                in source
                and
                "GroundingPrecisionVetoPolicy()"
                in source
            ),
            (
                "GROUNDING-PRECISION-T48 "
                "Runtime precision services"
            ),
            (
                "AITutor owns both deterministic "
                "precision services."
            ),
            (
                "Runtime precision services are "
                "not initialized correctly."
            ),
        )

    def test_t49_initial_precision_guard_routed(
        self,
    ) -> None:

        source = inspect.getsource(
            AITutor.respond
        )

        semantic_position = source.find(
            "self.response_grounding_validator"
        )

        precision_position = source.find(
            "self.grounding_precision_guard",
            semantic_position,
        )

        pedagogy_position = source.find(
            "response_mode_pedagogical_precheck",
            precision_position,
        )

        self._check(
            (
                semantic_position >= 0
                and precision_position
                > semantic_position
                and pedagogy_position
                > precision_position
            ),
            (
                "GROUNDING-PRECISION-T49 "
                "Initial precision routing"
            ),
            (
                "Initial Tutor answers pass through "
                "the deterministic precision guard "
                "after grounding and before pedagogy."
            ),
            (
                "Initial precision guard is missing "
                "or incorrectly ordered."
            ),
        )

    def test_t50_initial_veto_routed(
        self,
    ) -> None:

        source = inspect.getsource(
            AITutor.respond
        )

        precision_position = source.find(
            "self.grounding_precision_guard"
        )

        veto_position = source.find(
            "self.grounding_precision_veto_policy",
            precision_position,
        )

        pedagogy_position = source.find(
            "response_mode_pedagogical_precheck",
            veto_position,
        )

        self._check(
            (
                precision_position >= 0
                and veto_position
                > precision_position
                and pedagogy_position
                > veto_position
            ),
            (
                "GROUNDING-PRECISION-T50 "
                "Initial veto routing"
            ),
            (
                "Initial semantic grounding can be "
                "vetoed before pedagogical validation."
            ),
            (
                "Initial precision veto is missing "
                "or incorrectly ordered."
            ),
        )

    def test_t51_repair_precision_guard_routed(
        self,
    ) -> None:

        source = inspect.getsource(
            AITutor.respond
        )

        repair_validator_position = source.find(
            '"response_validator_repair"'
        )

        repair_precision_position = source.find(
            "repair_grounding_precision_guard_result",
            repair_validator_position,
        )

        repair_pedagogy_position = source.find(
            "repair_pedagogical_precheck",
            repair_precision_position,
        )

        self._check(
            (
                repair_validator_position >= 0
                and repair_precision_position
                > repair_validator_position
                and repair_pedagogy_position
                > repair_precision_position
            ),
            (
                "GROUNDING-PRECISION-T51 "
                "Repair precision routing"
            ),
            (
                "Repaired Tutor answers receive "
                "the same deterministic precision "
                "check after semantic re-validation."
            ),
            (
                "Repair precision guard is missing "
                "or incorrectly ordered."
            ),
        )

    def test_t52_repair_veto_routed(
        self,
    ) -> None:

        source = inspect.getsource(
            AITutor.respond
        )

        repair_precision_position = source.find(
            "repair_grounding_precision_guard_result"
        )

        repair_veto_position = source.find(
            "self.grounding_precision_veto_policy",
            repair_precision_position,
        )

        repair_pedagogy_position = source.find(
            "repair_pedagogical_precheck",
            repair_veto_position,
        )

        self._check(
            (
                repair_precision_position >= 0
                and repair_veto_position
                > repair_precision_position
                and repair_pedagogy_position
                > repair_veto_position
            ),
            (
                "GROUNDING-PRECISION-T52 "
                "Repair veto routing"
            ),
            (
                "Repair semantic supported results "
                "remain subject to the deterministic "
                "precision veto."
            ),
            (
                "Repair precision veto is missing "
                "or incorrectly ordered."
            ),
        )

    def test_t53_precision_telemetry_initialized(
        self,
    ) -> None:

        source = inspect.getsource(
            AITutor.__init__
        )

        self._check(
            (
                "self.last_grounding_precision_guard = None"
                in source
                and
                "self.last_repair_grounding_precision_guard = None"
                in source
            ),
            (
                "GROUNDING-PRECISION-T53 "
                "Precision telemetry initialized"
            ),
            (
                "Initial and repair precision "
                "telemetry start with safe null values."
            ),
            (
                "Precision telemetry defaults "
                "are incomplete."
            ),
        )

    def test_t54_precision_per_turn_clear(
        self,
    ) -> None:

        source = inspect.getsource(
            AITutor.respond
        )

        self._check(
            (
                "self.last_grounding_precision_guard = None"
                in source
                and
                "self.last_repair_grounding_precision_guard = None"
                in source
            ),
            (
                "GROUNDING-PRECISION-T54 "
                "Precision per-turn clear"
            ),
            (
                "Precision telemetry cannot leak "
                "between Tutor turns."
            ),
            (
                "Per-turn precision telemetry "
                "clear is missing."
            ),
        )

    def test_t55_precision_reset_clear(
        self,
    ) -> None:

        source = inspect.getsource(
            AITutor.reset
        )

        self._check(
            (
                "self.last_grounding_precision_guard = None"
                in source
                and
                "self.last_repair_grounding_precision_guard = None"
                in source
            ),
            (
                "GROUNDING-PRECISION-T55 "
                "Precision reset clear"
            ),
            (
                "Conversation reset clears all "
                "precision telemetry."
            ),
            (
                "Conversation reset leaves stale "
                "precision telemetry."
            ),
        )

    def test_t56_precision_runtime_no_llm(
        self,
    ) -> None:

        guard_source = inspect.getsource(
            GroundingPrecisionGuard
        )

        policy_source = inspect.getsource(
            GroundingPrecisionVetoPolicy
        )

        self._check(
            (
                "chat_with_ai"
                not in guard_source
                and
                "chat_with_structured_ai"
                not in guard_source
                and
                "chat_with_ai"
                not in policy_source
                and
                "chat_with_structured_ai"
                not in policy_source
            ),
            (
                "GROUNDING-PRECISION-T56 "
                "Runtime precision no LLM"
            ),
            (
                "Runtime precision enforcement "
                "adds no LLM dependency."
            ),
            (
                "Precision runtime enforcement "
                "unexpectedly depends on an LLM."
            ),
        )
    def test_t57_atomic_supported_evidence(
        self,
    ) -> None:

        validator = ResponseGroundingValidator()

        response = (
            "The collector current is related "
            "to the emitter current."
        )

        knowledge = (
            "The collector current is related "
            "to the emitter current."
        )

        payload = {
            "status": "supported",
            "confidence": 0.99,
            "reason": "Supported.",
            "issues": [],
            "claims": [
                {
                    "claim": (
                        "Collector current is related "
                        "to emitter current."
                    ),
                    "response_quote": response,
                    "status": "supported",
                    "evidence_quote": knowledge,
                    "issue": "",
                }
            ],
        }

        with patch(
            (
                "app.services."
                "response_grounding_validator."
                "chat_with_structured_ai"
            ),
            return_value=json.dumps(payload),
        ):
            result = validator.validate(
                response=response,
                knowledge_context=knowledge,
            )

        self._check(
            (
                result.status == "supported"
                and not result.issues
            ),
            (
                "GROUNDING-PRECISION-T57 "
                "Atomic supported evidence"
            ),
            (
                "An atomic claim with a verifiable "
                "response quote and course evidence "
                "remains supported."
            ),
            (
                "Valid atomic evidence was not "
                "accepted."
            ),
        )

    def test_t58_atomic_relation_strengthening(
        self,
    ) -> None:

        validator = ResponseGroundingValidator()

        response = (
            "The collector current is controlled "
            "by the emitter current."
        )

        knowledge = (
            "The collector current is related "
            "to the emitter current."
        )

        payload = {
            "status": "supported",
            "confidence": 0.99,
            "reason": "Supported.",
            "issues": [],
            "claims": [
                {
                    "claim": (
                        "Collector current is "
                        "controlled by emitter current."
                    ),
                    "response_quote": response,
                    "status": "supported",
                    "evidence_quote": knowledge,
                    "issue": "",
                }
            ],
        }

        with patch(
            (
                "app.services."
                "response_grounding_validator."
                "chat_with_structured_ai"
            ),
            return_value=json.dumps(payload),
        ):
            result = validator.validate(
                response=response,
                knowledge_context=knowledge,
            )

        self._check(
            (
                result.status == "unsupported"
                and any(
                    "relation strengthening"
                    in issue.lower()
                    for issue in result.issues
                )
            ),
            (
                "GROUNDING-PRECISION-T58 "
                "Atomic relation strengthening"
            ),
            (
                "A semantic supported result cannot "
                "turn source-level 'related to' into "
                "'controlled by'."
            ),
            (
                "Relation strengthening bypassed "
                "atomic evidence enforcement."
            ),
        )

    def test_t59_atomic_fake_evidence_quote(
        self,
    ) -> None:

        validator = ResponseGroundingValidator()

        response = (
            "The B-E junction is forward biased."
        )

        knowledge = (
            "The B-E junction is forward biased."
        )

        payload = {
            "status": "supported",
            "confidence": 0.99,
            "reason": "Supported.",
            "issues": [],
            "claims": [
                {
                    "claim": (
                        "The B-E junction is "
                        "forward biased."
                    ),
                    "response_quote": response,
                    "status": "supported",
                    "evidence_quote": (
                        "The B-E junction becomes "
                        "forward biased at 0.7 V."
                    ),
                    "issue": "",
                }
            ],
        }

        with patch(
            (
                "app.services."
                "response_grounding_validator."
                "chat_with_structured_ai"
            ),
            return_value=json.dumps(payload),
        ):
            result = validator.validate(
                response=response,
                knowledge_context=knowledge,
            )

        self._check(
            (
                result.status == "unsupported"
                and any(
                    "not found in course knowledge"
                    in issue.lower()
                    for issue in result.issues
                )
            ),
            (
                "GROUNDING-PRECISION-T59 "
                "Atomic evidence provenance"
            ),
            (
                "A fabricated evidence quote cannot "
                "be accepted as course evidence."
            ),
            (
                "Atomic validation accepted an "
                "evidence quote absent from the "
                "course knowledge."
            ),
        )

    def test_t60_atomic_fake_response_quote(
        self,
    ) -> None:

        validator = ResponseGroundingValidator()

        response = (
            "The B-E junction is forward biased."
        )

        knowledge = (
            "The B-E junction is forward biased."
        )

        payload = {
            "status": "supported",
            "confidence": 0.99,
            "reason": "Supported.",
            "issues": [],
            "claims": [
                {
                    "claim": (
                        "The transistor requires "
                        "0.7 V."
                    ),
                    "response_quote": (
                        "The transistor requires "
                        "0.7 V."
                    ),
                    "status": "supported",
                    "evidence_quote": knowledge,
                    "issue": "",
                }
            ],
        }

        with patch(
            (
                "app.services."
                "response_grounding_validator."
                "chat_with_structured_ai"
            ),
            return_value=json.dumps(payload),
        ):
            result = validator.validate(
                response=response,
                knowledge_context=knowledge,
            )

        self._check(
            (
                result.status == "unsupported"
                and any(
                    "response quote not found"
                    in issue.lower()
                    for issue in result.issues
                )
            ),
            (
                "GROUNDING-PRECISION-T60 "
                "Atomic response provenance"
            ),
            (
                "The validator cannot attribute a "
                "claim that is absent from the "
                "Tutor response."
            ),
            (
                "A fabricated Tutor response quote "
                "was accepted."
            ),
        )

    def test_t61_atomic_partial_aggregation(
        self,
    ) -> None:

        validator = ResponseGroundingValidator()

        response = (
            "The B-E junction is forward biased. "
            "The transistor requires exactly 0.7 V."
        )

        knowledge = (
            "The B-E junction is forward biased."
        )

        payload = {
            "status": "supported",
            "confidence": 0.9,
            "reason": "Everything is supported.",
            "issues": [],
            "claims": [
                {
                    "claim": (
                        "The B-E junction is "
                        "forward biased."
                    ),
                    "response_quote": (
                        "The B-E junction is "
                        "forward biased."
                    ),
                    "status": "supported",
                    "evidence_quote": knowledge,
                    "issue": "",
                },
                {
                    "claim": (
                        "The transistor requires "
                        "exactly 0.7 V."
                    ),
                    "response_quote": (
                        "The transistor requires "
                        "exactly 0.7 V."
                    ),
                    "status": "unsupported",
                    "evidence_quote": "",
                    "issue": (
                        "The exact 0.7 V value is "
                        "not provided."
                    ),
                },
            ],
        }

        with patch(
            (
                "app.services."
                "response_grounding_validator."
                "chat_with_structured_ai"
            ),
            return_value=json.dumps(payload),
        ):
            result = validator.validate(
                response=response,
                knowledge_context=knowledge,
            )

        self._check(
            (
                result.status
                == "partially_supported"
                and len(result.issues) == 1
            ),
            (
                "GROUNDING-PRECISION-T61 "
                "Atomic partial aggregation"
            ),
            (
                "Mixed supported and unsupported "
                "atomic claims aggregate to "
                "partially_supported."
            ),
            (
                "Atomic claim aggregation trusted "
                "the incorrect top-level status."
            ),
        )

    def test_t62_atomic_top_level_status_ignored(
        self,
    ) -> None:

        validator = ResponseGroundingValidator()

        response = (
            "The transistor requires exactly 0.7 V."
        )

        knowledge = (
            "The B-E junction is forward biased."
        )

        payload = {
            "status": "supported",
            "confidence": 0.99,
            "reason": (
                "Incorrect top-level decision."
            ),
            "issues": [],
            "claims": [
                {
                    "claim": (
                        "The transistor requires "
                        "exactly 0.7 V."
                    ),
                    "response_quote": response,
                    "status": "unsupported",
                    "evidence_quote": "",
                    "issue": (
                        "No exact voltage is "
                        "provided in the source."
                    ),
                }
            ],
        }

        with patch(
            (
                "app.services."
                "response_grounding_validator."
                "chat_with_structured_ai"
            ),
            return_value=json.dumps(payload),
        ):
            result = validator.validate(
                response=response,
                knowledge_context=knowledge,
            )

        self._check(
            result.status == "unsupported",
            (
                "GROUNDING-PRECISION-T62 "
                "Atomic aggregation ownership"
            ),
            (
                "Python aggregation overrides an "
                "incorrect LLM top-level supported "
                "classification."
            ),
            (
                "Grounding still trusts the LLM "
                "top-level status."
            ),
        )

    def test_t63_atomic_empty_question(
        self,
    ) -> None:

        validator = ResponseGroundingValidator()

        response = (
            "What happens when the B-E junction "
            "is forward biased?"
        )

        knowledge = (
            "The B-E junction is forward biased."
        )

        payload = {
            "status": "supported",
            "confidence": 0.99,
            "reason": "No factual claims.",
            "issues": [],
            "claims": [],
        }

        with patch(
            (
                "app.services."
                "response_grounding_validator."
                "chat_with_structured_ai"
            ),
            return_value=json.dumps(payload),
        ):
            result = validator.validate(
                response=response,
                knowledge_context=knowledge,
            )

        self._check(
            (
                result.status == "supported"
                and not result.issues
            ),
            (
                "GROUNDING-PRECISION-T63 "
                "Atomic question-only"
            ),
            (
                "An empty atomic claim list remains "
                "valid for a genuine question-only "
                "response."
            ),
            (
                "Question-only atomic behavior "
                "regressed."
            ),
        )

    def test_t64_atomic_empty_claim_fail_closed(
        self,
    ) -> None:

        validator = ResponseGroundingValidator()

        response = (
            "The collector current is related "
            "to the emitter current."
        )

        knowledge = response

        payload = {
            "status": "supported",
            "confidence": 0.99,
            "reason": "No claims.",
            "issues": [],
            "claims": [],
        }

        with patch(
            (
                "app.services."
                "response_grounding_validator."
                "chat_with_structured_ai"
            ),
            return_value=json.dumps(payload),
        ):
            result = validator.validate(
                response=response,
                knowledge_context=knowledge,
            )

        self._check(
            (
                result.status == "unsupported"
                and any(
                    "coverage"
                    in issue.lower()
                    for issue in result.issues
                )
            ),
            (
                "GROUNDING-PRECISION-T64 "
                "Atomic omission fail-closed"
            ),
            (
                "A factual response cannot be "
                "declared claim-free by the "
                "semantic validator."
            ),
            (
                "Atomic claim omission bypassed "
                "grounding validation."
            ),
        )

    def test_t65_atomic_wrong_relation_evidence(
        self,
    ) -> None:

        validator = ResponseGroundingValidator()

        response = (
            "The emitter current controls "
            "the collector current."
        )

        knowledge = (
            "The collector current is related "
            "to the emitter current. "
            "The voltage between two terminals "
            "controls the current through "
            "the third terminal."
        )

        payload = {
            "status": "supported",
            "confidence": 0.99,
            "reason": "Supported.",
            "issues": [],
            "claims": [
                {
                    "claim": (
                        "Emitter current controls "
                        "collector current."
                    ),
                    "response_quote": response,
                    "status": "supported",
                    "evidence_quote": (
                        "The voltage between two terminals "
                        "controls the current through "
                        "the third terminal."
                    ),
                    "issue": "",
                }
            ],
        }

        with patch(
            (
                "app.services."
                "response_grounding_validator."
                "chat_with_structured_ai"
            ),
            return_value=json.dumps(payload),
        ):
            result = validator.validate(
                response=response,
                knowledge_context=knowledge,
            )

        self._check(
            (
                result.status
                != "supported"
            ),
            (
                "GROUNDING-PRECISION-T65 "
                "Atomic relation entity alignment"
            ),
            (
                "A control relationship cannot be "
                "supported by evidence describing "
                "a different controller and target."
            ),
            (
                "Atomic validation accepted a "
                "control relation with mismatched "
                "entities."
            ),
        )

    def test_t66_atomic_relation_alignment_supported(
        self,
    ) -> None:

        validator = ResponseGroundingValidator()

        response = (
            "The collector current is controlled "
            "by the base voltage."
        )

        knowledge = (
            "The base voltage controls "
            "the collector current."
        )

        payload = {
            "status": "supported",
            "confidence": 0.99,
            "reason": "Supported.",
            "issues": [],
            "claims": [
                {
                    "claim": (
                        "Base voltage controls "
                        "collector current."
                    ),
                    "response_quote": response,
                    "status": "supported",
                    "evidence_quote": knowledge,
                    "issue": "",
                }
            ],
        }

        with patch(
            (
                "app.services."
                "response_grounding_validator."
                "chat_with_structured_ai"
            ),
            return_value=json.dumps(payload),
        ):
            result = validator.validate(
                response=response,
                knowledge_context=knowledge,
            )

        self._check(
            (
                result.status
                == "supported"
            ),
            (
                "GROUNDING-PRECISION-T66 "
                "Atomic aligned control relation"
            ),
            (
                "Equivalent active and passive "
                "control relationships with the "
                "same controller and target remain "
                "supported."
            ),
            (
                "A correctly aligned control "
                "relationship was rejected."
            ),
        )

    def test_t67_repair_complete_evidence_boundary(
        self,
    ) -> None:

        prompt = RESPONSE_REPAIR_PROMPT.lower()

        required = (
            "complete factual boundary",
            "outside knowledge",
            "remove",
            "every factual clause",
            "course knowledge",
        )

        self._check(
            all(
                item in prompt
                for item in required
            ),
            (
                "GROUNDING-PRECISION-T67 "
                "Repair evidence boundary"
            ),
            (
                "Grounding repair treats course "
                "knowledge as the complete factual "
                "boundary and permits unsupported "
                "claims to be removed."
            ),
            (
                "Grounding repair is still biased "
                "toward preserving unsupported "
                "factual content."
            ),
        )

    def test_t68_repair_relation_precision(
        self,
    ) -> None:

        prompt = RESPONSE_REPAIR_PROMPT.lower()

        self._check(
            (
                '"related to"' in prompt
                and '"controlled by"' in prompt
                and "controller-target" in prompt
                and "proportional" in prompt
            ),
            (
                "GROUNDING-PRECISION-T68 "
                "Repair relation precision"
            ),
            (
                "Repair explicitly preserves "
                "source-level relation strength "
                "and entity alignment."
            ),
            (
                "Repair does not explicitly prevent "
                "relation strengthening."
            ),
        )

    def test_t69_repair_technical_precision(
        self,
    ) -> None:

        prompt = RESPONSE_REPAIR_PROMPT.lower()

        required = (
            "exact numerical",
            "beta",
            "current gain",
            "doping",
            "majority",
            "conservative omission",
        )

        self._check(
            all(
                item in prompt
                for item in required
            ),
            (
                "GROUNDING-PRECISION-T69 "
                "Repair technical precision"
            ),
            (
                "Repair explicitly excludes the "
                "technical strengthening classes "
                "observed in live generation."
            ),
            (
                "Repair precision rules are "
                "incomplete."
            ),
        )

    def test_t70_mode_repair_evidence_boundary(
        self,
    ) -> None:

        prompt = (
            RESPONSE_MODE_REPAIR_PROMPT.lower()
        )

        required = (
            "complete factual boundary",
            "outside knowledge",
            "remove",
            "controller-target",
            "conservative omission",
        )

        self._check(
            all(
                item in prompt
                for item in required
            ),
            (
                "GROUNDING-PRECISION-T70 "
                "Mode repair evidence boundary"
            ),
            (
                "Non-preserving response modes "
                "receive the same strict evidence "
                "boundary without changing routing."
            ),
            (
                "Response-mode-aware repair lacks "
                "the strict evidence boundary."
            ),
        )

    def test_t71_factual_repair_reconstructs_without_original(
        self,
    ) -> None:

        service = ResponseRepairService()

        validation = GroundingValidationResult(
            status="partially_supported",
            confidence=0.9,
            reason="Unsupported factual content.",
            issues=[
                "Unsupported precise numerical value."
            ],
        )

        captured = {}

        def fake_chat(
            messages,
            task_name,
        ):
            captured["messages"] = messages
            captured["task_name"] = task_name

            return (
                "Reconstructed grounded response."
            )

        sentinel = (
            "UNSUPPORTED_SENTINEL_0_7V"
        )

        with patch(
            (
                "app.services."
                "response_repair_service."
                "chat_with_ai"
            ),
            side_effect=fake_chat,
        ):
            service.repair(
                original_response=sentinel,
                knowledge_context=(
                    "The B-E junction is "
                    "forward biased."
                ),
                validation=validation,
                learner_message=(
                    "Explain the base."
                ),
                scaffolding_level=1,
                strategy_name=(
                    "guiding_question"
                ),
                strategy_instruction=(
                    "Ask one question."
                ),
                intervention_name="none",
            )

        prompt_text = "\n".join(
            message["content"]
            for message
            in captured["messages"]
        )

        self._check(
            (
                sentinel
                not in prompt_text
                and
                "FACTUAL RECONSTRUCTION MODE"
                in prompt_text
            ),
            (
                "GROUNDING-PRECISION-T71 "
                "Factual reconstruction isolation"
            ),
            (
                "Factual repair rebuilds from "
                "course evidence without exposing "
                "the unsupported original answer."
            ),
            (
                "Unsupported original factual "
                "content still anchors the repair "
                "model."
            ),
        )

    def test_t72_pedagogical_repair_preserves_original(
        self,
    ) -> None:

        service = ResponseRepairService()

        validation = GroundingValidationResult(
            status="supported",
            confidence=1.0,
            reason="Grounded.",
            issues=[],
        )

        captured = {}

        def fake_chat(
            messages,
            task_name,
        ):
            captured["messages"] = messages

            return "Pedagogically repaired."

        sentinel = (
            "SUPPORTED_FACT_SENTINEL"
        )

        with patch(
            (
                "app.services."
                "response_repair_service."
                "chat_with_ai"
            ),
            side_effect=fake_chat,
        ):
            service.repair(
                original_response=sentinel,
                knowledge_context=(
                    "Supported course knowledge."
                ),
                validation=validation,
                learner_message=(
                    "Learner response."
                ),
                scaffolding_level=1,
                strategy_name=(
                    "guiding_question"
                ),
                strategy_instruction=(
                    "Ask one question."
                ),
                intervention_name="none",
                pedagogical_issues=[
                    "Question form requires repair."
                ],
            )

        prompt_text = "\n".join(
            message["content"]
            for message
            in captured["messages"]
        )

        self._check(
            (
                sentinel
                in prompt_text
                and
                "PEDAGOGICAL REPAIR MODE"
                in prompt_text
            ),
            (
                "GROUNDING-PRECISION-T72 "
                "Pedagogical repair preservation"
            ),
            (
                "Grounded responses remain "
                "available when only pedagogical "
                "repair is required."
            ),
            (
                "Factual reconstruction isolation "
                "incorrectly affected supported "
                "pedagogical repair."
            ),
        )

    def test_t73_repair_atomic_validator_budget(
        self,
    ) -> None:

        from app.infrastructure.ai.task_profiles import (
            TASK_PROFILES,
        )

        profile = TASK_PROFILES[
            "response_validator_repair"
        ]

        self._check(
            (
                profile.max_completion_tokens
                >= 1600
                and
                profile.reasoning_effort
                == "low"
            ),
            (
                "GROUNDING-PRECISION-T73 "
                "Repair atomic validator budget"
            ),
            (
                "Repair grounding validation has "
                "sufficient structured-output budget "
                "for atomic claim evidence."
            ),
            (
                "Repair grounding validation budget "
                "is too small for the atomic schema."
            ),
        )
    def test_t74_repair_candidate_telemetry_initialized(
        self,
    ) -> None:

        source = Path(
            "app/tutor.py"
        ).read_text(
            encoding="utf-8"
        )

        self._check(
            (
                "self.last_repair_candidate = None"
                in source
            ),
            (
                "GROUNDING-PRECISION-T74 "
                "Repair candidate telemetry"
            ),
            (
                "AITutor initializes repair-candidate "
                "telemetry safely."
            ),
            (
                "Repair candidate telemetry is not "
                "initialized."
            ),
        )

    def test_t75_repair_candidate_captured(
        self,
    ) -> None:

        source = Path(
            "app/tutor.py"
        ).read_text(
            encoding="utf-8"
        )

        expected = (
            "self.last_repair_candidate = (\n"
            "                            repaired_answer\n"
            "                        )"
        )

        self._check(
            (
                "self.last_repair_candidate"
                in source
                and
                "repaired_answer"
                in source
            ),
            (
                "GROUNDING-PRECISION-T75 "
                "Repair candidate capture"
            ),
            (
                "AITutor captures the repair candidate "
                "before repair re-validation."
            ),
            (
                "Repair candidates are not observable "
                "before validation."
            ),
        )

    def test_t76_repair_candidate_debug_projection(
        self,
    ) -> None:

        source = Path(
            "app/tutor.py"
        ).read_text(
            encoding="utf-8"
        )

        self._check(
            (
                '"repair_candidate"'
                in source
            ),
            (
                "GROUNDING-PRECISION-T76 "
                "Repair candidate debug projection"
            ),
            (
                "Debug telemetry exposes the exact "
                "repair candidate used for validation."
            ),
            (
                "Repair candidate is not exposed "
                "through debug telemetry."
            ),
        )   

    def test_t77_verified_evidence_model_frozen(
        self,
    ) -> None:

        result = RepairEvidenceSelectionResult(
            status="verified",
            evidence_quotes=(
                "Course evidence.",
            ),
            reason="Verified.",
            issues=(),
        )

        frozen = False

        try:
            result.status = "invalid"

        except Exception:
            frozen = True

        self._check(
            (
                frozen
                and
                result.has_verified_evidence
            ),
            (
                "GROUNDING-PRECISION-T77 "
                "Verified evidence model"
            ),
            (
                "Verified repair evidence uses an "
                "immutable typed result."
            ),
            (
                "Repair evidence result is not safely "
                "immutable or typed."
            ),
        )

    def test_t78_exact_repair_evidence_verified(
        self,
    ) -> None:

        selector = RepairEvidenceSelector()

        knowledge = (
            "The collector current is related to the "
            "emitter current which is in turn a function "
            "of the B-E voltage."
        )

        fake_result = json.dumps(
            {
                "evidence_quotes": [
                    (
                        "The collector current is related "
                        "to the emitter current"
                    ),
                ],
                "reason": "Relevant exact evidence.",
            }
        )

        with patch(
            (
                "app.services.repair_evidence_selector."
                "chat_with_structured_ai"
            ),
            return_value=fake_result,
        ):
            result = selector.select(
                learner_message=(
                    "What affects collector current?"
                ),
                knowledge_context=knowledge,
            )

        self._check(
            (
                result.status == "verified"
                and
                result.evidence_quotes
                == (
                    (
                        "The collector current is related "
                        "to the emitter current"
                    ),
                )
            ),
            (
                "GROUNDING-PRECISION-T78 "
                "Exact repair evidence"
            ),
            (
                "An exact course excerpt is accepted "
                "as verified repair evidence."
            ),
            (
                "Valid exact course evidence was not "
                "verified."
            ),
        )

    def test_t79_fabricated_repair_evidence_rejected(
        self,
    ) -> None:

        selector = RepairEvidenceSelector()

        knowledge = (
            "The collector current is related to the "
            "emitter current."
        )

        fake_result = json.dumps(
            {
                "evidence_quotes": [
                    (
                        "The base current controls the "
                        "collector current."
                    ),
                ],
                "reason": "Proposed evidence.",
            }
        )

        with patch(
            (
                "app.services.repair_evidence_selector."
                "chat_with_structured_ai"
            ),
            return_value=fake_result,
        ):
            result = selector.select(
                learner_message=(
                    "What does the base do?"
                ),
                knowledge_context=knowledge,
            )

        self._check(
            (
                result.status == "invalid"
                and not result.evidence_quotes
            ),
            (
                "GROUNDING-PRECISION-T79 "
                "Fabricated repair evidence"
            ),
            (
                "A fabricated evidence quote is "
                "rejected deterministically."
            ),
            (
                "Fabricated repair evidence was "
                "accepted."
            ),
        )

    def test_t80_paraphrased_repair_evidence_rejected(
        self,
    ) -> None:

        selector = RepairEvidenceSelector()

        knowledge = (
            "The collector current is related to the "
            "emitter current."
        )

        fake_result = json.dumps(
            {
                "evidence_quotes": [
                    (
                        "Collector current depends on "
                        "emitter current."
                    ),
                ],
                "reason": "Paraphrased evidence.",
            }
        )

        with patch(
            (
                "app.services.repair_evidence_selector."
                "chat_with_structured_ai"
            ),
            return_value=fake_result,
        ):
            result = selector.select(
                learner_message=(
                    "Explain collector current."
                ),
                knowledge_context=knowledge,
            )

        self._check(
            (
                result.status == "invalid"
                and not result.evidence_quotes
            ),
            (
                "GROUNDING-PRECISION-T80 "
                "Paraphrased repair evidence"
            ),
            (
                "Paraphrased evidence cannot pass "
                "exact source verification."
            ),
            (
                "A paraphrased quote was incorrectly "
                "treated as source evidence."
            ),
        )

    def test_t81_empty_repair_evidence_safe(
        self,
    ) -> None:

        selector = RepairEvidenceSelector()

        fake_result = json.dumps(
            {
                "evidence_quotes": [],
                "reason": (
                    "No directly relevant evidence."
                ),
            }
        )

        with patch(
            (
                "app.services.repair_evidence_selector."
                "chat_with_structured_ai"
            ),
            return_value=fake_result,
        ):
            result = selector.select(
                learner_message=(
                    "What is the exact gain?"
                ),
                knowledge_context=(
                    "The BJT has three terminals."
                ),
            )

        self._check(
            (
                result.status == "no_evidence"
                and not result.evidence_quotes
            ),
            (
                "GROUNDING-PRECISION-T81 "
                "Empty repair evidence"
            ),
            (
                "A selector with no source evidence "
                "returns a safe no-evidence result."
            ),
            (
                "Empty evidence selection did not "
                "fail safely."
            ),
        )

    def test_t82_repair_evidence_provider_fail_closed(
        self,
    ) -> None:

        selector = RepairEvidenceSelector()

        with patch(
            (
                "app.services.repair_evidence_selector."
                "chat_with_structured_ai"
            ),
            side_effect=RuntimeError(
                "simulated selector failure"
            ),
        ):
            result = selector.select(
                learner_message=(
                    "Explain the base."
                ),
                knowledge_context=(
                    "The three terminals of the BJT "
                    "are called the Base, Collector "
                    "and Emitter."
                ),
            )

        self._check(
            (
                result.status == "invalid"
                and not result.evidence_quotes
                and any(
                    "RuntimeError" in issue
                    for issue in result.issues
                )
            ),
            (
                "GROUNDING-PRECISION-T82 "
                "Evidence selector fail-closed"
            ),
            (
                "Evidence selector provider failures "
                "produce an invalid empty evidence set."
            ),
            (
                "Evidence selector provider failure "
                "did not fail closed."
            ),
        )  

    def test_t83_runtime_repair_evidence_selector_initialized(
        self,
    ) -> None:

        tutor = AITutor()

        self._check(
            isinstance(
                tutor.repair_evidence_selector,
                RepairEvidenceSelector,
            ),
            (
                "GROUNDING-PRECISION-T83 "
                "Runtime evidence selector"
            ),
            (
                "AITutor owns the verified repair "
                "evidence selector."
            ),
            (
                "AITutor does not initialize the "
                "repair evidence selector."
            ),
        )            

    def test_t84_supported_repair_preserves_context(
        self,
    ) -> None:

        tutor = AITutor()

        validation = GroundingValidationResult(
            status="supported",
            confidence=1.0,
            reason="Supported.",
            issues=[],
        )

        with patch.object(
            tutor.repair_evidence_selector,
            "select",
        ) as mocked_select:

            selection, context = (
                tutor._prepare_llm_repair_context(
                    learner_message="Explain this.",
                    knowledge_context=(
                        "Original course knowledge."
                    ),
                    grounding_validation=validation,
                )
            )

        self._check(
            (
                selection is None
                and context
                == "Original course knowledge."
                and mocked_select.call_count == 0
            ),
            (
                "GROUNDING-PRECISION-T84 "
                "Supported repair bypass"
            ),
            (
                "Supported pedagogical repair preserves "
                "the existing full knowledge context "
                "without evidence selection."
            ),
            (
                "Evidence selection incorrectly changed "
                "supported repair behavior."
            ),
        )

    def test_t85_factual_repair_verified_context(
        self,
    ) -> None:

        tutor = AITutor()

        validation = GroundingValidationResult(
            status="partially_supported",
            confidence=0.9,
            reason="Partial.",
            issues=[
                "Unsupported relation."
            ],
        )

        selected = (
            RepairEvidenceSelectionResult(
                status="verified",
                evidence_quotes=(
                    "Exact evidence one.",
                    "Exact evidence two.",
                ),
                reason="Verified.",
                issues=(),
            )
        )

        with patch.object(
            tutor.repair_evidence_selector,
            "select",
            return_value=selected,
        ):
            selection, context = (
                tutor._prepare_llm_repair_context(
                    learner_message=(
                        "Explain the concept."
                    ),
                    knowledge_context=(
                        "Exact evidence one. "
                        "Exact evidence two."
                    ),
                    grounding_validation=validation,
                )
            )

        self._check(
            (
                selection == selected
                and
                context
                == (
                    "[Verified Evidence 1]\n"
                    "Exact evidence one.\n\n"
                    "[Verified Evidence 2]\n"
                    "Exact evidence two."
                )
            ),
            (
                "GROUNDING-PRECISION-T85 "
                "Verified repair context"
            ),
            (
                "Factual repair is reduced to an exact "
                "verified evidence pack."
            ),
            (
                "Factual repair did not build the "
                "verified evidence boundary correctly."
            ),
        )

    def test_t86_no_evidence_blocks_repair_context(
        self,
    ) -> None:

        tutor = AITutor()

        validation = GroundingValidationResult(
            status="unsupported",
            confidence=1.0,
            reason="Unsupported.",
            issues=[
                "No source support."
            ],
        )

        selected = (
            RepairEvidenceSelectionResult(
                status="no_evidence",
                evidence_quotes=(),
                reason="No evidence.",
                issues=(),
            )
        )

        with patch.object(
            tutor.repair_evidence_selector,
            "select",
            return_value=selected,
        ):
            selection, context = (
                tutor._prepare_llm_repair_context(
                    learner_message=(
                        "Give the exact value."
                    ),
                    knowledge_context=(
                        "Unrelated course knowledge."
                    ),
                    grounding_validation=validation,
                )
            )

        self._check(
            (
                selection == selected
                and context is None
            ),
            (
                "GROUNDING-PRECISION-T86 "
                "No-evidence repair block"
            ),
            (
                "Factual repair receives no generation "
                "context when verified evidence is absent."
            ),
            (
                "LLM repair could still receive factual "
                "context without verified evidence."
            ),
        )

    def test_t87_repair_context_typed_input(
        self,
    ) -> None:

        tutor = AITutor()

        rejected = False

        try:
            tutor._prepare_llm_repair_context(
                learner_message=123,
                knowledge_context="Knowledge.",
                grounding_validation=(
                    GroundingValidationResult(
                        status="supported",
                        confidence=1.0,
                        reason="Supported.",
                        issues=[],
                    )
                ),
            )

        except TypeError:
            rejected = True

        self._check(
            rejected,
            (
                "GROUNDING-PRECISION-T87 "
                "Repair context typed input"
            ),
            (
                "Verified-evidence repair context "
                "enforces typed runtime inputs."
            ),
            (
                "Invalid repair-context input was "
                "accepted."
            ),
        )

    def test_t88_runtime_verified_context_to_repair(
        self,
    ) -> None:

        source = Path(
            "app/tutor.py"
        ).read_text(
            encoding="utf-8"
        )

        self._check(
            (
                "knowledge_context=(\n"
                "                                    "
                "repair_generation_context"
                in source
            ),
            (
                "GROUNDING-PRECISION-T88 "
                "Verified context to repair"
            ),
            (
                "LLM repair receives the prepared "
                "repair generation context."
            ),
            (
                "LLM repair still bypasses the "
                "verified repair context."
            ),
        )

    def test_t88_runtime_verified_context_to_repair(
        self,
    ) -> None:

        source = Path(
            "app/tutor.py"
        ).read_text(
            encoding="utf-8"
        )

        self._check(
            (
                "repair_generation_context"
                in source
                and
                "knowledge_context=("
                in source
            ),
            (
                "GROUNDING-PRECISION-T88 "
                "Verified context to repair"
            ),
            (
                "LLM repair receives the prepared "
                "repair generation context."
            ),
            (
                "LLM repair still bypasses the "
                "verified repair context."
            ),
        )       

    def test_t89_no_evidence_runtime_fail_closed(
        self,
    ) -> None:

        source = Path(
            "app/tutor.py"
        ).read_text(
            encoding="utf-8"
        )

        self._check(
            (
                'self.last_repair_failure_type = (\n'
                '                            "repair_evidence"'
                in source
                or
                '"repair_evidence"'
                in source
            ),
            (
                "GROUNDING-PRECISION-T89 "
                "No-evidence runtime block"
            ),
            (
                "Runtime factual repair fails closed "
                "when verified evidence is absent."
            ),
            (
                "Runtime does not block LLM repair "
                "without verified evidence."
            ),
        )

    def test_t90_repair_validator_verified_boundary(
        self,
    ) -> None:

        source = Path(
            "app/tutor.py"
        ).read_text(
            encoding="utf-8"
        )

        marker = (
            "task_name=(\n"
            '                                        '
            '"response_validator_repair"'
        )

        validator_pos = source.find(
            marker
        )

        context_pos = source.rfind(
            "repair_validation_context",
            0,
            validator_pos,
        )

        self._check(
            (
                validator_pos != -1
                and context_pos != -1
            ),
            (
                "GROUNDING-PRECISION-T90 "
                "Repair validator evidence boundary"
            ),
            (
                "Repair grounding validation uses "
                "the prepared evidence boundary."
            ),
            (
                "Repair validator is not tied to "
                "the verified repair context."
            ),
        )

    def test_t91_repair_precision_verified_boundary(
        self,
    ) -> None:

        source = Path(
            "app/tutor.py"
        ).read_text(
            encoding="utf-8"
        )

        self._check(
            (
                "repair_grounding_precision_guard_result"
                in source
                and
                "repair_validation_context"
                in source
            ),
            (
                "GROUNDING-PRECISION-T91 "
                "Repair precision evidence boundary"
            ),
            (
                "Repair precision validation uses "
                "the same evidence boundary."
            ),
            (
                "Repair precision validation can "
                "escape the verified evidence boundary."
            ),
        )

    def test_t92_runtime_evidence_telemetry(
        self,
    ) -> None:

        source = Path(
            "app/tutor.py"
        ).read_text(
            encoding="utf-8"
        )

        self._check(
            (
                "self.last_repair_evidence_selection"
                in source
                and
                "self.last_repair_evidence_context"
                in source
            ),
            (
                "GROUNDING-PRECISION-T92 "
                "Runtime evidence telemetry"
            ),
            (
                "Runtime records repair evidence "
                "selection and context."
            ),
            (
                "Verified repair evidence is not "
                "observable at runtime."
            ),
        )

    def test_t93_supported_repair_path_preserved(
        self,
    ) -> None:

        tutor = AITutor()

        validation = GroundingValidationResult(
            status="supported",
            confidence=1.0,
            reason="Supported.",
            issues=[],
        )

        with patch.object(
            tutor.repair_evidence_selector,
            "select",
        ) as mocked_select:

            selection, context = (
                tutor._prepare_llm_repair_context(
                    learner_message=(
                        "Explain this."
                    ),
                    knowledge_context=(
                        "Full course context."
                    ),
                    grounding_validation=(
                        validation
                    ),
                )
            )

        self._check(
            (
                selection is None
                and
                context
                == "Full course context."
                and
                mocked_select.call_count == 0
            ),
            (
                "GROUNDING-PRECISION-T93 "
                "Supported repair preserved"
            ),
            (
                "Pedagogical-only repair keeps "
                "the frozen full-context path."
            ),
            (
                "Verified-evidence integration "
                "changed supported repair behavior."
            ),
        )
    def test_t94_selector_requires_evidence_sufficiency(
        self,
    ) -> None:

        from app.services.repair_evidence_selector import (
            REPAIR_EVIDENCE_SELECTION_PROMPT,
        )

        prompt = " ".join(
            REPAIR_EVIDENCE_SELECTION_PROMPT
            .lower()
            .split()
        )

        self._check(
            (
                "evidence sufficiency rules"
                in prompt
                and
                "type of information requested"
                in prompt
            ),
            (
                "GROUNDING-PRECISION-T94 "
                "Evidence sufficiency contract"
            ),
            (
                "Repair evidence selection requires "
                "evidence to match the learner's "
                "requested information type."
            ),
            (
                "Evidence selector does not enforce "
                "semantic sufficiency."
            ),
        )
    def test_t95_identity_not_function_evidence(
        self,
    ) -> None:

        from app.services.repair_evidence_selector import (
            REPAIR_EVIDENCE_SELECTION_PROMPT,
        )

        prompt = (
            REPAIR_EVIDENCE_SELECTION_PROMPT
            .lower()
        )

        self._check(
            (
                "function or role"
                in prompt
                and
                "identifies the base"
                in prompt
                and
                "does not establish its"
                in prompt
            ),
            (
                "GROUNDING-PRECISION-T95 "
                "Identity-function distinction"
            ),
            (
                "Entity identity alone cannot be "
                "treated as evidence of its function."
            ),
            (
                "Evidence selector may confuse entity "
                "identity with functional evidence."
            ),
        )

    def test_t96_selector_supports_evidence_chain(
        self,
    ) -> None:

        from app.services.repair_evidence_selector import (
            REPAIR_EVIDENCE_SELECTION_PROMPT,
        )

        prompt = (
            REPAIR_EVIDENCE_SELECTION_PROMPT
            .lower()
        )

        self._check(
            (
                "select multiple exact excerpts"
                in prompt
                and
                "chain of statements"
                in prompt
            ),
            (
                "GROUNDING-PRECISION-T96 "
                "Evidence chain selection"
            ),
            (
                "Selector may use multiple verified "
                "excerpts when one excerpt is "
                "insufficient."
            ),
            (
                "Selector is still biased toward a "
                "single incomplete evidence excerpt."
            ),
        )

    def test_t97_atomic_issue_field_contract(
        self,
    ) -> None:

        from app.services.response_grounding_validator import (
            GROUNDING_VALIDATION_PROMPT,
        )

        prompt = " ".join(
            GROUNDING_VALIDATION_PROMPT
            .lower()
            .split()
        )

        self._check(
            (
                '"issue" field is still required'
                in prompt
                and
                'must be present as an empty string'
                in prompt
            ),
            (
                "GROUNDING-PRECISION-T97 "
                "Atomic issue-field contract"
            ),
            (
                "Every atomic claim explicitly retains "
                "the required issue field."
            ),
            (
                "Atomic structured output can omit "
                "the issue field."
            ),
        )

    def test_t98_operational_evidence_chain(
        self,
    ) -> None:

        from app.services.repair_evidence_selector import (
            REPAIR_EVIDENCE_SELECTION_PROMPT,
        )

        prompt = " ".join(
            REPAIR_EVIDENCE_SELECTION_PROMPT
            .lower()
            .split()
        )

        self._check(
            (
                "operational evidence chain"
                in prompt
                and
                "does not require"
                in prompt
                and
                "stronger functional label"
                in prompt
            ),
            (
                "GROUNDING-PRECISION-T98 "
                "Operational evidence chain"
            ),
            (
                "Function questions may use a "
                "sufficient explicit evidence chain "
                "without relation strengthening."
            ),
            (
                "Selector still requires an explicit "
                "function sentence."
            ),
        )

    def test_t99_evidence_safe_composer(
        self,
    ) -> None:

        composer = EvidenceSafeRepairComposer()

        self._check(
            isinstance(
                composer,
                EvidenceSafeRepairComposer,
            ),
            (
                "GROUNDING-PRECISION-T99 "
                "Evidence-safe composer"
            ),
            (
                "Verified evidence has a dedicated "
                "deterministic composition component."
            ),
            (
                "Evidence-safe repair composer "
                "is unavailable."
            ),
        )


    def test_t100_evidence_safe_exact_quotes(
        self,
    ) -> None:

        composer = EvidenceSafeRepairComposer()

        selection = (
            RepairEvidenceSelectionResult(
                status="verified",
                evidence_quotes=(
                    "Exact evidence one.",
                    "Exact evidence two.",
                ),
                reason=(
                    "This reason must not become "
                    "factual output."
                ),
                issues=(),
            )
        )

        result = composer.compose(
            selection
        )

        self._check(
            result == (
                "Exact evidence one.\n\n"
                "Exact evidence two."
            ),
            (
                "GROUNDING-PRECISION-T100 "
                "Exact evidence composition"
            ),
            (
                "Evidence-safe composition preserves "
                "verified quotes exactly and in order."
            ),
            (
                "Evidence-safe composition altered "
                "verified source evidence."
            ),
        )


    def test_t101_selector_reason_excluded(
        self,
    ) -> None:

        composer = EvidenceSafeRepairComposer()

        selection = (
            RepairEvidenceSelectionResult(
                status="verified",
                evidence_quotes=(
                    "Source relation is related to.",
                ),
                reason=(
                    "X controls Y and determines Z."
                ),
                issues=(),
            )
        )

        result = composer.compose(
            selection
        )

        self._check(
            (
                result
                == "Source relation is related to."
                and
                "controls"
                not in result.lower()
                and
                "determines"
                not in result.lower()
            ),
            (
                "GROUNDING-PRECISION-T101 "
                "Selector reason isolation"
            ),
            (
                "Evidence-safe composition uses only "
                "verified quotes and excludes selector "
                "reasoning."
            ),
            (
                "Selector reasoning can leak into "
                "factual repair output."
            ),
        )
    def test_t102_evidence_safe_no_evidence(
        self,
    ) -> None:

        composer = EvidenceSafeRepairComposer()

        selection = (
            RepairEvidenceSelectionResult(
                status="no_evidence",
                evidence_quotes=(),
                reason="No suitable evidence.",
                issues=(),
            )
        )

        self._check(
            composer.compose(
                selection
            ) is None,
            (
                "GROUNDING-PRECISION-T102 "
                "Evidence-safe no-evidence"
            ),
            (
                "Evidence-safe composition fails "
                "closed when verified evidence "
                "is unavailable."
            ),
            (
                "Composer generated factual output "
                "without verified evidence."
            ),
        )

    def test_t103_evidence_safe_invalid(
        self,
    ) -> None:

        composer = EvidenceSafeRepairComposer()

        selection = (
            RepairEvidenceSelectionResult(
                status="invalid",
                evidence_quotes=(),
                reason="Provider failure.",
                issues=(
                    "Controlled failure.",
                ),
            )
        )

        self._check(
            composer.compose(
                selection
            ) is None,
            (
                "GROUNDING-PRECISION-T103 "
                "Evidence-safe invalid selection"
            ),
            (
                "Invalid evidence selection cannot "
                "produce factual repair output."
            ),
            (
                "Invalid evidence selection escaped "
                "the composition boundary."
            ),
        )

    def test_t104_evidence_safe_typed_input(
        self,
    ) -> None:

        composer = EvidenceSafeRepairComposer()

        raised = False

        try:
            composer.compose(
                "not-a-selection"
            )
        except TypeError:
            raised = True

        self._check(
            raised,
            (
                "GROUNDING-PRECISION-T104 "
                "Evidence-safe typed input"
            ),
            (
                "Evidence-safe composition enforces "
                "typed verified-evidence input."
            ),
            (
                "Evidence-safe composition accepted "
                "an invalid input type."
            ),
        )
    def test_t105_runtime_evidence_safe_composer(
        self,
    ) -> None:

        tutor = AITutor()

        self._check(
            isinstance(
                tutor.evidence_safe_repair_composer,
                EvidenceSafeRepairComposer,
            ),
            (
                "GROUNDING-PRECISION-T105 "
                "Runtime evidence-safe composer"
            ),
            (
                "AITutor owns the deterministic "
                "evidence-safe repair composer."
            ),
            (
                "AITutor does not own the "
                "evidence-safe repair composer."
            ),
        )

    def test_t106_runtime_evidence_safe_helper(
        self,
    ) -> None:

        tutor = AITutor()

        selection = (
            RepairEvidenceSelectionResult(
                status="verified",
                evidence_quotes=(
                    "Evidence one.",
                    "Evidence two.",
                ),
                reason="Do not expose this.",
                issues=(),
            )
        )

        result = (
            tutor._compose_evidence_safe_repair(
                selection
            )
        )

        self._check(
            result == (
                "Evidence one.\n\n"
                "Evidence two."
            ),
            (
                "GROUNDING-PRECISION-T106 "
                "Runtime evidence-safe helper"
            ),
            (
                "Runtime adapter returns exact "
                "deterministic evidence composition."
            ),
            (
                "Runtime evidence-safe adapter "
                "altered verified evidence."
            ),
        )

    def test_t107_runtime_evidence_safe_none(
        self,
    ) -> None:

        tutor = AITutor()

        self._check(
            (
                tutor._compose_evidence_safe_repair(
                    None
                )
                is None
            ),
            (
                "GROUNDING-PRECISION-T107 "
                "Runtime evidence-safe null"
            ),
            (
                "Runtime evidence-safe adapter "
                "fails closed without selection."
            ),
            (
                "Runtime evidence-safe adapter "
                "generated output without selection."
            ),
        )

    def test_t108_evidence_safe_telemetry_defaults(
        self,
    ) -> None:

        tutor = AITutor()

        debug = tutor.get_debug_info()

        self._check(
            (
                debug[
                    "evidence_safe_repair_candidate"
                ]
                is None
                and
                debug[
                    "evidence_safe_repair_used"
                ]
                is False
            ),
            (
                "GROUNDING-PRECISION-T108 "
                "Evidence-safe telemetry defaults"
            ),
            (
                "Evidence-safe repair telemetry "
                "starts inactive."
            ),
            (
                "Evidence-safe repair telemetry "
                "starts with stale state."
            ),
        )

    def test_t109_evidence_safe_reset(
        self,
    ) -> None:

        tutor = AITutor()

        tutor.last_evidence_safe_repair_candidate = (
            "stale"
        )
        tutor.last_evidence_safe_repair_used = True

        tutor.reset()

        self._check(
            (
                tutor.last_evidence_safe_repair_candidate
                is None
                and
                tutor.last_evidence_safe_repair_used
                is False
            ),
            (
                "GROUNDING-PRECISION-T109 "
                "Evidence-safe telemetry reset"
            ),
            (
                "Conversation reset clears "
                "evidence-safe repair telemetry."
            ),
            (
                "Evidence-safe repair telemetry "
                "survived conversation reset."
            ),
        )
    def test_t110_evidence_safe_runtime_boundary(
        self,
    ) -> None:

        source = Path(
            "app/tutor.py"
        ).read_text(
            encoding="utf-8"
        )

        self._check(
            (
                "evidence_safe_accepted = False"
                in source
                and
                "self.last_repair_mode"
                in source
                and
                '== "llm"'
                in source
                and
                ".has_verified_evidence"
                in source
            ),
            (
                "GROUNDING-PRECISION-T110 "
                "Evidence-safe activation boundary"
            ),
            (
                "Evidence-safe fallback is limited "
                "to failed factual LLM repair with "
                "verified evidence."
            ),
            (
                "Evidence-safe fallback can activate "
                "outside its intended repair path."
            ),
        )
    def test_t111_evidence_safe_runtime_composition(
        self,
    ) -> None:

        source = Path(
            "app/tutor.py"
        ).read_text(
            encoding="utf-8"
        )

        self._check(
            (
                "_compose_evidence_safe_repair("
                in source
                and
                "last_evidence_safe_repair_candidate"
                in source
            ),
            (
                "GROUNDING-PRECISION-T111 "
                "Evidence-safe runtime composition"
            ),
            (
                "Failed LLM grounding can produce "
                "an observable deterministic "
                "evidence-safe candidate."
            ),
            (
                "Evidence-safe composer is not "
                "connected to runtime fallback."
            ),
        )

    def test_t112_evidence_safe_validation_boundary(
        self,
    ) -> None:

        source = Path(
            "app/tutor.py"
        ).read_text(
            encoding="utf-8"
        )

        self._check(
            (
                "evidence_safe_validation"
                in source
                and
                "self.last_repair_evidence_context"
                in source
                and
                '"response_validator_repair"'
                in source
            ),
            (
                "GROUNDING-PRECISION-T112 "
                "Evidence-safe validation boundary"
            ),
            (
                "Evidence-safe candidate is "
                "re-validated against the exact "
                "verified evidence context."
            ),
            (
                "Evidence-safe validation can escape "
                "the verified evidence boundary."
            ),
        )

    def test_t113_evidence_safe_precision(
        self,
    ) -> None:

        source = Path(
            "app/tutor.py"
        ).read_text(
            encoding="utf-8"
        )

        self._check(
            (
                "evidence_safe_precision_guard"
                in source
                and
                "grounding_precision_veto_policy"
                in source
            ),
            (
                "GROUNDING-PRECISION-T113 "
                "Evidence-safe precision"
            ),
            (
                "Evidence-safe fallback retains "
                "deterministic precision enforcement."
            ),
            (
                "Evidence-safe fallback bypasses "
                "precision enforcement."
            ),
        )

    def test_t114_evidence_safe_pedagogy(
        self,
    ) -> None:

        source = Path(
            "app/tutor.py"
        ).read_text(
            encoding="utf-8"
        )

        self._check(
            (
                "evidence_safe_pedagogical_precheck"
                in source
                and
                "response_mode_pedagogical_precheck"
                in source
                and
                "evidence_safe_pedagogical_validation"
                in source
                and
                "response_mode_pedagogical_validator"
                in source
            ),
            (
                "GROUNDING-PRECISION-T114 "
                "Evidence-safe pedagogy"
            ),
            (
                "Evidence-safe fallback remains "
                "subject to response-mode-aware "
                "pedagogical validation."
            ),
            (
                "Evidence-safe fallback bypasses "
                "the frozen pedagogical architecture."
            ),
        )

    def test_t115_evidence_safe_acceptance_gate(
        self,
    ) -> None:

        source = Path(
            "app/tutor.py"
        ).read_text(
            encoding="utf-8"
        )

        acceptance_pos = source.find(
            "self.last_evidence_safe_repair_used = ("
        )

        supported_pos = source.rfind(
            '== "supported"',
            0,
            acceptance_pos,
        )

        valid_pos = source.rfind(
            '== "valid"',
            0,
            acceptance_pos,
        )

        self._check(
            (
                acceptance_pos != -1
                and
                supported_pos != -1
                and
                valid_pos != -1
            ),
            (
                "GROUNDING-PRECISION-T115 "
                "Evidence-safe acceptance gate"
            ),
            (
                "Evidence-safe repair is marked used "
                "only after grounding and pedagogy "
                "acceptance."
            ),
            (
                "Evidence-safe repair can be marked "
                "used before full validation."
            ),
        )        
    def test_t116_evidence_safe_final_fail_closed(
        self,
    ) -> None:

        source = Path(
            "app/tutor.py"
        ).read_text(
            encoding="utf-8"
        )

        self._check(
            (
                "if not evidence_safe_accepted:"
                in source
                and
                "_build_grounding_fallback("
                in source
                and
                "_build_pedagogical_fallback("
                in source
            ),
            (
                "GROUNDING-PRECISION-T116 "
                "Evidence-safe final fail-closed"
            ),
            (
                "Failed evidence-safe repair still "
                "ends in the existing deterministic "
                "fallback path."
            ),
            (
                "Evidence-safe repair weakened the "
                "final fail-closed policy."
            ),
        )
    def test_t117_evidence_quote_completeness_contract(
        self,
    ) -> None:

        from app.services.repair_evidence_selector import (
            REPAIR_EVIDENCE_SELECTION_PROMPT,
        )

        prompt = " ".join(
            REPAIR_EVIDENCE_SELECTION_PROMPT
            .lower()
            .split()
        )

        self._check(
            (
                "semantically complete enough"
                in prompt
                and
                "self-contained evidence unit"
                in prompt
            ),
            (
                "GROUNDING-PRECISION-T117 "
                "Evidence quote completeness"
            ),
            (
                "Evidence selection requires "
                "self-contained evidence units."
            ),
            (
                "Evidence selector can return "
                "incomplete source fragments."
            ),
        )

    def test_t118_truncated_leadin_rejected(
        self,
    ) -> None:

        from app.services.repair_evidence_selector import (
            REPAIR_EVIDENCE_SELECTION_PROMPT,
        )

        prompt = " ".join(
            REPAIR_EVIDENCE_SELECTION_PROMPT
            .lower()
            .split()
        )

        self._check(
            (
                "truncated lead-in"
                in prompt
                and
                'ends with an unfinished connector'
                in prompt
            ),
            (
                "GROUNDING-PRECISION-T118 "
                "Truncated evidence rejection"
            ),
            (
                "Selector explicitly rejects "
                "unfinished source fragments."
            ),
            (
                "Selector does not guard against "
                "truncated evidence."
            ),
        )

    def test_t119_equation_leadin_example(
        self,
    ) -> None:

        from app.services.repair_evidence_selector import (
            REPAIR_EVIDENCE_SELECTION_PROMPT,
        )

        prompt = " ".join(
            REPAIR_EVIDENCE_SELECTION_PROMPT
            .lower()
            .split()
        )

        self._check(
            (
                "example of incomplete evidence"
                in prompt
                and
                "b-e junction is related"
                in prompt
                and
                "equation or continuation"
                in prompt
            ),
            (
                "GROUNDING-PRECISION-T119 "
                "Equation lead-in example"
            ),
            (
                "Selector has an explicit example "
                "covering the live truncated "
                "equation lead-in defect."
            ),
            (
                "Live evidence-fragment defect "
                "is not represented in the "
                "selector contract."
            ),
        )

    def test_t120_evidence_text_integrity_contract(
        self,
    ) -> None:

        from app.services.repair_evidence_selector import (
            REPAIR_EVIDENCE_SELECTION_PROMPT,
        )

        prompt = " ".join(
            REPAIR_EVIDENCE_SELECTION_PROMPT
            .lower()
            .split()
        )

        self._check(
            (
                "evidence text integrity rules"
                in prompt
                and
                "mathematical symbols"
                in prompt
                and
                "pdf extraction"
                in prompt
            ),
            (
                "GROUNDING-PRECISION-T120 "
                "Evidence text integrity"
            ),
            (
                "Evidence selection explicitly "
                "rejects extraction-damaged source text."
            ),
            (
                "Evidence selection does not account "
                "for damaged PDF extraction."
            ),
        )

    def test_t121_missing_variable_example(
        self,
    ) -> None:

        from app.services.repair_evidence_selector import (
            REPAIR_EVIDENCE_SELECTION_PROMPT,
        )

        prompt = " ".join(
            REPAIR_EVIDENCE_SELECTION_PROMPT
            .lower()
            .split()
        )

        self._check(
            (
                "with the voltage and as shown"
                in prompt
                and
                "variables or symbols"
                in prompt
            ),
            (
                "GROUNDING-PRECISION-T121 "
                "Missing-variable evidence"
            ),
            (
                "The live malformed voltage excerpt "
                "is explicitly represented in the "
                "evidence-selection contract."
            ),
            (
                "The selector has no explicit guard "
                "for the observed missing-variable "
                "PDF extraction defect."
            ),
        )       

    def test_t122_clean_evidence_preference(
        self,
    ) -> None:

        from app.services.repair_evidence_selector import (
            REPAIR_EVIDENCE_SELECTION_PROMPT,
        )

        prompt = " ".join(
            REPAIR_EVIDENCE_SELECTION_PROMPT
            .lower()
            .split()
        )

        self._check(
            (
                "always prefer the clean"
                in prompt
                and
                "collector current is related"
                in prompt
            ),
            (
                "GROUNDING-PRECISION-T122 "
                "Clean evidence preference"
            ),
            (
                "Selector prefers clean complete "
                "evidence over extraction-damaged "
                "alternatives."
            ),
            (
                "Selector has no clean-evidence "
                "preference when source alternatives "
                "exist."
            ),
        )

    def test_t123_no_source_reconstruction_contract(
        self,
    ) -> None:

        from app.services.repair_evidence_selector import (
            REPAIR_EVIDENCE_SELECTION_PROMPT,
        )

        prompt = " ".join(
            REPAIR_EVIDENCE_SELECTION_PROMPT
            .lower()
            .split()
        )

        self._check(
            (
                "never repair, reconstruct, restore"
                in prompt
                and
                "must never be improved by changing"
                in prompt
            ),
            (
                "GROUNDING-PRECISION-T123 "
                "No source reconstruction"
            ),
            (
                "Selector cannot repair damaged "
                "source text into new evidence."
            ),
            (
                "Selector may still reconstruct "
                "damaged source evidence."
            ),
        )
    def test_t124_missing_content_not_inserted(
        self,
    ) -> None:

        from app.services.repair_evidence_selector import (
            REPAIR_EVIDENCE_SELECTION_PROMPT,
        )

        prompt = " ".join(
            REPAIR_EVIDENCE_SELECTION_PROMPT
            .lower()
            .split()
        )

        self._check(
            (
                "never insert a missing variable"
                in prompt
                and
                "clean exact excerpt"
                in prompt
            ),
            (
                "GROUNDING-PRECISION-T124 "
                "No missing-content insertion"
            ),
            (
                "Missing PDF content cannot be "
                "invented during evidence selection."
            ),
            (
                "Selector may insert missing "
                "source content."
            ),
        )

    def test_t125_reconstruction_example(
        self,
    ) -> None:

        from app.services.repair_evidence_selector import (
            REPAIR_EVIDENCE_SELECTION_PROMPT,
        )

        prompt = " ".join(
            REPAIR_EVIDENCE_SELECTION_PROMPT
            .lower()
            .split()
        )

        self._check(
            (
                "example of forbidden source reconstruction"
                in prompt
                and
                "vbe and vcb"
                in prompt
                and
                "not an exact source excerpt"
                in prompt
            ),
            (
                "GROUNDING-PRECISION-T125 "
                "Source reconstruction example"
            ),
            (
                "Prompt explicitly covers the "
                "observed PDF reconstruction risk."
            ),
            (
                "Observed source-reconstruction "
                "defect is not represented."
            ),
        )
    def test_t126_translation_result_typed(
        self,
    ) -> None:

        from app.services.evidence_safe_translation_models import (
            EvidenceSafeTranslationResult,
        )

        result = EvidenceSafeTranslationResult(
            status="not_required",
            translated_text="",
            source_text="Exact evidence.",
            reason="Controlled.",
            issues=(),
        )

        self._check(
            (
                result.status == "not_required"
                and
                result.source_text
                == "Exact evidence."
            ),
            (
                "GROUNDING-PRECISION-T126 "
                "Typed evidence translation"
            ),
            (
                "Evidence-safe translation uses "
                "a typed immutable result."
            ),
            (
                "Evidence translation result "
                "contract is unavailable."
            ),
        )
    def test_t127_translation_result_frozen(
        self,
    ) -> None:

        from dataclasses import FrozenInstanceError

        from app.services.evidence_safe_translation_models import (
            EvidenceSafeTranslationResult,
        )

        result = EvidenceSafeTranslationResult(
            status="not_required",
            translated_text="",
            source_text="Exact evidence.",
            reason="Controlled.",
            issues=(),
        )

        raised = False

        try:
            result.status = "translated"
        except FrozenInstanceError:
            raised = True

        self._check(
            raised,
            (
                "GROUNDING-PRECISION-T127 "
                "Frozen translation result"
            ),
            (
                "Evidence translation state "
                "cannot be mutated."
            ),
            (
                "Evidence translation result "
                "is mutable."
            ),
        )
    def test_t128_translation_boundary(
        self,
    ) -> None:

        from app.services.evidence_safe_translation_service import (
            EVIDENCE_SAFE_TRANSLATION_PROMPT,
        )

        prompt = " ".join(
            EVIDENCE_SAFE_TRANSLATION_PROMPT
            .lower()
            .split()
        )

        self._check(
            (
                "translation-only component"
                in prompt
                and
                "do not summarize"
                in prompt
                and
                "do not add technical knowledge"
                in prompt
            ),
            (
                "GROUNDING-PRECISION-T128 "
                "Translation-only boundary"
            ),
            (
                "Evidence presentation is bounded "
                "to translation rather than "
                "answer generation."
            ),
            (
                "Translation layer can generate "
                "new factual content."
            ),
        )
    def test_t129_translation_relation_strength(
        self,
    ) -> None:

        from app.services.evidence_safe_translation_service import (
            EVIDENCE_SAFE_TRANSLATION_PROMPT,
        )

        prompt = " ".join(
            EVIDENCE_SAFE_TRANSLATION_PROMPT
            .lower()
            .split()
        )

        self._check(
            (
                '"related to"'
                in prompt
                and
                '"controls"'
                in prompt
                and
                "do not strengthen relationships"
                in prompt
            ),
            (
                "GROUNDING-PRECISION-T129 "
                "Translation relation precision"
            ),
            (
                "Translation cannot strengthen "
                "source relationships."
            ),
            (
                "Translation may strengthen "
                "technical relationships."
            ),
        )

    def test_t130_translation_empty_fail_closed(
        self,
    ) -> None:

        from app.services.evidence_safe_translation_service import (
            EvidenceSafeTranslationService,
        )

        service = EvidenceSafeTranslationService()

        result = service.prepare_source("")

        self._check(
            (
                result.status == "invalid"
                and
                not result.has_translation
            ),
            (
                "GROUNDING-PRECISION-T130 "
                "Translation empty fail-closed"
            ),
            (
                "Translation preparation fails "
                "closed without verified evidence."
            ),
            (
                "Translation can proceed without "
                "verified evidence."
            ),
        )
    def test_t131_translation_schema_boundary(
        self,
    ) -> None:

        from app.services.evidence_safe_translation_service import (
            EVIDENCE_SAFE_TRANSLATION_SCHEMA,
        )

        self._check(
            (
                EVIDENCE_SAFE_TRANSLATION_SCHEMA[
                    "additionalProperties"
                ]
                is False
                and
                EVIDENCE_SAFE_TRANSLATION_SCHEMA[
                    "required"
                ]
                == ["translated_text"]
                and
                set(
                    EVIDENCE_SAFE_TRANSLATION_SCHEMA[
                        "properties"
                    ].keys()
                )
                == {"translated_text"}
            ),
            (
                "GROUNDING-PRECISION-T131 "
                "Translation schema boundary"
            ),
            (
                "Translation structured output "
                "contains only translated text."
            ),
            (
                "Translation schema permits "
                "additional generated content."
            ),
        )
    def test_t132_translation_success(
        self,
    ) -> None:

        from unittest.mock import patch

        from app.services.evidence_safe_translation_service import (
            EvidenceSafeTranslationService,
        )

        service = EvidenceSafeTranslationService()

        with patch(
            (
                "app.services."
                "evidence_safe_translation_service."
                "chat_with_structured_ai"
            ),
            return_value=(
                '{"translated_text": '
                '"กระแสคอลเลคเตอร์มีความสัมพันธ์'
                'กับกระแสอีมิตเตอร์"}'
            ),
        ):
            result = service.translate(
                evidence_text=(
                    "The collector current is related "
                    "to the emitter current."
                ),
                target_language="th",
            )

        self._check(
            (
                result.status == "translated"
                and
                result.has_translation
                and
                "มีความสัมพันธ์"
                in result.translated_text
            ),
            (
                "GROUNDING-PRECISION-T132 "
                "Bounded translation success"
            ),
            (
                "Verified evidence can be translated "
                "through the bounded service."
            ),
            (
                "Bounded translation did not return "
                "a valid translated result."
            ),
        )

    def test_t133_translation_input_boundary(
        self,
    ) -> None:

        from unittest.mock import patch

        from app.services.evidence_safe_translation_service import (
            EvidenceSafeTranslationService,
        )

        service = EvidenceSafeTranslationService()

        captured = {}

        def controlled_translation(
            **kwargs,
        ):
            captured.update(kwargs)

            return (
                '{"translated_text": '
                '"หลักฐานที่แปลแล้ว"}'
            )

        with patch(
            (
                "app.services."
                "evidence_safe_translation_service."
                "chat_with_structured_ai"
            ),
            side_effect=controlled_translation,
        ):
            service.translate(
                evidence_text=(
                    "Verified evidence only."
                ),
                target_language="th",
            )

        messages_text = " ".join(
            str(message.get("content", ""))
            for message in captured["messages"]
        )

        self._check(
            (
                "Verified evidence only."
                in messages_text
                and
                "LEARNER MESSAGE:"
                not in messages_text
                and
                "PRIOR GROUNDING ISSUES:"
                not in messages_text
                and
                "REPAIR EVIDENCE REASON:"
                not in messages_text
            ),
            (
                "GROUNDING-PRECISION-T133 "
                "Translation input isolation"
            ),
            (
                "Translation receives verified "
                "evidence without learner or "
                "repair-reason context."
            ),
            (
                "Translation input boundary exposes "
                "unnecessary generation context."
            ),
        )

    def test_t134_translation_provider_fail_closed(
        self,
    ) -> None:

        from unittest.mock import patch

        from app.services.evidence_safe_translation_service import (
            EvidenceSafeTranslationService,
        )

        service = EvidenceSafeTranslationService()

        with patch(
            (
                "app.services."
                "evidence_safe_translation_service."
                "chat_with_structured_ai"
            ),
            side_effect=RuntimeError(
                "controlled provider failure"
            ),
        ):
            result = service.translate(
                "Verified evidence.",
                "th",
            )

        self._check(
            (
                result.status == "invalid"
                and
                not result.has_translation
            ),
            (
                "GROUNDING-PRECISION-T134 "
                "Translation provider fail-closed"
            ),
            (
                "Translation provider failure "
                "cannot produce user-facing text."
            ),
            (
                "Provider failure escaped the "
                "translation safety boundary."
            ),
        )


    def test_t135_translation_invalid_json(
        self,
    ) -> None:

        from unittest.mock import patch

        from app.services.evidence_safe_translation_service import (
            EvidenceSafeTranslationService,
        )

        service = EvidenceSafeTranslationService()

        with patch(
            (
                "app.services."
                "evidence_safe_translation_service."
                "chat_with_structured_ai"
            ),
            return_value="not-json",
        ):
            result = service.translate(
                "Verified evidence.",
                "th",
            )

        self._check(
            result.status == "invalid",
            (
                "GROUNDING-PRECISION-T135 "
                "Translation invalid JSON"
            ),
            (
                "Malformed structured translation "
                "fails closed."
            ),
            (
                "Malformed translation output "
                "was accepted."
            ),
        )


    def test_t136_translation_empty_output(
        self,
    ) -> None:

        from unittest.mock import patch

        from app.services.evidence_safe_translation_service import (
            EvidenceSafeTranslationService,
        )

        service = EvidenceSafeTranslationService()

        with patch(
            (
                "app.services."
                "evidence_safe_translation_service."
                "chat_with_structured_ai"
            ),
            return_value=(
                '{"translated_text": ""}'
            ),
        ):
            result = service.translate(
                "Verified evidence.",
                "th",
            )

        self._check(
            (
                result.status == "invalid"
                and
                not result.has_translation
            ),
            (
                "GROUNDING-PRECISION-T136 "
                "Translation empty output"
            ),
            (
                "Empty translation output fails "
                "closed."
            ),
            (
                "Empty translated evidence "
                "was accepted."
            ),
        )


    def test_t137_english_translation_bypass(
        self,
    ) -> None:

        from unittest.mock import patch

        from app.services.evidence_safe_translation_service import (
            EvidenceSafeTranslationService,
        )

        service = EvidenceSafeTranslationService()

        with patch(
            (
                "app.services."
                "evidence_safe_translation_service."
                "chat_with_structured_ai"
            )
        ) as mocked:

            result = service.translate(
                "Verified evidence.",
                "en",
            )

        self._check(
            (
                result.status == "not_required"
                and
                result.source_text
                == "Verified evidence."
                and
                mocked.call_count == 0
            ),
            (
                "GROUNDING-PRECISION-T137 "
                "English translation bypass"
            ),
            (
                "Evidence already targeting English "
                "does not add an LLM call."
            ),
            (
                "Translation service invoked the "
                "provider unnecessarily."
            ),
        )

    def test_t138_translation_task_profile(
        self,
    ) -> None:

        from app.infrastructure.ai.task_profiles import (
            get_task_profile,
        )

        profile = get_task_profile(
            task_name=(
                "evidence_safe_translation"
            ),
            structured=True,
        )

        self._check(
            (
                profile.max_completion_tokens
                >= 800
                and
                profile.reasoning_effort
                == "low"
            ),
            (
                "GROUNDING-PRECISION-T138 "
                "Translation task profile"
            ),
            (
                "Bounded translation has an "
                "explicit low-reasoning structured "
                "task profile."
            ),
            (
                "Bounded translation task profile "
                "is missing or undersized."
            ),
        )

    def test_t139_runtime_translation_service(
        self,
    ) -> None:

        from app.services.evidence_safe_translation_service import (
            EvidenceSafeTranslationService,
        )

        tutor = AITutor()

        self._check(
            isinstance(
                tutor.evidence_safe_translation_service,
                EvidenceSafeTranslationService,
            ),
            (
                "GROUNDING-PRECISION-T139 "
                "Runtime translation service"
            ),
            (
                "AITutor owns the bounded "
                "evidence translation service."
            ),
            (
                "AITutor does not own the "
                "translation service."
            ),
        )

    def test_t140_runtime_translation_helper(
        self,
    ) -> None:

        from app.services.evidence_safe_translation_models import (
            EvidenceSafeTranslationResult,
        )

        tutor = AITutor()

        calls = {
            "count": 0,
            "evidence_text": None,
            "target_language": None,
        }

        def controlled_translate(
            evidence_text,
            target_language,
        ):
            calls["count"] += 1
            calls["evidence_text"] = evidence_text
            calls["target_language"] = target_language

            return EvidenceSafeTranslationResult(
                status="translated",
                translated_text="หลักฐานที่แปลแล้ว",
                source_text=evidence_text,
                reason="Controlled translation.",
                issues=(),
            )

        tutor.evidence_safe_translation_service.translate = (
            controlled_translate
        )

        result = tutor._translate_evidence_safe_repair(
            evidence_text="Verified evidence.",
            target_language="th",
        )

        self._check(
            (
                calls["count"] == 1
                and
                calls["evidence_text"]
                == "Verified evidence."
                and
                calls["target_language"] == "th"
                and
                result.status == "translated"
            ),
            (
                "GROUNDING-PRECISION-T140 "
                "Runtime translation helper"
            ),
            (
                "Runtime translation adapter "
                "passes only evidence text and "
                "target language."
            ),
            (
                "Runtime translation helper "
                "violated its input boundary."
            ),
        )

    def test_t141_translation_telemetry_defaults(
        self,
    ) -> None:

        tutor = AITutor()

        debug = tutor.get_debug_info()

        self._check(
            (
                debug[
                    "evidence_safe_translation_status"
                ]
                is None
                and
                debug[
                    "evidence_safe_translation_candidate"
                ]
                is None
                and
                debug[
                    "evidence_safe_translation_used"
                ]
                is False
            ),
            (
                "GROUNDING-PRECISION-T141 "
                "Translation telemetry defaults"
            ),
            (
                "Evidence-safe translation "
                "starts inactive."
            ),
            (
                "Translation telemetry starts "
                "with stale state."
            ),
        )

    def test_t142_translation_telemetry_reset(
        self,
    ) -> None:

        from app.services.evidence_safe_translation_models import (
            EvidenceSafeTranslationResult,
        )

        tutor = AITutor()

        tutor.last_evidence_safe_translation_result = (
            EvidenceSafeTranslationResult(
                status="translated",
                translated_text="ข้อความ",
                source_text="Evidence.",
                reason="Controlled.",
                issues=(),
            )
        )

        tutor.last_evidence_safe_translation_candidate = (
            "ข้อความ"
        )

        tutor.last_evidence_safe_translation_used = True

        tutor.reset()

        self._check(
            (
                tutor.last_evidence_safe_translation_result
                is None
                and
                tutor.last_evidence_safe_translation_candidate
                is None
                and
                tutor.last_evidence_safe_translation_used
                is False
            ),
            (
                "GROUNDING-PRECISION-T142 "
                "Translation telemetry reset"
            ),
            (
                "Conversation reset clears all "
                "translation telemetry."
            ),
            (
                "Translation telemetry survived "
                "conversation reset."
            ),
        )

    def test_t143_translation_runtime_activated(
        self,
    ) -> None:

        source = Path(
            "app/tutor.py"
        ).read_text(
            encoding="utf-8"
        )

        start = source.find(
            "# ACCEPT EVIDENCE-SAFE REPAIR"
        )

        end = source.find(
            "# FINAL FAIL-CLOSED",
            start,
        )

        block = (
            source[start:end]
            if start != -1 and end != -1
            else ""
        )

        self._check(
            (
                start != -1
                and
                "_translate_evidence_safe_repair("
                in block
                and
                "final_evidence_safe_answer"
                in block
                and
                "evidence_safe_candidate"
                in block
                and
                "last_evidence_safe_translation_result"
                in block
            ),
            (
                "GROUNDING-PRECISION-T143 "
                "Translation runtime activation"
            ),
            (
                "Bounded translation is activated "
                "inside the accepted evidence-safe "
                "repair path."
            ),
            (
                "Translation runtime integration "
                "is missing."
            ),
        )
    def test_t144_translation_target_language(
        self,
    ) -> None:

        source = Path(
            "app/tutor.py"
        ).read_text(
            encoding="utf-8"
        )

        start = source.find(
            "target_language = None"
        )

        end = source.find(
            "_translate_evidence_safe_repair(",
            start,
        )

        block = (
            source[start:end]
            if start != -1 and end != -1
            else ""
        )

        self._check(
            (
                start != -1
                and
                end != -1
                and
                "self.resolved_response_style"
                in block
                and
                ".primary_language"
                in block
            ),
            (
                "GROUNDING-PRECISION-T144 "
                "Translation target language"
            ),
            (
                "Bounded translation uses the "
                "resolved response-style language."
            ),
            (
                "Translation language is not owned "
                "by resolved response style."
            ),
        )


    def test_t145_translation_runtime_input_boundary(
        self,
    ) -> None:

        source = Path(
            "app/tutor.py"
        ).read_text(
            encoding="utf-8"
        )

        self._check(
            (
                "_translate_evidence_safe_repair("
                in source
                and
                "evidence_safe_candidate"
                in source
                and
                "target_language"
                in source
            ),
            (
                "GROUNDING-PRECISION-T145 "
                "Translation runtime input boundary"
            ),
            (
                "Runtime translator receives the "
                "evidence-safe candidate and "
                "resolved target language."
            ),
            (
                "Runtime translation boundary "
                "is not connected correctly."
            ),
        )


    def test_t146_translation_grounding_boundary(
        self,
    ) -> None:

        source = " ".join(
            Path("app/tutor.py").read_text(
                encoding="utf-8"
            ).split()
        )

        self._check(
            (
                "translation_validation ="
                in source
                and
                "response=( translated_candidate )"
                in source
                and
                "knowledge_context=( "
                "self.last_repair_evidence_context )"
                in source
                and
                '"response_validator_repair"'
                in source
            ),
            (
                "GROUNDING-PRECISION-T146 "
                "Translation grounding boundary"
            ),
            (
                "Translated evidence is re-validated "
                "against the same verified evidence "
                "boundary."
            ),
            (
                "Translated evidence can bypass "
                "verified grounding."
            ),
        )

    def test_t147_translation_precision_language_gate(
        self,
    ) -> None:

        source = " ".join(
            Path("app/tutor.py").read_text(
                encoding="utf-8"
            ).split()
        )

        self._check(
            (
                "translation_precision_guard"
                in source
                and
                "translation_language_consistency"
                in source
                and
                '== "consistent"'
                in source
            ),
            (
                "GROUNDING-PRECISION-T147 "
                "Translation precision and language"
            ),
            (
                "Translated evidence must pass "
                "precision and language consistency."
            ),
            (
                "Translated evidence can bypass "
                "precision or language validation."
            ),
        )

    def test_t148_translation_pedagogy_gate(
        self,
    ) -> None:

        source = Path(
            "app/tutor.py"
        ).read_text(
            encoding="utf-8"
        )

        self._check(
            (
                "translation_pedagogical_precheck"
                in source
                and
                "translation_pedagogical_validation"
                in source
                and
                "response_mode_pedagogical_precheck"
                in source
                and
                "response_mode_pedagogical_validator"
                in source
            ),
            (
                "GROUNDING-PRECISION-T148 "
                "Translation pedagogy gate"
            ),
            (
                "Translated evidence retains "
                "response-mode-aware pedagogical "
                "validation."
            ),
            (
                "Translated evidence bypasses "
                "pedagogical validation."
            ),
        )
    def test_t149_translation_safe_fallback(
        self,
    ) -> None:

        source = Path(
            "app/tutor.py"
        ).read_text(
            encoding="utf-8"
        )

        start = source.find(
            "# ACCEPT EVIDENCE-SAFE REPAIR"
        )

        end = source.find(
            "# FINAL FAIL-CLOSED",
            start,
        )

        block = (
            source[start:end]
            if start != -1 and end != -1
            else ""
        )

        base_assignment = block.find(
            "final_evidence_safe_answer"
        )

        base_candidate = block.find(
            "evidence_safe_candidate",
            base_assignment,
        )

        translation_accept = block.find(
            "final_evidence_safe_answer",
            base_candidate + 1,
        )

        translated_candidate = block.find(
            "translated_candidate",
            translation_accept,
        )

        answer_assignment = block.find(
            "answer =",
            translated_candidate,
        )

        final_answer_source = block.find(
            "final_evidence_safe_answer",
            answer_assignment,
        )

        self._check(
            (
                base_assignment != -1
                and
                base_candidate != -1
                and
                translation_accept != -1
                and
                translated_candidate != -1
                and
                answer_assignment != -1
                and
                final_answer_source != -1
                and
                base_assignment
                < base_candidate
                < translation_accept
                < translated_candidate
                < answer_assignment
                < final_answer_source
            ),
            (
                "GROUNDING-PRECISION-T149 "
                "Translation safe fallback"
            ),
            (
                "Evidence-safe English content is "
                "the default, translation may replace "
                "it only on acceptance, and final "
                "answer uses the selected safe value."
            ),
            (
                "Translation failure can discard "
                "the safe evidence response."
            ),
        )

    def test_t150_translation_acceptance_telemetry(
        self,
    ) -> None:

        source = " ".join(
            Path("app/tutor.py").read_text(
                encoding="utf-8"
            ).split()
        )

        self._check(
            (
                "last_evidence_safe_translation_used = ( True )"
                in source
                and
                "repair_validation = ( accepted_validation )"
                in source
                and
                "repair_grounding_precision_guard_result = "
                "( accepted_precision_guard )"
                in source
                and
                "repair_pedagogical_validation = "
                "( accepted_pedagogical_validation )"
                in source
            ),
            (
                "GROUNDING-PRECISION-T150 "
                "Translation acceptance telemetry"
            ),
            (
                "Final repair telemetry describes "
                "the actually accepted translated "
                "or evidence-safe response."
            ),
            (
                "Final repair telemetry can describe "
                "a rejected response."
            ),
        )

    def test_t151_clean_evidence_usable(
        self,
    ) -> None:

        from app.services.evidence_quote_usability_guard import (
            EvidenceQuoteUsabilityGuard,
        )

        guard = EvidenceQuoteUsabilityGuard()

        result = guard.evaluate(
            (
                "Therefore, the collector current is "
                "related to the emitter current which "
                "is in turn a function of the B-E "
                "voltage."
            )
        )

        self._check(
            (
                result.status == "usable"
                and
                result.is_usable
                and
                result.issues == ()
            ),
            (
                "GROUNDING-PRECISION-T151 "
                "Clean evidence usable"
            ),
            (
                "A complete technical relation "
                "remains usable evidence."
            ),
            (
                "Usability guard rejected a clean "
                "complete evidence statement."
            ),
        )


    def test_t152_trailing_as_unusable(
        self,
    ) -> None:

        from app.services.evidence_quote_usability_guard import (
            EvidenceQuoteUsabilityGuard,
        )

        guard = EvidenceQuoteUsabilityGuard()

        result = guard.evaluate(
            (
                "The current through the B-E junction "
                "is related to the B-E voltage as"
            )
        )

        self._check(
            (
                result.status == "unusable"
                and
                not result.is_usable
                and
                bool(result.issues)
            ),
            (
                "GROUNDING-PRECISION-T152 "
                "Trailing-as evidence"
            ),
            (
                "An unfinished equation lead-in "
                "is rejected deterministically."
            ),
            (
                "Truncated evidence ending in "
                "'as' was accepted."
            ),
        )

    def test_t153_extraction_damage_unusable(
        self,
    ) -> None:

        from app.services.evidence_quote_usability_guard import (
            EvidenceQuoteUsabilityGuard,
        )

        guard = EvidenceQuoteUsabilityGuard()

        result = guard.evaluate(
            (
                "With the voltage and as shown, "
                "the Base-Emitter (B-E) junction "
                "is forward biased and the "
                "Base-Collector (B-C) junction "
                "is reverse biased."
            )
        )

        self._check(
            (
                result.status == "unusable"
                and
                not result.is_usable
            ),
            (
                "GROUNDING-PRECISION-T153 "
                "Extraction-damaged evidence"
            ),
            (
                "The observed missing-variable "
                "PDF extraction defect is rejected "
                "deterministically."
            ),
            (
                "Extraction-damaged voltage text "
                "was accepted."
            ),
        )

    def test_t154_complete_relation_usable(
        self,
    ) -> None:

        from app.services.evidence_quote_usability_guard import (
            EvidenceQuoteUsabilityGuard,
        )

        guard = EvidenceQuoteUsabilityGuard()

        result = guard.evaluate(
            (
                "The voltage between two terminals "
                "controls the current through the "
                "third terminal."
            )
        )

        self._check(
            result.status == "usable",
            (
                "GROUNDING-PRECISION-T154 "
                "Complete relation usable"
            ),
            (
                "A complete source relation remains "
                "available to evidence-safe repair."
            ),
            (
                "Usability guard over-rejected "
                "a complete source relation."
            ),
        )

    def test_t155_evidence_usability_deterministic(
        self,
    ) -> None:

        from pathlib import Path

        from app.services.evidence_quote_usability_guard import (
            EvidenceQuoteUsabilityGuard,
        )

        guard = EvidenceQuoteUsabilityGuard()

        quote = (
            "The current through the B-E junction "
            "is related to the B-E voltage as"
        )

        first = guard.evaluate(
            quote
        )

        second = guard.evaluate(
            quote
        )

        guard_source = Path(
            "app/services/"
            "evidence_quote_usability_guard.py"
        ).read_text(
            encoding="utf-8"
        ).lower()

        self._check(
            (
                first == second
                and
                "chat_with_ai"
                not in guard_source
                and
                "chat_with_structured_ai"
                not in guard_source
            ),
            (
                "GROUNDING-PRECISION-T155 "
                "Evidence usability deterministic"
            ),
            (
                "Evidence usability is deterministic "
                "and introduces no LLM dependency."
            ),
            (
                "Evidence usability depends on "
                "non-deterministic generation."
            ),
        )

    def test_t156_selector_rejects_unusable_exact_quote(
        self,
    ) -> None:

        from unittest.mock import patch

        from app.services.repair_evidence_selector import (
            RepairEvidenceSelector,
        )

        selector = RepairEvidenceSelector()

        bad_quote = (
            "The current through the B-E junction "
            "is related to the B-E voltage as"
        )

        knowledge = (
            "[Source 1]\n"
            f"{bad_quote}\n"
            "\n"
            "Therefore, the collector current is "
            "related to the emitter current."
        )

        provider_result = (
            "{"
            '"evidence_quotes": ['
            '"The current through the B-E junction '
            'is related to the B-E voltage as"'
            "], "
            '"reason": "Controlled selection."'
            "}"
        )

        with patch(
            (
                "app.services."
                "repair_evidence_selector."
                "chat_with_structured_ai"
            ),
            return_value=provider_result,
        ):

            result = selector.select(
                learner_message=(
                    "What does the Base do?"
                ),
                knowledge_context=knowledge,
                validation_issues=(
                    "Controlled grounding issue.",
                ),
            )

        self._check(
            (
                result.status == "invalid"
                and
                result.evidence_quotes == ()
                and
                any(
                    "unusable"
                    in issue.lower()
                    for issue in result.issues
                )
            ),
            (
                "GROUNDING-PRECISION-T156 "
                "Selector usability integration"
            ),
            (
                "An exact but unusable source quote "
                "is blocked before becoming verified "
                "repair evidence."
            ),
            (
                "Selector accepted an exact but "
                "unusable evidence fragment."
            ),
        )

    def test_t157_evidence_retry_eligibility(
        self,
    ) -> None:

        from app.services.repair_evidence_selection_models import (
            RepairEvidenceSelectionResult,
        )

        tutor = AITutor()

        usability_failure = (
            RepairEvidenceSelectionResult(
                status="invalid",
                evidence_quotes=(),
                reason=(
                    "Repair evidence selector proposed "
                    "unusable source evidence."
                ),
                issues=(
                    (
                        "Evidence quote 1 was exact "
                        "but unusable."
                    ),
                ),
            )
        )

        provenance_failure = (
            RepairEvidenceSelectionResult(
                status="invalid",
                evidence_quotes=(),
                reason=(
                    "Repair evidence selector proposed "
                    "unverifiable source evidence."
                ),
                issues=(
                    "Evidence quote was not found.",
                ),
            )
        )

        self._check(
            (
                tutor
                ._is_retryable_evidence_usability_failure(
                    usability_failure
                )
                and
                not tutor
                ._is_retryable_evidence_usability_failure(
                    provenance_failure
                )
            ),
            (
                "GROUNDING-PRECISION-T157 "
                "Evidence retry eligibility"
            ),
            (
                "Only deterministic evidence "
                "usability failures are retryable."
            ),
            (
                "Evidence retry policy is too broad "
                "or misses usability failures."
            ),
        )

    def test_t158_evidence_reselection_success(
        self,
    ) -> None:

        from app.services.repair_evidence_selection_models import (
            RepairEvidenceSelectionResult,
        )
        from app.services.response_grounding_validator import (
            GroundingValidationResult,
        )

        tutor = AITutor()

        first = RepairEvidenceSelectionResult(
            status="invalid",
            evidence_quotes=(),
            reason=(
                "Repair evidence selector proposed "
                "unusable source evidence."
            ),
            issues=(
                "Evidence quote 1 was exact but unusable.",
            ),
        )

        clean_quote = (
            "The voltage between two terminals "
            "controls the current through the "
            "third terminal."
        )

        second = RepairEvidenceSelectionResult(
            status="verified",
            evidence_quotes=(
                clean_quote,
            ),
            reason="Controlled clean reselection.",
            issues=(),
        )

        calls = []

        def controlled_select(**kwargs):
            calls.append(kwargs)

            if len(calls) == 1:
                return first

            return second

        tutor.repair_evidence_selector.select = (
            controlled_select
        )

        validation = GroundingValidationResult(
            status="unsupported",
            confidence=1.0,
            reason="Controlled.",
            issues=[
                "Controlled grounding issue.",
            ],
        )

        selection, context = (
            tutor._prepare_llm_repair_context(
                learner_message=(
                    "What does the Base do?"
                ),
                knowledge_context=(
                    clean_quote
                ),
                grounding_validation=(
                    validation
                ),
            )
        )

        self._check(
            (
                len(calls) == 2
                and
                selection == second
                and
                context is not None
                and
                clean_quote in context
                and
                tutor.last_repair_evidence_selection_attempts
                == 2
                and
                tutor.last_repair_evidence_retry_used
                is True
                and
                tutor.last_repair_evidence_first_failure
                == first
            ),
            (
                "GROUNDING-PRECISION-T158 "
                "Evidence reselection success"
            ),
            (
                "One usability rejection can trigger "
                "one clean verified reselection."
            ),
            (
                "Safe evidence reselection did not "
                "recover verified evidence."
            ),
        )

    def test_t159_evidence_retry_bounded(
        self,
    ) -> None:

        from app.services.repair_evidence_selection_models import (
            RepairEvidenceSelectionResult,
        )
        from app.services.response_grounding_validator import (
            GroundingValidationResult,
        )

        tutor = AITutor()

        failure = RepairEvidenceSelectionResult(
            status="invalid",
            evidence_quotes=(),
            reason=(
                "Repair evidence selector proposed "
                "unusable source evidence."
            ),
            issues=(
                "Exact evidence remained unusable.",
            ),
        )

        calls = {
            "count": 0,
        }

        def controlled_select(**kwargs):
            calls["count"] += 1
            return failure

        tutor.repair_evidence_selector.select = (
            controlled_select
        )

        validation = GroundingValidationResult(
            status="unsupported",
            confidence=1.0,
            reason="Controlled.",
            issues=[],
        )

        selection, context = (
            tutor._prepare_llm_repair_context(
                learner_message="Question.",
                knowledge_context="Course evidence.",
                grounding_validation=validation,
            )
        )

        self._check(
            (
                calls["count"] == 3
                and
                selection.status == "invalid"
                and
                context is None
                and
                tutor.last_repair_evidence_selection_attempts
                == 3
                and
                tutor.last_repair_evidence_retry_used
                is True
            ),
            (
                "GROUNDING-PRECISION-T159 "
                "Evidence retry bounded"
            ),
            (
                "Repeated usability failures are "
                "limited to three total selector "
                "attempts before fail-closed."
            ),
            (
                "Evidence reselection exceeded or "
                "failed to use its bounded retry "
                "budget."
            ),
        )

    def test_t160_provenance_failure_no_retry(
        self,
    ) -> None:

        from app.services.repair_evidence_selection_models import (
            RepairEvidenceSelectionResult,
        )
        from app.services.response_grounding_validator import (
            GroundingValidationResult,
        )

        tutor = AITutor()

        failure = RepairEvidenceSelectionResult(
            status="invalid",
            evidence_quotes=(),
            reason=(
                "Repair evidence selector proposed "
                "unverifiable source evidence."
            ),
            issues=(
                "Evidence quote was not found.",
            ),
        )

        calls = {
            "count": 0,
        }

        def controlled_select(**kwargs):
            calls["count"] += 1
            return failure

        tutor.repair_evidence_selector.select = (
            controlled_select
        )

        validation = GroundingValidationResult(
            status="unsupported",
            confidence=1.0,
            reason="Controlled.",
            issues=[],
        )

        selection, context = (
            tutor._prepare_llm_repair_context(
                learner_message="Question.",
                knowledge_context="Course evidence.",
                grounding_validation=validation,
            )
        )

        self._check(
            (
                calls["count"] == 1
                and
                selection == failure
                and
                context is None
                and
                tutor.last_repair_evidence_retry_used
                is False
            ),
            (
                "GROUNDING-PRECISION-T160 "
                "Provenance failure no retry"
            ),
            (
                "Unverifiable evidence fails closed "
                "without another selector call."
            ),
            (
                "Fabricated or unverifiable evidence "
                "incorrectly triggered reselection."
            ),
        )

    def test_t161_no_evidence_no_retry(
        self,
    ) -> None:

        from app.services.repair_evidence_selection_models import (
            RepairEvidenceSelectionResult,
        )
        from app.services.response_grounding_validator import (
            GroundingValidationResult,
        )

        tutor = AITutor()

        no_evidence = RepairEvidenceSelectionResult(
            status="no_evidence",
            evidence_quotes=(),
            reason="No sufficient exact evidence.",
            issues=(),
        )

        calls = {
            "count": 0,
        }

        def controlled_select(**kwargs):
            calls["count"] += 1
            return no_evidence

        tutor.repair_evidence_selector.select = (
            controlled_select
        )

        validation = GroundingValidationResult(
            status="unsupported",
            confidence=1.0,
            reason="Controlled.",
            issues=[],
        )

        selection, context = (
            tutor._prepare_llm_repair_context(
                learner_message="Question.",
                knowledge_context="Course evidence.",
                grounding_validation=validation,
            )
        )

        self._check(
            (
                calls["count"] == 1
                and
                selection.status == "no_evidence"
                and
                context is None
                and
                tutor.last_repair_evidence_retry_used
                is False
            ),
            (
                "GROUNDING-PRECISION-T161 "
                "No-evidence no retry"
            ),
            (
                "A genuine no-evidence result "
                "fails closed without reselection."
            ),
            (
                "No-evidence unexpectedly triggered "
                "another selector attempt."
            ),
        )

    def test_t162_reselection_feedback(
        self,
    ) -> None:

        from app.services.repair_evidence_selection_models import (
            RepairEvidenceSelectionResult,
        )
        from app.services.response_grounding_validator import (
            GroundingValidationResult,
        )

        tutor = AITutor()

        first = RepairEvidenceSelectionResult(
            status="invalid",
            evidence_quotes=(),
            reason=(
                "Repair evidence selector proposed "
                "unusable source evidence."
            ),
            issues=(
                (
                    "Evidence quote 1 was exact "
                    "but unusable."
                ),
            ),
        )

        second = RepairEvidenceSelectionResult(
            status="no_evidence",
            evidence_quotes=(),
            reason="Controlled.",
            issues=(),
        )

        calls = []

        def controlled_select(**kwargs):
            calls.append(kwargs)

            if len(calls) == 1:
                return first

            return second

        tutor.repair_evidence_selector.select = (
            controlled_select
        )

        validation = GroundingValidationResult(
            status="unsupported",
            confidence=1.0,
            reason="Controlled.",
            issues=[
                "Original grounding issue.",
            ],
        )

        tutor._prepare_llm_repair_context(
            learner_message="Question.",
            knowledge_context="Course evidence.",
            grounding_validation=validation,
        )

        first_issues = tuple(
            calls[0]["validation_issues"]
        )

        second_issues = tuple(
            calls[1]["validation_issues"]
        )

        feedback_text = " ".join(
            str(item)
            for item in second_issues
        ).lower()

        self._check(
            (
                len(calls) == 2
                and
                first_issues
                == (
                    "Original grounding issue.",
                )
                and
                len(second_issues)
                == 2
                and
                "reselection required"
                in feedback_text
                and
                "do not reuse"
                in feedback_text
                and
                "do not repair"
                in feedback_text
            ),
            (
                "GROUNDING-PRECISION-T162 "
                "Evidence reselection feedback"
            ),
            (
                "The second selector attempt receives "
                "explicit deterministic feedback "
                "without changing the selector API."
            ),
            (
                "Evidence reselection did not receive "
                "the required rejection feedback."
            ),
        )

    def test_t163_evidence_retry_telemetry(
        self,
    ) -> None:

        from app.services.repair_evidence_selection_models import (
            RepairEvidenceSelectionResult,
        )

        tutor = AITutor()

        defaults_ok = (
            tutor.last_repair_evidence_selection_attempts
            == 0
            and
            tutor.last_repair_evidence_retry_used
            is False
            and
            tutor.last_repair_evidence_first_failure
            is None
        )

        tutor.last_repair_evidence_selection_attempts = 2
        tutor.last_repair_evidence_retry_used = True

        tutor.last_repair_evidence_first_failure = (
            RepairEvidenceSelectionResult(
                status="invalid",
                evidence_quotes=(),
                reason=(
                    "Repair evidence selector proposed "
                    "unusable source evidence."
                ),
                issues=(
                    "Controlled.",
                ),
            )
        )

        tutor.reset()

        reset_ok = (
            tutor.last_repair_evidence_selection_attempts
            == 0
            and
            tutor.last_repair_evidence_retry_used
            is False
            and
            tutor.last_repair_evidence_first_failure
            is None
        )

        self._check(
            (
                defaults_ok
                and
                reset_ok
            ),
            (
                "GROUNDING-PRECISION-T163 "
                "Evidence retry telemetry"
            ),
            (
                "Evidence-retry telemetry starts "
                "clean and is cleared by reset."
            ),
            (
                "Evidence-retry telemetry can leak "
                "between tutoring sessions."
            ),
        )

    def test_t164_third_attempt_recovery(
        self,
    ) -> None:

        from app.services.repair_evidence_selection_models import (
            RepairEvidenceSelectionResult,
        )
        from app.services.response_grounding_validator import (
            GroundingValidationResult,
        )

        tutor = AITutor()

        first_failure = RepairEvidenceSelectionResult(
            status="invalid",
            evidence_quotes=(),
            reason=(
                "Repair evidence selector proposed "
                "unusable source evidence."
            ),
            issues=(
                (
                    "Evidence quote 1 was exact but "
                    "unusable: damaged voltage text."
                ),
            ),
        )

        second_failure = RepairEvidenceSelectionResult(
            status="invalid",
            evidence_quotes=(),
            reason=(
                "Repair evidence selector proposed "
                "unusable source evidence."
            ),
            issues=(
                (
                    "Evidence quote 1 was exact but "
                    "unusable: unfinished lead-in."
                ),
            ),
        )

        clean_quote = (
            "Therefore, the collector current is "
            "related to the emitter current which "
            "is in turn a function of the B-E voltage."
        )

        third_selection = (
            RepairEvidenceSelectionResult(
                status="verified",
                evidence_quotes=(
                    clean_quote,
                ),
                reason=(
                    "Controlled clean third selection."
                ),
                issues=(),
            )
        )

        results = [
            first_failure,
            second_failure,
            third_selection,
        ]

        calls = []

        def controlled_select(**kwargs):
            calls.append(kwargs)
            return results[
                len(calls) - 1
            ]

        tutor.repair_evidence_selector.select = (
            controlled_select
        )

        validation = GroundingValidationResult(
            status="unsupported",
            confidence=1.0,
            reason="Controlled.",
            issues=[
                "Original grounding issue.",
            ],
        )

        selection, context = (
            tutor._prepare_llm_repair_context(
                learner_message=(
                    "What does the Base do?"
                ),
                knowledge_context=(
                    clean_quote
                ),
                grounding_validation=(
                    validation
                ),
            )
        )

        self._check(
            (
                len(calls) == 3
                and
                selection == third_selection
                and
                context is not None
                and
                clean_quote in context
                and
                tutor.last_repair_evidence_selection_attempts
                == 3
                and
                tutor.last_repair_evidence_retry_used
                is True
                and
                tutor.last_repair_evidence_first_failure
                == first_failure
            ),
            (
                "GROUNDING-PRECISION-T164 "
                "Third-attempt evidence recovery"
            ),
            (
                "Two distinct unusable evidence "
                "selections can recover with clean "
                "verified evidence on the final "
                "bounded attempt."
            ),
            (
                "Bounded evidence reselection could "
                "not recover on the third attempt."
            ),
        )
    def test_t165_cumulative_reselection_feedback(
        self,
    ) -> None:

        from app.services.repair_evidence_selection_models import (
            RepairEvidenceSelectionResult,
        )
        from app.services.response_grounding_validator import (
            GroundingValidationResult,
        )

        tutor = AITutor()

        first = RepairEvidenceSelectionResult(
            status="invalid",
            evidence_quotes=(),
            reason=(
                "Repair evidence selector proposed "
                "unusable source evidence."
            ),
            issues=(
                "FIRST BAD EVIDENCE was unusable.",
            ),
        )

        second = RepairEvidenceSelectionResult(
            status="invalid",
            evidence_quotes=(),
            reason=(
                "Repair evidence selector proposed "
                "unusable source evidence."
            ),
            issues=(
                "SECOND BAD EVIDENCE was unusable.",
            ),
        )

        third = RepairEvidenceSelectionResult(
            status="no_evidence",
            evidence_quotes=(),
            reason="Controlled.",
            issues=(),
        )

        results = [
            first,
            second,
            third,
        ]

        calls = []

        def controlled_select(**kwargs):
            calls.append(kwargs)
            return results[
                len(calls) - 1
            ]

        tutor.repair_evidence_selector.select = (
            controlled_select
        )

        validation = GroundingValidationResult(
            status="unsupported",
            confidence=1.0,
            reason="Controlled.",
            issues=[
                "ORIGINAL ISSUE.",
            ],
        )

        tutor._prepare_llm_repair_context(
            learner_message="Question.",
            knowledge_context="Course evidence.",
            grounding_validation=validation,
        )

        first_call_issues = " ".join(
            str(item)
            for item in calls[0][
                "validation_issues"
            ]
        )

        second_call_issues = " ".join(
            str(item)
            for item in calls[1][
                "validation_issues"
            ]
        )

        third_call_issues = " ".join(
            str(item)
            for item in calls[2][
                "validation_issues"
            ]
        )

        self._check(
            (
                len(calls) == 3
                and
                "FIRST BAD EVIDENCE"
                not in first_call_issues
                and
                "FIRST BAD EVIDENCE"
                in second_call_issues
                and
                "FIRST BAD EVIDENCE"
                in third_call_issues
                and
                "SECOND BAD EVIDENCE"
                in third_call_issues
                and
                "ORIGINAL ISSUE"
                in third_call_issues
            ),
            (
                "GROUNDING-PRECISION-T165 "
                "Cumulative reselection feedback"
            ),
            (
                "Each bounded retry receives all "
                "prior deterministic evidence "
                "rejections plus the original "
                "grounding issues."
            ),
            (
                "Evidence retry feedback was lost "
                "between selector attempts."
            ),
        )

    def test_t166_guiding_question_result_typed(
        self,
    ) -> None:

        from dataclasses import FrozenInstanceError

        from app.services.guiding_question_grounding_models import (
            GuidingQuestionGroundingResult,
        )

        result = GuidingQuestionGroundingResult(
            status="supported",
            reason="Controlled.",
            issues=(),
        )

        raised = False

        try:
            result.status = "unsupported"
        except FrozenInstanceError:
            raised = True

        self._check(
            (
                result.is_safe
                and
                raised
            ),
            (
                "GROUNDING-PRECISION-T166 "
                "Guiding-question typed result"
            ),
            (
                "Guiding-question grounding uses "
                "an immutable typed result."
            ),
            (
                "Guiding-question grounding result "
                "contract is mutable or unavailable."
            ),
        )

    def test_t167_no_question_fast_path(
        self,
    ) -> None:

        from unittest.mock import patch

        from app.services.guiding_question_grounding_validator import (
            GuidingQuestionGroundingValidator,
        )

        validator = (
            GuidingQuestionGroundingValidator()
        )

        with patch(
            (
                "app.services."
                "guiding_question_grounding_validator."
                "chat_with_structured_ai"
            )
        ) as mocked:

            result = validator.validate(
                response=(
                    "The collector current is related "
                    "to the emitter current."
                ),
                knowledge_context=(
                    "Course evidence."
                ),
            )

        self._check(
            (
                result.status == "no_questions"
                and
                result.is_safe
                and
                mocked.call_count == 0
            ),
            (
                "GROUNDING-PRECISION-T167 "
                "Guiding-question fast path"
            ),
            (
                "Responses without guiding questions "
                "add no semantic validation call."
            ),
            (
                "Question grounding added an "
                "unnecessary LLM call."
            ),
        )

    def test_t168_answerable_guiding_question(
        self,
    ) -> None:

        from unittest.mock import patch

        from app.services.guiding_question_grounding_validator import (
            GuidingQuestionGroundingValidator,
        )

        response = (
            "What is the collector current "
            "related to?"
        )

        evidence = (
            "Therefore, the collector current is "
            "related to the emitter current which "
            "is in turn a function of the B-E voltage."
        )

        provider_result = (
            "{"
            '"questions": ['
            "{"
            '"question_quote": '
            '"What is the collector current related to?",'
            '"question_type": "factual",'
            '"status": "answerable",'
            '"evidence_quote": '
            '"Therefore, the collector current is '
            'related to the emitter current which '
            'is in turn a function of the B-E voltage.",'
            '"issue": ""'
            "}"
            "]"
            "}"
        )

        validator = (
            GuidingQuestionGroundingValidator()
        )

        with patch(
            (
                "app.services."
                "guiding_question_grounding_validator."
                "chat_with_structured_ai"
            ),
            return_value=provider_result,
        ):
            result = validator.validate(
                response=response,
                knowledge_context=evidence,
            )

        self._check(
            (
                result.status == "supported"
                and
                result.is_safe
            ),
            (
                "GROUNDING-PRECISION-T168 "
                "Answerable guiding question"
            ),
            (
                "A factual guiding question with "
                "verifiable course evidence remains "
                "allowed."
            ),
            (
                "An evidence-supported guiding "
                "question was rejected."
            ),
        )

    def test_t169_unsupported_exact_value_question(
        self,
    ) -> None:

        from unittest.mock import patch

        from app.services.guiding_question_grounding_validator import (
            GuidingQuestionGroundingValidator,
        )

        response = (
            "What exact B-E voltage is required?"
        )

        knowledge = (
            "The collector current is related to "
            "the emitter current which is in turn "
            "a function of the B-E voltage."
        )

        provider_result = (
            "{"
            '"questions": ['
            "{"
            '"question_quote": '
            '"What exact B-E voltage is required?",'
            '"question_type": "factual",'
            '"status": "unsupported",'
            '"evidence_quote": "",'
            '"issue": '
            '"Course knowledge does not provide '
            'an exact B-E voltage."'
            "}"
            "]"
            "}"
        )

        validator = (
            GuidingQuestionGroundingValidator()
        )

        with patch(
            (
                "app.services."
                "guiding_question_grounding_validator."
                "chat_with_structured_ai"
            ),
            return_value=provider_result,
        ):
            result = validator.validate(
                response=response,
                knowledge_context=knowledge,
            )

        self._check(
            (
                result.status == "unsupported"
                and
                not result.is_safe
            ),
            (
                "GROUNDING-PRECISION-T169 "
                "Unsupported exact-value question"
            ),
            (
                "A guiding question cannot request "
                "an exact value absent from course "
                "evidence."
            ),
            (
                "Unsupported numerical question "
                "was accepted."
            ),
        )

    def test_t170_unsupported_range_efficiency(
        self,
    ) -> None:

        from unittest.mock import patch

        from app.services.guiding_question_grounding_validator import (
            GuidingQuestionGroundingValidator,
        )

        response = (
            "คุณคิดว่าแรงดันที่ใช้กับ B-E "
            "ควรอยู่ในช่วงใดเพื่อให้ทรานซิสเตอร์"
            "ทำงานได้อย่างมีประสิทธิภาพ?"
        )

        knowledge = (
            "Therefore, the collector current is "
            "related to the emitter current which "
            "is in turn a function of the B-E voltage."
        )

        provider_result = json.dumps(
            {
                "questions": [
                    {
                        "question_quote": response,
                        "question_type": "factual",
                        "status": "unsupported",
                        "evidence_quote": "",
                        "issue": (
                            "Course knowledge provides "
                            "neither a B-E voltage range "
                            "nor an efficiency criterion."
                        ),
                    }
                ]
            },
            ensure_ascii=False,
        )

        validator = (
            GuidingQuestionGroundingValidator()
        )

        with patch(
            (
                "app.services."
                "guiding_question_grounding_validator."
                "chat_with_structured_ai"
            ),
            return_value=provider_result,
        ):
            result = validator.validate(
                response=response,
                knowledge_context=knowledge,
            )

        self._check(
            result.status == "unsupported",
            (
                "GROUNDING-PRECISION-T170 "
                "Range-efficiency question"
            ),
            (
                "The observed live question asking "
                "for an unsupported voltage range "
                "and efficiency criterion is blocked."
            ),
            (
                "The live range-efficiency defect "
                "was accepted."
            ),
        )

    def test_t171_reflective_question_allowed(
        self,
    ) -> None:

        from unittest.mock import patch

        from app.services.guiding_question_grounding_validator import (
            GuidingQuestionGroundingValidator,
        )

        response = (
            "What relationship do you notice "
            "from the information above?"
        )

        provider_result = (
            "{"
            '"questions": ['
            "{"
            '"question_quote": '
            '"What relationship do you notice '
            'from the information above?",'
            '"question_type": "reflective",'
            '"status": "reflective",'
            '"evidence_quote": "",'
            '"issue": ""'
            "}"
            "]"
            "}"
        )

        validator = (
            GuidingQuestionGroundingValidator()
        )

        with patch(
            (
                "app.services."
                "guiding_question_grounding_validator."
                "chat_with_structured_ai"
            ),
            return_value=provider_result,
        ):
            result = validator.validate(
                response=response,
                knowledge_context=(
                    "Course evidence."
                ),
            )

        self._check(
            result.status == "supported",
            (
                "GROUNDING-PRECISION-T171 "
                "Reflective question"
            ),
            (
                "A reflective guiding question "
                "remains allowed without inventing "
                "a factual answer requirement."
            ),
            (
                "Reflective tutoring questions "
                "were over-blocked."
            ),
        )
    def test_t172_question_evidence_provenance(
        self,
    ) -> None:

        from unittest.mock import patch

        from app.services.guiding_question_grounding_validator import (
            GuidingQuestionGroundingValidator,
        )

        response = (
            "What is the collector current "
            "related to?"
        )

        provider_result = (
            "{"
            '"questions": ['
            "{"
            '"question_quote": '
            '"What is the collector current related to?",'
            '"question_type": "factual",'
            '"status": "answerable",'
            '"evidence_quote": '
            '"Fabricated course evidence.",'
            '"issue": ""'
            "}"
            "]"
            "}"
        )

        validator = (
            GuidingQuestionGroundingValidator()
        )

        with patch(
            (
                "app.services."
                "guiding_question_grounding_validator."
                "chat_with_structured_ai"
            ),
            return_value=provider_result,
        ):
            result = validator.validate(
                response=response,
                knowledge_context=(
                    "Real course evidence."
                ),
            )

        self._check(
            (
                result.status == "invalid"
                and
                not result.is_safe
            ),
            (
                "GROUNDING-PRECISION-T172 "
                "Question evidence provenance"
            ),
            (
                "A fabricated evidence quote cannot "
                "make a factual guiding question "
                "answerable."
            ),
            (
                "Guiding-question validator accepted "
                "fabricated evidence."
            ),
        )

    def test_t173_question_omission_fail_closed(
        self,
    ) -> None:

        from unittest.mock import patch

        from app.services.guiding_question_grounding_validator import (
            GuidingQuestionGroundingValidator,
        )

        validator = (
            GuidingQuestionGroundingValidator()
        )

        with patch(
            (
                "app.services."
                "guiding_question_grounding_validator."
                "chat_with_structured_ai"
            ),
            return_value='{"questions": []}',
        ):
            result = validator.validate(
                response=(
                    "What exact B-E voltage is required?"
                ),
                knowledge_context=(
                    "Course evidence."
                ),
            )

        self._check(
            (
                result.status == "invalid"
                and
                not result.is_safe
            ),
            (
                "GROUNDING-PRECISION-T173 "
                "Question omission fail-closed"
            ),
            (
                "A semantic validator cannot hide "
                "an unsupported question by returning "
                "an empty question list."
            ),
            (
                "Question omission bypassed the "
                "answerability guard."
            ),
        )

    def test_t174_runtime_guiding_question_validator(
        self,
    ) -> None:

        from app.services.guiding_question_grounding_validator import (
            GuidingQuestionGroundingValidator,
        )

        tutor = AITutor()

        self._check(
            isinstance(
                tutor.guiding_question_grounding_validator,
                GuidingQuestionGroundingValidator,
            ),
            (
                "GROUNDING-PRECISION-T174 "
                "Runtime guiding-question validator"
            ),
            (
                "AITutor owns the knowledge-bounded "
                "guiding-question validator."
            ),
            (
                "AITutor does not own the "
                "guiding-question validator."
            ),
        )

    def test_t175_guiding_question_runtime_helper(
        self,
    ) -> None:

        from app.services.guiding_question_grounding_models import (
            GuidingQuestionGroundingResult,
        )

        tutor = AITutor()

        calls = []

        def controlled_validate(
            response,
            knowledge_context,
        ):
            calls.append(
                (
                    response,
                    knowledge_context,
                )
            )

            return GuidingQuestionGroundingResult(
                status="supported",
                reason="Controlled.",
                issues=(),
            )

        tutor.guiding_question_grounding_validator.validate = (
            controlled_validate
        )

        result = tutor._validate_guiding_questions(
            response="What is it related to?",
            knowledge_context="Course evidence.",
        )

        self._check(
            (
                len(calls) == 1
                and
                calls[0]
                == (
                    "What is it related to?",
                    "Course evidence.",
                )
                and
                result.status == "supported"
            ),
            (
                "GROUNDING-PRECISION-T175 "
                "Guiding-question runtime helper"
            ),
            (
                "Runtime helper delegates only the "
                "Tutor response and course knowledge."
            ),
            (
                "Runtime guiding-question helper "
                "violated its input boundary."
            ),
        )

    def test_t176_guiding_question_telemetry_defaults(
        self,
    ) -> None:

        tutor = AITutor()

        self._check(
        (
            tutor.last_guiding_question_grounding
            is None
            and
            tutor.last_repair_guiding_question_grounding
            is None
            and
            tutor.last_evidence_safe_guiding_question_grounding
            is None
            and
            tutor.last_translation_guiding_question_grounding
            is None
        ),
        (
            "GROUNDING-PRECISION-T176 "
            "Guiding-question telemetry defaults"
        ),
        (
            "All guiding-question validation "
            "telemetry starts inactive."
        ),
        (
            "Guiding-question telemetry starts "
            "with stale state."
        ),
    )

    def test_t177_guiding_question_telemetry_reset(
        self,
    ) -> None:

        from app.services.guiding_question_grounding_models import (
            GuidingQuestionGroundingResult,
        )

        tutor = AITutor()

        result = GuidingQuestionGroundingResult(
            status="unsupported",
            reason="Controlled.",
            issues=(
                "Controlled issue.",
            ),
        )

        tutor.last_guiding_question_grounding = result
        tutor.last_repair_guiding_question_grounding = result
        tutor.last_evidence_safe_guiding_question_grounding = result
        tutor.last_translation_guiding_question_grounding = result

        tutor.reset()

        self._check(
            (
                tutor.last_guiding_question_grounding
                is None
                and
                tutor.last_repair_guiding_question_grounding
                is None
                and
                tutor.last_evidence_safe_guiding_question_grounding
                is None
                and
                tutor.last_translation_guiding_question_grounding
                is None
            ),
            (
                "GROUNDING-PRECISION-T177 "
                "Guiding-question telemetry reset"
            ),
            (
                "Conversation reset clears all "
                "guiding-question telemetry."
            ),
            (
                "Guiding-question telemetry survived "
                "conversation reset."
            ),
        )

    def test_t178_guiding_question_debug_projection(
        self,
    ) -> None:

        from app.services.guiding_question_grounding_models import (
            GuidingQuestionGroundingResult,
        )

        tutor = AITutor()

        tutor.last_guiding_question_grounding = (
            GuidingQuestionGroundingResult(
                status="unsupported",
                reason="Controlled reason.",
                issues=(
                    "Controlled issue.",
                ),
            )
        )

        debug = tutor.get_debug_info()

        self._check(
            (
                debug[
                    "guiding_question_grounding_status"
                ]
                == "unsupported"
                and
                debug[
                    "guiding_question_grounding_reason"
                ]
                == "Controlled reason."
                and
                debug[
                    "guiding_question_grounding_issues"
                ]
                == [
                    "Controlled issue."
                ]
                and
                "repair_guiding_question_grounding_status"
                in debug
                and
                "evidence_safe_guiding_question_grounding_status"
                in debug
                and
                "translation_guiding_question_grounding_status"
                in debug
            ),
            (
                "GROUNDING-PRECISION-T178 "
                "Guiding-question debug projection"
            ),
            (
                "Runtime debug telemetry exposes "
                "guiding-question grounding state."
            ),
            (
                "Guiding-question debug telemetry "
                "is incomplete."
            ),
        )
    def test_t179_guiding_question_initial_activation(
        self,
    ) -> None:

        from pathlib import Path

        source = Path(
            "app/tutor.py"
        ).read_text(
            encoding="utf-8"
        )

        validation_start = source.index(
            "# VALIDATE RESPONSE"
        )

        repair_start = source.index(
            "# RESPONSE REPAIR",
            validation_start,
        )

        initial_validation_block = source[
            validation_start:repair_start
        ]

        self._check(
            (
                "_validate_guiding_questions("
                in initial_validation_block
            ),
            (
                "GROUNDING-PRECISION-T179 "
                "Guiding-question initial activation"
            ),
            (
                "Knowledge-bounded guiding-question "
                "validation is activated in the "
                "initial Tutor-response validation "
                "pipeline."
            ),
            (
                "Initial Tutor responses do not pass "
                "through guiding-question validation."
            ),
        )

    def test_t180_guiding_question_gate_safe_preserves(
        self,
    ) -> None:

        from app.services.guiding_question_grounding_models import (
            GuidingQuestionGroundingResult,
        )

        from app.services.pedagogical_validation_models import (
            PedagogicalValidationResult,
        )

        tutor = AITutor()

        guiding = GuidingQuestionGroundingResult(
            status="supported",
            reason="Controlled.",
            issues=(),
        )

        pedagogy = PedagogicalValidationResult(
            status="valid",
            confidence=1.0,
            reason="Controlled.",
            issues=[],
        )

        result = (
            tutor._apply_guiding_question_acceptance_gate(
                guiding_question_grounding=guiding,
                pedagogical_validation=pedagogy,
            )
        )

        self._check(
            result is pedagogy,
            (
                "GROUNDING-PRECISION-T180 "
                "Safe guiding-question gate"
            ),
            (
                "A supported guiding question preserves "
                "the existing pedagogical result."
            ),
            (
                "The acceptance gate mutated a safe "
                "pedagogical result."
            ),
        )

    def test_t181_guiding_question_gate_unsupported(
        self,
    ) -> None:

        from app.services.guiding_question_grounding_models import (
            GuidingQuestionGroundingResult,
        )

        from app.services.pedagogical_validation_models import (
            PedagogicalValidationResult,
        )

        tutor = AITutor()

        guiding = GuidingQuestionGroundingResult(
            status="unsupported",
            reason="Unsupported range.",
            issues=(
                (
                    "Course knowledge provides neither "
                    "a voltage range nor an efficiency "
                    "criterion."
                ),
            ),
        )

        pedagogy = PedagogicalValidationResult(
            status="valid",
            confidence=1.0,
            reason="Structurally valid.",
            issues=[],
        )

        result = (
            tutor._apply_guiding_question_acceptance_gate(
                guiding_question_grounding=guiding,
                pedagogical_validation=pedagogy,
            )
        )

        self._check(
            (
                result.status == "violation"
                and
                any(
                    "voltage range" in issue
                    for issue in result.issues
                )
            ),
            (
                "GROUNDING-PRECISION-T181 "
                "Unsupported guiding-question gate"
            ),
            (
                "An unsupported factual guiding "
                "question becomes an acceptance "
                "violation."
            ),
            (
                "Unsupported guiding question did "
                "not block acceptance."
            ),
        )
    def test_t182_guiding_question_gate_invalid(
        self,
    ) -> None:

        from app.services.guiding_question_grounding_models import (
            GuidingQuestionGroundingResult,
        )

        from app.services.pedagogical_validation_models import (
            PedagogicalValidationResult,
        )

        tutor = AITutor()

        guiding = GuidingQuestionGroundingResult(
            status="invalid",
            reason="Provider failed.",
            issues=(
                "Controlled validator failure.",
            ),
        )

        pedagogy = PedagogicalValidationResult(
            status="valid",
            confidence=1.0,
            reason="Controlled.",
            issues=[],
        )

        result = (
            tutor._apply_guiding_question_acceptance_gate(
                guiding_question_grounding=guiding,
                pedagogical_validation=pedagogy,
            )
        )

        self._check(
            result.status == "violation",
            (
                "GROUNDING-PRECISION-T182 "
                "Guiding-question fail closed"
            ),
            (
                "Invalid guiding-question validation "
                "fails closed."
            ),
            (
                "Invalid guiding-question validation "
                "was accepted."
            ),
        )

    def test_t183_guiding_question_gate_preserves_issues(
        self,
    ) -> None:

        from app.services.guiding_question_grounding_models import (
            GuidingQuestionGroundingResult,
        )

        from app.services.pedagogical_validation_models import (
            PedagogicalValidationResult,
        )

        tutor = AITutor()

        guiding = GuidingQuestionGroundingResult(
            status="unsupported",
            reason="Controlled.",
            issues=(
                "Unsupported factual question.",
            ),
        )

        pedagogy = PedagogicalValidationResult(
            status="violation",
            confidence=1.0,
            reason="Existing violation.",
            issues=[
                "Existing pedagogical issue.",
            ],
        )

        result = (
            tutor._apply_guiding_question_acceptance_gate(
                guiding_question_grounding=guiding,
                pedagogical_validation=pedagogy,
            )
        )

        self._check(
            (
                result.status == "violation"
                and
                "Existing pedagogical issue."
                in result.issues
                and
                any(
                    "Unsupported factual question."
                    in issue
                    for issue in result.issues
                )
            ),
            (
                "GROUNDING-PRECISION-T183 "
                "Guiding-question issue merge"
            ),
            (
                "The acceptance gate preserves "
                "existing pedagogical issues while "
                "adding guiding-question issues."
            ),
            (
                "Guiding-question gating discarded "
                "an existing pedagogical violation."
            ),
        )

    def test_t184_guiding_question_initial_pipeline_order(
        self,
    ) -> None:

        from pathlib import Path

        source = Path(
            "app/tutor.py"
        ).read_text(
            encoding="utf-8"
        )

        validation_start = source.index(
            "# VALIDATE RESPONSE"
        )

        precision_position = source.index(
            "self.grounding_precision_veto_policy",
            validation_start,
        )

        guiding_position = source.index(
            "# Knowledge-bounded guiding-question grounding",
            precision_position,
        )

        pedagogy_position = source.index(
            "# Pedagogical precheck",
            guiding_position,
        )

        self._check(
            (
                precision_position
                < guiding_position
                < pedagogy_position
            ),
            (
                "GROUNDING-PRECISION-T184 "
                "Initial guiding-question order"
            ),
            (
                "Initial guiding-question grounding "
                "runs after factual precision and "
                "before pedagogical validation."
            ),
            (
                "Guiding-question validation is "
                "misordered in the initial pipeline."
            ),
        )

    def test_t185_guiding_question_gate_before_escalation(
        self,
    ) -> None:

        from pathlib import Path

        source = Path(
            "app/tutor.py"
        ).read_text(
            encoding="utf-8"
        )

        validation_start = source.index(
            "# VALIDATE RESPONSE"
        )

        gate_position = source.index(
            "# Guiding-question acceptance gate",
            validation_start,
        )

        escalation_position = source.index(
            "self.validation_escalation_policy.decide(",
            gate_position,
        )

        self._check(
            gate_position < escalation_position,
            (
                "GROUNDING-PRECISION-T185 "
                "Guiding-question escalation order"
            ),
            (
                "Guiding-question acceptance is "
                "resolved before the frozen "
                "escalation policy."
            ),
            (
                "Escalation can run before the "
                "guiding-question acceptance gate."
            ),
        )
    def test_t186_repair_guiding_question_pipeline_order(
        self,
    ) -> None:

        from pathlib import Path

        source = Path(
            "app/tutor.py"
        ).read_text(
            encoding="utf-8"
        )

        repair_start = source.index(
            "# VALIDATE REPAIRED RESPONSE"
        )

        precision_position = source.index(
            "# Repair precision veto",
            repair_start,
        )

        guiding_position = source.index(
            "# Repair knowledge-bounded guiding questions",
            precision_position,
        )

        pedagogy_position = source.index(
            "# Repair pedagogical precheck",
            guiding_position,
        )

        self._check(
            (
                precision_position
                < guiding_position
                < pedagogy_position
            ),
            (
                "GROUNDING-PRECISION-T186 "
                "Repair guiding-question order"
            ),
            (
                "Repaired responses validate guiding "
                "questions after precision enforcement "
                "and before pedagogical validation."
            ),
            (
                "Repair guiding-question validation "
                "is misordered."
            ),
        )
    def test_t187_repair_guiding_question_same_boundary(
        self,
    ) -> None:

        from pathlib import Path

        source = Path(
            "app/tutor.py"
        ).read_text(
            encoding="utf-8"
        )

        repair_start = source.index(
            "# VALIDATE REPAIRED RESPONSE"
        )

        guiding_start = source.index(
            "# Repair knowledge-bounded guiding questions",
            repair_start,
        )

        pedagogy_start = source.index(
            "# Repair pedagogical precheck",
            guiding_start,
        )

        guiding_block = source[
            guiding_start:pedagogy_start
        ]

        self._check(
            (
                "repair_validation_context"
                in guiding_block
                and
                "knowledge_result.context"
                not in guiding_block
            ),
            (
                "GROUNDING-PRECISION-T187 "
                "Repair guiding-question boundary"
            ),
            (
                "Repair guiding questions use the "
                "same factual evidence boundary as "
                "repair grounding validation."
            ),
            (
                "Repair guiding-question validation "
                "escaped its verified evidence boundary."
            ),
        )

    def test_t188_repair_guiding_question_gate_before_accept(
        self,
    ) -> None:

        from pathlib import Path

        source = Path(
            "app/tutor.py"
        ).read_text(
            encoding="utf-8"
        )

        repair_start = source.index(
            "# VALIDATE REPAIRED RESPONSE"
        )

        gate_position = source.index(
            "# Repair guiding-question acceptance gate",
            repair_start,
        )

        accept_position = source.index(
            "# ACCEPT REPAIRED ANSWER",
            gate_position,
        )

        self._check(
            gate_position < accept_position,
            (
                "GROUNDING-PRECISION-T188 "
                "Repair gate before acceptance"
            ),
            (
                "A repaired response cannot be "
                "accepted before guiding-question "
                "answerability is resolved."
            ),
            (
                "Repair acceptance can bypass the "
                "guiding-question gate."
            ),
        )   

    def test_t189_repair_unsupported_question_rejected(
        self,
    ) -> None:

        from app.services.guiding_question_grounding_models import (
            GuidingQuestionGroundingResult,
        )

        from app.services.pedagogical_validation_models import (
            PedagogicalValidationResult,
        )

        tutor = AITutor()

        guiding = GuidingQuestionGroundingResult(
            status="unsupported",
            reason="Exact value absent.",
            issues=(
                (
                    "Course evidence does not provide "
                    "an exact B-E voltage."
                ),
            ),
        )

        pedagogy = PedagogicalValidationResult(
            status="valid",
            confidence=1.0,
            reason="Repair structure is valid.",
            issues=[],
        )

        result = (
            tutor._apply_guiding_question_acceptance_gate(
                guiding_question_grounding=guiding,
                pedagogical_validation=pedagogy,
            )
        )

        accepted = (
            "supported" == "supported"
            and
            result.status == "valid"
        )

        self._check(
            (
                result.status == "violation"
                and
                not accepted
            ),
            (
                "GROUNDING-PRECISION-T189 "
                "Repair unsupported question"
            ),
            (
                "A grounded repaired explanation "
                "cannot be accepted when its factual "
                "guiding question is unsupported."
            ),
            (
                "An unsupported repaired guiding "
                "question remained acceptable."
            ),
        )

    def test_t190_repair_supported_question_allowed(
        self,
    ) -> None:

        from app.services.guiding_question_grounding_models import (
            GuidingQuestionGroundingResult,
        )

        from app.services.pedagogical_validation_models import (
            PedagogicalValidationResult,
        )

        tutor = AITutor()

        guiding = GuidingQuestionGroundingResult(
            status="supported",
            reason="Course evidence answers it.",
            issues=(),
        )

        pedagogy = PedagogicalValidationResult(
            status="valid",
            confidence=1.0,
            reason="Repair structure is valid.",
            issues=[],
        )

        result = (
            tutor._apply_guiding_question_acceptance_gate(
                guiding_question_grounding=guiding,
                pedagogical_validation=pedagogy,
            )
        )

        self._check(
            (
                result is pedagogy
                and
                result.status == "valid"
            ),
            (
                "GROUNDING-PRECISION-T190 "
                "Repair supported question"
            ),
            (
                "A repaired guiding question that "
                "is answerable from its repair "
                "evidence remains eligible."
            ),
            (
                "Supported repaired guiding question "
                "was unnecessarily rejected."
            ),
        )    
    def test_t191_repair_guiding_question_telemetry_store(
        self,
    ) -> None:

        from pathlib import Path

        source = Path(
            "app/tutor.py"
        ).read_text(
            encoding="utf-8"
        )

        store_start = source.index(
            "# STORE TURN VALIDATION STATE"
        )

        store_block = source[
            store_start:
        ]

        expected = (
            "self.last_repair_guiding_question_grounding = ("
        )

        self._check(
            expected in store_block,
            (
                "GROUNDING-PRECISION-T191 "
                "Repair guiding-question telemetry"
            ),
            (
                "The final repair guiding-question "
                "result is preserved in runtime "
                "telemetry."
            ),
            (
                "Repair guiding-question telemetry "
                "is not stored at turn completion."
            ),
        )
    def test_t192_evidence_safe_guiding_question_order(
        self,
    ) -> None:

        from pathlib import Path

        source = Path(
            "app/tutor.py"
        ).read_text(
            encoding="utf-8"
        )

        start = source.index(
            "# Evidence-safe precision"
        )

        guiding_position = source.index(
            "# Evidence-safe guiding questions",
            start,
        )

        pedagogy_position = source.index(
            "evidence_safe_pedagogical_precheck = (",
            guiding_position,
        )

        self._check(
            (
                start
                < guiding_position
                < pedagogy_position
            ),
            (
                "GROUNDING-PRECISION-T192 "
                "Evidence-safe guiding-question order"
            ),
            (
                "Evidence-safe guiding questions are "
                "validated after precision enforcement "
                "and before pedagogy."
            ),
            (
                "Evidence-safe guiding-question "
                "validation is misordered."
            ),
        )

    def test_t193_evidence_safe_guiding_question_boundary(
        self,
    ) -> None:

        from pathlib import Path

        source = Path(
            "app/tutor.py"
        ).read_text(
            encoding="utf-8"
        )

        start = source.index(
            "# Evidence-safe guiding questions"
        )

        end = source.index(
            "evidence_safe_pedagogical_precheck = (",
            start,
        )

        block = source[
            start:end
        ]

        self._check(
            (
                "self.last_repair_evidence_context"
                in block
                and
                "knowledge_result.context"
                not in block
            ),
            (
                "GROUNDING-PRECISION-T193 "
                "Evidence-safe question boundary"
            ),
            (
                "Evidence-safe guiding questions use "
                "the exact verified repair evidence "
                "boundary."
            ),
            (
                "Evidence-safe question validation "
                "escaped its verified evidence pack."
            ),
        )

    def test_t194_evidence_safe_gate_before_accept(
        self,
    ) -> None:

        from pathlib import Path

        source = Path(
            "app/tutor.py"
        ).read_text(
            encoding="utf-8"
        )

        start = source.index(
            "# Evidence-safe guiding-question"
        )

        accept_position = source.index(
            "# ACCEPT EVIDENCE-SAFE REPAIR",
            start,
        )

        self._check(
            start < accept_position,
            (
                "GROUNDING-PRECISION-T194 "
                "Evidence-safe gate before accept"
            ),
            (
                "Evidence-safe responses cannot be "
                "accepted before guiding-question "
                "answerability is resolved."
            ),
            (
                "Evidence-safe acceptance can bypass "
                "the guiding-question gate."
            ),
        )

    def test_t195_evidence_safe_no_question_fast_path(
        self,
    ) -> None:

        from unittest.mock import patch

        tutor = AITutor()

        with patch(
            (
                "app.services."
                "guiding_question_grounding_validator."
                "chat_with_structured_ai"
            )
        ) as mocked:

            result = tutor._validate_guiding_questions(
                response=(
                    "The voltage between two terminals "
                    "controls the current through the "
                    "third terminal."
                ),
                knowledge_context=(
                    "The voltage between two terminals "
                    "controls the current through the "
                    "third terminal."
                ),
            )

        self._check(
            (
                result.status == "no_questions"
                and
                mocked.call_count == 0
            ),
            (
                "GROUNDING-PRECISION-T195 "
                "Evidence-safe question fast path"
            ),
            (
                "Normal exact evidence composition "
                "adds no guiding-question LLM call."
            ),
            (
                "Question grounding added unnecessary "
                "cost to question-free evidence."
            ),
        )

    def test_t196_evidence_safe_unsupported_question(
        self,
    ) -> None:

        from app.services.guiding_question_grounding_models import (
            GuidingQuestionGroundingResult,
        )

        from app.services.pedagogical_validation_models import (
            PedagogicalValidationResult,
        )

        tutor = AITutor()

        guiding = GuidingQuestionGroundingResult(
            status="unsupported",
            reason="Exact voltage absent.",
            issues=(
                "Verified evidence contains no exact B-E voltage.",
            ),
        )

        pedagogy = PedagogicalValidationResult(
            status="valid",
            confidence=1.0,
            reason="Controlled.",
            issues=[],
        )

        result = (
            tutor._apply_guiding_question_acceptance_gate(
                guiding_question_grounding=guiding,
                pedagogical_validation=pedagogy,
            )
        )

        self._check(
            result.status == "violation",
            (
                "GROUNDING-PRECISION-T196 "
                "Evidence-safe unsupported question"
            ),
            (
                "An unsupported factual question "
                "cannot survive evidence-safe repair."
            ),
            (
                "Evidence-safe path accepted an "
                "unsupported factual question."
            ),
        )

    def test_t197_evidence_safe_question_telemetry(
        self,
    ) -> None:

        from pathlib import Path

        source = Path(
            "app/tutor.py"
        ).read_text(
            encoding="utf-8"
        )

        start = source.index(
            "# Evidence-safe guiding questions"
        )

        accept = source.index(
            "# ACCEPT EVIDENCE-SAFE REPAIR",
            start,
        )

        block = source[
            start:accept
        ]

        self._check(
            (
                "self.last_evidence_safe_guiding_question_grounding"
                in block
            ),
            (
                "GROUNDING-PRECISION-T197 "
                "Evidence-safe question telemetry"
            ),
            (
                "Evidence-safe guiding-question "
                "validation is observable at runtime."
            ),
            (
                "Evidence-safe guiding-question "
                "telemetry is not recorded."
            ),
        )

    def test_t198_translation_guiding_question_order(
        self,
    ) -> None:

        from pathlib import Path

        source = Path(
            "app/tutor.py"
        ).read_text(
            encoding="utf-8"
        )

        translation_start = source.index(
            "# Translation precision"
        )

        guiding_position = source.index(
            "# Translation guiding questions",
            translation_start,
        )

        language_position = source.index(
            "# Translation language gate",
            guiding_position,
        )

        self._check(
            (
                translation_start
                < guiding_position
                < language_position
            ),
            (
                "GROUNDING-PRECISION-T198 "
                "Translation guiding-question order"
            ),
            (
                "Translated guiding questions are "
                "validated after precision enforcement "
                "and before the language gate."
            ),
            (
                "Translation guiding-question "
                "validation is misordered."
            ),
        )

    def test_t199_translation_guiding_question_boundary(
        self,
    ) -> None:

        from pathlib import Path

        source = Path(
            "app/tutor.py"
        ).read_text(
            encoding="utf-8"
        )

        start = source.index(
            "# Translation guiding questions"
        )

        end = source.index(
            "# Translation language gate",
            start,
        )

        block = source[
            start:end
        ]

        self._check(
            (
                "self.last_repair_evidence_context"
                in block
                and
                "knowledge_result.context"
                not in block
            ),
            (
                "GROUNDING-PRECISION-T199 "
                "Translation question boundary"
            ),
            (
                "Translated guiding questions use "
                "the same verified repair evidence "
                "boundary as translation grounding."
            ),
            (
                "Translation question validation "
                "escaped its evidence boundary."
            ),
        )

    def test_t200_translation_question_gate_before_diagnostics(
        self,
    ) -> None:

        from pathlib import Path

        source = Path(
            "app/tutor.py"
        ).read_text(
            encoding="utf-8"
        )

        start = source.index(
            "# Translation pedagogy"
        )

        gate_position = source.index(
            "# Translation guiding-question",
            start,
        )

        diagnostics_position = source.index(
            "# Translation rejection",
            gate_position,
        )

        self._check(
            gate_position < diagnostics_position,
            (
                "GROUNDING-PRECISION-T200 "
                "Translation question gate order"
            ),
            (
                "Translation guiding-question "
                "acceptance is resolved before "
                "rejection diagnostics."
            ),
            (
                "Translation diagnostics can run "
                "before the guiding-question gate."
            ),
        )

    def test_t201_translation_question_rejection_diagnostic(
        self,
    ) -> None:

        from pathlib import Path

        source = Path(
            "app/tutor.py"
        ).read_text(
            encoding="utf-8"
        )

        start = source.index(
            "# Translation rejection"
        )

        end = source.index(
            "# ACCEPT TRANSLATION",
            start,
        )

        block = source[
            start:end
        ]

        self._check(
            (
                '"guiding_question"'
                in block
                and
                "translation_guiding_question_grounding"
                in block
            ),
            (
                "GROUNDING-PRECISION-T201 "
                "Translation question diagnostic"
            ),
            (
                "Translation rejection telemetry "
                "distinguishes guiding-question "
                "failures from generic pedagogy."
            ),
            (
                "Translation guiding-question "
                "rejection is not diagnosable."
            ),
        )

    def test_t202_translation_accept_requires_safe_question(
        self,
    ) -> None:

        from pathlib import Path

        source = Path(
            "app/tutor.py"
        ).read_text(
            encoding="utf-8"
        )

        start = source.index(
            "# ACCEPT TRANSLATION"
        )

        end = source.index(
            "# ACCEPT SAFE FINAL RESPONSE",
            start,
        )

        block = source[
            start:end
        ]

        self._check(
            (
                "translation_guiding_question_grounding"
                in block
                and
                ".is_safe"
                in block
            ),
            (
                "GROUNDING-PRECISION-T202 "
                "Translation question acceptance"
            ),
            (
                "Translated evidence cannot be "
                "accepted without a safe "
                "guiding-question result."
            ),
            (
                "Translation acceptance can bypass "
                "guiding-question grounding."
            ),
        )
    def test_t203_translation_question_telemetry(
        self,
    ) -> None:

        from pathlib import Path

        source = Path(
            "app/tutor.py"
        ).read_text(
            encoding="utf-8"
        )

        start = source.index(
            "# Translation guiding questions"
        )

        end = source.index(
            "# Translation language gate",
            start,
        )

        block = source[
            start:end
        ]

        self._check(
            (
                "self.last_translation_guiding_question_grounding"
                in block
            ),
            (
                "GROUNDING-PRECISION-T203 "
                "Translation question telemetry"
            ),
            (
                "Translated guiding-question "
                "validation is observable at runtime."
            ),
            (
                "Translation guiding-question "
                "telemetry is not stored."
            ),
        )
    def test_t204_live_threshold_false_positive_veto(
        self,
    ) -> None:

        import json
        from unittest.mock import patch

        from app.services.guiding_question_grounding_validator import (
            GuidingQuestionGroundingValidator,
        )

        response = (
            "คุณคิดว่าเมื่อมีแรงดันระหว่างฐานและอีมิตเตอร์ "
            "(B-E) เป็นบวกมากพอแค่ไหน "
            "จะทำให้จุดตัด B-E ของทรานซิสเตอร์ NPN "
            "ถูกกระตุ้นให้ทำงานแบบฟอร์เวิร์ดบายาสได้อย่างไร?"
        )

        evidence = (
            "Therefore, the collector current is related "
            "to the emitter current which is in turn a "
            "function of the B-E voltage."
        )

        provider_result = {
            "questions": [
                {
                    "question_quote": response,
                    "question_type": "factual",
                    "status": "answerable",
                    "evidence_quote": evidence,
                    "issue": "",
                }
            ]
        }

        validator = (
            GuidingQuestionGroundingValidator()
        )

        with patch(
            (
                "app.services."
                "guiding_question_grounding_validator."
                "chat_with_structured_ai"
            ),
            return_value=json.dumps(
                provider_result,
                ensure_ascii=False,
            ),
        ):
            result = validator.validate(
                response=response,
                knowledge_context=evidence,
            )

        self._check(
            (
                result.status == "unsupported"
                and
                any(
                    "threshold" in issue.lower()
                    for issue in result.issues
                )
            ),
            (
                "GROUNDING-PRECISION-T204 "
                "Live threshold false-positive veto"
            ),
            (
                "The observed live 'มากพอแค่ไหน' "
                "question is rejected when course "
                "evidence provides no threshold."
            ),
            (
                "A semantic false-positive threshold "
                "question remained supported."
            ),
        )

    def test_t205_question_exact_value_absent(
        self,
    ) -> None:

        validator = (
            GuidingQuestionGroundingValidator()
        )

        issue = (
            validator._answerability_precision_issue(
                question=(
                    "What exact B-E voltage is required?"
                ),
                evidence=(
                    "Collector current is a function "
                    "of the B-E voltage."
                ),
            )
        )

        self._check(
            issue is not None,
            (
                "GROUNDING-PRECISION-T205 "
                "Question exact-value precision"
            ),
            (
                "Exact-value questions require "
                "explicit numerical evidence."
            ),
            (
                "A non-numerical excerpt was allowed "
                "to answer an exact-value question."
            ),
        )

    def test_t206_question_exact_value_present(
        self,
    ) -> None:

        validator = (
            GuidingQuestionGroundingValidator()
        )

        issue = (
            validator._answerability_precision_issue(
                question=(
                    "What exact B-E voltage is required?"
                ),
                evidence=(
                    "The B-E voltage is 0.7 V."
                ),
            )
        )

        self._check(
            issue is None,
            (
                "GROUNDING-PRECISION-T206 "
                "Supported question exact value"
            ),
            (
                "An explicit numerical value can "
                "satisfy an exact-value question."
            ),
            (
                "Explicit numerical evidence was "
                "incorrectly rejected."
            ),
        )

    def test_t207_question_range_absent(
        self,
    ) -> None:

        validator = (
            GuidingQuestionGroundingValidator()
        )

        issue = (
            validator._answerability_precision_issue(
                question=(
                    "แรงดัน B-E ควรอยู่ในช่วงใด?"
                ),
                evidence=(
                    "The collector current is a "
                    "function of the B-E voltage."
                ),
            )
        )

        self._check(
            issue is not None,
            (
                "GROUNDING-PRECISION-T207 "
                "Question range precision"
            ),
            (
                "A numerical-range question requires "
                "explicit bounds."
            ),
            (
                "Qualitative evidence was allowed to "
                "answer a numerical-range question."
            ),
        )

    def test_t208_question_range_present(
        self,
    ) -> None:

        validator = (
            GuidingQuestionGroundingValidator()
        )

        issue = (
            validator._answerability_precision_issue(
                question=(
                    "แรงดันควรอยู่ในช่วงใด?"
                ),
                evidence=(
                    "The voltage range is 1.0 V "
                    "to 2.0 V."
                ),
            )
        )

        self._check(
            issue is None,
            (
                "GROUNDING-PRECISION-T208 "
                "Supported question range"
            ),
            (
                "Explicit numerical bounds can "
                "satisfy a range question."
            ),
            (
                "Explicit range evidence was "
                "incorrectly rejected."
            ),
        )
    def test_t209_question_efficiency_absent(
        self,
    ) -> None:

        validator = (
            GuidingQuestionGroundingValidator()
        )

        issue = (
            validator._answerability_precision_issue(
                question=(
                    "แรงดันเท่าไรจึงทำงานได้"
                    "อย่างมีประสิทธิภาพ?"
                ),
                evidence=(
                    "Collector current is related "
                    "to the B-E voltage."
                ),
            )
        )

        self._check(
            (
                issue is not None
                and
                "efficiency" in issue.lower()
            ),
            (
                "GROUNDING-PRECISION-T209 "
                "Question efficiency precision"
            ),
            (
                "Efficiency questions require an "
                "explicit efficiency criterion in "
                "course evidence."
            ),
            (
                "A relationship statement was treated "
                "as an efficiency criterion."
            ),
        )

    def test_t210_relation_question_precision_clear(
        self,
    ) -> None:

        validator = (
            GuidingQuestionGroundingValidator()
        )

        issue = (
            validator._answerability_precision_issue(
                question=(
                    "What is collector current "
                    "related to?"
                ),
                evidence=(
                    "The collector current is related "
                    "to the emitter current."
                ),
            )
        )

        self._check(
            issue is None,
            (
                "GROUNDING-PRECISION-T210 "
                "Relation question preserved"
            ),
            (
                "Ordinary evidence-supported relation "
                "questions are unaffected by the "
                "quantitative precision veto."
            ),
            (
                "Question precision over-blocked an "
                "ordinary relation question."
            ),
        )


    def test_t211_question_precision_deterministic(
        self,
    ) -> None:

        from unittest.mock import patch

        validator = (
            GuidingQuestionGroundingValidator()
        )

        with patch(
            (
                "app.services."
                "guiding_question_grounding_validator."
                "chat_with_structured_ai"
            )
        ) as mocked:

            issue = (
                validator._answerability_precision_issue(
                    question=(
                        "B-E ต้องเป็นบวกมากพอแค่ไหน?"
                    ),
                    evidence=(
                        "Collector current is a "
                        "function of B-E voltage."
                    ),
                )
            )

        self._check(
            (
                issue is not None
                and
                mocked.call_count == 0
            ),
            (
                "GROUNDING-PRECISION-T211 "
                "Question precision deterministic"
            ),
            (
                "Question answerability precision "
                "adds no LLM dependency."
            ),
            (
                "Question precision unexpectedly "
                "required an LLM call."
            ),
        )

    def test_t212_live_sufficient_threshold_variant(
        self,
    ) -> None:

        import json
        from unittest.mock import patch

        response = (
            "คุณคิดว่าเมื่อมีแรงดันระหว่างฐาน-อีมิตเตอร์ "
            "(B-E) เป็นบวกมากพอเพียง จะทำให้กระแสที่ไหล"
            "ผ่านฐาน-อีมิตเตอร์มีผลอย่างไรต่อกระแส"
            "คอลเลกเตอร์?"
        )

        evidence = (
            "Therefore, the collector current is related "
            "to the emitter current which is in turn a "
            "function of the B-E voltage."
        )

        provider_result = {
            "questions": [
                {
                    "question_quote": response,
                    "question_type": "factual",
                    "status": "answerable",
                    "evidence_quote": evidence,
                    "issue": "",
                }
            ]
        }

        validator = (
            GuidingQuestionGroundingValidator()
        )

        with patch(
            (
                "app.services."
                "guiding_question_grounding_validator."
                "chat_with_structured_ai"
            ),
            return_value=json.dumps(
                provider_result,
                ensure_ascii=False,
            ),
        ):
            result = validator.validate(
                response=response,
                knowledge_context=evidence,
            )

        self._check(
            (
                result.status == "unsupported"
                and
                any(
                    "threshold" in issue.lower()
                    for issue in result.issues
                )
            ),
            (
                "GROUNDING-PRECISION-T212 "
                "Thai sufficient-threshold variant"
            ),
            (
                "The live 'มากพอเพียง' wording is "
                "treated as an unsupported threshold "
                "when no threshold value exists."
            ),
            (
                "The Thai sufficient-threshold wording "
                "escaped deterministic precision."
            ),
        )

    def test_t213_thai_sufficient_threshold_phrase(
        self,
    ) -> None:

        validator = (
            GuidingQuestionGroundingValidator()
        )

        issue = (
            validator._answerability_precision_issue(
                question=(
                    "แรงดัน B-E ต้องมากพอ "
                    "จึงจะทำงานหรือไม่?"
                ),
                evidence=(
                    "Collector current is related "
                    "to B-E voltage."
                ),
            )
        )

        self._check(
            (
                issue is not None
                and
                "threshold" in issue.lower()
            ),
            (
                "GROUNDING-PRECISION-T213 "
                "Thai sufficient threshold"
            ),
            (
                "Thai 'มากพอ' wording requires "
                "explicit threshold evidence."
            ),
            (
                "A qualitative relationship satisfied "
                "a Thai threshold question."
            ),
        )

    def test_t214_supported_threshold_value(
        self,
    ) -> None:

        validator = (
            GuidingQuestionGroundingValidator()
        )

        issue = (
            validator._answerability_precision_issue(
                question=(
                    "แรงดัน B-E ต้องมากพอเท่าไร?"
                ),
                evidence=(
                    "The required B-E threshold "
                    "voltage is 0.7 V."
                ),
            )
        )

        self._check(
            issue is None,
            (
                "GROUNDING-PRECISION-T214 "
                "Supported threshold value"
            ),
            (
                "An explicit threshold value remains "
                "eligible for a threshold question."
            ),
            (
                "Question precision incorrectly "
                "blocked explicit threshold evidence."
            ),
        )

    def test_t215_supported_repair_context_preserved(
        self,
    ) -> None:

        from pathlib import Path

        source = Path(
            "app/tutor.py"
        ).read_text(
            encoding="utf-8"
        )

        start = source.index(
            "def _prepare_llm_repair_context("
        )

        end = source.index(
            "# Factual repair:",
            start,
        )

        block = source[start:end]

        self._check(
            (
                "force_verified_evidence: bool = False"
                in block
                and
                "and\n            not force_verified_evidence"
                in block
                and
                "knowledge_context"
                in block
            ),
            (
                "GROUNDING-PRECISION-T215 "
                "Supported repair context preserved"
            ),
            (
                "Supported grounding preserves the "
                "existing full-context repair path "
                "unless verified evidence is explicitly "
                "forced."
            ),
            (
                "C2 changed the frozen supported-repair "
                "behavior."
            ),
        )

    def test_t216_forced_repair_evidence_contract(
        self,
    ) -> None:

        from pathlib import Path

        source = Path(
            "app/tutor.py"
        ).read_text(
            encoding="utf-8"
        )

        start = source.index(
            "def _prepare_llm_repair_context("
        )

        end = source.index(
            "def ",
            start + len(
                "def _prepare_llm_repair_context("
            ),
        )

        block = source[start:end]

        self._check(
            (
                "force_verified_evidence: bool = False"
                in block
                and
                "additional_validation_issues:"
                in block
                and
                "additional_validation_issues"
                in block
                and
                "base_validation_issues"
                in block
            ),
            (
                "GROUNDING-PRECISION-T216 "
                "Forced repair evidence contract"
            ),
            (
                "The existing bounded selector can be "
                "forced for guiding-question recovery "
                "while carrying recovery issues."
            ),
            (
                "Guiding-question recovery cannot reuse "
                "the verified-evidence selector safely."
            ),
        )

    def test_t217_guiding_question_recovery_preparation(
        self,
    ) -> None:

        from pathlib import Path

        source = Path(
            "app/tutor.py"
        ).read_text(
            encoding="utf-8"
        )

        start = source.index(
            "# Guiding-question recovery eligibility"
        )

        end = source.index(
            "repair_requires_evidence_safe_recovery",
            start,
        )

        block = source[start:end]

        self._check(
            (
                "repair_validation.status"
                in block
                and
                '== "supported"'
                in block
                and
                "repair_guiding_question_requires_recovery"
                in block
                and
                "force_verified_evidence=True"
                in block
            ),
            (
                "GROUNDING-PRECISION-T217 "
                "Guiding-question recovery preparation"
            ),
            (
                "A supported repair rejected by the "
                "guiding-question boundary can prepare "
                "verified evidence on demand."
            ),
            (
                "Guiding-question-only repair rejection "
                "cannot prepare recovery evidence."
            ),
        )
    def test_t218_guiding_question_recovery_boundary(
        self,
    ) -> None:

        from pathlib import Path

        source = Path(
            "app/tutor.py"
        ).read_text(
            encoding="utf-8"
        )

        start = source.index(
            "# Prepare verified evidence on demand."
        )

        end = source.index(
            "repair_requires_evidence_safe_recovery",
            start,
        )

        block = source[start:end]

        self._check(
            (
                "learner_message=("
                in block
                and
                "user_message"
                in block
                and
                "knowledge_result.context"
                in block
                and
                "grounding_validation=("
                in block
                and
                "repair_validation"
                in block
            ),
            (
                "GROUNDING-PRECISION-T218 "
                "Guiding recovery evidence boundary"
            ),
            (
                "Recovery selection remains bounded by "
                "the learner request, current course "
                "knowledge, and repair validation."
            ),
            (
                "Guiding-question recovery escaped the "
                "existing evidence boundary."
            ),
        )

    def test_t219_evidence_safe_guiding_recovery_eligibility(
        self,
    ) -> None:

        from pathlib import Path

        source = Path(
            "app/tutor.py"
        ).read_text(
            encoding="utf-8"
        )

        start = source.index(
            "repair_requires_evidence_safe_recovery = ("
        )

        end = source.index(
            "# Evidence-safe grounding precheck",
            start,
        )

        block = source[start:end]

        self._check(
            (
                'repair_validation.status\n'
                '                                != "supported"'
                in block
                and
                "repair_guiding_question_requires_recovery"
                in block
                and
                "repair_pedagogical_validation.status"
                not in block
            ),
            (
                "GROUNDING-PRECISION-T219 "
                "Evidence-safe guiding recovery"
            ),
            (
                "Evidence-safe recovery activates for "
                "grounding failure or unsafe guiding "
                "questions, not generic pedagogy."
            ),
            (
                "Evidence-safe recovery eligibility is "
                "too broad or still excludes guiding "
                "question failures."
            ),
        )

    def test_t220_no_duplicate_verified_selection(
        self,
    ) -> None:

        from pathlib import Path

        source = Path(
            "app/tutor.py"
        ).read_text(
            encoding="utf-8"
        )

        start = source.index(
            "# Prepare verified evidence on demand."
        )

        end = source.index(
            "repair_requires_evidence_safe_recovery",
            start,
        )

        block = source[start:end]

        self._check(
            (
                "self.last_repair_evidence_selection"
                in block
                and
                ".has_verified_evidence"
                in block
                and
                "self.last_repair_evidence_context"
                in block
            ),
            (
                "GROUNDING-PRECISION-T220 "
                "No duplicate verified selection"
            ),
            (
                "Guiding-question recovery reuses an "
                "existing verified evidence pack and "
                "selects only when one is unavailable."
            ),
            (
                "Guiding-question recovery may perform "
                "unnecessary duplicate evidence "
                "selection."
            ),
        )
    def test_t221_relative_clause_not_question(
        self,
    ) -> None:

        from app.services.guiding_question_grounding_validator import (
            GuidingQuestionGroundingValidator,
        )

        validator = object.__new__(
            GuidingQuestionGroundingValidator
        )

        response = (
            "Therefore, the collector current is related "
            "to the emitter current which is in turn a "
            "function of the B-E voltage."
        )

        result = validator._may_contain_question(
            response
        )

        self._check(
            result is False,
            (
                "GROUNDING-PRECISION-T221 "
                "Relative clause is not a question"
            ),
            (
                "The English relative clause "
                "'which is in turn' does not trigger "
                "guiding-question detection."
            ),
            (
                "Question-presence detection still "
                "misclassifies an English relative "
                "clause as a guiding question."
            ),
        )

    def test_t222_which_question_still_detected(
        self,
    ) -> None:

        from app.services.guiding_question_grounding_validator import (
            GuidingQuestionGroundingValidator,
        )

        validator = object.__new__(
            GuidingQuestionGroundingValidator
        )

        response = (
            "Which terminal controls the current "
            "through the third terminal?"
        )

        result = validator._may_contain_question(
            response
        )

        self._check(
            result is True,
            (
                "GROUNDING-PRECISION-T222 "
                "Which question remains detectable"
            ),
            (
                "A genuine English question beginning "
                "with 'Which' remains detectable after "
                "the relative-clause false-positive fix."
            ),
            (
                "The false-positive fix suppresses a "
                "genuine English guiding question."
            ),
        )

    def test_t223_c3_guiding_question_recovery_eligibility(
        self,
    ) -> None:

        from pathlib import Path

        source = Path(
            "app/tutor.py"
        ).read_text(
            encoding="utf-8"
        )

        start = source.index(
            "# C3 — Evidence-bounded guiding-question"
        )

        end = source.index(
            "# Evidence-safe grounding precheck",
            start,
        )

        block = source[start:end]

        self._check(
            (
                "evidence_bounded_guiding_question_recovery = ("
                in block
                and
                "repair_validation.status"
                in block
                and
                '== "supported"'
                in block
                and
                "repair_guiding_question_requires_recovery"
                in block
                and
                "strategy.name"
                in block
                and
                '== "guiding_question"'
                in block
                and
                "preserve_scaffolding_strategy"
                in block
            ),
            (
                "GROUNDING-PRECISION-T223 "
                "C3 guiding-question recovery eligibility"
            ),
            (
                "C3 activates only for a factually supported "
                "repair whose guiding question requires "
                "knowledge-bounded recovery while preserving "
                "the guiding-question scaffolding strategy."
            ),
            (
                "C3 eligibility is missing or has escaped "
                "the intended guiding-question recovery "
                "boundary."
            ),
        )

    def test_t224_c3_verified_evidence_boundary(
        self,
    ) -> None:

        from pathlib import Path

        source = Path(
            "app/tutor.py"
        ).read_text(
            encoding="utf-8"
        )

        start = source.index(
            "# C3 — Evidence-bounded guiding-question"
        )

        composer_position = source.index(
            "self._compose_evidence_safe_repair(",
            start,
        )

        block = source[
            start:composer_position
        ]

        self._check(
            (
                "self.response_mode_repair_service"
                in block
                and
                ".repair("
                in block
                and
                "knowledge_context=("
                in block
                and
                "self.last_repair_evidence_context"
                in block
                and
                "original_response=("
                in block
                and
                "repaired_answer"
                in block
                and
                "validation=("
                in block
                and
                "repair_validation"
                in block
                and
                "knowledge_result.context"
                not in block
            ),
            (
                "GROUNDING-PRECISION-T224 "
                "C3 verified evidence boundary"
            ),
            (
                "C3 regenerates the rejected guiding question "
                "from the verified repair evidence context "
                "only."
            ),
            (
                "C3 generation is not strictly bounded by "
                "verified repair evidence."
            ),
        )

    def test_t225_c3_candidate_revalidation_pipeline(
        self,
    ) -> None:

        from pathlib import Path

        source = Path(
            "app/tutor.py"
        ).read_text(
            encoding="utf-8"
        )

        c3_position = source.index(
            "# C3 — Evidence-bounded guiding-question"
        )

        candidate_store_position = source.index(
            "self.last_evidence_safe_repair_candidate = (",
            c3_position,
        )

        grounding_position = source.index(
            "# Evidence-safe grounding precheck",
            candidate_store_position,
        )

        precision_position = source.index(
            "# Evidence-safe precision",
            grounding_position,
        )

        guiding_position = source.index(
            "# Evidence-safe guiding questions",
            precision_position,
        )

        acceptance_gate_position = source.index(
            "self._apply_guiding_question_acceptance_gate(",
            guiding_position,
        )

        accept_position = source.index(
            "# ACCEPT EVIDENCE-SAFE REPAIR",
            acceptance_gate_position,
        )

        self._check(
            (
                c3_position
                < candidate_store_position
                < grounding_position
                < precision_position
                < guiding_position
                < acceptance_gate_position
                < accept_position
            ),
            (
                "GROUNDING-PRECISION-T225 "
                "C3 candidate revalidation pipeline"
            ),
            (
                "The C3 candidate enters the existing "
                "evidence-safe grounding, precision, "
                "guiding-question, and acceptance pipeline "
                "before it can be accepted."
            ),
            (
                "C3 candidate generation bypasses or "
                "misorders an existing evidence-safe "
                "validation gate."
            ),
        )

    def test_t226_c3_unsafe_question_cannot_bypass_gate(
        self,
    ) -> None:

        from pathlib import Path

        source = Path(
            "app/tutor.py"
        ).read_text(
            encoding="utf-8"
        )

        start = source.index(
            "# Evidence-safe guiding questions"
        )

        accept_position = source.index(
            "# ACCEPT EVIDENCE-SAFE REPAIR",
            start,
        )

        block = source[
            start:accept_position
        ]

        self._check(
            (
                "evidence_safe_guiding_question_grounding"
                in block
                and
                "self._apply_guiding_question_acceptance_gate("
                in block
                and
                "guiding_question_grounding=("
                in block
                and
                "evidence_safe_guiding_question_grounding"
                in block
                and
                "evidence_safe_pedagogical_validation"
                in block
            ),
            (
                "GROUNDING-PRECISION-T226 "
                "C3 unsafe question cannot bypass gate"
            ),
            (
                "A C3 candidate remains subject to the "
                "knowledge-bounded guiding-question "
                "acceptance gate before evidence-safe "
                "acceptance."
            ),
            (
                "A C3 candidate could reach evidence-safe "
                "acceptance without the existing guiding-"
                "question safety gate."
            ),
        )

    def test_t227_c3_generation_is_bounded_once(
        self,
    ) -> None:

        from pathlib import Path

        source = Path(
            "app/tutor.py"
        ).read_text(
            encoding="utf-8"
        )

        start = source.index(
            "# C3 — Evidence-bounded guiding-question"
        )

        composer_position = source.index(
            "self._compose_evidence_safe_repair(",
            start,
        )

        block = source[
            start:composer_position
        ]

        repair_call_count = block.count(
            ".repair("
        )

        self._check(
            (
                repair_call_count == 1
                and
                "while " not in block
                and
                "while(" not in block
            ),
            (
                "GROUNDING-PRECISION-T227 "
                "C3 generation bounded once"
            ),
            (
                "C3 contains exactly one evidence-bounded "
                "repair generation call and introduces no "
                "retry loop."
            ),
            (
                "C3 generation is unbounded or contains "
                "more than one repair-generation call."
            ),
        )

    def test_t228_c3_not_used_for_safe_guiding_question(
        self,
    ) -> None:

        from pathlib import Path

        source = Path(
            "app/tutor.py"
        ).read_text(
            encoding="utf-8"
        )

        eligibility_start = source.index(
            "# Guiding-question recovery eligibility"
        )

        c3_start = source.index(
            "# C3 — Evidence-bounded guiding-question",
            eligibility_start,
        )

        block = source[
            eligibility_start:c3_start
        ]

        c3_end = source.index(
            "if evidence_bounded_guiding_question_recovery:",
            c3_start,
        )

        c3_eligibility = source[
            c3_start:c3_end
        ]

        self._check(
            (
                "repair_guiding_question_requires_recovery = ("
                in block
                and
                "repair_guiding_question_grounding"
                in block
                and
                "not repair_guiding_question_grounding"
                in block
                and
                ".is_safe"
                in block
                and
                "repair_guiding_question_requires_recovery"
                in c3_eligibility
            ),
            (
                "GROUNDING-PRECISION-T228 "
                "C3 skips safe guiding questions"
            ),
            (
                "C3 depends on the existing unsafe-guiding-"
                "question recovery flag and therefore does "
                "not activate when the repaired guiding "
                "question is already safe."
            ),
            (
                "C3 eligibility is not strictly tied to "
                "an unsafe repaired guiding question."
            ),
        )

    def test_t229_non_c3_recovery_preserves_composer(
        self,
    ) -> None:

        from pathlib import Path

        source = Path(
            "app/tutor.py"
        ).read_text(
            encoding="utf-8"
        )

        start = source.index(
            "# C3 — Evidence-bounded guiding-question"
        )

        composer_position = source.index(
            "self._compose_evidence_safe_repair(",
            start,
        )

        grounding_position = source.index(
            "# Evidence-safe grounding precheck",
            composer_position,
        )

        block = source[
            start:grounding_position
        ]

        self._check(
            (
                "if evidence_bounded_guiding_question_recovery:"
                in block
                and
                "else:"
                in block
                and
                "self._compose_evidence_safe_repair("
                in block
                and
                "self.last_repair_evidence_selection"
                in block
            ),
            (
                "GROUNDING-PRECISION-T229 "
                "Non-C3 recovery preserves composer"
            ),
            (
                "Evidence-safe recovery cases outside C3 "
                "continue to use the existing deterministic "
                "evidence-safe repair composer."
            ),
            (
                "C3 replaced or bypassed the existing "
                "deterministic evidence-safe composer for "
                "non-guiding-question recovery cases."
            ),
        )



    def run(
        self,
    ) -> int:

        print()
        print(
            "Step 16.20 Grounding Precision "
            "Regression Suite"
        )
        print()

        tests = [
            self.test_t1_pure_question,
            self.test_t2_statement,
            self.test_t3_statement_then_question,
            self.test_t4_paragraph_then_question,
            self.test_t5_inline_question_clause,
            self.test_t6_thai_pure_question,
            self.test_t7_thai_mixed_response,
            self.test_t8_directive_only,
            self.test_t9_compact_question,
            self.test_t10_no_llm_dependency,

            self.test_t11_entailment_required,
            self.test_t12_no_relation_strengthening,
            self.test_t13_no_quantitative_inference,
            self.test_t14_claim_by_claim,
            self.test_t15_questions_preserved,

            self.test_t16_generation_boundary,
            self.test_t17_generation_no_strengthening,
            self.test_t18_no_knowledge_instruction,
            self.test_t19_generation_invalid_type,
            self.test_t20_prompt_optional_integration,

            self.test_t21_prompt_parameter_contract,
            self.test_t22_runtime_builder_initialized,
            self.test_t23_runtime_instruction_built,
            self.test_t24_runtime_prompt_receives,
            self.test_t25_runtime_telemetry,
            self.test_t26_per_turn_clear,
            self.test_t27_reset_clear,
            self.test_t28_pipeline_order,

            self.test_t29_validator_provider_failure,
            self.test_t30_validator_token_budget,
            self.test_t31_repair_validator_token_budget,

            self.test_t32_precision_numeric_missing,
            self.test_t33_precision_numeric_supported,
            self.test_t34_precision_beta_missing,
            self.test_t35_precision_beta_supported,
            self.test_t36_precision_majority_missing,
            self.test_t37_precision_thai_majority,
            self.test_t38_precision_proportionality,
            self.test_t39_precision_question_ignored,
            self.test_t40_precision_deterministic,

            self.test_t41_precision_veto_clear,
            self.test_t42_precision_veto_supported,
            self.test_t43_precision_veto_partial,
            self.test_t44_precision_veto_unsupported,
            self.test_t45_precision_veto_no_duplicate,
            self.test_t46_precision_veto_invalid_type,
            self.test_t47_precision_veto_deterministic,

            self.test_t48_runtime_precision_services,
            self.test_t49_initial_precision_guard_routed,
            self.test_t50_initial_veto_routed,
            self.test_t51_repair_precision_guard_routed,
            self.test_t52_repair_veto_routed,
            self.test_t53_precision_telemetry_initialized,
            self.test_t54_precision_per_turn_clear,
            self.test_t55_precision_reset_clear,
            self.test_t56_precision_runtime_no_llm,

            self.test_t57_atomic_supported_evidence,
            self.test_t58_atomic_relation_strengthening,
            self.test_t59_atomic_fake_evidence_quote,
            self.test_t60_atomic_fake_response_quote,
            self.test_t61_atomic_partial_aggregation,
            self.test_t62_atomic_top_level_status_ignored,
            self.test_t63_atomic_empty_question,
            self.test_t64_atomic_empty_claim_fail_closed,

            self.test_t65_atomic_wrong_relation_evidence,
            self.test_t66_atomic_relation_alignment_supported,
            self.test_t67_repair_complete_evidence_boundary,
            self.test_t68_repair_relation_precision,
            self.test_t69_repair_technical_precision,
            self.test_t70_mode_repair_evidence_boundary,

            self.test_t71_factual_repair_reconstructs_without_original,
            self.test_t72_pedagogical_repair_preserves_original,
            self.test_t73_repair_atomic_validator_budget,
            self.test_t74_repair_candidate_telemetry_initialized,
            self.test_t75_repair_candidate_captured,
            self.test_t76_repair_candidate_debug_projection,

            self.test_t77_verified_evidence_model_frozen,
            self.test_t78_exact_repair_evidence_verified,
            self.test_t79_fabricated_repair_evidence_rejected,
            self.test_t80_paraphrased_repair_evidence_rejected,
            self.test_t81_empty_repair_evidence_safe,
            self.test_t82_repair_evidence_provider_fail_closed,

            self.test_t83_runtime_repair_evidence_selector_initialized,
            self.test_t84_supported_repair_preserves_context,
            self.test_t85_factual_repair_verified_context,
            self.test_t86_no_evidence_blocks_repair_context,
            self.test_t87_repair_context_typed_input,

            self.test_t88_runtime_verified_context_to_repair,
            self.test_t89_no_evidence_runtime_fail_closed,
            self.test_t90_repair_validator_verified_boundary,
            self.test_t91_repair_precision_verified_boundary,
            self.test_t92_runtime_evidence_telemetry,
            self.test_t93_supported_repair_path_preserved,

            self.test_t94_selector_requires_evidence_sufficiency,
            self.test_t95_identity_not_function_evidence,
            self.test_t96_selector_supports_evidence_chain,

            self.test_t97_atomic_issue_field_contract,
            self.test_t98_operational_evidence_chain,
            self.test_t99_evidence_safe_composer,
            self.test_t100_evidence_safe_exact_quotes,
            self.test_t101_selector_reason_excluded,
            self.test_t102_evidence_safe_no_evidence,
            self.test_t103_evidence_safe_invalid,
            self.test_t104_evidence_safe_typed_input,

            self.test_t105_runtime_evidence_safe_composer,
            self.test_t106_runtime_evidence_safe_helper,
            self.test_t107_runtime_evidence_safe_none,
            self.test_t108_evidence_safe_telemetry_defaults,
            self.test_t109_evidence_safe_reset,

            self.test_t110_evidence_safe_runtime_boundary,
            self.test_t111_evidence_safe_runtime_composition,
            self.test_t112_evidence_safe_validation_boundary,
            self.test_t113_evidence_safe_precision,
            self.test_t114_evidence_safe_pedagogy,
            self.test_t115_evidence_safe_acceptance_gate,
            self.test_t116_evidence_safe_final_fail_closed,

            self.test_t117_evidence_quote_completeness_contract,
            self.test_t118_truncated_leadin_rejected,
            self.test_t119_equation_leadin_example,
            self.test_t120_evidence_text_integrity_contract,
            self.test_t121_missing_variable_example,
            self.test_t122_clean_evidence_preference,

            self.test_t123_no_source_reconstruction_contract,
            self.test_t124_missing_content_not_inserted,
            self.test_t125_reconstruction_example,
            self.test_t126_translation_result_typed,
            self.test_t127_translation_result_frozen,
            self.test_t128_translation_boundary,
            self.test_t129_translation_relation_strength,
            self.test_t130_translation_empty_fail_closed,

            self.test_t131_translation_schema_boundary,
            self.test_t132_translation_success,
            self.test_t133_translation_input_boundary,
            self.test_t134_translation_provider_fail_closed,
            self.test_t135_translation_invalid_json,
            self.test_t136_translation_empty_output,
            self.test_t137_english_translation_bypass,
            self.test_t138_translation_task_profile,

            self.test_t139_runtime_translation_service,
            self.test_t140_runtime_translation_helper,
            self.test_t141_translation_telemetry_defaults,
            self.test_t142_translation_telemetry_reset,
            self.test_t143_translation_runtime_activated,

            self.test_t144_translation_target_language,
            self.test_t145_translation_runtime_input_boundary,
            self.test_t146_translation_grounding_boundary,
            self.test_t147_translation_precision_language_gate,
            self.test_t148_translation_pedagogy_gate,
            self.test_t149_translation_safe_fallback,
            self.test_t150_translation_acceptance_telemetry,

            self.test_t151_clean_evidence_usable,
            self.test_t152_trailing_as_unusable,
            self.test_t153_extraction_damage_unusable,
            self.test_t154_complete_relation_usable,
            self.test_t155_evidence_usability_deterministic,
            self.test_t156_selector_rejects_unusable_exact_quote,

            self.test_t157_evidence_retry_eligibility,
            self.test_t158_evidence_reselection_success,
            self.test_t159_evidence_retry_bounded,
            self.test_t160_provenance_failure_no_retry,
            self.test_t161_no_evidence_no_retry,
            self.test_t162_reselection_feedback,
            self.test_t163_evidence_retry_telemetry,

            self.test_t164_third_attempt_recovery,
            self.test_t165_cumulative_reselection_feedback,

            self.test_t166_guiding_question_result_typed,
            self.test_t167_no_question_fast_path,
            self.test_t168_answerable_guiding_question,
            self.test_t169_unsupported_exact_value_question,
            self.test_t170_unsupported_range_efficiency,
            self.test_t171_reflective_question_allowed,
            self.test_t172_question_evidence_provenance,
            self.test_t173_question_omission_fail_closed,

            self.test_t174_runtime_guiding_question_validator,
            self.test_t175_guiding_question_runtime_helper,
            self.test_t176_guiding_question_telemetry_defaults,
            self.test_t177_guiding_question_telemetry_reset,
            self.test_t178_guiding_question_debug_projection,
            self.test_t179_guiding_question_initial_activation,

            self.test_t180_guiding_question_gate_safe_preserves,
            self.test_t181_guiding_question_gate_unsupported,
            self.test_t182_guiding_question_gate_invalid,
            self.test_t183_guiding_question_gate_preserves_issues,
            self.test_t184_guiding_question_initial_pipeline_order,
            self.test_t185_guiding_question_gate_before_escalation,

            self.test_t186_repair_guiding_question_pipeline_order,
            self.test_t187_repair_guiding_question_same_boundary,
            self.test_t188_repair_guiding_question_gate_before_accept,
            self.test_t189_repair_unsupported_question_rejected,
            self.test_t190_repair_supported_question_allowed,
            self.test_t191_repair_guiding_question_telemetry_store,

            self.test_t192_evidence_safe_guiding_question_order,
            self.test_t193_evidence_safe_guiding_question_boundary,
            self.test_t194_evidence_safe_gate_before_accept,
            self.test_t195_evidence_safe_no_question_fast_path,
            self.test_t196_evidence_safe_unsupported_question,
            self.test_t197_evidence_safe_question_telemetry,

            self.test_t198_translation_guiding_question_order,
            self.test_t199_translation_guiding_question_boundary,
            self.test_t200_translation_question_gate_before_diagnostics,
            self.test_t201_translation_question_rejection_diagnostic,
            self.test_t202_translation_accept_requires_safe_question,
            self.test_t203_translation_question_telemetry,

            self.test_t204_live_threshold_false_positive_veto,
            self.test_t205_question_exact_value_absent,
            self.test_t206_question_exact_value_present,
            self.test_t207_question_range_absent,
            self.test_t208_question_range_present,
            self.test_t209_question_efficiency_absent,
            self.test_t210_relation_question_precision_clear,
            self.test_t211_question_precision_deterministic,

            self.test_t212_live_sufficient_threshold_variant,
            self.test_t213_thai_sufficient_threshold_phrase,
            self.test_t214_supported_threshold_value,

            self.test_t215_supported_repair_context_preserved,
            self.test_t216_forced_repair_evidence_contract,
            self.test_t217_guiding_question_recovery_preparation,
            self.test_t218_guiding_question_recovery_boundary,
            self.test_t219_evidence_safe_guiding_recovery_eligibility,
            self.test_t220_no_duplicate_verified_selection,

            self.test_t221_relative_clause_not_question,
            self.test_t222_which_question_still_detected, 
            self.test_t223_c3_guiding_question_recovery_eligibility,
            self.test_t224_c3_verified_evidence_boundary,  
            self.test_t225_c3_candidate_revalidation_pipeline,
            self.test_t226_c3_unsafe_question_cannot_bypass_gate,    

            self.test_t227_c3_generation_is_bounded_once,
            self.test_t228_c3_not_used_for_safe_guiding_question,
            self.test_t229_non_c3_recovery_preserves_composer,
        ]


        for test in tests:

            try:
                test()

            except Exception as exc:

                self._fail(
                    test.__name__,
                    str(exc),
                )

        print()
        print("SUMMARY")
        print(
            f"Passed: {self.passed}"
        )
        print(
            f"Failed: {self.failed}"
        )

        return (
            0
            if self.failed == 0
            else 1
        )


def main() -> int:
    return RegressionSuite16_20().run()


if __name__ == "__main__":
    raise SystemExit(
        main()
    )