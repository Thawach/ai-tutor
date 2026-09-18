from app.services.validation_escalation_policy import (
    ValidationEscalationPolicy,
)

from app.services.deterministic_response_repair import (
    DeterministicResponseRepair,
)

from app.ai import (
    chat_with_ai,
    reset_ai_usage,
    get_ai_usage_summary,
)

from app.core.config import (
    ACTIVE_COURSE,
)

from app.courses.loader import (
    CourseProfileLoader,
)

from app.memory import ConversationMemory
from app.state import TutorState
from app.evaluator import evaluate_response
from app.prompts.builder import PromptBuilder

from app.services.knowledge_service import (
    KnowledgeService,
)

from app.services.grounding_guard import (
    GroundingGuard,
)

from app.services.grounding_claim_precheck import (
    GroundingClaimPrecheck,
)

from app.services.embedded_claim_term_guard import (
    EmbeddedClaimTermGuard,
)

from app.services.embedded_claim_term_guard import (
    EmbeddedClaimTermGuard,
)

from app.services.language_consistency_guard import (
    LanguageConsistencyGuard,
)

from app.services.response_grounding_validator import (
    ResponseGroundingValidator,
)

from app.services.grounding_validation_models import (
    GroundingValidationResult,
)

from app.services.pedagogical_response_validator import (
    PedagogicalResponseValidator,
)

from app.services.pedagogical_validation_models import (
    PedagogicalValidationResult,
)

from app.services.pedagogical_precheck import (
    PedagogicalPrecheck,
)

from app.services.response_repair_service import (
    ResponseRepairService,
)

from app.services.relevance_gate import (
    RelevanceGate,
)

from app.services.retrieval_query_builder import (
    RetrievalQueryBuilder,
)

from app.services.learner_turn_intent_guard import (
    LearnerTurnIntentGuard,
)

from app.services.learner_turn_routing_policy import (
    LearnerTurnRoutingPolicy,
)

from app.services.tutoring_response_strategy_policy import (
    TutoringResponseStrategyPolicy,
)

from app.services.tutoring_response_mode_instruction_builder import (
    TutoringResponseModeInstructionBuilder,
)

from app.services.learner_progress_snapshot_builder import (
    LearnerProgressSnapshotBuilder,
)

from app.services.learner_progress_history_store import (
    LearnerProgressHistoryStore,
)

from app.services.grounded_generation_instruction_builder import (
    GroundedGenerationInstructionBuilder,
)

from app.domain.scaffolding.policy import (
    ScaffoldingPolicy,
)

from app.domain.scaffolding.strategies import (
    get_strategy,
)

from app.domain.scaffolding.interventions import (
    get_intervention,
)

from app.domain.scaffolding.citation_policy import (
    CitationPolicy,
)

from app.services.response_style_resolver import (
    ResponseStyleResolver,
)

from app.services.response_style_instruction_builder import (
    ResponseStyleInstructionBuilder,
)
from app.services.response_quality_guard import (
    ResponseQualityGuard,
)

from app.services.response_quality_policy import (
    ResponseQualityPolicy,
)

from app.services.response_quality_history import (
    ResponseQualityHistory,
)

from app.services.response_quality_history_models import (
    ResponseQualityTurnRecord,
)

from app.services.response_quality_models import (
    ResponseQualityResult,
)

from app.services.response_quality_analytics import (
    ResponseQualityAnalytics,
)

from app.services.response_mode_pedagogical_precheck import (
    ResponseModePedagogicalPrecheck,
)

from app.services.response_mode_pedagogical_validator import (
    ResponseModePedagogicalValidator,
)
from app.services.response_mode_repair_service import (
    ResponseModeRepairService,
)

from app.services.grounding_precision_guard import (
    GroundingPrecisionGuard,
)

from app.services.grounding_precision_veto_policy import (
    GroundingPrecisionVetoPolicy,
)

from app.services.repair_evidence_selector import (
    RepairEvidenceSelector,
)

from app.services.repair_evidence_selection_models import (
    RepairEvidenceSelectionResult,
)

from app.services.evidence_safe_repair_composer import (
    EvidenceSafeRepairComposer,
)

from app.services.evidence_safe_translation_models import (
    EvidenceSafeTranslationResult,
)

from app.services.evidence_safe_translation_service import (
    EvidenceSafeTranslationService,
)
from app.services.guiding_question_grounding_models import (
    GuidingQuestionGroundingResult,
)

from app.services.guiding_question_grounding_validator import (
    GuidingQuestionGroundingValidator,
)

class AITutor:

    MAX_REPAIR_ATTEMPTS = 1

    def __init__(self):

        # -------------------------
        # Core learning state
        # -------------------------

        self.memory = ConversationMemory()
        self.state = TutorState()

        # -------------------------
        # Domain / services
        # -------------------------

        self.scaffolding_policy = (
            ScaffoldingPolicy()
        )

        self.prompt_builder = (
            PromptBuilder()
        )

        self.grounded_generation_instruction_builder = (
            GroundedGenerationInstructionBuilder()
        )

        self.course_profile = (
            CourseProfileLoader()
            .load(
                ACTIVE_COURSE
            )
        )

        self.response_style_resolver = (
            ResponseStyleResolver()
        )

        self.resolved_response_style = (
            self.response_style_resolver.resolve(
                self.course_profile
            )
        )

        self.response_style_instruction_builder = (
            ResponseStyleInstructionBuilder()
        )

        self.response_style_instruction = (
            self.response_style_instruction_builder.build(
                self.resolved_response_style
            )
        )

        self.validation_escalation_policy = (
            ValidationEscalationPolicy()
        )

        self.deterministic_response_repair = (
            DeterministicResponseRepair(
                course_profile=self.course_profile
            )
        )

        self.knowledge_service = (
            KnowledgeService(
                course_profile=(
                    self.course_profile
                )
            )
        )

        self.knowledge_service.validate_startup()
        
        self.embedded_claim_term_guard = (
            EmbeddedClaimTermGuard(
                course_profile=(
                    self.course_profile
                )
            )
        )

        self.language_consistency_guard = (
            LanguageConsistencyGuard(
                course_profile=(
                    self.course_profile
                )
            )
        )   

        self.last_response_quality = None

        response_quality_profile = (
            self.course_profile.response_quality
        )

        self.response_quality_guard = (
            ResponseQualityGuard(
                max_characters=(
                    response_quality_profile
                    .max_characters
                ),
                min_characters=(
                    response_quality_profile
                    .min_characters
                ),
                max_questions=(
                    response_quality_profile
                    .max_questions
                ),
                detect_repetition=(
                    response_quality_profile
                    .detect_repetition
                ),
            )
        )


        self.response_quality_policy = (
            ResponseQualityPolicy()
        )

        self.last_response_quality_policy = None

        self.response_quality_history = (
            ResponseQualityHistory()
        )

        self.response_quality_analytics = (
            ResponseQualityAnalytics()
        )

        self.learner_turn_intent_guard = (
            LearnerTurnIntentGuard()
        )
        self.learner_turn_routing_policy = (
            LearnerTurnRoutingPolicy()
        )

        self.tutoring_response_strategy_policy = (
            TutoringResponseStrategyPolicy()
        )

        self.tutoring_response_mode_instruction_builder = (
            TutoringResponseModeInstructionBuilder()
        )

        self.learner_progress_snapshot_builder = (
            LearnerProgressSnapshotBuilder()
        )

        self.learner_progress_history = (
            LearnerProgressHistoryStore()
        )

        self.last_learner_turn_routing = None
        self.last_learner_turn_intent = None
        self.last_tutoring_response_mode = None
        self.last_tutoring_response_mode_instruction = None


        self.relevance_gate = (
            RelevanceGate()
        )

        self.grounding_guard = (
            GroundingGuard()
        )

        self.grounding_claim_precheck = (
            GroundingClaimPrecheck()
        )

        self.response_grounding_validator = (
            ResponseGroundingValidator()
        )

        self.guiding_question_grounding_validator = (
            GuidingQuestionGroundingValidator()
        )

        self.grounding_precision_guard = (
            GroundingPrecisionGuard()
        )

        self.grounding_precision_veto_policy = (
            GroundingPrecisionVetoPolicy()
        )

        self.pedagogical_precheck = (
            PedagogicalPrecheck()
        )

        self.pedagogical_response_validator = (
            PedagogicalResponseValidator()
        )

        self.response_mode_pedagogical_precheck = (
            ResponseModePedagogicalPrecheck(
                base_precheck=(
                    self.pedagogical_precheck
                )
            )
        )

        self.response_mode_pedagogical_validator = (
            ResponseModePedagogicalValidator(
                base_validator=(
                    self.pedagogical_response_validator
                )
            )
        )

        self.repair_evidence_selector = (
            RepairEvidenceSelector()
        )

        self.evidence_safe_repair_composer = (
            EvidenceSafeRepairComposer()
        )

        self.evidence_safe_translation_service = (
            EvidenceSafeTranslationService()
        )

        self.response_repair_service = (
            ResponseRepairService()
        )

        self.response_repair_service = (
            ResponseRepairService()
        )

        self.response_mode_repair_service = (
            ResponseModeRepairService(
                base_repair_service=(
                    self.response_repair_service
                )
            )
        )

        self.retrieval_query_builder = (
            RetrievalQueryBuilder()
        )

        self.citation_policy = (
            CitationPolicy()
        )

        # -------------------------
        # Conversation state
        # -------------------------

        self.original_question = None
        self.waiting_for_response = False

        self.last_decision = None
        self.last_evaluation_result = None
        self.last_intervention = None

        # -------------------------
        # Interrupted-turn retry state
        # -------------------------

        self._pending_evaluated_retry_message = None
        self._pending_evaluated_retry_original_question = None
        self._pending_evaluated_retry_tutor_question = None
        self._pending_evaluated_retry_evaluation_result = None
        self._pending_evaluated_retry_decision = None

        # -------------------------
        # Prompt / RAG state
        # -------------------------

        self.last_system_prompt = None
        self.last_retrieval_query = None
        self.last_generated_answer = None

        self.last_grounding_precision_instruction = None

        self.last_knowledge_result = None
        self.last_relevance_result = None
        self.last_grounding_status = None

        # -------------------------
        # Validation state
        # -------------------------

        self.last_grounding_claim_precheck = None
        self.last_embedded_claim_term_guard = None
        self.last_grounding_validation = None
        self.last_grounding_precision_guard = None

        self.last_pedagogical_precheck = None
        self.last_pedagogical_validation = None

        self.last_guiding_question_grounding = None
        self.last_repair_guiding_question_grounding = None
        self.last_evidence_safe_guiding_question_grounding = None
        self.last_translation_guiding_question_grounding = None

        # -------------------------
        # Repair state
        # -------------------------

        self.last_response_repaired = False
        self.last_repair_candidate = None

        self.last_repair_evidence_selection = None
        self.last_repair_evidence_context = None

        self.last_repair_evidence_selection_attempts = 0
        self.last_repair_evidence_retry_used = False
        self.last_repair_evidence_first_failure = None

        self.last_evidence_safe_repair_candidate = None
        self.last_evidence_safe_repair_used = False

        self.last_evidence_safe_translation_result = None
        self.last_evidence_safe_translation_candidate = None
        self.last_evidence_safe_translation_used = False

        self.last_evidence_safe_translation_semantic_validation = None
        self.last_evidence_safe_translation_validation = None
        self.last_evidence_safe_translation_precision_guard = None
        self.last_evidence_safe_translation_language_consistency = None
        self.last_evidence_safe_translation_pedagogical_precheck = None
        self.last_evidence_safe_translation_pedagogical_validation = None
        self.last_evidence_safe_translation_rejection_gate = None

        self.last_repair_grounding_claim_precheck = None

        self.last_repair_embedded_claim_term_guard = None
        self.last_repair_validation = None
        self.last_repair_grounding_precision_guard = None

        self.last_repair_pedagogical_precheck = None
        self.last_repair_pedagogical_validation = None

        # -------------------------
        # Escalation / repair telemetry
        # -------------------------

        self.last_escalation_decision = None
        self.last_repair_mode = None
        self.last_deterministic_replacements = []

        self.last_repair_failed = False
        self.last_repair_failure_type = None
        self.last_repair_failure_reason = None

        self.last_repair_attempts = 0
        self.last_repair_limit_reached = False

        self.last_language_consistency = None

        # -------------------------
        # Usage state
        # -------------------------

        self.last_ai_usage_summary = None



    def _compose_evidence_safe_repair(
        self,
        selection: RepairEvidenceSelectionResult | None,
    ) -> str | None:

        if selection is None:
            return None

        candidate = (
            self.evidence_safe_repair_composer
            .compose(
                selection
            )
        )

        return candidate


    def _translate_evidence_safe_repair(
        self,
        evidence_text: str,
        target_language: str,
    ) -> EvidenceSafeTranslationResult:

        if not isinstance(
            evidence_text,
            str,
        ):
            raise TypeError(
                "evidence_text must be str."
            )

        if not isinstance(
            target_language,
            str,
        ):
            raise TypeError(
                "target_language must be str."
            )

        return (
            self.evidence_safe_translation_service
            .translate(
                evidence_text=evidence_text,
                target_language=target_language,
            )
        )
    # =========================================================
    # MAIN RESPONSE PIPELINE
    # =========================================================

    def respond(
        self,
        user_message: str,
    ) -> str:

        # -------------------------
        # Reset usage for this turn
        # -------------------------

        reset_ai_usage()

        self.last_grounding_precision_guard = None
        self.last_repair_grounding_precision_guard = None

        self.last_evidence_safe_repair_candidate = None
        self.last_evidence_safe_repair_used = False

        self.last_evidence_safe_translation_result = None
        self.last_evidence_safe_translation_candidate = None
        self.last_evidence_safe_translation_used = False

        self.last_evidence_safe_translation_semantic_validation = None
        self.last_evidence_safe_translation_validation = None
        self.last_evidence_safe_translation_precision_guard = None
        self.last_evidence_safe_translation_language_consistency = None
        self.last_evidence_safe_translation_pedagogical_precheck = None
        self.last_evidence_safe_translation_pedagogical_validation = None
        self.last_evidence_safe_translation_rejection_gate = None

        self.last_learner_turn_intent = None
        self.last_learner_turn_routing = None
        self.last_tutoring_response_mode = None
        self.last_tutoring_response_mode_instruction = None

        self.last_repair_failed = False
        self.last_repair_failure_type = None
        self.last_repair_failure_reason = None

        self.last_repair_candidate = None
        self.last_repair_evidence_selection = None
        self.last_repair_evidence_context = None

        self.last_repair_evidence_selection_attempts = 0
        self.last_repair_evidence_retry_used = False
        self.last_repair_evidence_first_failure = None        

        self.last_repair_attempts = 0
        self.last_repair_limit_reached = False

        self.last_language_consistency = None

        self.last_escalation_decision = None
        self.last_repair_mode = None
        self.last_deterministic_replacements = []

        self.last_grounding_precision_instruction = None

        # -----------------------------------------------------
        # Per-turn guiding-question telemetry reset
        #
        # Must occur before any early-return path such as
        # out-of-course handling, otherwise telemetry from the
        # previous Tutor turn can leak into debug state.
        # -----------------------------------------------------

        self.last_guiding_question_grounding = None
        self.last_repair_guiding_question_grounding = None
        self.last_evidence_safe_guiding_question_grounding = None
        self.last_translation_guiding_question_grounding = None


        evaluation_performed_this_turn = False

        # =====================================================
        # PRE-TURN ACTIVE-SEQUENCE SNAPSHOT
        #
        # Retrieval may fail after routing has already changed
        # the active question. Preserve the prior sequence
        # boundary so a failed follow-up can be restored
        # without rolling back a successful learner evaluation.
        # =====================================================

        active_question_before_turn = (
            self.original_question
        )

        waiting_for_response_before_turn = (
            self.waiting_for_response
        )

        new_learning_sequence_turn = (
            not self.waiting_for_response
        )

        # -------------------------
        # First learner question
        # -------------------------

        # =====================================================
        # LEARNING-SEQUENCE BOUNDARY
        # =====================================================

        new_learning_sequence_turn = (
            not self.waiting_for_response
        )

        starts_new_sequence_for_history = (
            new_learning_sequence_turn
        )

        if new_learning_sequence_turn:
            self._pending_evaluated_retry_message = None
            self._pending_evaluated_retry_original_question = None
            self._pending_evaluated_retry_tutor_question = None
            self._pending_evaluated_retry_evaluation_result = None
            self._pending_evaluated_retry_decision = None

        self.state.next_turn()

        if not self.waiting_for_response:

            self.original_question = (
                user_message
            )

            self.waiting_for_response = True

        else:

            # =================================================
            # STEP 16.18.3B — LEARNER TURN INTENT ROUTING
            # =================================================

            learner_turn_intent = (
                self.learner_turn_intent_guard
                .evaluate(
                    user_message
                )
            )

            self.last_learner_turn_intent = (
                learner_turn_intent
            )

            learner_turn_routing = (
                self.learner_turn_routing_policy
                .decide(
                    learner_turn_intent
                )
            )

            # -------------------------------------------------
            # Topic-change pedagogical state isolation
            # -------------------------------------------------

            if (
                learner_turn_routing.route
                == "topic_change"
            ):
                starts_new_sequence_for_history = (
                    True
                )

                self.state.reset_learning_sequence()

            self.last_learner_turn_routing = (
                learner_turn_routing
            )

            # -------------------------------------------------
            # A follow-up question or explicit topic change
            # becomes the active learner question.
            #
            # Clarification requests remain attached to the
            # existing tutoring sequence.
            # -------------------------------------------------

            if (
                learner_turn_routing
                .replace_active_question
            ):

                self.original_question = (
                    user_message
                )

            # -------------------------------------------------
            # Evaluate only answer-like / uncertain turns.
            #
            # Question-like learner turns must not be scored
            # as incorrect answers.
            # -------------------------------------------------

            if (
                learner_turn_routing
                .should_evaluate_response
            ):

                last_tutor_question = (
                    self.memory
                    .get_last_assistant_message()
                )

                # =============================================
                # STEP 16.22B-2
                # INTERRUPTED EVALUATED-TURN RETRY
                #
                # If the exact learner attempt was already
                # evaluated successfully before retrieval
                # failed, reuse that committed evaluation.
                #
                # Do NOT:
                # - call Evaluator again
                # - increment attempt_count again
                # - update streaks again
                # - add the same misconception again
                # - run scaffolding decision again
                # =============================================

                retry_committed_evaluation = (
                    self._pending_evaluated_retry_message
                    is not None
                    and
                    self._pending_evaluated_retry_message
                    == user_message
                    and
                    self._pending_evaluated_retry_original_question
                    == self.original_question
                    and
                    self._pending_evaluated_retry_tutor_question
                    == last_tutor_question
                    and
                    self._pending_evaluated_retry_evaluation_result
                    is not None
                    and
                    self._pending_evaluated_retry_decision
                    is not None
                )

                if retry_committed_evaluation:

                    evaluation_result = (
                        self
                        ._pending_evaluated_retry_evaluation_result
                    )

                    decision = (
                        self
                        ._pending_evaluated_retry_decision
                    )

                    # -----------------------------------------
                    # Consume the marker before continuing.
                    #
                    # If retrieval fails again later in this
                    # same retry, the retrieval exception
                    # boundary will re-arm it.
                    # -----------------------------------------

                    self._pending_evaluated_retry_message = None
                    self._pending_evaluated_retry_original_question = None
                    self._pending_evaluated_retry_tutor_question = None
                    self._pending_evaluated_retry_evaluation_result = None
                    self._pending_evaluated_retry_decision = None

                    self.last_evaluation_result = (
                        evaluation_result
                    )

                    self.last_decision = (
                        decision
                    )

                    evaluation_performed_this_turn = (
                        True
                    )

                else:

                    # -----------------------------------------
                    # Any non-matching learner answer is a
                    # genuinely new attempt. Invalidate the
                    # old retry marker before evaluating it.
                    # -----------------------------------------

                    self._pending_evaluated_retry_message = None
                    self._pending_evaluated_retry_original_question = None
                    self._pending_evaluated_retry_tutor_question = None
                    self._pending_evaluated_retry_evaluation_result = None
                    self._pending_evaluated_retry_decision = None

                    evaluation_result = (
                        evaluate_response(
                            original_question=(
                                self.original_question
                            ),
                            tutor_question=(
                                last_tutor_question
                            ),
                            learner_response=(
                                user_message
                            ),
                        )
                    )

                    self.last_evaluation_result = (
                        evaluation_result
                    )

                    evaluation_performed_this_turn = (
                        True
                    )

                    # -------------------------
                    # Store misconception
                    # -------------------------

                    if (
                        evaluation_result.classification
                        == "misconception"
                        and
                        evaluation_result.misconception
                    ):

                        self.state.add_misconception(
                            evaluation_result
                            .misconception
                        )

                    evaluation = (
                        evaluation_result
                        .classification
                    )

                    self.state.set_evaluation(
                        evaluation
                    )

                    # -------------------------
                    # Scaffolding decision
                    # -------------------------

                    decision = (
                        self.scaffolding_policy.decide(
                            state=self.state,
                            evaluation=evaluation,
                        )
                    )

                    self.last_decision = (
                        decision
                    )

            else:

                # ---------------------------------------------
                # Question-like / topic-change turns abandon
                # any pending exact-answer retry transaction.
                # ---------------------------------------------

                self._pending_evaluated_retry_message = None
                self._pending_evaluated_retry_original_question = None
                self._pending_evaluated_retry_tutor_question = None
                self._pending_evaluated_retry_evaluation_result = None
                self._pending_evaluated_retry_decision = None

                # ---------------------------------------------
                # Intent-aware bypass.
                #
                # Do not preserve stale per-turn evaluation
                # telemetry from the previous learner answer.
                #
                # Most importantly, do not call:
                # - state.set_evaluation()
                # - scaffolding_policy.decide()
                #
                # Therefore follow-up questions do not create
                # artificial failure streaks or escalation.
                # ---------------------------------------------

                self.last_evaluation_result = None
                self.last_decision = None


        # =====================================================
        # TUTORING RESPONSE MODE
        # Telemetry-only in Step 16.19.3.
        # This decision must not change prompt generation,
        # validation, repair, or scaffolding behavior yet.
        # =====================================================

        tutoring_response_mode = (
            self.tutoring_response_strategy_policy
            .decide(
                self.last_learner_turn_intent
            )
        )

        self.last_tutoring_response_mode = (
            tutoring_response_mode
        )

        tutoring_response_mode_instruction = (
            self.tutoring_response_mode_instruction_builder
            .build(
                tutoring_response_mode
            )
        )

        self.last_tutoring_response_mode_instruction = (
            tutoring_response_mode_instruction
        )

        # =====================================================
        # RETRIEVAL
        # =====================================================

        # =====================================================
        # RETRIEVAL CONTEXT BOUNDARY
        #
        # The first learner turn of a new learning sequence
        # must not inherit Tutor retrieval context from the
        # previous sequence.
        # =====================================================

        retrieval_tutor_question = None

        if not new_learning_sequence_turn:

            retrieval_tutor_question = (
                self.memory.get_last_assistant_message()
            )

        learner_turn_routing = (
            self.last_learner_turn_routing
        )

        # -----------------------------------------------------
        # Intent-aware retrieval
        # -----------------------------------------------------

        if (
            learner_turn_routing
            is not None
            and
            learner_turn_routing
            .use_current_message_for_retrieval
        ):

            # -------------------------------------------------
            # Follow-up / clarification:
            #
            # Current learner question is the primary query.
            # Previous Tutor message may provide local context.
            # -------------------------------------------------

            if (
                learner_turn_routing
                .include_previous_tutor_context
                and
                retrieval_tutor_question
            ):

                knowledge_query = (
                    f"{user_message}\n"
                    f"{retrieval_tutor_question}"
                )

            # -------------------------------------------------
            # Topic change:
            #
            # Previous Tutor context must not contaminate
            # retrieval for the new topic.
            # -------------------------------------------------

            else:

                knowledge_query = (
                    user_message
                )

        else:

            # -------------------------------------------------
            # Normal answer / uncertain intent / first question
            #
            # Preserve the existing retrieval behavior.
            # -------------------------------------------------

            knowledge_query = (
                self.retrieval_query_builder
                .build(
                    original_question=(
                        self.original_question
                        or user_message
                    ),
                    tutor_question=(
                        retrieval_tutor_question
                    ),
                    learner_response=(
                        user_message
                    ),
                )
            )

        self.last_retrieval_query = (
            knowledge_query
        )

        try:
            knowledge_result = (
                self.knowledge_service.retrieve(
                    query=knowledge_query,
                    n_results=5,
                )
            )

        except Exception:

            # -------------------------------------------------
            # Retrieval / RAG runtime failure.
            #
            # Do not fabricate an empty KnowledgeResult and
            # do not allow relevance, grounding, or Tutor
            # generation to run after unresolved retrieval.
            #
            # Recovery is route-aware because routing and
            # learner evaluation may already have completed
            # before retrieval begins.
            # -------------------------------------------------

            failed_route = getattr(
                learner_turn_routing,
                "route",
                None,
            )

            # -------------------------------------------------
            # STEP 16.22B-2
            # Preserve one successfully evaluated learner
            # attempt for an exact-message retry.
            #
            # This is armed only when evaluation completed
            # before retrieval failed.
            # -------------------------------------------------

            if (
                evaluation_performed_this_turn
                and
                waiting_for_response_before_turn
                and
                self.last_evaluation_result
                is not None
                and
                self.last_decision
                is not None
            ):

                self._pending_evaluated_retry_message = (
                    user_message
                )

                self._pending_evaluated_retry_original_question = (
                    self.original_question
                )

                self._pending_evaluated_retry_tutor_question = (
                    self.memory
                    .get_last_assistant_message()
                )

                self._pending_evaluated_retry_evaluation_result = (
                    self.last_evaluation_result
                )

                self._pending_evaluated_retry_decision = (
                    self.last_decision
                )


            # -------------------------------------------------
            # A newly opened sequence never received a Tutor
            # response. Close it completely.
            #
            # Explicit topic change already terminated the old
            # topic pedagogically. Do not revive that topic;
            # simply close the undelivered new sequence.
            # -------------------------------------------------

            if (
                new_learning_sequence_turn
                or
                failed_route == "topic_change"
            ):

                self.original_question = None
                self.waiting_for_response = False
                self.state.reset_learning_sequence()

            # -------------------------------------------------
            # A follow-up question was promoted to the active
            # question before retrieval. Since no Tutor reply
            # was delivered, restore the previously active
            # learning question.
            #
            # Do not reset LearnerState.
            # -------------------------------------------------

            elif failed_route == "follow_up_question":

                self.original_question = (
                    active_question_before_turn
                )

                self.waiting_for_response = (
                    waiting_for_response_before_turn
                )

            # -------------------------------------------------
            # Answer-like and clarification turns preserve the
            # existing active sequence.
            #
            # If learner evaluation completed successfully
            # before retrieval failed, its state transition is
            # retained as a completed learner-progress event.
            # -------------------------------------------------

            raise

        self.last_knowledge_result = (
            knowledge_result
        )

        # -------------------------
        # Relevance gate
        # -------------------------

        try:
            relevance_result = (
                self.relevance_gate.evaluate(
                    query=knowledge_query,
                    knowledge_result=(
                        knowledge_result
                    ),
                )
            )

        except Exception:

            # -------------------------------------------------
            # Relevance-provider/runtime failure occurred
            # before a Tutor response could be delivered.
            #
            # Do not fabricate a relevant / irrelevant result.
            # Close the incomplete learning sequence and
            # preserve the original provider exception.
            # -------------------------------------------------

            self.original_question = None
            self.waiting_for_response = False
            self.state.reset_learning_sequence()

            raise

        self.last_relevance_result = (
            relevance_result
        )

        # -------------------------
        # Grounding availability
        # -------------------------

        grounding_status = (
            self.grounding_guard.evaluate(
                result=knowledge_result,
                relevance=relevance_result,
            )
        )

        self.last_grounding_status = (
            grounding_status
        )


        # =====================================================
        # OUT-OF-COURSE / NO-GROUNDED-KNOWLEDGE GUARD
        # =====================================================

        if not grounding_status.has_knowledge:

            # ---------------------------------------------
            # Do not call Tutor LLM when the active course
            # does not contain relevant grounded knowledge.
            # ---------------------------------------------

            answer = (
                self._build_out_of_course_fallback(
                    learner_message=user_message,
                )
            )

            # ---------------------------------------------
            # No tutor generation occurred
            # ---------------------------------------------

            self.last_system_prompt = None
            self.last_generated_answer = None
            self.last_intervention = None

            # ---------------------------------------------
            # No response validation is required because
            # no model-generated tutor answer exists.
            # ---------------------------------------------

            self.last_grounding_claim_precheck = None
            self.last_embedded_claim_term_guard = None
            self.last_grounding_validation = None

            self.last_response_quality = None
            self.last_response_quality_policy = None

            self.last_pedagogical_precheck = None
            self.last_pedagogical_validation = None

            # ---------------------------------------------
            # No repair path
            # ---------------------------------------------

            self.last_response_repaired = False

            self.last_repair_grounding_claim_precheck = None
            self.last_repair_embedded_claim_term_guard = None
            self.last_repair_validation = None

            self.last_repair_pedagogical_precheck = None
            self.last_repair_pedagogical_validation = None

            # ---------------------------------------------
            # Collect usage.
            #
            # Normally only relevance_gate has used an LLM
            # on a first-turn out-of-course request.
            # ---------------------------------------------

            self.last_ai_usage_summary = (
                get_ai_usage_summary()
            )

            # ---------------------------------------------
            # End the tutoring sequence.
            #
            # The next learner input must be treated as
            # a NEW question rather than as an answer to
            # this fallback message.
            # ---------------------------------------------

            self.original_question = None
            self.waiting_for_response = False

            # Do NOT add this exchange to ConversationMemory.
            #
            # Otherwise the out-of-course fallback can
            # contaminate the next retrieval query because
            # RetrievalQueryBuilder uses the previous tutor
            # message.

            self._record_learner_progress_history(
                starts_new_sequence=(
                    starts_new_sequence_for_history
                ),
            )

            return answer

        # =====================================================
        # SCAFFOLDING
        # =====================================================

        strategy = get_strategy(
            self.state.scaffolding_level
        )

        intervention_evaluation = (
            self.state.last_evaluation
            if evaluation_performed_this_turn
            else None
        )

        intervention = get_intervention(
            intervention_evaluation
        )



        intervention_name = getattr(
            intervention,
            "name",
            "none",
        )

        intervention_instruction = getattr(
            intervention,
            "instruction",
            "",
        )

        self.last_intervention = (
            intervention
        )

        misconception = None

        if (
            self.last_evaluation_result
            is not None
        ):

            misconception = (
                self.last_evaluation_result
                .misconception
            )

        # =====================================================
        # EVIDENCE-BOUNDED GENERATION INSTRUCTION
        # =====================================================

        grounding_precision_instruction = (
            self.grounded_generation_instruction_builder
            .build(
                has_grounded_knowledge=(
                    grounding_status.has_knowledge
                )
            )
        )

        self.last_grounding_precision_instruction = (
            grounding_precision_instruction
        )


        # =====================================================
        # PROMPT
        # =====================================================

        system_prompt = (
            self.prompt_builder
            .build_system_prompt(
                strategy=strategy,
                intervention=intervention,
                misconception=misconception,
                knowledge_context=(
                    knowledge_result.context
                ),
                has_grounded_knowledge=(
                    grounding_status
                    .has_knowledge
                ),
                response_style_instruction=(
                    self.response_style_instruction
                ),
                response_mode_instruction=(
                    tutoring_response_mode_instruction
                ),
                grounding_precision_instruction=(
                    grounding_precision_instruction
                ),
            )
        )

        self.last_system_prompt = (
            system_prompt
        )

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            }
        ]

        messages.extend(
            self.memory.get_messages()
        )

        messages.append(
            {
                "role": "user",
                "content": user_message,
            }
        )

        # =====================================================
        # TUTOR GENERATION
        # =====================================================

        try:
            answer = chat_with_ai(
                messages,
                task_name="tutor",
            )

        except Exception:
            # Provider/runtime generation failure must not
            # leave an incomplete tutoring sequence active.
            #
            # Preserve the original exception contract:
            # cleanup Tutor state, then re-raise.
            self.original_question = None
            self.waiting_for_response = False

            self.state.reset_learning_sequence()

            raise

        # เก็บคำตอบดิบก่อน validation / repair
        self.last_generated_answer = answer

        # =====================================================
        # INITIALIZE VALIDATION STATE
        # =====================================================

        grounding_claim_precheck = None
        embedded_claim_term_guard = None
        grounding_validation = None
        grounding_precision_guard_result = None

        pedagogical_precheck = None
        pedagogical_validation = None

        repair_guiding_question_grounding = None

        evidence_safe_guiding_question_grounding = None
        translation_guiding_question_grounding = None   

        guiding_question_grounding = None

        repair_grounding_claim_precheck = None
        repair_embedded_claim_term_guard = None
        repair_validation = None
        repair_grounding_precision_guard_result = None

        repair_pedagogical_precheck = None
        repair_pedagogical_validation = None

        self.last_response_repaired = False

        # =====================================================
        # VALIDATE RESPONSE
        # =====================================================

        if grounding_status.has_knowledge:

            # -------------------------
            # Grounding claim precheck
            # -------------------------

            grounding_claim_precheck = (
                self.grounding_claim_precheck.evaluate(
                    response=answer,
                )
            )

            # -------------------------
            # Embedded claim /
            # technical term guard
            # -------------------------

            embedded_claim_term_guard = (
                self.embedded_claim_term_guard.evaluate(
                    response=answer,
                )
            )

            # -------------------------
            # Grounding validation
            # -------------------------

            if (
                embedded_claim_term_guard.status
                == "terminology_issue"
            ):

                # Technical terminology problem:
                # force repair without another LLM validator call.

                grounding_validation = (
                    GroundingValidationResult(
                        status="unsupported",
                        confidence=1.0,
                        reason=(
                            embedded_claim_term_guard.reason
                        ),
                        issues=(
                            embedded_claim_term_guard.issues
                        ),
                    )
                )

            elif (
                grounding_claim_precheck.status
                == "no_claims"
                and
                embedded_claim_term_guard.status
                == "clear"
            ):

                # Safe deterministic fast-path.

                grounding_validation = (
                    GroundingValidationResult(
                        status="supported",
                        confidence=1.0,
                        reason=(
                            "No explicit or embedded factual claim "
                            "required semantic grounding validation."
                        ),
                        issues=[],
                    )
                )

            else:

                # Either:
                # - factual statement exists
                # - embedded claim exists
                # - deterministic logic is uncertain

                grounding_validation = (
                    self.response_grounding_validator
                    .validate(
                        response=answer,
                        knowledge_context=(
                            knowledge_result.context
                        ),
                    )
                )

            # -------------------------------------------------
            # Deterministic grounding precision guard
            # -------------------------------------------------

            grounding_precision_guard_result = (
                self.grounding_precision_guard
                .evaluate(
                    response=answer,
                    knowledge_context=(
                        knowledge_result.context
                    ),
                )
            )

            # -------------------------------------------------
            # Precision veto
            #
            # Semantic "supported" cannot remain supported
            # when deterministic evidence precision detects
            # an unsupported high-risk detail.
            # -------------------------------------------------

            grounding_validation = (
                self.grounding_precision_veto_policy
                .apply(
                    validation=(
                        grounding_validation
                    ),
                    precision=(
                        grounding_precision_guard_result
                    ),
                )
            )

            # -------------------------------------------------
            # Knowledge-bounded guiding-question grounding
            #
            # This validates question answerability only.
            # It does not alter factual-statement grounding.
            # -------------------------------------------------

            guiding_question_grounding = (
                self._validate_guiding_questions(
                    response=answer,
                    knowledge_context=(
                        knowledge_result.context
                    ),
                )
            )

            # -------------------------
            # Pedagogical precheck
            # -------------------------

            pedagogical_precheck = (
                self.response_mode_pedagogical_precheck
                .evaluate(
                    response=answer,
                    strategy_name=(
                        strategy.name
                    ),
                    response_mode=(
                        tutoring_response_mode
                    ),
                )
            )

            # -------------------------
            # Pedagogical validation
            # -------------------------

            if (
                pedagogical_precheck.status
                == "valid"
            ):

                pedagogical_validation = (
                    PedagogicalValidationResult(
                        status="valid",
                        confidence=1.0,
                        reason=(
                            "Validated by deterministic "
                            "pedagogical precheck."
                        ),
                        issues=[],
                    )
                )

            elif (
                pedagogical_precheck.status
                == "violation"
            ):

                pedagogical_validation = (
                    PedagogicalValidationResult(
                        status="violation",
                        confidence=1.0,
                        reason=(
                            pedagogical_precheck.reason
                        ),
                        issues=(
                            pedagogical_precheck.issues
                        ),
                    )
                )

            else:
                pedagogical_validation = (
                    self.response_mode_pedagogical_validator
                    .validate(
                        response=answer,
                        scaffolding_level=(
                            self.state.scaffolding_level
                        ),
                        strategy_name=(
                            strategy.name
                        ),
                        strategy_instruction=(
                            strategy.instruction
                        ),
                        intervention_name=(
                            intervention_name
                        ),
                        intervention_instruction=(
                            intervention_instruction
                        ),
                        response_mode=(
                            tutoring_response_mode
                        ),
                        response_mode_instruction=(
                            tutoring_response_mode_instruction
                        ),
                    )
                )


            # -------------------------------------------------
            # Guiding-question acceptance gate
            #
            # Unsupported or unverifiable guiding questions
            # become a pedagogical acceptance violation so
            # the frozen escalation policy can route them
            # through the existing repair path.
            # -------------------------------------------------

            pedagogical_validation = (
                self._apply_guiding_question_acceptance_gate(
                    guiding_question_grounding=(
                        guiding_question_grounding
                    ),
                    pedagogical_validation=(
                        pedagogical_validation
                    ),
                )
            )

            # =================================================
            # REPAIR DECISION
            # =================================================

            escalation_decision = (
                self.validation_escalation_policy.decide(
                    grounding_validation=(
                        grounding_validation
                    ),
                    embedded_claim_term_guard=(
                        embedded_claim_term_guard
                    ),
                    pedagogical_validation=(
                        pedagogical_validation
                    ),
                )
            )

            self.last_escalation_decision = (
                escalation_decision
            )


            if (
                escalation_decision.action
                == "accept"
            ):

                self.last_repair_mode = None
                self.last_deterministic_replacements = []

            # =================================================
            # RESPONSE REPAIR
            # =================================================

            if (
                escalation_decision.action
                != "accept"
            ):

                # ---------------------------------------------
                # Repair attempt limit
                # ---------------------------------------------

                if (
                    self.last_repair_attempts
                    >= self.MAX_REPAIR_ATTEMPTS
                ):

                    self.last_repair_limit_reached = True
                    self.last_repair_failed = True
                    self.last_repair_failure_type = (
                        "repair_limit"
                    )
                    self.last_repair_failure_reason = (
                        "Maximum repair attempts reached."
                    )

                    answer = (
                        self._build_grounding_fallback(
                            learner_message=(
                                user_message
                            ),
                        )
                    )

                    # End this tutoring sequence safely.
                    self.original_question = None
                    self.waiting_for_response = False

                else:

                    self.last_repair_attempts += 1

                    pedagogical_issues = []

                    if (
                        pedagogical_validation
                        is not None
                    ):

                        pedagogical_issues = (
                            pedagogical_validation.issues
                        )

                    repaired_answer = None

                    # Full course context remains the default for:
                    # - deterministic repair
                    # - supported pedagogical-only LLM repair
                    #
                    # Factual LLM repair may replace this with a
                    # verified evidence-only context.
                    repair_validation_context = (
                        knowledge_result.context
                    )

                    # ---------------------------------------------
                    # 1. Deterministic repair
                    # ---------------------------------------------

                    if (
                        escalation_decision.action
                        == "deterministic_repair"
                    ):

                        deterministic_result = (
                            self.deterministic_response_repair
                            .repair_terminology(
                                response=answer
                            )
                        )

                        self.last_repair_mode = (
                            "deterministic"
                        )

                        self.last_deterministic_replacements = (
                            deterministic_result.replacements
                        )

                        if (
                            deterministic_result.repaired
                        ):

                            repaired_answer = (
                                deterministic_result.response
                            )

                        else:

                            self.last_repair_mode = (
                                "llm"
                            )
                            self.last_deterministic_replacements = []

                            (
                                repair_evidence_selection,
                                repair_generation_context,
                            ) = (
                                self._prepare_llm_repair_context(
                                    learner_message=(
                                        user_message
                                    ),
                                    knowledge_context=(
                                        knowledge_result.context
                                    ),
                                    grounding_validation=(
                                        grounding_validation
                                    ),
                                )
                            )

                            self.last_repair_evidence_selection = (
                                repair_evidence_selection
                            )

                            if (
                                repair_evidence_selection
                                is not None
                                and
                                repair_evidence_selection
                                .has_verified_evidence
                            ):
                                self.last_repair_evidence_context = (
                                    repair_generation_context
                                )

                            if (
                                repair_generation_context
                                is None
                            ):

                                self.last_repair_failed = True

                                self.last_repair_failure_type = (
                                    "repair_evidence"
                                )

                                self.last_repair_failure_reason = (
                                    (
                                        repair_evidence_selection.reason
                                        if repair_evidence_selection
                                        is not None
                                        else
                                        "No verified repair evidence "
                                        "was available."
                                    )
                                )

                                answer = (
                                    self._build_grounding_fallback(
                                        learner_message=(
                                            user_message
                                        ),
                                    )
                                )

                                self.original_question = None
                                self.waiting_for_response = False

                            else:

                                repair_validation_context = (
                                    repair_generation_context
                                )

                                repaired_answer = (
                                    self._run_response_repair(
                                        original_response=answer,
                                        knowledge_context=(
                                            repair_generation_context
                                        ),
                                        validation=(
                                            grounding_validation
                                        ),
                                        learner_message=(
                                            user_message
                                        ),
                                        scaffolding_level=(
                                            self.state.scaffolding_level
                                        ),
                                        strategy_name=(
                                            strategy.name
                                        ),
                                        strategy_instruction=(
                                            strategy.instruction
                                        ),
                                        intervention_name=(
                                            intervention_name
                                        ),
                                        pedagogical_issues=(
                                            pedagogical_issues
                                        ),
                                        response_mode=(
                                            tutoring_response_mode
                                        ),
                                        response_mode_instruction=(
                                            tutoring_response_mode_instruction
                                        ),
                                    )
                                )

                    # ---------------------------------------------
                    # 2. Semantic / LLM repair
                    # ---------------------------------------------

                    elif (
                        escalation_decision.action
                        == "llm_repair"
                    ):

                        self.last_repair_mode = (
                            "llm"
                        )

                        self.last_deterministic_replacements = []

                        (
                            repair_evidence_selection,
                            repair_generation_context,
                        ) = (
                            self._prepare_llm_repair_context(
                                learner_message=(
                                    user_message
                                ),
                                knowledge_context=(
                                    knowledge_result.context
                                ),
                                grounding_validation=(
                                    grounding_validation
                                ),
                            )
                        )

                        self.last_repair_evidence_selection = (
                            repair_evidence_selection
                        )

                        if (
                            repair_evidence_selection
                            is not None
                            and
                            repair_evidence_selection
                            .has_verified_evidence
                        ):
                            self.last_repair_evidence_context = (
                                repair_generation_context
                            )

                        if (
                            repair_generation_context
                            is None
                        ):

                            self.last_repair_failed = True

                            self.last_repair_failure_type = (
                                "repair_evidence"
                            )

                            self.last_repair_failure_reason = (
                                (
                                    repair_evidence_selection.reason
                                    if repair_evidence_selection
                                    is not None
                                    else
                                    "No verified repair evidence "
                                    "was available."
                                )
                            )

                            answer = (
                                self._build_grounding_fallback(
                                    learner_message=(
                                        user_message
                                    ),
                                )
                            )

                            self.original_question = None
                            self.waiting_for_response = False

                        else:

                            repair_validation_context = (
                                repair_generation_context
                            )

                            repaired_answer = (
                                self._run_response_repair(
                                    original_response=(
                                        answer
                                    ),
                                    knowledge_context=(
                                        repair_generation_context
                                    ),
                                    validation=(
                                        grounding_validation
                                    ),
                                    learner_message=(
                                        user_message
                                    ),
                                    scaffolding_level=(
                                        self.state
                                        .scaffolding_level
                                    ),
                                    strategy_name=(
                                        strategy.name
                                    ),
                                    strategy_instruction=(
                                        strategy.instruction
                                    ),
                                    intervention_name=(
                                        intervention_name
                                    ),
                                    pedagogical_issues=(
                                        pedagogical_issues
                                    ),
                                    response_mode=(
                                        tutoring_response_mode
                                    ),
                                    response_mode_instruction=(
                                        tutoring_response_mode_instruction
                                    ),
                                )
                            )

                    else:

                        # Unknown escalation actions must fail closed.
                        self.last_repair_failed = True
                        self.last_repair_failure_type = (
                            "unknown_repair_action"
                        )
                        self.last_repair_failure_reason = (
                            "Unsupported repair escalation action: "
                            f"{escalation_decision.action}"
                        )
                        answer = (
                            self._build_grounding_fallback(
                                learner_message=(
                                    user_message
                                ),
                            )
                        )
                        self.original_question = None
                        self.waiting_for_response = False


                    # =================================================
                    # VALIDATE REPAIRED RESPONSE
                    # =================================================

                    if repaired_answer is not None:

                        self.last_repair_candidate = (
                            repaired_answer
                        )

                        # ---------------------------------------------
                        # Repair grounding claim precheck
                        # ---------------------------------------------

                        repair_grounding_claim_precheck = (
                            self.grounding_claim_precheck
                            .evaluate(
                                response=repaired_answer,
                            )
                        )

                        # ---------------------------------------------
                        # Repair embedded claim / term guard
                        # ---------------------------------------------

                        repair_embedded_claim_term_guard = (
                            self.embedded_claim_term_guard
                            .evaluate(
                                response=repaired_answer,
                            )
                        )

                        # ---------------------------------------------
                        # Repair grounding validation
                        # ---------------------------------------------

                        if (
                            repair_embedded_claim_term_guard.status
                            == "terminology_issue"
                        ):

                            repair_validation = (
                                GroundingValidationResult(
                                    status="unsupported",
                                    confidence=1.0,
                                    reason=(
                                        repair_embedded_claim_term_guard
                                        .reason
                                    ),
                                    issues=(
                                        repair_embedded_claim_term_guard
                                        .issues
                                    ),
                                )
                            )

                        elif (
                            repair_grounding_claim_precheck.status
                            == "no_claims"
                            and
                            repair_embedded_claim_term_guard.status
                            == "clear"
                        ):

                            repair_validation = (
                                GroundingValidationResult(
                                    status="supported",
                                    confidence=1.0,
                                    reason=(
                                        "No explicit or embedded factual "
                                        "claim required semantic grounding "
                                        "validation."
                                    ),
                                    issues=[],
                                )
                            )

                        else:

                            repair_validation = (
                                self.response_grounding_validator
                                .validate(
                                    response=repaired_answer,
                                    knowledge_context=(
                                        repair_validation_context
                                    ),
                                    task_name=(
                                        "response_validator_repair"
                                    ),
                                )
                            )

                        # ---------------------------------------------
                        # Repair grounding precision guard
                        # ---------------------------------------------

                        repair_grounding_precision_guard_result = (
                            self.grounding_precision_guard
                            .evaluate(
                                response=repaired_answer,
                                knowledge_context=(
                                    repair_validation_context
                                ),
                            )
                        )

                        # ---------------------------------------------
                        # Repair precision veto
                        # ---------------------------------------------

                        repair_validation = (
                            self.grounding_precision_veto_policy
                            .apply(
                                validation=(
                                    repair_validation
                                ),
                                precision=(
                                    repair_grounding_precision_guard_result
                                ),
                            )
                        )

                                                # ---------------------------------------------
                        # Repair knowledge-bounded guiding questions
                        #
                        # Use the SAME factual boundary used by repair
                        # semantic grounding and precision validation.
                        # ---------------------------------------------

                        repair_guiding_question_grounding = (
                            self._validate_guiding_questions(
                                response=repaired_answer,
                                knowledge_context=(
                                    repair_validation_context
                                ),
                            )
                        )

                        # ---------------------------------------------
                        # Repair pedagogical precheck
                        # ---------------------------------------------

                        repair_pedagogical_precheck = (
                            self.response_mode_pedagogical_precheck
                            .evaluate(
                                response=repaired_answer,
                                strategy_name=(
                                    strategy.name
                                ),
                                response_mode=(
                                    tutoring_response_mode
                                ),
                            )
                        )

                        # ---------------------------------------------
                        # Repair pedagogical validation
                        # ---------------------------------------------

                        if (
                            repair_pedagogical_precheck.status
                            == "valid"
                        ):

                            repair_pedagogical_validation = (
                                PedagogicalValidationResult(
                                    status="valid",
                                    confidence=1.0,
                                    reason=(
                                        "Validated by deterministic "
                                        "pedagogical precheck."
                                    ),
                                    issues=[],
                                )
                            )

                        elif (
                            repair_pedagogical_precheck.status
                            == "violation"
                        ):

                            repair_pedagogical_validation = (
                                PedagogicalValidationResult(
                                    status="violation",
                                    confidence=1.0,
                                    reason=(
                                        repair_pedagogical_precheck
                                        .reason
                                    ),
                                    issues=(
                                        repair_pedagogical_precheck
                                        .issues
                                    ),
                                )
                            )

                        else:
                            repair_pedagogical_validation = (
                                self.response_mode_pedagogical_validator
                                .validate(
                                    response=repaired_answer,
                                    scaffolding_level=(
                                        self.state
                                        .scaffolding_level
                                    ),
                                    strategy_name=(
                                        strategy.name
                                    ),
                                    strategy_instruction=(
                                        strategy.instruction
                                    ),
                                    intervention_name=(
                                        intervention_name
                                    ),
                                    intervention_instruction=(
                                        intervention_instruction
                                    ),
                                    response_mode=(
                                        tutoring_response_mode
                                    ),
                                    response_mode_instruction=(
                                        tutoring_response_mode_instruction
                                    ),
                                    task_name=(
                                        "pedagogical_validator_repair"
                                    ),
                                )
                            )


                        # ---------------------------------------------
                        # Repair guiding-question acceptance gate
                        #
                        # Unsupported or unverifiable factual guiding
                        # questions cannot survive repair merely because
                        # the explanatory claims are grounded.
                        # ---------------------------------------------

                        repair_pedagogical_validation = (
                            self._apply_guiding_question_acceptance_gate(
                                guiding_question_grounding=(
                                    repair_guiding_question_grounding
                                ),
                                pedagogical_validation=(
                                    repair_pedagogical_validation
                                ),
                            )
                        )


                        # =============================================
                        # ACCEPT REPAIRED ANSWER
                        # =============================================

                        if (
                            repair_validation.status
                            == "supported"
                            and
                            repair_pedagogical_validation
                            is not None
                            and
                            repair_pedagogical_validation.status
                            == "valid"
                        ):

                            answer = (
                                repaired_answer
                            )
                            self.last_response_repaired = True

                        # =============================================
                        # REPAIR FAILED — TRY EVIDENCE-SAFE FALLBACK
                        # =============================================

                        else:

                            evidence_safe_accepted = False

                            # -----------------------------------------
                            # Guiding-question recovery eligibility
                            #
                            # A repair may be factually grounded but
                            # still be rejected because its guiding
                            # question asks for unsupported or
                            # unverifiable information.
                            # -----------------------------------------

                            repair_guiding_question_requires_recovery = (
                                repair_guiding_question_grounding
                                is not None
                                and
                                not repair_guiding_question_grounding
                                .is_safe
                            )

                            # -----------------------------------------
                            # Prepare verified evidence on demand.
                            #
                            # Supported-grounding pedagogical repair
                            # normally uses the full course context and
                            # therefore has no verified evidence pack.
                            #
                            # Only when the repair failed specifically
                            # at the guiding-question boundary do we
                            # force bounded verified-evidence selection.
                            # -----------------------------------------

                            if (
                                repair_validation.status
                                == "supported"
                                and
                                repair_guiding_question_requires_recovery
                                and
                                self.last_repair_mode
                                == "llm"
                                and
                                (
                                    self.last_repair_evidence_selection
                                    is None
                                    or
                                    not
                                    self.last_repair_evidence_selection
                                    .has_verified_evidence
                                    or
                                    self.last_repair_evidence_context
                                    is None
                                )
                            ):

                                recovery_validation_issues = tuple(
                                    repair_guiding_question_grounding
                                    .issues
                                )

                                if not recovery_validation_issues:
                                    recovery_validation_issues = (
                                        repair_guiding_question_grounding
                                        .reason,
                                    )

                                (
                                    recovery_evidence_selection,
                                    recovery_evidence_context,
                                ) = (
                                    self._prepare_llm_repair_context(
                                        learner_message=(
                                            user_message
                                        ),
                                        knowledge_context=(
                                            knowledge_result.context
                                        ),
                                        grounding_validation=(
                                            repair_validation
                                        ),
                                        force_verified_evidence=True,
                                        additional_validation_issues=(
                                            recovery_validation_issues
                                        ),
                                    )
                                )

                                self.last_repair_evidence_selection = (
                                    recovery_evidence_selection
                                )

                                if (
                                    recovery_evidence_selection
                                    is not None
                                    and
                                    recovery_evidence_selection
                                    .has_verified_evidence
                                ):
                                    self.last_repair_evidence_context = (
                                        recovery_evidence_context
                                    )
                                else:
                                    self.last_repair_evidence_context = (
                                        None
                                    )



                            repair_requires_evidence_safe_recovery = (
                                repair_validation.status
                                != "supported"
                                or
                                repair_guiding_question_requires_recovery
                            )

                            # -----------------------------------------
                            # Evidence-safe fallback is available only
                            # for an LLM repair that:
                            #
                            # - failed factual grounding; OR
                            # - failed the knowledge-bounded
                            #   guiding-question gate;
                            # - has verified repair evidence;
                            # - has the exact verified evidence context.
                            #
                            # Generic pedagogical-only failures do not
                            # activate evidence-safe reconstruction.
                            # -----------------------------------------

                            if (
                                repair_requires_evidence_safe_recovery
                                and
                                self.last_repair_mode
                                == "llm"
                                and
                                self.last_repair_evidence_selection
                                is not None
                                and
                                self.last_repair_evidence_selection
                                .has_verified_evidence
                                and
                                self.last_repair_evidence_context
                                is not None
                            ):

                                # -----------------------------------------
                                # C3 — Evidence-bounded guiding-question
                                # recovery.
                                #
                                # When the repaired response is factually
                                # supported but its guiding question is not
                                # safely answerable from course knowledge,
                                # make one additional bounded repair attempt
                                # using ONLY the verified evidence context.
                                #
                                # Other evidence-safe recovery cases retain
                                # the existing deterministic composer.
                                # -----------------------------------------

                                evidence_bounded_guiding_question_recovery = (
                                    repair_validation.status
                                    == "supported"
                                    and
                                    repair_guiding_question_requires_recovery
                                    and
                                    strategy.name
                                    == "guiding_question"
                                    and
                                    tutoring_response_mode
                                    .preserve_scaffolding_strategy
                                )

                                if evidence_bounded_guiding_question_recovery:

                                    evidence_bounded_guiding_question_issues = list(
                                        repair_guiding_question_grounding.issues
                                    )

                                    if not evidence_bounded_guiding_question_issues:
                                        evidence_bounded_guiding_question_issues = [
                                            repair_guiding_question_grounding.reason
                                        ]

                                    try:
                                        evidence_safe_candidate = (
                                            self.response_mode_repair_service
                                            .repair(
                                                original_response=(
                                                    repaired_answer
                                                ),
                                                knowledge_context=(
                                                    self.last_repair_evidence_context
                                                ),
                                                validation=(
                                                    repair_validation
                                                ),
                                                learner_message=(
                                                    user_message
                                                ),
                                                scaffolding_level=(
                                                    self.state.scaffolding_level
                                                ),
                                                strategy_name=(
                                                    strategy.name
                                                ),
                                                strategy_instruction=(
                                                    strategy.instruction
                                                ),
                                                intervention_name=(
                                                    intervention_name
                                                ),
                                                pedagogical_issues=(
                                                    evidence_bounded_guiding_question_issues
                                                ),
                                                response_mode=(
                                                    tutoring_response_mode
                                                ),
                                                response_mode_instruction=(
                                                    tutoring_response_mode_instruction
                                                ),
                                            )
                                        )

                                    except Exception as exc:

                                        self.last_repair_failed = True
                                        self.last_repair_failure_type = (
                                            "repair_provider"
                                        )
                                        self.last_repair_failure_reason = (
                                            str(exc)
                                        )

                                        self.original_question = None
                                        self.waiting_for_response = False

                                        self.state.reset_learning_sequence()

                                        raise

                                else:

                                    evidence_safe_candidate = (
                                        self._compose_evidence_safe_repair(
                                            self.last_repair_evidence_selection
                                        )
                                    )

                                self.last_evidence_safe_repair_candidate = (
                                    evidence_safe_candidate
                                )

                                if (
                                    evidence_safe_candidate
                                    is not None
                                ):

                                    # ---------------------------------
                                    # Evidence-safe grounding precheck
                                    # ---------------------------------

                                    evidence_safe_grounding_claim_precheck = (
                                        self.grounding_claim_precheck
                                        .evaluate(
                                            response=(
                                                evidence_safe_candidate
                                            ),
                                        )
                                    )

                                    # ---------------------------------
                                    # Evidence-safe embedded-term guard
                                    # ---------------------------------

                                    evidence_safe_embedded_claim_term_guard = (
                                        self.embedded_claim_term_guard
                                        .evaluate(
                                            response=(
                                                evidence_safe_candidate
                                            ),
                                        )
                                    )

                                    # ---------------------------------
                                    # Evidence-safe grounding validation
                                    #
                                    # IMPORTANT:
                                    # validate only against the exact
                                    # verified evidence context.
                                    # ---------------------------------

                                    if (
                                        evidence_safe_embedded_claim_term_guard
                                        .status
                                        == "terminology_issue"
                                    ):

                                        evidence_safe_validation = (
                                            GroundingValidationResult(
                                                status="unsupported",
                                                confidence=1.0,
                                                reason=(
                                                    evidence_safe_embedded_claim_term_guard
                                                    .reason
                                                ),
                                                issues=(
                                                    evidence_safe_embedded_claim_term_guard
                                                    .issues
                                                ),
                                            )
                                        )

                                    elif (
                                        evidence_safe_grounding_claim_precheck
                                        .status
                                        == "no_claims"
                                        and
                                        evidence_safe_embedded_claim_term_guard
                                        .status
                                        == "clear"
                                    ):

                                        evidence_safe_validation = (
                                            GroundingValidationResult(
                                                status="supported",
                                                confidence=1.0,
                                                reason=(
                                                    "No explicit or embedded "
                                                    "factual claim required "
                                                    "semantic grounding "
                                                    "validation."
                                                ),
                                                issues=[],
                                            )
                                        )

                                    else:

                                        evidence_safe_validation = (
                                            self.response_grounding_validator
                                            .validate(
                                                response=(
                                                    evidence_safe_candidate
                                                ),
                                                knowledge_context=(
                                                    self.last_repair_evidence_context
                                                ),
                                                task_name=(
                                                    "response_validator_repair"
                                                ),
                                            )
                                        )

                                    # ---------------------------------
                                    # Evidence-safe precision guard
                                    # ---------------------------------

                                    evidence_safe_precision_guard = (
                                        self.grounding_precision_guard
                                        .evaluate(
                                            response=(
                                                evidence_safe_candidate
                                            ),
                                            knowledge_context=(
                                                self.last_repair_evidence_context
                                            ),
                                        )
                                    )

                                    evidence_safe_validation = (
                                        self.grounding_precision_veto_policy
                                        .apply(
                                            validation=(
                                                evidence_safe_validation
                                            ),
                                            precision=(
                                                evidence_safe_precision_guard
                                            ),
                                        )
                                    )


                                    # ---------------------------------
                                    # Evidence-safe guiding questions
                                    #
                                    # Validate against the SAME exact
                                    # verified evidence boundary.
                                    # ---------------------------------

                                    evidence_safe_guiding_question_grounding = (
                                        self._validate_guiding_questions(
                                            response=(
                                                evidence_safe_candidate
                                            ),
                                            knowledge_context=(
                                                self.last_repair_evidence_context
                                            ),
                                        )
                                    )

                                    self.last_evidence_safe_guiding_question_grounding = (
                                        evidence_safe_guiding_question_grounding
                                    )

                                    # ---------------------------------
                                    # Evidence-safe pedagogy precheck
                                    # ---------------------------------

                                    evidence_safe_pedagogical_precheck = (
                                        self.response_mode_pedagogical_precheck
                                        .evaluate(
                                            response=(
                                                evidence_safe_candidate
                                            ),
                                            strategy_name=(
                                                strategy.name
                                            ),
                                            response_mode=(
                                                tutoring_response_mode
                                            ),
                                        )
                                    )

                                    # ---------------------------------
                                    # Evidence-safe pedagogy validation
                                    # ---------------------------------

                                    if (
                                        evidence_safe_pedagogical_precheck
                                        .status
                                        == "valid"
                                    ):

                                        evidence_safe_pedagogical_validation = (
                                            PedagogicalValidationResult(
                                                status="valid",
                                                confidence=1.0,
                                                reason=(
                                                    "Validated by deterministic "
                                                    "pedagogical precheck."
                                                ),
                                                issues=[],
                                            )
                                        )

                                    elif (
                                        evidence_safe_pedagogical_precheck
                                        .status
                                        == "violation"
                                    ):

                                        evidence_safe_pedagogical_validation = (
                                            PedagogicalValidationResult(
                                                status="violation",
                                                confidence=1.0,
                                                reason=(
                                                    evidence_safe_pedagogical_precheck
                                                    .reason
                                                ),
                                                issues=(
                                                    evidence_safe_pedagogical_precheck
                                                    .issues
                                                ),
                                            )
                                        )

                                    else:

                                        evidence_safe_pedagogical_validation = (
                                            self.response_mode_pedagogical_validator
                                            .validate(
                                                response=(
                                                    evidence_safe_candidate
                                                ),
                                                scaffolding_level=(
                                                    self.state
                                                    .scaffolding_level
                                                ),
                                                strategy_name=(
                                                    strategy.name
                                                ),
                                                strategy_instruction=(
                                                    strategy.instruction
                                                ),
                                                intervention_name=(
                                                    intervention_name
                                                ),
                                                intervention_instruction=(
                                                    intervention_instruction
                                                ),
                                                response_mode=(
                                                    tutoring_response_mode
                                                ),
                                                response_mode_instruction=(
                                                    tutoring_response_mode_instruction
                                                ),
                                                task_name=(
                                                    "pedagogical_validator_repair"
                                                ),
                                            )
                                        )


                                    # ---------------------------------
                                    # Evidence-safe guiding-question
                                    # acceptance gate
                                    # ---------------------------------

                                    evidence_safe_pedagogical_validation = (
                                        self._apply_guiding_question_acceptance_gate(
                                            guiding_question_grounding=(
                                                evidence_safe_guiding_question_grounding
                                            ),
                                            pedagogical_validation=(
                                                evidence_safe_pedagogical_validation
                                            ),
                                        )
                                    )    

                                    # =================================
                                    # ACCEPT EVIDENCE-SAFE REPAIR
                                    # =================================

                                    if (
                                        evidence_safe_validation.status
                                        == "supported"
                                        and
                                        evidence_safe_pedagogical_validation
                                        is not None
                                        and
                                        evidence_safe_pedagogical_validation
                                        .status
                                        == "valid"
                                    ):

                                        # =================================
                                        # Evidence-safe presentation base
                                        # =================================

                                        final_evidence_safe_answer = (
                                            evidence_safe_candidate
                                        )

                                        accepted_grounding_claim_precheck = (
                                            evidence_safe_grounding_claim_precheck
                                        )

                                        accepted_embedded_claim_term_guard = (
                                            evidence_safe_embedded_claim_term_guard
                                        )

                                        accepted_validation = (
                                            evidence_safe_validation
                                        )

                                        accepted_precision_guard = (
                                            evidence_safe_precision_guard
                                        )

                                        accepted_pedagogical_precheck = (
                                            evidence_safe_pedagogical_precheck
                                        )

                                        accepted_pedagogical_validation = (
                                            evidence_safe_pedagogical_validation
                                        )

                                        # =================================
                                        # BOUNDED TRANSLATION
                                        # =================================

                                        target_language = None

                                        if (
                                            self.resolved_response_style
                                            is not None
                                        ):
                                            target_language = (
                                                self.resolved_response_style
                                                .primary_language
                                            )

                                        if (
                                            isinstance(
                                                target_language,
                                                str,
                                            )
                                            and
                                            target_language.strip()
                                        ):

                                            translation_result = (
                                                self._translate_evidence_safe_repair(
                                                    evidence_text=(
                                                        evidence_safe_candidate
                                                    ),
                                                    target_language=(
                                                        target_language
                                                    ),
                                                )
                                            )

                                            self.last_evidence_safe_translation_result = (
                                                translation_result
                                            )

                                            self.last_evidence_safe_translation_result = (
                                                translation_result
                                            )

                                            # -------------------------
                                            # Translation output failure
                                            # diagnostics
                                            # -------------------------

                                            if (
                                                translation_result.status
                                                == "invalid"
                                            ):
                                                self.last_evidence_safe_translation_rejection_gate = (
                                                    "translation_output"
                                                )

                                            if (
                                                translation_result
                                                .has_translation
                                            ):

                                                translated_candidate = (
                                                    translation_result
                                                    .translated_text
                                                )

                                                self.last_evidence_safe_translation_candidate = (
                                                    translated_candidate
                                                )

                                                # -------------------------
                                                # Translation claim precheck
                                                # -------------------------

                                                translation_grounding_claim_precheck = (
                                                    self.grounding_claim_precheck
                                                    .evaluate(
                                                        response=(
                                                            translated_candidate
                                                        ),
                                                    )
                                                )

                                                # -------------------------
                                                # Translation term guard
                                                # -------------------------

                                                translation_embedded_claim_term_guard = (
                                                    self.embedded_claim_term_guard
                                                    .evaluate(
                                                        response=(
                                                            translated_candidate
                                                        ),
                                                    )
                                                )

                                                # -------------------------
                                                # Translation grounding
                                                #
                                                # Validate the translated
                                                # candidate against the same
                                                # verified English evidence
                                                # boundary.
                                                # -------------------------

                                                if (
                                                    translation_embedded_claim_term_guard
                                                    .status
                                                    == "terminology_issue"
                                                ):

                                                    translation_validation = (
                                                        GroundingValidationResult(
                                                            status="unsupported",
                                                            confidence=1.0,
                                                            reason=(
                                                                translation_embedded_claim_term_guard
                                                                .reason
                                                            ),
                                                            issues=(
                                                                translation_embedded_claim_term_guard
                                                                .issues
                                                            ),
                                                        )
                                                    )

                                                elif (
                                                    translation_grounding_claim_precheck
                                                    .status
                                                    == "no_claims"
                                                    and
                                                    translation_embedded_claim_term_guard
                                                    .status
                                                    == "clear"
                                                ):

                                                    translation_validation = (
                                                        GroundingValidationResult(
                                                            status="supported",
                                                            confidence=1.0,
                                                            reason=(
                                                                "No explicit or embedded "
                                                                "factual claim required "
                                                                "semantic grounding "
                                                                "validation."
                                                            ),
                                                            issues=[],
                                                        )
                                                    )

                                                else:

                                                    translation_validation = (
                                                        self.response_grounding_validator
                                                        .validate(
                                                            response=(
                                                                translated_candidate
                                                            ),
                                                            knowledge_context=(
                                                                self.last_repair_evidence_context
                                                            ),
                                                            task_name=(
                                                                "response_validator_repair"
                                                            ),
                                                        )
                                                    )


                                                self.last_evidence_safe_translation_semantic_validation = (
                                                    translation_validation
                                                )

                                                # -------------------------
                                                # Translation precision
                                                # -------------------------

                                                translation_precision_guard = (
                                                    self.grounding_precision_guard
                                                    .evaluate(
                                                        response=(
                                                            translated_candidate
                                                        ),
                                                        knowledge_context=(
                                                            self.last_repair_evidence_context
                                                        ),
                                                    )
                                                )

                                                self.last_evidence_safe_translation_precision_guard = (
                                                    translation_precision_guard
                                                )

                                                translation_validation = (
                                                    self.grounding_precision_veto_policy
                                                    .apply(
                                                        validation=(
                                                            translation_validation
                                                        ),
                                                        precision=(
                                                            translation_precision_guard
                                                        ),
                                                    )
                                                )

                                                self.last_evidence_safe_translation_validation = (
                                                    translation_validation
                                                )

                                                # -------------------------
                                                # Translation guiding questions
                                                #
                                                # Validate against the SAME
                                                # verified repair evidence
                                                # boundary used by translation
                                                # grounding and precision.
                                                # -------------------------

                                                if (
                                                    translation_validation.status
                                                    == "supported"
                                                ):
                                                    translation_guiding_question_grounding = (
                                                        self._validate_guiding_questions(
                                                            response=(
                                                                translated_candidate
                                                            ),
                                                            knowledge_context=(
                                                                self.last_repair_evidence_context
                                                            ),
                                                        )
                                                    )

                                                    self.last_translation_guiding_question_grounding = (
                                                        translation_guiding_question_grounding
                                                    )


                                                # -------------------------
                                                # Translation language gate
                                                # -------------------------

                                                translation_language_consistency = (
                                                    self.language_consistency_guard
                                                    .evaluate(
                                                        response=(
                                                            translated_candidate
                                                        ),
                                                    )
                                                )

                                                self.last_evidence_safe_translation_language_consistency = (
                                                    translation_language_consistency
                                                )   

                                                translation_pedagogical_precheck = None
                                                translation_pedagogical_validation = None

                                                # -------------------------
                                                # Translation pedagogy
                                                #
                                                # Do not spend another
                                                # pedagogical validation call
                                                # when grounding or language
                                                # already failed.
                                                # -------------------------

                                                if (
                                                    translation_validation.status
                                                    == "supported"
                                                    and
                                                    translation_language_consistency
                                                    .status
                                                    == "consistent"
                                                ):

                                                    translation_pedagogical_precheck = (
                                                        self.response_mode_pedagogical_precheck
                                                        .evaluate(
                                                            response=(
                                                                translated_candidate
                                                            ),
                                                            strategy_name=(
                                                                strategy.name
                                                            ),
                                                            response_mode=(
                                                                tutoring_response_mode
                                                            ),
                                                        )
                                                    )

                                                    if (
                                                        translation_pedagogical_precheck
                                                        .status
                                                        == "valid"
                                                    ):

                                                        translation_pedagogical_validation = (
                                                            PedagogicalValidationResult(
                                                                status="valid",
                                                                confidence=1.0,
                                                                reason=(
                                                                    "Validated by deterministic "
                                                                    "pedagogical precheck."
                                                                ),
                                                                issues=[],
                                                            )
                                                        )

                                                    elif (
                                                        translation_pedagogical_precheck
                                                        .status
                                                        == "violation"
                                                    ):

                                                        translation_pedagogical_validation = (
                                                            PedagogicalValidationResult(
                                                                status="violation",
                                                                confidence=1.0,
                                                                reason=(
                                                                    translation_pedagogical_precheck
                                                                    .reason
                                                                ),
                                                                issues=(
                                                                    translation_pedagogical_precheck
                                                                    .issues
                                                                ),
                                                            )
                                                        )

                                                    else:

                                                        translation_pedagogical_validation = (
                                                            self.response_mode_pedagogical_validator
                                                            .validate(
                                                                response=(
                                                                    translated_candidate
                                                                ),
                                                                scaffolding_level=(
                                                                    self.state
                                                                    .scaffolding_level
                                                                ),
                                                                strategy_name=(
                                                                    strategy.name
                                                                ),
                                                                strategy_instruction=(
                                                                    strategy.instruction
                                                                ),
                                                                intervention_name=(
                                                                    intervention_name
                                                                ),
                                                                intervention_instruction=(
                                                                    intervention_instruction
                                                                ),
                                                                response_mode=(
                                                                    tutoring_response_mode
                                                                ),
                                                                response_mode_instruction=(
                                                                    tutoring_response_mode_instruction
                                                                ),
                                                                task_name=(
                                                                    "pedagogical_validator_repair"
                                                                ),
                                                            )
                                                        )


                                                self.last_evidence_safe_translation_pedagogical_precheck = (
                                                    translation_pedagogical_precheck
                                                )

                                                # -------------------------
                                                # Translation guiding-question
                                                # acceptance gate
                                                # -------------------------

                                                if (
                                                    translation_guiding_question_grounding
                                                    is not None
                                                    and
                                                    translation_pedagogical_validation
                                                    is not None
                                                ):
                                                    translation_pedagogical_validation = (
                                                        self._apply_guiding_question_acceptance_gate(
                                                            guiding_question_grounding=(
                                                                translation_guiding_question_grounding
                                                            ),
                                                            pedagogical_validation=(
                                                                translation_pedagogical_validation
                                                            ),
                                                        )
                                                    )

                                                self.last_evidence_safe_translation_pedagogical_validation = (
                                                    translation_pedagogical_validation
                                                )

                                                # -------------------------
                                                # Translation rejection
                                                # diagnostics
                                                # -------------------------

                                                if (
                                                    translation_validation.status
                                                    != "supported"
                                                ):

                                                    if (
                                                        self.last_evidence_safe_translation_semantic_validation
                                                        is not None
                                                        and
                                                        self.last_evidence_safe_translation_semantic_validation
                                                        .status
                                                        == "supported"
                                                        and
                                                        translation_precision_guard
                                                        .status
                                                        == "precision_issue"
                                                    ):
                                                        self.last_evidence_safe_translation_rejection_gate = (
                                                            "precision"
                                                        )

                                                    else:
                                                        self.last_evidence_safe_translation_rejection_gate = (
                                                            "grounding"
                                                        )

                                                elif (
                                                    translation_language_consistency
                                                    .status
                                                    != "consistent"
                                                ):
                                                    self.last_evidence_safe_translation_rejection_gate = (
                                                        "language"
                                                    )

                                                elif (
                                                    translation_guiding_question_grounding
                                                    is None
                                                    or
                                                    not translation_guiding_question_grounding
                                                    .is_safe
                                                ):
                                                    self.last_evidence_safe_translation_rejection_gate = (
                                                        "guiding_question"
                                                    )


                                                elif (
                                                    translation_pedagogical_validation
                                                    is None
                                                    or
                                                    translation_pedagogical_validation
                                                    .status
                                                    != "valid"
                                                ):
                                                    self.last_evidence_safe_translation_rejection_gate = (
                                                        "pedagogy"
                                                    )

                                                else:
                                                    self.last_evidence_safe_translation_rejection_gate = (
                                                        None
                                                    )                                                

                                                # =========================
                                                # ACCEPT TRANSLATION
                                                # =========================

                                                if (
                                                    translation_validation.status
                                                    == "supported"
                                                    and
                                                    translation_language_consistency
                                                    .status
                                                    == "consistent"
                                                    and
                                                    translation_guiding_question_grounding
                                                    is not None
                                                    and
                                                    translation_guiding_question_grounding
                                                    .is_safe
                                                    and
                                                    translation_pedagogical_validation
                                                    is not None
                                                    and
                                                    translation_pedagogical_validation
                                                    .status
                                                    == "valid"
                                                ):

                                                    final_evidence_safe_answer = (
                                                        translated_candidate
                                                    )

                                                    self.last_evidence_safe_translation_used = (
                                                        True
                                                    )

                                                    self.last_evidence_safe_translation_rejection_gate = (
                                                        None
                                                    )

                                                    accepted_grounding_claim_precheck = (
                                                        translation_grounding_claim_precheck
                                                    )

                                                    accepted_embedded_claim_term_guard = (
                                                        translation_embedded_claim_term_guard
                                                    )

                                                    accepted_validation = (
                                                        translation_validation
                                                    )

                                                    accepted_precision_guard = (
                                                        translation_precision_guard
                                                    )

                                                    accepted_pedagogical_precheck = (
                                                        translation_pedagogical_precheck
                                                    )

                                                    accepted_pedagogical_validation = (
                                                        translation_pedagogical_validation
                                                    )

                                        # =================================
                                        # ACCEPT SAFE FINAL RESPONSE
                                        #
                                        # If translation is unavailable or
                                        # rejected, this remains the already
                                        # validated English evidence-safe
                                        # candidate.
                                        # =================================

                                        answer = (
                                            final_evidence_safe_answer
                                        )

                                        self.last_response_repaired = True

                                        self.last_evidence_safe_repair_used = (
                                            True
                                        )

                                        self.last_repair_failed = False
                                        self.last_repair_failure_type = None
                                        self.last_repair_failure_reason = None

                                        # Final repair telemetry describes
                                        # the response actually accepted.

                                        repair_grounding_claim_precheck = (
                                            accepted_grounding_claim_precheck
                                        )

                                        repair_embedded_claim_term_guard = (
                                            accepted_embedded_claim_term_guard
                                        )

                                        repair_validation = (
                                            accepted_validation
                                        )

                                        repair_grounding_precision_guard_result = (
                                            accepted_precision_guard
                                        )

                                        repair_pedagogical_precheck = (
                                            accepted_pedagogical_precheck
                                        )

                                        repair_pedagogical_validation = (
                                            accepted_pedagogical_validation
                                        )

                                        evidence_safe_accepted = True

                            # =========================================
                            # FINAL FAIL-CLOSED
                            # =========================================

                            if not evidence_safe_accepted:

                                self.last_repair_failed = True

                                if (
                                    repair_validation.status
                                    != "supported"
                                ):

                                    self.last_repair_failure_type = (
                                        "grounding"
                                    )

                                    self.last_repair_failure_reason = (
                                        repair_validation.reason
                                    )

                                    answer = (
                                        self._build_grounding_fallback(
                                            learner_message=(
                                                user_message
                                            ),
                                        )
                                    )

                                else:

                                    self.last_repair_failure_type = (
                                        "pedagogical"
                                    )

                                    if (
                                        repair_pedagogical_validation
                                        is not None
                                    ):

                                        self.last_repair_failure_reason = (
                                            repair_pedagogical_validation.reason
                                        )

                                    else:

                                        self.last_repair_failure_reason = (
                                            "Repaired response did not "
                                            "obtain a valid pedagogical "
                                            "evaluation."
                                        )

                                    answer = (
                                        self._build_pedagogical_fallback(
                                            learner_message=(
                                                user_message
                                            ),
                                            strategy_name=(
                                                strategy.name
                                            ),
                                        )
                                    )

                                # Any failed repair ends the current
                                # tutoring sequence.
                                self.original_question = None
                                self.waiting_for_response = False


        # =====================================================
        # STEP 16.17.1 — LANGUAGE CONSISTENCY DETECTION
        # Detection only. Do not modify the final answer.
        # =====================================================

        self.last_language_consistency = (
            self.language_consistency_guard
            .evaluate(
                response=answer,
            )
        )

        # =====================================================
        # RESPONSE QUALITY TELEMETRY
        # =====================================================

        response_quality = (
            self.response_quality_guard
            .evaluate(
                answer
            )
        )

        self.last_response_quality = (
            response_quality
        )

        response_quality_policy = (
            self.response_quality_policy
            .evaluate(
                response_quality
            )
        )

        self.last_response_quality_policy = (
            response_quality_policy
        )

        response_quality_record = (
            ResponseQualityTurnRecord(
                turn_number=(
                    self.state.turn_count
                ),
                quality_status=(
                    response_quality.status
                ),
                policy_status=(
                    response_quality_policy.status
                ),
                issue_count=(
                    response_quality_policy.issue_count
                ),
                requires_attention=(
                    response_quality_policy
                    .requires_attention
                ),
                character_count=(
                    response_quality.character_count
                ),
                token_like_count=(
                    response_quality.token_like_count
                ),
                question_count=(
                    response_quality.question_count
                ),
                sentence_count=(
                    response_quality.sentence_count
                ),
                too_long=(
                    response_quality.too_long
                ),
                too_short=(
                    response_quality.too_short
                ),
                too_many_questions=(
                    response_quality
                    .too_many_questions
                ),
                repetitive=(
                    response_quality.repetitive
                ),
                issues=tuple(
                    response_quality.issues
                ),
                max_characters=(
                    self.response_quality_guard
                    .max_characters
                ),
                min_characters=(
                    self.response_quality_guard
                    .min_characters
                ),
                max_questions=(
                    self.response_quality_guard
                    .max_questions
                ),
                detect_repetition=(
                    self.response_quality_guard
                    .detect_repetition
                ),
            )
        )

        self.response_quality_history.add(
            response_quality_record
        )
        # =====================================================
        # STORE TURN VALIDATION STATE
        # =====================================================

        self.last_grounding_claim_precheck = (
            grounding_claim_precheck
        )

        self.last_embedded_claim_term_guard = (
            embedded_claim_term_guard
        )
        self.last_grounding_validation = (
            grounding_validation
        )

        self.last_grounding_precision_guard = (
            grounding_precision_guard_result
        )

        self.last_guiding_question_grounding = (
            guiding_question_grounding
        )

        self.last_pedagogical_precheck = (
            pedagogical_precheck
        )

        self.last_pedagogical_validation = (
            pedagogical_validation
        )

        self.last_repair_grounding_claim_precheck = (
            repair_grounding_claim_precheck
        )

        self.last_repair_embedded_claim_term_guard = (
            repair_embedded_claim_term_guard
        )

        self.last_repair_validation = (
            repair_validation
        )

        self.last_repair_grounding_precision_guard = (
            repair_grounding_precision_guard_result
        )

        self.last_repair_guiding_question_grounding = (
            repair_guiding_question_grounding
        )

        self.last_repair_pedagogical_precheck = (
            repair_pedagogical_precheck
        )

        self.last_repair_pedagogical_validation = (
            repair_pedagogical_validation
        )

        # =====================================================
        # COLLECT AI USAGE
        # =====================================================

        self.last_ai_usage_summary = (
            get_ai_usage_summary()
        )

        # =====================================================
        # SAVE CONVERSATION
        # =====================================================
        if not self.last_repair_failed:

            self.memory.add_user_message(
                user_message
            )

            self.memory.add_assistant_message(
                answer
            )

        self._record_learner_progress_history(
            starts_new_sequence=(
                starts_new_sequence_for_history
            ),
        )

        return answer


    # =========================================================
    # LEARNER PROGRESS HISTORY
    # =========================================================

    def _record_learner_progress_history(
        self,
        *,
        starts_new_sequence: bool,
    ):
        """
        Record exactly one observation-only learner progress
        snapshot for one completed Tutor turn.

        This method must not affect:
        - routing
        - evaluation
        - scaffolding
        - retrieval
        - prompting
        - Tutor generation
        """

        if not isinstance(
            starts_new_sequence,
            bool,
        ):
            raise TypeError(
                "starts_new_sequence must be bool."
            )

        current_turn_evaluation = None

        if (
            self.last_evaluation_result
            is not None
        ):
            current_turn_evaluation = (
                self.last_evaluation_result
                .classification
            )

        learner_turn_intent = None

        if (
            self.last_learner_turn_intent
            is not None
        ):
            learner_turn_intent = (
                self.last_learner_turn_intent
                .intent
            )

        learner_turn_route = None

        if (
            self.last_learner_turn_routing
            is not None
        ):
            learner_turn_route = (
                self.last_learner_turn_routing
                .route
            )

        snapshot = (
            self.learner_progress_snapshot_builder
            .build(
                state=self.state,
                current_turn_evaluation=(
                    current_turn_evaluation
                ),
                learner_turn_intent=(
                    learner_turn_intent
                ),
                learner_turn_route=(
                    learner_turn_route
                ),
            )
        )

        return (
            self.learner_progress_history
            .record(
                snapshot,
                starts_new_sequence=(
                    starts_new_sequence
                ),
            )
        )


    # =========================================================
    # FALLBACKS
    # =========================================================

    def _build_out_of_course_fallback(
        self,
        learner_message: str,
    ) -> str:
        """
        Deterministic response when the active course
        does not contain sufficiently relevant grounded
        knowledge for the learner's question.

        No Tutor LLM is called in this path.
        """

        has_thai = any(
            "\u0E00" <= char <= "\u0E7F"
            for char in learner_message
        )

        if has_thai:

            course_label = (
                self.course_profile.description
                or
                self.course_profile.course_name
            )

            return (
                f"ฉันไม่พบเนื้อหาใน {course_label} "
                "ที่รองรับคำถามนี้อย่างเพียงพอ "
                "กรุณาถามคำถามที่เกี่ยวข้องกับ"
                "เนื้อหาในรายวิชาที่กำลังใช้งานอยู่"
            )

        return (
            "I could not find sufficiently relevant "
            f"content in the active course "
            f"'{self.course_profile.course_name}' "
            "to support this question. "
            "Please ask a question related to the "
            "currently active course."
        )

    def _build_grounding_fallback(
        self,
        learner_message: str,
    ) -> str:

        has_thai = any(
            "\u0E00" <= char <= "\u0E7F"
            for char in learner_message
        )

        if has_thai:

            return (
                "ข้อมูลจากเอกสารรายวิชาที่มีอยู่ยังไม่เพียงพอ"
                "ให้ฉันอธิบายประเด็นนี้ได้อย่างมั่นใจ "
                "ลองระบุส่วนของเนื้อหาที่ต้องการศึกษา"
                "ให้เจาะจงขึ้นอีกเล็กน้อยได้ไหม?"
            )

        return (
            "The available course material does not provide "
            "enough verified information for me to explain "
            "this point confidently. Could you make the "
            "learning point a little more specific?"
        )

    def _build_pedagogical_fallback(
        self,
        learner_message: str,
        strategy_name: str,
    ) -> str:

        has_thai = any(
            "\u0E00" <= char <= "\u0E7F"
            for char in learner_message
        )

        if has_thai:

            if strategy_name == "guiding_question":

                return (
                    "จากสิ่งที่คุณรู้อยู่แล้ว "
                    "คุณคิดว่าประเด็นสำคัญที่สุด"
                    "ที่ควรพิจารณาเพื่อหาคำตอบคืออะไร?"
                )

            if strategy_name == "hint":

                return (
                    "ลองพิจารณาข้อมูลสำคัญ"
                    "เพียงจุดเดียวก่อน "
                    "คุณคิดว่าข้อมูลนั้นช่วยให้"
                    "เข้าใกล้คำตอบอย่างไร?"
                )

            return (
                "ลองทบทวนแนวคิดสำคัญ"
                "จากข้อมูลที่มีอยู่ "
                "แล้วอธิบายความเข้าใจ"
                "ของคุณอีกครั้งได้ไหม?"
            )

        if strategy_name == "guiding_question":

            return (
                "Based on what you already know, "
                "what is the single most important "
                "point to consider here?"
            )

        if strategy_name == "hint":

            return (
                "Focus on one relevant clue first. "
                "How might that clue help you move "
                "toward the answer?"
            )

        return (
            "Review the key concept from the "
            "available information. "
            "How would you explain your "
            "understanding now?"
        )



    def _is_retryable_evidence_usability_failure(
        self,
        selection: RepairEvidenceSelectionResult,
    ) -> bool:

        if not isinstance(
            selection,
            RepairEvidenceSelectionResult,
        ):
            raise TypeError(
                "selection must be "
                "RepairEvidenceSelectionResult."
            )

        if selection.status != "invalid":
            return False

        if (
            selection.reason
            != (
                "Repair evidence selector proposed "
                "unusable source evidence."
            )
        ):
            return False

        return any(
            "unusable" in issue.lower()
            for issue in selection.issues
            if isinstance(issue, str)
        )


    def _validate_guiding_questions(
        self,
        response: str,
        knowledge_context: str,
    ) -> GuidingQuestionGroundingResult:

        if not isinstance(
            response,
            str,
        ):
            raise TypeError(
                "response must be str."
            )

        if not isinstance(
            knowledge_context,
            str,
        ):
            raise TypeError(
                "knowledge_context must be str."
            )

        return (
            self.guiding_question_grounding_validator
            .validate(
                response=response,
                knowledge_context=knowledge_context,
            )
        )

 

    def _apply_guiding_question_acceptance_gate(
        self,
        *,
        guiding_question_grounding: GuidingQuestionGroundingResult,
        pedagogical_validation: PedagogicalValidationResult,
    ) -> PedagogicalValidationResult:

        if not isinstance(
            guiding_question_grounding,
            GuidingQuestionGroundingResult,
        ):
            raise TypeError(
                "guiding_question_grounding must be "
                "GuidingQuestionGroundingResult."
            )

        if not isinstance(
            pedagogical_validation,
            PedagogicalValidationResult,
        ):
            raise TypeError(
                "pedagogical_validation must be "
                "PedagogicalValidationResult."
            )

        # -----------------------------------------------------
        # Safe guiding-question states preserve the existing
        # pedagogical result byte-for-byte / object-for-object.
        # -----------------------------------------------------

        if guiding_question_grounding.is_safe:
            return pedagogical_validation

        # -----------------------------------------------------
        # Unsupported or invalid guiding questions fail closed.
        #
        # Treat this as an acceptance/pedagogical violation,
        # NOT as a factual-statement grounding failure.
        # This preserves the frozen factual grounding contract
        # and routes through the existing repair architecture.
        # -----------------------------------------------------

        combined_issues = list(
            pedagogical_validation.issues
        )

        guiding_issues = [
            (
                "Knowledge-bounded guiding question: "
                f"{issue}"
            )
            for issue
            in guiding_question_grounding.issues
        ]

        if not guiding_issues:
            guiding_issues = [
                (
                    "Knowledge-bounded guiding-question "
                    "validation rejected the response "
                    f"with status "
                    f"'{guiding_question_grounding.status}'."
                )
            ]

        combined_issues.extend(
            guiding_issues
        )

        combined_issues = list(
            dict.fromkeys(
                combined_issues
            )
        )

        if (
            guiding_question_grounding.status
            == "unsupported"
        ):
            reason = (
                "The Tutor response contains a factual "
                "guiding question whose requested answer "
                "is not sufficiently supported by course "
                "knowledge."
            )

        else:
            reason = (
                "Guiding-question grounding validation "
                "could not establish a safe answerability "
                "boundary and therefore failed closed."
            )

        return PedagogicalValidationResult(
            status="violation",
            confidence=1.0,
            reason=reason,
            issues=combined_issues,
        )

    def _run_response_repair(
        self,
        **kwargs,
    ):
        """
        Run one response-repair generation call.

        Provider/runtime failures are observable through
        repair telemetry but remain provider exceptions.

        This helper does not:
        - create a fallback response
        - retry the provider
        - clear the active learning sequence
        - write ConversationMemory
        """

        try:
            return (
                self.response_mode_repair_service
                .repair(
                    **kwargs
                )
            )

        except Exception as exc:

            self.last_repair_failed = True
            self.last_repair_failure_type = (
                "repair_provider"
            )
            self.last_repair_failure_reason = (
                str(exc)
            )

            raise




    def _prepare_llm_repair_context(
        self,
        learner_message: str,
        knowledge_context: str,
        grounding_validation: GroundingValidationResult,
        *,
        force_verified_evidence: bool = False,
        additional_validation_issues: tuple[str, ...] = (),
    ) -> tuple[
        RepairEvidenceSelectionResult | None,
        str | None,
    ]:
        """
        Prepare the factual evidence boundary for an
        LLM repair.

        Supported factual grounding:
            Preserve the existing repair behavior and
            return the full knowledge context unless
            verified evidence is explicitly forced.

        Partially supported / unsupported grounding:
            Select exact course evidence first and return
            only the verified evidence pack.

        No verified evidence:
            Return None as the repair context so the
            runtime can fail closed without asking the
            repair model to invent content.
        """

        if not isinstance(
            learner_message,
            str,
        ):
            raise TypeError(
                "learner_message must be str."
            )

        if not isinstance(
            knowledge_context,
            str,
        ):
            raise TypeError(
                "knowledge_context must be str."
            )

        if not isinstance(
            grounding_validation,
            GroundingValidationResult,
        ):
            raise TypeError(
                "grounding_validation must be "
                "GroundingValidationResult."
            )

        if not isinstance(
            force_verified_evidence,
            bool,
        ):
            raise TypeError(
                "force_verified_evidence must be bool."
            )

        if (
            not isinstance(
                additional_validation_issues,
                tuple,
            )
            or
            not all(
                isinstance(issue, str)
                for issue
                in additional_validation_issues
            )
        ):
            raise TypeError(
                "additional_validation_issues must "
                "be tuple[str, ...]."
            )

        # -------------------------------------------------
        # Preserve the frozen supported-repair behavior
        # unless verified evidence is explicitly forced.
        # -------------------------------------------------
        
        if (
            grounding_validation.status
            == "supported"
            and
            not force_verified_evidence
        ):
            return (
                None,
                knowledge_context,
            )
 
        # -------------------------------------------------
        # Factual repair:
        # select source evidence before generation.
        # -------------------------------------------------

        # -------------------------------------------------
        # BOUNDED CUMULATIVE EVIDENCE RESELECTION
        #
        # Maximum:
        #     3 total selector attempts
        #
        # Retry only:
        #     deterministic usability failures
        #
        # Never retry:
        #     - unverifiable/fabricated evidence
        #     - provider failure
        #     - malformed structured output
        #     - genuine no_evidence
        #
        # Each retry receives all prior deterministic
        # rejection feedback.
        # -------------------------------------------------

        self.last_repair_evidence_selection_attempts = 0
        self.last_repair_evidence_retry_used = False
        self.last_repair_evidence_first_failure = None

        max_selection_attempts = 3

        base_validation_issues = tuple(
            dict.fromkeys(
                tuple(
                    grounding_validation.issues
                )
                +
                additional_validation_issues
            )
        )

        reselection_feedback: list[str] = []

        selection = (
            self.repair_evidence_selector.select(
                learner_message=(
                    learner_message
                ),
                knowledge_context=(
                    knowledge_context
                ),
                validation_issues=(
                    base_validation_issues
                ),
            )
        )

        self.last_repair_evidence_selection_attempts = 1

        while (
            self.last_repair_evidence_selection_attempts
            < max_selection_attempts
            and
            self._is_retryable_evidence_usability_failure(
                selection
            )
        ):

            if (
                self.last_repair_evidence_first_failure
                is None
            ):
                self.last_repair_evidence_first_failure = (
                    selection
                )

            self.last_repair_evidence_retry_used = True

            rejected_attempt = (
                self.last_repair_evidence_selection_attempts
            )

            rejection_detail = " | ".join(
                issue
                for issue in selection.issues
                if isinstance(
                    issue,
                    str,
                )
            )

            retry_feedback = (
                "EVIDENCE RESELECTION REQUIRED: "
                f"Selector attempt {rejected_attempt} "
                "was rejected by the deterministic "
                "evidence usability guard. "
                "Select different exact course evidence. "
                "The new evidence must be clean, complete, "
                "self-contained, and usable without "
                "reconstruction. "
                "Do not reuse ANY evidence excerpt "
                "identified in previous deterministic "
                "rejection feedback. "
                "Do not repair, rewrite, restore, complete, "
                "or infer missing source text. "
                "If no sufficient clean exact evidence "
                "exists, return no evidence."
            )

            if rejection_detail:
                retry_feedback = (
                    retry_feedback
                    + " Deterministic rejection: "
                    + rejection_detail
                )

            reselection_feedback.append(
                retry_feedback
            )

            retry_validation_issues = (
                base_validation_issues
                + tuple(
                    reselection_feedback
                )
            )

            selection = (
                self.repair_evidence_selector.select(
                    learner_message=(
                        learner_message
                    ),
                    knowledge_context=(
                        knowledge_context
                    ),
                    validation_issues=(
                        retry_validation_issues
                    ),
                )
            )

            self.last_repair_evidence_selection_attempts += 1

        if not selection.has_verified_evidence:
            return (
                selection,
                None,
            )

        # -------------------------------------------------
        # Build a repair-only evidence context.
        #
        # Each evidence quote remains byte-for-byte
        # present inside this context so later provenance
        # validation can verify it.
        # -------------------------------------------------

        evidence_blocks = []

        for index, quote in enumerate(
            selection.evidence_quotes,
            start=1,
        ):
            evidence_blocks.append(
                (
                    f"[Verified Evidence {index}]\n"
                    f"{quote}"
                )
            )

        evidence_context = "\n\n".join(
            evidence_blocks
        )

        return (
            selection,
            evidence_context,
        )
    # =========================================================
    # RESET
    # =========================================================

    def reset(self):

        self.memory.clear()
        self.state.reset()

        self.learner_progress_history.clear()

        self.response_quality_history.clear()

        self.last_evidence_safe_repair_candidate = None
        self.last_evidence_safe_repair_used = False

        self.last_evidence_safe_translation_result = None
        self.last_evidence_safe_translation_candidate = None
        self.last_evidence_safe_translation_used = False

        self.last_evidence_safe_translation_semantic_validation = None
        self.last_evidence_safe_translation_validation = None
        self.last_evidence_safe_translation_precision_guard = None
        self.last_evidence_safe_translation_language_consistency = None
        self.last_evidence_safe_translation_pedagogical_precheck = None
        self.last_evidence_safe_translation_pedagogical_validation = None
        self.last_evidence_safe_translation_rejection_gate = None

        self.last_response_quality = None
        self.last_response_quality_policy = None

        self.last_learner_turn_intent = None
        self.last_learner_turn_routing = None
        self.last_tutoring_response_mode = None
        self.last_tutoring_response_mode_instruction = None

        self.last_guiding_question_grounding = None
        self.last_repair_guiding_question_grounding = None
        self.last_evidence_safe_guiding_question_grounding = None
        self.last_translation_guiding_question_grounding = None

        reset_ai_usage()

        self.last_repair_attempts = 0
        self.last_repair_limit_reached = False

        self.last_language_consistency = None

        self.original_question = None
        self.waiting_for_response = False

        self.last_decision = None
        self.last_evaluation_result = None
        self.last_intervention = None

        self._pending_evaluated_retry_message = None
        self._pending_evaluated_retry_original_question = None
        self._pending_evaluated_retry_tutor_question = None
        self._pending_evaluated_retry_evaluation_result = None
        self._pending_evaluated_retry_decision = None


        self.last_system_prompt = None
        self.last_retrieval_query = None
        self.last_generated_answer = None

        self.last_grounding_precision_instruction = None

        self.last_knowledge_result = None
        self.last_relevance_result = None
        self.last_grounding_status = None

        self.last_grounding_claim_precheck = None
        self.last_embedded_claim_term_guard = None
        self.last_grounding_validation = None
        self.last_grounding_precision_guard = None

        self.last_pedagogical_precheck = None
        self.last_pedagogical_validation = None

        self.last_response_repaired = False
        self.last_repair_candidate = None

        self.last_repair_evidence_selection = None
        self.last_repair_evidence_context = None

        self.last_repair_evidence_selection_attempts = 0
        self.last_repair_evidence_retry_used = False
        self.last_repair_evidence_first_failure = None

        self.last_repair_grounding_claim_precheck = None
        
        self.last_repair_embedded_claim_term_guard = None
        self.last_repair_validation = None
        self.last_repair_grounding_precision_guard = None

        self.last_repair_pedagogical_precheck = None
        self.last_repair_pedagogical_validation = None

        self.last_escalation_decision = None
        self.last_repair_mode = None
        self.last_deterministic_replacements = []

        self.last_repair_failed = False
        self.last_repair_failure_type = None
        self.last_repair_failure_reason = None

        self.last_ai_usage_summary = None

    # =========================================================
    # BASIC GETTERS
    # =========================================================

    def get_last_system_prompt(self):
        return self.last_system_prompt

    def get_level(self):
        return self.state.scaffolding_level

    def get_evaluation(self):
        return self.state.last_evaluation

    # =========================================================
    # DEBUG INFO
    # =========================================================

    def get_debug_info(self) -> dict:

        strategy = get_strategy(
            self.state.scaffolding_level
        )

        # -------------------------
        # Decision
        # -------------------------

        decision_reason = None
        decision_action = None

        if self.last_decision is not None:

            decision_reason = (
                self.last_decision.reason
            )

            decision_action = (
                self.last_decision.action
            )

        # -------------------------
        # Evaluation
        # -------------------------

        evaluation_value = None
        evaluation_confidence = None
        evaluation_reason = None
        evaluation_misconception = None

        if (
            self.last_evaluation_result
            is not None
        ):

            evaluation_value = (
                self.last_evaluation_result
                .classification
            )

            evaluation_confidence = (
                self.last_evaluation_result
                .confidence
            )

            evaluation_reason = (
                self.last_evaluation_result
                .reason
            )

            evaluation_misconception = (
                self.last_evaluation_result
                .misconception
            )

        # -------------------------
        # Misconceptions
        # -------------------------

        active_misconceptions = (
            self.state.get_active_misconceptions()
        )

        # -------------------------
        # Intervention
        # -------------------------

        intervention_name = None

        if self.last_intervention is not None:

            intervention_name = (
                self.last_intervention.name
            )

        # -------------------------
        # Knowledge
        # -------------------------

        knowledge_sources = []
        knowledge_citations = []

        if self.last_knowledge_result is not None:

            knowledge_sources = [
                {
                    "source": item.source,
                    "page": item.page,
                    "distance": item.distance,
                }
                for item
                in self.last_knowledge_result.sources
            ]

            knowledge_citations = [
                {
                    "source": item.source,
                    "page": item.page,
                }
                for item
                in self.last_knowledge_result.citations
            ]

        # -------------------------
        # Citation
        # -------------------------

        show_citations = (
            self.citation_policy.should_show(
                scaffolding_level=(
                    self.state.scaffolding_level
                ),
                intervention=(
                    intervention_name
                ),
            )
        )

        # -------------------------
        # Relevance
        # -------------------------

        relevance_is_relevant = None
        relevance_confidence = None
        relevance_reason = None

        if self.last_relevance_result is not None:

            relevance_is_relevant = (
                self.last_relevance_result.is_relevant
            )

            relevance_confidence = (
                self.last_relevance_result.confidence
            )

            relevance_reason = (
                self.last_relevance_result.reason
            )

        # -------------------------
        # Grounding availability
        # -------------------------

        grounding_has_knowledge = None
        grounding_reason = None

        if self.last_grounding_status is not None:

            grounding_has_knowledge = (
                self.last_grounding_status.has_knowledge
            )

            grounding_reason = (
                self.last_grounding_status.reason
            )

        # -------------------------
        # Grounding claim precheck
        # -------------------------

        grounding_claim_precheck_status = None
        grounding_claim_precheck_reason = None

        if (
            self.last_grounding_claim_precheck
            is not None
        ):

            grounding_claim_precheck_status = (
                self.last_grounding_claim_precheck.status
            )

            grounding_claim_precheck_reason = (
                self.last_grounding_claim_precheck.reason
            )

        # -------------------------
        # Embedded claim /
        # technical term guard
        # -------------------------

        embedded_claim_term_guard_status = None
        embedded_claim_term_guard_reason = None
        embedded_claim_term_guard_issues = []

        if (
            self.last_embedded_claim_term_guard
            is not None
        ):

            embedded_claim_term_guard_status = (
                self.last_embedded_claim_term_guard.status
            )

            embedded_claim_term_guard_reason = (
                self.last_embedded_claim_term_guard.reason
            )

            embedded_claim_term_guard_issues = (
                self.last_embedded_claim_term_guard.issues
            )
        # -------------------------
        # Response grounding validation
        # -------------------------

        validation_status = None
        validation_confidence = None
        validation_reason = None
        validation_issues = []

        if self.last_grounding_validation is not None:

            validation_status = (
                self.last_grounding_validation.status
            )

            validation_confidence = (
                self.last_grounding_validation.confidence
            )

            validation_reason = (
                self.last_grounding_validation.reason
            )

            validation_issues = (
                self.last_grounding_validation.issues
            )

        # -------------------------
        # Pedagogical precheck
        # -------------------------

        pedagogical_precheck_status = None
        pedagogical_precheck_reason = None

        if self.last_pedagogical_precheck is not None:

            pedagogical_precheck_status = (
                self.last_pedagogical_precheck.status
            )

            pedagogical_precheck_reason = (
                self.last_pedagogical_precheck.reason
            )

        # -------------------------
        # Pedagogical validation
        # -------------------------

        pedagogical_status = None
        pedagogical_confidence = None
        pedagogical_reason = None
        pedagogical_issues = []

        if self.last_pedagogical_validation is not None:

            pedagogical_status = (
                self.last_pedagogical_validation.status
            )

            pedagogical_confidence = (
                self.last_pedagogical_validation.confidence
            )

            pedagogical_reason = (
                self.last_pedagogical_validation.reason
            )

            pedagogical_issues = (
                self.last_pedagogical_validation.issues
            )


        # -------------------------
        # Repair embedded claim /
        # technical term guard
        # -------------------------

        repair_embedded_claim_term_guard_status = None
        repair_embedded_claim_term_guard_reason = None
        repair_embedded_claim_term_guard_issues = []

        if (
            self.last_repair_embedded_claim_term_guard
            is not None
        ):

            repair_embedded_claim_term_guard_status = (
                self.last_repair_embedded_claim_term_guard
                .status
            )

            repair_embedded_claim_term_guard_reason = (
                self.last_repair_embedded_claim_term_guard
                .reason
            )

            repair_embedded_claim_term_guard_issues = (
                self.last_repair_embedded_claim_term_guard
                .issues
            )

        # -------------------------
        # Repair grounding claim precheck
        # -------------------------

        repair_grounding_claim_precheck_status = None
        repair_grounding_claim_precheck_reason = None

        if (
            self.last_repair_grounding_claim_precheck
            is not None
        ):

            repair_grounding_claim_precheck_status = (
                self.last_repair_grounding_claim_precheck
                .status
            )

            repair_grounding_claim_precheck_reason = (
                self.last_repair_grounding_claim_precheck
                .reason
            )

        # -------------------------
        # Repair grounding
        # -------------------------

        repair_status = None
        repair_confidence = None
        repair_reason = None
        repair_issues = []

        if self.last_repair_validation is not None:

            repair_status = (
                self.last_repair_validation.status
            )

            repair_confidence = (
                self.last_repair_validation.confidence
            )

            repair_reason = (
                self.last_repair_validation.reason
            )

            repair_issues = (
                self.last_repair_validation.issues
            )

        # -------------------------
        # Repair pedagogical precheck
        # -------------------------

        repair_pedagogical_precheck_status = None
        repair_pedagogical_precheck_reason = None

        if (
            self.last_repair_pedagogical_precheck
            is not None
        ):

            repair_pedagogical_precheck_status = (
                self.last_repair_pedagogical_precheck
                .status
            )

            repair_pedagogical_precheck_reason = (
                self.last_repair_pedagogical_precheck
                .reason
            )

        # -------------------------
        # Repair pedagogical validation
        # -------------------------

        repair_pedagogical_status = None
        repair_pedagogical_confidence = None
        repair_pedagogical_reason = None
        repair_pedagogical_issues = []

        if (
            self.last_repair_pedagogical_validation
            is not None
        ):

            repair_pedagogical_status = (
                self.last_repair_pedagogical_validation
                .status
            )

            repair_pedagogical_confidence = (
                self.last_repair_pedagogical_validation
                .confidence
            )

            repair_pedagogical_reason = (
                self.last_repair_pedagogical_validation
                .reason
            )

            repair_pedagogical_issues = (
                self.last_repair_pedagogical_validation
                .issues
            )

        # -------------------------
        # AI Usage
        # -------------------------

        ai_usage = (
            self.last_ai_usage_summary
            if self.last_ai_usage_summary is not None
            else {
                "calls": 0,
                "total_latency_ms": 0.0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "estimated_cost_usd": 0.0,
                "records": [],
            }
        )

        # =====================================================
        # RETURN DEBUG DATA
        # =====================================================


        escalation_action = None
        escalation_reason = None

        if (
            self.last_escalation_decision
            is not None
        ):

            escalation_action = (
                self.last_escalation_decision.action
            )

            escalation_reason = (
                self.last_escalation_decision.reason
            )


        # -----------------------------------------------------
        # Language consistency telemetry
        # -----------------------------------------------------

        language_consistency_status = None
        language_expected = None
        language_detected = None
        language_reason = None
        language_thai_chars = 0
        language_latin_chars = 0

        if (
            self.last_language_consistency
            is not None
        ):

            language_consistency_status = (
                self.last_language_consistency.status
            )

            language_expected = (
                self.last_language_consistency
                .expected_language
            )

            language_detected = (
                self.last_language_consistency
                .detected_language
            )

            language_reason = (
                self.last_language_consistency.reason
            )

            language_thai_chars = (
                self.last_language_consistency.thai_chars
            )

            language_latin_chars = (
                self.last_language_consistency.latin_chars
            )

        response_style_language = None
        response_style_technical_english = None
        response_style_term_format = None
        response_style_tone = None
        response_style_depth = None
        response_style_question_style = None
        response_style_max_guiding_questions = None

        if (
            self.resolved_response_style
            is not None
        ):

            response_style_language = (
                self.resolved_response_style
                .primary_language
            )

            response_style_technical_english = (
                self.resolved_response_style
                .allow_technical_english
            )

            response_style_term_format = (
                self.resolved_response_style
                .technical_term_format
            )

            response_style_tone = (
                self.resolved_response_style
                .tone
            )

            response_style_depth = (
                self.resolved_response_style
                .explanation_depth
            )

            response_style_question_style = (
                self.resolved_response_style
                .question_style
            )

            response_style_max_guiding_questions = (
                self.resolved_response_style
                .max_guiding_questions
            )


            response_quality_status = None
            response_quality_reason = None
            response_quality_issues = []

            response_quality_character_count = 0
            response_quality_token_like_count = 0
            response_quality_question_count = 0
            response_quality_sentence_count = 0

            response_quality_too_long = False
            response_quality_too_short = False
            response_quality_too_many_questions = False
            response_quality_repetitive = False

            if (
                self.last_response_quality
                is not None
            ):

                response_quality_status = (
                    self.last_response_quality.status
                )

                response_quality_reason = (
                    self.last_response_quality.reason
                )

                response_quality_issues = (
                    self.last_response_quality.issues
                )

                response_quality_character_count = (
                    self.last_response_quality
                    .character_count
                )

                response_quality_token_like_count = (
                    self.last_response_quality
                    .token_like_count
                )

                response_quality_question_count = (
                    self.last_response_quality
                    .question_count
                )

                response_quality_sentence_count = (
                    self.last_response_quality
                    .sentence_count
                )

                response_quality_too_long = (
                    self.last_response_quality
                    .too_long
                )

                response_quality_too_short = (
                    self.last_response_quality
                    .too_short
                )

                response_quality_too_many_questions = (
                    self.last_response_quality
                    .too_many_questions
                )

                response_quality_repetitive = (
                    self.last_response_quality
                    .repetitive
                )

        # -------------------------
        # Response Quality Policy
        # -------------------------

        response_quality_policy_status = None
        response_quality_policy_reason = None
        response_quality_policy_issues = []
        response_quality_policy_source_status = None
        response_quality_policy_issue_count = 0
        response_quality_policy_requires_attention = False

        if (
            self.last_response_quality_policy
            is not None
        ):

            response_quality_policy_status = (
                self.last_response_quality_policy
                .status
            )

            response_quality_policy_reason = (
                self.last_response_quality_policy
                .reason
            )

            response_quality_policy_issues = (
                self.last_response_quality_policy
                .issues
            )

            response_quality_policy_source_status = (
                self.last_response_quality_policy
                .source_status
            )

            response_quality_policy_issue_count = (
                self.last_response_quality_policy
                .issue_count
            )

            response_quality_policy_requires_attention = (
                self.last_response_quality_policy
                .requires_attention
            )

        # -------------------------
        # Response Quality Config
        # -------------------------

        response_quality_config_max_characters = (
            self.response_quality_guard
            .max_characters
        )

        response_quality_config_min_characters = (
            self.response_quality_guard
            .min_characters
        )

        response_quality_config_max_questions = (
            self.response_quality_guard
            .max_questions
        )

        response_quality_config_detect_repetition = (
            self.response_quality_guard
            .detect_repetition
        )

        # -------------------------
        # Response Quality Analytics
        # -------------------------

        response_quality_analytics = (
            self.response_quality_analytics
            .summarize(
                self.response_quality_history
            )
        )

        learner_turn_intent = None
        learner_turn_intent_reason = None
        learner_turn_intent_confidence = 0.0
        learner_turn_intent_signals = ()
        learner_turn_intent_is_question = False

        if (
            self.last_learner_turn_intent
            is not None
        ):

            learner_turn_intent = (
                self.last_learner_turn_intent.intent
            )

            learner_turn_intent_reason = (
                self.last_learner_turn_intent.reason
            )

            learner_turn_intent_confidence = (
                self.last_learner_turn_intent.confidence
            )

            learner_turn_intent_signals = (
                self.last_learner_turn_intent.signals
            )

            learner_turn_intent_is_question = (
                self.last_learner_turn_intent.is_question
            )
        # -----------------------------------------------------
        # Learner turn routing telemetry
        # -----------------------------------------------------

        learner_turn_route = None
        learner_turn_route_reason = None

        learner_turn_should_evaluate = None

        learner_turn_use_current_retrieval = None

        learner_turn_include_previous_context = None

        learner_turn_replace_active_question = None

        if (
            self.last_learner_turn_routing
            is not None
        ):

            learner_turn_route = (
                self.last_learner_turn_routing.route
            )

            learner_turn_route_reason = (
                self.last_learner_turn_routing.reason
            )

            learner_turn_should_evaluate = (
                self.last_learner_turn_routing
                .should_evaluate_response
            )

            learner_turn_use_current_retrieval = (
                self.last_learner_turn_routing
                .use_current_message_for_retrieval
            )

            learner_turn_include_previous_context = (
                self.last_learner_turn_routing
                .include_previous_tutor_context
            )

            learner_turn_replace_active_question = (
                self.last_learner_turn_routing
                .replace_active_question
            )

        # -----------------------------------------------------
        # STEP 16.19.1 — Learner Progress Snapshot
        #
        # Observation only.
        # This snapshot must not affect Tutor behavior.
        # -----------------------------------------------------

        learner_progress_snapshot = (
            self.learner_progress_snapshot_builder
            .build(
                state=self.state,
                current_turn_evaluation=(
                    evaluation_value
                ),
                learner_turn_intent=(
                    learner_turn_intent
                ),
                learner_turn_route=(
                    learner_turn_route
                ),
            )
        )

        # =====================================================
        # LEARNER PROGRESS HISTORY TELEMETRY
        # =====================================================

        learner_progress_history_count = (
            self.learner_progress_history
            .count()
        )

        learner_progress_history_sequence_count = (
            self.learner_progress_history
            .current_sequence_id()
        )

        learner_progress_history_latest = (
            self.learner_progress_history
            .get_latest()
        )

        learner_progress_history_latest_index = None
        learner_progress_history_latest_sequence = None
        learner_progress_history_latest_sequence_start = None
        learner_progress_history_latest_turn = None

        if (
            learner_progress_history_latest
            is not None
        ):

            learner_progress_history_latest_index = (
                learner_progress_history_latest
                .history_index
            )

            learner_progress_history_latest_sequence = (
                learner_progress_history_latest
                .learning_sequence_id
            )

            learner_progress_history_latest_sequence_start = (
                learner_progress_history_latest
                .starts_new_sequence
            )

            learner_progress_history_latest_turn = (
                learner_progress_history_latest
                .snapshot
                .session_turn
            )

        # -------------------------
        # Tutoring Response Mode
        # -------------------------

        tutoring_response_mode = None
        tutoring_response_mode_reason = None
        tutoring_direct_answer_required = None
        tutoring_guiding_question_policy = None
        tutoring_preserve_scaffolding_strategy = None
        tutoring_max_guiding_questions = None

        # This telemetry belongs to the prompt instruction
        # itself and must be safe even when no response-mode
        # decision exists yet.
        tutoring_response_mode_instruction_applied = (
            bool(
                self.last_tutoring_response_mode_instruction
            )
        )

        if (
            self.last_tutoring_response_mode
            is not None
        ):
            tutoring_response_mode = (
                self.last_tutoring_response_mode.mode
            )

            tutoring_response_mode_reason = (
                self.last_tutoring_response_mode.reason
            )

            tutoring_direct_answer_required = (
                self.last_tutoring_response_mode
                .direct_answer_required
            )

            tutoring_guiding_question_policy = (
                self.last_tutoring_response_mode
                .guiding_question_policy
            )

            tutoring_preserve_scaffolding_strategy = (
                self.last_tutoring_response_mode
                .preserve_scaffolding_strategy
            )

            tutoring_max_guiding_questions = (
                self.last_tutoring_response_mode
                .max_guiding_questions
            )

            tutoring_response_mode_instruction_applied = (
                bool(
                    self.last_tutoring_response_mode_instruction
                )
            )
        return {

            "level": (
                self.state.scaffolding_level
            ),

            "strategy": (
                strategy.name
            ),

            "intervention": (
                intervention_name
            ),

            "evaluation": (
                evaluation_value
            ),

            "evaluation_confidence": (
                evaluation_confidence
            ),

            "evaluation_reason": (
                evaluation_reason
            ),

            "evaluation_misconception": (
                evaluation_misconception
            ),

            "decision_action": (
                decision_action
            ),

            "decision_reason": (
                decision_reason
            ),

            "correct_streak": (
                self.state.correct_streak
            ),

            "partial_streak": (
                self.state.partial_streak
            ),

            "failure_streak": (
                self.state.failure_streak
            ),

            "turn": (
                self.state.turn_count
            ),

            "attempt": (
                self.state.attempt_count
            ),

            "learner_turn_intent": (
                learner_turn_intent
            ),

            "learner_turn_intent_reason": (
                learner_turn_intent_reason
            ),

            "learner_turn_intent_confidence": (
                learner_turn_intent_confidence
            ),

            "learner_turn_intent_signals": (
                learner_turn_intent_signals
            ),

            "learner_turn_intent_is_question": (
                learner_turn_intent_is_question
            ),

            "learner_turn_route": (
                learner_turn_route
            ),

            "learner_turn_route_reason": (
                learner_turn_route_reason
            ),

            "learner_turn_should_evaluate": (
                learner_turn_should_evaluate
            ),

            "learner_turn_use_current_retrieval": (
                learner_turn_use_current_retrieval
            ),

            "learner_turn_include_previous_context": (
                learner_turn_include_previous_context
            ),

            "learner_turn_replace_active_question": (
                learner_turn_replace_active_question
            ),


            "tutoring_response_mode": (
                tutoring_response_mode
            ),

            "tutoring_response_mode_reason": (
                tutoring_response_mode_reason
            ),

            "tutoring_direct_answer_required": (
                tutoring_direct_answer_required
            ),

            "tutoring_guiding_question_policy": (
                tutoring_guiding_question_policy
            ),

            "tutoring_preserve_scaffolding_strategy": (
                tutoring_preserve_scaffolding_strategy
            ),

            "tutoring_max_guiding_questions": (
                tutoring_max_guiding_questions
            ),

            "guiding_question_grounding_status": (
                self.last_guiding_question_grounding.status
                if self.last_guiding_question_grounding
                is not None
                else None
            ),

            "guiding_question_grounding_reason": (
                self.last_guiding_question_grounding.reason
                if self.last_guiding_question_grounding
                is not None
                else None
            ),

            "guiding_question_grounding_issues": (
                list(
                    self.last_guiding_question_grounding.issues
                )
                if self.last_guiding_question_grounding
                is not None
                else None
            ),

            "repair_guiding_question_grounding_status": (
                self.last_repair_guiding_question_grounding.status
                if self.last_repair_guiding_question_grounding
                is not None
                else None
            ),

            "evidence_safe_guiding_question_grounding_status": (
                self.last_evidence_safe_guiding_question_grounding.status
                if self.last_evidence_safe_guiding_question_grounding
                is not None
                else None
            ),

            "translation_guiding_question_grounding_status": (
                self.last_translation_guiding_question_grounding.status
                if self.last_translation_guiding_question_grounding
                is not None
                else None
            ),

            # =============================================
            # LEARNER PROGRESS HISTORY
            # =============================================

            "learner_progress_history_count": (
                learner_progress_history_count
            ),

            "learner_progress_history_sequence_count": (
                learner_progress_history_sequence_count
            ),

            "learner_progress_history_latest_record": (
                learner_progress_history_latest
            ),

            "learner_progress_history_latest_index": (
                learner_progress_history_latest_index
            ),

            "learner_progress_history_latest_sequence": (
                learner_progress_history_latest_sequence
            ),

            "learner_progress_history_latest_sequence_start": (
                learner_progress_history_latest_sequence_start
            ),

            "learner_progress_history_latest_turn": (
                learner_progress_history_latest_turn
            ),

            # -------------------------------------------------
            # Learner progress snapshot
            # -------------------------------------------------

            "learner_progress_snapshot": (
                learner_progress_snapshot
            ),

            "learner_progress_session_turn": (
                learner_progress_snapshot
                .session_turn
            ),

            "learner_progress_scaffolding_level": (
                learner_progress_snapshot
                .scaffolding_level
            ),

            "learner_progress_attempt_count": (
                learner_progress_snapshot
                .attempt_count
            ),

            "learner_progress_hint_count": (
                learner_progress_snapshot
                .hint_count
            ),

            "learner_progress_correct_streak": (
                learner_progress_snapshot
                .correct_streak
            ),

            "learner_progress_partial_streak": (
                learner_progress_snapshot
                .partial_streak
            ),

            "learner_progress_failure_streak": (
                learner_progress_snapshot
                .failure_streak
            ),

            "learner_progress_sequence_last_evaluation": (
                learner_progress_snapshot
                .sequence_last_evaluation
            ),

            "learner_progress_current_turn_evaluation": (
                learner_progress_snapshot
                .current_turn_evaluation
            ),

            "learner_progress_evaluation_performed": (
                learner_progress_snapshot
                .evaluation_performed
            ),

            "learner_progress_turn_intent": (
                learner_progress_snapshot
                .learner_turn_intent
            ),

            "learner_progress_turn_route": (
                learner_progress_snapshot
                .learner_turn_route
            ),

            "learner_progress_active_misconception_count": (
                learner_progress_snapshot
                .active_misconception_count
            ),

            "learner_progress_active_misconceptions": (
                learner_progress_snapshot
                .active_misconceptions
            ),


            # -------------------------
            # Retrieval
            # -------------------------

            "retrieval_query": (
                self.last_retrieval_query
            ),

            "generated_answer": (
                self.last_generated_answer
            ),

            "knowledge_sources": (
                knowledge_sources
            ),

            "knowledge_citations": (
                knowledge_citations
            ),

            "show_citations": (
                show_citations
            ),

            # -------------------------
            # Relevance
            # -------------------------

            "relevance_is_relevant": (
                relevance_is_relevant
            ),

            "relevance_confidence": (
                relevance_confidence
            ),

            "relevance_reason": (
                relevance_reason
            ),

            # -------------------------
            # Grounding
            # -------------------------

            "grounding_has_knowledge": (
                grounding_has_knowledge
            ),

            "grounding_reason": (
                grounding_reason
            ),

            # -------------------------
            # Grounding claim precheck
            # -------------------------

            "grounding_claim_precheck_status": (
                grounding_claim_precheck_status
            ),

            "grounding_claim_precheck_reason": (
                grounding_claim_precheck_reason
            ),

            "embedded_claim_term_guard_status": (
                embedded_claim_term_guard_status
            ),

            "embedded_claim_term_guard_reason": (
                embedded_claim_term_guard_reason
            ),

            "embedded_claim_term_guard_issues": (
                embedded_claim_term_guard_issues
            ),

            # -------------------------
            # Response validation
            # -------------------------

            "validation_status": (
                validation_status
            ),

            "validation_confidence": (
                validation_confidence
            ),

            "validation_reason": (
                validation_reason
            ),

            "validation_issues": (
                validation_issues
            ),

            # -------------------------
            # Pedagogical precheck
            # -------------------------

            "pedagogical_precheck_status": (
                pedagogical_precheck_status
            ),

            "pedagogical_precheck_reason": (
                pedagogical_precheck_reason
            ),

            # -------------------------
            # Pedagogical validation
            # -------------------------

            "pedagogical_status": (
                pedagogical_status
            ),

            "pedagogical_confidence": (
                pedagogical_confidence
            ),

            "pedagogical_reason": (
                pedagogical_reason
            ),

            "pedagogical_issues": (
                pedagogical_issues
            ),

            # -------------------------
            # Repair
            # -------------------------

            "response_repaired": (
                self.last_response_repaired
            ),

            "repair_candidate": (
                self.last_repair_candidate
            ),

            "repair_evidence_status": (
                self.last_repair_evidence_selection.status
                if self.last_repair_evidence_selection
                is not None
                else None
            ),

            "repair_evidence_selection_attempts": (
                self.last_repair_evidence_selection_attempts
            ),

            "repair_evidence_retry_used": (
                self.last_repair_evidence_retry_used
            ),

            "repair_evidence_first_failure_status": (
                self.last_repair_evidence_first_failure.status
                if self.last_repair_evidence_first_failure
                is not None
                else None
            ),

            "repair_evidence_first_failure_reason": (
                self.last_repair_evidence_first_failure.reason
                if self.last_repair_evidence_first_failure
                is not None
                else None
            ),

            "repair_evidence_first_failure_issues": (
                list(
                    self.last_repair_evidence_first_failure
                    .issues
                )
                if self.last_repair_evidence_first_failure
                is not None
                else None
            ),


            "evidence_safe_repair_candidate": (
                self.last_evidence_safe_repair_candidate
            ),
            "evidence_safe_repair_used": (
                self.last_evidence_safe_repair_used
            ),

            "evidence_safe_translation_status": (
                self.last_evidence_safe_translation_result.status
                if self.last_evidence_safe_translation_result
                is not None
                else None
            ),

            "evidence_safe_translation_candidate": (
                self.last_evidence_safe_translation_candidate
            ),

            "evidence_safe_translation_used": (
                self.last_evidence_safe_translation_used
            ),

            "evidence_safe_translation_semantic_status": (
                self.last_evidence_safe_translation_semantic_validation.status
                if self.last_evidence_safe_translation_semantic_validation
                is not None
                else None
            ),

            "evidence_safe_translation_semantic_reason": (
                self.last_evidence_safe_translation_semantic_validation.reason
                if self.last_evidence_safe_translation_semantic_validation
                is not None
                else None
            ),

            "evidence_safe_translation_semantic_issues": (
                list(
                    self.last_evidence_safe_translation_semantic_validation
                    .issues
                )
                if self.last_evidence_safe_translation_semantic_validation
                is not None
                else None
            ),

            "evidence_safe_translation_validation_status": (
                self.last_evidence_safe_translation_validation.status
                if self.last_evidence_safe_translation_validation
                is not None
                else None
            ),

            "evidence_safe_translation_validation_reason": (
                self.last_evidence_safe_translation_validation.reason
                if self.last_evidence_safe_translation_validation
                is not None
                else None
            ),

            "evidence_safe_translation_validation_issues": (
                list(
                    self.last_evidence_safe_translation_validation
                    .issues
                )
                if self.last_evidence_safe_translation_validation
                is not None
                else None
            ),

            "evidence_safe_translation_precision_status": (
                self.last_evidence_safe_translation_precision_guard.status
                if self.last_evidence_safe_translation_precision_guard
                is not None
                else None
            ),

            "evidence_safe_translation_precision_issues": (
                list(
                    self.last_evidence_safe_translation_precision_guard
                    .issues
                )
                if self.last_evidence_safe_translation_precision_guard
                is not None
                else None
            ),

            "evidence_safe_translation_language_status": (
                self.last_evidence_safe_translation_language_consistency.status
                if self.last_evidence_safe_translation_language_consistency
                is not None
                else None
            ),

            "evidence_safe_translation_pedagogy_status": (
                self.last_evidence_safe_translation_pedagogical_validation.status
                if self.last_evidence_safe_translation_pedagogical_validation
                is not None
                else None
            ),

            "evidence_safe_translation_rejection_gate": (
                self.last_evidence_safe_translation_rejection_gate
            ),



            "evidence_safe_translation_reason": (
                self.last_evidence_safe_translation_result.reason
                if self.last_evidence_safe_translation_result
                is not None
                else None
            ),

            "evidence_safe_translation_issues": (
                list(
                    self.last_evidence_safe_translation_result
                    .issues
                )
                if self.last_evidence_safe_translation_result
                is not None
                else None
            ),

            "repair_evidence_quotes": (
                list(
                    self.last_repair_evidence_selection
                    .evidence_quotes
                )
                if self.last_repair_evidence_selection
                is not None
                else []
            ),

            "repair_evidence_reason": (
                self.last_repair_evidence_selection.reason
                if self.last_repair_evidence_selection
                is not None
                else None
            ),

            "repair_evidence_issues": (
                list(
                    self.last_repair_evidence_selection
                    .issues
                )
                if self.last_repair_evidence_selection
                is not None
                else []
            ),

            "repair_evidence_context": (
                self.last_repair_evidence_context
            ),

            "repair_grounding_claim_precheck_status": (
                repair_grounding_claim_precheck_status
            ),

            "repair_grounding_claim_precheck_reason": (
                repair_grounding_claim_precheck_reason
            ),

            "repair_embedded_claim_term_guard_status": (
                repair_embedded_claim_term_guard_status
            ),

            "repair_embedded_claim_term_guard_reason": (
                repair_embedded_claim_term_guard_reason
            ),

            "repair_embedded_claim_term_guard_issues": (
                repair_embedded_claim_term_guard_issues
            ),

            "repair_status": (
                repair_status
            ),

            "escalation_action": (
                escalation_action
            ),

            "escalation_reason": (
                escalation_reason
            ),

            "repair_mode": (
                self.last_repair_mode
            ),

            "deterministic_replacements": (
                self.last_deterministic_replacements
            ),

            "repair_failed": (
                self.last_repair_failed
            ),

            "repair_failure_type": (
                self.last_repair_failure_type
            ),

            "repair_failure_reason": (
                self.last_repair_failure_reason
            ),

            "language_consistency_status": (
                language_consistency_status
            ),

            "language_expected": (
                language_expected
            ),

            "language_detected": (
                language_detected
            ),

            "language_reason": (
                language_reason
            ),

            "language_thai_chars": (
                language_thai_chars
            ),

            "language_latin_chars": (
                language_latin_chars
            ),

            "response_style_language": (
                response_style_language
            ),

            "response_style_technical_english": (
                response_style_technical_english
            ),

            "response_style_term_format": (
                response_style_term_format
            ),

            "response_style_tone": (
                response_style_tone
            ),

            "response_style_depth": (
                response_style_depth
            ),

            "response_style_question_style": (
                response_style_question_style
            ),

            "response_style_max_guiding_questions": (
                response_style_max_guiding_questions
            ),



            "repair_attempts": (
                self.last_repair_attempts
            ),

            "repair_limit_reached": (
                self.last_repair_limit_reached
            ),

            "max_repair_attempts": (
                self.MAX_REPAIR_ATTEMPTS
            ),

            "repair_confidence": (
                repair_confidence
            ),

            "repair_reason": (
                repair_reason
            ),

            "repair_issues": (
                repair_issues
            ),

            "repair_strategy": (
                strategy.name
                if self.last_response_repaired
                else None
            ),

            "repair_scaffolding_level": (
                self.state.scaffolding_level
                if self.last_response_repaired
                else None
            ),

            "repair_pedagogical_precheck_status": (
                repair_pedagogical_precheck_status
            ),

            "repair_pedagogical_precheck_reason": (
                repair_pedagogical_precheck_reason
            ),

            "repair_pedagogical_status": (
                repair_pedagogical_status
            ),

            "repair_pedagogical_confidence": (
                repair_pedagogical_confidence
            ),

            "repair_pedagogical_reason": (
                repair_pedagogical_reason
            ),

            "repair_pedagogical_issues": (
                repair_pedagogical_issues
            ),

            "response_quality_status": (
                response_quality_status
            ),

            "response_quality_reason": (
                response_quality_reason
            ),

            "response_quality_issues": (
                response_quality_issues
            ),

            "response_quality_character_count": (
                response_quality_character_count
            ),

            "response_quality_token_like_count": (
                response_quality_token_like_count
            ),

            "response_quality_question_count": (
                response_quality_question_count
            ),

            "response_quality_sentence_count": (
                response_quality_sentence_count
            ),

            "response_quality_too_long": (
                response_quality_too_long
            ),

            "response_quality_too_short": (
                response_quality_too_short
            ),

            "response_quality_too_many_questions": (
                response_quality_too_many_questions
            ),

            "response_quality_repetitive": (
                response_quality_repetitive
            ),

            "response_quality_analytics_total_turns": (
                response_quality_analytics.total_turns
            ),

            "response_quality_analytics_acceptable_turns": (
                response_quality_analytics.acceptable_turns
            ),

            "response_quality_analytics_advisory_turns": (
                response_quality_analytics.advisory_turns
            ),

            "response_quality_analytics_attention_turns": (
                response_quality_analytics.attention_turns
            ),

            "response_quality_analytics_issue_turns": (
                response_quality_analytics.issue_turns
            ),

            "response_quality_analytics_total_issues": (
                response_quality_analytics.total_issues
            ),

            "response_quality_analytics_attention_rate": (
                response_quality_analytics.attention_rate
            ),

            "response_quality_analytics_average_character_count": (
                response_quality_analytics
                .average_character_count
            ),

            "response_quality_analytics_average_question_count": (
                response_quality_analytics
                .average_question_count
            ),

            "response_quality_analytics_latest_quality_status": (
                response_quality_analytics
                .latest_quality_status
            ),

            "response_quality_analytics_latest_policy_status": (
                response_quality_analytics
                .latest_policy_status
            ),

            "response_quality_policy_status": (
                response_quality_policy_status
            ),

            "response_quality_policy_reason": (
                response_quality_policy_reason
            ),

            "response_quality_policy_issues": (
                response_quality_policy_issues
            ),

            "response_quality_policy_source_status": (
                response_quality_policy_source_status
            ),

            "response_quality_policy_issue_count": (
                response_quality_policy_issue_count
            ),

            "response_quality_policy_requires_attention": (
                response_quality_policy_requires_attention
            ),

            "response_quality_config_max_characters": (
                response_quality_config_max_characters
            ),

            "response_quality_config_min_characters": (
                response_quality_config_min_characters
            ),

            "response_quality_config_max_questions": (
                response_quality_config_max_questions
            ),

            "response_quality_config_detect_repetition": (
                response_quality_config_detect_repetition
            ),

            "tutoring_response_mode_instruction_applied": (
                tutoring_response_mode_instruction_applied
            ),

            # -------------------------
            # AI Usage
            # -------------------------

            "ai_usage": (
                ai_usage
            ),

            # -------------------------
            # Misconceptions
            # -------------------------

            "active_misconceptions": [
                {
                    "description": (
                        item.description
                    ),
                    "occurrence_count": (
                        item.occurrence_count
                    ),
                    "status": (
                        item.status
                    ),
                }
                for item
                in active_misconceptions
            ],
        }