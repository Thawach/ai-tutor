from rich.console import Console
from rich.panel import Panel

from app.config import (
    APP_NAME,
    AI_PROVIDER,
    ACTIVE_CHAT_MODEL,
)

from app.tutor import AITutor


console = Console()
tutor = AITutor()


def main():

    console.print(
        Panel.fit(
            f"[bold]{APP_NAME} v.4.1[/bold]\n"
            f"Provider: {AI_PROVIDER}\n"
            f"Model: {ACTIVE_CHAT_MODEL}\n\n"
            "Commands:\n"
            "exit  = ออกจากโปรแกรม\n"
            "reset = ล้างประวัติการสนทนา\n"
            "prompt = ดู System Prompt ล่าสุด\n",
            title="AI Tutor",
        )
    )

    while True:

        question = console.input(
            "\n[bold cyan]คุณ:[/bold cyan] "
        ).strip()

        if question.lower() in ["exit", "quit"]:
            console.print("\nปิด AI Tutor แล้ว")
            break

        if question.lower() == "reset":
            tutor.reset()

            console.print(
                "[yellow]"
                "ล้างประวัติการสนทนาแล้ว"
                "[/yellow]"
            )

            continue

        if question.lower() == "prompt":

            prompt = tutor.get_last_system_prompt()

            if prompt:
                console.print(
                    Panel(
                        prompt,
                        title="Current System Prompt",
                    )
                )
            else:
                console.print(
                    "[yellow]"
                    "ยังไม่มี System Prompt"
                    "[/yellow]"
                )

            continue

        if not question:
            continue

        try:

            console.print(
                "\n[bold green]"
                "AI Tutor:"
                "[/bold green]"
            )

            answer = tutor.respond(
                question
            )

            console.print(answer)

            # -------------------------
            # Debug information
            # -------------------------

            debug = tutor.get_debug_info()

            # -------------------------
            # Learner Evaluator
            # -------------------------

            if debug["evaluation"] is not None:

                console.print(
                    "\n[dim]"
                    f"Evaluator | "
                    f"Confidence={debug['evaluation_confidence']} | "
                    f"Reason={debug['evaluation_reason']} | "
                    f"Misconception={debug['evaluation_misconception']}"
                    "[/dim]"
                )

            # -------------------------
            # Active Misconceptions
            # -------------------------

            if debug["active_misconceptions"]:

                console.print(
                    "\n[bold yellow]"
                    "Active Misconceptions"
                    "[/bold yellow]"
                )

                for item in debug["active_misconceptions"]:

                    console.print(
                        "[dim]"
                        f"- {item['description']} "
                        f"(count={item['occurrence_count']})"
                        "[/dim]"
                    )

            if (
                debug.get(
                    "learner_turn_intent"
                )
                is not None
            ):

                console.print(
                    "[dim]"
                    f"Learner Turn Intent | "
                    f"Intent="
                    f"{debug.get('learner_turn_intent')} | "
                    f"Confidence="
                    f"{debug.get('learner_turn_intent_confidence')} | "
                    f"Question="
                    f"{debug.get('learner_turn_intent_is_question')}"
                    "[/dim]"
                )

                console.print(
                    "[dim]"
                    f"Learner Turn Intent Signals | "
                    f"{debug.get('learner_turn_intent_signals')}"
                    "[/dim]"
                )

                console.print(
                    "[dim]"
                    f"Learner Turn Intent Reason | "
                    f"{debug.get('learner_turn_intent_reason')}"
                    "[/dim]"
                )

            if debug.get(
                "learner_turn_route"
            ) is not None:

                console.print(
                    "[dim]"
                    f"Learner Turn Routing | "
                    f"Route="
                    f"{debug.get('learner_turn_route')} | "
                    f"Evaluate="
                    f"{debug.get('learner_turn_should_evaluate')} | "
                    f"CurrentRAG="
                    f"{debug.get('learner_turn_use_current_retrieval')} | "
                    f"PreviousContext="
                    f"{debug.get('learner_turn_include_previous_context')} | "
                    f"ReplaceActive="
                    f"{debug.get('learner_turn_replace_active_question')}"
                    "[/dim]"
                )

                console.print(
                    "[dim]"
                    f"Learner Turn Routing Reason | "
                    f"{debug.get('learner_turn_route_reason')}"
                    "[/dim]"
                )

            if (
                debug[
                    "tutoring_response_mode"
                ]
                is not None
            ):

                console.print(
                    "[dim]"
                    "Tutoring Response Mode | "
                    f"Mode="
                    f"{debug['tutoring_response_mode']} | "
                    f"DirectAnswer="
                    f"{debug['tutoring_direct_answer_required']} | "
                    f"GuidingQuestion="
                    f"{debug['tutoring_guiding_question_policy']} | "
                    f"PreserveScaffolding="
                    f"{debug['tutoring_preserve_scaffolding_strategy']} | "
                    f"PromptOverride="
                    f"{debug['tutoring_response_mode_instruction_applied']} | "
                    f"MaxQuestions="
                    f"{debug['tutoring_max_guiding_questions']}"
                    "[/dim]"
                )

                console.print(
                    "[dim]"
                    "Tutoring Response Mode Reason | "
                    f"{debug['tutoring_response_mode_reason']}"
                    "[/dim]"
                )               


            # =========================================================
            # LEARNER PROGRESS
            # Always visible, including initial turns.
            # =========================================================

            console.print(
                "[dim]"
                f"Learner Progress | "
                f"Turn="
                f"{debug.get('learner_progress_session_turn')} | "
                f"Level="
                f"{debug.get('learner_progress_scaffolding_level')} | "
                f"Attempts="
                f"{debug.get('learner_progress_attempt_count')} | "
                f"Hints="
                f"{debug.get('learner_progress_hint_count')} | "
                f"Evaluated="
                f"{debug.get('learner_progress_evaluation_performed')}"
                "[/dim]"
            )

            console.print(
                "[dim]"
                f"Learner Progress Evaluation | "
                f"SequenceLast="
                f"{debug.get('learner_progress_sequence_last_evaluation')} | "
                f"CurrentTurn="
                f"{debug.get('learner_progress_current_turn_evaluation')}"
                "[/dim]"
            )

            console.print(
                "[dim]"
                f"Learner Progress Streaks | "
                f"Correct="
                f"{debug.get('learner_progress_correct_streak')} | "
                f"Partial="
                f"{debug.get('learner_progress_partial_streak')} | "
                f"Failure="
                f"{debug.get('learner_progress_failure_streak')}"
                "[/dim]"
            )

            console.print(
                "[dim]"
                f"Learner Progress Context | "
                f"Intent="
                f"{debug.get('learner_progress_turn_intent')} | "
                f"Route="
                f"{debug.get('learner_progress_turn_route')} | "
                f"ActiveMisconceptions="
                f"{debug.get('learner_progress_active_misconception_count')}"
                "[/dim]"
            )


            # =========================================================
            # LEARNER PROGRESS HISTORY
            # =========================================================

            console.print(
                "[dim]"
                f"Learner Progress History | "
                f"Records="
                f"{debug.get('learner_progress_history_count')} | "
                f"Sequences="
                f"{debug.get('learner_progress_history_sequence_count')}"
                "[/dim]"
            )

            console.print(
                "[dim]"
                f"Learner Progress History Latest | "
                f"Index="
                f"{debug.get('learner_progress_history_latest_index')} | "
                f"Sequence="
                f"{debug.get('learner_progress_history_latest_sequence')} | "
                f"SequenceStart="
                f"{debug.get('learner_progress_history_latest_sequence_start')} | "
                f"Turn="
                f"{debug.get('learner_progress_history_latest_turn')}"
                "[/dim]"
            )

            # -------------------------
            # RAG Sources
            # Developer Debug
            # -------------------------

            if debug["knowledge_sources"]:

                console.print(
                    "\n[bold blue]"
                    "RAG Sources"
                    "[/bold blue]"
                )

                for item in debug["knowledge_sources"]:

                    console.print(
                        "[dim]"
                        f"- {item['source']} "
                        f"| Page={item['page']} "
                        f"| Distance={item['distance']}"
                        "[/dim]"
                    )                                                                                           

            # -------------------------
            # Retrieval Query
            # -------------------------

            if debug["retrieval_query"]:

                console.print(
                    "\n[bold magenta]"
                    "Retrieval Query"
                    "[/bold magenta]"
                )

                console.print(
                    "[dim]"
                    f"{debug['retrieval_query']}"
                    "[/dim]"
                )

            # -------------------------
            # Relevance Gate
            # -------------------------

            console.print(
                "\n[dim]"
                f"Relevance | "
                f"Relevant={debug['relevance_is_relevant']} | "
                f"Confidence={debug['relevance_confidence']} | "
                f"Reason={debug['relevance_reason']}"
                "[/dim]"
            )

            # -------------------------
            # Grounding Guard
            # -------------------------

            console.print(
                "\n[dim]"
                f"Grounding | "
                f"HasKnowledge={debug['grounding_has_knowledge']} | "
                f"Reason={debug['grounding_reason']}"
                "[/dim]"
            )

            # -------------------------
            # Original Tutor Generation
            # -------------------------

            if (
                debug.get("generated_answer")
                is not None
            ):

                console.print(
                    "\n[bold yellow]"
                    "Original Tutor Generation"
                    "[/bold yellow]"
                )

                console.print(
                    "[dim]"
                    f"{debug['generated_answer']}"
                    "[/dim]"
                )


            # -------------------------
            # Pedagogical Precheck
            # -------------------------

            if (
                debug["pedagogical_precheck_status"]
                is not None
            ):

                console.print(
                    "\n[dim]"
                    f"Pedagogical Precheck | "
                    f"Status="
                    f"{debug['pedagogical_precheck_status']} | "
                    f"Reason="
                    f"{debug['pedagogical_precheck_reason']}"
                    "[/dim]"
                )


            # -------------------------
            # Grounding Claim Precheck
            # -------------------------

            if (
                debug["grounding_claim_precheck_status"]
                is not None
            ):

                console.print(
                    "\n[dim]"
                    f"Grounding Claim Precheck | "
                    f"Status="
                    f"{debug['grounding_claim_precheck_status']} | "
                    f"Reason="
                    f"{debug['grounding_claim_precheck_reason']}"
                    "[/dim]"
                )

            # -------------------------
            # Embedded Claim /
            # Technical Term Guard
            # -------------------------

            if (
                debug["embedded_claim_term_guard_status"]
                is not None
            ):

                console.print(
                    "\n[dim]"
                    f"Embedded Claim/Term Guard | "
                    f"Status="
                    f"{debug['embedded_claim_term_guard_status']} | "
                    f"Reason="
                    f"{debug['embedded_claim_term_guard_reason']}"
                    "[/dim]"
                )

                if debug["embedded_claim_term_guard_issues"]:

                    console.print(
                        "[dim]"
                        "Embedded Claim/Term Issues:"
                        "[/dim]"
                    )

                    for issue in (
                        debug[
                            "embedded_claim_term_guard_issues"
                        ]
                    ):

                        console.print(
                            "[dim]"
                            f"- {issue}"
                            "[/dim]"
                        )

            # -------------------------
            # Response Grounding
            # Validation
            # -------------------------

            if debug["validation_status"] is not None:

                console.print(
                    "\n[dim]"
                    f"Response Validation | "
                    f"Status={debug['validation_status']} | "
                    f"Confidence={debug['validation_confidence']} | "
                    f"Reason={debug['validation_reason']}"
                    "[/dim]"
                )

                if debug["validation_issues"]:

                    console.print(
                        "[dim]"
                        "Validation Issues:"
                        "[/dim]"
                    )

                    for issue in debug["validation_issues"]:

                        console.print(
                            "[dim]"
                            f"- {issue}"
                            "[/dim]"
                        )


            # -------------------------
            # Pedagogical Validation
            # -------------------------

            if debug["pedagogical_status"] is not None:

                console.print(
                    "\n[dim]"
                    f"Pedagogical Validation | "
                    f"Status={debug['pedagogical_status']} | "
                    f"Confidence={debug['pedagogical_confidence']} | "
                    f"Reason={debug['pedagogical_reason']}"
                    "[/dim]"
                )

                if debug["pedagogical_issues"]:

                    console.print(
                        "[dim]"
                        "Pedagogical Issues:"
                        "[/dim]"
                    )

                    for issue in debug["pedagogical_issues"]:

                        console.print(
                            "[dim]"
                            f"- {issue}"
                            "[/dim]"
                        )       

            # -------------------------
            # Validation Escalation
            # -------------------------

            if (
                debug.get("escalation_action")
                is not None
            ):

                console.print(
                    "\n[dim]"
                    f"Validation Escalation | "
                    f"Action="
                    f"{debug['escalation_action']} | "
                    f"Reason="
                    f"{debug['escalation_reason']}"
                    "[/dim]"
                )


            # -------------------------
            # Repair Mode
            # -------------------------

            if (
                debug.get("repair_mode")
                is not None
            ):

                console.print(
                    "[dim]"
                    f"Repair Mode | "
                    f"{debug['repair_mode']}"
                    "[/dim]"
                )


            # -------------------------
            # Deterministic Replacements
            # -------------------------

            if debug.get(
                "deterministic_replacements"
            ):

                console.print(
                    "[dim]"
                    "Deterministic Replacements:"
                    "[/dim]"
                )

                for (
                    old_term,
                    new_term,
                ) in debug[
                    "deterministic_replacements"
                ]:

                    console.print(
                        "[dim]"
                        f"- {old_term} -> {new_term}"
                        "[/dim]"
                    )


            # -------------------------
            # Repair Failure
            # -------------------------

            if debug.get(
                "repair_failed"
            ):

                console.print(
                    "\n[dim]"
                    f"Repair Failure | "
                    f"Type="
                    f"{debug.get('repair_failure_type')} | "
                    f"Reason="
                    f"{debug.get('repair_failure_reason')}"
                    "[/dim]"
                )

            # -------------------------
            # Response Repair
            # -------------------------

            if debug["repair_status"] is not None:

                console.print(
                    "\n[bold yellow]"
                    "Response Repair"
                    "[/bold yellow]"
                )

                # -------------------------
                # Repair summary
                # -------------------------

                console.print(
                    "[dim]"
                    f"Repaired={debug['response_repaired']} | "
                    f"Level={debug['repair_scaffolding_level']} | "
                    f"Strategy={debug['repair_strategy']} | "
                    f"Status={debug['repair_status']} | "
                    f"Confidence={debug['repair_confidence']} | "
                    f"Reason={debug['repair_reason']}"
                    "[/dim]"
                )

                # -------------------------
                # Repair Grounding
                # Claim Precheck
                # -------------------------

                if (
                    debug[
                        "repair_grounding_claim_precheck_status"
                    ]
                    is not None
                ):

                    console.print(
                        "[dim]"
                        f"Repair Grounding Claim Precheck | "
                        f"Status="
                        f"{debug['repair_grounding_claim_precheck_status']} | "
                        f"Reason="
                        f"{debug['repair_grounding_claim_precheck_reason']}"
                        "[/dim]"
                    )

                # -------------------------
                # Repair Embedded Claim /
                # Technical Term Guard
                # -------------------------

                if (
                    debug[
                        "repair_embedded_claim_term_guard_status"
                    ]
                    is not None
                ):

                    console.print(
                        "[dim]"
                        f"Repair Embedded Claim/Term Guard | "
                        f"Status="
                        f"{debug['repair_embedded_claim_term_guard_status']} | "
                        f"Reason="
                        f"{debug['repair_embedded_claim_term_guard_reason']}"
                        "[/dim]"
                    )

                    if (
                        debug[
                            "repair_embedded_claim_term_guard_issues"
                        ]
                    ):

                        console.print(
                            "[dim]"
                            "Repair Embedded Claim/Term Issues:"
                            "[/dim]"
                        )

                        for issue in (
                            debug[
                                "repair_embedded_claim_term_guard_issues"
                            ]
                        ):

                            console.print(
                                "[dim]"
                                f"- {issue}"
                                "[/dim]"
                            )

                # -------------------------
                # Repair Pedagogy
                # -------------------------

                if (
                    debug["repair_pedagogical_status"]
                    is not None
                ):

                    console.print(
                        "[dim]"
                        f"Repair Pedagogy | "
                        f"Status="
                        f"{debug['repair_pedagogical_status']} | "
                        f"Confidence="
                        f"{debug['repair_pedagogical_confidence']} | "
                        f"Reason="
                        f"{debug['repair_pedagogical_reason']}"
                        "[/dim]"
                    )

                    if (
                        debug[
                            "repair_pedagogical_issues"
                        ]
                    ):

                        console.print(
                            "[dim]"
                            "Repair Pedagogical Issues:"
                            "[/dim]"
                        )

                        for issue in (
                            debug[
                                "repair_pedagogical_issues"
                            ]
                        ):

                            console.print(
                                "[dim]"
                                f"- {issue}"
                                "[/dim]"
                            )

                # -------------------------
                # Repair Validation Issues
                # -------------------------

                if debug["repair_issues"]:

                    console.print(
                        "[dim]"
                        "Repair Validation Issues:"
                        "[/dim]"
                    )

                    for issue in (
                        debug["repair_issues"]
                    ):

                        console.print(
                            "[dim]"
                            f"- {issue}"
                            "[/dim]"
                        )

            # -------------------------
            # Language Consistency
            # -------------------------

            if (
                debug.get(
                    "language_consistency_status"
                )
                is not None
            ):

                console.print(
                    "\n[dim]"
                    f"Language Consistency | "
                    f"Status="
                    f"{debug.get('language_consistency_status')} | "
                    f"Expected="
                    f"{debug.get('language_expected')} | "
                    f"Detected="
                    f"{debug.get('language_detected')} | "
                    f"Reason="
                    f"{debug.get('language_reason')}"
                    "[/dim]"
                )

                console.print(
                    "[dim]"
                    f"Language Characters | "
                    f"Thai="
                    f"{debug.get('language_thai_chars')} | "
                    f"Latin="
                    f"{debug.get('language_latin_chars')}"
                    "[/dim]"
                )


            # -------------------------
            # Response Style
            # -------------------------

            if (
                debug.get(
                    "response_style_language"
                )
                is not None
            ):

                console.print(
                    "\n[dim]"
                    f"Response Style | "
                    f"Language="
                    f"{debug.get('response_style_language')} | "
                    f"Tone="
                    f"{debug.get('response_style_tone')} | "
                    f"Depth="
                    f"{debug.get('response_style_depth')} | "
                    f"QuestionStyle="
                    f"{debug.get('response_style_question_style')} | "
                    f"MaxGuidingQuestions="
                    f"{debug.get('response_style_max_guiding_questions')}"
                    "[/dim]"
                )

                console.print(
                    "[dim]"
                    f"Response Style Terms | "
                    f"TechnicalEnglish="
                    f"{debug.get('response_style_technical_english')} | "
                    f"TermFormat="
                    f"{debug.get('response_style_term_format')}"
                    "[/dim]"
                )

            # -------------------------
            # Response Quality
            # -------------------------

            if (
                debug.get(
                    "response_quality_status"
                )
                is not None
            ):

                console.print(
                    "\n[dim]"
                    f"Response Quality | "
                    f"Status="
                    f"{debug.get('response_quality_status')} | "
                    f"Reason="
                    f"{debug.get('response_quality_reason')}"
                    "[/dim]"
                )

                console.print(
                    "[dim]"
                    f"Response Quality Config | "
                    f"MaxChars="
                    f"{debug.get('response_quality_config_max_characters')} | "
                    f"MinChars="
                    f"{debug.get('response_quality_config_min_characters')} | "
                    f"MaxQuestions="
                    f"{debug.get('response_quality_config_max_questions')} | "
                    f"DetectRepetition="
                    f"{debug.get('response_quality_config_detect_repetition')}"
                    "[/dim]"
                )


                console.print(
                    "[dim]"
                    f"Response Quality Metrics | "
                    f"Chars="
                    f"{debug.get('response_quality_character_count')} | "
                    f"TokensLike="
                    f"{debug.get('response_quality_token_like_count')} | "
                    f"Questions="
                    f"{debug.get('response_quality_question_count')} | "
                    f"Sentences="
                    f"{debug.get('response_quality_sentence_count')}"
                    "[/dim]"
                )

                console.print(
                    "[dim]"
                    f"Response Quality Flags | "
                    f"TooLong="
                    f"{debug.get('response_quality_too_long')} | "
                    f"TooShort="
                    f"{debug.get('response_quality_too_short')} | "
                    f"TooManyQuestions="
                    f"{debug.get('response_quality_too_many_questions')} | "
                    f"Repetitive="
                    f"{debug.get('response_quality_repetitive')}"
                    "[/dim]"
                )

            if (
                debug.get(
                    "response_quality_policy_status"
                )
                is not None
            ):

                console.print(
                    "[dim]"
                    f"Response Quality Policy | "
                    f"Status="
                    f"{debug.get('response_quality_policy_status')} | "
                    f"Source="
                    f"{debug.get('response_quality_policy_source_status')} | "
                    f"Issues="
                    f"{debug.get('response_quality_policy_issue_count')} | "
                    f"RequiresAttention="
                    f"{debug.get('response_quality_policy_requires_attention')}"
                    "[/dim]"
                )

                console.print(
                    "[dim]"
                    f"Response Quality Policy Reason | "
                    f"{debug.get('response_quality_policy_reason')}"
                    "[/dim]"
                )

                if debug.get(
                    "response_quality_issues"
                ):

                    console.print(
                        "[dim]"
                        "Response Quality Issues:"
                        "[/dim]"
                    )

                    for issue in debug[
                        "response_quality_issues"
                    ]:

                        console.print(
                            "[dim]"
                            f"- {issue}"
                            "[/dim]"
                        )

            if (
                debug.get(
                    "response_quality_analytics_total_turns",
                    0,
                )
                > 0
            ):

                console.print(
                    "[dim]"
                    f"Response Quality Analytics | "
                    f"Turns="
                    f"{debug.get('response_quality_analytics_total_turns')} | "
                    f"Acceptable="
                    f"{debug.get('response_quality_analytics_acceptable_turns')} | "
                    f"Advisory="
                    f"{debug.get('response_quality_analytics_advisory_turns')} | "
                    f"Attention="
                    f"{debug.get('response_quality_analytics_attention_turns')}"
                    "[/dim]"
                )

                console.print(
                    "[dim]"
                    f"Response Quality Trend | "
                    f"AttentionRate="
                    f"{debug.get('response_quality_analytics_attention_rate') * 100:.1f}% | "
                    f"IssueTurns="
                    f"{debug.get('response_quality_analytics_issue_turns')} | "
                    f"TotalIssues="
                    f"{debug.get('response_quality_analytics_total_issues')} | "
                    f"Latest="
                    f"{debug.get('response_quality_analytics_latest_policy_status')}"
                    "[/dim]"
                )

                console.print(
                    "[dim]"
                    f"Response Quality Averages | "
                    f"Chars="
                    f"{debug.get('response_quality_analytics_average_character_count'):.1f} | "
                    f"Questions="
                    f"{debug.get('response_quality_analytics_average_question_count'):.2f}"
                    "[/dim]"
                )


            # -------------------------
            # AI Usage
            # -------------------------

            ai_usage = debug["ai_usage"]

            if ai_usage["calls"] > 0:

                console.print(
                    "\n[bold bright_blue]"
                    "AI Usage"
                    "[/bold bright_blue]"
                )

                for record in ai_usage["records"]:

                    console.print(
                        "[dim]"
                        f"- {record['component']} | "
                        f"{record['provider']} | "
                        f"{record['latency_ms']:.0f} ms | "
                        f"Tokens={record['total_tokens']} | "
                        f"Cost=${record['estimated_cost_usd']:.6f}"
                        "[/dim]"
                    )

                console.print(
                    "[dim]"
                    f"TOTAL | "
                    f"Calls={ai_usage['calls']} | "
                    f"LLM Latency="
                    f"{ai_usage['total_latency_ms']:.0f} ms | "
                    f"Input={ai_usage['prompt_tokens']} | "
                    f"Output={ai_usage['completion_tokens']} | "
                    f"Tokens={ai_usage['total_tokens']} | "
                    f"Cost=${ai_usage['estimated_cost_usd']:.6f}"
                    "[/dim]"
                )                   

            # -------------------------
            # Learner-facing Citations
            # -------------------------

            if (
                debug["show_citations"]
                and debug["knowledge_citations"]
            ):

                console.print(
                    "\n[bold cyan]"
                    "Sources"
                    "[/bold cyan]"
                )

                for item in debug["knowledge_citations"]:

                    console.print(
                        "[dim]"
                        f"- {item['source']} "
                        f"| Page {item['page']}"
                        "[/dim]"
                    )

            # -------------------------
            # Scaffolding Debug
            # -------------------------

            console.print(
                "\n[dim]"
                f"Debug | "
                f"Level={debug['level']} | "
                f"Strategy={debug['strategy']} | "
                f"Intervention={debug['intervention']} | "
                f"Eval={debug['evaluation']} | "
                f"Action={debug['decision_action']} | "
                f"Reason={debug['decision_reason']} | "
                f"C={debug['correct_streak']} | "
                f"P={debug['partial_streak']} | "
                f"F={debug['failure_streak']} | "
                f"Turn={debug['turn']} | "
                f"Attempt={debug['attempt']}"
                "[/dim]"
            )

        except Exception as error:

            console.print(
                f"\n[bold red]"
                f"เกิดข้อผิดพลาด:"
                f"[/bold red] "
                f"{error}"
            )


if __name__ == "__main__":
    main()