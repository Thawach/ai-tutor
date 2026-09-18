from app.tutor import AITutor


def attr(obj, name):
    if obj is None:
        return None
    return getattr(obj, name, None)


def usage_tasks(summary):
    if not isinstance(summary, dict):
        return []

    records = summary.get(
        "records",
        [],
    )

    tasks = []

    for record in records:

        if isinstance(
            record,
            dict,
        ):
            task = (
                record.get("component")
                or record.get("task_name")
                or record.get("task")
            )

        else:
            task = (
                getattr(
                    record,
                    "component",
                    None,
                )
                or getattr(
                    record,
                    "task_name",
                    None,
                )
                or getattr(
                    record,
                    "task",
                    None,
                )
            )

        if task:
            tasks.append(
                task
            )

    return tasks


def final_pedagogical_status(tutor):
    if tutor.last_response_repaired:
        return attr(
            tutor.last_repair_pedagogical_validation,
            "status",
        )

    return attr(
        tutor.last_pedagogical_validation,
        "status",
    )


def run_turn(
    tutor,
    *,
    label,
    message,
    expected_intent,
    expected_mode,
    expected_override,
    expected_evaluation,
    expected_active_question,
    max_questions=None,
    expected_level=None,
    forbidden_retrieval_text=None,
):
    print()
    print("=" * 70)
    print(label)
    print("=" * 70)

    answer = tutor.respond(message)

    intent = attr(
        tutor.last_learner_turn_intent,
        "intent",
    )

    route = attr(
        tutor.last_learner_turn_routing,
        "route",
    )

    mode = attr(
        tutor.last_tutoring_response_mode,
        "mode",
    )

    direct_required = attr(
        tutor.last_tutoring_response_mode,
        "direct_answer_required",
    )

    question_policy = attr(
        tutor.last_tutoring_response_mode,
        "guiding_question_policy",
    )

    mode_max_questions = attr(
        tutor.last_tutoring_response_mode,
        "max_guiding_questions",
    )

    override = bool(
        tutor.last_tutoring_response_mode_instruction
    )

    evaluation = attr(
        tutor.last_evaluation_result,
        "classification",
    )

    relevance = attr(
        tutor.last_relevance_result,
        "is_relevant",
    )

    grounding = attr(
        tutor.last_grounding_validation,
        "status",
    )

    pedagogy = final_pedagogical_status(
        tutor
    )

    repair_precheck = attr(
        tutor.last_repair_pedagogical_precheck,
        "status",
    )

    repair_pedagogy = attr(
        tutor.last_repair_pedagogical_validation,
        "status",
    )

    question_count = (
        answer.count("?")
        + answer.count("？")
    )

    retrieval_query = (
        tutor.last_retrieval_query or ""
    )

    tasks = usage_tasks(
        tutor.last_ai_usage_summary
    )

    usage_summary = (
        tutor.last_ai_usage_summary
        if isinstance(
            tutor.last_ai_usage_summary,
            dict,
        )
        else {}
    )

    llm_calls = usage_summary.get(
        "calls",
        len(tasks),
    )

    print("USER:")
    print(message)

    print()
    print("TUTOR:")
    print(answer)

    print()
    print("--- TELEMETRY ---")
    print("Intent             =", intent)
    print("Route              =", route)
    print("Mode               =", mode)
    print("DirectRequired     =", direct_required)
    print("QuestionPolicy     =", question_policy)
    print("ModeMaxQuestions   =", mode_max_questions)
    print("PromptOverride     =", override)
    print("Evaluation         =", evaluation)
    print("ActiveQuestion     =", tutor.original_question)
    print("ScaffoldingLevel   =", tutor.state.scaffolding_level)
    print("TurnCount          =", tutor.state.turn_count)
    print("Relevance          =", relevance)
    print("Grounding          =", grounding)
    print("FinalPedagogy      =", pedagogy)
    print("ResponseRepaired   =", tutor.last_response_repaired)
    print("RepairMode         =", tutor.last_repair_mode)
    print("RepairFailed       =", tutor.last_repair_failed)
    print("RepairPrecheck     =", repair_precheck)
    print("RepairPedagogy     =", repair_pedagogy)
    print("QuestionCount      =", question_count)
    print("RetrievalQuery     =", retrieval_query)
    print("LLMCalls           =", llm_calls)
    print("LLMTasks           =", tasks)

    checks = []

    checks.append(
        (
            "intent",
            intent == expected_intent,
        )
    )

    checks.append(
        (
            "response_mode",
            mode == expected_mode,
        )
    )

    checks.append(
        (
            "prompt_override",
            override == expected_override,
        )
    )

    if expected_evaluation:
        checks.append(
            (
                "evaluation_performed",
                evaluation is not None,
            )
        )
    else:
        checks.append(
            (
                "evaluation_bypassed",
                evaluation is None,
            )
        )

    checks.append(
        (
            "active_question",
            tutor.original_question
            == expected_active_question,
        )
    )

    checks.append(
        (
            "repair_not_failed",
            tutor.last_repair_failed is False,
        )
    )

    checks.append(
        (
            "pedagogically_valid",
            pedagogy == "valid",
        )
    )

    if max_questions is not None:
        checks.append(
            (
                "question_limit",
                question_count <= max_questions,
            )
        )

    if expected_level is not None:
        checks.append(
            (
                "scaffolding_level",
                tutor.state.scaffolding_level
                == expected_level,
            )
        )

    if forbidden_retrieval_text:
        checks.append(
            (
                "retrieval_isolation",
                forbidden_retrieval_text
                not in retrieval_query,
            )
        )

    turn_passed = True

    print()
    print("--- ACCEPTANCE ---")

    for name, passed in checks:
        print(
            "[PASS]" if passed else "[FAIL]",
            name,
        )

        if not passed:
            turn_passed = False

    return turn_passed


def main():
    tutor = AITutor()

    results = []

    initial_question = (
        "ทรานซิสเตอร์ NPN มีโครงสร้างอย่างไร"
    )

    results.append(
        run_turn(
            tutor,
            label="LIVE-T1 Initial",
            message=initial_question,
            expected_intent=None,
            expected_mode="new_topic_scaffold",
            expected_override=False,
            expected_evaluation=False,
            expected_active_question=initial_question,
        )
    )

    results.append(
        run_turn(
            tutor,
            label="LIVE-T2 Learner Answer",
            message="มี 3 ชั้น คือ N P N",
            expected_intent="answer",
            expected_mode="scaffolded",
            expected_override=False,
            expected_evaluation=True,
            expected_active_question=initial_question,
        )
    )

    follow_up = (
        "เบสทำหน้าที่อะไร"
    )

    results.append(
        run_turn(
            tutor,
            label="LIVE-T3 Follow-up Question",
            message=follow_up,
            expected_intent="follow_up_question",
            expected_mode="answer_then_guide",
            expected_override=True,
            expected_evaluation=False,
            expected_active_question=follow_up,
            max_questions=1,
        )
    )

    results.append(
        run_turn(
            tutor,
            label="LIVE-T4 Clarification",
            message="คำว่าเบสหมายถึงอะไร",
            expected_intent="clarification_question",
            expected_mode="direct_clarification",
            expected_override=True,
            expected_evaluation=False,
            expected_active_question=follow_up,
            max_questions=1,
        )
    )

    topic_change = (
        "ขอถามอีกเรื่อง "
        "ทรานซิสเตอร์ NPN ทำงานอย่างไร"
    )

    results.append(
        run_turn(
            tutor,
            label="LIVE-T5 Topic Change",
            message=topic_change,
            expected_intent="topic_change",
            expected_mode="new_topic_scaffold",
            expected_override=False,
            expected_evaluation=False,
            expected_active_question=topic_change,
            expected_level=1,
            forbidden_retrieval_text="เบส",
        )
    )

    passed = sum(
        1
        for result in results
        if result
    )

    failed = len(results) - passed

    print()
    print("=" * 70)
    print("STEP 16.19 LIVE ACCEPTANCE SUMMARY")
    print("=" * 70)
    print("Passed turns:", passed)
    print("Failed turns:", failed)

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
