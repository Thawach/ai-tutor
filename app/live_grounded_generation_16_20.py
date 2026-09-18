from app.tutor import AITutor


QUESTION = "เบสทำหน้าที่อะไร"


def usage_tasks(tutor):
    summary = (
        tutor.last_ai_usage_summary
        if isinstance(
            tutor.last_ai_usage_summary,
            dict,
        )
        else {}
    )

    return [
        record.get("component")
        for record
        in summary.get("records", [])
    ]


def main():
    print()
    print(
        "Step 16.20.2B Grounded Generation Probe"
    )
    print()

    # Diagnostic run:
    # run only once to inspect the exact repair candidate
    # and retrieved course knowledge.
    for run_number in range(1, 2):

        tutor = AITutor()

        # -------------------------------------------------
        # Establish normal tutoring context first.
        # -------------------------------------------------

        tutor.respond(
            "ทรานซิสเตอร์ NPN มีโครงสร้างอย่างไร"
        )

        answer = tutor.respond(
            QUESTION
        )

        raw_answer = (
            tutor.last_generated_answer
        )

        # -------------------------------------------------
        # Initial grounding result
        # -------------------------------------------------

        grounding = getattr(
            tutor.last_grounding_validation,
            "status",
            None,
        )

        grounding_reason = getattr(
            tutor.last_grounding_validation,
            "reason",
            None,
        )

        grounding_issues = getattr(
            tutor.last_grounding_validation,
            "issues",
            None,
        )

        # -------------------------------------------------
        # Initial deterministic precision result
        # -------------------------------------------------

        initial_precision_status = getattr(
            tutor.last_grounding_precision_guard,
            "status",
            None,
        )

        initial_precision_issues = getattr(
            tutor.last_grounding_precision_guard,
            "issues",
            None,
        )

        # -------------------------------------------------
        # Repair grounding result
        # -------------------------------------------------

        repair_grounding = getattr(
            tutor.last_repair_validation,
            "status",
            None,
        )

        repair_grounding_reason = getattr(
            tutor.last_repair_validation,
            "reason",
            None,
        )

        repair_grounding_issues = getattr(
            tutor.last_repair_validation,
            "issues",
            None,
        )

        # -------------------------------------------------
        # Repair deterministic precision result
        # -------------------------------------------------

        repair_precision_status = getattr(
            tutor.last_repair_grounding_precision_guard,
            "status",
            None,
        )

        repair_precision_issues = getattr(
            tutor.last_repair_grounding_precision_guard,
            "issues",
            None,
        )

        # -------------------------------------------------
        # Final effective grounding result
        # -------------------------------------------------

        final_grounding = (
            repair_grounding
            if (
                tutor.last_response_repaired
                and repair_grounding is not None
            )
            else grounding
        )

        # -------------------------------------------------
        # Knowledge context used by generation / validation
        # -------------------------------------------------

        knowledge_context = (
            tutor.last_knowledge_result.context
            if tutor.last_knowledge_result is not None
            else None
        )

        # =================================================
        # OUTPUT
        # =================================================

        print("=" * 70)
        print(
            f"RUN {run_number}"
        )
        print("=" * 70)

        print("RAW GENERATED:")
        print(raw_answer)

        print()
        print("-" * 70)
        print("REPAIR CANDIDATE")
        print("-" * 70)
        print(
            tutor.last_repair_candidate
            if tutor.last_repair_candidate is not None
            else "<none>"
        )

        print()
        print("-" * 70)
        print("EVIDENCE-SAFE REPAIR")
        print("-" * 70)

        print(
            "EvidenceSafeRepairCandidate ="
        )

        print(
            tutor.last_evidence_safe_repair_candidate
            if tutor.last_evidence_safe_repair_candidate
            is not None
            else "<none>"
        )

        print(
            "EvidenceSafeRepairUsed =",
            tutor.last_evidence_safe_repair_used,
        )


        print()
        print("-" * 70)
        print("EVIDENCE-SAFE TRANSLATION")
        print("-" * 70)

        translation_result = (
            tutor.last_evidence_safe_translation_result
        )

        print(
            "EvidenceSafeTranslationStatus =",
            (
                translation_result.status
                if translation_result is not None
                else None
            ),
        )

        print(
            "EvidenceSafeTranslationCandidate ="
        )

        print(
            tutor.last_evidence_safe_translation_candidate
            if tutor.last_evidence_safe_translation_candidate
            is not None
            else "<none>"
        )

        print(
            "EvidenceSafeTranslationUsed =",
            tutor.last_evidence_safe_translation_used,
        )

        print(
            "EvidenceSafeTranslationReason =",
            (
                translation_result.reason
                if translation_result is not None
                else None
            ),
        )

        print(
            "EvidenceSafeTranslationIssues =",
            (
                translation_result.issues
                if translation_result is not None
                else None
            ),
        )

        print(
            "RepairEvidenceSelectionAttempts =",
            tutor.last_repair_evidence_selection_attempts,
        )

        print(
            "RepairEvidenceRetryUsed =",
            tutor.last_repair_evidence_retry_used,
        )

        print(
            "RepairEvidenceFirstFailure =",
            (
                tutor.last_repair_evidence_first_failure.reason
                if tutor.last_repair_evidence_first_failure
                is not None
                else None
            ),
        )
        print(
            "RepairEvidenceFirstFailureIssues =",
            (
                tutor.last_repair_evidence_first_failure.issues
                if tutor.last_repair_evidence_first_failure
                is not None
                else None
            ),
        )

        # ---------------------------------------------
        # Translation gate diagnostics
        # ---------------------------------------------
        
        translation_semantic = (
            tutor.last_evidence_safe_translation_semantic_validation
        )

        translation_validation = (
            tutor.last_evidence_safe_translation_validation
        )

        translation_precision = (
            tutor.last_evidence_safe_translation_precision_guard
        )

        translation_language = (
            tutor.last_evidence_safe_translation_language_consistency
        )

        translation_pedagogy = (
            tutor.last_evidence_safe_translation_pedagogical_validation
        )

        print(
            "TranslationSemanticGrounding =",
            (
                translation_semantic.status
                if translation_semantic is not None
                else None
            ),
        )

        print(
            "TranslationSemanticIssues =",
            (
                translation_semantic.issues
                if translation_semantic is not None
                else None
            ),
        )

        print(
            "TranslationFinalGrounding =",
            (
                translation_validation.status
                if translation_validation is not None
                else None
            ),
        )

        print(
            "TranslationFinalIssues =",
            (
                translation_validation.issues
                if translation_validation is not None
                else None
            ),
        )

        print(
            "TranslationPrecision =",
            (
                translation_precision.status
                if translation_precision is not None
                else None
            ),
        )

        print(
            "TranslationPrecisionIssues =",
            (
                translation_precision.issues
                if translation_precision is not None
                else None
            ),
        )

        print(
            "TranslationLanguage =",
            (
                translation_language.status
                if translation_language is not None
                else None
            ),
        )

        print(
            "TranslationPedagogy =",
            (
                translation_pedagogy.status
                if translation_pedagogy is not None
                else None
            ),
        )

        print(
            "TranslationRejectionGate =",
            tutor.last_evidence_safe_translation_rejection_gate,
        )

        print()
        print("-" * 70)
        print("REPAIR EVIDENCE")
        print("-" * 70)

        repair_evidence_selection = (
            tutor.last_repair_evidence_selection
        )

        print(
            "RepairEvidenceStatus =",
            (
                repair_evidence_selection.status
                if repair_evidence_selection is not None
                else None
            ),
        )

        print(
            "RepairEvidenceReason =",
            (
                repair_evidence_selection.reason
                if repair_evidence_selection is not None
                else None
            ),
        )

        print(
            "RepairEvidenceIssues =",
            (
                repair_evidence_selection.issues
                if repair_evidence_selection is not None
                else ()
            ),
        )

        print(
            "RepairEvidenceQuotes =",
            (
                repair_evidence_selection.evidence_quotes
                if repair_evidence_selection is not None
                else ()
            ),
        )

        print()
        print("RepairEvidenceContext:")
        print(
            tutor.last_repair_evidence_context
            if tutor.last_repair_evidence_context is not None
            else "<none>"
        )

        print()
        print("-" * 70)
        print("KNOWLEDGE CONTEXT")
        print("-" * 70)
        print(
            knowledge_context
            if knowledge_context is not None
            else "<none>"
        )

        print()
        print("-" * 70)
        print("FINAL ANSWER")
        print("-" * 70)
        print(answer)

        print()

        print(
            "InitialGrounding   =",
            grounding,
        )

        print(
            "InitialReason      =",
            grounding_reason,
        )

        print(
            "InitialIssues      =",
            grounding_issues,
        )

        print(
            "InitialPrecision   =",
            initial_precision_status,
        )

        print(
            "InitialPrecIssues  =",
            initial_precision_issues,
        )

        print(
            "RepairPrecision    =",
            repair_precision_status,
        )

        print(
            "RepairPrecIssues   =",
            repair_precision_issues,
        )

        print(
            "RepairGrounding    =",
            repair_grounding,
        )

        print(
            "RepairReason       =",
            repair_grounding_reason,
        )

        print(
            "RepairIssues       =",
            repair_grounding_issues,
        )

        print(
            "FinalGrounding     =",
            final_grounding,
        )

        print(
            "ResponseRepaired   =",
            tutor.last_response_repaired,
        )

        print(
            "RepairMode         =",
            tutor.last_repair_mode,
        )

        print(
            "RepairFailed       =",
            tutor.last_repair_failed,
        )

        print(
            "Tasks              =",
            usage_tasks(tutor),
        )

        print()


if __name__ == "__main__":
    main()