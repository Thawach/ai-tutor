from app.services.response_quality_analytics_models import (
    ResponseQualityAnalyticsResult,
)

from app.services.response_quality_history import (
    ResponseQualityHistory,
)


class ResponseQualityAnalytics:
    """
    Deterministic analytics over ResponseQualityHistory.

    This service does not:
    - call an LLM
    - mutate quality history
    - modify Tutor responses
    - trigger repair
    - trigger escalation
    """

    def summarize(
        self,
        history: ResponseQualityHistory,
    ) -> ResponseQualityAnalyticsResult:

        if not isinstance(
            history,
            ResponseQualityHistory,
        ):
            raise TypeError(
                "history must be a "
                "ResponseQualityHistory."
            )

        records = history.records

        total_turns = len(
            records
        )

        if total_turns == 0:

            return (
                ResponseQualityAnalyticsResult()
            )

        acceptable_turns = sum(
            record.policy_status
            == "acceptable"
            for record in records
        )

        advisory_turns = sum(
            record.policy_status
            == "advisory"
            for record in records
        )

        attention_turns = sum(
            record.policy_status
            == "attention"
            for record in records
        )

        issue_turns = sum(
            record.issue_count > 0
            for record in records
        )

        total_issues = sum(
            record.issue_count
            for record in records
        )

        attention_rate = (
            attention_turns
            / total_turns
        )

        average_character_count = (
            sum(
                record.character_count
                for record in records
            )
            / total_turns
        )

        average_question_count = (
            sum(
                record.question_count
                for record in records
            )
            / total_turns
        )

        latest = records[-1]

        return ResponseQualityAnalyticsResult(
            total_turns=total_turns,
            acceptable_turns=acceptable_turns,
            advisory_turns=advisory_turns,
            attention_turns=attention_turns,
            issue_turns=issue_turns,
            total_issues=total_issues,
            attention_rate=attention_rate,
            average_character_count=(
                average_character_count
            ),
            average_question_count=(
                average_question_count
            ),
            latest_quality_status=(
                latest.quality_status
            ),
            latest_policy_status=(
                latest.policy_status
            ),
        )