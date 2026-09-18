from app.services.response_quality_history_models import (
    ResponseQualityTurnRecord,
)


class ResponseQualityHistory:
    """
    In-memory ordered response-quality history.

    This container is analytics-only and does not:
    - call an LLM
    - modify Tutor responses
    - trigger repair
    - trigger escalation
    """

    def __init__(self) -> None:

        self._records: list[
            ResponseQualityTurnRecord
        ] = []

    # =========================================================
    # Mutation
    # =========================================================

    def add(
        self,
        record: ResponseQualityTurnRecord,
    ) -> None:

        if not isinstance(
            record,
            ResponseQualityTurnRecord,
        ):
            raise TypeError(
                "record must be a "
                "ResponseQualityTurnRecord."
            )

        self._records.append(
            record
        )

    def clear(
        self,
    ) -> None:

        self._records.clear()

    # =========================================================
    # Read-only access
    # =========================================================

    @property
    def records(
        self,
    ) -> tuple[
        ResponseQualityTurnRecord,
        ...
    ]:

        return tuple(
            self._records
        )

    @property
    def count(
        self,
    ) -> int:

        return len(
            self._records
        )

    @property
    def latest(
        self,
    ) -> ResponseQualityTurnRecord | None:

        if not self._records:
            return None

        return self._records[-1]