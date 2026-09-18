from dataclasses import (
    dataclass,
    field,
)


@dataclass(frozen=True)
class ResponseQualityPolicyResult:
    """
    Deterministic interpretation of response-quality telemetry.

    Status:
    - acceptable
    - advisory
    - attention

    This model is observational only.
    """

    status: str

    reason: str

    issues: list[str] = field(
        default_factory=list
    )

    source_status: str = "uncertain"

    issue_count: int = 0

    requires_attention: bool = False