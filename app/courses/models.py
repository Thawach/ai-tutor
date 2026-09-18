from dataclasses import dataclass, field

@dataclass(frozen=True)
class ResponseStyleProfile:
    """
    Course-level response style configuration.

    This configuration describes preferred Tutor response style.

    Step 16.17.2B is configuration-only.
    These values must not modify Tutor behavior until explicitly
    integrated in a later step.
    """

    primary_language: str = "th"

    allow_technical_english: bool = True

    technical_term_format: str = (
        "thai_with_english_parentheses"
    )

    tone: str = (
        "supportive_academic"
    )

    explanation_depth: str = (
        "adaptive"
    )

    question_style: str = (
        "socratic"
    )

    max_guiding_questions: int = 1


@dataclass(frozen=True)
class ResponseQualityProfile:
    """
    Course-level deterministic response-quality configuration.

    This profile controls observation thresholds only.
    It does not trigger repair, regeneration, or LLM calls.
    """

    max_characters: int = 1200
    min_characters: int = 3
    max_questions: int = 2
    detect_repetition: bool = True

@dataclass(frozen=True)
class CourseProfile:
    """
    Course-specific configuration used by the AI Tutor.

    The Tutor core remains course-agnostic.
    Course-specific behavior and configuration are supplied
    through this profile.
    """

    course_id: str
    course_name: str
    language: str

    document_path: str
    chroma_path: str

    description: str = ""

    technical_terms: dict[str, str] = field(
        default_factory=dict
    )

    embedded_claim_patterns: list[str] = field(
        default_factory=list
    )

    response_style: ResponseStyleProfile = field(
        default_factory=ResponseStyleProfile
    )

    response_quality: ResponseQualityProfile = field(
        default_factory=ResponseQualityProfile
    )

    metadata: dict = field(
        default_factory=dict
    )