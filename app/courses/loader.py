import json
from pathlib import Path

from app.courses.models import (
    CourseProfile,
    ResponseQualityProfile,
    ResponseStyleProfile,
)


class CourseProfileLoader:
    """
    โหลด Course Profile จาก:

    app/courses/<course_id>/course.json
    """

    def __init__(
        self,
        base_path: str | Path | None = None,
    ):

        if base_path is None:

            base_path = (
                Path(__file__)
                .resolve()
                .parent
            )

        self.base_path = Path(base_path)

    def load(
        self,
        course_id: str,
    ) -> CourseProfile:

        normalized_course_id = (
            course_id.strip().lower()
        )

        profile_path = (
            self.base_path
            / normalized_course_id
            / "course.json"
        )

        if not profile_path.exists():

            raise FileNotFoundError(
                "Course profile not found: "
                f"{profile_path}"
            )

        with profile_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(file)

        self._validate(
            data=data,
            profile_path=profile_path,
        )

        # ---------------------------------------------------------
        # Course identity integrity
        # ---------------------------------------------------------

        profile_course_id = data["course_id"]

        if (
            not isinstance(
                profile_course_id,
                str,
            )
            or
            profile_course_id.strip().lower()
            != normalized_course_id
        ):

            raise ValueError(
                "Invalid course profile "
                f"{profile_path}. "
                "'course_id' does not match "
                "the requested course. "
                f"Requested: '{normalized_course_id}', "
                f"profile declares: "
                f"'{profile_course_id}'."
            )

        response_style_data = data.get(
            "response_style",
            {},
        )

        response_style = ResponseStyleProfile(
            primary_language=(
                response_style_data.get(
                    "primary_language",
                    data.get(
                        "language",
                        "th",
                    ),
                )
            ),
            allow_technical_english=(
                response_style_data.get(
                    "allow_technical_english",
                    True,
                )
            ),
            technical_term_format=(
                response_style_data.get(
                    "technical_term_format",
                    "thai_with_english_parentheses",
                )
            ),
            tone=(
                response_style_data.get(
                    "tone",
                    "supportive_academic",
                )
            ),
            explanation_depth=(
                response_style_data.get(
                    "explanation_depth",
                    "adaptive",
                )
            ),
            question_style=(
                response_style_data.get(
                    "question_style",
                    "socratic",
                )
            ),
            max_guiding_questions=(
                response_style_data.get(
                    "max_guiding_questions",
                    1,
                )
            ),
        )

        # =========================================================
        # Response Quality Profile
        # =========================================================

        quality_data = data.get(
            "response_quality",
            {},
        )

        response_quality = ResponseQualityProfile(
            max_characters=quality_data.get(
                "max_characters",
                1200,
            ),
            min_characters=quality_data.get(
                "min_characters",
                3,
            ),
            max_questions=quality_data.get(
                "max_questions",
                2,
            ),
            detect_repetition=quality_data.get(
                "detect_repetition",
                True,
            ),
        )

        return CourseProfile(
            course_id=normalized_course_id,
            course_name=data["course_name"],
            language=data.get(
                "language",
                "th",
            ),
            document_path=data["document_path"],
            chroma_path=data["chroma_path"],
            description=data.get(
                "description",
                "",
            ),
            technical_terms=data.get(
                "technical_terms",
                {},
            ),
            embedded_claim_patterns=data.get(
                "embedded_claim_patterns",
                [],
            ),

            response_style=response_style,
            response_quality=response_quality,
            
            metadata=data.get(
                "metadata",
                {},
            ),
        )

    def _validate(
        self,
        data: dict,
        profile_path: Path,
    ) -> None:

        required_fields = (
            "course_id",
            "course_name",
            "document_path",
            "chroma_path",
        )

        missing_fields = [
            field
            for field in required_fields
            if not data.get(field)
        ]

        if missing_fields:

            raise ValueError(
                "Invalid course profile "
                f"{profile_path}. "
                "Missing required fields: "
                f"{', '.join(missing_fields)}"
            )

        # =====================================================
        # Validate response_style container
        # =====================================================

        response_style_data = data.get(
            "response_style",
            {},
        )

        if not isinstance(
            response_style_data,
            dict,
        ):
            raise ValueError(
                "Invalid course profile "
                f"{profile_path}. "
                "'response_style' must be an object."
            )

        # =====================================================
        # ADD TYPE VALIDATION HERE
        # =====================================================

        string_fields = (
            "primary_language",
            "technical_term_format",
            "tone",
            "explanation_depth",
            "question_style",
        )

        for field_name in string_fields:

            if (
                field_name in response_style_data
                and
                not isinstance(
                    response_style_data[field_name],
                    str,
                )
            ):
                raise ValueError(
                    "Invalid course profile "
                    f"{profile_path}. "
                    f"'response_style.{field_name}' "
                    "must be a string."
                )

        if (
            "allow_technical_english"
            in response_style_data
            and
            not isinstance(
                response_style_data[
                    "allow_technical_english"
                ],
                bool,
            )
        ):
            raise ValueError(
                "Invalid course profile "
                f"{profile_path}. "
                "'response_style."
                "allow_technical_english' "
                "must be a boolean."
            )

        if (
            "max_guiding_questions"
            in response_style_data
        ):

            value = response_style_data[
                "max_guiding_questions"
            ]

            if (
                not isinstance(
                    value,
                    int,
                )
                or
                isinstance(
                    value,
                    bool,
                )
                or
                value < 0
            ):
                raise ValueError(
                    "Invalid course profile "
                    f"{profile_path}. "
                    "'response_style."
                    "max_guiding_questions' "
                    "must be a non-negative integer."
                ) 
        # =========================================================
        # Response Quality Validation
        # =========================================================

        response_quality_data = data.get(
            "response_quality",
        )

        if response_quality_data is not None:

            if not isinstance(
                response_quality_data,
                dict,
            ):
                raise ValueError(
                    "Invalid course profile "
                    f"{profile_path}. "
                    "'response_quality' "
                    "must be an object."
                )

            integer_fields = (
                "max_characters",
                "min_characters",
                "max_questions",
            )

            for field_name in integer_fields:

                if (
                    field_name
                    not in response_quality_data
                ):
                    continue

                value = response_quality_data[
                    field_name
                ]

                if (
                    not isinstance(
                        value,
                        int,
                    )
                    or
                    isinstance(
                        value,
                        bool,
                    )
                    or
                    value < 0
                ):
                    raise ValueError(
                        "Invalid course profile "
                        f"{profile_path}. "
                        "'response_quality."
                        f"{field_name}' "
                        "must be a non-negative integer."
                    )

            if (
                "detect_repetition"
                in response_quality_data
            ):

                value = response_quality_data[
                    "detect_repetition"
                ]

                if not isinstance(
                    value,
                    bool,
                ):
                    raise ValueError(
                        "Invalid course profile "
                        f"{profile_path}. "
                        "'response_quality."
                        "detect_repetition' "
                        "must be a boolean."
                    )

            max_characters = (
                response_quality_data.get(
                    "max_characters",
                    1200,
                )
            )

            min_characters = (
                response_quality_data.get(
                    "min_characters",
                    3,
                )
            )

            if (
                min_characters
                > max_characters
            ):
                raise ValueError(
                    "Invalid course profile "
                    f"{profile_path}. "
                    "'response_quality."
                    "min_characters' "
                    "must not exceed "
                    "'response_quality."
                    "max_characters'."
                )

            max_characters = response_quality_data.get(
                "max_characters",
                1200,
            )

            min_characters = response_quality_data.get(
                "min_characters",
                3,
            )

            if min_characters > max_characters:
                raise ValueError(
                    "Invalid course profile "
                    f"{profile_path}. "
                    "'response_quality.min_characters' "
                    "must not exceed "
                    "'response_quality.max_characters'."
                )