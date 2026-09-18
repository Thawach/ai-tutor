import time

from groq import (
    Groq,
    BadRequestError,
)

from app.infrastructure.ai.usage import (
    AIUsage,
)



from app.core.config import (
    GROQ_API_KEY,
    GROQ_MODEL,
)


class GroqProvider:
    """
    Cloud LLM provider สำหรับ Groq.

    Provider นี้ใช้สำหรับ text/chat generation เท่านั้น

    Embedding ยังคงใช้ OllamaProvider
    และ nomic-embed-text ในเครื่อง
    """

    def __init__(self):

        if not GROQ_API_KEY.strip():
            raise ValueError(
                "GROQ_API_KEY is not configured."
            )

        self.last_usage = None

        self.client = Groq(
            api_key=GROQ_API_KEY
        )


    def _create_completion(
        self,
        messages: list,
        model: str,
        max_completion_tokens: int,
        reasoning_effort: str,
        temperature: float = 0.2,
    ):

        return self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=max_completion_tokens,
            reasoning_effort=reasoning_effort,
            include_reasoning=False,
            temperature=temperature,
            tool_choice="none",
        )

    def _build_safe_retry_messages(
        self,
        messages: list,
    ) -> list:

        safety_instruction = {
            "role": "system",
            "content": (
                "IMPORTANT OUTPUT RULE:\n"
                "Return only a normal plain-text assistant response.\n"
                "Do not call tools.\n"
                "Do not generate function calls.\n"
                "Do not generate tool-call syntax.\n"
                "Do not output internal channel markers or protocol tokens.\n"
                "Respond directly to the learner."
            ),
        }

        return [
            safety_instruction,
            *messages,
        ]

    def generate(
        self,
        messages: list,
        model: str | None = None,
        max_completion_tokens: int = 500,
        reasoning_effort: str = "low",
    ) -> str:

        start_time = time.perf_counter()

        selected_model = (
            model or GROQ_MODEL
        )

        try:

            response = self._create_completion(
                messages=messages,
                model=selected_model,
                max_completion_tokens=max_completion_tokens,
                reasoning_effort=reasoning_effort,
                temperature=0.2,
            )

        except BadRequestError as error:

            error_text = str(error)

            if "tool_use_failed" not in error_text:
                raise

            retry_messages = (
                self._build_safe_retry_messages(
                    messages
                )
            )

            response = self._create_completion(
                messages=retry_messages,
                model=selected_model,
                max_completion_tokens=max_completion_tokens,
                reasoning_effort=reasoning_effort,
                temperature=0.1,
            )

        # -------------------------
        # Latency
        # -------------------------

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        # -------------------------
        # Token usage
        # -------------------------
        
        usage = response.usage

        prompt_tokens = (
            usage.prompt_tokens
            if usage is not None
            else 0
        )

        completion_tokens = (
            usage.completion_tokens
            if usage is not None
            else 0
        )

        total_tokens = (
            usage.total_tokens
            if usage is not None
            else 0
        )

        # -------------------------
        # Estimated cost
        # -------------------------

        input_cost = (
            prompt_tokens
            / 1_000_000
            * 0.075
        )

        output_cost = (
            completion_tokens
            / 1_000_000
            * 0.30
        )

        estimated_cost = (
            input_cost
            + output_cost
        )

        self.last_usage = AIUsage(
            provider="groq",
            model=model or GROQ_MODEL,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=estimated_cost,
        )

        # -------------------------
        # Response content
        # -------------------------                                               

        content = (
            response
            .choices[0]
            .message
            .content
        )

        if content is None:
            return ""

        return content.strip()

    def generate_structured(
        self,
        messages: list,
        schema_name: str,
        schema: dict,
        model: str | None = None,
        max_completion_tokens: int = 800,
        reasoning_effort: str = "low",
    ) -> str:
        """
        Generate JSON output constrained by JSON Schema.

        ใช้สำหรับ internal AI services เช่น:
        - evaluator
        - relevance_gate
        - response_validator

        ไม่ใช้สำหรับ tutor response ที่ส่งให้ learner
        """

        start_time = time.perf_counter()

        selected_model = (
            model or GROQ_MODEL
        )

        response = self.client.chat.completions.create(
            model=selected_model,
            messages=messages,
            max_completion_tokens=max_completion_tokens,
            reasoning_effort=reasoning_effort,
            include_reasoning=False,
            temperature=0.0,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                },
            },
        )

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        usage = response.usage

        prompt_tokens = (
            usage.prompt_tokens
            if usage is not None
            else 0
        )

        completion_tokens = (
            usage.completion_tokens
            if usage is not None
            else 0
        )

        total_tokens = (
            usage.total_tokens
            if usage is not None
            else 0
        )

        input_cost = (
            prompt_tokens
            / 1_000_000
            * 0.075
        )

        output_cost = (
            completion_tokens
            / 1_000_000
            * 0.30
        )

        estimated_cost = (
            input_cost
            + output_cost
        )

        self.last_usage = AIUsage(
            provider="groq",
            model=selected_model,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=estimated_cost,
        )

        content = (
            response
            .choices[0]
            .message
            .content
        )

        if content is None:
            return "{}"

        return content.strip()