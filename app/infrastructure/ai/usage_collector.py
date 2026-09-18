from dataclasses import dataclass

from app.infrastructure.ai.usage import (
    AIUsage,
)


@dataclass
class AIUsageRecord:
    """
    Usage ของ LLM call หนึ่งครั้ง
    พร้อมชื่อ component ที่เรียกใช้
    """

    component: str
    usage: AIUsage


class AIUsageCollector:
    """
    เก็บ AI usage หลาย calls ภายในหนึ่ง Tutor turn.

    ตัวอย่าง:
    - evaluator
    - relevance_gate
    - tutor
    - response_validator
    """

    def __init__(self):
        self.records: list[AIUsageRecord] = []

    def reset(self):
        """
        เริ่มเก็บ usage สำหรับ turn ใหม่
        """

        self.records = []

    def add(
        self,
        component: str,
        usage: AIUsage | None,
    ):
        """
        เพิ่ม usage ของ LLM call หนึ่งครั้ง
        """

        if usage is None:
            return

        self.records.append(
            AIUsageRecord(
                component=component,
                usage=usage,
            )
        )

    def get_records(
        self,
    ) -> list[AIUsageRecord]:

        return list(self.records)

    def get_summary(
        self,
    ) -> dict:
        """
        สรุป usage ทั้งหมดของ turn ปัจจุบัน
        """

        total_latency_ms = sum(
            record.usage.latency_ms
            for record in self.records
        )

        total_prompt_tokens = sum(
            record.usage.prompt_tokens
            for record in self.records
        )

        total_completion_tokens = sum(
            record.usage.completion_tokens
            for record in self.records
        )

        total_tokens = sum(
            record.usage.total_tokens
            for record in self.records
        )

        total_cost_usd = sum(
            record.usage.estimated_cost_usd
            for record in self.records
        )

        return {
            "calls": len(self.records),

            "total_latency_ms": total_latency_ms,

            "prompt_tokens": total_prompt_tokens,

            "completion_tokens": (
                total_completion_tokens
            ),

            "total_tokens": total_tokens,

            "estimated_cost_usd": (
                total_cost_usd
            ),

            "records": [
                {
                    "component": record.component,
                    "provider": (
                        record.usage.provider
                    ),
                    "model": (
                        record.usage.model
                    ),
                    "latency_ms": (
                        record.usage.latency_ms
                    ),
                    "prompt_tokens": (
                        record.usage.prompt_tokens
                    ),
                    "completion_tokens": (
                        record.usage.completion_tokens
                    ),
                    "total_tokens": (
                        record.usage.total_tokens
                    ),
                    "estimated_cost_usd": (
                        record.usage.estimated_cost_usd
                    ),
                }
                for record in self.records
            ],
        }