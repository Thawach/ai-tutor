from dataclasses import dataclass


@dataclass
class AIUsage:
    provider: str
    model: str

    latency_ms: float = 0.0

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    estimated_cost_usd: float = 0.0