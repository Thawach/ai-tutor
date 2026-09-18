from app.domain.scaffolding.strategies import (
    get_strategy,
)


def get_scaffolding_instruction(level: int) -> str:
    """
    Compatibility function สำหรับโค้ดเดิม

    คืนเฉพาะ instruction ของ strategy
    ตาม scaffolding level
    """

    strategy = get_strategy(level)

    return strategy.instruction