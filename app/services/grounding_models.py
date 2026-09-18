from dataclasses import dataclass


@dataclass
class GroundingStatus:
    has_knowledge: bool
    reason: str