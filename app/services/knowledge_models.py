from dataclasses import dataclass, field


@dataclass
class KnowledgeSource:
    text: str
    source: str
    page: int
    distance: float | None = None


@dataclass
class Citation:
    source: str
    page: int


@dataclass
class KnowledgeResult:
    query: str
    context: str
    sources: list[KnowledgeSource] = field(
        default_factory=list
    )
    citations: list[Citation] = field(
        default_factory=list
    )

    @property
    def has_context(self) -> bool:
        return bool(self.context.strip())