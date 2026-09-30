from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HandEvent:
    kind: str
    actor: int | None = None
    amount: int = 0
    cards: tuple[str, ...] = ()
