from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PublicHandState:
    """Public betting values expressed in integer chips."""

    pot: int
    to_call: int
    stack: int
    min_raise: int
    can_raise: bool = True

    def __post_init__(self) -> None:
        values = (self.pot, self.to_call, self.stack, self.min_raise)
        if any(value < 0 for value in values):
            raise ValueError("chip values cannot be negative")
        if self.to_call > self.stack:
            raise ValueError("amount to call cannot exceed the stack")
