import time


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    def __init__(self, seconds: float):
        self.started = time.monotonic()
        self.deadline = self.started + seconds

    def check(self):
        if time.monotonic() >= self.deadline:
            raise BudgetExceeded('wall-time budget exhausted')

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started
