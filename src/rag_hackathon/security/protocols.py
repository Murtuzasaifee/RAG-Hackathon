from __future__ import annotations

from typing import Protocol


class ScanResult:
    __slots__ = ("is_valid", "sanitized", "score", "reasons")

    def __init__(
        self,
        is_valid: bool,
        sanitized: str = "",
        score: float = 0.0,
        reasons: list[str] | None = None,
    ) -> None:
        self.is_valid = is_valid
        self.sanitized = sanitized
        self.score = score
        self.reasons = reasons or []


class InputGuard(Protocol):
    async def scan_input(self, prompt: str) -> ScanResult: ...


class OutputGuard(Protocol):
    async def scan_output(
        self, prompt: str, output: str
    ) -> ScanResult: ...
