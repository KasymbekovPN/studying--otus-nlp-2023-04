import re

from src.hw_005_bot.message.processing.determinant.result import DeterminationResult
from src.hw_005_bot.engine.engine_strategies import BaseEngineStrategy


class BaseDeterminant:
    def __call__(self, *args, **kwargs) -> DeterminationResult | None:
        text = kwargs.get('text') if 'text' in kwargs else None
        return self._determinate(text)

    def _determinate(self, text: str | None) -> DeterminationResult | None:
        return DeterminationResult.create_for_unknown(text)


class SpecificCommandDeterminant(BaseDeterminant):
    def __init__(self, command: str, strategy: BaseEngineStrategy) -> None:
        self._command = command
        self._strategy = strategy

    def _determinate(self, text: str | None) -> DeterminationResult | None:
        if self._command == text:
            return DeterminationResult.create_for_spec_command(text, self._strategy)
        return None


class AnyCommandDeterminant(BaseDeterminant):
    def __init__(self):
        self._re = re.compile('/[a-z][a-zA-Z0-9_]+')

    def _determinate(self, text: str | None) -> DeterminationResult | None:
        if text is not None:
            match = self._re.match(text)
            if match is not None and len(text) == self._re.match(text).span()[1]:
                return DeterminationResult.create_for_unknown_command(text)
        return None


class TextDeterminant(BaseDeterminant):
    def _determinate(self, text: str | None) -> DeterminationResult | None:
        return None if text is None else DeterminationResult.create_for_text(text)


class DefaultDeterminant(BaseDeterminant):
    pass
