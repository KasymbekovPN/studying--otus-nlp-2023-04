from src.hw_005_bot.engine.engine_strategies import (
    BaseEngineStrategy,
    UnknownCommandEngineStrategy,
    TextEngineStrategy
)


class DeterminationResult:
    KIND_UNKNOWN = -1
    KIND_UNKNOWN_COMMAND = 0
    KIND_SPEC_COMMAND = 1
    KIND_TEXT = 2

    KIND_NAMES = {
        KIND_UNKNOWN: 'UNKNOWN',
        KIND_UNKNOWN_COMMAND: 'UNKNOWN_COMMAND',
        KIND_SPEC_COMMAND: 'SPEC_COMMAND',
        KIND_TEXT: 'TEXT'
    }

    def __init__(self,
                 kind: int,
                 text: str | None,
                 strategy=BaseEngineStrategy()) -> None:
        self._kind = kind
        self._text = text
        self._strategy = strategy

    def __repr__(self):
        kind = self.kind if self.kind in self.KIND_NAMES else self.KIND_UNKNOWN
        return f'{{kind: {self.KIND_NAMES[kind]}, text: {self.text} }}'

    @property
    def kind(self):
        return self._kind

    @kind.setter
    def kind(self, value):
        raise RuntimeError('Setter for kind is not supported')

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, value):
        raise RuntimeError('Setter for value is not supported')

    @property
    def strategy(self):
        return self._strategy

    @strategy.setter
    def strategy(self, value):
        raise RuntimeError('Setter for strategy is not supported')

    @staticmethod
    def create_for_unknown(text: str):
        r = DeterminationResult(DeterminationResult.KIND_UNKNOWN, text)
        return r

    @staticmethod
    def create_for_spec_command(text: str, strategy: BaseEngineStrategy):
        r = DeterminationResult(DeterminationResult.KIND_SPEC_COMMAND, text, strategy)
        return r

    @staticmethod
    def create_for_unknown_command(text: str):
        r = DeterminationResult(
            DeterminationResult.KIND_UNKNOWN_COMMAND,
            text,
            UnknownCommandEngineStrategy()
        )
        return r

    @staticmethod
    def create_for_text(text: str):
        r = DeterminationResult(
            DeterminationResult.KIND_TEXT,
            text,
            TextEngineStrategy()
        )
        return r
