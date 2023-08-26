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
                 raw_text: str,
                 command: str | None,
                 text: str | None,
                 strategy=BaseEngineStrategy()) -> None:
        self.kind = kind
        self.raw_text = raw_text
        self.command = command
        self.text = text
        self._strategy = strategy

    def __repr__(self):
        kind = self.kind if self.kind in self.KIND_NAMES else self.KIND_UNKNOWN
        return f'{{kind: {self.KIND_NAMES[kind]}, command: {self.command}, text: {self.text}, raw_text: {self.raw_text} }}'

    @property
    def strategy(self):
        return self._strategy

    @strategy.setter
    def strategy(self, value):
        raise RuntimeError('Setter for strategy is not supported')

    @staticmethod
    def create_for_unknown(raw_text: str):
        r = DeterminationResult(DeterminationResult.KIND_UNKNOWN, raw_text, None, None)
        return r

    @staticmethod
    def create_for_spec_command(raw_text: str, command: str, strategy: BaseEngineStrategy):
        r = DeterminationResult(DeterminationResult.KIND_SPEC_COMMAND, raw_text, command, None, strategy)
        return r

    @staticmethod
    def create_for_unknown_command(raw_text: str, command: str):
        r = DeterminationResult(
            DeterminationResult.KIND_UNKNOWN_COMMAND,
            raw_text,
            command,
            None,
            UnknownCommandEngineStrategy()
        )
        return r

    @staticmethod
    def create_for_text(raw_text: str):
        r = DeterminationResult(
            DeterminationResult.KIND_TEXT,
            raw_text,
            None,
            raw_text,
            TextEngineStrategy()
        )
        return r
