class DeterminationResult:
    KIND_UNKNOWN = -1
    KIND_COMMAND = 0
    KIND_TEXT = 1

    KIND_NAMES = {-1: 'UNKNOWN', 0: 'COMMAND', 1: 'TEXT'}

    def __init__(self,
                 kind: int,
                 raw_text: str,
                 command: str | None,
                 text: str | None) -> None:
        self.kind = kind
        self.raw_text = raw_text
        self.command = command
        self.text = text

    def __repr__(self):
        kind = self.kind if self.kind in self.KIND_NAMES else self.KIND_UNKNOWN
        return f'{{kind: {self.KIND_NAMES[kind]}, command: {self.command}, text: {self.text}, raw_text: {self.raw_text} }}'

    @staticmethod
    def create_for_unknown(raw_text: str):
        r = DeterminationResult(DeterminationResult.KIND_UNKNOWN, raw_text, None, None)
        return r

    @staticmethod
    def create_for_command(raw_text: str, command: str):
        r = DeterminationResult(DeterminationResult.KIND_COMMAND, raw_text, command, None)
        return r

    @staticmethod
    def create_for_text(raw_text: str):
        r = DeterminationResult(DeterminationResult.KIND_TEXT, raw_text, None, raw_text)
        return r
