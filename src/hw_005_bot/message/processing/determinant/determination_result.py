
class DeterminationResult:
    KIND_COMMAND = 0
    KIND_TASK = 1
    KIND_UNKNOWN = 2

    def __init__(self,
                 raw_text: str,
                 command: str | None,
                 question: str | None,
                 passage: str | None) -> None:
        self.kind = self.KIND_UNKNOWN
        self.raw_text = raw_text
        self.command = command
        self.question = question
        self.passage = passage

    @staticmethod
    def create_for_unknown(raw_text: str):
        r = DeterminationResult(raw_text, None, None, None)
        return r

    @staticmethod
    def create_for_command(raw_text: str, command: str):
        r = DeterminationResult(raw_text, command, None, None)
        return r

    @staticmethod
    def create_for_task(raw_text: str, question: str, passage: str):
        r = DeterminationResult(raw_text, None, question, passage)
        return r
