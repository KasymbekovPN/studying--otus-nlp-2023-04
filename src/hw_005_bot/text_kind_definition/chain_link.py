
# todo ???
class BaseDeterminant:
    def __call__(self, *args, **kwargs):
        text = kwargs.get('text') if 'text' in kwargs else None
        return self._execute(text)

    def _execute(self, text: str | None):
        return None


class KnownCommandDeterminant(BaseDeterminant):
    def _execute(self, text: str | None):
        pass


class UnknownCommandDeterminant:
    def _execute(self, text: str | None):
        pass


class TaskDeterminant:
    def _execute(self, text: str | None):
        pass


class DefaultDeterminant:
    pass
